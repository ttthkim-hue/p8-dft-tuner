"""Blackwell Ozaki regime advisor, VRAM-wall & MLIP manifold guards, and QE `.in` parser (`v1.1.0`)."""
from __future__ import annotations

import math
import pathlib
import platform
import re
from typing import Any


# Calibrated v3 closed-form defaults (synced to `model_fit_metrics.json` schema v3 reduced_core)
DEFAULT_OZ_SLICE_BASE = 0.20
DEFAULT_OZ_SLICE_PER_BIT = 0.008
DEFAULT_OZ_SMALL_MATRIX_OVH = 12.96
DEFAULT_OZ_REF_NBND = 128.0
PAW_TOKEN_RE = re.compile(r"(?:^|[._-])(kj)?paw(?:[._-]|$)", re.IGNORECASE)


class OzakiPolicyViolation(RuntimeError):
    """Raised when `strict=True` and user requests an unsafe Ozaki mode (VRAM wall or small-cell cliff)."""


def _parse_fortran_float(token: str) -> float:
    """Parse standard or Fortran D/d exponent floats (e.g., '4.5d1' -> 45.0, '1.0D-8' -> 1e-8)."""
    cleaned = token.strip().replace("D", "e").replace("d", "e")
    return float(cleaned)


def detect_wsl2() -> bool:
    """Return True if running inside Windows Subsystem for Linux 2."""
    rel = platform.release().lower()
    if "microsoft" in rel or "wsl" in rel:
        return True
    pv = pathlib.Path("/proc/version")
    if pv.is_file():
        try:
            txt = pv.read_text(encoding="utf-8", errors="ignore").lower()
            return "microsoft" in txt or "wsl" in txt
        except OSError:
            return False
    return False


def parse_qe_input_summary(qe_in_path: pathlib.Path | str) -> dict[str, Any]:
    """Extract structural, k-point, and functional descriptors from a Quantum ESPRESSO `.in` file.

    Raises `ValueError` if input is empty, missing `ATOMIC_POSITIONS`, or specifies `nat <= 0`.
    """
    p = pathlib.Path(qe_in_path)
    raw_txt = p.read_text(encoding="utf-8", errors="ignore")
    if not raw_txt.strip():
        raise ValueError(f"QE input '{p}' is empty")

    # Strip Fortran inline `!` comments before keyword and numeric scanning
    clean_lines = [ln.split("!", 1)[0] for ln in raw_txt.splitlines()]
    clean_txt = "\n".join(clean_lines)
    if not clean_txt.strip():
        raise ValueError(f"QE input '{p}' contains only comments/whitespace")

    m_nat = re.search(r"\bnat\s*=\s*([+-]?\d+)", clean_txt, flags=re.IGNORECASE)
    m_nspin = re.search(r"\bnspin\s*=\s*([+-]?\d+)", clean_txt, flags=re.IGNORECASE)
    m_nbnd = re.search(r"\bnbnd\s*=\s*([+-]?\d+)", clean_txt, flags=re.IGNORECASE)
    m_ecut = re.search(r"\becutwfc\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?)", clean_txt, flags=re.IGNORECASE)

    nspin = int(m_nspin.group(1)) if m_nspin else 1
    if nspin < 1:
        raise ValueError(f"Invalid nspin={nspin} in '{p}'")
    ecutwfc = _parse_fortran_float(m_ecut.group(1)) if m_ecut else 60.0

    has_u = bool(
        re.search(r"^\s*HUBBARD\b", clean_txt, flags=re.IGNORECASE | re.MULTILINE)
        or re.search(r"\blda_plus_u\s*=\s*\.true\.", clean_txt, flags=re.IGNORECASE)
        or re.search(r"\bHubbard_U\s*\(", clean_txt, flags=re.IGNORECASE)
    )

    paw_species: set[str] = set()
    for i, ln in enumerate(clean_lines):
        if ln.strip().upper().startswith("ATOMIC_SPECIES"):
            for sl in clean_lines[i + 1 :]:
                s = sl.strip()
                if not s:
                    continue
                if re.match(r"^(ATOMIC_POSITIONS|CELL_PARAMETERS|K_POINTS|HUBBARD|CONSTRAINTS|&|/)\b", s, flags=re.I):
                    break
                toks = s.split()
                if len(toks) >= 3 and PAW_TOKEN_RE.search(toks[2]):
                    paw_species.add(toks[0])
            break

    saw_positions = False
    pos_count = 0
    n_paw = 0
    for i, ln in enumerate(clean_lines):
        if ln.strip().upper().startswith("ATOMIC_POSITIONS"):
            saw_positions = True
            for sl in clean_lines[i + 1 :]:
                s = sl.strip()
                if not s:
                    continue
                if re.match(r"^(ATOMIC_SPECIES|CELL_PARAMETERS|K_POINTS|HUBBARD|CONSTRAINTS|&|/)\b", s, flags=re.I):
                    break
                toks = s.split()
                if len(toks) >= 4:
                    pos_count += 1
                    if toks[0] in paw_species:
                        n_paw += 1
            break

    if not saw_positions or pos_count == 0:
        raise ValueError(f"QE input '{p}' is missing a populated ATOMIC_POSITIONS card")

    nat = int(m_nat.group(1)) if m_nat else pos_count
    if nat < 1:
        raise ValueError(f"Invalid nat={nat} in '{p}' (must be >= 1)")

    # Parse spatial k-points from K_POINTS card
    nks = 1
    for i, ln in enumerate(clean_lines):
        s_hdr = ln.strip().upper()
        if s_hdr.startswith("K_POINTS"):
            if "GAMMA" in s_hdr:
                nks = 1
            elif "AUTOMATIC" in s_hdr:
                for sl in clean_lines[i + 1 :]:
                    toks = sl.strip().split()
                    if len(toks) >= 3:
                        nks = max(1, int(toks[0]) * int(toks[1]) * int(toks[2]))
                        break
            else:
                for sl in clean_lines[i + 1 :]:
                    toks = sl.strip().split()
                    if toks:
                        try:
                            nks = max(1, int(toks[0]))
                        except ValueError:
                            nks = 1
                        break
            break

    nbnd_est = int(m_nbnd.group(1)) if m_nbnd else max(64, nat * 4)
    if nbnd_est < 1:
        raise ValueError(f"Invalid nbnd={nbnd_est} in '{p}'")

    return {
        "nat": nat,
        "n_paw": n_paw,
        "has_hubbard_u": has_u,
        "nspin": nspin,
        "nks": nks,
        "nbnd": nbnd_est,
        "ecutwfc_ry": ecutwfc,
    }


def estimate_ozaki_crossover_nbnd(mantissa_bits: int = 53) -> int:
    """Compute mantissa-dependent analytical compute crossover band count nbnd*(m)."""
    if int(mantissa_bits) < 24:
        raise ValueError(f"mantissa_bits must be >= 24, got {mantissa_bits}")
    slice_term = min(0.95, DEFAULT_OZ_SLICE_BASE + DEFAULT_OZ_SLICE_PER_BIT * max(0, int(mantissa_bits) - 39))
    bit_scale_time = max(0.45, (max(24, int(mantissa_bits)) - 22.0) / 31.0)
    if slice_term >= 1.0:
        return 10_000
    nbnd_star = DEFAULT_OZ_REF_NBND * math.sqrt((DEFAULT_OZ_SMALL_MATRIX_OVH * bit_scale_time) / (1.0 - slice_term))
    return int(round(nbnd_star))


def estimate_ozaki_eager_multiplier(nbnd: int, mantissa_bits: int = 53) -> float:
    """Predict GEMM Ozaki eager latency multiplier relative to native FP64."""
    if int(nbnd) < 1:
        raise ValueError(f"nbnd must be >= 1, got {nbnd}")
    if int(mantissa_bits) < 24:
        raise ValueError(f"mantissa_bits must be >= 24, got {mantissa_bits}")
    ratio = max(int(nbnd), 32) / DEFAULT_OZ_REF_NBND
    slice_term = min(0.90, max(0.20, DEFAULT_OZ_SLICE_BASE + DEFAULT_OZ_SLICE_PER_BIT * max(0, mantissa_bits - 39)))
    bit_scale = max(0.45, (max(24, mantissa_bits) - 22.0) / 31.0)
    return max(0.30, slice_term + (DEFAULT_OZ_SMALL_MATRIX_OVH * bit_scale) / (ratio * ratio))


def estimate_ozaki_vram_mib(
    *,
    nat: int = 35,
    nbnd: int,
    npw_smooth: int,
    nks: int = 1,
    nspin: int = 1,
    mantissa_bits: int = 53,
    concurrent_workers: int = 1,
) -> dict[str, float]:
    """Estimate baseline and Ozaki-eager peak GPU VRAM (MiB) matching v3 physics."""
    if int(nbnd) < 1 or int(npw_smooth) < 1 or int(nks) < 1 or int(nspin) < 1 or int(concurrent_workers) < 1:
        raise ValueError("nbnd, npw_smooth, nks, nspin, and concurrent_workers must all be >= 1")
    npw_m = npw_smooth / 1_000_000.0
    bnd_100 = nbnd / 100.0
    spin_factor = 1.0 + 0.35 * max(0, nspin - 1)
    w = int(concurrent_workers)
    k = int(nks)
    phi_spin = k * npw_m * spin_factor

    vram_off = 2758.0 * w + w * (3180.5 * npw_m + 1035.1 * bnd_100 * phi_spin)
    footprint = bnd_100 * phi_spin
    bit_scale = max(0.35, (max(24, mantissa_bits) - 20.0) / 33.0)
    oz_spike = (10662.6 * (footprint - 0.60) * bit_scale * w) if footprint > 0.60 else 0.0
    vram_eager = vram_off + oz_spike
    return {
        "estimated_vram_off_mib": round(vram_off, 1),
        "estimated_ozaki_workspace_spike_mib": round(oz_spike, 1),
        "estimated_vram_eager_mib": round(vram_eager, 1),
    }


def advise_execution_regime(
    *,
    nat: int,
    nbnd: int,
    n_paw: int | None = None,
    has_hubbard_u: bool = False,
    nks: int = 1,
    nspin: int = 1,
    npw_smooth: int | None = None,
    cuda_version: str = "13.2",
    ozaki_mode: str = "auto",
    mantissa_bits: int = 53,
    cudart_mode: str = "static",
    in_wsl2: bool | None = None,
    physical_cores: int = 24,
    gpu_vram_mib: int = 48935,
    vram_safe_cap_mib: int = 43008,
    mlip_pre_relax: str = "auto",
    strict: bool = False,
    allow_slowdown: bool = False,
    crossover_nbnd: int | None = None,
) -> dict[str, Any]:
    """Recommend concurrency, solver, MPI pool, CUDA Ozaki, and MLIP manifold policies with hard safety guards."""
    if int(nat) < 1:
        raise ValueError(f"nat must be >= 1, got {nat}")
    if int(nbnd) < 1:
        raise ValueError(f"nbnd must be >= 1, got {nbnd}")
    if n_paw is not None and int(n_paw) < 0:
        raise ValueError(f"n_paw must be >= 0, got {n_paw}")
    if int(nks) < 1:
        raise ValueError(f"nks must be >= 1, got {nks}")
    if int(nspin) < 1:
        raise ValueError(f"nspin must be >= 1, got {nspin}")
    if npw_smooth is not None and int(npw_smooth) < 1:
        raise ValueError(f"npw_smooth must be >= 1, got {npw_smooth}")
    if int(mantissa_bits) < 24:
        raise ValueError(f"mantissa_bits must be >= 24, got {mantissa_bits}")
    if int(physical_cores) < 1:
        raise ValueError(f"physical_cores must be >= 1, got {physical_cores}")
    if int(gpu_vram_mib) <= 0 or int(vram_safe_cap_mib) <= 0:
        raise ValueError("gpu_vram_mib and vram_safe_cap_mib must be > 0")

    paw_atoms = int(n_paw if n_paw is not None else nat)
    est_npw = int(npw_smooth if npw_smooth is not None else max(250_000, nat * 11_000))
    wsl2 = bool(detect_wsl2() if in_wsl2 is None else in_wsl2)
    req_oz = (ozaki_mode or "auto").lower()
    req_mlip = (mlip_pre_relax or "auto").lower()
    eff_crossover = int(crossover_nbnd) if crossover_nbnd is not None else estimate_ozaki_crossover_nbnd(mantissa_bits)

    cuda_major = 0
    try:
        cuda_major = int(str(cuda_version).split(".")[0])
    except ValueError:
        cuda_major = 0

    warnings: list[str] = []
    policy_action = "accepted"
    recommended_oz = req_oz
    alpha_eager = estimate_ozaki_eager_multiplier(nbnd, mantissa_bits=mantissa_bits)
    vram_est = estimate_ozaki_vram_mib(
        nat=nat,
        nbnd=nbnd,
        npw_smooth=est_npw,
        nks=nks,
        nspin=nspin,
        mantissa_bits=mantissa_bits,
        concurrent_workers=1,
    )
    vram_off_1 = vram_est["estimated_vram_off_mib"]
    eager_vram = vram_est["estimated_vram_eager_mib"]
    vram_spike = vram_est["estimated_ozaki_workspace_spike_mib"]
    hits_vram_wall = eager_vram > float(vram_safe_cap_mib)

    # Guard 1: Hardware VRAM-capacity safety & strict policy are NEVER bypassed by allow_slowdown
    if req_oz == "eager":
        if hits_vram_wall:
            msg = (
                f"Blocked ozaki_mode='eager': predicted peak VRAM {eager_vram:.0f} MiB "
                f"(+{vram_spike:.0f} MiB Ozaki workspace) exceeds safe cap {vram_safe_cap_mib} MiB "
                f"(GPU {gpu_vram_mib} MiB); switched to ozaki_mode='off'."
            )
            if strict:
                raise OzakiPolicyViolation(msg)
            warnings.append(msg)
            recommended_oz = "off"
            policy_action = "blocked_vram_capacity_wall"
        elif nbnd < eff_crossover and not allow_slowdown:
            msg = (
                f"Blocked ozaki_mode='eager': nbnd={nbnd} < analytical compute crossover {eff_crossover} "
                f"for {mantissa_bits}-bit (predicted {alpha_eager:.2f}x GEMM overhead); switched to ozaki_mode='off'."
            )
            if strict:
                raise OzakiPolicyViolation(msg)
            warnings.append(msg)
            recommended_oz = "off"
            policy_action = "blocked_small_cell_eager"
    elif req_oz == "performant" and 160 < nbnd < eff_crossover:
        warnings.append(
            f"Note: ozaki_mode='performant' in intermediate band range (nbnd={nbnd}, 160 < nbnd < {eff_crossover}) "
            "can regress (~2-12%) vs CUDA 13.2 ozaki_mode='off'; prefer 'off' unless benchmarking."
        )
    elif req_oz == "auto":
        if cuda_major >= 13 and nbnd <= 160 and est_npw <= 600_000 and paw_atoms == 0:
            recommended_oz = "performant"
            policy_action = "auto_small_cell_performant_sweet_spot"
        else:
            recommended_oz = "off"
            policy_action = "auto_cuda13_off_safe_default"

    # Guard 2: Warn loudly on WSL2 dynamic CUDA >= 13 libcudart.so mismatch risk
    if wsl2 and cuda_major >= 13 and (cudart_mode or "static").lower() == "dynamic":
        warnings.append(
            "WSL2 + CUDA >= 13.0 with dynamic cudart risks libcuda/libcudart version mismatch; "
            "link static libcudart (--cudart=static) or validate against bare-metal Linux driver."
        )

    # Guard 3: MLIP+QE functional-manifold matching rule
    if has_hubbard_u:
        if req_mlip == "u_matched":
            mlip_recommended = True
            mlip_mode = "u_matched"
            mlip_policy = "allowed_u_matched_mlip"
        else:
            mlip_recommended = False
            mlip_mode = "none"
            mlip_policy = "blocked_pbe_mlip_dft_u_mismatch"
            if req_mlip in ("auto", "pbe_universal"):
                warnings.append(
                    "Disabled universal GGA/PBE MLIP pre-relax on DFT+U structure: GGA MLIP distorts "
                    "correlated M-O bond manifold, increasing subsequent QE SCF/BFGS work unless U-matched."
                )
    else:
        mlip_recommended = req_mlip != "none"
        mlip_mode = "pbe_universal" if mlip_recommended else "none"
        mlip_policy = "recommended_same_pbe_manifold" if mlip_recommended else "disabled_by_request"

    # Estimate CPU serial fraction & enforce multi-worker VRAM fit (w * vram_off <= safe_cap)
    paw_work = 0.08 * max(0, paw_atoms) + (30.0 if has_hubbard_u else 0.0)
    gpu_work = 12.0 + 0.65 * max(1, nks) * ((max(16, nbnd) / 100.0) ** 3)
    f_cpu_est = min(0.9, max(0.0, paw_work / max(paw_work + gpu_work, 1e-6)))

    if physical_cores >= 24:
        w_cores = 3 if (f_cpu_est >= 0.15 or vram_off_1 <= 14_000.0) else 2
        cores_per_slot = 4
    elif physical_cores >= 12:
        w_cores = 2
        cores_per_slot = max(2, physical_cores // 4)
    else:
        w_cores = 1
        cores_per_slot = max(1, physical_cores)

    max_w_vram = max(1, int(float(vram_safe_cap_mib) // max(vram_off_1, 1.0)))
    w_star = max(1, min(w_cores, max_w_vram))
    if w_star < w_cores:
        warnings.append(
            f"Reduced concurrent_workers from {w_cores} to {w_star} so {w_star} x {vram_off_1:.0f} MiB "
            f"({w_star * vram_off_1:.0f} MiB) fits within safe VRAM cap {vram_safe_cap_mib} MiB."
        )

    recommended_np = 2 if (nks >= 2 and w_star == 1) else 1
    recommended_npool = 2 if recommended_np == 2 else 1

    return {
        "recommended_ozaki_mode": recommended_oz,
        "requested_ozaki_mode": req_oz,
        "policy_action": policy_action,
        "ozaki_crossover_nbnd": eff_crossover,
        "mantissa_bits": mantissa_bits,
        "estimated_eager_gemm_multiplier": round(alpha_eager, 3),
        "estimated_npw_smooth": est_npw,
        "estimated_vram_off_mib": vram_off_1,
        "estimated_ozaki_workspace_spike_mib": vram_spike,
        "estimated_vram_eager_mib": eager_vram,
        "vram_safe_cap_mib": vram_safe_cap_mib,
        "hits_vram_capacity_wall": hits_vram_wall,
        "mlip_pre_relax_recommended": mlip_recommended,
        "recommended_mlip_mode": mlip_mode,
        "mlip_policy": mlip_policy,
        "recommended_diagonalization": "david",
        "recommended_mpi_ranks": recommended_np,
        "recommended_npool": recommended_npool,
        "recommended_omp_threads": 1,
        "recommended_concurrent_workers": w_star,
        "cores_per_gpu_slot": cores_per_slot,
        "estimated_cpu_serial_fraction": round(f_cpu_est, 4),
        "recommended_cpu_lane_cores": max(0, physical_cores - w_star * cores_per_slot),
        "wsl2_detected": wsl2,
        "cudart_mode": cudart_mode,
        "warnings": warnings,
    }

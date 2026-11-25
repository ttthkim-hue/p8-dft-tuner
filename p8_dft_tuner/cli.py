"""Portable CLI entrypoint (`p8-dft-tuner`) for topology, runtime env, MLIP guard, and Blackwell advice."""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

from .blackwell_advisor import advise_execution_regime, parse_qe_input_summary
from .env_guard import build_runtime_env
from .locks import verify_non_overlapping_masks
from .topology import compute_disjoint_lane_plan, detect_cpu_topology

CLI_VERSION = "1.1.0"


def _cmd_recommend(args: argparse.Namespace) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if args.qe_in is not None:
        meta = parse_qe_input_summary(args.qe_in)

    nat = int(args.nat if args.nat is not None else meta.get("nat", 35))
    nbnd = int(args.nbnd if args.nbnd is not None else meta.get("nbnd", max(64, nat * 4)))
    n_paw = int(args.n_paw if args.n_paw is not None else meta.get("n_paw", nat))
    nks = int(args.nks if args.nks is not None else meta.get("nks", 1))
    nspin = int(args.nspin if args.nspin is not None else meta.get("nspin", 1))
    has_u = bool(args.hubbard_u or meta.get("has_hubbard_u", False))

    topo = detect_cpu_topology(logical_cpus_override=args.logical_cpus)
    advice = advise_execution_regime(
        nat=nat,
        nbnd=nbnd,
        n_paw=n_paw,
        has_hubbard_u=has_u,
        nks=nks,
        nspin=nspin,
        npw_smooth=args.npw_smooth,
        cuda_version=args.cuda_version,
        ozaki_mode=args.ozaki_mode,
        mantissa_bits=args.mantissa_bits,
        cudart_mode=args.cudart_mode,
        in_wsl2=args.wsl2,
        physical_cores=topo.physical_cores,
        gpu_vram_mib=args.gpu_vram_mib,
        vram_safe_cap_mib=args.vram_safe_cap_mib,
        mlip_pre_relax=args.mlip_pre_relax,
        strict=args.strict,
        allow_slowdown=args.allow_slowdown,
    )
    lane_plan = compute_disjoint_lane_plan(
        topo,
        gpu_slots=advice["recommended_concurrent_workers"],
        cores_per_gpu_slot=advice["cores_per_gpu_slot"],
    )
    gpu_masks = [s["cpu_list"] for s in lane_plan["gpu_slots"]]  # type: ignore[index]
    cpu_mask = str(lane_plan["cpu_lane"]["cpu_list"])  # type: ignore[index]
    verify_non_overlapping_masks(gpu_masks, cpu_mask)
    env = build_runtime_env(
        mpi_ranks=advice["recommended_mpi_ranks"],
        omp_threads=advice["recommended_omp_threads"],
    )
    return {
        "advice": advice,
        "lane_plan": lane_plan,
        "runtime_env": {
            k: env[k]
            for k in (
                "OMP_NUM_THREADS",
                "OMP_PROC_BIND",
                "OMP_WAIT_POLICY",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
    }


def _cmd_topology(args: argparse.Namespace) -> dict[str, Any]:
    topo = detect_cpu_topology(logical_cpus_override=args.logical_cpus)
    return compute_disjoint_lane_plan(
        topo,
        gpu_slots=args.workers,
        cores_per_gpu_slot=args.cores_per_slot,
        reserve_os_core0=args.reserve_core0,
    )


def _cmd_env(args: argparse.Namespace) -> dict[str, Any]:
    return build_runtime_env(
        mpi_ranks=args.mpi_ranks,
        omp_threads=args.omp_threads,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="p8-dft-tuner",
        description="Portable DFT co-scheduling, MLIP manifold guard & Blackwell Ozaki advisor",
    )
    ap.add_argument("-V", "--version", action="version", version=f"%(prog)s {CLI_VERSION}")
    sub = ap.add_subparsers(dest="command", required=True)

    p_rec = sub.add_parser("recommend", help="Recommend Ozaki policy, MLIP guard, W*, CPU masks, and OMP env")
    p_rec.add_argument("--qe-in", type=pathlib.Path, default=None, help="Optional QE input file to inspect")
    p_rec.add_argument("--nat", type=int, default=None)
    p_rec.add_argument("--nbnd", type=int, default=None)
    p_rec.add_argument("--n-paw", type=int, default=None)
    p_rec.add_argument("--nks", type=int, default=None)
    p_rec.add_argument("--nspin", type=int, default=None)
    p_rec.add_argument("--npw-smooth", type=int, default=None)
    p_rec.add_argument("--hubbard-u", action="store_true")
    p_rec.add_argument("--cuda-version", default="13.2")
    p_rec.add_argument("--ozaki-mode", default="auto", choices=["auto", "off", "performant", "eager"])
    p_rec.add_argument("--mantissa-bits", type=int, default=53)
    p_rec.add_argument("--cudart-mode", default="static", choices=["static", "dynamic"])
    p_rec.add_argument("--gpu-vram-mib", type=int, default=48935)
    p_rec.add_argument("--vram-safe-cap-mib", type=int, default=43008)
    p_rec.add_argument(
        "--mlip-pre-relax",
        default="auto",
        choices=["auto", "none", "pbe_universal", "u_matched"],
    )
    p_rec.add_argument("--wsl2", action=argparse.BooleanOptionalAction, default=None)
    p_rec.add_argument("--logical-cpus", type=int, default=None)
    p_rec.add_argument("--strict", action="store_true")
    p_rec.add_argument("--allow-slowdown", action="store_true")

    p_top = sub.add_parser("topology", help="Inspect SMT topology and non-overlapping GPU/CPU masks")
    p_top.add_argument("--workers", type=int, default=3)
    p_top.add_argument("--cores-per-slot", type=int, default=4)
    p_top.add_argument("--reserve-core0", action="store_true")
    p_top.add_argument("--logical-cpus", type=int, default=None)

    p_env = sub.add_parser("env", help="Emit OpenMP/BLAS-guarded environment mapping")
    p_env.add_argument("--mpi-ranks", type=int, default=2)
    p_env.add_argument("--omp-threads", type=int, default=1)

    args = ap.parse_args(argv)
    try:
        if args.command == "recommend":
            res = _cmd_recommend(args)
        elif args.command == "topology":
            res = _cmd_topology(args)
        else:
            res = _cmd_env(args)
    except ValueError as exc:
        ap.error(str(exc))
        return 2
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

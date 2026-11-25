"""Tests for OpenMP guard, Ozaki VRAM-wall guard, mantissa crossover, MLIP rules, and adversarial QE inputs."""
from __future__ import annotations

import pathlib

import pytest

from p8_dft_tuner.blackwell_advisor import (
    OzakiPolicyViolation,
    advise_execution_regime,
    estimate_ozaki_crossover_nbnd,
    parse_qe_input_summary,
)
from p8_dft_tuner.env_guard import build_runtime_env


def test_env_guard_prevents_multirank_omp_pin_collapse() -> None:
    env_multi = build_runtime_env(
        mpi_ranks=2,
        omp_threads=1,
        base_env={"OMP_PROC_BIND": "close", "OMP_PLACES": "cores"},
    )
    assert env_multi["OMP_PROC_BIND"] == "false"
    assert "OMP_PLACES" not in env_multi
    assert env_multi["OPENBLAS_NUM_THREADS"] == "1"
    assert env_multi["MKL_NUM_THREADS"] == "1"

    env_single = build_runtime_env(mpi_ranks=1, omp_threads=4, prefer_bind_close_single_rank=True)
    assert env_single["OMP_PROC_BIND"] == "close"
    assert env_single["OMP_PLACES"] == "cores"
    assert env_single["OMP_NUM_THREADS"] == "4"


def test_env_guard_rejects_nonpositive_inputs() -> None:
    with pytest.raises(ValueError):
        build_runtime_env(mpi_ranks=0, omp_threads=1)
    with pytest.raises(ValueError):
        build_runtime_env(mpi_ranks=1, omp_threads=0)


def test_advisor_blocks_vram_wall_even_with_allow_slowdown_and_warns_wsl2() -> None:
    rec = advise_execution_regime(
        nat=60,
        nbnd=128,
        nks=4,
        npw_smooth=1438601,
        cuda_version="13.2",
        ozaki_mode="eager",
        cudart_mode="dynamic",
        in_wsl2=True,
        physical_cores=24,
        allow_slowdown=True,  # allow_slowdown MUST NOT bypass hardware VRAM wall
    )
    assert rec["recommended_ozaki_mode"] == "off"
    assert rec["hits_vram_capacity_wall"] is True
    assert rec["policy_action"] == "blocked_vram_capacity_wall"
    assert any("WSL2" in w and "cudart" in w.lower() for w in rec["warnings"])
    assert any("Blocked ozaki_mode='eager'" in w for w in rec["warnings"])

    with pytest.raises(OzakiPolicyViolation):
        advise_execution_regime(
            nat=108,
            nbnd=468,
            npw_smooth=1413619,
            cuda_version="13.2",
            ozaki_mode="eager",
            allow_slowdown=True,
            strict=True,
        )


def test_advisor_mantissa_dynamic_crossover_and_multiworker_vram_clamp() -> None:
    c39 = estimate_ozaki_crossover_nbnd(39)
    c53 = estimate_ozaki_crossover_nbnd(53)
    assert c39 < c53
    assert 350 <= c39 <= 500
    assert 480 <= c53 <= 680

    # Large wavefunction cell (~22 GiB per worker) clamps W* from 3 down to 1 to fit 42 GiB safe cap
    large = advise_execution_regime(
        nat=216,
        nbnd=432,
        n_paw=0,
        nks=2,
        npw_smooth=1_780_000,
        ozaki_mode="auto",
        vram_safe_cap_mib=43008,
    )
    assert large["recommended_concurrent_workers"] == 1
    assert large["recommended_ozaki_mode"] == "off"


def test_advisor_auto_ozaki_sweet_spot_and_intermediate_off_fallback() -> None:
    small = advise_execution_regime(
        nat=64,
        nbnd=128,
        n_paw=0,
        npw_smooth=352443,
        cuda_version="13.2",
        ozaki_mode="auto",
    )
    assert small["recommended_ozaki_mode"] == "performant"
    assert small["policy_action"] == "auto_small_cell_performant_sweet_spot"

    mid = advise_execution_regime(
        nat=75,
        nbnd=325,
        n_paw=0,
        npw_smooth=981709,
        cuda_version="13.2",
        ozaki_mode="auto",
    )
    assert mid["recommended_ozaki_mode"] == "off"
    assert mid["policy_action"] == "auto_cuda13_off_safe_default"


def test_mlip_functional_manifold_matching_rule() -> None:
    non_u = advise_execution_regime(nat=26, nbnd=114, n_paw=0, has_hubbard_u=False, mlip_pre_relax="auto")
    assert non_u["mlip_pre_relax_recommended"] is True
    assert non_u["recommended_mlip_mode"] == "pbe_universal"

    dft_u_auto = advise_execution_regime(nat=35, nbnd=168, n_paw=33, has_hubbard_u=True, mlip_pre_relax="auto")
    assert dft_u_auto["mlip_pre_relax_recommended"] is False
    assert dft_u_auto["mlip_policy"] == "blocked_pbe_mlip_dft_u_mismatch"

    dft_u_matched = advise_execution_regime(nat=35, nbnd=168, n_paw=33, has_hubbard_u=True, mlip_pre_relax="u_matched")
    assert dft_u_matched["mlip_pre_relax_recommended"] is True
    assert dft_u_matched["recommended_mlip_mode"] == "u_matched"


def test_advisor_rejects_invalid_numeric_inputs() -> None:
    for bad_kwargs in (
        {"nat": 0, "nbnd": 64},
        {"nat": 10, "nbnd": 0},
        {"nat": 10, "nbnd": 64, "n_paw": -1},
        {"nat": 10, "nbnd": 64, "nks": 0},
        {"nat": 10, "nbnd": 64, "nspin": 0},
        {"nat": 10, "nbnd": 64, "npw_smooth": 0},
        {"nat": 10, "nbnd": 64, "mantissa_bits": 16},
        {"nat": 10, "nbnd": 64, "gpu_vram_mib": 0},
    ):
        with pytest.raises(ValueError):
            advise_execution_regime(**bad_kwargs)  # type: ignore[arg-type]


def test_parse_qe_input_adversarial(tmp_path: pathlib.Path) -> None:
    # Fortran comments, D exponent, legacy lda_plus_u, pawless UPF exclusion, K_POINTS automatic
    qein = tmp_path / "adv.in"
    qein.write_text(
        "! Header comment with nat = 999 HUBBARD\n"
        "&SYSTEM\n"
        "  nat = 4 ! inline comment\n"
        "  nspin = 2\n"
        "  ecutwfc = 4.5d1 ! 45.0 Ry\n"
        "  lda_plus_u = .true.\n"
        "  Hubbard_U(1) = 7.0d0\n"
        "/\n"
        "ATOMIC_SPECIES\n"
        "  Cu 63.5 Cu.kjpaw.UPF ! real PAW\n"
        "  Si 28.1 Si_pawless.UPF ! NOT paw\n"
        "ATOMIC_POSITIONS angstrom\n"
        "  Cu 0 0 0\n"
        "  Cu 1 0 0\n"
        "  Si 0 1 0 ! Cu.paw.UPF decoy comment\n"
        "  Si 1 1 0\n"
        "K_POINTS automatic\n"
        "  2 2 1 0 0 0\n",
        encoding="utf-8",
    )
    meta = parse_qe_input_summary(qein)
    assert meta["nat"] == 4
    assert meta["nspin"] == 2
    assert meta["n_paw"] == 2  # excludes Si_pawless.UPF
    assert meta["has_hubbard_u"] is True
    assert meta["nks"] == 4
    assert abs(meta["ecutwfc_ry"] - 45.0) < 1e-6

    empty_in = tmp_path / "empty.in"
    empty_in.write_text("! only comments\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_qe_input_summary(empty_in)

    missing_pos = tmp_path / "nopos.in"
    missing_pos.write_text("&SYSTEM\n nat = 2\n/\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_qe_input_summary(missing_pos)

    bad_nat = tmp_path / "badnat.in"
    bad_nat.write_text("&SYSTEM\n nat = 0\n/\nATOMIC_POSITIONS\n H 0 0 0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_qe_input_summary(bad_nat)

    explicit_kp = tmp_path / "expk.in"
    explicit_kp.write_text(
        "&SYSTEM\n nat = 2\n/\nATOMIC_SPECIES\n Si 28.1 Si.uspp.UPF\n"
        "ATOMIC_POSITIONS\n Si 0 0 0\n Si 1 1 1\nK_POINTS crystal\n 3\n 0 0 0 1\n",
        encoding="utf-8",
    )
    assert parse_qe_input_summary(explicit_kp)["nks"] == 3


def test_advisor_intermediate_performant_warning_and_invalid_caps() -> None:
    mid_perf = advise_execution_regime(nat=75, nbnd=325, n_paw=0, ozaki_mode="performant")
    assert any("regress" in w for w in mid_perf["warnings"])
    with pytest.raises(ValueError):
        advise_execution_regime(nat=10, nbnd=64, vram_safe_cap_mib=0)
    with pytest.raises(ValueError):
        advise_execution_regime(nat=10, nbnd=64, physical_cores=0)


def test_parse_qe_input_gamma_and_lowercase_d_exponent(tmp_path: pathlib.Path) -> None:
    qein = tmp_path / "gamma_d.in"
    qein.write_text(
        "&SYSTEM\n  nat = 2\n  ecutwfc = 6.5D1\n  nbnd = 32\n/\n"
        "ATOMIC_SPECIES\n  Si 28.1 Si.uspp.UPF\n"
        "ATOMIC_POSITIONS crystal\n  Si 0 0 0\n  Si 0.25 0.25 0.25\n"
        "K_POINTS gamma\n",
        encoding="utf-8",
    )
    meta = parse_qe_input_summary(qein)
    assert meta["nks"] == 1
    assert meta["n_paw"] == 0
    assert abs(meta["ecutwfc_ry"] - 65.0) < 1e-6


def test_advisor_allow_slowdown_permits_small_matrix_when_vram_safe() -> None:
    rec = advise_execution_regime(
        nat=64,
        nbnd=128,
        n_paw=0,
        npw_smooth=352443,
        cuda_version="13.2",
        ozaki_mode="eager",
        allow_slowdown=True,
        vram_safe_cap_mib=43008,
    )
    assert rec["hits_vram_capacity_wall"] is False
    assert rec["recommended_ozaki_mode"] == "eager"
    assert rec["policy_action"] == "accepted"

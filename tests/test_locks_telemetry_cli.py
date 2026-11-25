"""Tests for lock manager (`slot_status` non-creating), fail-closed gate, telemetry, and CLI v1.1.0."""
from __future__ import annotations

import json
import pathlib
import pytest

from p8_dft_tuner.cli import CLI_VERSION, main as cli_main
from p8_dft_tuner.locks import (
    SlotAllocationError,
    SlotLockManager,
    verify_non_overlapping_masks,
)
from p8_dft_tuner.telemetry import (
    check_maintenance_gate,
    load_profile_with_fallback,
    record_run,
)


def test_slot_locks_status_no_side_effect_and_validation(tmp_path: pathlib.Path) -> None:
    lock_dir = tmp_path / "locks"
    mgr = SlotLockManager(lock_dir=lock_dir)
    # slot_status must NOT create empty lockfiles when probing an idle directory
    st_empty = mgr.slot_status(num_slots=3)
    assert all(s["busy"] is False for s in st_empty)
    assert list(lock_dir.glob("*.lock")) == []

    with pytest.raises(ValueError):
        mgr.slot_status(num_slots=0)
    with pytest.raises(ValueError):
        with mgr.acquire_first_free_slot(num_slots=0):
            pass

    with mgr.acquire_slot(0, metadata={"worker": "w0"}) as lease0:
        assert lease0.slot_id == 0
        with pytest.raises(SlotAllocationError):
            with mgr.acquire_slot(0, timeout_s=0.02):
                pass
        with mgr.acquire_first_free_slot(num_slots=3) as lease1:
            assert lease1.slot_id == 1

    assert verify_non_overlapping_masks(["0,2", "4,6"], "8,10") == []
    with pytest.raises(ValueError):
        verify_non_overlapping_masks(["0,2", "4,6"], "2,10")


def test_maintenance_gate_fail_closed_profile_and_telemetry(tmp_path: pathlib.Path) -> None:
    gate = tmp_path / "maint.flag"
    assert check_maintenance_gate(gate)["active"] is False
    gate.write_text('{"active": true, "reason": "bench"}', encoding="utf-8")
    assert check_maintenance_gate(gate)["active"] is True
    # Corrupt gate file fails closed (reports active=True)
    gate.write_text("{broken_json", encoding="utf-8")
    assert check_maintenance_gate(gate)["active"] is True

    prof = tmp_path / "profile.json"
    assert load_profile_with_fallback(prof)["profile_source"] == "defaults"
    prof.write_text('{"concurrent_gpu_workers": 2}', encoding="utf-8")
    loaded = load_profile_with_fallback(prof)
    assert loaded["concurrent_gpu_workers"] == 2

    prof.write_text("{corrupt_json", encoding="utf-8")
    assert load_profile_with_fallback(prof)["profile_source"] == "defaults_fallback_after_parse_error"

    tlog = tmp_path / "sub" / "runs.jsonl"
    assert record_run({"run_id": "t1", "wall_s": 12.5}, log_path=tlog) is True
    rows = [json.loads(line) for line in tlog.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["run_id"] == "t1"

    bad_dir = tmp_path / "not_a_dir"
    bad_dir.write_text("file", encoding="utf-8")
    assert record_run({"run_id": "t2"}, log_path=bad_dir / "runs.jsonl") is False


def test_cli_recommend_topology_env_version_and_validation(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as ver_exc:
        cli_main(["--version"])
    assert ver_exc.value.code == 0
    assert CLI_VERSION in capsys.readouterr().out

    rc = cli_main(
        [
            "recommend",
            "--nat",
            "60",
            "--nbnd",
            "128",
            "--nks",
            "4",
            "--npw-smooth",
            "1438601",
            "--ozaki-mode",
            "eager",
            "--cudart-mode",
            "dynamic",
            "--wsl2",
            "--logical-cpus",
            "48",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["advice"]["recommended_ozaki_mode"] == "off"
    assert out["advice"]["hits_vram_capacity_wall"] is True
    assert out["lane_plan"]["overlap_cpus"] == []

    assert cli_main(["topology", "--workers", "3", "--cores-per-slot", "4", "--logical-cpus", "48"]) == 0
    top = json.loads(capsys.readouterr().out)
    assert len(top["gpu_slots"]) == 3

    assert cli_main(["env", "--mpi-ranks", "2", "--omp-threads", "1"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["OMP_PROC_BIND"] == "false"

    for bad_argv in (
        ["recommend", "--nat", "0"],
        ["topology", "--workers", "0"],
        ["env", "--mpi-ranks", "0"],
    ):
        with pytest.raises(SystemExit) as exc:
            cli_main(bad_argv)
        assert exc.value.code == 2


def test_locks_negative_slot_and_cli_from_qe_file(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    mgr = SlotLockManager(lock_dir=tmp_path / "locks2")
    with pytest.raises(ValueError):
        with mgr.acquire_slot(-1):
            pass

    qein = tmp_path / "cuo.in"
    qein.write_text(
        "&SYSTEM\n  nat = 4\n  nspin = 2\n  lda_plus_u = .true.\n/\n"
        "ATOMIC_SPECIES\n  Cu 63.5 Cu.kjpaw.UPF\n  O 16.0 O.uspp.UPF\n"
        "ATOMIC_POSITIONS angstrom\n  Cu 0 0 0\n  Cu 1 0 0\n  O 0 1 0\n  O 0 2 0\n",
        encoding="utf-8",
    )
    assert cli_main(["recommend", "--qe-in", str(qein), "--ozaki-mode", "auto", "--logical-cpus", "48"]) == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["advice"]["mlip_policy"] == "blocked_pbe_mlip_dft_u_mismatch"
    assert rec["advice"]["recommended_ozaki_mode"] == "off"


def test_locks_slot_status_while_busy_and_release_cleanup(tmp_path: pathlib.Path) -> None:
    mgr = SlotLockManager(lock_dir=tmp_path / "locks_busy")
    with mgr.acquire_slot(0, metadata={"job": "alpha"}):
        statuses = mgr.slot_status(num_slots=2)
        assert statuses[0]["busy"] is True
        assert statuses[0]["holder"]["job"] == "alpha"
        assert statuses[1]["busy"] is False
    assert all(s["busy"] is False for s in mgr.slot_status(num_slots=2))

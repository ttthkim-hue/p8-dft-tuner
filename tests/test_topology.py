"""Tests for SMT-aware topology detection, Intel split fallback, and disjoint validation."""
from __future__ import annotations

import pathlib

import pytest

from p8_dft_tuner.topology import (
    compute_disjoint_lane_plan,
    detect_cpu_topology,
    parse_cpulist,
)


def _write_mock_sysfs(root: pathlib.Path, sibling_groups: list[tuple[int, int]]) -> pathlib.Path:
    for a, b in sibling_groups:
        for cid in (a, b):
            tdir = root / f"cpu{cid}" / "topology"
            tdir.mkdir(parents=True, exist_ok=True)
            (tdir / "thread_siblings_list").write_text(f"{a},{b}\n", encoding="utf-8")
    return root


def test_parse_cpulist_valid_and_rejects_reversed() -> None:
    assert parse_cpulist("0,2-4,8") == [0, 2, 3, 4, 8]
    with pytest.raises(ValueError):
        parse_cpulist("4-2")
    with pytest.raises(ValueError):
        parse_cpulist("-1")


def test_threadripper_adjacent_smt_and_disjoint_lanes(tmp_path: pathlib.Path) -> None:
    pairs = [(2 * c, 2 * c + 1) for c in range(24)]
    sysfs_root = _write_mock_sysfs(tmp_path / "sys_cpu", pairs)
    topo = detect_cpu_topology(sysfs_cpu_root=sysfs_root, cpuinfo_text="vendor_id : AuthenticAMD\n")
    assert topo.source == "sysfs"
    assert topo.physical_cores == 24
    assert topo.logical_cpus == 48
    assert topo.sibling_pairs[0] == (0, 1)
    assert topo.sibling_pairs[12] == (24, 25)
    assert topo.cpu_list_for_physical_cores([0, 1, 2, 3]) == "0,2,4,6"

    plan = compute_disjoint_lane_plan(topo, gpu_slots=3, cores_per_gpu_slot=4)
    assert plan["overlap_cpus"] == []
    slots = plan["gpu_slots"]
    assert len(slots) == 3  # type: ignore[arg-type]
    assert slots[0]["cpu_list"] == "0,2,4,6"  # type: ignore[index]
    assert slots[1]["cpu_list"] == "8,10,12,14"  # type: ignore[index]
    assert slots[2]["cpu_list"] == "16,18,20,22"  # type: ignore[index]
    assert plan["cpu_lane"]["core_count"] == 12  # type: ignore[index]
    assert plan["cpu_lane"]["cpu_list"] == "24,26,28,30,32,34,36,38,40,42,44,46"  # type: ignore[index]


def test_intel_defaults_to_split_smt_and_validates_overrides() -> None:
    topo = detect_cpu_topology(
        sysfs_cpu_root=pathlib.Path("/nonexistent_sysfs_dir"),
        cpuinfo_text="vendor_id : GenuineIntel",
        logical_cpus_override=32,
    )
    assert topo.source == "heuristic_split_smt"
    assert topo.physical_cores == 16
    assert topo.sibling_pairs[0] == (0, 16)
    plan = compute_disjoint_lane_plan(topo, gpu_slots=2, cores_per_gpu_slot=4, reserve_os_core0=True)
    assert plan["reserved_os_cores"] == [0]
    assert plan["overlap_cpus"] == []
    assert plan["gpu_slots"][0]["physical_cores"] == [1, 2, 3, 4]  # type: ignore[index]

    with pytest.raises(ValueError):
        compute_disjoint_lane_plan(topo, gpu_slots=5, cores_per_gpu_slot=4)
    with pytest.raises(ValueError):
        compute_disjoint_lane_plan(topo, gpu_slots=0, cores_per_gpu_slot=4)
    with pytest.raises(ValueError):
        detect_cpu_topology(logical_cpus_override=8, smt_pairing_override="bogus")
    with pytest.raises(ValueError):
        detect_cpu_topology(logical_cpus_override=0)


def test_topology_core_bounds_and_physical_cores_helper() -> None:
    from p8_dft_tuner.topology import physical_cores_to_cpu_list

    topo = detect_cpu_topology(logical_cpus_override=8, smt_pairing_override="adjacent")
    assert physical_cores_to_cpu_list([0, 2], topology=topo) == "0,4"
    assert physical_cores_to_cpu_list([0, 2], include_smt_siblings=True, topology=topo) == "0,1,4,5"
    with pytest.raises(ValueError):
        topo.cpu_list_for_physical_cores([4])

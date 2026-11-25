"""Hardware SMT topology detection and non-overlapping CPU-GPU lane partitioning."""
from __future__ import annotations

import dataclasses
import os
import pathlib
import re
from typing import Iterable


@dataclasses.dataclass(frozen=True)
class CpuTopology:
    """Detected physical-to-logical CPU SMT sibling mapping."""

    physical_cores: int
    logical_cpus: int
    vendor: str
    sibling_pairs: tuple[tuple[int, ...], ...]
    source: str

    def cpu_list_for_physical_cores(
        self,
        core_indices: Iterable[int],
        *,
        include_smt_siblings: bool = False,
    ) -> str:
        """Convert 0-based physical core indices into a `taskset -c` CPU list string."""
        selected: list[int] = []
        seen: set[int] = set()
        for c in core_indices:
            if c < 0 or c >= len(self.sibling_pairs):
                raise ValueError(f"Physical core index {c} out of bounds [0, {len(self.sibling_pairs) - 1}]")
            sibs = self.sibling_pairs[c]
            chosen = sibs if include_smt_siblings else (sibs[0],)
            for cpu_id in chosen:
                if cpu_id not in seen:
                    seen.add(cpu_id)
                    selected.append(cpu_id)
        return ",".join(str(x) for x in selected)

    def cpuset_for_physical_cores(
        self,
        core_indices: Iterable[int],
        *,
        include_smt_siblings: bool = False,
    ) -> set[int]:
        """Return set of OS logical CPU IDs corresponding to `core_indices`."""
        s = self.cpu_list_for_physical_cores(core_indices, include_smt_siblings=include_smt_siblings)
        return {int(x) for x in s.split(",") if x} if s else set()


def parse_cpulist(spec: str) -> list[int]:
    """Parse Linux sysfs cpulist syntax (e.g. '0,24' or '0-3,8-11') into sorted ints."""
    out: list[int] = []
    for tok in spec.strip().split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            lo_s, hi_s = tok.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo < 0 or hi < 0 or lo > hi:
                raise ValueError(f"Invalid CPU range '{tok}' in cpulist '{spec}'")
            out.extend(range(lo, hi + 1))
        else:
            val = int(tok)
            if val < 0:
                raise ValueError(f"Negative CPU index {val} in cpulist '{spec}'")
            out.append(val)
    return sorted(set(out))


def _detect_from_sysfs(sysfs_cpu_root: pathlib.Path) -> list[tuple[int, ...]] | None:
    if not sysfs_cpu_root.is_dir():
        return None
    groups: set[tuple[int, ...]] = set()
    for entry in sorted(sysfs_cpu_root.iterdir()):
        if not re.fullmatch(r"cpu\d+", entry.name):
            continue
        sib_file = entry / "topology" / "thread_siblings_list"
        if not sib_file.is_file():
            continue
        try:
            sibs = tuple(parse_cpulist(sib_file.read_text(encoding="utf-8", errors="ignore")))
        except OSError:
            continue
        if sibs:
            groups.add(sibs)
    if not groups:
        return None
    return sorted(groups, key=lambda g: g[0])


def _detect_vendor_from_cpuinfo(cpuinfo_text: str | None) -> str:
    txt = cpuinfo_text or ""
    if not txt:
        p = pathlib.Path("/proc/cpuinfo")
        if p.is_file():
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")[:16384]
            except OSError:
                txt = ""
    low = txt.lower()
    if "authenticamd" in low or "amd " in low or "threadripper" in low:
        return "AuthenticAMD"
    if "genuineintel" in low or "intel" in low:
        return "GenuineIntel"
    return "unknown"


def detect_cpu_topology(
    *,
    sysfs_cpu_root: pathlib.Path | None = None,
    cpuinfo_text: str | None = None,
    logical_cpus_override: int | None = None,
    smt_pairing_override: str | None = None,
) -> CpuTopology:
    """Detect physical core -> SMT sibling pairs portably across bare-metal Linux, WSL2, or mocks.

    Supports both adjacent Zen/Threadripper SMT pairs `(2c, 2c+1)` and split-bank Intel/NUMA
    SMT pairs `(c, c + N)` discovered automatically from Linux sysfs `thread_siblings_list`.
    """
    vendor = _detect_vendor_from_cpuinfo(cpuinfo_text)
    root = sysfs_cpu_root if sysfs_cpu_root is not None else pathlib.Path("/sys/devices/system/cpu")
    sysfs_groups = _detect_from_sysfs(root) if logical_cpus_override is None else None

    if sysfs_groups:
        logical = sum(len(g) for g in sysfs_groups)
        return CpuTopology(
            physical_cores=len(sysfs_groups),
            logical_cpus=logical,
            vendor=vendor,
            sibling_pairs=tuple(sysfs_groups),
            source="sysfs",
        )

    if logical_cpus_override is not None and int(logical_cpus_override) < 1:
        raise ValueError(f"logical_cpus_override must be >= 1, got {logical_cpus_override}")
    if smt_pairing_override is not None and smt_pairing_override.lower() not in {"adjacent", "split", "none"}:
        raise ValueError(f"Unsupported smt_pairing_override='{smt_pairing_override}' (expected adjacent|split|none)")

    logical = logical_cpus_override if logical_cpus_override is not None else (os.cpu_count() or 1)
    logical = max(1, int(logical))
    if smt_pairing_override is not None:
        mode = smt_pairing_override.lower()
    elif vendor == "GenuineIntel" and logical >= 2 and logical % 2 == 0:
        mode = "split"
    elif (vendor == "AuthenticAMD" or logical % 2 == 0) and logical >= 2 and logical % 2 == 0:
        mode = "adjacent"
    else:
        mode = "none"

    if mode == "adjacent" and logical >= 2 and logical % 2 == 0:
        pairs = tuple((2 * c, 2 * c + 1) for c in range(logical // 2))
        return CpuTopology(
            physical_cores=len(pairs),
            logical_cpus=logical,
            vendor=vendor,
            sibling_pairs=pairs,
            source="heuristic_adjacent_smt",
        )
    if mode == "split" and logical >= 2 and logical % 2 == 0:
        half = logical // 2
        pairs = tuple((c, c + half) for c in range(half))
        return CpuTopology(
            physical_cores=half,
            logical_cpus=logical,
            vendor=vendor,
            sibling_pairs=pairs,
            source="heuristic_split_smt",
        )

    pairs = tuple((c,) for c in range(logical))
    return CpuTopology(
        physical_cores=logical,
        logical_cpus=logical,
        vendor=vendor,
        sibling_pairs=pairs,
        source="flat_fallback",
    )


def physical_cores_to_cpu_list(
    physical_cores: Iterable[int],
    *,
    include_smt_siblings: bool = False,
    topology: CpuTopology | None = None,
) -> str:
    """Map physical core indices to OS logical CPU IDs using detected SMT topology."""
    topo = topology if topology is not None else detect_cpu_topology()
    return topo.cpu_list_for_physical_cores(physical_cores, include_smt_siblings=include_smt_siblings)


def compute_disjoint_lane_plan(
    topology: CpuTopology | None = None,
    *,
    gpu_slots: int = 3,
    cores_per_gpu_slot: int = 4,
    reserve_os_core0: bool = False,
    include_smt_siblings_gpu: bool = False,
    include_smt_siblings_cpu: bool = False,
) -> dict[str, object]:
    """Partition physical cores into strictly disjoint GPU feeder slots and CPU batch lane."""
    if int(gpu_slots) < 1:
        raise ValueError(f"gpu_slots must be >= 1, got {gpu_slots}")
    if int(cores_per_gpu_slot) < 1:
        raise ValueError(f"cores_per_gpu_slot must be >= 1, got {cores_per_gpu_slot}")
    topo = topology if topology is not None else detect_cpu_topology()
    start_core = 1 if (reserve_os_core0 and topo.physical_cores >= gpu_slots * cores_per_gpu_slot + 2) else 0
    total_avail = topo.physical_cores - start_core
    needed_gpu = int(gpu_slots) * int(cores_per_gpu_slot)
    if needed_gpu > total_avail:
        raise ValueError(
            f"Insufficient physical cores ({total_avail} available) for {gpu_slots} slots x {cores_per_gpu_slot} cores"
        )

    gpu_slot_phys: list[list[int]] = []
    gpu_slot_masks: list[str] = []
    gpu_cpu_union: set[int] = set()
    cursor = start_core
    for _ in range(gpu_slots):
        phys = list(range(cursor, cursor + cores_per_gpu_slot))
        cursor += cores_per_gpu_slot
        mask = topo.cpu_list_for_physical_cores(phys, include_smt_siblings=include_smt_siblings_gpu)
        gpu_slot_phys.append(phys)
        gpu_slot_masks.append(mask)
        gpu_cpu_union.update(int(x) for x in mask.split(",") if x)

    cpu_phys = list(range(cursor, topo.physical_cores))
    cpu_mask = topo.cpu_list_for_physical_cores(cpu_phys, include_smt_siblings=include_smt_siblings_cpu)
    cpu_set = {int(x) for x in cpu_mask.split(",") if x} if cpu_mask else set()
    overlap = sorted(gpu_cpu_union & cpu_set)
    if overlap:
        raise AssertionError(f"Non-disjoint CPU allocation detected: {overlap}")

    return {
        "topology_source": topo.source,
        "physical_cores": topo.physical_cores,
        "logical_cpus": topo.logical_cpus,
        "reserved_os_cores": [0] if start_core == 1 else [],
        "gpu_slots": [
            {"slot": i, "physical_cores": gpu_slot_phys[i], "cpu_list": gpu_slot_masks[i]}
            for i in range(gpu_slots)
        ],
        "cpu_lane": {
            "physical_cores": cpu_phys,
            "core_count": len(cpu_phys),
            "cpu_list": cpu_mask,
        },
        "overlap_cpus": overlap,
    }

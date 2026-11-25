"""p8_dft_tuner: Portable scheduler, OpenMP guard, and Blackwell Ozaki advisor for DFT."""
from __future__ import annotations

from .blackwell_advisor import (
    OzakiPolicyViolation,
    advise_execution_regime,
    parse_qe_input_summary,
)
from .env_guard import build_runtime_env
from .locks import (
    SlotAllocationError,
    SlotLockManager,
    verify_non_overlapping_masks,
)
from .telemetry import (
    check_maintenance_gate,
    load_profile_with_fallback,
    record_run,
)
from .topology import (
    CpuTopology,
    compute_disjoint_lane_plan,
    detect_cpu_topology,
    physical_cores_to_cpu_list,
)

__version__ = "1.1.0"

__all__ = [
    "CpuTopology",
    "OzakiPolicyViolation",
    "SlotAllocationError",
    "SlotLockManager",
    "__version__",
    "advise_execution_regime",
    "build_runtime_env",
    "check_maintenance_gate",
    "compute_disjoint_lane_plan",
    "detect_cpu_topology",
    "load_profile_with_fallback",
    "parse_qe_input_summary",
    "physical_cores_to_cpu_list",
    "record_run",
    "verify_non_overlapping_masks",
]

"""Safe OpenMP, BLAS, and CUDA runtime environment builders preventing multi-rank pin regressions."""
from __future__ import annotations

import copy
from typing import Mapping


def build_runtime_env(
    *,
    mpi_ranks: int = 1,
    omp_threads: int = 1,
    omp_places: str = "cores",
    prefer_bind_close_single_rank: bool = True,
    cuda_device_order: str = "PCI_BUS_ID",
    visible_devices: str | None = None,
    extra_env: Mapping[str, str] | None = None,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Construct an OpenMP/BLAS/CUDA-safe environment mapping for Quantum ESPRESSO launches.

    Enforces critical multi-rank correctness & performance invariants:
    - When `mpi_ranks >= 2`: forces `OMP_PROC_BIND=false` and clears `OMP_PLACES` so MPI ranks
      never collapse onto Core 0 under OpenMP affinity (avoiding >2x slowdown observed in Stage 2.1).
    - Clamps nested CPU BLAS threads (`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
      `VECLIB_MAXIMUM_THREADS=1`) to avoid thread oversubscription across concurrent workers.
    - Sets `OMP_WAIT_POLICY=PASSIVE` so idle worker threads yield cleanly to co-scheduled jobs.
    """
    if int(mpi_ranks) < 1:
        raise ValueError(f"mpi_ranks must be >= 1, got {mpi_ranks}")
    if int(omp_threads) < 1:
        raise ValueError(f"omp_threads must be >= 1, got {omp_threads}")
    env = dict(base_env) if base_env is not None else {}
    ranks = int(mpi_ranks)
    threads = int(omp_threads)

    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_WAIT_POLICY"] = "PASSIVE"
    env["OMP_DYNAMIC"] = "FALSE"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["CUDA_DEVICE_ORDER"] = cuda_device_order
    if visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(visible_devices)

    if ranks >= 2:
        # Multi-rank guard: prevent multiple MPI ranks from contending on identical OMP_PLACES
        env["OMP_PROC_BIND"] = "false"
        env.pop("OMP_PLACES", None)
    else:
        if threads > 1 and prefer_bind_close_single_rank:
            env["OMP_PROC_BIND"] = "close"
            env["OMP_PLACES"] = omp_places
        else:
            env["OMP_PROC_BIND"] = "false"
            env.pop("OMP_PLACES", None)

    if extra_env:
        for k, v in extra_env.items():
            env[str(k)] = str(v)
    return copy.deepcopy(env)

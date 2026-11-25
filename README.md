# `p8-dft-tuner` (`v1.1.0`)

**Portable Hardware-Topology, Concurrency-Slot, OpenMP-Guard, MLIP-Manifold & Blackwell Ozaki Advisor for CPU-GPU Plane-Wave DFT**

`p8-dft-tuner` is a standalone, zero-dependency, MIT-licensed Python package distilled from empirical Quantum ESPRESSO 7.6 optimization and Blackwell tensor-core FP64-emulation studies. It enforces scheduling and numerical-safety invariants without hardcoding any workstation-specific paths:

1. **SMT-Aware Topology & Disjoint Lane Partitioning (`p8_dft_tuner.topology`)**
   Detects Linux `sysfs` `thread_siblings_list` (including adjacent Zen/Threadripper `(2c, 2c+1)` pairs and split-bank Intel/NUMA SMT layouts) and partitions physical cores into strictly non-overlapping GPU feeder slots and concurrent CPU batch lanes.
2. **Cross-Platform Slot Locking (`p8_dft_tuner.locks`)**
   Uses non-blocking/timed `fcntl.flock` on POSIX and `msvcrt.locking` on Windows hosts in configurable lock directories (`P8_DFT_TUNER_LOCK_DIR` or OS temp directory) to serialize `W` GPU slots while asserting zero CPU core overlap.
3. **Multi-Rank OpenMP Anti-Pinning Guard (`p8_dft_tuner.env_guard`)**
   Prevents `OMP_PROC_BIND=close`/`OMP_PLACES` from collapsing multiple `pw.x` MPI ranks onto Core 0 (`OMP_PROC_BIND=false` when `mpi_ranks >= 2`), while clamping nested OpenBLAS/MKL threads.
4. **Blackwell CUDA 13 Ozaki Regime & VRAM-Capacity Wall Advisor (`p8_dft_tuner.blackwell_advisor`)**
   - Blocks `ozaki_mode="eager"` unconditionally when workspace memory spikes (+13.7 to +32.2 GiB across multi-band cells) breach the GPU VRAM capacity wall before reaching compute break-even (even under `--allow-slowdown`), or when `nbnd` falls below the mantissa-aware analytical crossover cliff.
   - Selects `ozaki_mode="performant"` when `paw_atoms == 0` in compact pure-USPP/ONCV cells (`nbnd <= 160`, `npw_smooth <= 600k`) and sets `CUBLAS_EMULATE_DOUBLE_PRECISION=0` (`ozaki_mode="off"`) in intermediate-to-large regimes (`nbnd = 256..468`) to preserve CUDA 13.2 speedups without memory bloat.
   - Verifies multi-worker footprint (`w * vram_off <= safe_cap`) and warns when running dynamic `libcudart.so` under WSL2 + CUDA $\ge 13.0$.
5. **MLIP+QE Functional-Manifold Matching Guard (`p8_dft_tuner.blackwell_advisor`)**
   Enables universal GGA/PBE MLIP pre-relaxation (`pbe_universal`) on matching non-$U$ PBE systems, while blocking generic PBE MLIP pre-relaxation on `DFT+U` structures (`blocked_pbe_mlip_dft_u_mismatch`) unless an explicit `u_matched` potential is supplied.
6. **Fail-Closed Maintenance Gate & Fail-Safe Telemetry (`p8_dft_tuner.telemetry`)**
   Maintains fail-closed maintenance gate checks, live JSON profile reload with safe fallback, and lock-protected JSONL telemetry logging (`P8_DFT_TUNER_TELEMETRY_FILE`).

## Installation

```bash
pip install -e .
```

## Quick CLI Usage

```bash
# Check package version
p8-dft-tuner --version

# Check Ozaki VRAM-wall guard + MLIP DFT+U manifold guard + disjoint CPU/GPU plan
p8-dft-tuner recommend --nat 108 --nbnd 468 --n-paw 0 --npw-smooth 1413619 --ozaki-mode eager --wsl2 --cudart-mode dynamic

# Non-U PBE MoS2 relaxation advice (same-manifold MLIP enabled, CUDA 13.2 off default)
p8-dft-tuner recommend --nat 26 --nbnd 114 --n-paw 0 --mlip-pre-relax auto

# Inspect SMT-aware disjoint lane masks for 3 GPU slots x 4 cores
p8-dft-tuner topology --workers 3 --cores-per-slot 4

# Emit OpenMP-safe environment for 2 MPI ranks
p8-dft-tuner env --mpi-ranks 2 --omp-threads 1
```

## Running Tests

```bash
pytest -v
```

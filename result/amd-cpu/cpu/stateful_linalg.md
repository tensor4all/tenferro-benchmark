# CPU Stateful Incremental QR

- Suite: `cpu/stateful_linalg`
- Raw run: `data/results/amd-cpu/cpu/stateful_linalg/20261013_091743`

## CPU Information

- Model: `Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `16`
- Sockets: `1`
- Cores per socket: `16`
- Threads per core: `1`
- NUMA nodes: `unknown`
- Python platform: `Linux-7.0.14-linuxkit-x86_64-with-glibc2.39`

F64, compact column-major inputs, initial rank 5, nine width-3 appends, final rank 32. Each workflow interleaves append, R snapshot, and the three new Q columns. Release execution; seven fresh processes per case, alternating path order; three warmups and three measured 64-workflow batches per process. One entered session surrounds all shared-route samples. Fresh routes include per-operation session entry and are explicitly workflow diagnostics. Both eager paths reuse one runtime; capture is inactive and values untracked.

Initial factors, input blocks, restoration, output-retention containers, runtime/backend/session setup, validation and destruction are untimed. All compact states, R/Q snapshots and explicit-Q BCGS2 intermediates survive clock stop. Numerical checks cover every interleaved prefix, final reconstruction and orthogonality. Attribution is a separate diagnostic with individual call clocks; ordinary workflow rows contain no component clocks.

Medians are microseconds per complete sequence; CoV uses the seven process medians. CoV >10%, missing provider/thread observations, missing replicates or a process-median batch shorter than 1 ms is inconclusive. No result here claims a speedup against historical hardware.

| Rows | Path | Threads | Median sequence µs | Process CoV | Minimum process-median batch ms | Reconstruction error | Orthogonality error | Status |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 64 | attributed | 1 | 169.395 | 1.8% | 10.700 | 2.98e-16 | 8.88e-16 | observed |
| 64 | bcgs2 | 1 | 1258.937 | 1.4% | 78.283 | 4.48e-16 | 1.11e-15 | observed |
| 64 | bcgs2-attributed | 1 | 1263.680 | 1.3% | 78.739 | 4.48e-16 | 1.11e-15 | observed |
| 64 | eager-fresh | 1 | 377.905 | 3.1% | 23.373 | 2.98e-16 | 8.88e-16 | observed |
| 64 | eager-shared | 1 | 320.065 | 1.0% | 20.298 | 2.98e-16 | 8.88e-16 | observed |
| 64 | fresh | 1 | 212.367 | 1.6% | 13.124 | 2.98e-16 | 8.88e-16 | observed |
| 64 | shared | 1 | 170.356 | 1.8% | 10.738 | 2.98e-16 | 8.88e-16 | observed |
| 128 | attributed | 1 | 178.197 | 3.8% | 11.199 | 5.18e-16 | 1.11e-15 | observed |
| 128 | bcgs2 | 1 | 1326.558 | 3.7% | 83.115 | 5.09e-16 | 1.11e-15 | observed |
| 128 | bcgs2-attributed | 1 | 1323.633 | 1.8% | 82.849 | 5.09e-16 | 1.11e-15 | observed |
| 128 | eager-fresh | 1 | 419.963 | 1.8% | 26.486 | 5.18e-16 | 1.11e-15 | observed |
| 128 | eager-shared | 1 | 364.186 | 2.2% | 22.849 | 5.18e-16 | 1.11e-15 | observed |
| 128 | fresh | 1 | 207.004 | 2.0% | 13.030 | 5.18e-16 | 1.11e-15 | observed |
| 128 | shared | 1 | 176.253 | 1.7% | 10.957 | 5.18e-16 | 1.11e-15 | observed |
| 256 | attributed | 1 | 261.979 | 24.9% | 16.369 | 5.52e-16 | 2.66e-15 | inconclusive |
| 256 | bcgs2 | 1 | 1427.607 | 1.8% | 89.091 | 1.32e-15 | 2.22e-15 | observed |
| 256 | bcgs2-attributed | 1 | 1443.829 | 1.7% | 90.647 | 1.32e-15 | 2.22e-15 | observed |
| 256 | eager-fresh | 1 | 507.838 | 1.3% | 31.903 | 5.52e-16 | 2.66e-15 | observed |
| 256 | eager-shared | 1 | 452.531 | 3.8% | 28.467 | 5.52e-16 | 2.66e-15 | observed |
| 256 | fresh | 1 | 290.670 | 1.7% | 18.327 | 5.52e-16 | 2.66e-15 | observed |
| 256 | shared | 1 | 261.099 | 4.9% | 16.317 | 5.52e-16 | 2.66e-15 | observed |
| 64 | attributed | 4 | 203.768 | 7.2% | 12.591 | 3.85e-16 | 7.77e-16 | observed |
| 64 | bcgs2 | 4 | 7668.270 | 5.3% | 487.027 | 4.48e-16 | 1.11e-15 | observed |
| 64 | bcgs2-attributed | 4 | 7727.689 | 4.2% | 484.938 | 4.48e-16 | 1.11e-15 | observed |
| 64 | eager-fresh | 4 | 2285.393 | 7.1% | 130.160 | 3.85e-16 | 7.77e-16 | observed |
| 64 | eager-shared | 4 | 2097.117 | 9.3% | 122.979 | 3.85e-16 | 7.77e-16 | observed |
| 64 | fresh | 4 | 245.914 | 5.3% | 14.656 | 3.85e-16 | 7.77e-16 | observed |
| 64 | shared | 4 | 207.493 | 6.0% | 12.035 | 3.85e-16 | 7.77e-16 | observed |
| 128 | attributed | 4 | 220.306 | 7.4% | 13.732 | 3.52e-16 | 1.11e-15 | observed |
| 128 | bcgs2 | 4 | 7780.584 | 1.4% | 493.907 | 5.09e-16 | 1.11e-15 | observed |
| 128 | bcgs2-attributed | 4 | 7841.317 | 2.1% | 490.668 | 5.09e-16 | 1.11e-15 | observed |
| 128 | eager-fresh | 4 | 2352.911 | 7.1% | 138.593 | 3.52e-16 | 1.11e-15 | observed |
| 128 | eager-shared | 4 | 2201.343 | 5.6% | 126.290 | 3.52e-16 | 1.11e-15 | observed |
| 128 | fresh | 4 | 257.140 | 7.8% | 15.739 | 3.52e-16 | 1.11e-15 | observed |
| 128 | shared | 4 | 218.859 | 11.9% | 13.597 | 3.52e-16 | 1.11e-15 | inconclusive |
| 256 | attributed | 4 | 271.167 | 5.2% | 16.170 | 5.49e-16 | 2.66e-15 | observed |
| 256 | bcgs2 | 4 | 8117.525 | 3.2% | 497.561 | 1.32e-15 | 2.22e-15 | observed |
| 256 | bcgs2-attributed | 4 | 7999.539 | 1.7% | 498.176 | 1.32e-15 | 2.22e-15 | observed |
| 256 | eager-fresh | 4 | 2349.113 | 4.6% | 138.903 | 5.49e-16 | 2.66e-15 | observed |
| 256 | eager-shared | 4 | 2220.674 | 5.4% | 130.099 | 5.49e-16 | 2.66e-15 | observed |
| 256 | fresh | 4 | 320.980 | 4.5% | 19.334 | 5.49e-16 | 2.66e-15 | observed |
| 256 | shared | 4 | 260.290 | 5.0% | 15.903 | 5.49e-16 | 2.66e-15 | observed |

| Rows | Threads | Append µs | R µs | New Q µs | Complete/(append+R+Q) | Fresh/shared | Eager fresh/shared concrete | Eager shared/shared concrete | Compact append/BCGS2 append | Quality |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 1 | 94.897 | 11.671 | 60.764 | 1.018 | 1.247 | 2.218 | 1.879 | 0.077 | observed diagnostic ratios |
| 128 | 1 | 91.910 | 6.690 | 76.624 | 1.006 | 1.174 | 2.383 | 2.066 | 0.071 | observed diagnostic ratios |
| 256 | 1 | 139.749 | 6.775 | 113.192 | 1.005 | 1.113 | 1.945 | 1.733 | 0.099 | inconclusive |
| 64 | 4 | 103.377 | 7.399 | 91.824 | 1.024 | 1.185 | 11.014 | 10.107 | 0.013 | observed diagnostic ratios |
| 128 | 4 | 113.735 | 7.576 | 95.837 | 1.008 | 1.175 | 10.751 | 10.058 | 0.015 | inconclusive |
| 256 | 4 | 145.685 | 8.461 | 112.791 | 0.975 | 1.233 | 9.025 | 8.532 | 0.018 | observed diagnostic ratios |

BCGS2 uses two QᵀB projections, two residual corrections, QR of the new residual block, and public concatenation to update Q/R. It is an explicit-Q diagnostic using the same provider, not accumulated full QR. Compact/BCGS2 compares the separately attributed append work; extraction/retention clocks are excluded from that component. Ratios between fresh and shared paths include session policy differences, so they do not isolate eager wrapper cost.

## run_t1

```yaml
timestamp: '2026-10-13T09:17:43Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 8b13220c1e0397a29a27be65557c24105352a35b
  dirty: false
  features:
  - blas-mkl
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: mkl
  version: 2026.0.1
  root: /opt/intel/oneapi/mkl/latest
  library: /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so
environment:
  hostname: f88adb25e9bd
  os: Linux-7.0.14-linuxkit-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  env:
    OMP_NUM_THREADS: '1'
    OMP_THREAD_LIMIT: '1'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: amd-cpu
    BENCHMARK_COMMIT: 43f1f3ed14a850e2fde79d15b169955d0762c851
    OPENBLAS_ROOT: /opt/openblas
    MKLROOT: /opt/intel/oneapi/mkl/latest
    OPENBLAS_NUM_THREADS: '1'
    GOTO_NUM_THREADS: '1'
    MKL_NUM_THREADS: '1'
    MKL_DYNAMIC: 'FALSE'
    VECLIB_MAXIMUM_THREADS: '1'
    VECLIB_NUM_THREADS: '1'
    NUMEXPR_NUM_THREADS: '1'
    BLIS_NUM_THREADS: '1'
    PJRT_NPROC: '1'
    XLA_FLAGS: --xla_cpu_multi_thread_eigen=false --xla_cpu_experimental_ynn_fusion_type=
      intra_op_parallelism_threads=1
    JULIA_NUM_THREADS: '1'
    CARGO_TARGET_DIR: /tmp/tenferro-candidates-target
    USE_CUDA: '0'
collection:
  harness_commit: 43f1f3ed14a850e2fde79d15b169955d0762c851
  build_profile: release
  case_manifest: data/instances/stateful_linalg.json
  manifest_version: 1
  process_replicates: 7
  samples_per_process: 3
  warmups: 3
  initialization_passes: 1
  process_order: ascending case order on even rounds; reverse on odd rounds
  command: scripts/run_cpu_stateful_linalg.sh
  affinity: none
  scope: setup untimed; retained-output whole interleaved workflow batches
```

## run_t4

```yaml
timestamp: '2026-10-13T09:19:27Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 8b13220c1e0397a29a27be65557c24105352a35b
  dirty: false
  features:
  - blas-mkl
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: mkl
  version: 2026.0.1
  root: /opt/intel/oneapi/mkl/latest
  library: /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so
environment:
  hostname: f88adb25e9bd
  os: Linux-7.0.14-linuxkit-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  env:
    OMP_NUM_THREADS: '4'
    OMP_THREAD_LIMIT: '4'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '4'
    BENCHMARK_TARGET_PROFILE: amd-cpu
    BENCHMARK_COMMIT: 43f1f3ed14a850e2fde79d15b169955d0762c851
    OPENBLAS_ROOT: /opt/openblas
    MKLROOT: /opt/intel/oneapi/mkl/latest
    OPENBLAS_NUM_THREADS: '4'
    GOTO_NUM_THREADS: '4'
    MKL_NUM_THREADS: '4'
    MKL_DYNAMIC: 'FALSE'
    VECLIB_MAXIMUM_THREADS: '4'
    VECLIB_NUM_THREADS: '4'
    NUMEXPR_NUM_THREADS: '4'
    BLIS_NUM_THREADS: '4'
    PJRT_NPROC: '4'
    XLA_FLAGS: --xla_cpu_multi_thread_eigen=true --xla_cpu_experimental_ynn_fusion_type=
      intra_op_parallelism_threads=4
    JULIA_NUM_THREADS: '4'
    CARGO_TARGET_DIR: /tmp/tenferro-candidates-target
    USE_CUDA: '0'
collection:
  harness_commit: 43f1f3ed14a850e2fde79d15b169955d0762c851
  build_profile: release
  case_manifest: data/instances/stateful_linalg.json
  manifest_version: 1
  process_replicates: 7
  samples_per_process: 3
  warmups: 3
  initialization_passes: 1
  process_order: ascending case order on even rounds; reverse on odd rounds
  command: scripts/run_cpu_stateful_linalg.sh
  affinity: none
  scope: setup untimed; retained-output whole interleaved workflow batches
```

## Case status

- Manifest: `cpu/stateful_linalg` version 1, coverage `full`, effort `standard`
- Expected 42, selected 42, executed 42, unsupported 0, failed 0, missing 0, noisy 2.
- Complete: **yes** — only executed cases count as covered; unsupported, failed, missing and unselected cases never do.
- Noisy: `incremental_qr_m256_attributed@t1`, `incremental_qr_m128_shared@t4`


Wall time `wall_phases.json`:
```json
{
  "build_and_validation_seconds": 1.818129524,
  "metadata_startup_and_collection_seconds": 374.01111874
}
```

Wall time `wall_t1.json`:
```json
{
  "collection_wall_seconds": 99.08717461300012,
  "phase_policy": "one process per case: input/setup, warmup, timed intervals, correctness and teardown; per-case wall plus binary initialization_ms recorded",
  "process_order": "ascending case order on even rounds; reverse on odd rounds"
}
```

Wall time `wall_t4.json`:
```json
{
  "collection_wall_seconds": 264.569193414,
  "phase_policy": "one process per case: input/setup, warmup, timed intervals, correctness and teardown; per-case wall plus binary initialization_ms recorded",
  "process_order": "ascending case order on even rounds; reverse on odd rounds"
}
```

# CPU Small-Work Benchmark Results

- Suite: `cpu/small_work`
- Raw run: `data/results/zen5-cpu/cpu/small_work/20261028_011041`

## CPU Information

- Model: `AMD Ryzen AI 9 HX 470 w/ Radeon 890M`
- Vendor: `AuthenticAMD`
- Logical CPUs: `24`
- Sockets: `1`
- Cores per socket: `12`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3,12-15; 4-11,16-23`
- CPU boost: `enabled`

## run_t1

Full metadata: `data/results/zen5-cpu/cpu/small_work/20261028_011041/run_t1.yaml`

```yaml
timestamp: '2026-10-28T01:10:42Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
  commit: 8fec1ca70852b930c092e7177c30bb8af7594a57
  dirty: false
  features:
  - blas-mkl
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: mkl
  version: 2026.0.2
  root: /opt/intel/oneapi/mkl/latest
  library: /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so
environment:
  hostname: bcfefbe95606
  os: Linux-7.0.0-38-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: taskset
    requested_measured_cpus: '0'
    observer_cpus: 4-11,16-23
    override_of: null
    thread_budget: '1'
    measured_cpus: '0'
    l3_domains:
    - 0-3,12-15
    - 4-11,16-23
    cpu_boost: enabled
    scaling_governor: performance
    scaling_driver: amd-pstate-epp
  env:
    OMP_NUM_THREADS: '1'
    OMP_THREAD_LIMIT: '1'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: zen5-cpu
    BENCHMARK_COMMIT: 3f31f38e0491084b9414035754a3809ef0f5ae94
    BENCH_COMMAND_BUDGET_SECONDS: '120'
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
    PUBLICATION_GATE_PROFILE: full
    USE_CUDA: '0'
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.1 (58cf8f9c3 2026-10-13)
    BENCH_INSTANCE: null
```

## run_t4

Full metadata: `data/results/zen5-cpu/cpu/small_work/20261028_011041/run_t4.yaml`

```yaml
timestamp: '2026-10-28T01:10:48Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
  commit: 8fec1ca70852b930c092e7177c30bb8af7594a57
  dirty: false
  features:
  - blas-mkl
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: mkl
  version: 2026.0.2
  root: /opt/intel/oneapi/mkl/latest
  library: /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so
environment:
  hostname: bcfefbe95606
  os: Linux-7.0.0-38-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: taskset
    requested_measured_cpus: 0-3
    observer_cpus: 4-11,16-23
    override_of: null
    thread_budget: '4'
    measured_cpus: 0-3
    l3_domains:
    - 0-3,12-15
    - 4-11,16-23
    cpu_boost: enabled
    scaling_governor: performance
    scaling_driver: amd-pstate-epp
  env:
    OMP_NUM_THREADS: '4'
    OMP_THREAD_LIMIT: '4'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '4'
    BENCHMARK_TARGET_PROFILE: zen5-cpu
    BENCHMARK_COMMIT: 3f31f38e0491084b9414035754a3809ef0f5ae94
    BENCH_COMMAND_BUDGET_SECONDS: '120'
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
    PUBLICATION_GATE_PROFILE: full
    USE_CUDA: '0'
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.1 (58cf8f9c3 2026-10-13)
    BENCH_INSTANCE: null
```

## run_t8

Full metadata: `data/results/zen5-cpu/cpu/small_work/20261028_011041/run_t8.yaml`

```yaml
timestamp: '2026-10-28T01:10:56Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
  commit: 8fec1ca70852b930c092e7177c30bb8af7594a57
  dirty: false
  features:
  - blas-mkl
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: mkl
  version: 2026.0.2
  root: /opt/intel/oneapi/mkl/latest
  library: /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so
environment:
  hostname: bcfefbe95606
  os: Linux-7.0.0-38-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: taskset
    requested_measured_cpus: 4-11
    observer_cpus: 0-3,12-15
    override_of: null
    thread_budget: '8'
    measured_cpus: 4-11
    l3_domains:
    - 0-3,12-15
    - 4-11,16-23
    cpu_boost: enabled
    scaling_governor: performance
    scaling_driver: amd-pstate-epp
  env:
    OMP_NUM_THREADS: '8'
    OMP_THREAD_LIMIT: '8'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '8'
    BENCHMARK_TARGET_PROFILE: zen5-cpu
    BENCHMARK_COMMIT: 3f31f38e0491084b9414035754a3809ef0f5ae94
    BENCH_COMMAND_BUDGET_SECONDS: '120'
    OPENBLAS_ROOT: /opt/openblas
    MKLROOT: /opt/intel/oneapi/mkl/latest
    OPENBLAS_NUM_THREADS: '8'
    GOTO_NUM_THREADS: '8'
    MKL_NUM_THREADS: '8'
    MKL_DYNAMIC: 'FALSE'
    VECLIB_MAXIMUM_THREADS: '8'
    VECLIB_NUM_THREADS: '8'
    NUMEXPR_NUM_THREADS: '8'
    BLIS_NUM_THREADS: '8'
    PJRT_NPROC: '8'
    XLA_FLAGS: --xla_cpu_multi_thread_eigen=true --xla_cpu_experimental_ynn_fusion_type=
      intra_op_parallelism_threads=8
    JULIA_NUM_THREADS: '8'
    PUBLICATION_GATE_PROFILE: full
    USE_CUDA: '0'
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.1 (58cf8f9c3 2026-10-13)
    BENCH_INSTANCE: null
```

## Case status

- Manifest: `cpu/small_work` version 3, coverage `quick`, effort `standard`
- Expected 216, selected 216, executed 216, unsupported 0, failed 0, missing 0, noisy 5.
- Complete: **yes** — only executed cases count as covered; unsupported, failed, missing and unselected cases never do.
- Noisy: `nonad_solve_n16_col_major_contiguous_concrete-shared@t1`, `einsum_f64_prepared-repeat_n32_single@t4`, `einsum_c64_prepared-repeat_n32_single@t4`, `einsum_f64_prepared-repeat_n32_single@t8`, `einsum_c64_prepared-repeat_n32_single@t8`

Numerical checks: **216/216 recorded rows passed** (case × thread count).

## Timing

Sequential release runs; explicit initialization and warmups, then calibration toward the declared batch duration (2 ms by default), then the measured batches of the recorded effort (standard: 3 warmups and 15 batches; scan: a low-repetition screen that only marks suspects). Legacy single-call routes start calibration at 1024 workflows; matched non-AD and dependent-chain routes start at 16 and bound retained memory. Each batch uses one wall-clock interval; all outputs are retained until it stops. Statistics divide batch time by iterations. Median and inclusive IQR are in nanoseconds (ns); CoV is sample standard deviation / mean. NOISY means CoV > 10%, a descriptive label only. No noisy rows are excluded.

Workflow time is the total for one call or the complete dependent chain. The separate ns/op column divides the chain median by its operation count, not an independently measured call. Setup rows measure preparation only, not execution.

Inputs and backend construction, independent numerical checks and AD backward checks are outside timing. Returned outputs/plans and every dependent-chain intermediate are retained until after each batch interval; retention containers are prepared before the clock. value downloads/checks are outside. Eager AD rows measure forward execution/recording, not backward.

| Case ID | Operation | API route | Route (path / output / repr / session) | Dtype | Shape | Layout | Workflow | Threads | Provider | Operations/batch | Median batch ms | Median workflow ns | IQR ns | CoV | ns/op (chain only) | Status |
|---|---|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 4096 | 1.809752 | 441.83 | 1.14 | 2.2% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 1 | blas | 5120 | 2.274899 | 4443.16 | 14.58 | 0.2% | 444.32 | ok |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 4096 | 1.816455 | 443.47 | 0.71 | 2.0% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 1 | blas | 5120 | 2.224536 | 4344.80 | 21.68 | 0.3% | 434.48 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 4096 | 2.021744 | 493.59 | 1.86 | 4.3% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 1 | blas | 5120 | 2.369254 | 4627.45 | 71.51 | 2.9% | 462.75 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 3.110098 | 1518.60 | 2.62 | 0.1% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 2048 | 3.293375 | 1608.09 | 2.99 | 0.2% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 1.926408 | 1881.26 | 6.33 | 2.5% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 3.255084 | 1589.40 | 1.45 | 0.1% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 1 | blas | 2048 | 3.257018 | 1590.34 | 1.79 | 0.2% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.211072 | 2159.25 | 5.65 | 2.0% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 4096 | 2.576705 | 629.08 | 2.43 | 0.8% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 4096 | 2.782925 | 679.43 | 0.63 | 0.7% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.144349 | 2094.09 | 183.72 | 4.4% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 2048 | 3.328459 | 1625.22 | 2.82 | 0.2% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 3.435828 | 3355.30 | 30.68 | 3.5% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 1.759640 | 1718.40 | 11.59 | 0.4% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 6.341037 | 6192.42 | 127.48 | 1.9% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 4096 | 2.560806 | 625.20 | 1.39 | 1.4% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 4096 | 2.582005 | 630.37 | 1.15 | 1.5% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 2048 | 1.553371 | 758.48 | 1.41 | 1.7% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 4096 | 2.153867 | 525.85 | 0.74 | 1.7% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 4096 | 2.135914 | 521.46 | 0.95 | 1.6% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 4096 | 2.193730 | 535.58 | 1.14 | 1.8% | — | ok |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 2.742191 | 2677.92 | 20.56 | 0.6% | — | ok |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 2048 | 3.368504 | 1644.78 | 1.07 | 6.7% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 2048 | 3.258992 | 1591.30 | 2.00 | 2.4% | — | ok |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 2048 | 3.326877 | 1624.45 | 4.84 | 0.2% | — | ok |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 4096 | 3.295850 | 804.65 | 3.62 | 0.7% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 4096 | 3.084841 | 753.14 | 1.28 | 0.9% | — | ok |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 1024 | 2.232833 | 2180.50 | 2.94 | 0.4% | — | ok |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 2048 | 3.002970 | 1466.29 | 1.38 | 0.8% | — | ok |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 2048 | 2.914237 | 1422.97 | 1.67 | 0.7% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.510484 | 4903.29 | 19.96 | 0.2% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 3.193862 | 1559.50 | 2.43 | 1.5% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.478054 | 4839.95 | 12.03 | 0.3% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 3.120067 | 1523.47 | 6.19 | 1.5% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 2.829441 | 11052.50 | 17.77 | 0.4% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.336623 | 4563.72 | 15.44 | 0.2% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 2.826566 | 11041.27 | 11.66 | 0.2% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.348244 | 4586.41 | 12.83 | 0.3% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 4096 | 2.892898 | 706.27 | 0.79 | 3.4% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.687229 | 2624.25 | 13.10 | 0.4% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 4096 | 2.860428 | 698.35 | 0.94 | 3.7% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.672391 | 2609.76 | 5.82 | 0.2% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.794076 | 1364.29 | 5.91 | 2.2% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 3.553484 | 3470.20 | 10.41 | 0.3% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.782334 | 1358.56 | 2.45 | 2.0% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 3.544809 | 3461.73 | 6.89 | 0.2% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 4096 | 2.538085 | 619.65 | 1.77 | 3.6% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.576795 | 2516.40 | 14.51 | 0.4% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 4096 | 2.551389 | 622.90 | 1.40 | 3.9% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.599247 | 2538.33 | 14.27 | 0.4% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 1.958227 | 1912.33 | 3.12 | 1.0% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.394119 | 4676.01 | 4.57 | 0.1% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 1.992039 | 1945.35 | 5.65 | 1.8% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.412253 | 4711.43 | 9.60 | 0.4% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.401052 | 4689.55 | 12.23 | 0.5% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 3.047232 | 2975.81 | 12.29 | 0.6% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.378250 | 4645.02 | 12.85 | 0.2% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 3.048715 | 2977.26 | 8.11 | 0.6% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.047691 | 3999.40 | 13.94 | 0.3% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.122549 | 2072.80 | 194.63 | 11.0% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.058581 | 4020.67 | 12.32 | 0.5% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.158214 | 2107.63 | 190.41 | 4.4% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.402354 | 4692.10 | 5.47 | 0.1% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 4096 | 2.801649 | 684.00 | 0.50 | 3.8% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 4096 | 2.592234 | 632.87 | 1.31 | 3.8% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.377108 | 4642.79 | 7.47 | 0.4% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 3.068731 | 2996.81 | 7.88 | 0.6% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.119644 | 2069.96 | 191.66 | 4.5% | — | ok |
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 4096 | 2.482742 | 606.14 | 5.99 | 1.4% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 4 | blas | 5120 | 2.990388 | 5840.60 | 14.89 | 0.3% | 584.06 | ok |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 4096 | 2.536642 | 619.30 | 4.27 | 1.1% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 4 | blas | 5120 | 3.112743 | 6079.58 | 26.81 | 0.3% | 607.96 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 4096 | 2.617911 | 639.14 | 3.09 | 3.3% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 4 | blas | 2560 | 1.658374 | 6478.02 | 31.09 | 0.4% | 647.80 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 3.599850 | 1757.74 | 5.35 | 0.4% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 2048 | 3.516166 | 1716.88 | 3.73 | 0.3% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.180175 | 2129.08 | 4.57 | 0.3% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 3.574253 | 1745.24 | 2.31 | 0.1% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 4 | blas | 2048 | 3.605220 | 1760.36 | 3.77 | 0.2% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.439723 | 2382.54 | 19.17 | 2.1% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 4096 | 3.281513 | 801.15 | 1.03 | 0.5% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 4096 | 3.428214 | 836.97 | 2.97 | 0.7% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.326345 | 2271.82 | 194.29 | 4.2% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 2048 | 3.679647 | 1796.70 | 2.15 | 0.1% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 39.112065 | 38195.38 | 10553.19 | 47.9% | — | NOISY |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 2.011796 | 1964.64 | 20.77 | 0.8% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 7.326123 | 7154.42 | 104.12 | 96.9% | — | NOISY |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 4096 | 3.248934 | 793.20 | 4.64 | 0.9% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 4096 | 3.365308 | 821.61 | 4.47 | 1.1% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 2048 | 1.974647 | 964.18 | 4.63 | 1.5% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 4096 | 2.821146 | 688.76 | 2.85 | 1.2% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 4096 | 2.838228 | 692.93 | 1.02 | 1.2% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 4096 | 2.883601 | 704.00 | 1.77 | 1.2% | — | ok |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 2.989907 | 2919.83 | 10.49 | 0.6% | — | ok |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 2048 | 3.832398 | 1871.29 | 3.27 | 0.2% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 2048 | 3.671301 | 1792.63 | 3.07 | 0.1% | — | ok |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 1024 | 1.887146 | 1842.92 | 4.70 | 0.3% | — | ok |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 2048 | 2.008549 | 980.74 | 3.72 | 0.4% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 4096 | 3.816509 | 931.76 | 1.54 | 0.6% | — | ok |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 1024 | 2.434664 | 2377.60 | 6.72 | 0.5% | — | ok |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 2048 | 3.378462 | 1649.64 | 2.65 | 2.4% | — | ok |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 2048 | 3.330894 | 1626.41 | 1.94 | 6.6% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.340220 | 4570.74 | 11.14 | 0.2% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.809513 | 1767.10 | 15.68 | 0.9% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.324160 | 4539.38 | 13.37 | 0.3% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 3.558764 | 1737.68 | 6.02 | 1.9% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 3.022527 | 11806.75 | 27.06 | 0.3% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.675097 | 5224.80 | 15.39 | 0.3% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 3.034810 | 11854.73 | 36.40 | 0.5% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.615848 | 5109.08 | 10.74 | 0.3% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 1.720699 | 840.19 | 3.88 | 2.4% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 3.012398 | 2941.79 | 18.79 | 0.4% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 1.736048 | 847.68 | 3.12 | 1.9% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.953710 | 2884.48 | 16.04 | 0.4% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 3.110008 | 1518.56 | 1.79 | 1.9% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 1.935675 | 3780.62 | 14.64 | 0.4% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 3.125797 | 1526.27 | 1.98 | 1.9% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 1.933271 | 3775.92 | 18.29 | 0.3% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 1.624212 | 793.07 | 2.79 | 2.0% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.872280 | 2804.96 | 9.59 | 0.3% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 4096 | 3.221503 | 786.50 | 0.79 | 2.9% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.847565 | 2780.83 | 12.60 | 0.4% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.217204 | 2165.24 | 4.62 | 0.4% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.574692 | 5028.70 | 13.45 | 0.2% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.256045 | 2203.17 | 2.99 | 0.7% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.564955 | 5009.68 | 12.74 | 0.2% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.553824 | 4987.94 | 25.54 | 0.4% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 3.291242 | 3214.10 | 9.81 | 0.9% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.579320 | 5037.73 | 14.88 | 0.7% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 3.327308 | 3249.32 | 10.22 | 0.6% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.194822 | 4286.76 | 14.48 | 0.7% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.358604 | 2303.32 | 196.76 | 4.2% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.216562 | 4329.22 | 9.29 | 0.2% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.331935 | 2277.28 | 202.55 | 4.4% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.340811 | 4571.90 | 14.72 | 0.2% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 1.724166 | 841.88 | 1.15 | 2.3% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 4096 | 3.207728 | 783.14 | 1.71 | 3.4% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.585172 | 5049.16 | 21.15 | 0.4% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 3.303865 | 3226.43 | 8.52 | 0.5% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.315073 | 2260.81 | 196.82 | 4.3% | — | ok |
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 8 | blas | 2048 | 1.942257 | 948.37 | 4.24 | 1.4% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 8 | blas | 2560 | 2.342675 | 9151.07 | 29.43 | 0.4% | 915.11 | ok |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 8 | blas | 2048 | 1.934483 | 944.57 | 6.32 | 1.3% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 8 | blas | 2560 | 2.384752 | 9315.44 | 40.54 | 0.3% | 931.54 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 8 | blas | 2048 | 2.040087 | 996.14 | 6.66 | 2.4% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 8 | blas | 2560 | 2.485157 | 9707.64 | 63.16 | 0.8% | 970.76 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 1024 | 2.675398 | 2612.69 | 6.42 | 0.2% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 8 | blas | 1024 | 2.644130 | 2582.16 | 10.29 | 0.4% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 1024 | 3.336916 | 3258.71 | 13.36 | 0.4% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 8 | blas | 1024 | 2.637929 | 2576.10 | 11.70 | 0.4% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 8 | blas | 1024 | 2.696416 | 2633.22 | 7.62 | 0.2% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 8 | blas | 1024 | 3.723849 | 3636.57 | 15.67 | 0.8% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 2048 | 2.437860 | 1190.36 | 11.36 | 0.8% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 8 | blas | 2048 | 2.610578 | 1274.70 | 3.57 | 0.2% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 1024 | 3.622402 | 3537.50 | 278.03 | 3.9% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 8 | blas | 1024 | 2.804095 | 2738.37 | 8.11 | 0.3% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 8 | blas | 1024 | 5.952824 | 5813.30 | 13081.06 | 103.1% | — | NOISY |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 8 | blas | 1024 | 3.045480 | 2974.10 | 24.48 | 0.4% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 8 | blas | 1024 | 11.030418 | 10771.89 | 71.72 | 98.7% | — | NOISY |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 8 | blas | 2048 | 2.582116 | 1260.80 | 8.15 | 1.3% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 8 | blas | 2048 | 2.663406 | 1300.49 | 8.70 | 1.1% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 8 | blas | 1024 | 1.526061 | 1490.29 | 6.30 | 1.0% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 8 | blas | 2048 | 2.217594 | 1082.81 | 4.01 | 1.2% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 8 | blas | 2048 | 2.255274 | 1101.21 | 4.87 | 1.1% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 8 | blas | 2048 | 2.230419 | 1089.07 | 3.19 | 1.2% | — | ok |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 8 | blas | 1024 | 4.502663 | 4397.13 | 10.17 | 0.5% | — | ok |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 8 | blas | 1024 | 2.869986 | 2802.72 | 6.80 | 0.4% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 8 | blas | 1024 | 2.855460 | 2788.54 | 11.24 | 0.3% | — | ok |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 8 | blas | 1024 | 2.761517 | 2696.79 | 6.32 | 0.2% | — | ok |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 8 | blas | 2048 | 3.075655 | 1501.78 | 3.92 | 0.2% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 8 | blas | 2048 | 2.955794 | 1443.26 | 4.79 | 0.3% | — | ok |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 8 | blas | 1024 | 3.823803 | 3734.18 | 13.42 | 0.5% | — | ok |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 8 | blas | 1024 | 2.603396 | 2542.38 | 9.34 | 0.9% | — | ok |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 8 | blas | 1024 | 2.523589 | 2464.44 | 5.06 | 0.6% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 512 | 4.018683 | 7848.99 | 20.73 | 0.3% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 1024 | 2.784950 | 2719.68 | 14.50 | 0.6% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 512 | 3.873776 | 7565.97 | 28.12 | 0.3% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 1024 | 2.712566 | 2648.99 | 9.58 | 0.3% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 128 | 2.358484 | 18425.66 | 107.62 | 0.5% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 256 | 2.030299 | 7930.86 | 74.69 | 1.7% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 128 | 2.352112 | 18375.88 | 162.09 | 0.5% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 256 | 2.003250 | 7825.20 | 63.55 | 0.6% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 2048 | 2.642147 | 1290.11 | 5.20 | 2.0% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 512 | 2.300527 | 4493.22 | 24.45 | 0.4% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 2048 | 2.699572 | 1318.15 | 7.52 | 2.0% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 512 | 2.276402 | 4446.10 | 9.70 | 0.4% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 1024 | 2.393919 | 2337.81 | 12.53 | 1.2% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 512 | 2.964010 | 5789.08 | 38.99 | 0.7% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 1024 | 2.456906 | 2399.32 | 13.31 | 0.7% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 512 | 2.926109 | 5715.06 | 11.48 | 0.3% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 2048 | 2.379223 | 1161.73 | 7.08 | 2.2% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 512 | 2.197778 | 4292.54 | 51.42 | 0.6% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 2048 | 2.447308 | 1194.97 | 3.79 | 2.2% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 512 | 2.183842 | 4265.32 | 58.90 | 0.7% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 1024 | 3.305068 | 3227.61 | 18.75 | 1.0% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 256 | 1.953939 | 7632.57 | 28.62 | 0.3% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 1024 | 3.385295 | 3305.95 | 70.64 | 2.2% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 256 | 1.955231 | 7637.62 | 38.39 | 0.4% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 256 | 1.924835 | 7518.89 | 40.15 | 0.4% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 512 | 2.477384 | 4838.64 | 30.19 | 1.1% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 256 | 1.957896 | 7648.03 | 26.85 | 1.4% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 512 | 2.494305 | 4871.69 | 22.00 | 0.7% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 512 | 3.350321 | 6543.60 | 26.38 | 0.4% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 1024 | 3.566950 | 3483.35 | 294.71 | 4.1% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 512 | 3.361221 | 6564.88 | 40.40 | 0.5% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 1024 | 3.508472 | 3426.24 | 304.23 | 4.3% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 512 | 4.012501 | 7836.92 | 50.72 | 0.4% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 2048 | 2.662334 | 1299.97 | 4.23 | 2.0% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 2048 | 2.452427 | 1197.47 | 6.53 | 2.3% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 256 | 1.974347 | 7712.29 | 23.93 | 0.5% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 512 | 2.503472 | 4889.59 | 19.00 | 0.4% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 1024 | 3.567210 | 3483.60 | 293.78 | 4.2% | — | ok |

## Matched non-AD eager comparisons (#98)

All requested QR/SVD outputs are retained. Ordinary einsum uses the same parsed integer notation and default Auto strategy; planning/cache lookup stays inside both public calls. These are public-call diagnostics, not prepared execution or an isolated AD cost. Fresh pairs include session entry; shared pairs exclude it. Views are constructed untimed and consumed as views on both routes. No output to_tensor conversion is timed. Capture is inactive by construction in each fresh process, with no capture_trace guard; inputs and validation outputs are checked untracked.

| Operation | Shape | Layout | Session | Threads | Direct ns | Eager ns | Eager − direct ns | Eager/direct | Quality |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 1 | 4903.29 | 1559.50 | -3343.79 | 0.318 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | shared | 1 | 4839.95 | 1523.47 | -3316.48 | 0.315 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 4 | 4570.74 | 1767.10 | -2803.64 | 0.387 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | shared | 4 | 4539.38 | 1737.68 | -2801.70 | 0.383 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 8 | 7848.99 | 2719.68 | -5129.31 | 0.347 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | shared | 8 | 7565.97 | 2648.99 | -4916.98 | 0.350 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 1 | 4711.43 | 1945.35 | -2766.08 | 0.413 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | shared | 1 | 4676.01 | 1912.33 | -2763.68 | 0.409 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 4 | 5009.68 | 2203.17 | -2806.51 | 0.440 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | shared | 4 | 5028.70 | 2165.24 | -2863.46 | 0.431 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 8 | 7637.62 | 3305.95 | -4331.67 | 0.433 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | shared | 8 | 7632.57 | 3227.61 | -4404.96 | 0.423 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 1 | 11052.50 | 4563.72 | -6488.78 | 0.413 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | shared | 1 | 11041.27 | 4586.41 | -6454.86 | 0.415 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 4 | 11806.75 | 5224.80 | -6581.95 | 0.443 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | shared | 4 | 11854.73 | 5109.08 | -6745.65 | 0.431 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 8 | 18425.66 | 7930.86 | -10494.80 | 0.430 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | shared | 8 | 18375.88 | 7825.20 | -10550.68 | 0.426 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 1 | 706.27 | 2624.25 | 1917.98 | 3.716 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | shared | 1 | 698.35 | 2609.76 | 1911.41 | 3.737 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 4 | 840.19 | 2941.79 | 2101.60 | 3.501 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | shared | 4 | 847.68 | 2884.48 | 2036.80 | 3.403 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 8 | 1290.11 | 4493.22 | 3203.11 | 3.483 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | shared | 8 | 1318.15 | 4446.10 | 3127.95 | 3.373 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 1 | 2977.26 | 4645.02 | 1667.76 | 1.560 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | shared | 1 | 2975.81 | 4689.55 | 1713.74 | 1.576 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 4 | 3249.32 | 5037.73 | 1788.41 | 1.550 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | shared | 4 | 3214.10 | 4987.94 | 1773.84 | 1.552 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 8 | 4871.69 | 7648.03 | 2776.34 | 1.570 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | shared | 8 | 4838.64 | 7518.89 | 2680.25 | 1.554 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 1 | 619.65 | 2516.40 | 1896.75 | 4.061 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | shared | 1 | 622.90 | 2538.33 | 1915.43 | 4.075 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 4 | 793.07 | 2804.96 | 2011.89 | 3.537 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | shared | 4 | 786.50 | 2780.83 | 1994.33 | 3.536 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 8 | 1161.73 | 4292.54 | 3130.81 | 3.695 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | shared | 8 | 1194.97 | 4265.32 | 3070.35 | 3.569 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 1 | 2107.63 | 4020.67 | 1913.04 | 1.908 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | shared | 1 | 2072.80 | 3999.40 | 1926.60 | 1.929 | inconclusive |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 4 | 2277.28 | 4329.22 | 2051.94 | 1.901 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | shared | 4 | 2303.32 | 4286.76 | 1983.44 | 1.861 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 8 | 3426.24 | 6564.88 | 3138.64 | 1.916 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | shared | 8 | 3483.35 | 6543.60 | 3060.25 | 1.879 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1364.29 | 3470.20 | 2105.91 | 2.544 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | shared | 1 | 1358.56 | 3461.73 | 2103.17 | 2.548 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 4 | 1518.56 | 3780.62 | 2262.06 | 2.490 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | shared | 4 | 1526.27 | 3775.92 | 2249.65 | 2.474 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 8 | 2337.81 | 5789.08 | 3451.27 | 2.476 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | shared | 8 | 2399.32 | 5715.06 | 3315.74 | 2.382 | observed; no speedup claim |

## Separate allocation diagnostics

Instrumented executable is built separately after latency collection. Calling-thread Rust allocator requests only; excludes foreign BLAS and worker-thread allocations. Realloc counts its requested new size, not net retained bytes. Output destruction and instrumentation setup are excluded. Helper references and multi-output retention use stack storage. No allocation-instrumented run contributes latency samples.

| Corresponding case ID | Threads | Median allocation requests/op | Median requested bytes/op | Status |
|---|---:|---:|---:|---|
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | 1 | 116 | 12951 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | 1 | 15 | 4120 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | 1 | 116 | 12951 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-shared | 1 | 15 | 4120 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | 1 | 265 | 51829 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | 1 | 48 | 20952 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | 1 | 265 | 51829 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | 1 | 48 | 20952 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | 1 | 6 | 1280 | passed |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | 1 | 39 | 15515 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-shared | 1 | 6 | 1280 | passed |
| nonad_qr_n2_col_major_contiguous_eager-shared | 1 | 39 | 15515 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | 1 | 8 | 2088 | passed |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | 1 | 46 | 17651 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-shared | 1 | 8 | 2088 | passed |
| nonad_svd_n2_col_major_contiguous_eager-shared | 1 | 46 | 17651 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | 1 | 3 | 80 | passed |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | 1 | 35 | 16195 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-shared | 1 | 3 | 80 | passed |
| nonad_solve_n2_col_major_contiguous_eager-shared | 1 | 35 | 16195 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-shared | 1 | 15 | 4120 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | 1 | 116 | 12951 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | 1 | 15 | 4120 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | 1 | 116 | 12951 | passed |
| nonad_qr_n16_col_major_contiguous_eager-shared | 1 | 39 | 19547 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-shared | 1 | 6 | 5312 | passed |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | 1 | 39 | 19547 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | 1 | 6 | 5312 | passed |
| nonad_solve_n16_col_major_contiguous_eager-shared | 1 | 35 | 18211 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-shared | 1 | 3 | 4112 | passed |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | 1 | 35 | 18211 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | 1 | 3 | 4112 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | 1 | 116 | 12951 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | 1 | 6 | 1280 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | 1 | 3 | 80 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | 1 | 116 | 12951 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | 1 | 6 | 5312 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | 1 | 3 | 4112 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | 4 | 116 | 12951 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | 4 | 15 | 4120 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | 4 | 116 | 12951 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-shared | 4 | 15 | 4120 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | 4 | 265 | 51829 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | 4 | 48 | 20952 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | 4 | 265 | 51829 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | 4 | 48 | 20952 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | 4 | 6 | 1280 | passed |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | 4 | 39 | 15515 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-shared | 4 | 6 | 1280 | passed |
| nonad_qr_n2_col_major_contiguous_eager-shared | 4 | 39 | 15515 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | 4 | 8 | 2088 | passed |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | 4 | 46 | 17651 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-shared | 4 | 8 | 2088 | passed |
| nonad_svd_n2_col_major_contiguous_eager-shared | 4 | 46 | 17651 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | 4 | 3 | 80 | passed |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | 4 | 35 | 16195 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-shared | 4 | 3 | 80 | passed |
| nonad_solve_n2_col_major_contiguous_eager-shared | 4 | 35 | 16195 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-shared | 4 | 15 | 4120 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | 4 | 116 | 12951 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | 4 | 15 | 4120 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | 4 | 116 | 12951 | passed |
| nonad_qr_n16_col_major_contiguous_eager-shared | 4 | 39 | 19547 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-shared | 4 | 6 | 5312 | passed |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | 4 | 39 | 19547 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | 4 | 6 | 5312 | passed |
| nonad_solve_n16_col_major_contiguous_eager-shared | 4 | 35 | 18211 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-shared | 4 | 3 | 4112 | passed |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | 4 | 35 | 18211 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | 4 | 3 | 4112 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | 4 | 116 | 12951 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | 4 | 6 | 1280 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | 4 | 3 | 80 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | 4 | 116 | 12951 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | 4 | 6 | 5312 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | 4 | 3 | 4112 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | 8 | 116 | 12951 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | 8 | 15 | 4120 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | 8 | 116 | 12951 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-shared | 8 | 15 | 4120 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | 8 | 265 | 51829 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | 8 | 48 | 20952 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | 8 | 265 | 51829 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | 8 | 48 | 20952 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | 8 | 6 | 1280 | passed |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | 8 | 39 | 15515 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-shared | 8 | 6 | 1280 | passed |
| nonad_qr_n2_col_major_contiguous_eager-shared | 8 | 39 | 15515 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | 8 | 8 | 2088 | passed |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | 8 | 46 | 17651 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-shared | 8 | 8 | 2088 | passed |
| nonad_svd_n2_col_major_contiguous_eager-shared | 8 | 46 | 17651 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | 8 | 3 | 80 | passed |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | 8 | 35 | 16195 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-shared | 8 | 3 | 80 | passed |
| nonad_solve_n2_col_major_contiguous_eager-shared | 8 | 35 | 16195 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-shared | 8 | 15 | 4120 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | 8 | 116 | 12951 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | 8 | 15 | 4120 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | 8 | 116 | 12951 | passed |
| nonad_qr_n16_col_major_contiguous_eager-shared | 8 | 39 | 19547 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-shared | 8 | 6 | 5312 | passed |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | 8 | 39 | 19547 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | 8 | 6 | 5312 | passed |
| nonad_solve_n16_col_major_contiguous_eager-shared | 8 | 35 | 18211 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-shared | 8 | 3 | 4112 | passed |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | 8 | 35 | 18211 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | 8 | 3 | 4112 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | 8 | 116 | 12951 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | 8 | 6 | 1280 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | 8 | 3 | 80 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | 8 | 116 | 12951 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | 8 | 6 | 5312 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | 8 | 3 | 4112 | passed |

## Per-route timing boundaries

Shared-session routes enter/exit the session once around warmup/calibration/all samples, outside timing. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions reuse the entered executor context. Earlier BLAS revisions included per-operation executor entry; consult the recorded tenferro commit. Fresh routes enter/exit for each operation. Prepared-repeat excludes plan preparation; compiled-repeat excludes tracing/compilation and runtime construction. Prepared-into-repeat and dot-general-into-shared write into one destination preallocated outside timing (no per-call output allocation) and re-check it after the last batch.

- **add / concrete-shared** — inside: add, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **einsum / concrete-fresh** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), per-call session entry/exit diagnostic, ordinary einsum default Auto planning/cache lookup included equally; parsed integer notation; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **einsum / concrete-scoped** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), per-call session entry/exit diagnostic, ordinary einsum default Auto planning/cache lookup included equally; parsed integer notation; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction, caller-held engine construction; no execution scope exists after tenferro-rs #2206.
- **einsum / concrete-shared** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), session entry excluded, ordinary einsum default Auto planning/cache lookup included equally; parsed integer notation; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **einsum / dot-general-into-shared** — inside: dot_general_read_into; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, dot_general_config_construction, output_preallocation, output_destruction, operation_config_construction.
- **einsum / eager-no-ad** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), per-call session entry/exit diagnostic, ordinary einsum default Auto planning/cache lookup included equally; parsed integer notation; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **einsum / eager-shared** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), session entry excluded, ordinary einsum default Auto planning/cache lookup included equally; parsed integer notation; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **einsum / prepared-into-repeat** — inside: prepared_execute_into; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, plan_preparation, output_preallocation, output_destruction, operation_config_construction.
- **einsum / prepared-repeat** — inside: prepared_execute, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, plan_preparation, output_destruction, operation_config_construction.
- **einsum3 / concrete-fresh** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), per-call session entry/exit diagnostic, ordinary einsum default Auto planning/cache lookup included equally; parsed integer notation; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **einsum3 / concrete-shared** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), session entry excluded, ordinary einsum default Auto planning/cache lookup included equally; parsed integer notation; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **einsum3 / eager-no-ad** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), per-call session entry/exit diagnostic, ordinary einsum default Auto planning/cache lookup included equally; parsed integer notation; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **einsum3 / eager-shared** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), session entry excluded, ordinary einsum default Auto planning/cache lookup included equally; parsed integer notation; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **gather / concrete-shared** — inside: gather, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **qr / concrete-fresh** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), per-call session entry/exit diagnostic, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **qr / concrete-scoped** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), per-call session entry/exit diagnostic, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction, caller-held engine construction; no execution scope exists after tenferro-rs #2206.
- **qr / concrete-shared** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), session entry excluded, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **qr / eager-no-ad** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), per-call session entry/exit diagnostic, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **qr / eager-shared** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), session entry excluded, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **reduce_sum / concrete-shared** — inside: reduce_sum, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **solve / concrete-fresh** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), per-call session entry/exit diagnostic, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **solve / concrete-scoped** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), per-call session entry/exit diagnostic, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction, caller-held engine construction; no execution scope exists after tenferro-rs #2206.
- **solve / concrete-shared** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), session entry excluded, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **solve / eager-no-ad** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), per-call session entry/exit diagnostic, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **solve / eager-shared** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), session entry excluded, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **svd / concrete-fresh** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), per-call session entry/exit diagnostic, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **svd / concrete-shared** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), session entry excluded, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **svd / eager-no-ad** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), per-call session entry/exit diagnostic, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **svd / eager-shared** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), session entry excluded, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.

Wall observations `wall_allocations_t1.json`:
```json
{
  "collection_total_seconds": 0.2648727179985144,
  "case_process_seconds": 0.24471323900070274
}
```

Wall observations `wall_allocations_t4.json`:
```json
{
  "collection_total_seconds": 0.28862033499899553,
  "case_process_seconds": 0.2686069859919371
}
```

Wall observations `wall_allocations_t8.json`:
```json
{
  "collection_total_seconds": 0.4542855830004555,
  "case_process_seconds": 0.42260702500789193
}
```

Wall observations `wall_command.json`:
```json
{
  "command_seconds_before_final_report_write": 144.446973372,
  "report_generation_seconds": 0.115398898,
  "scope": "command startup through report generation; excludes final report file writes",
  "budget_seconds": 120.0,
  "budget_status": "over_budget; diagnostic only"
}
```

Wall observations `wall_measure_t1.json`:
```json
{
  "collection_total_seconds": 3.85861651700543,
  "case_process_seconds": 3.828471669970895
}
```

Wall observations `wall_measure_t4.json`:
```json
{
  "collection_total_seconds": 4.610908747003123,
  "case_process_seconds": 4.580426032953255
}
```

Wall observations `wall_measure_t8.json`:
```json
{
  "collection_total_seconds": 4.777414668002166,
  "case_process_seconds": 4.732087237003725
}
```

Wall observations `wall_phases.json`:
```json
{
  "validation_and_latency_build_seconds": 0.349622481,
  "metadata_and_latency_collection_seconds": 23.366921638,
  "allocation_diagnostic_build_seconds": 119.129524419,
  "allocation_collection_seconds": 1.485505936,
  "command_before_report_seconds": 144.331574474
}
```

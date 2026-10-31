# CPU Small-Work Benchmark Results

- Suite: `cpu/small_work`
- Raw run: `data/results/zen5-cpu/cpu/small_work/20261031_030610`

## CPU Information

- Model: `AMD Ryzen AI 9 HX 470 w/ Radeon 890M`
- Vendor: `AuthenticAMD`
- Logical CPUs: `24`
- Sockets: `1`
- Cores per socket: `12`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3,12-15; 4-11,16-23`
- CPU boost: `disabled`

## run_t1

Full metadata: `data/results/zen5-cpu/cpu/small_work/20261031_030610/run_t1.yaml`

```yaml
timestamp: '2026-10-31T03:06:11Z'
tenferro_rs:
  path: /home/shinaoka/projects/tensor4all/tenferro-benchmark/extern/tenferro-rs
  commit: a9f494cc4679387c75af698b299c316fd2295fe4
  dirty: false
  features:
  - blas-openblas
  resolved_path: /home/shinaoka/projects/tensor4all/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: openblas
  version: unknown
  root: /usr
  library: /usr
environment:
  hostname: shinaoka-EVO-X1Pro
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
    cpu_boost: disabled
    scaling_governor: performance
    scaling_driver: amd-pstate-epp
  env:
    OMP_NUM_THREADS: '1'
    OMP_THREAD_LIMIT: '1'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: zen5-cpu
    BENCHMARK_COMMIT: dae444a6a8f82e39fb37b05009ddc67d7d143d60
    BENCH_COMMAND_BUDGET_SECONDS: '120'
    OPENBLAS_ROOT: /usr
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
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.0 (b940084d7 2026-09-28)
    BENCH_INSTANCE: null
```

## run_t4

Full metadata: `data/results/zen5-cpu/cpu/small_work/20261031_030610/run_t4.yaml`

```yaml
timestamp: '2026-10-31T03:06:20Z'
tenferro_rs:
  path: /home/shinaoka/projects/tensor4all/tenferro-benchmark/extern/tenferro-rs
  commit: a9f494cc4679387c75af698b299c316fd2295fe4
  dirty: false
  features:
  - blas-openblas
  resolved_path: /home/shinaoka/projects/tensor4all/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: openblas
  version: unknown
  root: /usr
  library: /usr
environment:
  hostname: shinaoka-EVO-X1Pro
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
    cpu_boost: disabled
    scaling_governor: performance
    scaling_driver: amd-pstate-epp
  env:
    OMP_NUM_THREADS: '4'
    OMP_THREAD_LIMIT: '4'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '4'
    BENCHMARK_TARGET_PROFILE: zen5-cpu
    BENCHMARK_COMMIT: dae444a6a8f82e39fb37b05009ddc67d7d143d60
    BENCH_COMMAND_BUDGET_SECONDS: '120'
    OPENBLAS_ROOT: /usr
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
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.0 (b940084d7 2026-09-28)
    BENCH_INSTANCE: null
```

## run_t8

Full metadata: `data/results/zen5-cpu/cpu/small_work/20261031_030610/run_t8.yaml`

```yaml
timestamp: '2026-10-31T03:06:30Z'
tenferro_rs:
  path: /home/shinaoka/projects/tensor4all/tenferro-benchmark/extern/tenferro-rs
  commit: a9f494cc4679387c75af698b299c316fd2295fe4
  dirty: false
  features:
  - blas-openblas
  resolved_path: /home/shinaoka/projects/tensor4all/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: openblas
  version: unknown
  root: /usr
  library: /usr
environment:
  hostname: shinaoka-EVO-X1Pro
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
    cpu_boost: disabled
    scaling_governor: performance
    scaling_driver: amd-pstate-epp
  env:
    OMP_NUM_THREADS: '8'
    OMP_THREAD_LIMIT: '8'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '8'
    BENCHMARK_TARGET_PROFILE: zen5-cpu
    BENCHMARK_COMMIT: dae444a6a8f82e39fb37b05009ddc67d7d143d60
    BENCH_COMMAND_BUDGET_SECONDS: '120'
    OPENBLAS_ROOT: /usr
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
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.0 (b940084d7 2026-09-28)
    BENCH_INSTANCE: null
```

## Case status

- Manifest: `cpu/small_work` version 3, coverage `quick`, effort `standard`
- Expected 216, selected 216, executed 216, unsupported 0, failed 0, missing 0, noisy 50.
- Complete: **yes** — only executed cases count as covered; unsupported, failed, missing and unselected cases never do.
- Noisy: `einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single@t1`, `add_f64_concrete-shared_size4_single@t4`, `add_f64_concrete-shared_size4_dependent10@t4`, `add_f64_concrete-shared_size16_dependent10@t4`, `add_f64_concrete-shared_size256_dependent10@t4`, `einsum_f64_prepared-repeat_n2_single@t4`, `solve_f64_concrete-shared_n4_single@t4`, `einsum_f64_prepared-repeat_n8_single@t4`, `einsum_c64_prepared-repeat_n8_single@t4`, `gather_f64_concrete-shared_size256_single@t4`, `reduce_sum_f64_concrete-shared_size4_single@t4`, `reduce_sum_f64_concrete-shared_size16_single@t4`, `einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single@t4`, `einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single@t4`, `einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single@t4`, `nonad_einsum_n2_col_major_contiguous_concrete-fresh@t4`, `nonad_einsum_n2_col_major_contiguous_eager-no-ad@t4`, `nonad_einsum_n2_col_major_contiguous_concrete-shared@t4`, `nonad_einsum3_n2_col_major_contiguous_eager-shared@t4`, `nonad_svd_n2_col_major_contiguous_eager-shared@t4`, `nonad_qr_n16_col_major_contiguous_eager-no-ad@t4`, `nonad_qr_n16_col_major_contiguous_concrete-fresh@t4`, `nonad_solve_n16_col_major_contiguous_concrete-shared@t4`, `nonad_solve_n16_col_major_contiguous_concrete-fresh@t4`, `nonad_einsum_n2_col_major_contiguous_concrete-scoped@t4`, `nonad_qr_n2_col_major_contiguous_concrete-scoped@t4`, `add_f64_concrete-shared_size4_dependent10@t8`, `einsum_f64_prepared-repeat_n4_single@t8`, `einsum_f64_prepared-repeat_n16_single@t8`, `einsum_c64_prepared-repeat_n2_single@t8`, `solve_f64_concrete-shared_n2_single@t8`, `solve_f64_concrete-shared_n16_single@t8`, `einsum_f64_prepared-repeat_n8_single@t8`, `einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single@t8`, `einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single@t8`, `einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single@t8`, `einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single@t8`, `einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single@t8`, `einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single@t8`, `nonad_einsum_n2_col_major_contiguous_concrete-shared@t8` (+10 more)

Numerical checks: **216/216 recorded rows passed** (case × thread count).

## Timing

Sequential release runs; explicit initialization and warmups, then calibration toward the declared batch duration (2 ms by default), then the measured batches of the recorded effort (standard: 3 warmups and 15 batches; scan: a low-repetition screen that only marks suspects). Legacy single-call routes start calibration at 1024 workflows; matched non-AD and dependent-chain routes start at 16 and bound retained memory. Each batch uses one wall-clock interval; all outputs are retained until it stops. Statistics divide batch time by iterations. Median and inclusive IQR are in nanoseconds (ns); CoV is sample standard deviation / mean. NOISY means CoV > 10%, a descriptive label only. No noisy rows are excluded.

Workflow time is the total for one call or the complete dependent chain. The separate ns/op column divides the chain median by its operation count, not an independently measured call. Setup rows measure preparation only, not execution.

Inputs and backend construction, independent numerical checks and AD backward checks are outside timing. Returned outputs/plans and every dependent-chain intermediate are retained until after each batch interval; retention containers are prepared before the clock. value downloads/checks are outside. Eager AD rows measure forward execution/recording, not backward.

| Case ID | Operation | API route | Route (path / output / repr / session) | Dtype | Shape | Layout | Workflow | Threads | Provider | Operations/batch | Median batch ms | Median workflow ns | IQR ns | CoV | ns/op (chain only) | Status |
|---|---|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 2048 | 2.507738 | 1224.48 | 7.94 | 2.4% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 1 | blas | 2560 | 2.888136 | 11281.78 | 69.35 | 0.7% | 1128.18 | ok |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 2048 | 2.425295 | 1184.23 | 3.41 | 2.2% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 1 | blas | 2560 | 2.954751 | 11542.00 | 38.43 | 0.5% | 1154.20 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 1024 | 1.409994 | 1376.95 | 93.74 | 4.6% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 1 | blas | 1280 | 1.712508 | 13378.97 | 136.86 | 1.4% | 1337.90 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 1.765657 | 1724.27 | 11.34 | 1.5% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 2048 | 3.518450 | 1717.99 | 7.95 | 1.7% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.075954 | 2027.30 | 14.04 | 1.5% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 1.876553 | 1832.57 | 15.11 | 1.6% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 2.013539 | 1966.35 | 5.32 | 1.3% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 4.068734 | 3973.37 | 259.58 | 3.8% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.043965 | 1996.06 | 9.99 | 1.2% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 2.335868 | 2281.12 | 10.94 | 1.3% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 8.541019 | 8340.84 | 363.47 | 2.3% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 1.829216 | 1786.34 | 30.42 | 2.1% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 4.081208 | 3985.55 | 13.78 | 0.7% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 2.282630 | 2229.13 | 50.04 | 6.2% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 15.507778 | 15144.31 | 254.97 | 1.0% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 1024 | 1.716495 | 1676.26 | 12.69 | 1.6% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 1024 | 1.770215 | 1728.73 | 4.98 | 1.8% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 1024 | 2.086364 | 2037.46 | 13.50 | 1.6% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 2048 | 2.835909 | 1384.72 | 3.06 | 2.0% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 2048 | 2.817996 | 1375.97 | 4.27 | 1.8% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 2048 | 2.897334 | 1414.71 | 6.48 | 1.8% | — | ok |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 3.403476 | 3323.71 | 27.92 | 0.9% | — | ok |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 3.333465 | 3255.34 | 15.17 | 1.0% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 3.285055 | 3208.06 | 24.93 | 1.0% | — | ok |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 1024 | 1.976710 | 1930.38 | 7.23 | 1.7% | — | ok |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 1024 | 2.056779 | 2008.57 | 13.03 | 1.9% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 1024 | 2.005694 | 1958.69 | 6.52 | 1.6% | — | ok |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 1024 | 3.227117 | 3151.48 | 28.66 | 1.2% | — | ok |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 1024 | 3.365555 | 3286.67 | 9.35 | 0.9% | — | ok |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 1024 | 3.207802 | 3132.62 | 17.19 | 10.1% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.237876 | 2185.43 | 11.15 | 3.1% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.010533 | 3926.82 | 33.86 | 2.4% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.259617 | 2206.66 | 10.65 | 3.2% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 1.979134 | 3865.50 | 21.74 | 3.6% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 64 | 1.989875 | 31091.80 | 240.21 | 0.8% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 2.944762 | 11502.98 | 96.06 | 1.1% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 64 | 1.978363 | 30911.92 | 136.43 | 0.7% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 2.937008 | 11472.69 | 69.07 | 1.1% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 1.959908 | 1913.97 | 19.58 | 4.1% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 1.915195 | 7481.23 | 125.02 | 1.5% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 1.979355 | 1932.96 | 11.94 | 4.1% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 1.900167 | 7422.53 | 93.02 | 1.6% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.150373 | 4199.95 | 21.55 | 2.7% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 2.698934 | 10542.71 | 60.41 | 0.7% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.163788 | 4226.15 | 42.47 | 2.2% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 2.670281 | 10430.79 | 62.99 | 0.7% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.002859 | 1955.92 | 17.51 | 3.7% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 1.849754 | 7225.60 | 122.00 | 2.0% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.018318 | 1971.01 | 16.60 | 3.1% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 3.623917 | 7077.96 | 45.69 | 1.2% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.226204 | 4348.05 | 15.55 | 2.1% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.660182 | 2597.83 | 18.70 | 2.9% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.290464 | 4473.56 | 43.71 | 1.7% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.659330 | 2597.00 | 13.08 | 3.1% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 128 | 2.047191 | 15993.68 | 181.90 | 0.9% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 2.548304 | 9954.31 | 55.12 | 1.1% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 128 | 2.026322 | 15830.64 | 142.92 | 1.3% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 2.589420 | 10114.92 | 78.70 | 1.0% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 3.443039 | 13449.37 | 68.52 | 0.9% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 2.078720 | 8120.00 | 391.37 | 3.0% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 3.466173 | 13539.74 | 144.88 | 1.0% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 2.127711 | 8311.37 | 390.47 | 3.0% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.252533 | 2199.74 | 18.41 | 3.4% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 1.964998 | 1918.94 | 13.57 | 4.5% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 1.997849 | 1951.02 | 7.90 | 4.0% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.619657 | 2558.26 | 20.73 | 3.0% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 2.548113 | 9953.57 | 55.24 | 0.5% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 2.097765 | 8194.39 | 401.37 | 2.5% | — | ok |
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 1024 | 1.885319 | 1841.13 | 8.58 | 45.9% | — | NOISY |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 4 | blas | 1280 | 2.238988 | 17492.09 | 73.07 | 39.3% | 1749.21 | NOISY |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 1024 | 1.831259 | 1788.34 | 11.03 | 1.4% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 4 | blas | 1280 | 2.250279 | 17580.30 | 90.13 | 37.3% | 1758.03 | NOISY |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 1.937567 | 1892.16 | 18.63 | 2.0% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 4 | blas | 1280 | 2.368919 | 18507.18 | 76.70 | 21.7% | 1850.72 | NOISY |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.312405 | 2258.21 | 11.91 | 30.7% | — | NOISY |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 2.332021 | 2277.36 | 15.66 | 1.3% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.589330 | 2528.64 | 10.88 | 1.2% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.501436 | 2442.81 | 21.03 | 1.0% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 2.558323 | 2498.36 | 19.05 | 1.1% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 4.311146 | 4210.10 | 7.33 | 9.6% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 3.243939 | 3167.91 | 5.31 | 0.8% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 3.697383 | 3610.73 | 905.02 | 19.6% | — | NOISY |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 7.385619 | 7212.52 | 362.01 | 2.5% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 2.367898 | 2312.40 | 6.62 | 21.5% | — | NOISY |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 4.507031 | 4401.40 | 8.98 | 1.0% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 2.788371 | 2723.02 | 3.86 | 27.7% | — | NOISY |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 15.611872 | 15245.97 | 251.25 | 0.9% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 1024 | 2.415908 | 2359.29 | 26.20 | 1.2% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 1024 | 2.390039 | 2334.02 | 21.03 | 1.4% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 2.713061 | 2649.47 | 2.02 | 28.3% | — | NOISY |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 1024 | 2.043384 | 1995.49 | 4.52 | 43.3% | — | NOISY |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 1024 | 2.018428 | 1971.12 | 7.52 | 39.5% | — | NOISY |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 2.131418 | 2081.46 | 18.13 | 1.2% | — | ok |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 3.869454 | 3778.76 | 12.95 | 8.0% | — | ok |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 3.963779 | 3870.88 | 13.19 | 9.8% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 3.854025 | 3763.70 | 5.85 | 0.8% | — | ok |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 1024 | 2.600952 | 2539.99 | 6.55 | 20.3% | — | NOISY |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 1024 | 2.629866 | 2568.23 | 9.77 | 1.0% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 1024 | 2.551780 | 2491.97 | 50.55 | 35.1% | — | NOISY |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 1024 | 3.859686 | 3769.22 | 11.50 | 0.6% | — | ok |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 1024 | 3.917112 | 3825.30 | 11.95 | 20.6% | — | NOISY |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 1024 | 3.783794 | 3695.11 | 8.97 | 0.8% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 1.384047 | 2703.22 | 12.31 | 38.8% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 1.142967 | 4464.71 | 39.60 | 11.9% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 1.378406 | 2692.20 | 8.34 | 26.5% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.243817 | 4382.46 | 17.81 | 1.8% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 64 | 2.002849 | 31294.52 | 122.10 | 0.3% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 128 | 1.684847 | 13162.87 | 84.22 | 0.7% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 64 | 1.980316 | 30942.44 | 182.05 | 0.5% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 128 | 1.703781 | 13310.79 | 119.36 | 30.0% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.474967 | 2416.96 | 17.19 | 2.8% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 1.970018 | 7695.38 | 36.24 | 0.8% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 1.255357 | 2451.87 | 15.01 | 2.8% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 1.944920 | 7597.34 | 50.31 | 1.0% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.395840 | 4679.38 | 12.77 | 1.5% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 2.660824 | 10393.84 | 49.04 | 0.3% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.409115 | 4705.30 | 7.76 | 1.4% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 2.641989 | 10320.27 | 69.96 | 34.3% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 3.193976 | 3119.12 | 11.36 | 1.9% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 2.140806 | 8362.52 | 1605.66 | 9.2% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 3.188245 | 3113.52 | 13.63 | 1.8% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 2.523538 | 9857.57 | 1665.83 | 10.0% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 1.207608 | 4717.22 | 27.06 | 1.3% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 1.526511 | 2981.47 | 17.15 | 2.3% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.444170 | 4773.77 | 19.72 | 1.1% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 3.089481 | 3017.07 | 6.93 | 2.3% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 128 | 2.015702 | 15747.67 | 68.65 | 0.4% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 2.561168 | 10004.56 | 16.46 | 0.2% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 128 | 1.986398 | 15518.73 | 66.38 | 35.6% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 2.626059 | 10258.04 | 47.33 | 13.8% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 3.164130 | 12359.88 | 348.97 | 3.2% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 1.904094 | 7437.87 | 1551.45 | 16.0% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 128 | 1.579220 | 12337.66 | 96.32 | 2.0% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 128 | 1.274633 | 9958.07 | 410.77 | 10.5% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 1.393504 | 2721.69 | 10.77 | 49.1% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 1.253283 | 2447.82 | 17.15 | 37.7% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 3.124497 | 3051.27 | 13.55 | 2.0% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 1.532383 | 2992.94 | 8.23 | 2.3% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 2.536252 | 9907.23 | 25.46 | 0.4% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 128 | 0.906698 | 7083.58 | 195.01 | 3.4% | — | INCONCLUSIVE (<1 ms batch) |
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 8 | blas | 1024 | 1.905858 | 1861.19 | 17.46 | 1.4% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 8 | blas | 1280 | 2.329156 | 18196.53 | 98.51 | 16.1% | 1819.65 | NOISY |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 8 | blas | 1024 | 1.902872 | 1858.27 | 16.80 | 1.3% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 8 | blas | 1280 | 2.292438 | 17909.67 | 121.56 | 0.5% | 1790.97 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 8 | blas | 1024 | 1.952244 | 1906.49 | 12.30 | 1.6% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 8 | blas | 1280 | 2.379479 | 18589.68 | 42.34 | 0.4% | 1858.97 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 1024 | 2.410617 | 2354.12 | 8.48 | 0.9% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 8 | blas | 1024 | 2.360243 | 2304.92 | 17.38 | 30.4% | — | NOISY |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 1024 | 2.622291 | 2560.83 | 13.13 | 29.5% | — | NOISY |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 8 | blas | 1024 | 2.545008 | 2485.36 | 15.71 | 15.0% | — | NOISY |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 8 | blas | 1024 | 2.625377 | 2563.84 | 14.93 | 1.0% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 8 | blas | 1024 | 4.376377 | 4273.81 | 13.88 | 2.0% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 1024 | 5.011569 | 4894.11 | 1614.11 | 19.1% | — | NOISY |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 8 | blas | 1024 | 3.722740 | 3635.49 | 11.66 | 0.8% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 1024 | 7.847128 | 7663.21 | 506.73 | 12.7% | — | NOISY |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 8 | blas | 1024 | 2.391332 | 2335.29 | 9.01 | 15.3% | — | NOISY |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 8 | blas | 1024 | 4.585717 | 4478.24 | 26.62 | 0.9% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 8 | blas | 1024 | 2.785275 | 2720.00 | 24.40 | 1.5% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 8 | blas | 1024 | 15.748547 | 15379.44 | 282.95 | 1.1% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 8 | blas | 1024 | 2.480859 | 2422.71 | 16.72 | 1.4% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 8 | blas | 1024 | 2.509011 | 2450.21 | 18.12 | 1.2% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 8 | blas | 1024 | 2.827704 | 2761.43 | 33.07 | 1.2% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 8 | blas | 1024 | 2.141517 | 2091.33 | 27.71 | 1.5% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 8 | blas | 1024 | 2.162846 | 2112.15 | 8.61 | 1.1% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 8 | blas | 1024 | 2.179638 | 2128.55 | 17.69 | 1.4% | — | ok |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 8 | blas | 1024 | 3.881225 | 3790.26 | 14.97 | 0.8% | — | ok |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 8 | blas | 1024 | 3.992142 | 3898.58 | 17.33 | 27.6% | — | NOISY |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 8 | blas | 1024 | 3.899819 | 3808.42 | 19.67 | 0.9% | — | ok |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 8 | blas | 1024 | 2.686922 | 2623.95 | 8.74 | 50.0% | — | NOISY |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 8 | blas | 1024 | 2.710276 | 2646.75 | 10.14 | 0.6% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 8 | blas | 1024 | 2.570766 | 2510.51 | 14.82 | 29.0% | — | NOISY |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 8 | blas | 1024 | 3.897395 | 3806.05 | 15.77 | 66.2% | — | NOISY |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 8 | blas | 1024 | 3.977915 | 3884.68 | 16.72 | 27.9% | — | NOISY |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 8 | blas | 1024 | 3.869013 | 3778.33 | 12.47 | 20.7% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 128 | 0.355572 | 2777.91 | 30.57 | 0.7% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 512 | 2.280405 | 4453.92 | 31.24 | 1.7% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 512 | 1.417569 | 2768.69 | 14.13 | 48.7% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 256 | 1.148438 | 4486.09 | 29.61 | 1.5% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 64 | 2.048203 | 32003.17 | 222.98 | 0.6% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 256 | 3.439744 | 13436.50 | 43.68 | 12.3% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 64 | 2.005945 | 31342.89 | 106.69 | 0.2% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 128 | 1.692451 | 13222.27 | 85.16 | 42.6% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 1024 | 2.544667 | 2485.03 | 19.11 | 2.8% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 256 | 1.999101 | 7808.99 | 35.20 | 1.0% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 512 | 1.253043 | 2447.35 | 9.84 | 2.5% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 256 | 1.968294 | 7688.65 | 45.46 | 44.1% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 512 | 2.439811 | 4765.26 | 23.06 | 1.7% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 256 | 2.714714 | 10604.35 | 72.61 | 0.5% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 256 | 1.211084 | 4730.80 | 31.90 | 16.4% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 128 | 1.338482 | 10456.89 | 59.14 | 0.5% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 512 | 2.479907 | 4843.57 | 46.78 | 1.5% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 256 | 2.125617 | 8303.19 | 92.11 | 1.0% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 256 | 1.244366 | 4860.80 | 13.95 | 0.6% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 256 | 2.578030 | 10070.43 | 66.92 | 63.6% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 512 | 2.446975 | 4779.25 | 28.54 | 1.4% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 1024 | 3.100722 | 3028.05 | 12.60 | 2.4% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 512 | 2.486799 | 4857.03 | 41.77 | 1.4% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 256 | 0.787616 | 3076.62 | 16.34 | 33.5% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 128 | 2.003179 | 15649.84 | 122.93 | 0.5% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 256 | 2.616581 | 10221.02 | 33.11 | 0.2% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 128 | 2.023557 | 15809.04 | 118.50 | 0.5% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 256 | 2.618084 | 10226.89 | 42.89 | 0.3% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 64 | 0.949617 | 14837.77 | 112.79 | 0.9% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 256 | 2.499193 | 9762.47 | 315.41 | 2.1% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 128 | 1.589279 | 12416.24 | 2560.55 | 9.9% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 256 | 2.487310 | 9716.05 | 145.15 | 10.2% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 1024 | 2.871355 | 2804.06 | 28.83 | 24.0% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 512 | 1.271026 | 2482.47 | 11.59 | 29.3% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 8 | blas | 1024 | 3.270839 | 3194.18 | 13.43 | 2.3% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 1024 | 3.089972 | 3017.55 | 13.67 | 2.4% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 256 | 2.592496 | 10126.94 | 27.89 | 0.4% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 8 | blas | 256 | 2.031862 | 7936.96 | 2012.91 | 12.9% | — | NOISY |

## Matched non-AD eager comparisons (#98)

All requested QR/SVD outputs are retained. Ordinary einsum uses the same parsed integer notation and default Auto strategy; planning/cache lookup stays inside both public calls. These are public-call diagnostics, not prepared execution or an isolated AD cost. Fresh pairs include session entry; shared pairs exclude it. Views are constructed untimed and consumed as views on both routes. No output to_tensor conversion is timed. Capture is inactive by construction in each fresh process, with no capture_trace guard; inputs and validation outputs are checked untracked.

| Operation | Shape | Layout | Session | Threads | Direct ns | Eager ns | Eager − direct ns | Eager/direct | Quality |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 1 | 2185.43 | 3926.82 | 1741.39 | 1.797 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | shared | 1 | 2206.66 | 3865.50 | 1658.84 | 1.752 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 4 | 2703.22 | 4464.71 | 1761.49 | 1.652 | inconclusive |
| einsum | 2×2 | col_major_contiguous | shared | 4 | 2692.20 | 4382.46 | 1690.26 | 1.628 | inconclusive |
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 8 | 2777.91 | 4453.92 | 1676.01 | 1.603 | inconclusive |
| einsum | 2×2 | col_major_contiguous | shared | 8 | 2768.69 | 4486.09 | 1717.40 | 1.620 | inconclusive |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 1 | 2597.00 | 4473.56 | 1876.56 | 1.723 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | shared | 1 | 2597.83 | 4348.05 | 1750.22 | 1.674 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 4 | 3017.07 | 4773.77 | 1756.70 | 1.582 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | shared | 4 | 2981.47 | 4717.22 | 1735.75 | 1.582 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 8 | 3076.62 | 4857.03 | 1780.41 | 1.579 | inconclusive |
| einsum | 16×16 | col_major_contiguous | shared | 8 | 3028.05 | 4779.25 | 1751.20 | 1.578 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 1 | 31091.80 | 11502.98 | -19588.82 | 0.370 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | shared | 1 | 30911.92 | 11472.69 | -19439.23 | 0.371 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 4 | 31294.52 | 13162.87 | -18131.65 | 0.421 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | shared | 4 | 30942.44 | 13310.79 | -17631.65 | 0.430 | inconclusive |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 8 | 32003.17 | 13436.50 | -18566.67 | 0.420 | inconclusive |
| einsum3 | 2×2 | col_major_contiguous | shared | 8 | 31342.89 | 13222.27 | -18120.62 | 0.422 | inconclusive |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1913.97 | 7481.23 | 5567.26 | 3.909 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | shared | 1 | 1932.96 | 7422.53 | 5489.57 | 3.840 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 4 | 2416.96 | 7695.38 | 5278.42 | 3.184 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | shared | 4 | 2451.87 | 7597.34 | 5145.47 | 3.099 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 8 | 2485.03 | 7808.99 | 5323.96 | 3.142 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | shared | 8 | 2447.35 | 7688.65 | 5241.30 | 3.142 | inconclusive |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 1 | 10114.92 | 15830.64 | 5715.72 | 1.565 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | shared | 1 | 9954.31 | 15993.68 | 6039.37 | 1.607 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 4 | 10258.04 | 15518.73 | 5260.69 | 1.513 | inconclusive |
| qr | 16×16 | col_major_contiguous | shared | 4 | 10004.56 | 15747.67 | 5743.11 | 1.574 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 8 | 10226.89 | 15809.04 | 5582.15 | 1.546 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | shared | 8 | 10221.02 | 15649.84 | 5428.82 | 1.531 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1955.92 | 7225.60 | 5269.68 | 3.694 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | shared | 1 | 1971.01 | 7077.96 | 5106.95 | 3.591 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 4 | 3119.12 | 8362.52 | 5243.40 | 2.681 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | shared | 4 | 3113.52 | 9857.57 | 6744.05 | 3.166 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 8 | 4843.57 | 8303.19 | 3459.62 | 1.714 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | shared | 8 | 4860.80 | 10070.43 | 5209.63 | 2.072 | inconclusive |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 1 | 8311.37 | 13539.74 | 5228.37 | 1.629 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | shared | 1 | 8120.00 | 13449.37 | 5329.37 | 1.656 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 4 | 9958.07 | 12337.66 | 2379.59 | 1.239 | inconclusive |
| solve | 16×16 | col_major_contiguous | shared | 4 | 7437.87 | 12359.88 | 4922.01 | 1.662 | inconclusive |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 8 | 9716.05 | 12416.24 | 2700.19 | 1.278 | inconclusive |
| solve | 16×16 | col_major_contiguous | shared | 8 | 9762.47 | 14837.77 | 5075.30 | 1.520 | inconclusive |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 1 | 4199.95 | 10542.71 | 6342.76 | 2.510 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | shared | 1 | 4226.15 | 10430.79 | 6204.64 | 2.468 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 4 | 4679.38 | 10393.84 | 5714.46 | 2.221 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | shared | 4 | 4705.30 | 10320.27 | 5614.97 | 2.193 | inconclusive |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 8 | 4765.26 | 10604.35 | 5839.09 | 2.225 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | shared | 8 | 4730.80 | 10456.89 | 5726.09 | 2.210 | inconclusive |

## Separate allocation diagnostics

Instrumented executable is built separately after latency collection. Calling-thread Rust allocator requests only; excludes foreign BLAS and worker-thread allocations. Realloc counts its requested new size, not net retained bytes. Output destruction and instrumentation setup are excluded. Helper references and multi-output retention use stack storage. No allocation-instrumented run contributes latency samples.

| Corresponding case ID | Threads | Median allocation requests/op | Median requested bytes/op | Status |
|---|---:|---:|---:|---|
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | 1 | 8 | 120 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | 1 | 15 | 4120 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | 1 | 8 | 120 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-shared | 1 | 15 | 4120 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | 1 | 265 | 51829 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | 1 | 48 | 20952 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | 1 | 265 | 51829 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | 1 | 48 | 20952 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | 1 | 6 | 1760 | passed |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | 1 | 39 | 15995 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-shared | 1 | 6 | 1760 | passed |
| nonad_qr_n2_col_major_contiguous_eager-shared | 1 | 39 | 15995 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | 1 | 8 | 2952 | passed |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | 1 | 46 | 18515 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-shared | 1 | 8 | 2952 | passed |
| nonad_svd_n2_col_major_contiguous_eager-shared | 1 | 46 | 18515 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | 1 | 3 | 80 | passed |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | 1 | 35 | 16195 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-shared | 1 | 3 | 80 | passed |
| nonad_solve_n2_col_major_contiguous_eager-shared | 1 | 35 | 16195 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-shared | 1 | 15 | 4120 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | 1 | 8 | 120 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | 1 | 15 | 4120 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | 1 | 8 | 120 | passed |
| nonad_qr_n16_col_major_contiguous_eager-shared | 1 | 39 | 21595 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-shared | 1 | 6 | 7360 | passed |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | 1 | 39 | 21595 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | 1 | 6 | 7360 | passed |
| nonad_solve_n16_col_major_contiguous_eager-shared | 1 | 35 | 18211 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-shared | 1 | 3 | 4112 | passed |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | 1 | 35 | 18211 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | 1 | 3 | 4112 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | 1 | 8 | 120 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | 1 | 6 | 1760 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | 1 | 3 | 80 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | 1 | 8 | 120 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | 1 | 6 | 7360 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | 1 | 3 | 4112 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | 4 | 8 | 120 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | 4 | 15 | 4120 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | 4 | 8 | 120 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-shared | 4 | 15 | 4120 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | 4 | 265 | 51829 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | 4 | 48 | 20952 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | 4 | 265 | 51829 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | 4 | 48 | 20952 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | 4 | 6 | 1760 | passed |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | 4 | 39 | 15995 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-shared | 4 | 6 | 1760 | passed |
| nonad_qr_n2_col_major_contiguous_eager-shared | 4 | 39 | 15995 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | 4 | 8 | 2952 | passed |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | 4 | 46 | 18515 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-shared | 4 | 8 | 2952 | passed |
| nonad_svd_n2_col_major_contiguous_eager-shared | 4 | 46 | 18515 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | 4 | 3 | 80 | passed |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | 4 | 35 | 16195 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-shared | 4 | 3 | 80 | passed |
| nonad_solve_n2_col_major_contiguous_eager-shared | 4 | 35 | 16195 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-shared | 4 | 15 | 4120 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | 4 | 8 | 120 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | 4 | 15 | 4120 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | 4 | 8 | 120 | passed |
| nonad_qr_n16_col_major_contiguous_eager-shared | 4 | 39 | 21595 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-shared | 4 | 6 | 7360 | passed |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | 4 | 39 | 21595 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | 4 | 6 | 7360 | passed |
| nonad_solve_n16_col_major_contiguous_eager-shared | 4 | 35 | 18211 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-shared | 4 | 3 | 4112 | passed |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | 4 | 35 | 18211 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | 4 | 3 | 4112 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | 4 | 8 | 120 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | 4 | 6 | 1760 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | 4 | 3 | 80 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | 4 | 8 | 120 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | 4 | 6 | 7360 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | 4 | 3 | 4112 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | 8 | 8 | 120 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | 8 | 15 | 4120 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | 8 | 8 | 120 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-shared | 8 | 15 | 4120 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | 8 | 265 | 51829 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | 8 | 48 | 20952 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | 8 | 265 | 51829 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | 8 | 48 | 20952 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | 8 | 6 | 1760 | passed |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | 8 | 39 | 15995 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-shared | 8 | 6 | 1760 | passed |
| nonad_qr_n2_col_major_contiguous_eager-shared | 8 | 39 | 15995 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | 8 | 8 | 2952 | passed |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | 8 | 46 | 18515 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-shared | 8 | 8 | 2952 | passed |
| nonad_svd_n2_col_major_contiguous_eager-shared | 8 | 46 | 18515 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | 8 | 3 | 80 | passed |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | 8 | 35 | 16195 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-shared | 8 | 3 | 80 | passed |
| nonad_solve_n2_col_major_contiguous_eager-shared | 8 | 35 | 16195 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-shared | 8 | 15 | 4120 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | 8 | 8 | 120 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | 8 | 15 | 4120 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | 8 | 8 | 120 | passed |
| nonad_qr_n16_col_major_contiguous_eager-shared | 8 | 39 | 21595 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-shared | 8 | 6 | 7360 | passed |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | 8 | 39 | 21595 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | 8 | 6 | 7360 | passed |
| nonad_solve_n16_col_major_contiguous_eager-shared | 8 | 35 | 18211 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-shared | 8 | 3 | 4112 | passed |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | 8 | 35 | 18211 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | 8 | 3 | 4112 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | 8 | 8 | 120 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | 8 | 6 | 1760 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | 8 | 3 | 80 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | 8 | 8 | 120 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | 8 | 6 | 7360 | passed |
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
  "collection_total_seconds": 0.19729610701324418,
  "case_process_seconds": 0.14848609396722168
}
```

Wall observations `wall_allocations_t4.json`:
```json
{
  "collection_total_seconds": 0.6699708619853482,
  "case_process_seconds": 0.6218139498960227
}
```

Wall observations `wall_allocations_t8.json`:
```json
{
  "collection_total_seconds": 1.2789048239937983,
  "case_process_seconds": 1.2289379217545502
}
```

Wall observations `wall_command.json`:
```json
{
  "command_seconds_through_collection": 229.005444437,
  "scope": "command startup through collection; excludes report rendering and publication",
  "budget_seconds": 120.0,
  "budget_status": "over_budget; diagnostic only"
}
```

Wall observations `wall_measure_t1.json`:
```json
{
  "collection_total_seconds": 3.9671798269846477,
  "case_process_seconds": 3.8908388559357263
}
```

Wall observations `wall_measure_t4.json`:
```json
{
  "collection_total_seconds": 4.826058076985646,
  "case_process_seconds": 4.7615454234182835
}
```

Wall observations `wall_measure_t8.json`:
```json
{
  "collection_total_seconds": 5.85726036602864,
  "case_process_seconds": 5.787733261182439
}
```

Wall observations `wall_phases.json`:
```json
{
  "validation_and_latency_build_seconds": 0.500118368,
  "metadata_and_latency_collection_seconds": 31.265003053,
  "allocation_diagnostic_build_seconds": 193.451259788,
  "allocation_collection_seconds": 3.708777864,
  "command_before_report_seconds": 228.925159073
}
```

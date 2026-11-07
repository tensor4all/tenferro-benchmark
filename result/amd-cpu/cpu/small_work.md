# CPU Small-Work Benchmark Results

- Suite: `cpu/small_work`
- Raw run: `data/results/amd-cpu/cpu/small_work/20261107_194911`

## CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1019-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`
- Container: `devcontainer`

## run_t1

Full metadata: `data/results/amd-cpu/cpu/small_work/20261107_194911/run_t1.yaml`

```yaml
timestamp: '2026-11-07T19:49:11Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
  commit: 9a850b06155405acc99ecd0648829e9e6d3866ba
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
  container: devcontainer
  hostname: 7824261bb622
  os: Linux-7.0.0-1019-azure-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '1'
    measured_cpus: 0-3
    l3_domains:
    - 0-3
    cpu_boost: null
    scaling_governor: null
    scaling_driver: null
  env:
    OMP_NUM_THREADS: '1'
    OMP_THREAD_LIMIT: '1'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: amd-cpu
    BENCHMARK_COMMIT: d6655b567e09aa29c3201655e1de418716934876
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
    PUBLICATION_GATE_PROFILE: quick
    USE_CUDA: '0'
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.1 (58cf8f9c3 2026-10-13)
    BENCH_INSTANCE: null
```

## run_t4

Full metadata: `data/results/amd-cpu/cpu/small_work/20261107_194911/run_t4.yaml`

```yaml
timestamp: '2026-11-07T19:49:19Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
  commit: 9a850b06155405acc99ecd0648829e9e6d3866ba
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
  container: devcontainer
  hostname: 7824261bb622
  os: Linux-7.0.0-1019-azure-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '4'
    measured_cpus: 0-3
    l3_domains:
    - 0-3
    cpu_boost: null
    scaling_governor: null
    scaling_driver: null
  env:
    OMP_NUM_THREADS: '4'
    OMP_THREAD_LIMIT: '4'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '4'
    BENCHMARK_TARGET_PROFILE: amd-cpu
    BENCHMARK_COMMIT: d6655b567e09aa29c3201655e1de418716934876
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
    PUBLICATION_GATE_PROFILE: quick
    USE_CUDA: '0'
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.1 (58cf8f9c3 2026-10-13)
    BENCH_INSTANCE: null
```

## Case status

- Manifest: `cpu/small_work` version 3, coverage `quick`, effort `standard`
- Expected 144, selected 144, executed 144, unsupported 0, failed 0, skipped 0, missing 0, noisy 17.
- Complete: **yes** — only executed cases count as covered; unsupported, skipped, failed, missing and unselected cases never do.

| expected | selected | executed | unsupported | failed | skipped | missing | noisy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 144 | 144 | 144 | 0 | 0 | 0 | 0 | 17 |

- Noisy: `reduce_sum_f64_concrete-shared_size256_single@t1`, `nonad_einsum_n2_col_major_contiguous_concrete-shared@t1`, `add_f64_concrete-shared_size4_single@t4`, `add_f64_concrete-shared_size4_dependent10@t4`, `add_f64_concrete_shared@t4`, `add_f64_concrete-shared_size16_dependent10@t4`, `add_f64_concrete-shared_size256_single@t4`, `einsum_c64_prepared-repeat_n2_single@t4`, `gather_f64_concrete-shared_size4_single@t4`, `gather_f64_concrete_shared@t4`, `gather_f64_concrete-shared_size256_single@t4`, `reduce_sum_f64_concrete-shared_size4_single@t4`, `reduce_sum_f64_concrete-shared_size16_single@t4`, `reduce_sum_f64_concrete-shared_size256_single@t4`, `nonad_einsum3_n2_col_major_contiguous_concrete-fresh@t4`, `nonad_einsum3_n2_col_major_contiguous_concrete-shared@t4`, `nonad_svd_n2_col_major_contiguous_eager-no-ad@t4`

Numerical checks: **144/144 recorded rows passed** (case × thread count).

## Timing

Sequential release runs; explicit initialization and warmups, then calibration toward the declared batch duration (2 ms by default), then the measured batches of the recorded effort (standard: 3 warmups and 15 batches; scan: a low-repetition screen that only marks suspects). Legacy single-call routes start calibration at 1024 workflows; matched non-AD and dependent-chain routes start at 16 and bound retained memory. Each batch uses one wall-clock interval; all outputs are retained until it stops. Statistics divide batch time by iterations. Median and inclusive IQR are in nanoseconds (ns); CoV is sample standard deviation / mean. NOISY means CoV > 10%, a descriptive label only. No noisy rows are excluded.

Workflow time is the total for one call or the complete dependent chain. The separate ns/op column divides the chain median by its operation count, not an independently measured call. Setup rows measure preparation only, not execution.

Inputs and backend construction, independent numerical checks and AD backward checks are outside timing. Returned outputs/plans and every dependent-chain intermediate are retained until after each batch interval; retention containers are prepared before the clock. value downloads/checks are outside. Eager AD rows measure forward execution/recording, not backward.

| Case ID | Operation | API route | Route (path / output / repr / session) | Dtype | Shape | Layout | Workflow | Threads | Provider | Operations/batch | Median batch ms | Median workflow ns | IQR ns | CoV | ns/op (chain only) | Status |
|---|---|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 2048 | 1.794954 | 876.44 | 4.57 | 3.8% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 1 | blas | 2560 | 2.220576 | 8674.12 | 62.14 | 1.5% | 867.41 | ok |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 2048 | 1.843313 | 900.06 | 12.51 | 3.9% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 1 | blas | 2560 | 2.186722 | 8541.88 | 63.52 | 0.7% | 854.19 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 1024 | 0.976158 | 953.28 | 10.91 | 4.4% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 1 | blas | 2560 | 2.383419 | 9310.23 | 52.46 | 0.9% | 931.02 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 3.313662 | 1618.00 | 12.88 | 1.0% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 2048 | 3.292582 | 1607.71 | 6.66 | 0.4% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.360627 | 2305.30 | 12.49 | 0.8% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 3.111906 | 1519.49 | 15.07 | 2.6% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 1.726837 | 1686.36 | 18.61 | 1.1% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.725637 | 2661.75 | 18.82 | 1.1% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.735435 | 1335.66 | 5.24 | 0.9% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 2048 | 2.776632 | 1355.78 | 3.59 | 0.4% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 4.281054 | 4180.72 | 29.58 | 0.6% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 1.726727 | 1686.26 | 15.66 | 1.3% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 5.158088 | 5037.20 | 26.22 | 1.0% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 2.190079 | 2138.75 | 13.85 | 0.8% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 8.576253 | 8375.25 | 72.63 | 1.2% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 2048 | 2.411061 | 1177.28 | 4.55 | 3.4% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 2048 | 2.454091 | 1198.29 | 7.47 | 2.8% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 1024 | 1.494424 | 1459.40 | 17.36 | 3.1% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 2048 | 2.333978 | 1139.64 | 5.17 | 3.6% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 2048 | 2.358673 | 1151.70 | 6.64 | 3.3% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 2048 | 2.510215 | 1225.69 | 274.22 | 17.2% | — | NOISY |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 3.499889 | 3417.86 | 9.50 | 1.0% | — | ok |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 3.520748 | 3438.23 | 17.36 | 0.4% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 3.398259 | 3318.61 | 7.32 | 1.0% | — | ok |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 1024 | 1.707901 | 1667.87 | 10.59 | 0.7% | — | ok |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 2048 | 3.575179 | 1745.69 | 4.88 | 0.4% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 2048 | 3.322608 | 1622.37 | 6.94 | 0.5% | — | ok |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 1024 | 3.160447 | 3086.37 | 9.82 | 1.4% | — | ok |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 1024 | 3.321667 | 3243.82 | 38.32 | 1.5% | — | ok |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 1024 | 3.212012 | 3136.73 | 16.12 | 1.1% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.134796 | 2084.76 | 23.35 | 6.4% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 3.512161 | 3429.84 | 25.78 | 0.9% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 0.997127 | 1947.51 | 909.55 | 24.5% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 3.366049 | 3287.16 | 20.55 | 0.9% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 64 | 1.694325 | 26473.83 | 445.28 | 1.5% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 2.415779 | 9436.64 | 50.53 | 0.6% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 128 | 3.331195 | 26024.96 | 164.75 | 0.5% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 2.382798 | 9307.80 | 78.58 | 1.2% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.925059 | 1428.25 | 13.52 | 4.3% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.403928 | 4695.17 | 26.43 | 0.6% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 3.062184 | 1495.21 | 5.68 | 4.0% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.337264 | 4564.97 | 42.12 | 0.8% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.918196 | 2849.80 | 16.35 | 2.0% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 3.337095 | 6517.76 | 78.66 | 7.9% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.962928 | 2893.48 | 12.49 | 0.6% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 3.424679 | 6688.83 | 94.02 | 1.0% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.664482 | 1301.02 | 16.66 | 2.8% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.442780 | 4771.05 | 24.49 | 0.5% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.614290 | 1276.51 | 2.54 | 2.6% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.457758 | 4800.31 | 32.90 | 0.9% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.115119 | 4131.09 | 37.20 | 0.8% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.763387 | 2698.62 | 5.79 | 0.8% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.092217 | 4086.36 | 17.47 | 2.1% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.776512 | 2711.44 | 11.87 | 0.7% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 2.162228 | 8446.20 | 44.12 | 1.0% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 3.573786 | 6980.05 | 28.30 | 0.3% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 2.211860 | 8640.08 | 96.02 | 1.8% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 3.598442 | 7028.21 | 14.78 | 0.2% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 2.045580 | 7990.55 | 71.21 | 2.3% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.127482 | 4155.24 | 37.41 | 0.5% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 2.040009 | 7968.79 | 129.85 | 1.0% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.108007 | 4117.20 | 14.60 | 0.7% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 1.965521 | 1919.45 | 7.20 | 1.3% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.938754 | 1434.94 | 8.01 | 4.3% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.650176 | 1294.03 | 3.64 | 2.7% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.764549 | 2699.75 | 8.20 | 0.2% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 3.609914 | 7050.61 | 26.65 | 1.1% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.106824 | 4114.89 | 37.42 | 2.1% | — | ok |
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 1024 | 1.493353 | 1458.35 | 14.37 | 19.7% | — | NOISY |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 4 | blas | 1280 | 1.808880 | 14131.88 | 110.64 | 12.4% | 1413.19 | NOISY |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 1024 | 1.480328 | 1445.63 | 55.17 | 19.7% | — | NOISY |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 4 | blas | 1280 | 1.783773 | 13935.73 | 75.22 | 13.1% | 1393.57 | NOISY |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 1.554306 | 1517.88 | 119.59 | 17.4% | — | NOISY |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 4 | blas | 1280 | 1.924355 | 15034.02 | 144.18 | 8.5% | 1503.40 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.135959 | 2085.90 | 17.77 | 0.9% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 2.195020 | 2143.57 | 16.80 | 1.0% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.996283 | 2926.06 | 15.58 | 1.1% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.168251 | 2117.43 | 18.71 | 14.4% | — | NOISY |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 2.235094 | 2182.71 | 12.50 | 0.5% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 3.334223 | 3256.08 | 25.63 | 1.0% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.817637 | 1775.04 | 9.67 | 0.8% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 1.956866 | 1911.00 | 5.86 | 0.4% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 4.739140 | 4628.07 | 44.37 | 0.8% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 2.311287 | 2257.12 | 15.25 | 0.7% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 27.943235 | 27288.32 | 124.72 | 0.4% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 2.834101 | 2767.68 | 12.06 | 0.4% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 38.625980 | 37720.68 | 207.01 | 0.4% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 1024 | 1.841201 | 1798.05 | 20.16 | 19.2% | — | NOISY |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 1024 | 1.875484 | 1831.53 | 25.14 | 18.5% | — | NOISY |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 2.073443 | 2024.85 | 15.71 | 16.3% | — | NOISY |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 1024 | 1.747456 | 1706.50 | 64.25 | 17.2% | — | NOISY |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 1024 | 1.795936 | 1753.84 | 29.82 | 16.2% | — | NOISY |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 1.740954 | 1700.15 | 17.50 | 18.3% | — | NOISY |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 4.020562 | 3926.33 | 28.07 | 0.6% | — | ok |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 4.170491 | 4072.75 | 23.40 | 0.5% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 3.941725 | 3849.34 | 6.10 | 0.4% | — | ok |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 1024 | 2.332476 | 2277.81 | 7.53 | 0.9% | — | ok |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 1024 | 2.331504 | 2276.86 | 14.12 | 1.5% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 1024 | 2.197244 | 2145.75 | 7.49 | 0.5% | — | ok |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 1024 | 3.745448 | 3657.66 | 13.54 | 3.7% | — | ok |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 1024 | 3.846437 | 3756.29 | 16.08 | 5.5% | — | ok |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 1024 | 3.682923 | 3596.60 | 16.90 | 5.4% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.628247 | 2566.65 | 19.00 | 0.9% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.075497 | 4053.71 | 22.63 | 1.0% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.631844 | 2570.16 | 8.07 | 0.3% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.036244 | 3977.04 | 16.58 | 0.4% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 64 | 1.751363 | 27365.05 | 622.80 | 14.7% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 3.044012 | 11890.67 | 156.48 | 0.9% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 64 | 1.738790 | 27168.59 | 1068.71 | 14.8% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 2.907638 | 11357.96 | 78.78 | 0.6% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.089344 | 2040.38 | 22.55 | 0.8% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.635822 | 5148.09 | 68.17 | 1.6% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.005968 | 1958.95 | 5.66 | 0.5% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.606016 | 5089.88 | 41.82 | 0.8% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 3.369849 | 3290.87 | 20.21 | 0.7% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 1.928183 | 7531.96 | 3498.57 | 39.8% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 1.701791 | 3323.81 | 25.02 | 1.2% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 1.817967 | 7101.43 | 62.36 | 1.1% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.865015 | 1821.30 | 6.69 | 0.7% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.832958 | 5533.12 | 14.93 | 1.1% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.868331 | 1824.54 | 5.58 | 0.5% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.800718 | 5470.15 | 29.81 | 0.4% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.463270 | 4811.07 | 30.18 | 1.0% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 1.732418 | 3383.63 | 47.38 | 3.5% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.528963 | 4939.38 | 43.80 | 0.7% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 1.678428 | 3278.18 | 23.23 | 2.1% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 2.312108 | 9031.67 | 61.62 | 0.5% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 3.911177 | 7639.02 | 40.58 | 1.0% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 2.356231 | 9204.03 | 65.29 | 0.8% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 3.867206 | 7553.14 | 57.01 | 0.6% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 2.230887 | 8714.40 | 70.82 | 1.0% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.386968 | 4662.05 | 28.56 | 0.4% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 2.220908 | 8675.42 | 76.55 | 0.6% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.434997 | 4755.85 | 20.00 | 0.4% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.565149 | 2505.03 | 7.59 | 0.4% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.974388 | 1928.11 | 3.05 | 0.2% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.795425 | 1753.34 | 7.75 | 1.5% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 1.724483 | 3368.13 | 11.22 | 0.6% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 3.923620 | 7663.32 | 36.51 | 0.3% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.374163 | 4637.04 | 29.52 | 1.0% | — | ok |

## Matched non-AD eager comparisons (#98)

All requested QR/SVD outputs are retained. Ordinary einsum uses the same parsed integer notation and default Auto strategy; planning/cache lookup stays inside both public calls. These are public-call diagnostics, not prepared execution or an isolated AD cost. Fresh pairs include session entry; shared pairs exclude it. Views are constructed untimed and consumed as views on both routes. No output to_tensor conversion is timed. Capture is inactive by construction in each fresh process, with no capture_trace guard; inputs and validation outputs are checked untracked.

| Operation | Shape | Layout | Session | Threads | Direct ns | Eager ns | Eager − direct ns | Eager/direct | Quality |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 1 | 2084.76 | 3429.84 | 1345.08 | 1.645 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | shared | 1 | 1947.51 | 3287.16 | 1339.65 | 1.688 | inconclusive |
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 4 | 2566.65 | 4053.71 | 1487.06 | 1.579 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | shared | 4 | 2570.16 | 3977.04 | 1406.88 | 1.547 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 1 | 2711.44 | 4086.36 | 1374.92 | 1.507 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | shared | 1 | 2698.62 | 4131.09 | 1432.47 | 1.531 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 4 | 3278.18 | 4939.38 | 1661.20 | 1.507 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | shared | 4 | 3383.63 | 4811.07 | 1427.44 | 1.422 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 1 | 26473.83 | 9436.64 | -17037.19 | 0.356 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | shared | 1 | 26024.96 | 9307.80 | -16717.16 | 0.358 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 4 | 27365.05 | 11890.67 | -15474.38 | 0.435 | inconclusive |
| einsum3 | 2×2 | col_major_contiguous | shared | 4 | 27168.59 | 11357.96 | -15810.63 | 0.418 | inconclusive |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1428.25 | 4695.17 | 3266.92 | 3.287 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | shared | 1 | 1495.21 | 4564.97 | 3069.76 | 3.053 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 4 | 2040.38 | 5148.09 | 3107.71 | 2.523 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | shared | 4 | 1958.95 | 5089.88 | 3130.93 | 2.598 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 1 | 7028.21 | 8640.08 | 1611.87 | 1.229 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | shared | 1 | 6980.05 | 8446.20 | 1466.15 | 1.210 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 4 | 7553.14 | 9204.03 | 1650.89 | 1.219 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | shared | 4 | 7639.02 | 9031.67 | 1392.65 | 1.182 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1301.02 | 4771.05 | 3470.03 | 3.667 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | shared | 1 | 1276.51 | 4800.31 | 3523.80 | 3.760 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 4 | 1821.30 | 5533.12 | 3711.82 | 3.038 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | shared | 4 | 1824.54 | 5470.15 | 3645.61 | 2.998 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 1 | 4117.20 | 7968.79 | 3851.59 | 1.935 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | shared | 1 | 4155.24 | 7990.55 | 3835.31 | 1.923 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 4 | 4755.85 | 8675.42 | 3919.57 | 1.824 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | shared | 4 | 4662.05 | 8714.40 | 4052.35 | 1.869 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 1 | 2849.80 | 6517.76 | 3667.96 | 2.287 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | shared | 1 | 2893.48 | 6688.83 | 3795.35 | 2.312 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 4 | 3290.87 | 7531.96 | 4241.09 | 2.289 | inconclusive |
| svd | 2×2 | col_major_contiguous | shared | 4 | 3323.81 | 7101.43 | 3777.62 | 2.137 | observed; no speedup claim |

## Separate allocation diagnostics

Instrumented executable is built separately after latency collection. Calling-thread Rust allocator requests only; excludes foreign BLAS and worker-thread allocations. Realloc counts its requested new size, not net retained bytes. Output destruction and instrumentation setup are excluded. Helper references and multi-output retention use stack storage. No allocation-instrumented run contributes latency samples.

| Corresponding case ID | Threads | Median allocation requests/op | Median requested bytes/op | Status |
|---|---:|---:|---:|---|
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | 1 | 8 | 120 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | 1 | 15 | 3984 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | 1 | 8 | 120 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-shared | 1 | 15 | 3984 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | 1 | 265 | 43093 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | 1 | 48 | 19728 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | 1 | 265 | 43093 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | 1 | 48 | 19728 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | 1 | 6 | 1280 | passed |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | 1 | 39 | 15371 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-shared | 1 | 6 | 1280 | passed |
| nonad_qr_n2_col_major_contiguous_eager-shared | 1 | 39 | 15371 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | 1 | 7 | 1248 | passed |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | 1 | 45 | 16419 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-shared | 1 | 7 | 1248 | passed |
| nonad_svd_n2_col_major_contiguous_eager-shared | 1 | 45 | 16419 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | 1 | 2 | 48 | passed |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | 1 | 35 | 16083 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-shared | 1 | 2 | 48 | passed |
| nonad_solve_n2_col_major_contiguous_eager-shared | 1 | 35 | 16083 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-shared | 1 | 15 | 3984 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | 1 | 8 | 120 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | 1 | 15 | 3984 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | 1 | 8 | 120 | passed |
| nonad_qr_n16_col_major_contiguous_eager-shared | 1 | 39 | 19403 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-shared | 1 | 6 | 5312 | passed |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | 1 | 39 | 19403 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | 1 | 6 | 5312 | passed |
| nonad_solve_n16_col_major_contiguous_eager-shared | 1 | 35 | 18099 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-shared | 1 | 2 | 2064 | passed |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | 1 | 35 | 18099 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | 1 | 2 | 2064 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | 1 | 8 | 120 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | 1 | 6 | 1280 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | 1 | 2 | 48 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | 1 | 8 | 120 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | 1 | 6 | 5312 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | 1 | 2 | 2064 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | 4 | 8 | 120 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | 4 | 15 | 3984 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | 4 | 8 | 120 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-shared | 4 | 15 | 3984 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | 4 | 265 | 43093 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | 4 | 48 | 19728 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | 4 | 265 | 43093 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | 4 | 48 | 19728 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | 4 | 6 | 1280 | passed |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | 4 | 39 | 15371 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-shared | 4 | 6 | 1280 | passed |
| nonad_qr_n2_col_major_contiguous_eager-shared | 4 | 39 | 15371 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | 4 | 7 | 1248 | passed |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | 4 | 45 | 16419 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-shared | 4 | 7 | 1248 | passed |
| nonad_svd_n2_col_major_contiguous_eager-shared | 4 | 45 | 16419 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | 4 | 2 | 48 | passed |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | 4 | 35 | 16083 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-shared | 4 | 2 | 48 | passed |
| nonad_solve_n2_col_major_contiguous_eager-shared | 4 | 35 | 16083 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-shared | 4 | 15 | 3984 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | 4 | 8 | 120 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | 4 | 15 | 3984 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | 4 | 8 | 120 | passed |
| nonad_qr_n16_col_major_contiguous_eager-shared | 4 | 39 | 19403 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-shared | 4 | 6 | 5312 | passed |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | 4 | 39 | 19403 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | 4 | 6 | 5312 | passed |
| nonad_solve_n16_col_major_contiguous_eager-shared | 4 | 35 | 18099 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-shared | 4 | 2 | 2064 | passed |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | 4 | 35 | 18099 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | 4 | 2 | 2064 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | 4 | 8 | 120 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | 4 | 6 | 1280 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | 4 | 2 | 48 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | 4 | 8 | 120 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | 4 | 6 | 5312 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | 4 | 2 | 2064 | passed |

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
  "collection_total_seconds": 0.519847751999805,
  "case_process_seconds": 0.47047357800329337
}
```

Wall observations `wall_allocations_t4.json`:
```json
{
  "collection_total_seconds": 0.6495039629999155,
  "case_process_seconds": 0.6000940289995924
}
```

Wall observations `wall_command.json`:
```json
{
  "command_seconds_through_collection": 416.85112429,
  "scope": "command startup through collection; excludes report rendering and publication",
  "budget_seconds": 120.0,
  "budget_status": "over_budget; diagnostic only"
}
```

Wall observations `wall_measure_t1.json`:
```json
{
  "collection_total_seconds": 4.373129282000264,
  "case_process_seconds": 4.310476839998046
}
```

Wall observations `wall_measure_t4.json`:
```json
{
  "collection_total_seconds": 5.231609529000707,
  "case_process_seconds": 5.1666741560002265
}
```

Wall observations `wall_phases.json`:
```json
{
  "validation_and_latency_build_seconds": 4.448706143,
  "metadata_and_latency_collection_seconds": 16.281266222,
  "allocation_diagnostic_build_seconds": 394.723264107,
  "allocation_collection_seconds": 1.333303349,
  "command_before_report_seconds": 416.786539821
}
```

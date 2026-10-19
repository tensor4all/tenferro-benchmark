# CPU Small-Work Benchmark Results

- Suite: `cpu/small_work`
- Raw run: `data/results/amd-cpu/cpu/small_work/20261019_184832`

## CPU Information

- Model: `AMD EPYC 9V45 96-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`

## run_t1

Full metadata: `data/results/amd-cpu/cpu/small_work/20261019_184832/run_t1.yaml`

```yaml
timestamp: '2026-10-19T18:48:32Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
  commit: 14bc1ab933e3c7a9c966e08355670331ab9d993c
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
  hostname: a5c7960fb148
  os: Linux-6.17.0-1022-azure-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  env:
    OMP_NUM_THREADS: '1'
    OMP_THREAD_LIMIT: '1'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: amd-cpu
    BENCHMARK_COMMIT: ce89bccec28bc7c863b30c80764d3570747f6df6
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

Full metadata: `data/results/amd-cpu/cpu/small_work/20261019_184832/run_t4.yaml`

```yaml
timestamp: '2026-10-19T18:48:38Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
  commit: 14bc1ab933e3c7a9c966e08355670331ab9d993c
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
  hostname: a5c7960fb148
  os: Linux-6.17.0-1022-azure-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  env:
    OMP_NUM_THREADS: '4'
    OMP_THREAD_LIMIT: '4'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '4'
    BENCHMARK_TARGET_PROFILE: amd-cpu
    BENCHMARK_COMMIT: ce89bccec28bc7c863b30c80764d3570747f6df6
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
- Expected 144, selected 144, executed 144, unsupported 0, failed 0, missing 0, noisy 14.
- Complete: **yes** — only executed cases count as covered; unsupported, failed, missing and unselected cases never do.
- Noisy: `add_f64_concrete_shared@t1`, `add_f64_concrete-shared_size256_single@t1`, `add_f64_concrete-shared_size256_dependent10@t1`, `einsum_f64_prepared-repeat_n16_single@t1`, `solve_f64_concrete-shared_n16_single@t1`, `gather_f64_concrete_shared@t1`, `gather_f64_concrete-shared_size256_single@t1`, `nonad_solve_n16_col_major_contiguous_concrete-shared@t1`, `nonad_solve_n16_col_major_contiguous_concrete-fresh@t1`, `nonad_solve_n2_col_major_contiguous_concrete-scoped@t1`, `nonad_solve_n16_col_major_contiguous_concrete-scoped@t1`, `solve_f64_concrete-shared_n16_single@t4`, `nonad_solve_n16_col_major_contiguous_concrete-shared@t4`, `nonad_solve_n16_col_major_contiguous_concrete-scoped@t4`

Numerical checks: **144/144 recorded rows passed** (case × thread count).

## Timing

Sequential release runs; explicit initialization and warmups, then calibration toward the declared batch duration (2 ms by default), then the measured batches of the recorded effort (standard: 3 warmups and 15 batches; scan: a low-repetition screen that only marks suspects). Legacy single-call routes start calibration at 1024 workflows; matched non-AD and dependent-chain routes start at 16 and bound retained memory. Each batch uses one wall-clock interval; all outputs are retained until it stops. Statistics divide batch time by iterations. Median and inclusive IQR are in nanoseconds (ns); CoV is sample standard deviation / mean. NOISY means CoV > 10%, a descriptive label only. No noisy rows are excluded.

Workflow time is the total for one call or the complete dependent chain. The separate ns/op column divides the chain median by its operation count, not an independently measured call. Setup rows measure preparation only, not execution.

Inputs and backend construction, independent numerical checks and AD backward checks are outside timing. Returned outputs/plans and every dependent-chain intermediate are retained until after each batch interval; retention containers are prepared before the clock. value downloads/checks are outside. Eager AD rows measure forward execution/recording, not backward.

| Case ID | Operation | API route | Route (path / output / repr / session) | Dtype | Shape | Layout | Workflow | Threads | Provider | Operations/batch | Median batch ms | Median workflow ns | IQR ns | CoV | ns/op (chain only) | Status |
|---|---|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 4096 | 2.035869 | 497.04 | 5.96 | 5.0% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 1 | blas | 5120 | 2.408010 | 4703.14 | 21.73 | 0.7% | 470.31 | ok |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 4096 | 2.066516 | 504.52 | 6.02 | 12.4% | — | NOISY |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 1 | blas | 5120 | 2.434140 | 4754.18 | 239.56 | 8.7% | 475.42 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 2048 | 1.132066 | 552.77 | 242.45 | 33.9% | — | NOISY |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 1 | blas | 2560 | 1.399490 | 5466.76 | 1754.68 | 26.5% | 546.68 | NOISY |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 1.944873 | 1899.29 | 83.73 | 4.7% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 1.876098 | 1832.13 | 33.24 | 4.3% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 5.025567 | 4907.78 | 1554.84 | 29.2% | — | NOISY |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.379768 | 2323.99 | 180.26 | 8.7% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 1.875117 | 1831.17 | 174.43 | 5.1% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.626300 | 2564.75 | 62.45 | 1.9% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 4096 | 2.919381 | 712.74 | 7.98 | 2.0% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 4096 | 3.265233 | 797.18 | 17.83 | 1.9% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.710327 | 2646.80 | 633.81 | 12.4% | — | NOISY |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 1.930360 | 1885.12 | 54.79 | 1.5% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 4.177559 | 4079.65 | 199.51 | 3.4% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 2.182731 | 2131.57 | 37.27 | 1.7% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 7.195429 | 7026.79 | 386.24 | 6.1% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 4096 | 3.061716 | 747.49 | 5.49 | 3.4% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 4096 | 3.402700 | 830.74 | 100.25 | 20.7% | — | NOISY |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 1024 | 1.088411 | 1062.90 | 346.33 | 19.4% | — | NOISY |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 4096 | 2.204414 | 538.19 | 7.03 | 5.1% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 4096 | 2.287539 | 558.48 | 21.52 | 4.6% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 4096 | 2.471777 | 603.46 | 22.42 | 5.6% | — | ok |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 3.160185 | 3086.12 | 168.71 | 3.4% | — | ok |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 1.974597 | 1928.32 | 106.32 | 3.0% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 1.931472 | 1886.20 | 97.00 | 2.8% | — | ok |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 1024 | 1.967797 | 1921.68 | 87.62 | 2.9% | — | ok |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 2048 | 2.025544 | 989.04 | 37.30 | 2.4% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 2048 | 1.886434 | 921.11 | 36.27 | 2.1% | — | ok |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 1024 | 2.623776 | 2562.28 | 80.62 | 2.0% | — | ok |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 2048 | 3.474348 | 1696.46 | 96.90 | 3.2% | — | ok |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 2048 | 3.345373 | 1633.48 | 9.72 | 1.9% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 3.065432 | 5987.17 | 142.27 | 2.3% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.780603 | 2715.43 | 59.76 | 2.1% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.644097 | 5164.25 | 137.01 | 2.2% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 1.831311 | 1788.39 | 21.74 | 0.8% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 3.520711 | 13752.78 | 127.19 | 0.9% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 3.076974 | 6009.71 | 26.63 | 0.6% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 3.390535 | 13244.28 | 71.38 | 0.9% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.688610 | 5251.19 | 16.32 | 0.4% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 3.254150 | 1588.94 | 28.22 | 3.2% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 1.981128 | 3869.39 | 66.47 | 0.9% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 1.621858 | 791.92 | 8.60 | 5.8% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.982021 | 2912.13 | 12.45 | 0.4% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.476922 | 2418.87 | 14.49 | 0.8% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.402180 | 4691.76 | 29.13 | 0.8% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 3.314361 | 1618.34 | 23.33 | 5.1% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.121518 | 4143.59 | 17.22 | 0.5% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 3.043804 | 1486.23 | 14.40 | 3.3% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 1.946936 | 3802.61 | 55.89 | 1.0% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 1.525643 | 744.94 | 9.19 | 5.3% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.904955 | 2836.87 | 23.51 | 0.6% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.260106 | 2207.13 | 8.36 | 0.5% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.805476 | 5479.45 | 48.16 | 0.6% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 3.197225 | 3122.29 | 55.38 | 1.4% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 3.325909 | 6495.92 | 116.84 | 1.2% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.640798 | 5157.81 | 105.90 | 1.2% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 4.398020 | 4294.94 | 85.60 | 1.2% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 3.088021 | 6031.29 | 63.70 | 0.7% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.657895 | 5191.20 | 47.65 | 0.6% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.346036 | 4582.10 | 44.83 | 0.6% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.383582 | 2327.72 | 658.23 | 13.0% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.787329 | 5444.00 | 41.52 | 0.5% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 3.298477 | 3221.17 | 657.18 | 10.1% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.789973 | 5449.17 | 49.86 | 0.7% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.248278 | 1097.79 | 18.07 | 4.5% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.223331 | 1085.61 | 239.52 | 25.8% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.958296 | 5777.92 | 42.09 | 1.3% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.413017 | 4712.92 | 30.43 | 2.3% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 3.034741 | 2963.61 | 621.52 | 10.2% | — | NOISY |
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 1024 | 7.597995 | 7419.92 | 101.71 | 2.6% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 4 | blas | 320 | 2.426785 | 75837.03 | 2316.88 | 2.3% | 7583.70 | ok |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 1024 | 7.479387 | 7304.09 | 285.74 | 2.5% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 4 | blas | 320 | 2.346269 | 73320.91 | 1142.73 | 1.1% | 7332.09 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 7.634862 | 7455.92 | 203.19 | 7.2% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 4 | blas | 320 | 2.375608 | 74237.75 | 1571.53 | 3.7% | 7423.77 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 3.685920 | 1799.77 | 10.80 | 0.6% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 2048 | 3.650736 | 1782.59 | 11.69 | 0.4% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.262312 | 2209.29 | 22.79 | 1.8% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 3.672971 | 1793.44 | 8.20 | 0.7% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 1.833325 | 1790.36 | 9.08 | 0.6% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.742787 | 2678.50 | 96.72 | 3.0% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 4096 | 2.903239 | 708.80 | 17.58 | 2.7% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 4096 | 3.127628 | 763.58 | 4.28 | 2.0% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.453751 | 2396.24 | 626.50 | 12.1% | — | NOISY |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 1.853205 | 1809.77 | 5.97 | 0.3% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 6.030025 | 5888.70 | 105.20 | 2.6% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 2.161550 | 2110.89 | 17.28 | 1.4% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 10.356161 | 10113.44 | 265.97 | 2.7% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 1024 | 8.163563 | 7972.23 | 463.90 | 7.3% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 1024 | 7.579582 | 7401.94 | 48.95 | 4.2% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 7.985760 | 7798.59 | 280.68 | 3.0% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 1024 | 7.436452 | 7262.16 | 200.38 | 5.1% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 1024 | 7.507373 | 7331.42 | 238.81 | 7.2% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 7.542497 | 7365.72 | 231.25 | 3.5% | — | ok |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 3.153837 | 3079.92 | 38.36 | 1.1% | — | ok |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 1.950652 | 1904.93 | 4.53 | 0.5% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 1.884502 | 1840.33 | 23.11 | 0.8% | — | ok |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 1024 | 1.947428 | 1901.79 | 4.20 | 0.3% | — | ok |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 2048 | 2.055220 | 1003.53 | 7.71 | 0.7% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 4096 | 3.663476 | 894.40 | 2.33 | 1.7% | — | ok |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 1024 | 2.621554 | 2560.11 | 16.92 | 1.5% | — | ok |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 2048 | 3.421280 | 1670.55 | 13.22 | 2.0% | — | ok |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 2048 | 3.267268 | 1595.35 | 7.78 | 1.9% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 3.001037 | 5861.40 | 10.11 | 0.2% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.673843 | 2611.17 | 11.15 | 0.3% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.711070 | 5295.06 | 26.18 | 0.5% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.805954 | 1763.63 | 6.85 | 0.5% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 3.454821 | 13495.39 | 65.63 | 0.7% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 128 | 2.036798 | 15912.48 | 740.41 | 2.8% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 3.276212 | 12797.70 | 84.88 | 0.7% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 128 | 1.922270 | 15017.73 | 447.22 | 1.7% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 3.158734 | 1542.35 | 8.33 | 3.0% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 3.347769 | 13077.22 | 402.68 | 3.9% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 1.631851 | 796.80 | 3.63 | 5.4% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 2.841186 | 11098.38 | 1365.35 | 6.2% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.374251 | 2318.60 | 6.83 | 0.3% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 3.713031 | 14504.03 | 485.30 | 2.9% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 3.214017 | 1569.34 | 8.95 | 5.6% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 3.391566 | 13248.30 | 309.79 | 1.7% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 3.014060 | 1471.71 | 6.42 | 2.7% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 1.879031 | 3669.98 | 16.22 | 0.4% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 1.471036 | 718.28 | 3.91 | 5.4% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.908802 | 2840.63 | 9.98 | 0.2% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.273104 | 2219.83 | 12.54 | 0.7% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.734959 | 5341.72 | 17.32 | 0.4% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 3.155824 | 3081.86 | 17.82 | 0.5% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 3.139469 | 6131.78 | 21.54 | 0.4% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 128 | 1.927609 | 15059.45 | 458.86 | 2.4% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 4.323944 | 4222.60 | 17.32 | 0.7% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 128 | 2.070839 | 16178.43 | 537.74 | 2.7% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.535309 | 4951.78 | 14.40 | 0.4% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.291481 | 4475.55 | 37.03 | 1.1% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.448117 | 2390.74 | 648.11 | 12.5% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.719467 | 5311.46 | 29.78 | 0.7% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 3.264217 | 3187.71 | 644.18 | 9.4% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.698364 | 5270.24 | 20.65 | 0.5% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 2.241436 | 1094.45 | 8.16 | 4.4% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 2.100034 | 1025.41 | 6.48 | 4.3% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.912046 | 5687.59 | 41.12 | 0.6% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.309569 | 4510.88 | 23.71 | 0.4% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.828230 | 2761.94 | 653.45 | 11.1% | — | NOISY |

## Matched non-AD eager comparisons (#98)

All requested QR/SVD outputs are retained. Ordinary einsum uses the same parsed integer notation and default Auto strategy; planning/cache lookup stays inside both public calls. These are public-call diagnostics, not prepared execution or an isolated AD cost. Fresh pairs include session entry; shared pairs exclude it. Views are constructed untimed and consumed as views on both routes. No output to_tensor conversion is timed. Capture is inactive by construction in each fresh process, with no capture_trace guard; inputs and validation outputs are checked untracked.

| Operation | Shape | Layout | Session | Threads | Direct ns | Eager ns | Eager − direct ns | Eager/direct | Quality |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 1 | 5987.17 | 2715.43 | -3271.74 | 0.454 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | shared | 1 | 5164.25 | 1788.39 | -3375.86 | 0.346 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 4 | 5861.40 | 2611.17 | -3250.23 | 0.445 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | shared | 4 | 5295.06 | 1763.63 | -3531.43 | 0.333 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 1 | 6495.92 | 3122.29 | -3373.63 | 0.481 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | shared | 1 | 5479.45 | 2207.13 | -3272.32 | 0.403 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 4 | 6131.78 | 3081.86 | -3049.92 | 0.503 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | shared | 4 | 5341.72 | 2219.83 | -3121.89 | 0.416 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 1 | 13752.78 | 6009.71 | -7743.07 | 0.437 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | shared | 1 | 13244.28 | 5251.19 | -7993.09 | 0.396 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 4 | 13495.39 | 15912.48 | 2417.09 | 1.179 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | shared | 4 | 12797.70 | 15017.73 | 2220.03 | 1.173 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1588.94 | 3869.39 | 2280.45 | 2.435 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | shared | 1 | 791.92 | 2912.13 | 2120.21 | 3.677 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 4 | 1542.35 | 13077.22 | 11534.87 | 8.479 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | shared | 4 | 796.80 | 11098.38 | 10301.58 | 13.929 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 1 | 5191.20 | 6031.29 | 840.09 | 1.162 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | shared | 1 | 4294.94 | 5157.81 | 862.87 | 1.201 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 4 | 4951.78 | 16178.43 | 11226.65 | 3.267 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | shared | 4 | 4222.60 | 15059.45 | 10836.85 | 3.566 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1486.23 | 3802.61 | 2316.38 | 2.559 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | shared | 1 | 744.94 | 2836.87 | 2091.93 | 3.808 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 4 | 1471.71 | 3669.98 | 2198.27 | 2.494 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | shared | 4 | 718.28 | 2840.63 | 2122.35 | 3.955 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 1 | 3221.17 | 5444.00 | 2222.83 | 1.690 | inconclusive |
| solve | 16×16 | col_major_contiguous | shared | 1 | 2327.72 | 4582.10 | 2254.38 | 1.968 | inconclusive |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 4 | 3187.71 | 5311.46 | 2123.75 | 1.666 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | shared | 4 | 2390.74 | 4475.55 | 2084.81 | 1.872 | inconclusive |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 1 | 2418.87 | 4691.76 | 2272.89 | 1.940 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | shared | 1 | 1618.34 | 4143.59 | 2525.25 | 2.560 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 4 | 2318.60 | 14504.03 | 12185.43 | 6.256 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | shared | 4 | 1569.34 | 13248.30 | 11678.96 | 8.442 | observed; no speedup claim |

## Separate allocation diagnostics

Instrumented executable is built separately after latency collection. Calling-thread Rust allocator requests only; excludes foreign BLAS and worker-thread allocations. Realloc counts its requested new size, not net retained bytes. Output destruction and instrumentation setup are excluded. Helper references and multi-output retention use stack storage. No allocation-instrumented run contributes latency samples.

| Corresponding case ID | Threads | Median allocation requests/op | Median requested bytes/op | Status |
|---|---:|---:|---:|---|
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | 1 | 118 | 13215 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | 1 | 17 | 4384 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | 1 | 116 | 12951 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-shared | 1 | 15 | 4120 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | 1 | 267 | 52141 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | 1 | 47 | 12352 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | 1 | 265 | 51877 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | 1 | 45 | 12088 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | 1 | 8 | 1544 | passed |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | 1 | 40 | 6995 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-shared | 1 | 6 | 1280 | passed |
| nonad_qr_n2_col_major_contiguous_eager-shared | 1 | 38 | 6731 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | 1 | 11 | 2544 | passed |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | 1 | 48 | 9323 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-shared | 1 | 8 | 2088 | passed |
| nonad_svd_n2_col_major_contiguous_eager-shared | 1 | 45 | 8867 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | 1 | 6 | 536 | passed |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | 1 | 37 | 7867 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-shared | 1 | 3 | 80 | passed |
| nonad_solve_n2_col_major_contiguous_eager-shared | 1 | 34 | 7411 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-shared | 1 | 15 | 4120 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | 1 | 116 | 12951 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | 1 | 17 | 4384 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | 1 | 118 | 13215 | passed |
| nonad_qr_n16_col_major_contiguous_eager-shared | 1 | 38 | 10763 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-shared | 1 | 6 | 5312 | passed |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | 1 | 40 | 11027 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | 1 | 8 | 5576 | passed |
| nonad_solve_n16_col_major_contiguous_eager-shared | 1 | 34 | 9427 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-shared | 1 | 3 | 4112 | passed |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | 1 | 37 | 9883 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | 1 | 6 | 4568 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | 1 | 117 | 13143 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | 1 | 7 | 1472 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | 1 | 5 | 464 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | 1 | 117 | 13143 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | 1 | 7 | 5504 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | 1 | 5 | 4496 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | 4 | 118 | 13215 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | 4 | 17 | 4384 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | 4 | 116 | 12951 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-shared | 4 | 15 | 4120 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | 4 | 267 | 52141 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | 4 | 47 | 12352 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | 4 | 265 | 51877 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | 4 | 45 | 12088 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | 4 | 8 | 1544 | passed |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | 4 | 39 | 6803 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-shared | 4 | 6 | 1280 | passed |
| nonad_qr_n2_col_major_contiguous_eager-shared | 4 | 38 | 6731 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | 4 | 11 | 2544 | passed |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | 4 | 47 | 9131 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-shared | 4 | 8 | 2088 | passed |
| nonad_svd_n2_col_major_contiguous_eager-shared | 4 | 45 | 8867 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | 4 | 6 | 536 | passed |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | 4 | 37 | 7867 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-shared | 4 | 3 | 80 | passed |
| nonad_solve_n2_col_major_contiguous_eager-shared | 4 | 34 | 7411 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-shared | 4 | 15 | 4120 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | 4 | 116 | 12951 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | 4 | 17 | 4384 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | 4 | 118 | 13215 | passed |
| nonad_qr_n16_col_major_contiguous_eager-shared | 4 | 38 | 10763 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-shared | 4 | 6 | 5312 | passed |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | 4 | 39 | 10835 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | 4 | 8 | 5576 | passed |
| nonad_solve_n16_col_major_contiguous_eager-shared | 4 | 34 | 9427 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-shared | 4 | 3 | 4112 | passed |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | 4 | 37 | 9883 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | 4 | 6 | 4568 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | 4 | 117 | 13143 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | 4 | 7 | 1472 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | 4 | 5 | 464 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | 4 | 117 | 13143 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | 4 | 7 | 5504 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | 4 | 5 | 4496 | passed |

## Per-route timing boundaries

Shared-session routes enter/exit the session once around warmup/calibration/all samples, outside timing. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions reuse the entered executor context. Earlier BLAS revisions included per-operation executor entry; consult the recorded tenferro commit. Fresh routes enter/exit for each operation. Prepared-repeat excludes plan preparation; compiled-repeat excludes tracing/compilation and runtime construction. Prepared-into-repeat and dot-general-into-shared write into one destination preallocated outside timing (no per-call output allocation) and re-check it after the last batch.

- **add / concrete-shared** — inside: add, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **einsum / concrete-fresh** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), per-call session entry/exit diagnostic, ordinary einsum default Auto planning/cache lookup included equally; parsed integer notation; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **einsum / concrete-scoped** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), per-call session entry/exit diagnostic, ordinary einsum default Auto planning/cache lookup included equally; parsed integer notation; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction, execution scope admission/exit; cloned witness setup.
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
- **qr / concrete-scoped** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), per-call session entry/exit diagnostic, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction, execution scope admission/exit; cloned witness setup.
- **qr / concrete-shared** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), session entry excluded, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **qr / eager-no-ad** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), per-call session entry/exit diagnostic, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **qr / eager-shared** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), session entry excluded, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **reduce_sum / concrete-shared** — inside: reduce_sum, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **solve / concrete-fresh** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), per-call session entry/exit diagnostic, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **solve / concrete-scoped** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), per-call session entry/exit diagnostic, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction, execution scope admission/exit; cloned witness setup.
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
  "collection_total_seconds": 0.46513777399968603,
  "case_process_seconds": 0.4396538369987866
}
```

Wall observations `wall_allocations_t4.json`:
```json
{
  "collection_total_seconds": 0.4697276380002222,
  "case_process_seconds": 0.4442246739999973
}
```

Wall observations `wall_command.json`:
```json
{
  "command_seconds_before_final_report_write": 233.475905285,
  "report_generation_seconds": 0.088164446,
  "scope": "command startup through report generation; excludes final report file writes",
  "budget_seconds": 120.0,
  "budget_status": "over_budget; diagnostic only"
}
```

Wall observations `wall_measure_t1.json`:
```json
{
  "collection_total_seconds": 4.229802161999942,
  "case_process_seconds": 4.193315183000323
}
```

Wall observations `wall_measure_t4.json`:
```json
{
  "collection_total_seconds": 5.006564179999714,
  "case_process_seconds": 4.971883152998544
}
```

Wall observations `wall_phases.json`:
```json
{
  "validation_and_latency_build_seconds": 3.021165808,
  "metadata_and_latency_collection_seconds": 13.770352992,
  "allocation_diagnostic_build_seconds": 215.54155458,
  "allocation_collection_seconds": 1.054667459,
  "command_before_report_seconds": 233.387740839
}
```

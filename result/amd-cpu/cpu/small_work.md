# CPU Small-Work Benchmark Results

- Suite: `cpu/small_work`
- Raw run: `data/results/amd-cpu/cpu/small_work/20261021_191107`

## CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`

## run_t1

Full metadata: `data/results/amd-cpu/cpu/small_work/20261021_191107/run_t1.yaml`

```yaml
timestamp: '2026-10-21T19:11:07Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
  commit: f1d4d8d961ff6b937768111888ad2a493314892a
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
  hostname: 42da23793576
  os: Linux-6.17.0-1022-azure-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  env:
    OMP_NUM_THREADS: '1'
    OMP_THREAD_LIMIT: '1'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: amd-cpu
    BENCHMARK_COMMIT: aecee8b998098c968f4a0f59d6228ac81d3e7b09
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

Full metadata: `data/results/amd-cpu/cpu/small_work/20261021_191107/run_t4.yaml`

```yaml
timestamp: '2026-10-21T19:11:15Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
  commit: f1d4d8d961ff6b937768111888ad2a493314892a
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
  hostname: 42da23793576
  os: Linux-6.17.0-1022-azure-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  env:
    OMP_NUM_THREADS: '4'
    OMP_THREAD_LIMIT: '4'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '4'
    BENCHMARK_TARGET_PROFILE: amd-cpu
    BENCHMARK_COMMIT: aecee8b998098c968f4a0f59d6228ac81d3e7b09
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
- Expected 144, selected 144, executed 144, unsupported 0, failed 0, missing 0, noisy 1.
- Complete: **yes** — only executed cases count as covered; unsupported, failed, missing and unselected cases never do.
- Noisy: `einsum_c64_prepared-repeat_n16_single@t4`

Numerical checks: **144/144 recorded rows passed** (case × thread count).

## Timing

Sequential release runs; explicit initialization and warmups, then calibration toward the declared batch duration (2 ms by default), then the measured batches of the recorded effort (standard: 3 warmups and 15 batches; scan: a low-repetition screen that only marks suspects). Legacy single-call routes start calibration at 1024 workflows; matched non-AD and dependent-chain routes start at 16 and bound retained memory. Each batch uses one wall-clock interval; all outputs are retained until it stops. Statistics divide batch time by iterations. Median and inclusive IQR are in nanoseconds (ns); CoV is sample standard deviation / mean. NOISY means CoV > 10%, a descriptive label only. No noisy rows are excluded.

Workflow time is the total for one call or the complete dependent chain. The separate ns/op column divides the chain median by its operation count, not an independently measured call. Setup rows measure preparation only, not execution.

Inputs and backend construction, independent numerical checks and AD backward checks are outside timing. Returned outputs/plans and every dependent-chain intermediate are retained until after each batch interval; retention containers are prepared before the clock. value downloads/checks are outside. Eager AD rows measure forward execution/recording, not backward.

| Case ID | Operation | API route | Route (path / output / repr / session) | Dtype | Shape | Layout | Workflow | Threads | Provider | Operations/batch | Median batch ms | Median workflow ns | IQR ns | CoV | ns/op (chain only) | Status |
|---|---|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 2048 | 1.693214 | 826.76 | 6.84 | 3.9% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 1 | blas | 2560 | 2.038096 | 7961.31 | 106.51 | 1.2% | 796.13 | ok |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 2048 | 1.808318 | 882.97 | 15.08 | 4.0% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 1 | blas | 2560 | 2.138694 | 8354.27 | 130.73 | 1.3% | 835.43 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 1024 | 0.959888 | 937.39 | 33.15 | 5.9% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 1 | blas | 2560 | 2.388178 | 9328.82 | 529.72 | 3.5% | 932.88 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 3.298574 | 3221.26 | 13.15 | 0.8% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 3.320014 | 3242.20 | 19.96 | 0.4% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 4.163054 | 4065.48 | 27.69 | 0.7% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 3.358796 | 3280.07 | 20.91 | 1.3% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 3.372101 | 3293.07 | 16.72 | 0.4% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 4.586163 | 4478.67 | 24.84 | 1.0% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.648303 | 1293.12 | 11.76 | 0.6% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 2048 | 2.761534 | 1348.41 | 17.49 | 1.9% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 4.597234 | 4489.49 | 877.85 | 9.5% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 3.430290 | 3349.89 | 36.30 | 1.1% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 7.169184 | 7001.16 | 75.47 | 1.2% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 3.878705 | 3787.80 | 5.50 | 0.4% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 10.628779 | 10379.67 | 213.57 | 1.7% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 2048 | 2.370195 | 1157.32 | 12.85 | 3.5% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 2048 | 2.412424 | 1177.94 | 5.99 | 3.2% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 1024 | 1.509181 | 1473.81 | 29.77 | 3.8% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 2048 | 2.012789 | 982.81 | 5.58 | 3.6% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 2048 | 2.027397 | 989.94 | 10.22 | 3.4% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 2048 | 1.955573 | 954.87 | 5.69 | 4.5% | — | ok |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 5.695209 | 5561.73 | 37.97 | 0.7% | — | ok |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 3.616717 | 3531.95 | 31.17 | 1.8% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 3.405995 | 3326.17 | 24.81 | 0.5% | — | ok |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 1024 | 3.537509 | 3454.60 | 50.43 | 1.0% | — | ok |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 2048 | 3.560191 | 1738.37 | 18.59 | 0.7% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 2048 | 3.524415 | 1720.91 | 18.70 | 0.7% | — | ok |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 1024 | 4.859613 | 4745.72 | 50.12 | 1.5% | — | ok |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 1024 | 3.190633 | 3115.85 | 35.49 | 1.7% | — | ok |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 1024 | 3.181436 | 3106.87 | 60.44 | 2.1% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 2.252366 | 8798.30 | 83.83 | 0.6% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 1.731395 | 3381.63 | 19.85 | 0.7% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 2.226377 | 8696.79 | 65.32 | 0.8% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 1.752204 | 3422.27 | 61.01 | 1.0% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 128 | 3.223805 | 25185.98 | 471.30 | 1.1% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 2.663702 | 10405.09 | 151.04 | 1.4% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 128 | 3.222122 | 25172.83 | 307.88 | 0.8% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 2.515776 | 9827.25 | 146.47 | 1.1% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.816958 | 1375.47 | 19.82 | 4.3% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.700750 | 5274.90 | 40.89 | 0.7% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.746135 | 1340.89 | 6.71 | 4.5% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.595344 | 5069.03 | 66.77 | 0.9% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.865718 | 2798.55 | 14.28 | 0.7% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 1.889510 | 7380.90 | 137.62 | 1.7% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.893790 | 2825.97 | 28.59 | 0.8% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 2.075136 | 8106.00 | 75.55 | 0.6% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.618458 | 1278.54 | 17.49 | 4.1% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.387527 | 4663.14 | 77.56 | 1.5% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.641330 | 1289.71 | 12.61 | 4.4% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.506739 | 4895.97 | 157.75 | 3.0% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.112635 | 4126.24 | 33.49 | 0.8% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 2.557804 | 9991.42 | 73.41 | 0.9% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.155175 | 4209.33 | 17.51 | 0.7% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 2.503984 | 9781.19 | 58.43 | 0.4% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 2.425969 | 9476.44 | 98.03 | 1.1% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 3.483168 | 6803.06 | 25.33 | 0.3% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 2.417633 | 9443.88 | 96.18 | 0.8% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 3.500631 | 6837.17 | 30.53 | 0.6% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 2.000416 | 7814.12 | 164.27 | 1.4% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.128485 | 4157.20 | 22.55 | 0.8% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 2.013851 | 7866.61 | 140.24 | 1.3% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.114949 | 4130.76 | 52.53 | 1.0% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 2.270710 | 8869.96 | 93.65 | 1.4% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.775671 | 1355.31 | 12.13 | 4.2% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.614511 | 1276.62 | 9.79 | 5.0% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 2.528670 | 9877.62 | 85.73 | 1.0% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 3.485492 | 6807.60 | 47.55 | 1.1% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.133013 | 4166.04 | 55.74 | 0.7% | — | ok |
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 2048 | 1.829238 | 893.18 | 7.06 | 4.0% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 4 | blas | 2560 | 2.189750 | 8553.71 | 69.41 | 1.1% | 855.37 | ok |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 2048 | 1.939584 | 947.06 | 9.42 | 5.0% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 4 | blas | 2560 | 2.199438 | 8591.55 | 108.33 | 0.8% | 859.15 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 1.011354 | 987.65 | 16.04 | 6.2% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 4 | blas | 2560 | 2.450646 | 9572.84 | 254.36 | 2.6% | 957.28 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 3.330826 | 3252.76 | 19.39 | 1.6% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 3.354010 | 3275.40 | 30.96 | 0.6% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 4.215614 | 4116.81 | 7.91 | 1.3% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 3.324885 | 3246.96 | 7.53 | 0.2% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 3.435532 | 3355.01 | 32.60 | 0.7% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 4.695299 | 4585.25 | 97.10 | 12.0% | — | NOISY |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 2.611486 | 1275.14 | 3.70 | 0.6% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 2048 | 2.751597 | 1343.55 | 2.75 | 0.4% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 4.388136 | 4285.29 | 854.72 | 9.4% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 3.562298 | 3478.81 | 14.48 | 0.3% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 12.712177 | 12414.24 | 218.60 | 1.5% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 3.950541 | 3857.95 | 23.66 | 1.1% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 23.527519 | 22976.09 | 113.34 | 1.1% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 2048 | 2.418727 | 1181.02 | 9.25 | 3.3% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 2048 | 2.494779 | 1218.15 | 12.70 | 3.1% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 1.491389 | 1456.43 | 25.81 | 2.9% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 2048 | 2.074275 | 1012.83 | 10.69 | 3.5% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 2048 | 2.015515 | 984.14 | 2.75 | 3.7% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 2048 | 2.128757 | 1039.43 | 3.92 | 3.7% | — | ok |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 5.657872 | 5525.27 | 62.73 | 0.8% | — | ok |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 3.621808 | 3536.92 | 27.87 | 0.7% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 3.609666 | 3525.06 | 9.17 | 0.5% | — | ok |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 1024 | 3.550456 | 3467.24 | 16.71 | 1.1% | — | ok |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 2048 | 3.572587 | 1744.43 | 5.11 | 0.5% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 2048 | 3.340114 | 1630.92 | 4.56 | 1.5% | — | ok |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 1024 | 4.923093 | 4807.71 | 19.04 | 1.6% | — | ok |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 1024 | 3.238845 | 3162.93 | 19.76 | 2.1% | — | ok |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 1024 | 3.130593 | 3057.22 | 28.66 | 1.6% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 2.312730 | 9034.10 | 154.08 | 0.9% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 1.726737 | 3372.53 | 55.71 | 1.3% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 2.283195 | 8918.73 | 90.05 | 1.1% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 3.374218 | 3295.13 | 26.85 | 0.6% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 128 | 3.162753 | 24709.01 | 223.74 | 0.6% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 2.586560 | 10103.75 | 59.08 | 1.3% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 128 | 3.174305 | 24799.26 | 299.82 | 1.2% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 2.571542 | 10045.09 | 141.00 | 1.5% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 2.776163 | 1355.55 | 5.76 | 4.5% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.697025 | 5267.63 | 53.34 | 0.9% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 2.866151 | 1399.49 | 15.38 | 4.6% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.682408 | 5239.08 | 37.66 | 0.8% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.836305 | 2769.83 | 7.65 | 0.5% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 1.948310 | 7610.59 | 56.53 | 0.8% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.766475 | 2701.64 | 15.55 | 0.9% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 1.907805 | 7452.36 | 54.67 | 1.1% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 2.724406 | 1330.28 | 12.78 | 4.2% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.360369 | 4610.10 | 13.79 | 2.6% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 2.593092 | 1266.16 | 10.37 | 4.2% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.387839 | 4663.75 | 58.98 | 1.0% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.112296 | 4125.58 | 49.98 | 0.9% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 2.499778 | 9764.76 | 47.38 | 0.5% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.118297 | 4137.30 | 104.93 | 2.3% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 2.589566 | 10115.49 | 129.91 | 0.9% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 2.484088 | 9703.47 | 117.97 | 1.6% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 3.461129 | 6760.02 | 34.02 | 0.5% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 2.479600 | 9685.94 | 166.78 | 2.8% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 3.515701 | 6866.60 | 25.60 | 0.5% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 2.020625 | 7893.07 | 64.34 | 2.1% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.162680 | 4223.98 | 14.42 | 0.2% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 2.029702 | 7928.52 | 72.40 | 1.1% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.156428 | 4211.77 | 34.71 | 0.8% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 2.347133 | 9168.49 | 102.57 | 0.8% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 2.772727 | 1353.87 | 8.17 | 4.4% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 2.682057 | 1309.60 | 10.91 | 5.1% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 2.558367 | 9993.62 | 102.56 | 0.6% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 3.576213 | 6984.79 | 55.86 | 0.7% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.140609 | 4180.88 | 39.15 | 0.6% | — | ok |

## Matched non-AD eager comparisons (#98)

All requested QR/SVD outputs are retained. Ordinary einsum uses the same parsed integer notation and default Auto strategy; planning/cache lookup stays inside both public calls. These are public-call diagnostics, not prepared execution or an isolated AD cost. Fresh pairs include session entry; shared pairs exclude it. Views are constructed untimed and consumed as views on both routes. No output to_tensor conversion is timed. Capture is inactive by construction in each fresh process, with no capture_trace guard; inputs and validation outputs are checked untracked.

| Operation | Shape | Layout | Session | Threads | Direct ns | Eager ns | Eager − direct ns | Eager/direct | Quality |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 1 | 8798.30 | 3381.63 | -5416.67 | 0.384 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | shared | 1 | 8696.79 | 3422.27 | -5274.52 | 0.394 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 4 | 9034.10 | 3372.53 | -5661.57 | 0.373 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | shared | 4 | 8918.73 | 3295.13 | -5623.60 | 0.369 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 1 | 9781.19 | 4209.33 | -5571.86 | 0.430 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | shared | 1 | 9991.42 | 4126.24 | -5865.18 | 0.413 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 4 | 10115.49 | 4137.30 | -5978.19 | 0.409 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | shared | 4 | 9764.76 | 4125.58 | -5639.18 | 0.422 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 1 | 25185.98 | 10405.09 | -14780.89 | 0.413 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | shared | 1 | 25172.83 | 9827.25 | -15345.58 | 0.390 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 4 | 24709.01 | 10103.75 | -14605.26 | 0.409 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | shared | 4 | 24799.26 | 10045.09 | -14754.17 | 0.405 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1375.47 | 5274.90 | 3899.43 | 3.835 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | shared | 1 | 1340.89 | 5069.03 | 3728.14 | 3.780 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 4 | 1355.55 | 5267.63 | 3912.08 | 3.886 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | shared | 4 | 1399.49 | 5239.08 | 3839.59 | 3.744 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 1 | 6837.17 | 9443.88 | 2606.71 | 1.381 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | shared | 1 | 6803.06 | 9476.44 | 2673.38 | 1.393 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 4 | 6866.60 | 9685.94 | 2819.34 | 1.411 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | shared | 4 | 6760.02 | 9703.47 | 2943.45 | 1.435 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1278.54 | 4663.14 | 3384.60 | 3.647 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | shared | 1 | 1289.71 | 4895.97 | 3606.26 | 3.796 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 4 | 1330.28 | 4610.10 | 3279.82 | 3.466 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | shared | 4 | 1266.16 | 4663.75 | 3397.59 | 3.683 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 1 | 4130.76 | 7866.61 | 3735.85 | 1.904 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | shared | 1 | 4157.20 | 7814.12 | 3656.92 | 1.880 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 4 | 4211.77 | 7928.52 | 3716.75 | 1.882 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | shared | 4 | 4223.98 | 7893.07 | 3669.09 | 1.869 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 1 | 2798.55 | 7380.90 | 4582.35 | 2.637 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | shared | 1 | 2825.97 | 8106.00 | 5280.03 | 2.868 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 4 | 2769.83 | 7610.59 | 4840.76 | 2.748 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | shared | 4 | 2701.64 | 7452.36 | 4750.72 | 2.758 | observed; no speedup claim |

## Separate allocation diagnostics

Instrumented executable is built separately after latency collection. Calling-thread Rust allocator requests only; excludes foreign BLAS and worker-thread allocations. Realloc counts its requested new size, not net retained bytes. Output destruction and instrumentation setup are excluded. Helper references and multi-output retention use stack storage. No allocation-instrumented run contributes latency samples.

| Corresponding case ID | Threads | Median allocation requests/op | Median requested bytes/op | Status |
|---|---:|---:|---:|---|
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | 1 | 116 | 12951 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | 1 | 15 | 4120 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | 1 | 116 | 12951 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-shared | 1 | 15 | 4120 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | 1 | 265 | 51877 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | 1 | 48 | 20952 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | 1 | 265 | 51877 | passed |
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
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | 4 | 265 | 51877 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | 4 | 48 | 20952 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | 4 | 265 | 51877 | passed |
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
  "collection_total_seconds": 0.5399920580002799,
  "case_process_seconds": 0.49013505799939594
}
```

Wall observations `wall_allocations_t4.json`:
```json
{
  "collection_total_seconds": 0.5641951199995674,
  "case_process_seconds": 0.5138263360013298
}
```

Wall observations `wall_command.json`:
```json
{
  "command_seconds_before_final_report_write": 355.87908766,
  "report_generation_seconds": 0.135740722,
  "scope": "command startup through report generation; excludes final report file writes",
  "budget_seconds": 120.0,
  "budget_status": "over_budget; diagnostic only"
}
```

Wall observations `wall_measure_t1.json`:
```json
{
  "collection_total_seconds": 4.641152544999386,
  "case_process_seconds": 4.575543316003859
}
```

Wall observations `wall_measure_t4.json`:
```json
{
  "collection_total_seconds": 5.020427753999684,
  "case_process_seconds": 4.95474746799664
}
```

Wall observations `wall_phases.json`:
```json
{
  "validation_and_latency_build_seconds": 4.423849679,
  "metadata_and_latency_collection_seconds": 16.59825583,
  "allocation_diagnostic_build_seconds": 333.453880134,
  "allocation_collection_seconds": 1.267361295,
  "command_before_report_seconds": 355.743346938
}
```

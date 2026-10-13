# CPU Small-Work Benchmark Results

- Suite: `cpu/small_work`
- Raw run: `data/results/amd-cpu/cpu/small_work/20261013_101616`

## CPU Information

- Model: `Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `16`
- Sockets: `1`
- Cores per socket: `16`
- Threads per core: `1`
- NUMA nodes: `unknown`
- Python platform: `Linux-7.0.14-linuxkit-x86_64-with-glibc2.39`

## run_t1

Full metadata: `data/results/amd-cpu/cpu/small_work/20261013_101616/run_t1.yaml`

```yaml
timestamp: '2026-10-13T10:16:16Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
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
    BENCHMARK_COMMIT: b87c857a1e658102ec94436ec7c889ae8e8816ec
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
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.0 (b940084d7 2026-09-28)
    BENCH_INSTANCE: null
```

## run_t4

Full metadata: `data/results/amd-cpu/cpu/small_work/20261013_101616/run_t4.yaml`

```yaml
timestamp: '2026-10-13T10:17:22Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
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
    BENCHMARK_COMMIT: b87c857a1e658102ec94436ec7c889ae8e8816ec
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
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.0 (b940084d7 2026-09-28)
    BENCH_INSTANCE: null
```

## Case status

- Manifest: `cpu/small_work` version 3, coverage `full`, effort `standard`
- Expected 584, selected 584, executed 584, unsupported 0, failed 0, missing 0, noisy 15.
- Complete: **yes** — only executed cases count as covered; unsupported, failed, missing and unselected cases never do.
- Noisy: `einsum_f64_borrowed-fresh_n4_strided_single@t1`, `einsum_f64_borrowed-shared_n16_row_major_contiguous_single@t4`, `einsum_c64_eager-ad_n4_single@t4`, `einsum_f64_compiled-setup_n2_single@t4`, `nonad_qr_n4_col_major_contiguous_eager-shared@t4`, `nonad_solve_n16_col_major_contiguous_eager-no-ad@t4`, `nonad_einsum_n64_col_major_contiguous_concrete-fresh@t4`, `nonad_einsum_n64_col_major_contiguous_eager-no-ad@t4`, `nonad_einsum_n64_col_major_contiguous_concrete-shared@t4`, `nonad_einsum_n64_col_major_contiguous_eager-shared@t4`, `nonad_einsum3_n64_col_major_contiguous_eager-no-ad@t4`, `nonad_einsum3_n64_col_major_contiguous_concrete-shared@t4`, `nonad_qr_n64_col_major_contiguous_concrete-shared@t4`, `nonad_solve_n64_col_major_contiguous_concrete-fresh@t4`, `nonad_solve_n64_col_major_contiguous_eager-shared@t4`

Numerical checks: **584/584 recorded rows passed** (case × thread count).

## Timing

Sequential release runs; explicit initialization and warmups, then calibration toward the declared batch duration (2 ms by default), then the measured batches of the recorded effort (standard: 3 warmups and 15 batches; scan: a low-repetition screen that only marks suspects). Legacy single-call routes start calibration at 1024 workflows; matched non-AD and dependent-chain routes start at 16 and bound retained memory. Each batch uses one wall-clock interval; all outputs are retained until it stops. Statistics divide batch time by iterations. Median and inclusive IQR are in nanoseconds (ns); CoV is sample standard deviation / mean. NOISY means CoV > 10%, a descriptive label only. No noisy rows are excluded.

Workflow time is the total for one call or the complete dependent chain. The separate ns/op column divides the chain median by its operation count, not an independently measured call. Setup rows measure preparation only, not execution.

Inputs and backend construction, independent numerical checks and AD backward checks are outside timing. Returned outputs/plans and every dependent-chain intermediate are retained until after each batch interval; retention containers are prepared before the clock. value downloads/checks are outside. Eager AD rows measure forward execution/recording, not backward.

| Case ID | Operation | API route | Route (path / output / repr / session) | Dtype | Shape | Layout | Workflow | Threads | Provider | Operations/batch | Median batch ms | Median workflow ns | IQR ns | CoV | ns/op (chain only) | Status |
|---|---|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 2048 | 2.110753 | 1030.64 | 58.01 | 3.9% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 1 | blas | 2560 | 2.598992 | 10152.31 | 153.43 | 1.8% | 1015.23 | ok |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 2048 | 2.196819 | 1072.67 | 13.63 | 3.4% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 1 | blas | 2560 | 2.564762 | 10018.60 | 146.12 | 1.6% | 1001.86 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 1024 | 1.183916 | 1156.17 | 70.09 | 4.8% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 1 | blas | 2560 | 2.808414 | 10970.37 | 363.50 | 3.5% | 1097.04 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 11.892576 | 11613.84 | 157.36 | 1.5% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 12.205884 | 11919.81 | 350.13 | 2.9% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 13.005397 | 12700.58 | 380.45 | 3.2% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 14.515958 | 14175.74 | 218.87 | 1.4% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 14.702574 | 14357.98 | 219.38 | 1.3% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 15.901295 | 15528.61 | 229.49 | 1.2% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.227853 | 1087.82 | 9.56 | 1.2% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 2048 | 2.491260 | 1216.44 | 42.04 | 2.7% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 4.079761 | 3984.14 | 139.47 | 2.4% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 12.284488 | 11996.57 | 332.34 | 1.9% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 13.805380 | 13481.82 | 350.41 | 2.0% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 14.613141 | 14270.65 | 147.64 | 1.0% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 21.199567 | 20702.70 | 329.86 | 1.4% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 2048 | 2.423075 | 1183.14 | 46.14 | 3.2% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 2048 | 2.465619 | 1203.92 | 15.16 | 2.7% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 1024 | 1.508093 | 1472.75 | 79.87 | 3.5% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 2048 | 1.944591 | 949.51 | 30.65 | 2.8% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 2048 | 2.070590 | 1011.03 | 33.27 | 3.8% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 2048 | 1.967844 | 960.86 | 19.78 | 2.4% | — | ok |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 19.137390 | 18688.86 | 362.14 | 1.3% | — | ok |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 14.344046 | 14007.86 | 351.11 | 1.7% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 13.546917 | 13229.41 | 305.51 | 1.4% | — | ok |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 1024 | 12.402350 | 12111.67 | 255.37 | 1.8% | — | ok |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 1024 | 9.267021 | 9049.83 | 244.27 | 2.1% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 1024 | 8.460140 | 8261.86 | 313.37 | 2.2% | — | ok |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 1024 | 13.764219 | 13441.62 | 383.38 | 2.1% | — | ok |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 1024 | 10.277262 | 10036.39 | 357.97 | 2.4% | — | ok |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 1024 | 9.779209 | 9550.01 | 267.52 | 2.1% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 128 | 3.144564 | 24566.91 | 1352.84 | 3.4% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 3.775243 | 14747.04 | 801.80 | 6.6% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 128 | 2.803163 | 21899.71 | 1249.95 | 3.9% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 3.207050 | 12527.54 | 724.53 | 5.2% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 64 | 1.892064 | 29563.50 | 2550.59 | 8.8% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 64 | 2.226136 | 34783.38 | 2524.01 | 5.3% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 128 | 3.444248 | 26908.19 | 938.66 | 4.7% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 64 | 2.006601 | 31353.14 | 1011.39 | 4.0% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.309956 | 2255.82 | 43.13 | 2.6% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 2.292114 | 8953.57 | 471.95 | 4.5% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.699535 | 1318.13 | 42.06 | 4.6% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 3.265119 | 6377.19 | 230.52 | 3.8% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.163230 | 4225.06 | 213.33 | 5.1% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 3.307299 | 12919.14 | 767.27 | 4.2% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.858623 | 2791.62 | 96.02 | 2.5% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 2.461847 | 9616.59 | 369.25 | 2.9% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.110445 | 2060.98 | 41.83 | 4.2% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 3.727231 | 7279.75 | 286.66 | 3.4% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.186715 | 1067.73 | 63.93 | 3.5% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.908328 | 5680.33 | 218.97 | 2.4% | — | ok |
| nonad_einsum_n4_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 256 | 3.218931 | 12573.95 | 647.34 | 3.5% | — | ok |
| nonad_einsum_n4_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 128 | 2.911358 | 22744.98 | 700.82 | 2.7% | — | ok |
| nonad_einsum_n4_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 256 | 3.789858 | 14804.13 | 515.57 | 3.7% | — | ok |
| nonad_einsum_n4_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 128 | 3.173298 | 24791.39 | 984.31 | 2.9% | — | ok |
| nonad_einsum3_n4_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 64 | 2.079786 | 32496.66 | 1207.55 | 2.8% | — | ok |
| nonad_einsum3_n4_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 128 | 3.567852 | 27873.84 | 888.05 | 2.9% | — | ok |
| nonad_einsum3_n4_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 64 | 2.285746 | 35714.78 | 1737.48 | 4.6% | — | ok |
| nonad_einsum3_n4_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 128 | 3.840045 | 30000.35 | 1570.38 | 4.3% | — | ok |
| nonad_qr_n4_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 256 | 1.741434 | 6802.48 | 298.29 | 4.3% | — | ok |
| nonad_qr_n4_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 2048 | 3.171315 | 1548.49 | 55.92 | 4.8% | — | ok |
| nonad_qr_n4_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 256 | 2.325683 | 9084.70 | 415.47 | 3.8% | — | ok |
| nonad_qr_n4_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 2.690088 | 2627.04 | 126.87 | 3.2% | — | ok |
| nonad_svd_n4_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 128 | 1.785040 | 13945.62 | 252.14 | 5.0% | — | ok |
| nonad_svd_n4_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 512 | 2.987297 | 5834.56 | 262.05 | 3.2% | — | ok |
| nonad_svd_n4_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 128 | 2.240201 | 17501.57 | 828.11 | 3.3% | — | ok |
| nonad_svd_n4_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 256 | 1.876217 | 7328.97 | 214.92 | 3.1% | — | ok |
| nonad_solve_n4_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 512 | 2.946137 | 5754.17 | 253.22 | 3.8% | — | ok |
| nonad_solve_n4_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 2048 | 2.415981 | 1179.68 | 32.16 | 2.6% | — | ok |
| nonad_solve_n4_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 256 | 1.869959 | 7304.53 | 253.35 | 3.1% | — | ok |
| nonad_solve_n4_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 2.089482 | 2040.51 | 67.24 | 2.5% | — | ok |
| nonad_einsum_n8_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 128 | 3.144278 | 24564.67 | 590.01 | 3.8% | — | ok |
| nonad_einsum_n8_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 256 | 3.821134 | 14926.30 | 690.07 | 3.1% | — | ok |
| nonad_einsum_n8_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 128 | 2.854221 | 22298.60 | 646.63 | 3.7% | — | ok |
| nonad_einsum_n8_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 256 | 3.112583 | 12158.53 | 584.56 | 5.1% | — | ok |
| nonad_einsum3_n8_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 64 | 1.984631 | 31009.86 | 456.49 | 3.4% | — | ok |
| nonad_einsum3_n8_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 64 | 2.304059 | 36000.92 | 1777.95 | 5.2% | — | ok |
| nonad_einsum3_n8_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 128 | 3.444190 | 26907.73 | 865.45 | 2.9% | — | ok |
| nonad_einsum3_n8_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 64 | 2.048437 | 32006.83 | 481.35 | 3.4% | — | ok |
| nonad_qr_n8_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 3.343188 | 3264.83 | 112.60 | 2.4% | — | ok |
| nonad_qr_n8_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 256 | 2.603776 | 10171.00 | 522.27 | 3.3% | — | ok |
| nonad_qr_n8_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 2.361516 | 2306.17 | 50.84 | 3.6% | — | ok |
| nonad_qr_n8_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 256 | 1.936501 | 7564.46 | 302.79 | 3.7% | — | ok |
| nonad_svd_n8_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 128 | 1.983732 | 15497.91 | 853.08 | 6.4% | — | ok |
| nonad_svd_n8_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 128 | 3.199947 | 24999.59 | 778.98 | 4.3% | — | ok |
| nonad_svd_n8_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 256 | 3.656418 | 14282.88 | 651.71 | 4.1% | — | ok |
| nonad_svd_n8_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 128 | 2.829359 | 22104.37 | 588.09 | 3.2% | — | ok |
| nonad_solve_n8_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 2.845217 | 2778.53 | 39.65 | 2.4% | — | ok |
| nonad_solve_n8_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 256 | 2.081291 | 8130.04 | 360.43 | 5.2% | — | ok |
| nonad_solve_n8_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 1.906197 | 1861.52 | 28.62 | 3.4% | — | ok |
| nonad_solve_n8_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 512 | 3.272226 | 6391.07 | 312.22 | 2.9% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 3.244740 | 12674.77 | 771.81 | 5.6% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 128 | 2.896763 | 22630.96 | 913.73 | 3.5% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 128 | 1.968042 | 15375.33 | 796.74 | 6.7% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 128 | 3.194451 | 24956.65 | 958.75 | 3.2% | — | ok |
| nonad_einsum_n16_row_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 256 | 3.462625 | 13525.88 | 69.32 | 2.7% | — | ok |
| nonad_einsum_n16_row_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 128 | 2.977760 | 23263.75 | 562.68 | 4.5% | — | ok |
| nonad_einsum_n16_row_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 128 | 2.004980 | 15663.91 | 595.44 | 3.1% | — | ok |
| nonad_einsum_n16_row_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 128 | 3.224816 | 25193.88 | 1099.91 | 4.2% | — | ok |
| nonad_einsum3_n16_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 64 | 2.073924 | 32405.06 | 1566.34 | 3.8% | — | ok |
| nonad_einsum3_n16_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 64 | 1.819413 | 28428.33 | 682.98 | 2.1% | — | ok |
| nonad_einsum3_n16_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 64 | 2.281376 | 35646.50 | 2435.47 | 7.7% | — | ok |
| nonad_einsum3_n16_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 128 | 3.809969 | 29765.38 | 554.77 | 1.4% | — | ok |
| nonad_einsum3_n16_row_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 64 | 2.099994 | 32812.41 | 1638.20 | 5.7% | — | ok |
| nonad_einsum3_n16_row_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 128 | 3.614164 | 28235.66 | 930.56 | 2.9% | — | ok |
| nonad_einsum3_n16_row_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 64 | 2.395338 | 37427.16 | 995.45 | 4.1% | — | ok |
| nonad_einsum3_n16_row_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 64 | 1.950142 | 30470.97 | 1358.22 | 5.2% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 2.626021 | 10257.89 | 466.94 | 3.6% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.819204 | 5506.26 | 81.66 | 2.3% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 3.168303 | 12376.18 | 491.83 | 3.0% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 3.488607 | 6813.69 | 207.39 | 2.0% | — | ok |
| nonad_qr_n16_row_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 256 | 2.809228 | 10973.55 | 613.22 | 3.2% | — | ok |
| nonad_qr_n16_row_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 512 | 3.750164 | 7324.54 | 403.47 | 3.9% | — | ok |
| nonad_qr_n16_row_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 256 | 3.407260 | 13309.61 | 470.54 | 4.7% | — | ok |
| nonad_qr_n16_row_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 256 | 1.880785 | 7346.82 | 247.72 | 5.3% | — | ok |
| nonad_svd_n16_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 64 | 2.773673 | 43338.64 | 2247.07 | 4.9% | — | ok |
| nonad_svd_n16_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 64 | 2.198851 | 34357.05 | 1241.84 | 4.0% | — | ok |
| nonad_svd_n16_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 64 | 2.987634 | 46681.78 | 2998.71 | 4.7% | — | ok |
| nonad_svd_n16_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 64 | 2.312953 | 36139.89 | 1463.22 | 3.1% | — | ok |
| nonad_svd_n16_row_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 64 | 2.778066 | 43407.28 | 1219.64 | 3.9% | — | ok |
| nonad_svd_n16_row_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 64 | 2.352575 | 36758.98 | 1377.91 | 3.5% | — | ok |
| nonad_svd_n16_row_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 64 | 3.102977 | 48484.02 | 3010.52 | 3.9% | — | ok |
| nonad_svd_n16_row_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 64 | 2.544108 | 39751.69 | 2613.77 | 4.8% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 2.136200 | 8344.53 | 479.34 | 4.0% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 4.005691 | 3911.81 | 118.94 | 1.7% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 2.692856 | 10518.97 | 406.01 | 3.4% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.269296 | 4432.22 | 120.30 | 3.8% | — | ok |
| nonad_solve_n16_row_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 256 | 2.310468 | 9025.27 | 433.77 | 3.7% | — | ok |
| nonad_solve_n16_row_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 512 | 1.637383 | 3198.01 | 61.66 | 2.6% | — | ok |
| nonad_solve_n16_row_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 256 | 2.845125 | 11113.77 | 545.56 | 5.7% | — | ok |
| nonad_solve_n16_row_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 512 | 2.180421 | 4258.63 | 128.32 | 3.1% | — | ok |
| nonad_einsum_n64_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 64 | 2.256241 | 35253.77 | 2156.91 | 3.5% | — | ok |
| nonad_einsum_n64_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 128 | 3.172935 | 24788.55 | 1671.66 | 4.3% | — | ok |
| nonad_einsum_n64_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 64 | 2.043537 | 31930.27 | 276.12 | 1.6% | — | ok |
| nonad_einsum_n64_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 128 | 2.918938 | 22804.20 | 929.07 | 4.9% | — | ok |
| nonad_einsum3_n64_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 64 | 3.331562 | 52055.66 | 2508.49 | 3.2% | — | ok |
| nonad_einsum3_n64_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 32 | 1.872258 | 58508.06 | 3633.88 | 4.0% | — | ok |
| nonad_einsum3_n64_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 64 | 3.164354 | 49443.03 | 3411.70 | 4.8% | — | ok |
| nonad_einsum3_n64_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 32 | 1.811132 | 56597.88 | 1686.59 | 2.1% | — | ok |
| nonad_qr_n64_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 32 | 1.762111 | 55065.97 | 923.19 | 7.8% | — | ok |
| nonad_qr_n64_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 16 | 1.112479 | 69529.94 | 5740.88 | 6.1% | — | ok |
| nonad_qr_n64_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 32 | 1.784434 | 55763.56 | 659.84 | 7.2% | — | ok |
| nonad_qr_n64_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 32 | 2.137559 | 66798.72 | 5261.91 | 7.1% | — | ok |
| nonad_svd_n64_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 16 | 6.701487 | 418842.94 | 11829.19 | 2.2% | — | ok |
| nonad_svd_n64_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 16 | 6.941692 | 433855.75 | 13996.31 | 1.9% | — | ok |
| nonad_svd_n64_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 16 | 6.676652 | 417290.75 | 18975.31 | 5.2% | — | ok |
| nonad_svd_n64_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 16 | 6.857167 | 428572.94 | 12469.03 | 1.7% | — | ok |
| nonad_solve_n64_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 64 | 2.854207 | 44596.98 | 1731.76 | 5.9% | — | ok |
| nonad_solve_n64_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 64 | 2.623232 | 40988.00 | 2067.16 | 4.9% | — | ok |
| nonad_solve_n64_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 64 | 2.777643 | 43400.67 | 2265.24 | 4.9% | — | ok |
| nonad_solve_n64_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 64 | 2.498101 | 39032.83 | 2230.28 | 5.1% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 128 | 2.938618 | 22957.95 | 655.93 | 3.0% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 3.433357 | 1676.44 | 60.68 | 5.0% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 3.128181 | 1527.43 | 57.89 | 3.3% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 128 | 3.077856 | 24045.75 | 1343.33 | 3.4% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 3.110816 | 6075.81 | 132.04 | 3.3% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 1.909220 | 3728.95 | 121.82 | 2.7% | — | ok |
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 1024 | 88.560315 | 86484.68 | 2540.33 | 2.0% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 4 | blas | 160 | 13.843915 | 865244.69 | 25107.22 | 4.1% | 86524.47 | ok |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 1024 | 88.588230 | 86511.94 | 1609.51 | 1.6% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 4 | blas | 160 | 13.764118 | 860257.38 | 18292.31 | 3.7% | 86025.74 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 89.535178 | 87436.70 | 2523.41 | 2.7% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 4 | blas | 160 | 14.287666 | 892979.12 | 38541.09 | 3.3% | 89297.91 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 12.232514 | 11945.81 | 230.11 | 1.4% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 12.608071 | 12312.57 | 263.19 | 1.9% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 12.662545 | 12365.77 | 360.64 | 1.8% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 14.622796 | 14280.07 | 355.34 | 2.1% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 14.639340 | 14296.23 | 246.31 | 1.4% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 15.971948 | 15597.61 | 396.16 | 1.6% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 2.395600 | 1169.73 | 63.13 | 4.1% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 2048 | 2.755975 | 1345.69 | 99.08 | 5.7% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 4.281988 | 4181.63 | 214.35 | 4.7% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 12.258606 | 11971.29 | 306.03 | 1.9% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 19.020864 | 18575.06 | 615.35 | 5.4% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 14.916817 | 14567.20 | 274.88 | 1.9% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 21.682249 | 21174.07 | 1128.66 | 4.0% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 1024 | 86.176347 | 84156.59 | 1074.64 | 1.9% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 1024 | 84.895099 | 82905.37 | 1380.39 | 1.3% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 86.549517 | 84521.01 | 2573.64 | 5.1% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 1024 | 85.978885 | 83963.75 | 1710.28 | 2.3% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 1024 | 86.349485 | 84325.67 | 2396.56 | 2.0% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 86.827659 | 84792.64 | 1719.40 | 1.6% | — | ok |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 19.450487 | 18994.62 | 450.13 | 1.4% | — | ok |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 14.745535 | 14399.94 | 495.96 | 2.9% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 14.497726 | 14157.94 | 270.93 | 1.3% | — | ok |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 1024 | 13.165643 | 12857.07 | 212.21 | 1.2% | — | ok |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 1024 | 8.849597 | 8642.18 | 401.25 | 3.1% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 1024 | 8.953526 | 8743.68 | 355.73 | 2.4% | — | ok |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 1024 | 13.980629 | 13652.96 | 353.24 | 1.8% | — | ok |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 1024 | 9.996499 | 9762.21 | 345.02 | 2.5% | — | ok |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 1024 | 9.757468 | 9528.78 | 453.82 | 3.5% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 128 | 3.190842 | 24928.45 | 741.79 | 3.6% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 3.718903 | 14526.96 | 505.30 | 3.6% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 128 | 2.813464 | 21980.19 | 1069.72 | 4.7% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 3.278433 | 12806.38 | 646.20 | 4.6% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 128 | 3.817940 | 29827.66 | 965.48 | 3.1% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 16 | 2.692033 | 168252.06 | 9815.03 | 3.4% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 128 | 3.475843 | 27155.02 | 1279.38 | 3.0% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 16 | 2.622510 | 163906.88 | 7112.25 | 3.4% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.314739 | 2260.49 | 106.38 | 3.1% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 16 | 2.225088 | 139068.00 | 7087.22 | 4.2% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 2.763355 | 1349.29 | 85.24 | 5.9% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 16 | 2.197395 | 137337.19 | 10944.62 | 6.2% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.121553 | 4143.66 | 198.82 | 4.2% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 16 | 2.275852 | 142240.75 | 6253.69 | 3.2% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.687040 | 2624.06 | 154.61 | 5.6% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 16 | 2.206920 | 137932.50 | 5357.53 | 3.7% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.134573 | 2084.54 | 88.17 | 3.3% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 1.835618 | 7170.38 | 225.08 | 2.3% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 2.277813 | 1112.21 | 46.69 | 4.0% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.967232 | 5795.38 | 366.39 | 4.1% | — | ok |
| nonad_einsum_n4_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 256 | 3.310185 | 12930.41 | 374.49 | 3.3% | — | ok |
| nonad_einsum_n4_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 128 | 2.832633 | 22129.95 | 1092.45 | 7.3% | — | ok |
| nonad_einsum_n4_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 128 | 1.986398 | 15518.73 | 1434.38 | 6.9% | — | ok |
| nonad_einsum_n4_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 128 | 3.159546 | 24683.95 | 2196.52 | 9.2% | — | ok |
| nonad_einsum3_n4_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 16 | 2.658183 | 166136.44 | 8120.72 | 4.3% | — | ok |
| nonad_einsum3_n4_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 128 | 3.488316 | 27252.47 | 1039.81 | 3.2% | — | ok |
| nonad_einsum3_n4_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 16 | 2.747834 | 171739.62 | 16975.03 | 5.5% | — | ok |
| nonad_einsum3_n4_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 128 | 3.831270 | 29931.80 | 1303.27 | 4.8% | — | ok |
| nonad_qr_n4_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 16 | 2.228947 | 139309.19 | 16851.28 | 11.9% | — | NOISY |
| nonad_qr_n4_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 2048 | 3.272726 | 1598.01 | 71.09 | 4.9% | — | ok |
| nonad_qr_n4_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 16 | 2.181681 | 136355.06 | 7691.94 | 7.5% | — | ok |
| nonad_qr_n4_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 2.672273 | 2609.64 | 124.66 | 3.8% | — | ok |
| nonad_svd_n4_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 16 | 2.258648 | 141165.50 | 9138.94 | 5.2% | — | ok |
| nonad_svd_n4_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 512 | 3.076941 | 6009.65 | 362.75 | 4.6% | — | ok |
| nonad_svd_n4_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 16 | 2.402363 | 150147.69 | 4641.34 | 3.8% | — | ok |
| nonad_svd_n4_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 256 | 1.998355 | 7806.07 | 337.53 | 4.0% | — | ok |
| nonad_solve_n4_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 512 | 2.918719 | 5700.62 | 185.13 | 4.0% | — | ok |
| nonad_solve_n4_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 2048 | 2.461839 | 1202.07 | 61.64 | 4.5% | — | ok |
| nonad_solve_n4_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 256 | 1.906151 | 7445.90 | 419.71 | 5.0% | — | ok |
| nonad_solve_n4_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 2.246914 | 2194.25 | 91.85 | 3.4% | — | ok |
| nonad_einsum_n8_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 128 | 3.277779 | 25607.65 | 1179.09 | 4.7% | — | ok |
| nonad_einsum_n8_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 128 | 1.978376 | 15456.06 | 838.30 | 4.2% | — | ok |
| nonad_einsum_n8_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 128 | 2.915632 | 22778.38 | 906.11 | 6.0% | — | ok |
| nonad_einsum_n8_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 256 | 3.136382 | 12251.49 | 710.80 | 5.2% | — | ok |
| nonad_einsum3_n8_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 64 | 2.013383 | 31459.11 | 856.35 | 1.7% | — | ok |
| nonad_einsum3_n8_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 16 | 2.655063 | 165941.44 | 7224.12 | 3.2% | — | ok |
| nonad_einsum3_n8_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 128 | 3.548650 | 27723.83 | 1369.67 | 6.0% | — | ok |
| nonad_einsum3_n8_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 16 | 2.647284 | 165455.25 | 6084.25 | 2.6% | — | ok |
| nonad_qr_n8_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 3.454675 | 3373.71 | 167.99 | 3.1% | — | ok |
| nonad_qr_n8_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 16 | 2.214239 | 138389.94 | 4981.28 | 3.3% | — | ok |
| nonad_qr_n8_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 2.486145 | 2427.88 | 105.77 | 7.0% | — | ok |
| nonad_qr_n8_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 16 | 2.184837 | 136552.31 | 11628.47 | 5.2% | — | ok |
| nonad_svd_n8_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 128 | 2.012047 | 15719.12 | 1003.09 | 4.8% | — | ok |
| nonad_svd_n8_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 16 | 2.528677 | 158042.31 | 7682.72 | 3.9% | — | ok |
| nonad_svd_n8_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 128 | 1.858424 | 14518.94 | 617.93 | 4.2% | — | ok |
| nonad_svd_n8_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 16 | 2.484539 | 155283.69 | 8976.88 | 3.9% | — | ok |
| nonad_solve_n8_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 2.926833 | 2858.24 | 169.56 | 3.7% | — | ok |
| nonad_solve_n8_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 256 | 2.210455 | 8634.59 | 251.32 | 3.1% | — | ok |
| nonad_solve_n8_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 1.919021 | 1874.04 | 134.29 | 4.2% | — | ok |
| nonad_solve_n8_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 512 | 3.298305 | 6442.00 | 262.83 | 3.2% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 3.200298 | 12501.16 | 447.18 | 2.6% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 128 | 2.998997 | 23429.66 | 673.99 | 4.2% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 128 | 2.071547 | 16183.96 | 830.09 | 4.3% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 128 | 3.226687 | 25208.49 | 798.68 | 4.8% | — | ok |
| nonad_einsum_n16_row_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 256 | 3.435989 | 13421.83 | 620.71 | 2.9% | — | ok |
| nonad_einsum_n16_row_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 128 | 2.981913 | 23296.20 | 850.89 | 5.5% | — | ok |
| nonad_einsum_n16_row_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 128 | 1.985965 | 15515.35 | 702.21 | 5.9% | — | ok |
| nonad_einsum_n16_row_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 128 | 3.298588 | 25770.22 | 800.54 | 2.1% | — | ok |
| nonad_einsum3_n16_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 16 | 2.638904 | 164931.50 | 9635.12 | 5.6% | — | ok |
| nonad_einsum3_n16_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 64 | 1.830555 | 28602.42 | 1924.20 | 6.0% | — | ok |
| nonad_einsum3_n16_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 16 | 2.816527 | 176032.94 | 9224.91 | 3.1% | — | ok |
| nonad_einsum3_n16_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 64 | 2.164135 | 33814.61 | 2901.54 | 7.3% | — | ok |
| nonad_einsum3_n16_row_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 16 | 2.731014 | 170688.38 | 17231.97 | 7.9% | — | ok |
| nonad_einsum3_n16_row_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 128 | 3.757703 | 29357.05 | 2422.10 | 4.8% | — | ok |
| nonad_einsum3_n16_row_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 16 | 2.791393 | 174462.06 | 6378.88 | 2.8% | — | ok |
| nonad_einsum3_n16_row_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 64 | 2.308051 | 36063.30 | 4306.50 | 7.8% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 16 | 2.358391 | 147399.44 | 13966.41 | 7.1% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 3.206994 | 6263.66 | 80.12 | 1.3% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 16 | 2.303231 | 143951.94 | 8835.78 | 4.1% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 3.693002 | 7212.89 | 188.37 | 3.0% | — | ok |
| nonad_qr_n16_row_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 16 | 2.400325 | 150020.31 | 12280.91 | 5.5% | — | ok |
| nonad_qr_n16_row_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 16 | 2.087462 | 130466.38 | 11098.69 | 4.5% | — | ok |
| nonad_qr_n16_row_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 16 | 2.447679 | 152979.94 | 13340.75 | 7.2% | — | ok |
| nonad_qr_n16_row_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 16 | 2.125396 | 132837.25 | 7904.69 | 7.5% | — | ok |
| nonad_svd_n16_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 16 | 3.026738 | 189171.12 | 12730.94 | 5.0% | — | ok |
| nonad_svd_n16_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 64 | 2.495548 | 38992.94 | 4108.72 | 8.6% | — | ok |
| nonad_svd_n16_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 16 | 3.123160 | 195197.50 | 11841.41 | 4.4% | — | ok |
| nonad_svd_n16_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 64 | 2.532199 | 39565.61 | 1638.69 | 4.5% | — | ok |
| nonad_svd_n16_row_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 16 | 3.236425 | 202276.56 | 9476.25 | 3.6% | — | ok |
| nonad_svd_n16_row_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 16 | 3.037171 | 189823.19 | 6897.31 | 3.9% | — | ok |
| nonad_svd_n16_row_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 16 | 3.233318 | 202082.38 | 21053.75 | 5.8% | — | ok |
| nonad_svd_n16_row_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 16 | 2.893222 | 180826.38 | 8422.56 | 4.2% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 2.106284 | 8227.67 | 306.75 | 6.7% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 1.738540 | 3395.59 | 87.14 | 2.6% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 2.814426 | 10993.85 | 1052.11 | 12.2% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.415434 | 4717.64 | 163.34 | 8.1% | — | ok |
| nonad_solve_n16_row_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 256 | 2.442873 | 9542.47 | 468.73 | 6.6% | — | ok |
| nonad_solve_n16_row_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 512 | 1.832237 | 3578.59 | 360.80 | 9.6% | — | ok |
| nonad_solve_n16_row_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 256 | 2.955288 | 11544.09 | 1261.53 | 6.1% | — | ok |
| nonad_solve_n16_row_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 512 | 2.362823 | 4614.89 | 479.65 | 8.3% | — | ok |
| nonad_einsum_n64_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 64 | 2.424092 | 37876.44 | 9113.02 | 12.8% | — | NOISY |
| nonad_einsum_n64_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 128 | 3.592942 | 28069.86 | 3607.77 | 13.7% | — | NOISY |
| nonad_einsum_n64_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 32 | 1.049595 | 32799.84 | 7870.80 | 27.0% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_einsum_n64_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 64 | 1.293468 | 20210.44 | 1279.64 | 15.1% | — | NOISY |
| nonad_einsum3_n64_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 64 | 3.184217 | 49753.39 | 4220.16 | 6.9% | — | ok |
| nonad_einsum3_n64_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 16 | 3.929260 | 245578.75 | 25245.00 | 11.8% | — | NOISY |
| nonad_einsum3_n64_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 32 | 1.597813 | 49931.66 | 5501.12 | 11.2% | — | NOISY |
| nonad_einsum3_n64_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 16 | 4.422051 | 276378.19 | 23903.53 | 7.0% | — | ok |
| nonad_qr_n64_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 32 | 2.592892 | 81027.88 | 5467.23 | 8.2% | — | ok |
| nonad_qr_n64_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 16 | 4.568840 | 285552.50 | 32363.00 | 7.6% | — | ok |
| nonad_qr_n64_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 32 | 2.605026 | 81407.06 | 9561.73 | 10.3% | — | NOISY |
| nonad_qr_n64_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 16 | 4.681419 | 292588.69 | 29443.53 | 9.9% | — | ok |
| nonad_svd_n64_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 16 | 7.545335 | 471583.44 | 31957.72 | 5.1% | — | ok |
| nonad_svd_n64_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 16 | 10.900512 | 681282.00 | 17807.69 | 2.9% | — | ok |
| nonad_svd_n64_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 16 | 7.410454 | 463153.38 | 24642.19 | 3.8% | — | ok |
| nonad_svd_n64_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 16 | 11.075934 | 692245.88 | 51049.69 | 4.8% | — | ok |
| nonad_solve_n64_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 64 | 3.127215 | 48862.73 | 5694.34 | 11.0% | — | NOISY |
| nonad_solve_n64_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 64 | 3.001850 | 46903.91 | 4475.54 | 9.7% | — | ok |
| nonad_solve_n64_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 64 | 3.029915 | 47342.42 | 2037.84 | 4.6% | — | ok |
| nonad_solve_n64_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 32 | 1.376764 | 43023.88 | 5875.39 | 11.1% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 128 | 3.001213 | 23446.98 | 462.13 | 4.6% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.867472 | 1823.70 | 69.21 | 2.9% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 3.085952 | 1506.81 | 41.48 | 3.7% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 128 | 3.115455 | 24339.49 | 1395.92 | 4.7% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 3.413256 | 6666.52 | 488.92 | 4.6% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.022032 | 3949.28 | 281.73 | 5.5% | — | ok |

## Matched non-AD eager comparisons (#98)

All requested QR/SVD outputs are retained. Ordinary einsum uses the same parsed integer notation and default Auto strategy; planning/cache lookup stays inside both public calls. These are public-call diagnostics, not prepared execution or an isolated AD cost. Fresh pairs include session entry; shared pairs exclude it. Views are constructed untimed and consumed as views on both routes. No output to_tensor conversion is timed. Capture is inactive by construction in each fresh process, with no capture_trace guard; inputs and validation outputs are checked untracked.

| Operation | Shape | Layout | Session | Threads | Direct ns | Eager ns | Eager − direct ns | Eager/direct | Quality |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 1 | 24566.91 | 14747.04 | -9819.87 | 0.600 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | shared | 1 | 21899.71 | 12527.54 | -9372.17 | 0.572 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 4 | 24928.45 | 14526.96 | -10401.49 | 0.583 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | shared | 4 | 21980.19 | 12806.38 | -9173.81 | 0.583 | observed; no speedup claim |
| einsum | 4×4 | col_major_contiguous | per call diagnostic | 1 | 24791.39 | 14804.13 | -9987.26 | 0.597 | observed; no speedup claim |
| einsum | 4×4 | col_major_contiguous | shared | 1 | 22744.98 | 12573.95 | -10171.03 | 0.553 | observed; no speedup claim |
| einsum | 4×4 | col_major_contiguous | per call diagnostic | 4 | 24683.95 | 15518.73 | -9165.22 | 0.629 | observed; no speedup claim |
| einsum | 4×4 | col_major_contiguous | shared | 4 | 22129.95 | 12930.41 | -9199.54 | 0.584 | observed; no speedup claim |
| einsum | 8×8 | col_major_contiguous | per call diagnostic | 1 | 24564.67 | 14926.30 | -9638.37 | 0.608 | observed; no speedup claim |
| einsum | 8×8 | col_major_contiguous | shared | 1 | 22298.60 | 12158.53 | -10140.07 | 0.545 | observed; no speedup claim |
| einsum | 8×8 | col_major_contiguous | per call diagnostic | 4 | 25607.65 | 15456.06 | -10151.59 | 0.604 | observed; no speedup claim |
| einsum | 8×8 | col_major_contiguous | shared | 4 | 22778.38 | 12251.49 | -10526.89 | 0.538 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 1 | 24956.65 | 15375.33 | -9581.32 | 0.616 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | shared | 1 | 22630.96 | 12674.77 | -9956.19 | 0.560 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 4 | 25208.49 | 16183.96 | -9024.53 | 0.642 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | shared | 4 | 23429.66 | 12501.16 | -10928.50 | 0.534 | observed; no speedup claim |
| einsum | 16×16 | row_major_contiguous | per call diagnostic | 1 | 25193.88 | 15663.91 | -9529.97 | 0.622 | observed; no speedup claim |
| einsum | 16×16 | row_major_contiguous | shared | 1 | 23263.75 | 13525.88 | -9737.87 | 0.581 | observed; no speedup claim |
| einsum | 16×16 | row_major_contiguous | per call diagnostic | 4 | 25770.22 | 15515.35 | -10254.87 | 0.602 | observed; no speedup claim |
| einsum | 16×16 | row_major_contiguous | shared | 4 | 23296.20 | 13421.83 | -9874.37 | 0.576 | observed; no speedup claim |
| einsum | 64×64 | col_major_contiguous | per call diagnostic | 1 | 35253.77 | 24788.55 | -10465.22 | 0.703 | observed; no speedup claim |
| einsum | 64×64 | col_major_contiguous | shared | 1 | 31930.27 | 22804.20 | -9126.07 | 0.714 | observed; no speedup claim |
| einsum | 64×64 | col_major_contiguous | per call diagnostic | 4 | 37876.44 | 28069.86 | -9806.58 | 0.741 | inconclusive |
| einsum | 64×64 | col_major_contiguous | shared | 4 | 32799.84 | 20210.44 | -12589.40 | 0.616 | inconclusive |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 1 | 29563.50 | 34783.38 | 5219.88 | 1.177 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | shared | 1 | 26908.19 | 31353.14 | 4444.95 | 1.165 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 4 | 29827.66 | 168252.06 | 138424.40 | 5.641 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | shared | 4 | 27155.02 | 163906.88 | 136751.86 | 6.036 | observed; no speedup claim |
| einsum3 | 4×4 | col_major_contiguous | per call diagnostic | 1 | 30000.35 | 35714.78 | 5714.43 | 1.190 | observed; no speedup claim |
| einsum3 | 4×4 | col_major_contiguous | shared | 1 | 27873.84 | 32496.66 | 4622.82 | 1.166 | observed; no speedup claim |
| einsum3 | 4×4 | col_major_contiguous | per call diagnostic | 4 | 29931.80 | 171739.62 | 141807.82 | 5.738 | observed; no speedup claim |
| einsum3 | 4×4 | col_major_contiguous | shared | 4 | 27252.47 | 166136.44 | 138883.97 | 6.096 | observed; no speedup claim |
| einsum3 | 8×8 | col_major_contiguous | per call diagnostic | 1 | 31009.86 | 36000.92 | 4991.06 | 1.161 | observed; no speedup claim |
| einsum3 | 8×8 | col_major_contiguous | shared | 1 | 26907.73 | 32006.83 | 5099.10 | 1.190 | observed; no speedup claim |
| einsum3 | 8×8 | col_major_contiguous | per call diagnostic | 4 | 31459.11 | 165941.44 | 134482.33 | 5.275 | observed; no speedup claim |
| einsum3 | 8×8 | col_major_contiguous | shared | 4 | 27723.83 | 165455.25 | 137731.42 | 5.968 | observed; no speedup claim |
| einsum3 | 16×16 | col_major_contiguous | per call diagnostic | 1 | 29765.38 | 35646.50 | 5881.12 | 1.198 | observed; no speedup claim |
| einsum3 | 16×16 | col_major_contiguous | shared | 1 | 28428.33 | 32405.06 | 3976.73 | 1.140 | observed; no speedup claim |
| einsum3 | 16×16 | col_major_contiguous | per call diagnostic | 4 | 33814.61 | 176032.94 | 142218.33 | 5.206 | observed; no speedup claim |
| einsum3 | 16×16 | col_major_contiguous | shared | 4 | 28602.42 | 164931.50 | 136329.08 | 5.766 | observed; no speedup claim |
| einsum3 | 16×16 | row_major_contiguous | per call diagnostic | 1 | 30470.97 | 37427.16 | 6956.19 | 1.228 | observed; no speedup claim |
| einsum3 | 16×16 | row_major_contiguous | shared | 1 | 28235.66 | 32812.41 | 4576.75 | 1.162 | observed; no speedup claim |
| einsum3 | 16×16 | row_major_contiguous | per call diagnostic | 4 | 36063.30 | 174462.06 | 138398.76 | 4.838 | observed; no speedup claim |
| einsum3 | 16×16 | row_major_contiguous | shared | 4 | 29357.05 | 170688.38 | 141331.33 | 5.814 | observed; no speedup claim |
| einsum3 | 64×64 | col_major_contiguous | per call diagnostic | 1 | 52055.66 | 58508.06 | 6452.40 | 1.124 | observed; no speedup claim |
| einsum3 | 64×64 | col_major_contiguous | shared | 1 | 49443.03 | 56597.88 | 7154.85 | 1.145 | observed; no speedup claim |
| einsum3 | 64×64 | col_major_contiguous | per call diagnostic | 4 | 49753.39 | 245578.75 | 195825.36 | 4.936 | inconclusive |
| einsum3 | 64×64 | col_major_contiguous | shared | 4 | 49931.66 | 276378.19 | 226446.53 | 5.535 | inconclusive |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 1 | 2255.82 | 8953.57 | 6697.75 | 3.969 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | shared | 1 | 1318.13 | 6377.19 | 5059.06 | 4.838 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 4 | 2260.49 | 139068.00 | 136807.51 | 61.521 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | shared | 4 | 1349.29 | 137337.19 | 135987.90 | 101.785 | observed; no speedup claim |
| qr | 4×4 | col_major_contiguous | per call diagnostic | 1 | 2627.04 | 9084.70 | 6457.66 | 3.458 | observed; no speedup claim |
| qr | 4×4 | col_major_contiguous | shared | 1 | 1548.49 | 6802.48 | 5253.99 | 4.393 | observed; no speedup claim |
| qr | 4×4 | col_major_contiguous | per call diagnostic | 4 | 2609.64 | 136355.06 | 133745.42 | 52.251 | observed; no speedup claim |
| qr | 4×4 | col_major_contiguous | shared | 4 | 1598.01 | 139309.19 | 137711.18 | 87.177 | inconclusive |
| qr | 8×8 | col_major_contiguous | per call diagnostic | 1 | 3264.83 | 10171.00 | 6906.17 | 3.115 | observed; no speedup claim |
| qr | 8×8 | col_major_contiguous | shared | 1 | 2306.17 | 7564.46 | 5258.29 | 3.280 | observed; no speedup claim |
| qr | 8×8 | col_major_contiguous | per call diagnostic | 4 | 3373.71 | 138389.94 | 135016.23 | 41.020 | observed; no speedup claim |
| qr | 8×8 | col_major_contiguous | shared | 4 | 2427.88 | 136552.31 | 134124.43 | 56.243 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 1 | 6813.69 | 12376.18 | 5562.49 | 1.816 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | shared | 1 | 5506.26 | 10257.89 | 4751.63 | 1.863 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 4 | 7212.89 | 143951.94 | 136739.05 | 19.958 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | shared | 4 | 6263.66 | 147399.44 | 141135.78 | 23.532 | observed; no speedup claim |
| qr | 16×16 | row_major_contiguous | per call diagnostic | 1 | 7346.82 | 13309.61 | 5962.79 | 1.812 | observed; no speedup claim |
| qr | 16×16 | row_major_contiguous | shared | 1 | 7324.54 | 10973.55 | 3649.01 | 1.498 | observed; no speedup claim |
| qr | 16×16 | row_major_contiguous | per call diagnostic | 4 | 132837.25 | 152979.94 | 20142.69 | 1.152 | observed; no speedup claim |
| qr | 16×16 | row_major_contiguous | shared | 4 | 130466.38 | 150020.31 | 19553.93 | 1.150 | observed; no speedup claim |
| qr | 64×64 | col_major_contiguous | per call diagnostic | 1 | 55065.97 | 69529.94 | 14463.97 | 1.263 | observed; no speedup claim |
| qr | 64×64 | col_major_contiguous | shared | 1 | 55763.56 | 66798.72 | 11035.16 | 1.198 | observed; no speedup claim |
| qr | 64×64 | col_major_contiguous | per call diagnostic | 4 | 81027.88 | 285552.50 | 204524.62 | 3.524 | observed; no speedup claim |
| qr | 64×64 | col_major_contiguous | shared | 4 | 81407.06 | 292588.69 | 211181.63 | 3.594 | inconclusive |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 1 | 2060.98 | 7279.75 | 5218.77 | 3.532 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | shared | 1 | 1067.73 | 5680.33 | 4612.60 | 5.320 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 4 | 2084.54 | 7170.38 | 5085.84 | 3.440 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | shared | 4 | 1112.21 | 5795.38 | 4683.17 | 5.211 | observed; no speedup claim |
| solve | 4×4 | col_major_contiguous | per call diagnostic | 1 | 2040.51 | 7304.53 | 5264.02 | 3.580 | observed; no speedup claim |
| solve | 4×4 | col_major_contiguous | shared | 1 | 1179.68 | 5754.17 | 4574.49 | 4.878 | observed; no speedup claim |
| solve | 4×4 | col_major_contiguous | per call diagnostic | 4 | 2194.25 | 7445.90 | 5251.65 | 3.393 | observed; no speedup claim |
| solve | 4×4 | col_major_contiguous | shared | 4 | 1202.07 | 5700.62 | 4498.55 | 4.742 | observed; no speedup claim |
| solve | 8×8 | col_major_contiguous | per call diagnostic | 1 | 2778.53 | 8130.04 | 5351.51 | 2.926 | observed; no speedup claim |
| solve | 8×8 | col_major_contiguous | shared | 1 | 1861.52 | 6391.07 | 4529.55 | 3.433 | observed; no speedup claim |
| solve | 8×8 | col_major_contiguous | per call diagnostic | 4 | 2858.24 | 8634.59 | 5776.35 | 3.021 | observed; no speedup claim |
| solve | 8×8 | col_major_contiguous | shared | 4 | 1874.04 | 6442.00 | 4567.96 | 3.437 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 1 | 4432.22 | 10518.97 | 6086.75 | 2.373 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | shared | 1 | 3911.81 | 8344.53 | 4432.72 | 2.133 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 4 | 4717.64 | 10993.85 | 6276.21 | 2.330 | inconclusive |
| solve | 16×16 | col_major_contiguous | shared | 4 | 3395.59 | 8227.67 | 4832.08 | 2.423 | observed; no speedup claim |
| solve | 16×16 | row_major_contiguous | per call diagnostic | 1 | 4258.63 | 11113.77 | 6855.14 | 2.610 | observed; no speedup claim |
| solve | 16×16 | row_major_contiguous | shared | 1 | 3198.01 | 9025.27 | 5827.26 | 2.822 | observed; no speedup claim |
| solve | 16×16 | row_major_contiguous | per call diagnostic | 4 | 4614.89 | 11544.09 | 6929.20 | 2.501 | observed; no speedup claim |
| solve | 16×16 | row_major_contiguous | shared | 4 | 3578.59 | 9542.47 | 5963.88 | 2.667 | observed; no speedup claim |
| solve | 64×64 | col_major_contiguous | per call diagnostic | 1 | 44596.98 | 40988.00 | -3608.98 | 0.919 | observed; no speedup claim |
| solve | 64×64 | col_major_contiguous | shared | 1 | 43400.67 | 39032.83 | -4367.84 | 0.899 | observed; no speedup claim |
| solve | 64×64 | col_major_contiguous | per call diagnostic | 4 | 48862.73 | 46903.91 | -1958.82 | 0.960 | inconclusive |
| solve | 64×64 | col_major_contiguous | shared | 4 | 47342.42 | 43023.88 | -4318.54 | 0.909 | inconclusive |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 1 | 4225.06 | 12919.14 | 8694.08 | 3.058 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | shared | 1 | 2791.62 | 9616.59 | 6824.97 | 3.445 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 4 | 4143.66 | 142240.75 | 138097.09 | 34.327 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | shared | 4 | 2624.06 | 137932.50 | 135308.44 | 52.565 | observed; no speedup claim |
| svd | 4×4 | col_major_contiguous | per call diagnostic | 1 | 7328.97 | 17501.57 | 10172.60 | 2.388 | observed; no speedup claim |
| svd | 4×4 | col_major_contiguous | shared | 1 | 5834.56 | 13945.62 | 8111.06 | 2.390 | observed; no speedup claim |
| svd | 4×4 | col_major_contiguous | per call diagnostic | 4 | 7806.07 | 150147.69 | 142341.62 | 19.235 | observed; no speedup claim |
| svd | 4×4 | col_major_contiguous | shared | 4 | 6009.65 | 141165.50 | 135155.85 | 23.490 | observed; no speedup claim |
| svd | 8×8 | col_major_contiguous | per call diagnostic | 1 | 15497.91 | 24999.59 | 9501.68 | 1.613 | observed; no speedup claim |
| svd | 8×8 | col_major_contiguous | shared | 1 | 14282.88 | 22104.37 | 7821.49 | 1.548 | observed; no speedup claim |
| svd | 8×8 | col_major_contiguous | per call diagnostic | 4 | 15719.12 | 158042.31 | 142323.19 | 10.054 | observed; no speedup claim |
| svd | 8×8 | col_major_contiguous | shared | 4 | 14518.94 | 155283.69 | 140764.75 | 10.695 | observed; no speedup claim |
| svd | 16×16 | col_major_contiguous | per call diagnostic | 1 | 36139.89 | 46681.78 | 10541.89 | 1.292 | observed; no speedup claim |
| svd | 16×16 | col_major_contiguous | shared | 1 | 34357.05 | 43338.64 | 8981.59 | 1.261 | observed; no speedup claim |
| svd | 16×16 | col_major_contiguous | per call diagnostic | 4 | 39565.61 | 195197.50 | 155631.89 | 4.934 | observed; no speedup claim |
| svd | 16×16 | col_major_contiguous | shared | 4 | 38992.94 | 189171.12 | 150178.18 | 4.851 | observed; no speedup claim |
| svd | 16×16 | row_major_contiguous | per call diagnostic | 1 | 39751.69 | 48484.02 | 8732.33 | 1.220 | observed; no speedup claim |
| svd | 16×16 | row_major_contiguous | shared | 1 | 36758.98 | 43407.28 | 6648.30 | 1.181 | observed; no speedup claim |
| svd | 16×16 | row_major_contiguous | per call diagnostic | 4 | 180826.38 | 202082.38 | 21256.00 | 1.118 | observed; no speedup claim |
| svd | 16×16 | row_major_contiguous | shared | 4 | 189823.19 | 202276.56 | 12453.37 | 1.066 | observed; no speedup claim |
| svd | 64×64 | col_major_contiguous | per call diagnostic | 1 | 418842.94 | 433855.75 | 15012.81 | 1.036 | observed; no speedup claim |
| svd | 64×64 | col_major_contiguous | shared | 1 | 417290.75 | 428572.94 | 11282.19 | 1.027 | observed; no speedup claim |
| svd | 64×64 | col_major_contiguous | per call diagnostic | 4 | 471583.44 | 681282.00 | 209698.56 | 1.445 | observed; no speedup claim |
| svd | 64×64 | col_major_contiguous | shared | 4 | 463153.38 | 692245.88 | 229092.50 | 1.495 | observed; no speedup claim |

## Separate allocation diagnostics

Instrumented executable is built separately after latency collection. Calling-thread Rust allocator requests only; excludes foreign BLAS and worker-thread allocations. Realloc counts its requested new size, not net retained bytes. Output destruction and instrumentation setup are excluded. Helper references and multi-output retention use stack storage. No allocation-instrumented run contributes latency samples.

| Corresponding case ID | Threads | Median allocation requests/op | Median requested bytes/op | Status |
|---|---:|---:|---:|---|
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | 1 | 219 | 18140 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | 1 | 120 | 14653 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | 1 | 217 | 17876 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-shared | 1 | 118 | 14389 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | 1 | 312 | 26669 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | 1 | 274 | 30642 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | 1 | 310 | 26405 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | 1 | 272 | 30378 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | 1 | 8 | 1336 | passed |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | 1 | 47 | 12051 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-shared | 1 | 6 | 1072 | passed |
| nonad_qr_n2_col_major_contiguous_eager-shared | 1 | 45 | 11787 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | 1 | 11 | 2232 | passed |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | 1 | 55 | 14259 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-shared | 1 | 8 | 1776 | passed |
| nonad_svd_n2_col_major_contiguous_eager-shared | 1 | 52 | 13803 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | 1 | 6 | 192 | passed |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | 1 | 41 | 17795 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-shared | 1 | 5 | 120 | passed |
| nonad_solve_n2_col_major_contiguous_eager-shared | 1 | 40 | 17723 | passed |
| nonad_einsum_n4_col_major_contiguous_eager-shared | 1 | 118 | 14517 | passed |
| nonad_einsum_n4_col_major_contiguous_concrete-shared | 1 | 217 | 18004 | passed |
| nonad_einsum_n4_col_major_contiguous_eager-no-ad | 1 | 120 | 14781 | passed |
| nonad_einsum_n4_col_major_contiguous_concrete-fresh | 1 | 219 | 18268 | passed |
| nonad_einsum3_n4_col_major_contiguous_eager-shared | 1 | 272 | 30634 | passed |
| nonad_einsum3_n4_col_major_contiguous_concrete-shared | 1 | 310 | 26405 | passed |
| nonad_einsum3_n4_col_major_contiguous_eager-no-ad | 1 | 274 | 30898 | passed |
| nonad_einsum3_n4_col_major_contiguous_concrete-fresh | 1 | 312 | 26669 | passed |
| nonad_qr_n4_col_major_contiguous_eager-shared | 1 | 45 | 11979 | passed |
| nonad_qr_n4_col_major_contiguous_concrete-shared | 1 | 6 | 1264 | passed |
| nonad_qr_n4_col_major_contiguous_eager-no-ad | 1 | 47 | 12243 | passed |
| nonad_qr_n4_col_major_contiguous_concrete-fresh | 1 | 8 | 1528 | passed |
| nonad_svd_n4_col_major_contiguous_eager-shared | 1 | 52 | 14395 | passed |
| nonad_svd_n4_col_major_contiguous_concrete-shared | 1 | 8 | 2368 | passed |
| nonad_svd_n4_col_major_contiguous_eager-no-ad | 1 | 55 | 14851 | passed |
| nonad_svd_n4_col_major_contiguous_concrete-fresh | 1 | 11 | 2824 | passed |
| nonad_solve_n4_col_major_contiguous_eager-shared | 1 | 40 | 17923 | passed |
| nonad_solve_n4_col_major_contiguous_concrete-shared | 1 | 5 | 416 | passed |
| nonad_solve_n4_col_major_contiguous_eager-no-ad | 1 | 41 | 17995 | passed |
| nonad_solve_n4_col_major_contiguous_concrete-fresh | 1 | 6 | 488 | passed |
| nonad_einsum_n8_col_major_contiguous_concrete-fresh | 1 | 219 | 18524 | passed |
| nonad_einsum_n8_col_major_contiguous_eager-no-ad | 1 | 120 | 15037 | passed |
| nonad_einsum_n8_col_major_contiguous_concrete-shared | 1 | 217 | 18260 | passed |
| nonad_einsum_n8_col_major_contiguous_eager-shared | 1 | 118 | 14773 | passed |
| nonad_einsum3_n8_col_major_contiguous_concrete-fresh | 1 | 312 | 26669 | passed |
| nonad_einsum3_n8_col_major_contiguous_eager-no-ad | 1 | 273 | 31402 | passed |
| nonad_einsum3_n8_col_major_contiguous_concrete-shared | 1 | 310 | 26405 | passed |
| nonad_einsum3_n8_col_major_contiguous_eager-shared | 1 | 271 | 31138 | passed |
| nonad_qr_n8_col_major_contiguous_concrete-fresh | 1 | 8 | 2296 | passed |
| nonad_qr_n8_col_major_contiguous_eager-no-ad | 1 | 47 | 13011 | passed |
| nonad_qr_n8_col_major_contiguous_concrete-shared | 1 | 6 | 2032 | passed |
| nonad_qr_n8_col_major_contiguous_eager-shared | 1 | 45 | 12747 | passed |
| nonad_svd_n8_col_major_contiguous_concrete-fresh | 1 | 11 | 4968 | passed |
| nonad_svd_n8_col_major_contiguous_eager-no-ad | 1 | 55 | 16995 | passed |
| nonad_svd_n8_col_major_contiguous_concrete-shared | 1 | 8 | 4512 | passed |
| nonad_svd_n8_col_major_contiguous_eager-shared | 1 | 52 | 16539 | passed |
| nonad_solve_n8_col_major_contiguous_concrete-fresh | 1 | 6 | 1656 | passed |
| nonad_solve_n8_col_major_contiguous_eager-no-ad | 1 | 41 | 18779 | passed |
| nonad_solve_n8_col_major_contiguous_concrete-shared | 1 | 5 | 1584 | passed |
| nonad_solve_n8_col_major_contiguous_eager-shared | 1 | 40 | 18707 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-shared | 1 | 118 | 15285 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | 1 | 217 | 18772 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | 1 | 120 | 15549 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | 1 | 219 | 19036 | passed |
| nonad_einsum_n16_row_major_contiguous_eager-shared | 1 | 128 | 17605 | passed |
| nonad_einsum_n16_row_major_contiguous_concrete-shared | 1 | 217 | 18772 | passed |
| nonad_einsum_n16_row_major_contiguous_eager-no-ad | 1 | 130 | 17869 | passed |
| nonad_einsum_n16_row_major_contiguous_concrete-fresh | 1 | 219 | 19036 | passed |
| nonad_einsum3_n16_col_major_contiguous_eager-shared | 1 | 266 | 32086 | passed |
| nonad_einsum3_n16_col_major_contiguous_concrete-shared | 1 | 310 | 26405 | passed |
| nonad_einsum3_n16_col_major_contiguous_eager-no-ad | 1 | 268 | 32350 | passed |
| nonad_einsum3_n16_col_major_contiguous_concrete-fresh | 1 | 312 | 26669 | passed |
| nonad_einsum3_n16_row_major_contiguous_eager-shared | 1 | 281 | 35566 | passed |
| nonad_einsum3_n16_row_major_contiguous_concrete-shared | 1 | 309 | 24437 | passed |
| nonad_einsum3_n16_row_major_contiguous_eager-no-ad | 1 | 283 | 35830 | passed |
| nonad_einsum3_n16_row_major_contiguous_concrete-fresh | 1 | 311 | 24701 | passed |
| nonad_qr_n16_col_major_contiguous_eager-shared | 1 | 45 | 15819 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-shared | 1 | 6 | 5104 | passed |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | 1 | 47 | 16083 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | 1 | 8 | 5368 | passed |
| nonad_qr_n16_row_major_contiguous_eager-shared | 1 | 55 | 17075 | passed |
| nonad_qr_n16_row_major_contiguous_concrete-shared | 1 | 16 | 5360 | passed |
| nonad_qr_n16_row_major_contiguous_eager-no-ad | 1 | 57 | 17339 | passed |
| nonad_qr_n16_row_major_contiguous_concrete-fresh | 1 | 18 | 5624 | passed |
| nonad_svd_n16_col_major_contiguous_eager-shared | 1 | 52 | 24667 | passed |
| nonad_svd_n16_col_major_contiguous_concrete-shared | 1 | 8 | 12640 | passed |
| nonad_svd_n16_col_major_contiguous_eager-no-ad | 1 | 55 | 25123 | passed |
| nonad_svd_n16_col_major_contiguous_concrete-fresh | 1 | 11 | 13096 | passed |
| nonad_svd_n16_row_major_contiguous_eager-shared | 1 | 62 | 25923 | passed |
| nonad_svd_n16_row_major_contiguous_concrete-shared | 1 | 18 | 12896 | passed |
| nonad_svd_n16_row_major_contiguous_eager-no-ad | 1 | 65 | 26379 | passed |
| nonad_svd_n16_row_major_contiguous_concrete-fresh | 1 | 21 | 13352 | passed |
| nonad_solve_n16_col_major_contiguous_eager-shared | 1 | 40 | 21811 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-shared | 1 | 5 | 6224 | passed |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | 1 | 41 | 21883 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | 1 | 6 | 6296 | passed |
| nonad_solve_n16_row_major_contiguous_eager-shared | 1 | 50 | 24099 | passed |
| nonad_solve_n16_row_major_contiguous_concrete-shared | 1 | 4 | 4176 | passed |
| nonad_solve_n16_row_major_contiguous_eager-no-ad | 1 | 51 | 24171 | passed |
| nonad_solve_n16_row_major_contiguous_concrete-fresh | 1 | 5 | 4248 | passed |
| nonad_einsum_n64_col_major_contiguous_concrete-fresh | 1 | 219 | 22108 | passed |
| nonad_einsum_n64_col_major_contiguous_eager-no-ad | 1 | 120 | 18621 | passed |
| nonad_einsum_n64_col_major_contiguous_concrete-shared | 1 | 217 | 21844 | passed |
| nonad_einsum_n64_col_major_contiguous_eager-shared | 1 | 118 | 18357 | passed |
| nonad_einsum3_n64_col_major_contiguous_concrete-fresh | 1 | 334 | 27037 | passed |
| nonad_einsum3_n64_col_major_contiguous_eager-no-ad | 1 | 268 | 38494 | passed |
| nonad_einsum3_n64_col_major_contiguous_concrete-shared | 1 | 332 | 26773 | passed |
| nonad_einsum3_n64_col_major_contiguous_eager-shared | 1 | 266 | 38230 | passed |
| nonad_qr_n64_col_major_contiguous_concrete-fresh | 1 | 8 | 66808 | passed |
| nonad_qr_n64_col_major_contiguous_eager-no-ad | 1 | 47 | 77523 | passed |
| nonad_qr_n64_col_major_contiguous_concrete-shared | 1 | 6 | 66544 | passed |
| nonad_qr_n64_col_major_contiguous_eager-shared | 1 | 45 | 77259 | passed |
| nonad_svd_n64_col_major_contiguous_concrete-fresh | 1 | 11 | 169384 | passed |
| nonad_svd_n64_col_major_contiguous_eager-no-ad | 1 | 55 | 181411 | passed |
| nonad_svd_n64_col_major_contiguous_concrete-shared | 1 | 8 | 168928 | passed |
| nonad_svd_n64_col_major_contiguous_eager-shared | 1 | 52 | 180955 | passed |
| nonad_solve_n64_col_major_contiguous_concrete-fresh | 1 | 6 | 98648 | passed |
| nonad_solve_n64_col_major_contiguous_eager-no-ad | 1 | 41 | 83515 | passed |
| nonad_solve_n64_col_major_contiguous_concrete-shared | 1 | 5 | 98576 | passed |
| nonad_solve_n64_col_major_contiguous_eager-shared | 1 | 40 | 83443 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | 1 | 218 | 18068 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | 1 | 7 | 1264 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | 1 | 5 | 120 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | 1 | 218 | 18964 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | 1 | 7 | 5296 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | 1 | 5 | 6224 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | 4 | 219 | 18140 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | 4 | 120 | 14653 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | 4 | 217 | 17876 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-shared | 4 | 118 | 14389 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | 4 | 312 | 26669 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | 4 | 252 | 30202 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | 4 | 310 | 26405 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | 4 | 250 | 29938 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | 4 | 8 | 1336 | passed |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | 4 | 41 | 11715 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-shared | 4 | 6 | 1072 | passed |
| nonad_qr_n2_col_major_contiguous_eager-shared | 4 | 40 | 11643 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | 4 | 11 | 2232 | passed |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | 4 | 49 | 13923 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-shared | 4 | 8 | 1776 | passed |
| nonad_svd_n2_col_major_contiguous_eager-shared | 4 | 47 | 13659 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | 4 | 6 | 192 | passed |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | 4 | 41 | 17795 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-shared | 4 | 5 | 120 | passed |
| nonad_solve_n2_col_major_contiguous_eager-shared | 4 | 40 | 17723 | passed |
| nonad_einsum_n4_col_major_contiguous_eager-shared | 4 | 118 | 14517 | passed |
| nonad_einsum_n4_col_major_contiguous_concrete-shared | 4 | 217 | 18004 | passed |
| nonad_einsum_n4_col_major_contiguous_eager-no-ad | 4 | 120 | 14781 | passed |
| nonad_einsum_n4_col_major_contiguous_concrete-fresh | 4 | 219 | 18268 | passed |
| nonad_einsum3_n4_col_major_contiguous_eager-shared | 4 | 250 | 30194 | passed |
| nonad_einsum3_n4_col_major_contiguous_concrete-shared | 4 | 310 | 26405 | passed |
| nonad_einsum3_n4_col_major_contiguous_eager-no-ad | 4 | 252 | 30458 | passed |
| nonad_einsum3_n4_col_major_contiguous_concrete-fresh | 4 | 312 | 26669 | passed |
| nonad_qr_n4_col_major_contiguous_eager-shared | 4 | 40 | 11835 | passed |
| nonad_qr_n4_col_major_contiguous_concrete-shared | 4 | 6 | 1264 | passed |
| nonad_qr_n4_col_major_contiguous_eager-no-ad | 4 | 41 | 11907 | passed |
| nonad_qr_n4_col_major_contiguous_concrete-fresh | 4 | 8 | 1528 | passed |
| nonad_svd_n4_col_major_contiguous_eager-shared | 4 | 47 | 14251 | passed |
| nonad_svd_n4_col_major_contiguous_concrete-shared | 4 | 8 | 2368 | passed |
| nonad_svd_n4_col_major_contiguous_eager-no-ad | 4 | 49 | 14515 | passed |
| nonad_svd_n4_col_major_contiguous_concrete-fresh | 4 | 11 | 2824 | passed |
| nonad_solve_n4_col_major_contiguous_eager-shared | 4 | 40 | 17923 | passed |
| nonad_solve_n4_col_major_contiguous_concrete-shared | 4 | 5 | 416 | passed |
| nonad_solve_n4_col_major_contiguous_eager-no-ad | 4 | 41 | 17995 | passed |
| nonad_solve_n4_col_major_contiguous_concrete-fresh | 4 | 6 | 488 | passed |
| nonad_einsum_n8_col_major_contiguous_concrete-fresh | 4 | 219 | 18524 | passed |
| nonad_einsum_n8_col_major_contiguous_eager-no-ad | 4 | 120 | 15037 | passed |
| nonad_einsum_n8_col_major_contiguous_concrete-shared | 4 | 217 | 18260 | passed |
| nonad_einsum_n8_col_major_contiguous_eager-shared | 4 | 118 | 14773 | passed |
| nonad_einsum3_n8_col_major_contiguous_concrete-fresh | 4 | 312 | 26669 | passed |
| nonad_einsum3_n8_col_major_contiguous_eager-no-ad | 4 | 252 | 30970 | passed |
| nonad_einsum3_n8_col_major_contiguous_concrete-shared | 4 | 310 | 26405 | passed |
| nonad_einsum3_n8_col_major_contiguous_eager-shared | 4 | 250 | 30706 | passed |
| nonad_qr_n8_col_major_contiguous_concrete-fresh | 4 | 8 | 2296 | passed |
| nonad_qr_n8_col_major_contiguous_eager-no-ad | 4 | 41 | 12675 | passed |
| nonad_qr_n8_col_major_contiguous_concrete-shared | 4 | 6 | 2032 | passed |
| nonad_qr_n8_col_major_contiguous_eager-shared | 4 | 40 | 12603 | passed |
| nonad_svd_n8_col_major_contiguous_concrete-fresh | 4 | 11 | 4968 | passed |
| nonad_svd_n8_col_major_contiguous_eager-no-ad | 4 | 49 | 16659 | passed |
| nonad_svd_n8_col_major_contiguous_concrete-shared | 4 | 8 | 4512 | passed |
| nonad_svd_n8_col_major_contiguous_eager-shared | 4 | 47 | 16395 | passed |
| nonad_solve_n8_col_major_contiguous_concrete-fresh | 4 | 6 | 1656 | passed |
| nonad_solve_n8_col_major_contiguous_eager-no-ad | 4 | 41 | 18779 | passed |
| nonad_solve_n8_col_major_contiguous_concrete-shared | 4 | 5 | 1584 | passed |
| nonad_solve_n8_col_major_contiguous_eager-shared | 4 | 40 | 18707 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-shared | 4 | 118 | 15285 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | 4 | 217 | 18772 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | 4 | 120 | 15549 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | 4 | 219 | 19036 | passed |
| nonad_einsum_n16_row_major_contiguous_eager-shared | 4 | 128 | 17605 | passed |
| nonad_einsum_n16_row_major_contiguous_concrete-shared | 4 | 217 | 18772 | passed |
| nonad_einsum_n16_row_major_contiguous_eager-no-ad | 4 | 130 | 17869 | passed |
| nonad_einsum_n16_row_major_contiguous_concrete-fresh | 4 | 219 | 19036 | passed |
| nonad_einsum3_n16_col_major_contiguous_eager-shared | 4 | 250 | 31730 | passed |
| nonad_einsum3_n16_col_major_contiguous_concrete-shared | 4 | 310 | 26405 | passed |
| nonad_einsum3_n16_col_major_contiguous_eager-no-ad | 4 | 252 | 31994 | passed |
| nonad_einsum3_n16_col_major_contiguous_concrete-fresh | 4 | 312 | 26669 | passed |
| nonad_einsum3_n16_row_major_contiguous_eager-shared | 4 | 265 | 35210 | passed |
| nonad_einsum3_n16_row_major_contiguous_concrete-shared | 4 | 309 | 24437 | passed |
| nonad_einsum3_n16_row_major_contiguous_eager-no-ad | 4 | 267 | 35474 | passed |
| nonad_einsum3_n16_row_major_contiguous_concrete-fresh | 4 | 311 | 24701 | passed |
| nonad_qr_n16_col_major_contiguous_eager-shared | 4 | 40 | 15675 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-shared | 4 | 6 | 5104 | passed |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | 4 | 41 | 15747 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | 4 | 8 | 5368 | passed |
| nonad_qr_n16_row_major_contiguous_eager-shared | 4 | 45 | 16819 | passed |
| nonad_qr_n16_row_major_contiguous_concrete-shared | 4 | 6 | 5104 | passed |
| nonad_qr_n16_row_major_contiguous_eager-no-ad | 4 | 46 | 16891 | passed |
| nonad_qr_n16_row_major_contiguous_concrete-fresh | 4 | 7 | 5176 | passed |
| nonad_svd_n16_col_major_contiguous_eager-shared | 4 | 47 | 24523 | passed |
| nonad_svd_n16_col_major_contiguous_concrete-shared | 4 | 8 | 12640 | passed |
| nonad_svd_n16_col_major_contiguous_eager-no-ad | 4 | 49 | 24787 | passed |
| nonad_svd_n16_col_major_contiguous_concrete-fresh | 4 | 11 | 13096 | passed |
| nonad_svd_n16_row_major_contiguous_eager-shared | 4 | 52 | 25667 | passed |
| nonad_svd_n16_row_major_contiguous_concrete-shared | 4 | 8 | 12640 | passed |
| nonad_svd_n16_row_major_contiguous_eager-no-ad | 4 | 54 | 25931 | passed |
| nonad_svd_n16_row_major_contiguous_concrete-fresh | 4 | 10 | 12904 | passed |
| nonad_solve_n16_col_major_contiguous_eager-shared | 4 | 40 | 21811 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-shared | 4 | 5 | 6224 | passed |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | 4 | 41 | 21883 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | 4 | 6 | 6296 | passed |
| nonad_solve_n16_row_major_contiguous_eager-shared | 4 | 50 | 24099 | passed |
| nonad_solve_n16_row_major_contiguous_concrete-shared | 4 | 4 | 4176 | passed |
| nonad_solve_n16_row_major_contiguous_eager-no-ad | 4 | 51 | 24171 | passed |
| nonad_solve_n16_row_major_contiguous_concrete-fresh | 4 | 5 | 4248 | passed |
| nonad_einsum_n64_col_major_contiguous_concrete-fresh | 4 | 219 | 22108 | passed |
| nonad_einsum_n64_col_major_contiguous_eager-no-ad | 4 | 120 | 18621 | passed |
| nonad_einsum_n64_col_major_contiguous_concrete-shared | 4 | 217 | 21844 | passed |
| nonad_einsum_n64_col_major_contiguous_eager-shared | 4 | 118 | 18357 | passed |
| nonad_einsum3_n64_col_major_contiguous_concrete-fresh | 4 | 334 | 27037 | passed |
| nonad_einsum3_n64_col_major_contiguous_eager-no-ad | 4 | 252 | 38138 | passed |
| nonad_einsum3_n64_col_major_contiguous_concrete-shared | 4 | 332 | 26773 | passed |
| nonad_einsum3_n64_col_major_contiguous_eager-shared | 4 | 250 | 37874 | passed |
| nonad_qr_n64_col_major_contiguous_concrete-fresh | 4 | 8 | 66808 | passed |
| nonad_qr_n64_col_major_contiguous_eager-no-ad | 4 | 41 | 77187 | passed |
| nonad_qr_n64_col_major_contiguous_concrete-shared | 4 | 6 | 66544 | passed |
| nonad_qr_n64_col_major_contiguous_eager-shared | 4 | 40 | 77115 | passed |
| nonad_svd_n64_col_major_contiguous_concrete-fresh | 4 | 11 | 169384 | passed |
| nonad_svd_n64_col_major_contiguous_eager-no-ad | 4 | 49 | 181075 | passed |
| nonad_svd_n64_col_major_contiguous_concrete-shared | 4 | 8 | 168928 | passed |
| nonad_svd_n64_col_major_contiguous_eager-shared | 4 | 47 | 180811 | passed |
| nonad_solve_n64_col_major_contiguous_concrete-fresh | 4 | 6 | 98648 | passed |
| nonad_solve_n64_col_major_contiguous_eager-no-ad | 4 | 41 | 83515 | passed |
| nonad_solve_n64_col_major_contiguous_concrete-shared | 4 | 5 | 98576 | passed |
| nonad_solve_n64_col_major_contiguous_eager-shared | 4 | 40 | 83443 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | 4 | 218 | 18068 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | 4 | 7 | 1264 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | 4 | 5 | 120 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | 4 | 218 | 18964 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | 4 | 7 | 5296 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | 4 | 5 | 6224 | passed |

## Explicit setup diagnostics (not operation comparisons)

| Case | Median ns | IQR ns | Status |
|---|---:|---:|---|
| add_f64_concrete_fresh | 1831.09 | 14.08 | ok |
| add_f64_concrete-fresh_size4_dependent10 | 19243.67 | 360.36 | ok |
| add_f64_concrete-fresh_size16_single | 1857.78 | 53.57 | ok |
| add_f64_concrete-fresh_size16_dependent10 | 18394.80 | 531.09 | ok |
| add_f64_concrete-fresh_size256_single | 1993.37 | 92.46 | ok |
| add_f64_concrete-fresh_size256_dependent10 | 19526.44 | 1009.85 | ok |
| add_f64_eager-no-ad_size4_single | 4144.82 | 85.89 | ok |
| add_f64_eager-no-ad_size4_dependent10 | 37887.91 | 1932.28 | ok |
| add_f64_eager_no_ad | 4362.92 | 195.50 | ok |
| add_f64_eager-no-ad_size16_dependent10 | 37122.77 | 859.39 | ok |
| add_f64_eager-no-ad_size256_single | 4450.95 | 122.03 | ok |
| add_f64_eager-no-ad_size256_dependent10 | 39289.30 | 1360.48 | ok |
| add_f64_eager_active_ad | 9912.23 | 348.93 | ok |
| add_f64_eager-ad_size4_dependent10 | 91900.31 | 1948.11 | ok |
| add_f64_eager-ad_size16_single | 10379.22 | 414.61 | ok |
| add_f64_eager-ad_size16_dependent10 | 91639.06 | 2873.02 | ok |
| add_f64_eager-ad_size256_single | 10900.74 | 496.96 | ok |
| add_f64_eager-ad_size256_dependent10 | 93263.09 | 5279.11 | ok |
| einsum_f64_concrete-fresh_n2_single | 26446.08 | 358.29 | ok |
| einsum_f64_concrete-fresh_n4_single | 26975.28 | 485.74 | ok |
| einsum_f64_concrete-fresh_n16_single | 27364.68 | 528.58 | ok |
| einsum_f64_concrete-shared_n2_single | 23688.68 | 387.28 | ok |
| einsum_f64_concrete-shared_n4_single | 24332.70 | 331.92 | ok |
| einsum_f64_concrete-shared_n16_single | 24396.14 | 465.04 | ok |
| einsum_f64_eager-no-ad_n2_single | 14680.63 | 532.27 | ok |
| einsum_f64_eager-no-ad_n4_single | 14772.96 | 510.18 | ok |
| einsum_f64_eager-no-ad_n16_single | 15370.86 | 184.11 | ok |
| einsum_f64_eager-ad_n2_single | 25288.83 | 473.21 | ok |
| einsum_f64_eager-ad_n4_single | 24577.27 | 517.23 | ok |
| einsum_f64_eager-ad_n16_single | 25832.09 | 499.43 | ok |
| einsum_f64_prepared-setup_n2_single | 6264.40 | 169.99 | ok |
| einsum_f64_prepared-setup_n4_single | 6288.57 | 109.00 | ok |
| einsum_f64_prepared-setup_n16_single | 6206.56 | 76.45 | ok |
| einsum_f64_borrowed-fresh_n2_col_major_contiguous_single | 27081.12 | 437.34 | ok |
| einsum_f64_borrowed-fresh_n2_row_major_contiguous_single | 26711.83 | 249.24 | ok |
| einsum_f64_borrowed-fresh_n2_strided_single | 29827.78 | 769.24 | ok |
| einsum_f64_borrowed-shared_n2_col_major_contiguous_single | 24080.00 | 211.03 | ok |
| einsum_f64_borrowed-shared_n2_row_major_contiguous_single | 23866.12 | 313.91 | ok |
| einsum_f64_borrowed-shared_n2_strided_single | 27414.57 | 226.01 | ok |
| einsum_f64_borrowed-fresh_n4_col_major_contiguous_single | 27202.84 | 467.56 | ok |
| einsum_f64_borrowed-fresh_n4_row_major_contiguous_single | 26852.58 | 433.64 | ok |
| einsum_f64_borrowed-fresh_n4_strided_single | 30071.01 | 530.95 | NOISY |
| einsum_f64_borrowed-shared_n4_col_major_contiguous_single | 24144.01 | 423.15 | ok |
| einsum_f64_borrowed-shared_n4_row_major_contiguous_single | 24493.62 | 491.15 | ok |
| einsum_f64_borrowed-shared_n4_strided_single | 27366.22 | 446.95 | ok |
| einsum_f64_borrowed-fresh_n16_col_major_contiguous_single | 27308.97 | 392.84 | ok |
| einsum_f64_borrowed-fresh_n16_row_major_contiguous_single | 27830.61 | 582.96 | ok |
| einsum_f64_borrowed-fresh_n16_strided_single | 30608.90 | 259.35 | ok |
| einsum_f64_borrowed-shared_n16_col_major_contiguous_single | 24991.42 | 428.66 | ok |
| einsum_f64_borrowed-shared_n16_row_major_contiguous_single | 25309.11 | 694.35 | ok |
| einsum_f64_borrowed-shared_n16_strided_single | 28545.30 | 207.58 | ok |
| einsum_f64_compiled-repeat_n2_single | 24867.85 | 508.80 | ok |
| einsum_f64_compiled-repeat_n4_single | 24245.58 | 429.80 | ok |
| einsum_f64_compiled-repeat_n16_single | 24975.83 | 389.08 | ok |
| einsum_c64_concrete-fresh_n2_single | 29789.40 | 399.36 | ok |
| einsum_c64_concrete-shared_n2_single | 26924.35 | 538.46 | ok |
| einsum_c64_concrete-fresh_n4_single | 30577.66 | 221.68 | ok |
| einsum_c64_concrete-shared_n4_single | 27551.39 | 528.57 | ok |
| einsum_c64_concrete-fresh_n16_single | 31147.11 | 491.87 | ok |
| einsum_c64_concrete-shared_n16_single | 28021.35 | 268.55 | ok |
| einsum_c64_prepared-setup_n2_single | 6267.96 | 48.11 | ok |
| einsum_c64_prepared-setup_n4_single | 6276.66 | 92.86 | ok |
| einsum_c64_prepared-setup_n16_single | 6323.04 | 125.54 | ok |
| einsum_c64_eager-no-ad_n2_single | 17807.57 | 413.94 | ok |
| einsum_c64_compiled-repeat_n2_single | 27120.46 | 644.13 | ok |
| einsum_c64_eager-no-ad_n4_single | 17684.01 | 355.84 | ok |
| einsum_c64_compiled-repeat_n4_single | 26808.18 | 379.81 | ok |
| einsum_c64_eager-no-ad_n16_single | 19183.10 | 440.11 | ok |
| einsum_c64_compiled-repeat_n16_single | 27567.96 | 687.98 | ok |
| einsum_c64_eager-ad_n2_single | 28048.67 | 311.40 | ok |
| einsum_c64_eager-ad_n4_single | 28351.57 | 811.78 | ok |
| einsum_c64_eager-ad_n16_single | 30232.60 | 774.72 | ok |
| einsum_f64_compiled-setup_n2_single | 5630.01 | 59.99 | ok |
| einsum_f64_compiled-setup_n4_single | 5554.80 | 130.76 | ok |
| einsum_f64_compiled-setup_n16_single | 5497.59 | 70.70 | ok |
| einsum_c64_compiled-setup_n2_single | 5587.04 | 93.67 | ok |
| einsum_c64_compiled-setup_n4_single | 5666.13 | 107.71 | ok |
| einsum_c64_compiled-setup_n16_single | 5577.96 | 179.87 | ok |
| solve_f64_concrete-fresh_n2_single | 2006.59 | 82.56 | ok |
| solve_f64_concrete-fresh_n4_single | 2054.42 | 36.81 | ok |
| solve_f64_concrete-fresh_n16_single | 4990.34 | 75.15 | ok |
| einsum_f64_borrowed-fresh_n2_broadcast_single | 30256.88 | 437.41 | ok |
| einsum_f64_borrowed-shared_n2_broadcast_single | 27189.01 | 254.75 | ok |
| einsum_f64_borrowed-fresh_n4_broadcast_single | 29945.56 | 511.31 | ok |
| einsum_f64_borrowed-shared_n4_broadcast_single | 27121.47 | 594.27 | ok |
| einsum_f64_borrowed-fresh_n16_broadcast_single | 31444.32 | 1118.23 | ok |
| einsum_f64_borrowed-shared_n16_broadcast_single | 28463.17 | 737.98 | ok |
| einsum_f64_concrete-fresh_n8_single | 27843.57 | 489.66 | ok |
| einsum_f64_concrete-shared_n8_single | 25325.28 | 454.35 | ok |
| einsum_f64_prepared-setup_n8_single | 6389.48 | 145.44 | ok |
| einsum_f64_concrete-fresh_n32_single | 29003.30 | 662.33 | ok |
| einsum_f64_concrete-shared_n32_single | 26354.51 | 384.39 | ok |
| einsum_f64_prepared-setup_n32_single | 6232.65 | 101.81 | ok |
| einsum_c64_concrete-fresh_n8_single | 30462.27 | 471.46 | ok |
| einsum_c64_concrete-shared_n8_single | 27733.89 | 455.72 | ok |
| einsum_c64_prepared-setup_n8_single | 6314.12 | 171.10 | ok |
| einsum_c64_concrete-fresh_n32_single | 36770.11 | 955.66 | ok |
| einsum_c64_concrete-shared_n32_single | 33797.83 | 701.68 | ok |
| einsum_c64_prepared-setup_n32_single | 6467.56 | 130.14 | ok |
| gather_f64_concrete_fresh | 2005.21 | 76.35 | ok |
| gather_f64_concrete-fresh_size16_single | 2001.25 | 21.20 | ok |
| gather_f64_concrete-fresh_size256_single | 2255.84 | 127.24 | ok |
| einsum_c64_borrowed-fresh_n2_col_major_contiguous_single | 29491.68 | 338.80 | ok |
| einsum_c64_borrowed-fresh_n2_row_major_contiguous_single | 29835.17 | 485.02 | ok |
| einsum_c64_borrowed-fresh_n2_strided_single | 32611.43 | 380.37 | ok |
| einsum_c64_borrowed-shared_n2_col_major_contiguous_single | 27197.16 | 427.23 | ok |
| einsum_c64_borrowed-shared_n2_row_major_contiguous_single | 26548.05 | 600.51 | ok |
| einsum_c64_borrowed-shared_n2_strided_single | 29299.50 | 646.56 | ok |
| einsum_c64_borrowed-fresh_n4_col_major_contiguous_single | 30734.72 | 360.67 | ok |
| einsum_c64_borrowed-fresh_n4_row_major_contiguous_single | 30926.13 | 556.02 | ok |
| einsum_c64_borrowed-fresh_n4_strided_single | 32466.73 | 488.87 | ok |
| einsum_c64_borrowed-shared_n4_col_major_contiguous_single | 27625.71 | 566.80 | ok |
| einsum_c64_borrowed-shared_n4_row_major_contiguous_single | 27899.95 | 373.57 | ok |
| einsum_c64_borrowed-shared_n4_strided_single | 30553.07 | 454.44 | ok |
| einsum_c64_borrowed-fresh_n16_col_major_contiguous_single | 31838.54 | 339.16 | ok |
| einsum_c64_borrowed-fresh_n16_row_major_contiguous_single | 31630.18 | 425.68 | ok |
| einsum_c64_borrowed-fresh_n16_strided_single | 35312.57 | 567.08 | ok |
| einsum_c64_borrowed-shared_n16_col_major_contiguous_single | 28669.11 | 459.99 | ok |
| einsum_c64_borrowed-shared_n16_row_major_contiguous_single | 28314.33 | 341.32 | ok |
| einsum_c64_borrowed-shared_n16_strided_single | 31711.37 | 513.66 | ok |
| einsum_c64_borrowed-fresh_n2_broadcast_single | 32094.11 | 401.43 | ok |
| einsum_c64_borrowed-shared_n2_broadcast_single | 29207.19 | 377.11 | ok |
| einsum_c64_borrowed-fresh_n4_broadcast_single | 33910.68 | 633.75 | ok |
| einsum_c64_borrowed-shared_n4_broadcast_single | 31478.65 | 668.10 | ok |
| einsum_c64_borrowed-fresh_n16_broadcast_single | 35689.77 | 668.06 | ok |
| einsum_c64_borrowed-shared_n16_broadcast_single | 32236.20 | 457.37 | ok |
| reduce_sum_f64_concrete-fresh_size4_single | 1808.53 | 97.73 | ok |
| reduce_sum_f64_concrete-fresh_size16_single | 1727.28 | 66.26 | ok |
| reduce_sum_f64_concrete-fresh_size256_single | 1776.63 | 45.33 | ok |
| einsum_f64_concrete-shared_abcd-dbef-acef_d4_single | 34365.38 | 483.66 | ok |
| einsum_f64_concrete-shared_ax-asb-xsb_d4s2_single | 25279.09 | 225.78 | ok |
| einsum_f64_concrete-shared_ij-jk-ik_m95k95n1_single | 26007.56 | 370.83 | ok |
| add_f64_concrete_fresh | 111250.02 | 991.68 | ok |
| add_f64_concrete-fresh_size4_dependent10 | 1123731.44 | 24526.09 | ok |
| add_f64_concrete-fresh_size16_single | 110966.55 | 2030.53 | ok |
| add_f64_concrete-fresh_size16_dependent10 | 1112577.06 | 37692.28 | ok |
| add_f64_concrete-fresh_size256_single | 109952.49 | 901.09 | ok |
| add_f64_concrete-fresh_size256_dependent10 | 1109424.38 | 39315.09 | ok |
| add_f64_eager-no-ad_size4_single | 122053.85 | 592.04 | ok |
| add_f64_eager-no-ad_size4_dependent10 | 1228204.19 | 14264.16 | ok |
| add_f64_eager_no_ad | 122133.90 | 1801.10 | ok |
| add_f64_eager-no-ad_size16_dependent10 | 1229208.56 | 35085.53 | ok |
| add_f64_eager-no-ad_size256_single | 122474.88 | 534.04 | ok |
| add_f64_eager-no-ad_size256_dependent10 | 1223108.00 | 18639.53 | ok |
| add_f64_eager_active_ad | 134611.05 | 1876.09 | ok |
| add_f64_eager-ad_size4_dependent10 | 1322466.00 | 21591.69 | ok |
| add_f64_eager-ad_size16_single | 134417.07 | 1712.62 | ok |
| add_f64_eager-ad_size16_dependent10 | 1336697.06 | 28226.03 | ok |
| add_f64_eager-ad_size256_single | 135059.29 | 816.88 | ok |
| add_f64_eager-ad_size256_dependent10 | 1330825.00 | 14509.34 | ok |
| einsum_f64_concrete-fresh_n2_single | 26629.60 | 473.84 | ok |
| einsum_f64_concrete-fresh_n4_single | 27486.92 | 341.15 | ok |
| einsum_f64_concrete-fresh_n16_single | 26731.76 | 404.50 | ok |
| einsum_f64_concrete-shared_n2_single | 24151.09 | 340.91 | ok |
| einsum_f64_concrete-shared_n4_single | 24953.77 | 344.28 | ok |
| einsum_f64_concrete-shared_n16_single | 24429.98 | 544.43 | ok |
| einsum_f64_eager-no-ad_n2_single | 14492.65 | 334.30 | ok |
| einsum_f64_eager-no-ad_n4_single | 15185.70 | 463.00 | ok |
| einsum_f64_eager-no-ad_n16_single | 15279.18 | 200.15 | ok |
| einsum_f64_eager-ad_n2_single | 24844.28 | 371.57 | ok |
| einsum_f64_eager-ad_n4_single | 24651.28 | 572.46 | ok |
| einsum_f64_eager-ad_n16_single | 25720.77 | 1043.11 | ok |
| einsum_f64_prepared-setup_n2_single | 6360.23 | 116.99 | ok |
| einsum_f64_prepared-setup_n4_single | 6349.75 | 134.16 | ok |
| einsum_f64_prepared-setup_n16_single | 6450.89 | 208.61 | ok |
| einsum_f64_borrowed-fresh_n2_col_major_contiguous_single | 27064.90 | 460.62 | ok |
| einsum_f64_borrowed-fresh_n2_row_major_contiguous_single | 26714.19 | 372.36 | ok |
| einsum_f64_borrowed-fresh_n2_strided_single | 30003.68 | 514.06 | ok |
| einsum_f64_borrowed-shared_n2_col_major_contiguous_single | 24257.52 | 277.71 | ok |
| einsum_f64_borrowed-shared_n2_row_major_contiguous_single | 24281.60 | 258.59 | ok |
| einsum_f64_borrowed-shared_n2_strided_single | 27518.36 | 622.54 | ok |
| einsum_f64_borrowed-fresh_n4_col_major_contiguous_single | 26913.49 | 420.01 | ok |
| einsum_f64_borrowed-fresh_n4_row_major_contiguous_single | 27002.55 | 637.46 | ok |
| einsum_f64_borrowed-fresh_n4_strided_single | 30098.01 | 250.44 | ok |
| einsum_f64_borrowed-shared_n4_col_major_contiguous_single | 25448.68 | 404.99 | ok |
| einsum_f64_borrowed-shared_n4_row_major_contiguous_single | 25000.51 | 362.76 | ok |
| einsum_f64_borrowed-shared_n4_strided_single | 28111.76 | 505.16 | ok |
| einsum_f64_borrowed-fresh_n16_col_major_contiguous_single | 27619.92 | 622.53 | ok |
| einsum_f64_borrowed-fresh_n16_row_major_contiguous_single | 28257.80 | 449.40 | ok |
| einsum_f64_borrowed-fresh_n16_strided_single | 31232.31 | 273.46 | ok |
| einsum_f64_borrowed-shared_n16_col_major_contiguous_single | 25496.10 | 318.90 | ok |
| einsum_f64_borrowed-shared_n16_row_major_contiguous_single | 24964.46 | 434.66 | NOISY |
| einsum_f64_borrowed-shared_n16_strided_single | 28650.66 | 375.49 | ok |
| einsum_f64_compiled-repeat_n2_single | 24635.26 | 407.72 | ok |
| einsum_f64_compiled-repeat_n4_single | 23684.94 | 253.74 | ok |
| einsum_f64_compiled-repeat_n16_single | 24168.75 | 525.07 | ok |
| einsum_c64_concrete-fresh_n2_single | 29829.10 | 427.84 | ok |
| einsum_c64_concrete-shared_n2_single | 26860.22 | 541.15 | ok |
| einsum_c64_concrete-fresh_n4_single | 30274.17 | 505.11 | ok |
| einsum_c64_concrete-shared_n4_single | 27741.99 | 245.07 | ok |
| einsum_c64_concrete-fresh_n16_single | 31507.71 | 514.39 | ok |
| einsum_c64_concrete-shared_n16_single | 29329.01 | 603.06 | ok |
| einsum_c64_prepared-setup_n2_single | 6383.27 | 169.24 | ok |
| einsum_c64_prepared-setup_n4_single | 6288.47 | 136.42 | ok |
| einsum_c64_prepared-setup_n16_single | 6358.09 | 181.86 | ok |
| einsum_c64_eager-no-ad_n2_single | 17727.24 | 304.73 | ok |
| einsum_c64_compiled-repeat_n2_single | 27136.34 | 521.73 | ok |
| einsum_c64_eager-no-ad_n4_single | 17700.48 | 470.42 | ok |
| einsum_c64_compiled-repeat_n4_single | 26999.27 | 398.69 | ok |
| einsum_c64_eager-no-ad_n16_single | 19116.80 | 411.31 | ok |
| einsum_c64_compiled-repeat_n16_single | 28780.02 | 575.52 | ok |
| einsum_c64_eager-ad_n2_single | 28088.76 | 911.42 | ok |
| einsum_c64_eager-ad_n4_single | 28349.82 | 1213.27 | NOISY |
| einsum_c64_eager-ad_n16_single | 30002.94 | 724.21 | ok |
| einsum_f64_compiled-setup_n2_single | 6103.78 | 479.89 | NOISY |
| einsum_f64_compiled-setup_n4_single | 5759.20 | 230.38 | ok |
| einsum_f64_compiled-setup_n16_single | 5764.35 | 348.55 | ok |
| einsum_c64_compiled-setup_n2_single | 6074.07 | 272.24 | ok |
| einsum_c64_compiled-setup_n4_single | 5873.18 | 481.23 | ok |
| einsum_c64_compiled-setup_n16_single | 5905.73 | 332.47 | ok |
| solve_f64_concrete-fresh_n2_single | 2051.76 | 133.92 | ok |
| solve_f64_concrete-fresh_n4_single | 2135.87 | 187.71 | ok |
| solve_f64_concrete-fresh_n16_single | 5481.29 | 304.10 | ok |
| einsum_f64_borrowed-fresh_n2_broadcast_single | 30188.80 | 564.60 | ok |
| einsum_f64_borrowed-shared_n2_broadcast_single | 27099.29 | 387.94 | ok |
| einsum_f64_borrowed-fresh_n4_broadcast_single | 30664.10 | 615.95 | ok |
| einsum_f64_borrowed-shared_n4_broadcast_single | 27645.54 | 482.28 | ok |
| einsum_f64_borrowed-fresh_n16_broadcast_single | 31648.45 | 576.59 | ok |
| einsum_f64_borrowed-shared_n16_broadcast_single | 29325.23 | 664.98 | ok |
| einsum_f64_concrete-fresh_n8_single | 28150.63 | 634.87 | ok |
| einsum_f64_concrete-shared_n8_single | 25131.82 | 579.88 | ok |
| einsum_f64_prepared-setup_n8_single | 6437.94 | 121.62 | ok |
| einsum_f64_concrete-fresh_n32_single | 36316.02 | 1902.25 | ok |
| einsum_f64_concrete-shared_n32_single | 31358.41 | 802.96 | ok |
| einsum_f64_prepared-setup_n32_single | 6883.68 | 276.29 | ok |
| einsum_c64_concrete-fresh_n8_single | 32265.72 | 1426.66 | ok |
| einsum_c64_concrete-shared_n8_single | 30238.82 | 811.00 | ok |
| einsum_c64_prepared-setup_n8_single | 6750.72 | 275.44 | ok |
| einsum_c64_concrete-fresh_n32_single | 39827.43 | 1056.11 | ok |
| einsum_c64_concrete-shared_n32_single | 35630.70 | 1258.59 | ok |
| einsum_c64_prepared-setup_n32_single | 6936.94 | 239.98 | ok |
| gather_f64_concrete_fresh | 98905.05 | 2357.79 | ok |
| gather_f64_concrete-fresh_size16_single | 97138.01 | 2087.45 | ok |
| gather_f64_concrete-fresh_size256_single | 101412.15 | 1034.22 | ok |
| einsum_c64_borrowed-fresh_n2_col_major_contiguous_single | 30172.05 | 977.01 | ok |
| einsum_c64_borrowed-fresh_n2_row_major_contiguous_single | 30799.73 | 630.68 | ok |
| einsum_c64_borrowed-fresh_n2_strided_single | 34602.88 | 755.74 | ok |
| einsum_c64_borrowed-shared_n2_col_major_contiguous_single | 28484.34 | 1261.91 | ok |
| einsum_c64_borrowed-shared_n2_row_major_contiguous_single | 28754.16 | 227.94 | ok |
| einsum_c64_borrowed-shared_n2_strided_single | 31448.80 | 996.86 | ok |
| einsum_c64_borrowed-fresh_n4_col_major_contiguous_single | 30919.16 | 459.33 | ok |
| einsum_c64_borrowed-fresh_n4_row_major_contiguous_single | 31288.47 | 1637.03 | ok |
| einsum_c64_borrowed-fresh_n4_strided_single | 34957.54 | 2032.88 | ok |
| einsum_c64_borrowed-shared_n4_col_major_contiguous_single | 28004.40 | 1253.44 | ok |
| einsum_c64_borrowed-shared_n4_row_major_contiguous_single | 27077.49 | 359.11 | ok |
| einsum_c64_borrowed-shared_n4_strided_single | 30924.60 | 613.90 | ok |
| einsum_c64_borrowed-fresh_n16_col_major_contiguous_single | 31779.05 | 403.81 | ok |
| einsum_c64_borrowed-fresh_n16_row_major_contiguous_single | 31741.88 | 1380.86 | ok |
| einsum_c64_borrowed-fresh_n16_strided_single | 34563.53 | 573.43 | ok |
| einsum_c64_borrowed-shared_n16_col_major_contiguous_single | 29057.97 | 755.98 | ok |
| einsum_c64_borrowed-shared_n16_row_major_contiguous_single | 29107.10 | 520.69 | ok |
| einsum_c64_borrowed-shared_n16_strided_single | 32546.30 | 467.39 | ok |
| einsum_c64_borrowed-fresh_n2_broadcast_single | 32859.18 | 495.01 | ok |
| einsum_c64_borrowed-shared_n2_broadcast_single | 29662.02 | 314.33 | ok |
| einsum_c64_borrowed-fresh_n4_broadcast_single | 33002.91 | 482.51 | ok |
| einsum_c64_borrowed-shared_n4_broadcast_single | 30697.97 | 485.31 | ok |
| einsum_c64_borrowed-fresh_n16_broadcast_single | 34345.02 | 446.08 | ok |
| einsum_c64_borrowed-shared_n16_broadcast_single | 31660.22 | 666.09 | ok |
| reduce_sum_f64_concrete-fresh_size4_single | 98454.72 | 1694.12 | ok |
| reduce_sum_f64_concrete-fresh_size16_single | 97806.66 | 2105.79 | ok |
| reduce_sum_f64_concrete-fresh_size256_single | 98396.70 | 2367.73 | ok |
| einsum_f64_concrete-shared_abcd-dbef-acef_d4_single | 34572.24 | 514.88 | ok |
| einsum_f64_concrete-shared_ax-asb-xsb_d4s2_single | 26974.19 | 387.59 | ok |
| einsum_f64_concrete-shared_ij-jk-ik_m95k95n1_single | 26082.82 | 406.37 | ok |

## Per-route timing boundaries

Shared-session routes enter/exit the session once around warmup/calibration/all samples, outside timing. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions reuse the entered executor context. Earlier BLAS revisions included per-operation executor entry; consult the recorded tenferro commit. Fresh routes enter/exit for each operation. Prepared-repeat excludes plan preparation; compiled-repeat excludes tracing/compilation and runtime construction. Prepared-into-repeat and dot-general-into-shared write into one destination preallocated outside timing (no per-call output allocation) and re-check it after the last batch.

- **add / concrete-fresh** — inside: session_entry_exit, add, output_allocation; outside: backend_construction, input_construction, correctness_check, output_destruction, operation_config_construction.
- **add / concrete-shared** — inside: add, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **add / eager-ad** — inside: add_forward_recording, eager_dispatch, output_allocation; outside: backend_construction, input_construction, eager_runtime_construction, backward, correctness_check, output_destruction, operation_config_construction.
- **add / eager-no-ad** — inside: add, eager_dispatch, output_allocation; outside: backend_construction, input_construction, eager_runtime_construction, correctness_check, output_destruction, operation_config_construction.
- **einsum / borrowed-fresh** — inside: session_entry_exit, einsum, output_allocation; outside: backend_construction, input_construction, correctness_check, output_destruction, operation_config_construction.
- **einsum / borrowed-shared** — inside: einsum, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **einsum / compiled-repeat** — inside: runtime_admission_execution, output_allocation; outside: backend_construction, input_construction, trace_compile, runtime_construction, input_bindings, correctness_check, output_destruction, operation_config_construction.
- **einsum / compiled-setup** — inside: trace_compile, program_allocation; outside: backend_construction, input_construction, runtime_construction, input_bindings, correctness_check, output_destruction, operation_config_construction.
- **einsum / concrete-fresh** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), per-call session entry/exit diagnostic, ordinary einsum default Auto planning/cache lookup included equally; parsed integer notation; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **einsum / concrete-scoped** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), per-call session entry/exit diagnostic, ordinary einsum default Auto planning/cache lookup included equally; parsed integer notation; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction, execution scope admission/exit; cloned witness setup.
- **einsum / concrete-shared** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), session entry excluded, ordinary einsum default Auto planning/cache lookup included equally; parsed integer notation; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **einsum / dot-general-into-shared** — inside: dot_general_read_into; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, dot_general_config_construction, output_preallocation, output_destruction, operation_config_construction.
- **einsum / eager-ad** — inside: einsum_forward_recording, eager_dispatch, output_allocation; outside: backend_construction, input_construction, eager_runtime_construction, backward, correctness_check, output_destruction, operation_config_construction.
- **einsum / eager-no-ad** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), per-call session entry/exit diagnostic, ordinary einsum default Auto planning/cache lookup included equally; parsed integer notation; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **einsum / eager-shared** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), session entry excluded, ordinary einsum default Auto planning/cache lookup included equally; parsed integer notation; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **einsum / prepared-into-repeat** — inside: prepared_execute_into; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, plan_preparation, output_preallocation, output_destruction, operation_config_construction.
- **einsum / prepared-repeat** — inside: prepared_execute, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, plan_preparation, output_destruction, operation_config_construction.
- **einsum / prepared-setup** — inside: prepare, plan_allocation; outside: backend_construction, input_construction, correctness_check, output_destruction, operation_config_construction.
- **einsum3 / concrete-fresh** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), per-call session entry/exit diagnostic, ordinary einsum default Auto planning/cache lookup included equally; parsed integer notation; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **einsum3 / concrete-shared** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), session entry excluded, ordinary einsum default Auto planning/cache lookup included equally; parsed integer notation; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **einsum3 / eager-no-ad** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), per-call session entry/exit diagnostic, ordinary einsum default Auto planning/cache lookup included equally; parsed integer notation; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **einsum3 / eager-shared** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), session entry excluded, ordinary einsum default Auto planning/cache lookup included equally; parsed integer notation; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **gather / concrete-fresh** — inside: session_entry_exit, gather, output_allocation; outside: backend_construction, input_construction, correctness_check, output_destruction, operation_config_construction.
- **gather / concrete-shared** — inside: gather, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **qr / concrete-fresh** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), per-call session entry/exit diagnostic, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **qr / concrete-scoped** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), per-call session entry/exit diagnostic, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction, execution scope admission/exit; cloned witness setup.
- **qr / concrete-shared** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), session entry excluded, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **qr / eager-no-ad** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), per-call session entry/exit diagnostic, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **qr / eager-shared** — inside: public extension call, intrinsic public output allocation; stack output retention enum (no helper heap allocation), session entry excluded, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **reduce_sum / concrete-fresh** — inside: session_entry_exit, reduce_sum, output_allocation; outside: backend_construction, input_construction, correctness_check, output_destruction, operation_config_construction.
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
  "collection_total_seconds": 2.2285881609996068,
  "case_process_seconds": 2.132261948998348
}
```

Wall observations `wall_allocations_t4.json`:
```json
{
  "collection_total_seconds": 2.4951596950004387,
  "case_process_seconds": 2.4007424389992593
}
```

Wall observations `wall_command.json`:
```json
{
  "command_seconds_before_final_report_write": 181.018847208,
  "report_generation_seconds": 0.24788065,
  "scope": "command startup through report generation; excludes final report file writes",
  "budget_seconds": null,
  "budget_status": "unconfigured"
}
```

Wall observations `wall_measure_t1.json`:
```json
{
  "collection_total_seconds": 60.2361601270004,
  "case_process_seconds": 60.05427672300175
}
```

Wall observations `wall_measure_t4.json`:
```json
{
  "collection_total_seconds": 104.22329099500075,
  "case_process_seconds": 104.04157787400436
}
```

Wall observations `wall_phases.json`:
```json
{
  "validation_and_latency_build_seconds": 0.484055744,
  "metadata_and_latency_collection_seconds": 175.110579676,
  "allocation_diagnostic_build_seconds": 0.243865433,
  "allocation_collection_seconds": 4.932465705,
  "command_before_report_seconds": 180.770966558
}
```

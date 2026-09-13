# CPU Small-Work Benchmark Results

- Suite: `cpu/small_work`
- Raw run: `data/results/mac-cpu/cpu/small_work/20260913_035116`

## CPU Information

- Model: `Apple M5 Max`
- Vendor: `Apple`
- Logical CPUs: `18`
- Physical CPUs: `18`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Super: 6 physical / 6 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 6 CPUs/L2); Performance: 12 physical / 12 logical (L1i 128 KiB, L1d 64 KiB, L2 8 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5.1-arm64-arm-64bit-Mach-O`

## run_t1

Full metadata: `data/results/mac-cpu/cpu/small_work/20260913_035116/run_t1.yaml`

```yaml
timestamp: '2026-09-13T03:51:16Z'
tenferro_rs:
  path: /Users/hiroshi/projects/tensor4all/tenferro-benchmark-m5-refresh/extern/tenferro-rs
  commit: a48866b1a0bb52e6f9712c485925105b14ea30b9
  dirty: false
  features:
  - system-accelerate
blas:
  implementation: accelerate
  version: unknown
  library: /System/Library/Frameworks/Accelerate.framework
environment:
  hostname: MacBook-Pro-7.local
  os: macOS-26.5.1-arm64-arm-64bit-Mach-O
  arch: arm64
  cpu: arm
  env:
    OMP_NUM_THREADS: '1'
    OMP_THREAD_LIMIT: '1'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '1'
    TENFERRO_CPU_BACKEND_KIND: blas
    BENCHMARK_TARGET_PROFILE: mac-cpu
    OPENBLAS_NUM_THREADS: '1'
    GOTO_NUM_THREADS: '1'
    MKL_NUM_THREADS: '1'
    VECLIB_MAXIMUM_THREADS: '1'
    VECLIB_NUM_THREADS: '1'
    NUMEXPR_NUM_THREADS: '1'
    BLIS_NUM_THREADS: '1'
    XLA_FLAGS: --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1
    JULIA_NUM_THREADS: '1'
    BENCHMARK_COMMIT: e6674a6704adaa5cd79bd3ca323b9cbcf8242983
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.96.0 (ac68faa20 2026-05-25)
    BENCH_INSTANCE: all 154
```

## run_t4

Full metadata: `data/results/mac-cpu/cpu/small_work/20260913_035116/run_t4.yaml`

```yaml
timestamp: '2026-09-13T03:51:23Z'
tenferro_rs:
  path: /Users/hiroshi/projects/tensor4all/tenferro-benchmark-m5-refresh/extern/tenferro-rs
  commit: a48866b1a0bb52e6f9712c485925105b14ea30b9
  dirty: false
  features:
  - system-accelerate
blas:
  implementation: accelerate
  version: unknown
  library: /System/Library/Frameworks/Accelerate.framework
environment:
  hostname: MacBook-Pro-7.local
  os: macOS-26.5.1-arm64-arm-64bit-Mach-O
  arch: arm64
  cpu: arm
  env:
    OMP_NUM_THREADS: '4'
    OMP_THREAD_LIMIT: '4'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '4'
    TENFERRO_CPU_BACKEND_KIND: blas
    BENCHMARK_TARGET_PROFILE: mac-cpu
    OPENBLAS_NUM_THREADS: '4'
    GOTO_NUM_THREADS: '4'
    MKL_NUM_THREADS: '4'
    VECLIB_MAXIMUM_THREADS: '4'
    VECLIB_NUM_THREADS: '4'
    NUMEXPR_NUM_THREADS: '4'
    BLIS_NUM_THREADS: '4'
    XLA_FLAGS: --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4
    JULIA_NUM_THREADS: '4'
    BENCHMARK_COMMIT: e6674a6704adaa5cd79bd3ca323b9cbcf8242983
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.96.0 (ac68faa20 2026-05-25)
    BENCH_INSTANCE: all 154
```

Numerical checks: **308/308 recorded rows passed** (case × thread count).

## Timing

Sequential release runs; 3 warmups, then calibration to a 1 ms batch and 15 measured batches per case. Statistics describe batch elapsed time / iterations, not individually timed calls. Median and inclusive IQR are in nanoseconds (ns); CoV is sample standard deviation / mean. NOISY means CoV > 10%, a descriptive label only. No noisy rows are excluded.

Workflow time is the total for one call or the complete dependent chain. The separate ns/op column divides the chain median by its operation count, not an independently measured call. Setup rows measure preparation only, not execution.

Inputs and backend construction, independent numerical checks and AD backward checks are outside timing. Returned outputs/plans are consumed with black_box and dropped inside timing; value downloads/checks are outside. Eager AD rows measure forward execution/recording, not backward.

| Case ID | Operation | API route | Dtype | Shape | Layout | Workflow | Threads | Median workflow ns | IQR ns | CoV | ns/op (chain only) | Status |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| add_f64_concrete_fresh | add | concrete-fresh | f64 | 4 | col_major_contiguous | single | 1 | 1822.43 | 579.39 | 17.1% | — | NOISY |
| add_f64_concrete-fresh_size4_dependent10 | add | concrete-fresh | f64 | 4 | col_major_contiguous | dependent10 | 1 | 15626.31 | 528.32 | 5.6% | 1562.63 | ok |
| add_f64_concrete-fresh_size16_single | add | concrete-fresh | f64 | 16 | col_major_contiguous | single | 1 | 1361.17 | 110.25 | 4.5% | — | ok |
| add_f64_concrete-fresh_size16_dependent10 | add | concrete-fresh | f64 | 16 | col_major_contiguous | dependent10 | 1 | 11320.64 | 648.77 | 3.3% | 1132.06 | ok |
| add_f64_concrete-fresh_size256_single | add | concrete-fresh | f64 | 256 | col_major_contiguous | single | 1 | 1207.93 | 30.11 | 2.0% | — | ok |
| add_f64_concrete-fresh_size256_dependent10 | add | concrete-fresh | f64 | 256 | col_major_contiguous | dependent10 | 1 | 11727.54 | 108.07 | 1.2% | 1172.75 | ok |
| add_f64_concrete-shared_size4_single | add | concrete-shared | f64 | 4 | col_major_contiguous | single | 1 | 1065.51 | 19.12 | 1.4% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | f64 | 4 | col_major_contiguous | dependent10 | 1 | 10230.80 | 168.13 | 1.7% | 1023.08 | ok |
| add_f64_concrete_shared | add | concrete-shared | f64 | 16 | col_major_contiguous | single | 1 | 1056.68 | 20.12 | 1.7% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | f64 | 16 | col_major_contiguous | dependent10 | 1 | 10511.39 | 329.43 | 1.8% | 1051.14 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | f64 | 256 | col_major_contiguous | single | 1 | 1099.53 | 25.15 | 1.9% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | f64 | 256 | col_major_contiguous | dependent10 | 1 | 10899.09 | 182.45 | 1.8% | 1089.91 | ok |
| add_f64_eager-no-ad_size4_single | add | eager-no-ad | f64 | 4 | col_major_contiguous | single | 1 | 3844.64 | 105.43 | 1.7% | — | ok |
| add_f64_eager-no-ad_size4_dependent10 | add | eager-no-ad | f64 | 4 | col_major_contiguous | dependent10 | 1 | 38511.72 | 966.80 | 1.9% | 3851.17 | ok |
| add_f64_eager_no_ad | add | eager-no-ad | f64 | 16 | col_major_contiguous | single | 1 | 3968.18 | 110.35 | 1.7% | — | ok |
| add_f64_eager-no-ad_size16_dependent10 | add | eager-no-ad | f64 | 16 | col_major_contiguous | dependent10 | 1 | 38533.88 | 632.16 | 1.3% | 3853.39 | ok |
| add_f64_eager-no-ad_size256_single | add | eager-no-ad | f64 | 256 | col_major_contiguous | single | 1 | 4004.15 | 66.85 | 1.0% | — | ok |
| add_f64_eager-no-ad_size256_dependent10 | add | eager-no-ad | f64 | 256 | col_major_contiguous | dependent10 | 1 | 39695.31 | 792.97 | 1.9% | 3969.53 | ok |
| add_f64_eager_active_ad | add | eager-ad | f64 | 4 | col_major_contiguous | single | 1 | 3571.86 | 120.36 | 3.4% | — | ok |
| add_f64_eager-ad_size4_dependent10 | add | eager-ad | f64 | 4 | col_major_contiguous | dependent10 | 1 | 34699.22 | 1060.55 | 3.5% | 3469.92 | ok |
| add_f64_eager-ad_size16_single | add | eager-ad | f64 | 16 | col_major_contiguous | single | 1 | 3604.49 | 106.69 | 3.2% | — | ok |
| add_f64_eager-ad_size16_dependent10 | add | eager-ad | f64 | 16 | col_major_contiguous | dependent10 | 1 | 34423.19 | 2140.00 | 5.5% | 3442.32 | ok |
| add_f64_eager-ad_size256_single | add | eager-ad | f64 | 256 | col_major_contiguous | single | 1 | 3739.18 | 192.14 | 5.4% | — | ok |
| add_f64_eager-ad_size256_dependent10 | add | eager-ad | f64 | 256 | col_major_contiguous | dependent10 | 1 | 35345.03 | 1759.12 | 5.3% | 3534.50 | ok |
| einsum_f64_concrete-fresh_n2_single | einsum | concrete-fresh | f64 | 2×2 | col_major_contiguous | single | 1 | 5787.60 | 164.55 | 2.4% | — | ok |
| einsum_f64_concrete-fresh_n4_single | einsum | concrete-fresh | f64 | 4×4 | col_major_contiguous | single | 1 | 5790.85 | 215.33 | 4.0% | — | ok |
| einsum_f64_concrete-fresh_n16_single | einsum | concrete-fresh | f64 | 16×16 | col_major_contiguous | single | 1 | 7335.61 | 566.16 | 5.4% | — | ok |
| einsum_f64_concrete-shared_n2_single | einsum | concrete-shared | f64 | 2×2 | col_major_contiguous | single | 1 | 5714.19 | 232.50 | 3.5% | — | ok |
| einsum_f64_concrete-shared_n4_single | einsum | concrete-shared | f64 | 4×4 | col_major_contiguous | single | 1 | 5759.93 | 222.17 | 3.1% | — | ok |
| einsum_f64_concrete-shared_n16_single | einsum | concrete-shared | f64 | 16×16 | col_major_contiguous | single | 1 | 7291.50 | 216.23 | 3.2% | — | ok |
| einsum_f64_eager-no-ad_n2_single | einsum | eager-no-ad | f64 | 2×2 | col_major_contiguous | single | 1 | 5267.09 | 208.09 | 3.9% | — | ok |
| einsum_f64_eager-no-ad_n4_single | einsum | eager-no-ad | f64 | 4×4 | col_major_contiguous | single | 1 | 5192.05 | 178.71 | 4.7% | — | ok |
| einsum_f64_eager-no-ad_n16_single | einsum | eager-no-ad | f64 | 16×16 | col_major_contiguous | single | 1 | 6799.97 | 74.14 | 3.1% | — | ok |
| einsum_f64_eager-ad_n2_single | einsum | eager-ad | f64 | 2×2 | col_major_contiguous | single | 1 | 5457.03 | 472.82 | 7.8% | — | ok |
| einsum_f64_eager-ad_n4_single | einsum | eager-ad | f64 | 4×4 | col_major_contiguous | single | 1 | 5443.04 | 251.79 | 8.1% | — | ok |
| einsum_f64_eager-ad_n16_single | einsum | eager-ad | f64 | 16×16 | col_major_contiguous | single | 1 | 6976.40 | 306.97 | 3.9% | — | ok |
| einsum_f64_prepared-setup_n2_single | einsum | prepared-setup | f64 | 2×2 | col_major_contiguous | single | 1 | 2550.94 | 28.08 | 2.1% | — | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | f64 | 2×2 | col_major_contiguous | single | 1 | 2647.13 | 57.17 | 1.9% | — | ok |
| einsum_f64_prepared-setup_n4_single | einsum | prepared-setup | f64 | 4×4 | col_major_contiguous | single | 1 | 2526.77 | 79.59 | 2.5% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | f64 | 4×4 | col_major_contiguous | single | 1 | 2676.27 | 59.20 | 2.6% | — | ok |
| einsum_f64_prepared-setup_n16_single | einsum | prepared-setup | f64 | 16×16 | col_major_contiguous | single | 1 | 2643.39 | 62.58 | 2.1% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | f64 | 16×16 | col_major_contiguous | single | 1 | 3798.01 | 170.66 | 2.8% | — | ok |
| einsum_f64_borrowed-fresh_n2_col_major_contiguous_single | einsum | borrowed-fresh | f64 | 2×2 | col_major_contiguous | single | 1 | 5955.24 | 267.01 | 2.9% | — | ok |
| einsum_f64_borrowed-fresh_n2_row_major_contiguous_single | einsum | borrowed-fresh | f64 | 2×2 | row_major_contiguous | single | 1 | 5906.09 | 114.99 | 4.1% | — | ok |
| einsum_f64_borrowed-fresh_n2_strided_single | einsum | borrowed-fresh | f64 | 2×2 | strided | single | 1 | 8933.59 | 258.46 | 1.8% | — | ok |
| einsum_f64_borrowed-shared_n2_col_major_contiguous_single | einsum | borrowed-shared | f64 | 2×2 | col_major_contiguous | single | 1 | 5800.13 | 174.07 | 2.8% | — | ok |
| einsum_f64_borrowed-shared_n2_row_major_contiguous_single | einsum | borrowed-shared | f64 | 2×2 | row_major_contiguous | single | 1 | 5833.82 | 184.82 | 2.4% | — | ok |
| einsum_f64_borrowed-shared_n2_strided_single | einsum | borrowed-shared | f64 | 2×2 | strided | single | 1 | 8697.92 | 239.10 | 2.6% | — | ok |
| einsum_f64_borrowed-fresh_n4_col_major_contiguous_single | einsum | borrowed-fresh | f64 | 4×4 | col_major_contiguous | single | 1 | 5889.97 | 250.49 | 2.4% | — | ok |
| einsum_f64_borrowed-fresh_n4_row_major_contiguous_single | einsum | borrowed-fresh | f64 | 4×4 | row_major_contiguous | single | 1 | 5978.52 | 96.11 | 1.6% | — | ok |
| einsum_f64_borrowed-fresh_n4_strided_single | einsum | borrowed-fresh | f64 | 4×4 | strided | single | 1 | 8673.17 | 257.16 | 2.8% | — | ok |
| einsum_f64_borrowed-shared_n4_col_major_contiguous_single | einsum | borrowed-shared | f64 | 4×4 | col_major_contiguous | single | 1 | 5728.19 | 118.90 | 2.6% | — | ok |
| einsum_f64_borrowed-shared_n4_row_major_contiguous_single | einsum | borrowed-shared | f64 | 4×4 | row_major_contiguous | single | 1 | 5787.76 | 177.98 | 3.0% | — | ok |
| einsum_f64_borrowed-shared_n4_strided_single | einsum | borrowed-shared | f64 | 4×4 | strided | single | 1 | 9337.89 | 406.57 | 4.7% | — | ok |
| einsum_f64_borrowed-fresh_n16_col_major_contiguous_single | einsum | borrowed-fresh | f64 | 16×16 | col_major_contiguous | single | 1 | 7456.71 | 194.01 | 1.8% | — | ok |
| einsum_f64_borrowed-fresh_n16_row_major_contiguous_single | einsum | borrowed-fresh | f64 | 16×16 | row_major_contiguous | single | 1 | 7524.74 | 199.55 | 1.8% | — | ok |
| einsum_f64_borrowed-fresh_n16_strided_single | einsum | borrowed-fresh | f64 | 16×16 | strided | single | 1 | 11214.52 | 680.34 | 4.5% | — | ok |
| einsum_f64_borrowed-shared_n16_col_major_contiguous_single | einsum | borrowed-shared | f64 | 16×16 | col_major_contiguous | single | 1 | 7285.16 | 305.66 | 3.2% | — | ok |
| einsum_f64_borrowed-shared_n16_row_major_contiguous_single | einsum | borrowed-shared | f64 | 16×16 | row_major_contiguous | single | 1 | 7090.01 | 111.16 | 1.4% | — | ok |
| einsum_f64_borrowed-shared_n16_strided_single | einsum | borrowed-shared | f64 | 16×16 | strided | single | 1 | 10783.85 | 563.80 | 3.9% | — | ok |
| einsum_f64_compiled-repeat_n2_single | einsum | compiled-repeat | f64 | 2×2 | col_major_contiguous | single | 1 | 14166.34 | 489.42 | 2.9% | — | ok |
| einsum_f64_compiled-repeat_n4_single | einsum | compiled-repeat | f64 | 4×4 | col_major_contiguous | single | 1 | 14135.09 | 406.74 | 2.6% | — | ok |
| einsum_f64_compiled-repeat_n16_single | einsum | compiled-repeat | f64 | 16×16 | col_major_contiguous | single | 1 | 16193.36 | 550.29 | 3.2% | — | ok |
| einsum_c64_concrete-fresh_n2_single | einsum | concrete-fresh | c64 | 2×2 | col_major_contiguous | single | 1 | 5787.60 | 148.68 | 2.0% | — | ok |
| einsum_c64_concrete-shared_n2_single | einsum | concrete-shared | c64 | 2×2 | col_major_contiguous | single | 1 | 5851.40 | 226.16 | 3.1% | — | ok |
| einsum_c64_concrete-fresh_n4_single | einsum | concrete-fresh | c64 | 4×4 | col_major_contiguous | single | 1 | 6105.79 | 358.07 | 3.4% | — | ok |
| einsum_c64_concrete-shared_n4_single | einsum | concrete-shared | c64 | 4×4 | col_major_contiguous | single | 1 | 6297.04 | 318.93 | 3.5% | — | ok |
| einsum_c64_concrete-fresh_n16_single | einsum | concrete-fresh | c64 | 16×16 | col_major_contiguous | single | 1 | 8703.12 | 189.12 | 2.2% | — | ok |
| einsum_c64_concrete-shared_n16_single | einsum | concrete-shared | c64 | 16×16 | col_major_contiguous | single | 1 | 8571.62 | 330.90 | 2.7% | — | ok |
| einsum_c64_prepared-setup_n2_single | einsum | prepared-setup | c64 | 2×2 | col_major_contiguous | single | 1 | 2671.30 | 76.82 | 6.2% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | c64 | 2×2 | col_major_contiguous | single | 1 | 2729.17 | 111.65 | 2.8% | — | ok |
| einsum_c64_prepared-setup_n4_single | einsum | prepared-setup | c64 | 4×4 | col_major_contiguous | single | 1 | 2599.20 | 95.05 | 2.4% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | c64 | 4×4 | col_major_contiguous | single | 1 | 2787.35 | 29.42 | 1.3% | — | ok |
| einsum_c64_prepared-setup_n16_single | einsum | prepared-setup | c64 | 16×16 | col_major_contiguous | single | 1 | 2653.40 | 139.61 | 4.5% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | c64 | 16×16 | col_major_contiguous | single | 1 | 5316.57 | 182.94 | 2.2% | — | ok |
| einsum_c64_eager-no-ad_n2_single | einsum | eager-no-ad | c64 | 2×2 | col_major_contiguous | single | 1 | 5264.98 | 55.74 | 2.1% | — | ok |
| einsum_c64_compiled-repeat_n2_single | einsum | compiled-repeat | c64 | 2×2 | col_major_contiguous | single | 1 | 14455.73 | 289.55 | 2.0% | — | ok |
| einsum_c64_eager-no-ad_n4_single | einsum | eager-no-ad | c64 | 4×4 | col_major_contiguous | single | 1 | 5305.99 | 168.21 | 2.9% | — | ok |
| einsum_c64_compiled-repeat_n4_single | einsum | compiled-repeat | c64 | 4×4 | col_major_contiguous | single | 1 | 14311.52 | 348.96 | 2.4% | — | ok |
| einsum_c64_eager-no-ad_n16_single | einsum | eager-no-ad | c64 | 16×16 | col_major_contiguous | single | 1 | 8098.95 | 358.23 | 3.0% | — | ok |
| einsum_c64_compiled-repeat_n16_single | einsum | compiled-repeat | c64 | 16×16 | col_major_contiguous | single | 1 | 17869.14 | 684.91 | 3.3% | — | ok |
| einsum_c64_eager-ad_n2_single | einsum | eager-ad | c64 | 2×2 | col_major_contiguous | single | 1 | 5464.68 | 302.33 | 4.3% | — | ok |
| einsum_c64_eager-ad_n4_single | einsum | eager-ad | c64 | 4×4 | col_major_contiguous | single | 1 | 5541.18 | 173.58 | 3.1% | — | ok |
| einsum_c64_eager-ad_n16_single | einsum | eager-ad | c64 | 16×16 | col_major_contiguous | single | 1 | 8252.28 | 261.39 | 4.3% | — | ok |
| einsum_f64_compiled-setup_n2_single | einsum | compiled-setup | f64 | 2×2 | col_major_contiguous | single | 1 | 2463.13 | 103.80 | 4.0% | — | ok |
| einsum_f64_compiled-setup_n4_single | einsum | compiled-setup | f64 | 4×4 | col_major_contiguous | single | 1 | 2551.35 | 85.90 | 2.4% | — | ok |
| einsum_f64_compiled-setup_n16_single | einsum | compiled-setup | f64 | 16×16 | col_major_contiguous | single | 1 | 2501.30 | 522.54 | 10.1% | — | NOISY |
| einsum_c64_compiled-setup_n2_single | einsum | compiled-setup | c64 | 2×2 | col_major_contiguous | single | 1 | 2537.03 | 90.58 | 3.4% | — | ok |
| einsum_c64_compiled-setup_n4_single | einsum | compiled-setup | c64 | 4×4 | col_major_contiguous | single | 1 | 2500.24 | 30.76 | 1.8% | — | ok |
| einsum_c64_compiled-setup_n16_single | einsum | compiled-setup | c64 | 16×16 | col_major_contiguous | single | 1 | 2545.08 | 60.71 | 1.6% | — | ok |
| solve_f64_concrete-fresh_n2_single | solve | concrete-fresh | f64 | 2×2 | col_major_contiguous | single | 1 | 2158.77 | 46.47 | 1.3% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | f64 | 2×2 | col_major_contiguous | single | 1 | 2020.43 | 15.01 | 1.0% | — | ok |
| solve_f64_concrete-fresh_n4_single | solve | concrete-fresh | f64 | 4×4 | col_major_contiguous | single | 1 | 2253.74 | 110.43 | 3.1% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | f64 | 4×4 | col_major_contiguous | single | 1 | 2136.80 | 48.10 | 2.9% | — | ok |
| solve_f64_concrete-fresh_n16_single | solve | concrete-fresh | f64 | 16×16 | col_major_contiguous | single | 1 | 3906.25 | 71.53 | 1.9% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | f64 | 16×16 | col_major_contiguous | single | 1 | 3779.05 | 97.49 | 1.7% | — | ok |
| einsum_f64_borrowed-fresh_n2_broadcast_single | einsum | borrowed-fresh | f64 | 2×2 | broadcast | single | 1 | 8751.30 | 325.19 | 2.4% | — | ok |
| einsum_f64_borrowed-shared_n2_broadcast_single | einsum | borrowed-shared | f64 | 2×2 | broadcast | single | 1 | 8645.19 | 311.36 | 2.7% | — | ok |
| einsum_f64_borrowed-fresh_n4_broadcast_single | einsum | borrowed-fresh | f64 | 4×4 | broadcast | single | 1 | 8710.28 | 170.57 | 3.5% | — | ok |
| einsum_f64_borrowed-shared_n4_broadcast_single | einsum | borrowed-shared | f64 | 4×4 | broadcast | single | 1 | 8595.05 | 439.95 | 3.4% | — | ok |
| einsum_f64_borrowed-fresh_n16_broadcast_single | einsum | borrowed-fresh | f64 | 16×16 | broadcast | single | 1 | 11211.26 | 573.89 | 3.4% | — | ok |
| einsum_f64_borrowed-shared_n16_broadcast_single | einsum | borrowed-shared | f64 | 16×16 | broadcast | single | 1 | 10681.64 | 216.46 | 2.3% | — | ok |
| einsum_f64_concrete-fresh_n8_single | einsum | concrete-fresh | f64 | 8×8 | col_major_contiguous | single | 1 | 5700.36 | 159.92 | 2.3% | — | ok |
| einsum_f64_concrete-shared_n8_single | einsum | concrete-shared | f64 | 8×8 | col_major_contiguous | single | 1 | 5652.18 | 125.08 | 1.8% | — | ok |
| einsum_f64_prepared-setup_n8_single | einsum | prepared-setup | f64 | 8×8 | col_major_contiguous | single | 1 | 2622.64 | 66.98 | 2.3% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | f64 | 8×8 | col_major_contiguous | single | 1 | 2714.52 | 70.23 | 1.6% | — | ok |
| einsum_f64_concrete-fresh_n32_single | einsum | concrete-fresh | f64 | 32×32 | col_major_contiguous | single | 1 | 7611.82 | 266.36 | 2.3% | — | ok |
| einsum_f64_concrete-shared_n32_single | einsum | concrete-shared | f64 | 32×32 | col_major_contiguous | single | 1 | 7611.33 | 162.43 | 1.9% | — | ok |
| einsum_f64_prepared-setup_n32_single | einsum | prepared-setup | f64 | 32×32 | col_major_contiguous | single | 1 | 2729.82 | 168.74 | 3.7% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | f64 | 32×32 | col_major_contiguous | single | 1 | 4139.65 | 133.14 | 2.6% | — | ok |
| einsum_c64_concrete-fresh_n8_single | einsum | concrete-fresh | c64 | 8×8 | col_major_contiguous | single | 1 | 6201.50 | 182.13 | 1.8% | — | ok |
| einsum_c64_concrete-shared_n8_single | einsum | concrete-shared | c64 | 8×8 | col_major_contiguous | single | 1 | 6099.61 | 157.14 | 3.0% | — | ok |
| einsum_c64_prepared-setup_n8_single | einsum | prepared-setup | c64 | 8×8 | col_major_contiguous | single | 1 | 2628.09 | 79.59 | 2.1% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | c64 | 8×8 | col_major_contiguous | single | 1 | 3014.81 | 28.48 | 1.3% | — | ok |
| einsum_c64_concrete-fresh_n32_single | einsum | concrete-fresh | c64 | 32×32 | col_major_contiguous | single | 1 | 12782.55 | 646.97 | 3.3% | — | ok |
| einsum_c64_concrete-shared_n32_single | einsum | concrete-shared | c64 | 32×32 | col_major_contiguous | single | 1 | 12487.63 | 553.39 | 3.5% | — | ok |
| einsum_c64_prepared-setup_n32_single | einsum | prepared-setup | c64 | 32×32 | col_major_contiguous | single | 1 | 2576.17 | 56.07 | 2.3% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | c64 | 32×32 | col_major_contiguous | single | 1 | 9086.91 | 303.06 | 2.6% | — | ok |
| gather_f64_concrete_fresh | gather | concrete-fresh | f64 | 4 | col_major_contiguous | single | 1 | 977.54 | 18.20 | 1.7% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | f64 | 4 | col_major_contiguous | single | 1 | 880.39 | 19.32 | 1.7% | — | ok |
| gather_f64_concrete-fresh_size16_single | gather | concrete-fresh | f64 | 16 | col_major_contiguous | single | 1 | 981.77 | 17.01 | 2.2% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | f64 | 16 | col_major_contiguous | single | 1 | 884.16 | 21.25 | 2.1% | — | ok |
| gather_f64_concrete-fresh_size256_single | gather | concrete-fresh | f64 | 256 | col_major_contiguous | single | 1 | 1142.50 | 19.35 | 2.5% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | f64 | 256 | col_major_contiguous | single | 1 | 1067.79 | 11.54 | 1.4% | — | ok |
| einsum_c64_borrowed-fresh_n2_col_major_contiguous_single | einsum | borrowed-fresh | c64 | 2×2 | col_major_contiguous | single | 1 | 6015.46 | 158.85 | 2.4% | — | ok |
| einsum_c64_borrowed-fresh_n2_row_major_contiguous_single | einsum | borrowed-fresh | c64 | 2×2 | row_major_contiguous | single | 1 | 5896.00 | 135.09 | 1.5% | — | ok |
| einsum_c64_borrowed-fresh_n2_strided_single | einsum | borrowed-fresh | c64 | 2×2 | strided | single | 1 | 8911.46 | 158.70 | 2.7% | — | ok |
| einsum_c64_borrowed-shared_n2_col_major_contiguous_single | einsum | borrowed-shared | c64 | 2×2 | col_major_contiguous | single | 1 | 5964.19 | 182.94 | 3.5% | — | ok |
| einsum_c64_borrowed-shared_n2_row_major_contiguous_single | einsum | borrowed-shared | c64 | 2×2 | row_major_contiguous | single | 1 | 5750.00 | 155.60 | 2.2% | — | ok |
| einsum_c64_borrowed-shared_n2_strided_single | einsum | borrowed-shared | c64 | 2×2 | strided | single | 1 | 8718.43 | 219.08 | 2.5% | — | ok |
| einsum_c64_borrowed-fresh_n4_col_major_contiguous_single | einsum | borrowed-fresh | c64 | 4×4 | col_major_contiguous | single | 1 | 6032.88 | 201.99 | 3.1% | — | ok |
| einsum_c64_borrowed-fresh_n4_row_major_contiguous_single | einsum | borrowed-fresh | c64 | 4×4 | row_major_contiguous | single | 1 | 5996.42 | 167.81 | 1.7% | — | ok |
| einsum_c64_borrowed-fresh_n4_strided_single | einsum | borrowed-fresh | c64 | 4×4 | strided | single | 1 | 8960.94 | 440.76 | 3.1% | — | ok |
| einsum_c64_borrowed-shared_n4_col_major_contiguous_single | einsum | borrowed-shared | c64 | 4×4 | col_major_contiguous | single | 1 | 5922.53 | 115.40 | 1.9% | — | ok |
| einsum_c64_borrowed-shared_n4_row_major_contiguous_single | einsum | borrowed-shared | c64 | 4×4 | row_major_contiguous | single | 1 | 5883.14 | 96.03 | 1.4% | — | ok |
| einsum_c64_borrowed-shared_n4_strided_single | einsum | borrowed-shared | c64 | 4×4 | strided | single | 1 | 8817.71 | 220.38 | 2.2% | — | ok |
| einsum_c64_borrowed-fresh_n16_col_major_contiguous_single | einsum | borrowed-fresh | c64 | 16×16 | col_major_contiguous | single | 1 | 8807.62 | 340.33 | 2.5% | — | ok |
| einsum_c64_borrowed-fresh_n16_row_major_contiguous_single | einsum | borrowed-fresh | c64 | 16×16 | row_major_contiguous | single | 1 | 8657.88 | 266.93 | 1.8% | — | ok |
| einsum_c64_borrowed-fresh_n16_strided_single | einsum | borrowed-fresh | c64 | 16×16 | strided | single | 1 | 12463.55 | 406.74 | 2.7% | — | ok |
| einsum_c64_borrowed-shared_n16_col_major_contiguous_single | einsum | borrowed-shared | c64 | 16×16 | col_major_contiguous | single | 1 | 8594.41 | 370.28 | 3.0% | — | ok |
| einsum_c64_borrowed-shared_n16_row_major_contiguous_single | einsum | borrowed-shared | c64 | 16×16 | row_major_contiguous | single | 1 | 8776.70 | 222.66 | 2.0% | — | ok |
| einsum_c64_borrowed-shared_n16_strided_single | einsum | borrowed-shared | c64 | 16×16 | strided | single | 1 | 12090.50 | 673.18 | 3.9% | — | ok |
| einsum_c64_borrowed-fresh_n2_broadcast_single | einsum | borrowed-fresh | c64 | 2×2 | broadcast | single | 1 | 8851.23 | 205.73 | 2.3% | — | ok |
| einsum_c64_borrowed-shared_n2_broadcast_single | einsum | borrowed-shared | c64 | 2×2 | broadcast | single | 1 | 8727.54 | 200.03 | 2.0% | — | ok |
| einsum_c64_borrowed-fresh_n4_broadcast_single | einsum | borrowed-fresh | c64 | 4×4 | broadcast | single | 1 | 9126.62 | 212.08 | 2.0% | — | ok |
| einsum_c64_borrowed-shared_n4_broadcast_single | einsum | borrowed-shared | c64 | 4×4 | broadcast | single | 1 | 8871.41 | 205.89 | 2.0% | — | ok |
| einsum_c64_borrowed-fresh_n16_broadcast_single | einsum | borrowed-fresh | c64 | 16×16 | broadcast | single | 1 | 12274.09 | 701.18 | 3.8% | — | ok |
| einsum_c64_borrowed-shared_n16_broadcast_single | einsum | borrowed-shared | c64 | 16×16 | broadcast | single | 1 | 12179.69 | 576.66 | 3.1% | — | ok |
| reduce_sum_f64_concrete-fresh_size4_single | reduce_sum | concrete-fresh | f64 | 4 | col_major_contiguous | single | 1 | 773.93 | 14.58 | 2.4% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | f64 | 4 | col_major_contiguous | single | 1 | 633.73 | 19.11 | 2.5% | — | ok |
| reduce_sum_f64_concrete-fresh_size16_single | reduce_sum | concrete-fresh | f64 | 16 | col_major_contiguous | single | 1 | 757.06 | 11.94 | 1.3% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | f64 | 16 | col_major_contiguous | single | 1 | 642.90 | 19.71 | 2.2% | — | ok |
| reduce_sum_f64_concrete-fresh_size256_single | reduce_sum | concrete-fresh | f64 | 256 | col_major_contiguous | single | 1 | 774.52 | 6.43 | 1.8% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | f64 | 256 | col_major_contiguous | single | 1 | 664.92 | 8.32 | 1.6% | — | ok |
| add_f64_concrete_fresh | add | concrete-fresh | f64 | 4 | col_major_contiguous | single | 4 | 11712.89 | 3000.33 | 17.0% | — | NOISY |
| add_f64_concrete-fresh_size4_dependent10 | add | concrete-fresh | f64 | 4 | col_major_contiguous | dependent10 | 4 | 152823.00 | 37458.25 | 16.7% | 15282.30 | NOISY |
| add_f64_concrete-fresh_size16_single | add | concrete-fresh | f64 | 16 | col_major_contiguous | single | 4 | 13754.88 | 1484.70 | 9.5% | — | ok |
| add_f64_concrete-fresh_size16_dependent10 | add | concrete-fresh | f64 | 16 | col_major_contiguous | dependent10 | 4 | 104369.75 | 7059.88 | 10.7% | 10436.98 | NOISY |
| add_f64_concrete-fresh_size256_single | add | concrete-fresh | f64 | 256 | col_major_contiguous | single | 4 | 9554.37 | 1205.08 | 8.4% | — | ok |
| add_f64_concrete-fresh_size256_dependent10 | add | concrete-fresh | f64 | 256 | col_major_contiguous | dependent10 | 4 | 99369.81 | 17462.25 | 9.8% | 9936.98 | ok |
| add_f64_concrete-shared_size4_single | add | concrete-shared | f64 | 4 | col_major_contiguous | single | 4 | 8598.96 | 1823.57 | 14.3% | — | NOISY |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | f64 | 4 | col_major_contiguous | dependent10 | 4 | 98007.81 | 16134.09 | 13.3% | 9800.78 | NOISY |
| add_f64_concrete_shared | add | concrete-shared | f64 | 16 | col_major_contiguous | single | 4 | 8248.70 | 1542.64 | 14.2% | — | NOISY |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | f64 | 16 | col_major_contiguous | dependent10 | 4 | 85549.44 | 19675.78 | 15.1% | 8554.94 | NOISY |
| add_f64_concrete-shared_size256_single | add | concrete-shared | f64 | 256 | col_major_contiguous | single | 4 | 8519.53 | 1844.56 | 18.0% | — | NOISY |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | f64 | 256 | col_major_contiguous | dependent10 | 4 | 96015.62 | 19009.16 | 11.5% | 9601.56 | NOISY |
| add_f64_eager-no-ad_size4_single | add | eager-no-ad | f64 | 4 | col_major_contiguous | single | 4 | 32787.75 | 4071.93 | 11.4% | — | NOISY |
| add_f64_eager-no-ad_size4_dependent10 | add | eager-no-ad | f64 | 4 | col_major_contiguous | dependent10 | 4 | 306791.75 | 57677.00 | 16.6% | 30679.17 | NOISY |
| add_f64_eager_no_ad | add | eager-no-ad | f64 | 16 | col_major_contiguous | single | 4 | 30945.31 | 5203.14 | 15.8% | — | NOISY |
| add_f64_eager-no-ad_size16_dependent10 | add | eager-no-ad | f64 | 16 | col_major_contiguous | dependent10 | 4 | 313343.75 | 29213.50 | 11.1% | 31334.38 | NOISY |
| add_f64_eager-no-ad_size256_single | add | eager-no-ad | f64 | 256 | col_major_contiguous | single | 4 | 33356.78 | 3582.69 | 9.2% | — | ok |
| add_f64_eager-no-ad_size256_dependent10 | add | eager-no-ad | f64 | 256 | col_major_contiguous | dependent10 | 4 | 343604.25 | 86838.62 | 19.1% | 34360.43 | NOISY |
| add_f64_eager_active_ad | add | eager-ad | f64 | 4 | col_major_contiguous | single | 4 | 24733.09 | 4797.53 | 54.9% | — | NOISY |
| add_f64_eager-ad_size4_dependent10 | add | eager-ad | f64 | 4 | col_major_contiguous | dependent10 | 4 | 198583.38 | 56487.00 | 15.7% | 19858.34 | NOISY |
| add_f64_eager-ad_size16_single | add | eager-ad | f64 | 16 | col_major_contiguous | single | 4 | 15949.22 | 2852.21 | 16.0% | — | NOISY |
| add_f64_eager-ad_size16_dependent10 | add | eager-ad | f64 | 16 | col_major_contiguous | dependent10 | 4 | 166927.12 | 36138.12 | 12.1% | 16692.71 | NOISY |
| add_f64_eager-ad_size256_single | add | eager-ad | f64 | 256 | col_major_contiguous | single | 4 | 16412.75 | 2493.48 | 21.0% | — | NOISY |
| add_f64_eager-ad_size256_dependent10 | add | eager-ad | f64 | 256 | col_major_contiguous | dependent10 | 4 | 159875.00 | 30401.06 | 14.2% | 15987.50 | NOISY |
| einsum_f64_concrete-fresh_n2_single | einsum | concrete-fresh | f64 | 2×2 | col_major_contiguous | single | 4 | 17436.20 | 846.03 | 14.2% | — | NOISY |
| einsum_f64_concrete-fresh_n4_single | einsum | concrete-fresh | f64 | 4×4 | col_major_contiguous | single | 4 | 17707.03 | 1707.03 | 13.1% | — | NOISY |
| einsum_f64_concrete-fresh_n16_single | einsum | concrete-fresh | f64 | 16×16 | col_major_contiguous | single | 4 | 19956.38 | 2166.34 | 8.7% | — | ok |
| einsum_f64_concrete-shared_n2_single | einsum | concrete-shared | f64 | 2×2 | col_major_contiguous | single | 4 | 19580.08 | 3640.30 | 12.4% | — | NOISY |
| einsum_f64_concrete-shared_n4_single | einsum | concrete-shared | f64 | 4×4 | col_major_contiguous | single | 4 | 18177.08 | 2552.41 | 10.0% | — | ok |
| einsum_f64_concrete-shared_n16_single | einsum | concrete-shared | f64 | 16×16 | col_major_contiguous | single | 4 | 18736.33 | 3656.57 | 10.5% | — | NOISY |
| einsum_f64_eager-no-ad_n2_single | einsum | eager-no-ad | f64 | 2×2 | col_major_contiguous | single | 4 | 30605.47 | 5852.88 | 11.9% | — | NOISY |
| einsum_f64_eager-no-ad_n4_single | einsum | eager-no-ad | f64 | 4×4 | col_major_contiguous | single | 4 | 31487.62 | 4308.91 | 12.9% | — | NOISY |
| einsum_f64_eager-no-ad_n16_single | einsum | eager-no-ad | f64 | 16×16 | col_major_contiguous | single | 4 | 34071.62 | 3276.03 | 10.6% | — | NOISY |
| einsum_f64_eager-ad_n2_single | einsum | eager-ad | f64 | 2×2 | col_major_contiguous | single | 4 | 18774.73 | 2603.19 | 10.3% | — | NOISY |
| einsum_f64_eager-ad_n4_single | einsum | eager-ad | f64 | 4×4 | col_major_contiguous | single | 4 | 18393.23 | 2873.05 | 12.5% | — | NOISY |
| einsum_f64_eager-ad_n16_single | einsum | eager-ad | f64 | 16×16 | col_major_contiguous | single | 4 | 19367.84 | 4184.57 | 11.8% | — | NOISY |
| einsum_f64_prepared-setup_n2_single | einsum | prepared-setup | f64 | 2×2 | col_major_contiguous | single | 4 | 2601.64 | 235.47 | 4.9% | — | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | f64 | 2×2 | col_major_contiguous | single | 4 | 12128.91 | 1102.70 | 6.8% | — | ok |
| einsum_f64_prepared-setup_n4_single | einsum | prepared-setup | f64 | 4×4 | col_major_contiguous | single | 4 | 2627.69 | 322.35 | 7.5% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | f64 | 4×4 | col_major_contiguous | single | 4 | 12020.18 | 479.65 | 6.5% | — | ok |
| einsum_f64_prepared-setup_n16_single | einsum | prepared-setup | f64 | 16×16 | col_major_contiguous | single | 4 | 2747.72 | 223.14 | 4.9% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | f64 | 16×16 | col_major_contiguous | single | 4 | 13320.63 | 928.22 | 5.0% | — | ok |
| einsum_f64_borrowed-fresh_n2_col_major_contiguous_single | einsum | borrowed-fresh | f64 | 2×2 | col_major_contiguous | single | 4 | 18164.72 | 665.04 | 9.3% | — | ok |
| einsum_f64_borrowed-fresh_n2_row_major_contiguous_single | einsum | borrowed-fresh | f64 | 2×2 | row_major_contiguous | single | 4 | 17780.61 | 1475.91 | 9.2% | — | ok |
| einsum_f64_borrowed-fresh_n2_strided_single | einsum | borrowed-fresh | f64 | 2×2 | strided | single | 4 | 21129.56 | 815.43 | 6.2% | — | ok |
| einsum_f64_borrowed-shared_n2_col_major_contiguous_single | einsum | borrowed-shared | f64 | 2×2 | col_major_contiguous | single | 4 | 17874.34 | 1150.39 | 6.1% | — | ok |
| einsum_f64_borrowed-shared_n2_row_major_contiguous_single | einsum | borrowed-shared | f64 | 2×2 | row_major_contiguous | single | 4 | 17523.44 | 874.34 | 5.0% | — | ok |
| einsum_f64_borrowed-shared_n2_strided_single | einsum | borrowed-shared | f64 | 2×2 | strided | single | 4 | 20575.52 | 911.46 | 6.3% | — | ok |
| einsum_f64_borrowed-fresh_n4_col_major_contiguous_single | einsum | borrowed-fresh | f64 | 4×4 | col_major_contiguous | single | 4 | 18273.44 | 1562.18 | 9.6% | — | ok |
| einsum_f64_borrowed-fresh_n4_row_major_contiguous_single | einsum | borrowed-fresh | f64 | 4×4 | row_major_contiguous | single | 4 | 18009.77 | 933.28 | 4.0% | — | ok |
| einsum_f64_borrowed-fresh_n4_strided_single | einsum | borrowed-fresh | f64 | 4×4 | strided | single | 4 | 21125.66 | 1170.90 | 5.2% | — | ok |
| einsum_f64_borrowed-shared_n4_col_major_contiguous_single | einsum | borrowed-shared | f64 | 4×4 | col_major_contiguous | single | 4 | 17705.72 | 1755.86 | 6.9% | — | ok |
| einsum_f64_borrowed-shared_n4_row_major_contiguous_single | einsum | borrowed-shared | f64 | 4×4 | row_major_contiguous | single | 4 | 17348.95 | 576.18 | 4.6% | — | ok |
| einsum_f64_borrowed-shared_n4_strided_single | einsum | borrowed-shared | f64 | 4×4 | strided | single | 4 | 19974.61 | 1200.53 | 5.6% | — | ok |
| einsum_f64_borrowed-fresh_n16_col_major_contiguous_single | einsum | borrowed-fresh | f64 | 16×16 | col_major_contiguous | single | 4 | 18310.55 | 1026.05 | 4.6% | — | ok |
| einsum_f64_borrowed-fresh_n16_row_major_contiguous_single | einsum | borrowed-fresh | f64 | 16×16 | row_major_contiguous | single | 4 | 20209.62 | 2969.08 | 11.5% | — | NOISY |
| einsum_f64_borrowed-fresh_n16_strided_single | einsum | borrowed-fresh | f64 | 16×16 | strided | single | 4 | 23171.88 | 2174.80 | 10.3% | — | NOISY |
| einsum_f64_borrowed-shared_n16_col_major_contiguous_single | einsum | borrowed-shared | f64 | 16×16 | col_major_contiguous | single | 4 | 18144.53 | 1061.20 | 4.8% | — | ok |
| einsum_f64_borrowed-shared_n16_row_major_contiguous_single | einsum | borrowed-shared | f64 | 16×16 | row_major_contiguous | single | 4 | 21577.47 | 4341.47 | 12.0% | — | NOISY |
| einsum_f64_borrowed-shared_n16_strided_single | einsum | borrowed-shared | f64 | 16×16 | strided | single | 4 | 22426.44 | 1456.70 | 9.8% | — | ok |
| einsum_f64_compiled-repeat_n2_single | einsum | compiled-repeat | f64 | 2×2 | col_major_contiguous | single | 4 | 29992.19 | 1420.58 | 6.2% | — | ok |
| einsum_f64_compiled-repeat_n4_single | einsum | compiled-repeat | f64 | 4×4 | col_major_contiguous | single | 4 | 29561.19 | 1839.84 | 8.0% | — | ok |
| einsum_f64_compiled-repeat_n16_single | einsum | compiled-repeat | f64 | 16×16 | col_major_contiguous | single | 4 | 27544.92 | 2465.16 | 9.6% | — | ok |
| einsum_c64_concrete-fresh_n2_single | einsum | concrete-fresh | c64 | 2×2 | col_major_contiguous | single | 4 | 17391.27 | 974.94 | 13.7% | — | NOISY |
| einsum_c64_concrete-shared_n2_single | einsum | concrete-shared | c64 | 2×2 | col_major_contiguous | single | 4 | 17321.61 | 1037.44 | 6.8% | — | ok |
| einsum_c64_concrete-fresh_n4_single | einsum | concrete-fresh | c64 | 4×4 | col_major_contiguous | single | 4 | 17404.30 | 1003.57 | 4.3% | — | ok |
| einsum_c64_concrete-shared_n4_single | einsum | concrete-shared | c64 | 4×4 | col_major_contiguous | single | 4 | 17082.69 | 559.58 | 5.2% | — | ok |
| einsum_c64_concrete-fresh_n16_single | einsum | concrete-fresh | c64 | 16×16 | col_major_contiguous | single | 4 | 19748.70 | 1135.09 | 5.2% | — | ok |
| einsum_c64_concrete-shared_n16_single | einsum | concrete-shared | c64 | 16×16 | col_major_contiguous | single | 4 | 19886.06 | 1026.05 | 3.8% | — | ok |
| einsum_c64_prepared-setup_n2_single | einsum | prepared-setup | c64 | 2×2 | col_major_contiguous | single | 4 | 2687.42 | 189.29 | 4.0% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | c64 | 2×2 | col_major_contiguous | single | 4 | 11888.34 | 552.08 | 5.2% | — | ok |
| einsum_c64_prepared-setup_n4_single | einsum | prepared-setup | c64 | 4×4 | col_major_contiguous | single | 4 | 2580.16 | 307.74 | 6.2% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | c64 | 4×4 | col_major_contiguous | single | 4 | 12106.12 | 942.05 | 5.2% | — | ok |
| einsum_c64_prepared-setup_n16_single | einsum | prepared-setup | c64 | 16×16 | col_major_contiguous | single | 4 | 2574.46 | 87.12 | 3.5% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | c64 | 16×16 | col_major_contiguous | single | 4 | 15106.45 | 744.63 | 4.8% | — | ok |
| einsum_c64_eager-no-ad_n2_single | einsum | eager-no-ad | c64 | 2×2 | col_major_contiguous | single | 4 | 29319.00 | 1711.58 | 5.8% | — | ok |
| einsum_c64_compiled-repeat_n2_single | einsum | compiled-repeat | c64 | 2×2 | col_major_contiguous | single | 4 | 28450.53 | 2219.07 | 5.5% | — | ok |
| einsum_c64_eager-no-ad_n4_single | einsum | eager-no-ad | c64 | 4×4 | col_major_contiguous | single | 4 | 30034.50 | 2924.15 | 6.9% | — | ok |
| einsum_c64_compiled-repeat_n4_single | einsum | compiled-repeat | c64 | 4×4 | col_major_contiguous | single | 4 | 28526.03 | 3088.55 | 27.2% | — | NOISY |
| einsum_c64_eager-no-ad_n16_single | einsum | eager-no-ad | c64 | 16×16 | col_major_contiguous | single | 4 | 51776.03 | 4386.72 | 8.7% | — | ok |
| einsum_c64_compiled-repeat_n16_single | einsum | compiled-repeat | c64 | 16×16 | col_major_contiguous | single | 4 | 43709.66 | 3258.47 | 4.8% | — | ok |
| einsum_c64_eager-ad_n2_single | einsum | eager-ad | c64 | 2×2 | col_major_contiguous | single | 4 | 25192.70 | 2204.75 | 12.0% | — | NOISY |
| einsum_c64_eager-ad_n4_single | einsum | eager-ad | c64 | 4×4 | col_major_contiguous | single | 4 | 24217.44 | 1337.88 | 7.4% | — | ok |
| einsum_c64_eager-ad_n16_single | einsum | eager-ad | c64 | 16×16 | col_major_contiguous | single | 4 | 29014.97 | 1656.91 | 7.4% | — | ok |
| einsum_f64_compiled-setup_n2_single | einsum | compiled-setup | f64 | 2×2 | col_major_contiguous | single | 4 | 2477.05 | 192.79 | 5.9% | — | ok |
| einsum_f64_compiled-setup_n4_single | einsum | compiled-setup | f64 | 4×4 | col_major_contiguous | single | 4 | 2682.37 | 303.92 | 12.1% | — | NOISY |
| einsum_f64_compiled-setup_n16_single | einsum | compiled-setup | f64 | 16×16 | col_major_contiguous | single | 4 | 2762.29 | 350.67 | 8.4% | — | ok |
| einsum_c64_compiled-setup_n2_single | einsum | compiled-setup | c64 | 2×2 | col_major_contiguous | single | 4 | 2646.08 | 169.07 | 8.6% | — | ok |
| einsum_c64_compiled-setup_n4_single | einsum | compiled-setup | c64 | 4×4 | col_major_contiguous | single | 4 | 2517.58 | 249.23 | 8.8% | — | ok |
| einsum_c64_compiled-setup_n16_single | einsum | compiled-setup | c64 | 16×16 | col_major_contiguous | single | 4 | 2584.47 | 328.53 | 9.8% | — | ok |
| solve_f64_concrete-fresh_n2_single | solve | concrete-fresh | f64 | 2×2 | col_major_contiguous | single | 4 | 14843.75 | 2041.99 | 7.9% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | f64 | 2×2 | col_major_contiguous | single | 4 | 13524.73 | 2157.71 | 11.5% | — | NOISY |
| solve_f64_concrete-fresh_n4_single | solve | concrete-fresh | f64 | 4×4 | col_major_contiguous | single | 4 | 13930.66 | 1411.62 | 8.4% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | f64 | 4×4 | col_major_contiguous | single | 4 | 12422.20 | 1362.63 | 10.6% | — | NOISY |
| solve_f64_concrete-fresh_n16_single | solve | concrete-fresh | f64 | 16×16 | col_major_contiguous | single | 4 | 13253.91 | 899.41 | 9.1% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | f64 | 16×16 | col_major_contiguous | single | 4 | 12778.97 | 1459.96 | 6.6% | — | ok |
| einsum_f64_borrowed-fresh_n2_broadcast_single | einsum | borrowed-fresh | f64 | 2×2 | broadcast | single | 4 | 22234.38 | 3761.06 | 11.4% | — | NOISY |
| einsum_f64_borrowed-shared_n2_broadcast_single | einsum | borrowed-shared | f64 | 2×2 | broadcast | single | 4 | 21651.03 | 3468.10 | 9.6% | — | ok |
| einsum_f64_borrowed-fresh_n4_broadcast_single | einsum | borrowed-fresh | f64 | 4×4 | broadcast | single | 4 | 21674.48 | 3381.51 | 9.6% | — | ok |
| einsum_f64_borrowed-shared_n4_broadcast_single | einsum | borrowed-shared | f64 | 4×4 | broadcast | single | 4 | 21963.55 | 1918.29 | 9.7% | — | ok |
| einsum_f64_borrowed-fresh_n16_broadcast_single | einsum | borrowed-fresh | f64 | 16×16 | broadcast | single | 4 | 23986.98 | 2171.22 | 10.7% | — | NOISY |
| einsum_f64_borrowed-shared_n16_broadcast_single | einsum | borrowed-shared | f64 | 16×16 | broadcast | single | 4 | 23639.31 | 2286.14 | 6.5% | — | ok |
| einsum_f64_concrete-fresh_n8_single | einsum | concrete-fresh | f64 | 8×8 | col_major_contiguous | single | 4 | 19733.08 | 1519.54 | 7.8% | — | ok |
| einsum_f64_concrete-shared_n8_single | einsum | concrete-shared | f64 | 8×8 | col_major_contiguous | single | 4 | 20113.94 | 3773.11 | 12.7% | — | NOISY |
| einsum_f64_prepared-setup_n8_single | einsum | prepared-setup | f64 | 8×8 | col_major_contiguous | single | 4 | 2648.76 | 96.48 | 4.8% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | f64 | 8×8 | col_major_contiguous | single | 4 | 11817.71 | 930.01 | 7.7% | — | ok |
| einsum_f64_concrete-fresh_n32_single | einsum | concrete-fresh | f64 | 32×32 | col_major_contiguous | single | 4 | 18763.02 | 1624.98 | 12.3% | — | NOISY |
| einsum_f64_concrete-shared_n32_single | einsum | concrete-shared | f64 | 32×32 | col_major_contiguous | single | 4 | 19279.30 | 4404.30 | 12.6% | — | NOISY |
| einsum_f64_prepared-setup_n32_single | einsum | prepared-setup | f64 | 32×32 | col_major_contiguous | single | 4 | 2668.86 | 346.92 | 7.9% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | f64 | 32×32 | col_major_contiguous | single | 4 | 13470.38 | 827.14 | 4.8% | — | ok |
| einsum_c64_concrete-fresh_n8_single | einsum | concrete-fresh | c64 | 8×8 | col_major_contiguous | single | 4 | 17871.09 | 947.92 | 6.0% | — | ok |
| einsum_c64_concrete-shared_n8_single | einsum | concrete-shared | c64 | 8×8 | col_major_contiguous | single | 4 | 17490.88 | 613.28 | 4.4% | — | ok |
| einsum_c64_prepared-setup_n8_single | einsum | prepared-setup | c64 | 8×8 | col_major_contiguous | single | 4 | 2660.81 | 83.09 | 2.0% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | c64 | 8×8 | col_major_contiguous | single | 4 | 12233.72 | 748.21 | 5.0% | — | ok |
| einsum_c64_concrete-fresh_n32_single | einsum | concrete-fresh | c64 | 32×32 | col_major_contiguous | single | 4 | 22651.70 | 1547.84 | 3.8% | — | ok |
| einsum_c64_concrete-shared_n32_single | einsum | concrete-shared | c64 | 32×32 | col_major_contiguous | single | 4 | 22220.70 | 566.41 | 5.3% | — | ok |
| einsum_c64_prepared-setup_n32_single | einsum | prepared-setup | c64 | 32×32 | col_major_contiguous | single | 4 | 2670.65 | 35.04 | 0.9% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | c64 | 32×32 | col_major_contiguous | single | 4 | 17712.89 | 583.99 | 5.4% | — | ok |
| gather_f64_concrete_fresh | gather | concrete-fresh | f64 | 4 | col_major_contiguous | single | 4 | 8116.86 | 355.30 | 6.6% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | f64 | 4 | col_major_contiguous | single | 4 | 7673.51 | 212.24 | 9.6% | — | ok |
| gather_f64_concrete-fresh_size16_single | gather | concrete-fresh | f64 | 16 | col_major_contiguous | single | 4 | 8046.55 | 226.57 | 6.1% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | f64 | 16 | col_major_contiguous | single | 4 | 7631.51 | 884.20 | 5.8% | — | ok |
| gather_f64_concrete-fresh_size256_single | gather | concrete-fresh | f64 | 256 | col_major_contiguous | single | 4 | 8449.88 | 421.38 | 7.6% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | f64 | 256 | col_major_contiguous | single | 4 | 7814.29 | 323.89 | 6.1% | — | ok |
| einsum_c64_borrowed-fresh_n2_col_major_contiguous_single | einsum | borrowed-fresh | c64 | 2×2 | col_major_contiguous | single | 4 | 17619.80 | 995.12 | 7.7% | — | ok |
| einsum_c64_borrowed-fresh_n2_row_major_contiguous_single | einsum | borrowed-fresh | c64 | 2×2 | row_major_contiguous | single | 4 | 17867.83 | 902.67 | 7.9% | — | ok |
| einsum_c64_borrowed-fresh_n2_strided_single | einsum | borrowed-fresh | c64 | 2×2 | strided | single | 4 | 20984.38 | 1184.89 | 5.7% | — | ok |
| einsum_c64_borrowed-shared_n2_col_major_contiguous_single | einsum | borrowed-shared | c64 | 2×2 | col_major_contiguous | single | 4 | 17755.20 | 586.26 | 6.8% | — | ok |
| einsum_c64_borrowed-shared_n2_row_major_contiguous_single | einsum | borrowed-shared | c64 | 2×2 | row_major_contiguous | single | 4 | 17520.19 | 1563.48 | 6.3% | — | ok |
| einsum_c64_borrowed-shared_n2_strided_single | einsum | borrowed-shared | c64 | 2×2 | strided | single | 4 | 20364.58 | 1085.94 | 7.6% | — | ok |
| einsum_c64_borrowed-fresh_n4_col_major_contiguous_single | einsum | borrowed-fresh | c64 | 4×4 | col_major_contiguous | single | 4 | 17767.58 | 999.02 | 7.4% | — | ok |
| einsum_c64_borrowed-fresh_n4_row_major_contiguous_single | einsum | borrowed-fresh | c64 | 4×4 | row_major_contiguous | single | 4 | 18079.44 | 640.95 | 7.5% | — | ok |
| einsum_c64_borrowed-fresh_n4_strided_single | einsum | borrowed-fresh | c64 | 4×4 | strided | single | 4 | 20923.83 | 589.19 | 4.1% | — | ok |
| einsum_c64_borrowed-shared_n4_col_major_contiguous_single | einsum | borrowed-shared | c64 | 4×4 | col_major_contiguous | single | 4 | 18283.86 | 2187.50 | 9.6% | — | ok |
| einsum_c64_borrowed-shared_n4_row_major_contiguous_single | einsum | borrowed-shared | c64 | 4×4 | row_major_contiguous | single | 4 | 17285.80 | 1198.24 | 6.6% | — | ok |
| einsum_c64_borrowed-shared_n4_strided_single | einsum | borrowed-shared | c64 | 4×4 | strided | single | 4 | 20507.81 | 612.96 | 7.9% | — | ok |
| einsum_c64_borrowed-fresh_n16_col_major_contiguous_single | einsum | borrowed-fresh | c64 | 16×16 | col_major_contiguous | single | 4 | 19839.84 | 876.95 | 4.8% | — | ok |
| einsum_c64_borrowed-fresh_n16_row_major_contiguous_single | einsum | borrowed-fresh | c64 | 16×16 | row_major_contiguous | single | 4 | 20051.44 | 608.07 | 6.3% | — | ok |
| einsum_c64_borrowed-fresh_n16_strided_single | einsum | borrowed-fresh | c64 | 16×16 | strided | single | 4 | 23942.70 | 1035.48 | 3.9% | — | ok |
| einsum_c64_borrowed-shared_n16_col_major_contiguous_single | einsum | borrowed-shared | c64 | 16×16 | col_major_contiguous | single | 4 | 19757.16 | 1051.44 | 6.9% | — | ok |
| einsum_c64_borrowed-shared_n16_row_major_contiguous_single | einsum | borrowed-shared | c64 | 16×16 | row_major_contiguous | single | 4 | 19638.67 | 997.08 | 5.9% | — | ok |
| einsum_c64_borrowed-shared_n16_strided_single | einsum | borrowed-shared | c64 | 16×16 | strided | single | 4 | 23488.94 | 1355.15 | 5.7% | — | ok |
| einsum_c64_borrowed-fresh_n2_broadcast_single | einsum | borrowed-fresh | c64 | 2×2 | broadcast | single | 4 | 20854.81 | 885.75 | 4.4% | — | ok |
| einsum_c64_borrowed-shared_n2_broadcast_single | einsum | borrowed-shared | c64 | 2×2 | broadcast | single | 4 | 20435.55 | 1597.98 | 7.5% | — | ok |
| einsum_c64_borrowed-fresh_n4_broadcast_single | einsum | borrowed-fresh | c64 | 4×4 | broadcast | single | 4 | 21373.05 | 1855.80 | 9.4% | — | ok |
| einsum_c64_borrowed-shared_n4_broadcast_single | einsum | borrowed-shared | c64 | 4×4 | broadcast | single | 4 | 20208.33 | 1967.78 | 9.1% | — | ok |
| einsum_c64_borrowed-fresh_n16_broadcast_single | einsum | borrowed-fresh | c64 | 16×16 | broadcast | single | 4 | 23701.83 | 1348.95 | 6.1% | — | ok |
| einsum_c64_borrowed-shared_n16_broadcast_single | einsum | borrowed-shared | c64 | 16×16 | broadcast | single | 4 | 23777.34 | 1803.39 | 8.0% | — | ok |
| reduce_sum_f64_concrete-fresh_size4_single | reduce_sum | concrete-fresh | f64 | 4 | col_major_contiguous | single | 4 | 8676.76 | 1525.55 | 12.8% | — | NOISY |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | f64 | 4 | col_major_contiguous | single | 4 | 7929.69 | 1486.00 | 15.3% | — | NOISY |
| reduce_sum_f64_concrete-fresh_size16_single | reduce_sum | concrete-fresh | f64 | 16 | col_major_contiguous | single | 4 | 9037.60 | 1243.16 | 10.4% | — | NOISY |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | f64 | 16 | col_major_contiguous | single | 4 | 7795.25 | 592.12 | 18.6% | — | NOISY |
| reduce_sum_f64_concrete-fresh_size256_single | reduce_sum | concrete-fresh | f64 | 256 | col_major_contiguous | single | 4 | 8189.12 | 1240.23 | 15.4% | — | NOISY |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | f64 | 256 | col_major_contiguous | single | 4 | 8356.12 | 2176.43 | 20.9% | — | NOISY |

## Per-route timing boundaries

Shared-session routes enter/exit the session once around warmup/calibration/all samples, outside timing. Fresh routes enter/exit for each operation. Prepared-repeat excludes plan preparation; compiled-repeat excludes tracing/compilation and runtime construction.

- **add / concrete-fresh** — inside: session_entry_exit, add, output_lifetime; outside: backend_construction, input_construction, correctness_check.
- **add / concrete-shared** — inside: add, output_lifetime; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check.
- **add / eager-ad** — inside: add_forward_recording, eager_dispatch, output_lifetime; outside: backend_construction, input_construction, eager_runtime_construction, backward, correctness_check.
- **add / eager-no-ad** — inside: add, eager_dispatch, output_lifetime; outside: backend_construction, input_construction, eager_runtime_construction, correctness_check.
- **einsum / borrowed-fresh** — inside: session_entry_exit, einsum, output_lifetime; outside: backend_construction, input_construction, correctness_check.
- **einsum / borrowed-shared** — inside: einsum, output_lifetime; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check.
- **einsum / compiled-repeat** — inside: runtime_admission_execution, output_lifetime; outside: backend_construction, input_construction, trace_compile, runtime_construction, input_bindings, correctness_check.
- **einsum / compiled-setup** — inside: trace_compile, program_lifetime; outside: backend_construction, input_construction, runtime_construction, input_bindings, correctness_check.
- **einsum / concrete-fresh** — inside: session_entry_exit, einsum, output_lifetime; outside: backend_construction, input_construction, correctness_check.
- **einsum / concrete-shared** — inside: einsum, output_lifetime; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check.
- **einsum / eager-ad** — inside: einsum_forward_recording, eager_dispatch, output_lifetime; outside: backend_construction, input_construction, eager_runtime_construction, backward, correctness_check.
- **einsum / eager-no-ad** — inside: einsum, eager_dispatch, output_lifetime; outside: backend_construction, input_construction, eager_runtime_construction, correctness_check.
- **einsum / prepared-repeat** — inside: prepared_execute, output_lifetime; outside: backend_construction, input_construction, plan_preparation, session_entry_exit, shared_admission_exit, correctness_check.
- **einsum / prepared-setup** — inside: prepare, plan_lifetime; outside: backend_construction, input_construction, correctness_check.
- **gather / concrete-fresh** — inside: index_config_construction, session_entry_exit, gather, output_lifetime; outside: backend_construction, input_construction, correctness_check.
- **gather / concrete-shared** — inside: index_config_construction, gather, output_lifetime; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check.
- **reduce_sum / concrete-fresh** — inside: session_entry_exit, reduce_sum, output_lifetime; outside: backend_construction, input_construction, correctness_check.
- **reduce_sum / concrete-shared** — inside: reduce_sum, output_lifetime; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check.
- **solve / concrete-fresh** — inside: session_entry_exit, solve, output_lifetime; outside: backend_construction, input_construction, correctness_check.
- **solve / concrete-shared** — inside: solve, output_lifetime; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check.

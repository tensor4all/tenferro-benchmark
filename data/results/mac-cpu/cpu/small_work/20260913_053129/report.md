# CPU Small-Work Benchmark Results

- Suite: `cpu/small_work`
- Raw run: `data/results/mac-cpu/cpu/small_work/20260913_053129`

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

Full metadata: `data/results/mac-cpu/cpu/small_work/20260913_053129/run_t1.yaml`

```yaml
timestamp: '2026-09-13T05:31:29Z'
tenferro_rs:
  path: /Users/hiroshi/projects/tensor4all/tenferro-benchmark-m5-refresh/extern/tenferro-rs
  commit: a793c2e95693f053722fbff8d0db25722336c21f
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
    BENCHMARK_COMMIT: 15c25b1c003c993e0a5b4f4dc05a97a4a836a282
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.96.0 (ac68faa20 2026-05-25)
    BENCH_INSTANCE: all 154
```

## run_t4

Full metadata: `data/results/mac-cpu/cpu/small_work/20260913_053129/run_t4.yaml`

```yaml
timestamp: '2026-09-13T05:31:33Z'
tenferro_rs:
  path: /Users/hiroshi/projects/tensor4all/tenferro-benchmark-m5-refresh/extern/tenferro-rs
  commit: a793c2e95693f053722fbff8d0db25722336c21f
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
    BENCHMARK_COMMIT: 15c25b1c003c993e0a5b4f4dc05a97a4a836a282
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.96.0 (ac68faa20 2026-05-25)
    BENCH_INSTANCE: all 154
```

Numerical checks: **110/110 recorded rows passed** (case × thread count).

## Timing

Sequential release runs; 3 warmups, then calibration to a 1 ms batch and 15 measured batches per case. Statistics describe the sum of per-call execution intervals / iterations. Clock overhead is included. Median and inclusive IQR are in nanoseconds (ns); CoV is sample standard deviation / mean. NOISY means CoV > 10%, a descriptive label only. No noisy rows are excluded.

Workflow time is the total for one call or the complete dependent chain. The separate ns/op column divides the chain median by its operation count, not an independently measured call. Setup rows measure preparation only, not execution.

Inputs and backend construction, independent numerical checks and AD backward checks are outside timing. Returned outputs/plans are retained until after each execution interval; value downloads/checks are outside. Eager AD rows measure forward execution/recording, not backward.

| Case ID | Operation | API route | Dtype | Shape | Layout | Workflow | Threads | Median workflow ns | IQR ns | CoV | ns/op (chain only) | Status |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| add_f64_concrete-shared_size4_single | add | concrete-shared | f64 | 4 | col_major_contiguous | single | 1 | 1523.46 | 122.30 | 7.4% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | f64 | 4 | col_major_contiguous | dependent10 | 1 | 12181.66 | 904.81 | 5.0% | 1218.17 | ok |
| add_f64_concrete_shared | add | concrete-shared | f64 | 16 | col_major_contiguous | single | 1 | 1067.12 | 29.26 | 1.8% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | f64 | 16 | col_major_contiguous | dependent10 | 1 | 10246.04 | 298.35 | 1.5% | 1024.60 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | f64 | 256 | col_major_contiguous | single | 1 | 1042.65 | 17.42 | 1.5% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | f64 | 256 | col_major_contiguous | dependent10 | 1 | 10551.74 | 123.18 | 1.1% | 1055.17 | ok |
| add_f64_eager-no-ad_size4_single | add | eager-no-ad | f64 | 4 | col_major_contiguous | single | 1 | 3797.71 | 60.31 | 1.0% | — | ok |
| add_f64_eager-no-ad_size4_dependent10 | add | eager-no-ad | f64 | 4 | col_major_contiguous | dependent10 | 1 | 37815.12 | 632.22 | 1.8% | 3781.51 | ok |
| add_f64_eager_no_ad | add | eager-no-ad | f64 | 16 | col_major_contiguous | single | 1 | 3841.37 | 48.91 | 0.9% | — | ok |
| add_f64_eager-no-ad_size16_dependent10 | add | eager-no-ad | f64 | 16 | col_major_contiguous | dependent10 | 1 | 38479.19 | 604.81 | 1.5% | 3847.92 | ok |
| add_f64_eager-no-ad_size256_single | add | eager-no-ad | f64 | 256 | col_major_contiguous | single | 1 | 3926.50 | 69.53 | 1.2% | — | ok |
| add_f64_eager-no-ad_size256_dependent10 | add | eager-no-ad | f64 | 256 | col_major_contiguous | dependent10 | 1 | 39350.16 | 658.09 | 1.4% | 3935.02 | ok |
| add_f64_eager_active_ad | add | eager-ad | f64 | 4 | col_major_contiguous | single | 1 | 3385.43 | 125.81 | 4.5% | — | ok |
| add_f64_eager-ad_size4_dependent10 | add | eager-ad | f64 | 4 | col_major_contiguous | dependent10 | 1 | 33489.66 | 1235.70 | 3.4% | 3348.97 | ok |
| add_f64_eager-ad_size16_single | add | eager-ad | f64 | 16 | col_major_contiguous | single | 1 | 3315.93 | 90.13 | 4.2% | — | ok |
| add_f64_eager-ad_size16_dependent10 | add | eager-ad | f64 | 16 | col_major_contiguous | dependent10 | 1 | 32128.97 | 1176.45 | 3.9% | 3212.90 | ok |
| add_f64_eager-ad_size256_single | add | eager-ad | f64 | 256 | col_major_contiguous | single | 1 | 3488.61 | 103.40 | 4.6% | — | ok |
| add_f64_eager-ad_size256_dependent10 | add | eager-ad | f64 | 256 | col_major_contiguous | dependent10 | 1 | 34104.28 | 1171.81 | 4.6% | 3410.43 | ok |
| einsum_f64_eager-no-ad_n2_single | einsum | eager-no-ad | f64 | 2×2 | col_major_contiguous | single | 1 | 4436.48 | 74.46 | 1.5% | — | ok |
| einsum_f64_eager-no-ad_n4_single | einsum | eager-no-ad | f64 | 4×4 | col_major_contiguous | single | 1 | 4492.52 | 113.41 | 1.8% | — | ok |
| einsum_f64_eager-no-ad_n16_single | einsum | eager-no-ad | f64 | 16×16 | col_major_contiguous | single | 1 | 6175.94 | 145.75 | 1.9% | — | ok |
| einsum_f64_eager-ad_n2_single | einsum | eager-ad | f64 | 2×2 | col_major_contiguous | single | 1 | 4337.38 | 135.74 | 3.7% | — | ok |
| einsum_f64_eager-ad_n4_single | einsum | eager-ad | f64 | 4×4 | col_major_contiguous | single | 1 | 4392.12 | 87.47 | 3.2% | — | ok |
| einsum_f64_eager-ad_n16_single | einsum | eager-ad | f64 | 16×16 | col_major_contiguous | single | 1 | 5961.91 | 250.24 | 2.9% | — | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | f64 | 2×2 | col_major_contiguous | single | 1 | 2579.11 | 67.21 | 1.6% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | f64 | 4×4 | col_major_contiguous | single | 1 | 2594.61 | 47.83 | 1.6% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | f64 | 16×16 | col_major_contiguous | single | 1 | 3613.72 | 63.12 | 1.4% | — | ok |
| einsum_f64_compiled-repeat_n2_single | einsum | compiled-repeat | f64 | 2×2 | col_major_contiguous | single | 1 | 14128.91 | 759.00 | 3.5% | — | ok |
| einsum_f64_compiled-repeat_n4_single | einsum | compiled-repeat | f64 | 4×4 | col_major_contiguous | single | 1 | 13914.34 | 501.84 | 2.3% | — | ok |
| einsum_f64_compiled-repeat_n16_single | einsum | compiled-repeat | f64 | 16×16 | col_major_contiguous | single | 1 | 16539.16 | 890.29 | 4.3% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | c64 | 2×2 | col_major_contiguous | single | 1 | 2635.27 | 58.39 | 1.8% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | c64 | 4×4 | col_major_contiguous | single | 1 | 2756.50 | 77.56 | 1.9% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | c64 | 16×16 | col_major_contiguous | single | 1 | 5253.42 | 123.44 | 1.6% | — | ok |
| einsum_c64_eager-no-ad_n2_single | einsum | eager-no-ad | c64 | 2×2 | col_major_contiguous | single | 1 | 4514.15 | 93.37 | 1.7% | — | ok |
| einsum_c64_compiled-repeat_n2_single | einsum | compiled-repeat | c64 | 2×2 | col_major_contiguous | single | 1 | 14207.05 | 278.69 | 2.5% | — | ok |
| einsum_c64_eager-no-ad_n4_single | einsum | eager-no-ad | c64 | 4×4 | col_major_contiguous | single | 1 | 4577.31 | 64.27 | 1.4% | — | ok |
| einsum_c64_compiled-repeat_n4_single | einsum | compiled-repeat | c64 | 4×4 | col_major_contiguous | single | 1 | 14370.75 | 368.86 | 2.3% | — | ok |
| einsum_c64_eager-no-ad_n16_single | einsum | eager-no-ad | c64 | 16×16 | col_major_contiguous | single | 1 | 7534.48 | 93.10 | 1.4% | — | ok |
| einsum_c64_compiled-repeat_n16_single | einsum | compiled-repeat | c64 | 16×16 | col_major_contiguous | single | 1 | 17918.56 | 494.05 | 4.4% | — | ok |
| einsum_c64_eager-ad_n2_single | einsum | eager-ad | c64 | 2×2 | col_major_contiguous | single | 1 | 4376.63 | 172.19 | 3.0% | — | ok |
| einsum_c64_eager-ad_n4_single | einsum | eager-ad | c64 | 4×4 | col_major_contiguous | single | 1 | 4577.50 | 156.69 | 3.5% | — | ok |
| einsum_c64_eager-ad_n16_single | einsum | eager-ad | c64 | 16×16 | col_major_contiguous | single | 1 | 7449.89 | 180.59 | 3.1% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | f64 | 2×2 | col_major_contiguous | single | 1 | 2055.67 | 38.38 | 1.6% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | f64 | 4×4 | col_major_contiguous | single | 1 | 2135.02 | 45.03 | 1.8% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | f64 | 16×16 | col_major_contiguous | single | 1 | 3728.26 | 63.99 | 1.4% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | f64 | 8×8 | col_major_contiguous | single | 1 | 2652.89 | 60.87 | 1.9% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | f64 | 32×32 | col_major_contiguous | single | 1 | 4022.12 | 94.17 | 1.7% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | c64 | 8×8 | col_major_contiguous | single | 1 | 2976.24 | 68.24 | 3.0% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | c64 | 32×32 | col_major_contiguous | single | 1 | 9041.33 | 484.73 | 5.6% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | f64 | 4 | col_major_contiguous | single | 1 | 876.25 | 39.85 | 3.5% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | f64 | 16 | col_major_contiguous | single | 1 | 878.49 | 78.92 | 5.4% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | f64 | 256 | col_major_contiguous | single | 1 | 1021.45 | 47.94 | 3.6% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | f64 | 4 | col_major_contiguous | single | 1 | 725.62 | 78.76 | 7.8% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | f64 | 16 | col_major_contiguous | single | 1 | 671.62 | 34.32 | 5.6% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | f64 | 256 | col_major_contiguous | single | 1 | 688.04 | 48.71 | 4.2% | — | ok |
| add_f64_concrete-shared_size4_single | add | concrete-shared | f64 | 4 | col_major_contiguous | single | 4 | 10644.52 | 2357.57 | 15.4% | — | NOISY |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | f64 | 4 | col_major_contiguous | dependent10 | 4 | 106520.62 | 16420.72 | 16.6% | 10652.06 | NOISY |
| add_f64_concrete_shared | add | concrete-shared | f64 | 16 | col_major_contiguous | single | 4 | 9945.91 | 1151.23 | 15.3% | — | NOISY |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | f64 | 16 | col_major_contiguous | dependent10 | 4 | 92036.56 | 11778.50 | 10.2% | 9203.66 | NOISY |
| add_f64_concrete-shared_size256_single | add | concrete-shared | f64 | 256 | col_major_contiguous | single | 4 | 9402.30 | 1823.25 | 22.6% | — | NOISY |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | f64 | 256 | col_major_contiguous | dependent10 | 4 | 85981.62 | 8135.53 | 9.0% | 8598.16 | ok |
| add_f64_eager-no-ad_size4_single | add | eager-no-ad | f64 | 4 | col_major_contiguous | single | 4 | 27725.30 | 1254.79 | 8.3% | — | ok |
| add_f64_eager-no-ad_size4_dependent10 | add | eager-no-ad | f64 | 4 | col_major_contiguous | dependent10 | 4 | 277979.25 | 21562.38 | 15.2% | 27797.92 | NOISY |
| add_f64_eager_no_ad | add | eager-no-ad | f64 | 16 | col_major_contiguous | single | 4 | 27259.19 | 2292.30 | 8.1% | — | ok |
| add_f64_eager-no-ad_size16_dependent10 | add | eager-no-ad | f64 | 16 | col_major_contiguous | dependent10 | 4 | 269302.00 | 20062.25 | 5.2% | 26930.20 | ok |
| add_f64_eager-no-ad_size256_single | add | eager-no-ad | f64 | 256 | col_major_contiguous | single | 4 | 27471.95 | 1009.16 | 4.5% | — | ok |
| add_f64_eager-no-ad_size256_dependent10 | add | eager-no-ad | f64 | 256 | col_major_contiguous | dependent10 | 4 | 271500.25 | 17062.50 | 7.7% | 27150.03 | ok |
| add_f64_eager_active_ad | add | eager-ad | f64 | 4 | col_major_contiguous | single | 4 | 14450.13 | 690.71 | 4.1% | — | ok |
| add_f64_eager-ad_size4_dependent10 | add | eager-ad | f64 | 4 | col_major_contiguous | dependent10 | 4 | 136020.75 | 2973.81 | 3.6% | 13602.08 | ok |
| add_f64_eager-ad_size16_single | add | eager-ad | f64 | 16 | col_major_contiguous | single | 4 | 14141.97 | 475.09 | 3.7% | — | ok |
| add_f64_eager-ad_size16_dependent10 | add | eager-ad | f64 | 16 | col_major_contiguous | dependent10 | 4 | 142109.25 | 5760.56 | 3.9% | 14210.92 | ok |
| add_f64_eager-ad_size256_single | add | eager-ad | f64 | 256 | col_major_contiguous | single | 4 | 14419.98 | 695.02 | 4.0% | — | ok |
| add_f64_eager-ad_size256_dependent10 | add | eager-ad | f64 | 256 | col_major_contiguous | dependent10 | 4 | 140901.12 | 5054.69 | 5.4% | 14090.11 | ok |
| einsum_f64_eager-no-ad_n2_single | einsum | eager-no-ad | f64 | 2×2 | col_major_contiguous | single | 4 | 28022.14 | 940.14 | 3.1% | — | ok |
| einsum_f64_eager-no-ad_n4_single | einsum | eager-no-ad | f64 | 4×4 | col_major_contiguous | single | 4 | 28244.77 | 784.50 | 3.0% | — | ok |
| einsum_f64_eager-no-ad_n16_single | einsum | eager-no-ad | f64 | 16×16 | col_major_contiguous | single | 4 | 30065.09 | 1266.75 | 3.0% | — | ok |
| einsum_f64_eager-ad_n2_single | einsum | eager-ad | f64 | 2×2 | col_major_contiguous | single | 4 | 16285.12 | 559.30 | 2.9% | — | ok |
| einsum_f64_eager-ad_n4_single | einsum | eager-ad | f64 | 4×4 | col_major_contiguous | single | 4 | 16333.30 | 726.55 | 2.9% | — | ok |
| einsum_f64_eager-ad_n16_single | einsum | eager-ad | f64 | 16×16 | col_major_contiguous | single | 4 | 16857.47 | 942.12 | 4.6% | — | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | f64 | 2×2 | col_major_contiguous | single | 4 | 11738.20 | 486.59 | 4.3% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | f64 | 4×4 | col_major_contiguous | single | 4 | 12025.39 | 827.48 | 7.0% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | f64 | 16×16 | col_major_contiguous | single | 4 | 12794.58 | 829.92 | 5.0% | — | ok |
| einsum_f64_compiled-repeat_n2_single | einsum | compiled-repeat | f64 | 2×2 | col_major_contiguous | single | 4 | 30246.78 | 1177.05 | 2.5% | — | ok |
| einsum_f64_compiled-repeat_n4_single | einsum | compiled-repeat | f64 | 4×4 | col_major_contiguous | single | 4 | 29272.83 | 828.51 | 2.2% | — | ok |
| einsum_f64_compiled-repeat_n16_single | einsum | compiled-repeat | f64 | 16×16 | col_major_contiguous | single | 4 | 27439.53 | 2094.43 | 5.4% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | c64 | 2×2 | col_major_contiguous | single | 4 | 12032.26 | 693.89 | 12.2% | — | NOISY |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | c64 | 4×4 | col_major_contiguous | single | 4 | 12129.57 | 305.18 | 4.1% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | c64 | 16×16 | col_major_contiguous | single | 4 | 14446.61 | 355.41 | 4.3% | — | ok |
| einsum_c64_eager-no-ad_n2_single | einsum | eager-no-ad | c64 | 2×2 | col_major_contiguous | single | 4 | 28500.61 | 945.27 | 8.6% | — | ok |
| einsum_c64_compiled-repeat_n2_single | einsum | compiled-repeat | c64 | 2×2 | col_major_contiguous | single | 4 | 28530.00 | 1524.69 | 9.1% | — | ok |
| einsum_c64_eager-no-ad_n4_single | einsum | eager-no-ad | c64 | 4×4 | col_major_contiguous | single | 4 | 30795.55 | 3099.95 | 7.0% | — | ok |
| einsum_c64_compiled-repeat_n4_single | einsum | compiled-repeat | c64 | 4×4 | col_major_contiguous | single | 4 | 28420.59 | 2685.86 | 7.2% | — | ok |
| einsum_c64_eager-no-ad_n16_single | einsum | eager-no-ad | c64 | 16×16 | col_major_contiguous | single | 4 | 33505.25 | 4585.30 | 12.4% | — | NOISY |
| einsum_c64_compiled-repeat_n16_single | einsum | compiled-repeat | c64 | 16×16 | col_major_contiguous | single | 4 | 32826.84 | 1289.09 | 8.6% | — | ok |
| einsum_c64_eager-ad_n2_single | einsum | eager-ad | c64 | 2×2 | col_major_contiguous | single | 4 | 16465.47 | 1268.52 | 9.2% | — | ok |
| einsum_c64_eager-ad_n4_single | einsum | eager-ad | c64 | 4×4 | col_major_contiguous | single | 4 | 17115.23 | 1956.02 | 13.0% | — | NOISY |
| einsum_c64_eager-ad_n16_single | einsum | eager-ad | c64 | 16×16 | col_major_contiguous | single | 4 | 19994.05 | 1048.83 | 13.0% | — | NOISY |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | f64 | 2×2 | col_major_contiguous | single | 4 | 9853.17 | 1241.02 | 9.1% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | f64 | 4×4 | col_major_contiguous | single | 4 | 9172.20 | 626.34 | 8.3% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | f64 | 16×16 | col_major_contiguous | single | 4 | 11217.12 | 646.14 | 12.3% | — | NOISY |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | f64 | 8×8 | col_major_contiguous | single | 4 | 12422.87 | 689.44 | 9.5% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | f64 | 32×32 | col_major_contiguous | single | 4 | 14022.75 | 2435.71 | 12.1% | — | NOISY |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | c64 | 8×8 | col_major_contiguous | single | 4 | 12987.59 | 2101.92 | 11.1% | — | NOISY |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | c64 | 32×32 | col_major_contiguous | single | 4 | 18559.19 | 993.15 | 6.7% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | f64 | 4 | col_major_contiguous | single | 4 | 7927.25 | 732.84 | 7.3% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | f64 | 16 | col_major_contiguous | single | 4 | 7770.37 | 1243.41 | 8.9% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | f64 | 256 | col_major_contiguous | single | 4 | 7937.50 | 929.43 | 10.4% | — | NOISY |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | f64 | 4 | col_major_contiguous | single | 4 | 8597.16 | 1436.89 | 13.7% | — | NOISY |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | f64 | 16 | col_major_contiguous | single | 4 | 7660.95 | 1254.08 | 8.9% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | f64 | 256 | col_major_contiguous | single | 4 | 7736.15 | 844.96 | 7.0% | — | ok |

## Per-route timing boundaries

Shared-session routes enter/exit the session once around warmup/calibration/all samples, outside timing. Fresh routes enter/exit for each operation. Prepared-repeat excludes plan preparation; compiled-repeat excludes tracing/compilation and runtime construction.

- **add / concrete-shared** — inside: add, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **add / eager-ad** — inside: add_forward_recording, eager_dispatch, output_allocation; outside: backend_construction, input_construction, eager_runtime_construction, backward, correctness_check, output_destruction, operation_config_construction.
- **add / eager-no-ad** — inside: add, eager_dispatch, output_allocation; outside: backend_construction, input_construction, eager_runtime_construction, correctness_check, output_destruction, operation_config_construction.
- **einsum / compiled-repeat** — inside: runtime_admission_execution, output_allocation; outside: backend_construction, input_construction, trace_compile, runtime_construction, input_bindings, correctness_check, output_destruction, operation_config_construction.
- **einsum / eager-ad** — inside: einsum_forward_recording, eager_dispatch, output_allocation; outside: backend_construction, input_construction, eager_runtime_construction, backward, correctness_check, output_destruction, operation_config_construction.
- **einsum / eager-no-ad** — inside: einsum, eager_dispatch, output_allocation; outside: backend_construction, input_construction, eager_runtime_construction, correctness_check, output_destruction, operation_config_construction.
- **einsum / prepared-repeat** — inside: prepared_execute, output_allocation; outside: backend_construction, input_construction, plan_preparation, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **gather / concrete-shared** — inside: gather, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **reduce_sum / concrete-shared** — inside: reduce_sum, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **solve / concrete-shared** — inside: solve, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.

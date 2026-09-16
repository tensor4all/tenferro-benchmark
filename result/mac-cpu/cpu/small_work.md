# CPU Small-Work Benchmark Results

- Suite: `cpu/small_work`
- Raw run: `data/results/mac-cpu/cpu/small_work/20260916_014938`

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

Full metadata: `data/results/mac-cpu/cpu/small_work/20260916_014938/run_t1.yaml`

```yaml
timestamp: '2026-09-16T01:49:38Z'
tenferro_rs:
  path: /Users/hiroshi/projects/tensor4all/tenferro-benchmark-m5-refresh/extern/tenferro-rs
  commit: bca2d54a5884586b1aeae8006caa9b8c0ba44a26
  dirty: false
  features:
  - system-accelerate
blas:
  implementation: accelerate
  version: unknown
  library: /System/Library/Frameworks/Accelerate.framework
environment:
  hostname: MacBook-Pro-8.local
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
    BENCHMARK_COMMIT: f983e88b9d07e558989427d047f1fa4676df2ea8
    OPENBLAS_NUM_THREADS: '1'
    GOTO_NUM_THREADS: '1'
    MKL_NUM_THREADS: '1'
    VECLIB_MAXIMUM_THREADS: '1'
    VECLIB_NUM_THREADS: '1'
    NUMEXPR_NUM_THREADS: '1'
    BLIS_NUM_THREADS: '1'
    PJRT_NPROC: '1'
    XLA_FLAGS: --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1
    JULIA_NUM_THREADS: '1'
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.96.0 (ac68faa20 2026-05-25)
    BENCH_INSTANCE: all 154
```

## run_t4

Full metadata: `data/results/mac-cpu/cpu/small_work/20260916_014938/run_t4.yaml`

```yaml
timestamp: '2026-09-16T01:49:42Z'
tenferro_rs:
  path: /Users/hiroshi/projects/tensor4all/tenferro-benchmark-m5-refresh/extern/tenferro-rs
  commit: bca2d54a5884586b1aeae8006caa9b8c0ba44a26
  dirty: false
  features:
  - system-accelerate
blas:
  implementation: accelerate
  version: unknown
  library: /System/Library/Frameworks/Accelerate.framework
environment:
  hostname: MacBook-Pro-8.local
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
    BENCHMARK_COMMIT: f983e88b9d07e558989427d047f1fa4676df2ea8
    OPENBLAS_NUM_THREADS: '4'
    GOTO_NUM_THREADS: '4'
    MKL_NUM_THREADS: '4'
    VECLIB_MAXIMUM_THREADS: '4'
    VECLIB_NUM_THREADS: '4'
    NUMEXPR_NUM_THREADS: '4'
    BLIS_NUM_THREADS: '4'
    PJRT_NPROC: '4'
    XLA_FLAGS: --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4
    JULIA_NUM_THREADS: '4'
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.96.0 (ac68faa20 2026-05-25)
    BENCH_INSTANCE: all 154
```

Numerical checks: **50/50 recorded rows passed** (case × thread count).

## Timing

Sequential release runs; 3 warmups, then calibration from at least 1024 operations toward a 1 ms batch and 15 measured batches per case. Each batch uses one wall-clock interval; all outputs are retained until it stops. Statistics divide batch time by iterations. Median and inclusive IQR are in nanoseconds (ns); CoV is sample standard deviation / mean. NOISY means CoV > 10%, a descriptive label only. No noisy rows are excluded.

Workflow time is the total for one call or the complete dependent chain. The separate ns/op column divides the chain median by its operation count, not an independently measured call. Setup rows measure preparation only, not execution.

Inputs and backend construction, independent numerical checks and AD backward checks are outside timing. Returned outputs/plans are retained until after each batch interval; value downloads/checks are outside. Eager AD rows measure forward execution/recording, not backward.

| Case ID | Operation | API route | Dtype | Shape | Layout | Workflow | Threads | Provider | Operations/batch | Median batch ms | Median workflow ns | IQR ns | CoV | ns/op (chain only) | Status |
|---|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| add_f64_concrete-shared_size4_single | add | concrete-shared | f64 | 4 | col_major_contiguous | single | 1 | blas | 1024 | 1.415958 | 1382.77 | 165.16 | 12.9% | — | NOISY |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | f64 | 4 | col_major_contiguous | dependent10 | 1 | blas | 10240 | 8.610917 | 8409.10 | 476.56 | 7.1% | 840.91 | ok |
| add_f64_concrete_shared | add | concrete-shared | f64 | 16 | col_major_contiguous | single | 1 | blas | 2048 | 1.764084 | 861.37 | 19.47 | 1.7% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | f64 | 16 | col_major_contiguous | dependent10 | 1 | blas | 10240 | 8.754375 | 8549.19 | 122.80 | 1.7% | 854.92 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | f64 | 256 | col_major_contiguous | single | 1 | blas | 1024 | 0.930250 | 908.45 | 27.51 | 2.0% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | f64 | 256 | col_major_contiguous | dependent10 | 1 | blas | 10240 | 8.953375 | 8743.53 | 75.09 | 0.8% | 874.35 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.445125 | 2387.82 | 56.62 | 1.7% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 2.450292 | 2392.86 | 32.53 | 1.4% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 3.504709 | 3422.57 | 71.98 | 1.6% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | c64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.563917 | 2503.83 | 31.25 | 1.0% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | c64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 2.565250 | 2505.13 | 69.89 | 3.0% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | c64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 5.110708 | 4990.93 | 80.93 | 1.1% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.039500 | 1991.70 | 48.08 | 1.4% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 2.126791 | 2076.94 | 35.64 | 1.6% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 3.837084 | 3747.15 | 44.17 | 1.3% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 2.511000 | 2452.15 | 31.03 | 1.3% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | f64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 3.909167 | 3817.55 | 67.95 | 1.8% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | c64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 2.816375 | 2750.37 | 50.78 | 1.2% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | c64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 9.500875 | 9278.20 | 126.67 | 1.2% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | f64 | 4 | col_major_contiguous | single | 1 | blas | 2048 | 1.641792 | 801.66 | 17.02 | 1.5% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | f64 | 16 | col_major_contiguous | single | 1 | blas | 2048 | 1.658250 | 809.69 | 14.05 | 1.2% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | f64 | 256 | col_major_contiguous | single | 1 | blas | 1024 | 1.000208 | 976.77 | 22.93 | 1.8% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | f64 | 4 | col_major_contiguous | single | 1 | blas | 2048 | 1.269250 | 619.75 | 13.55 | 1.2% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | f64 | 16 | col_major_contiguous | single | 1 | blas | 2048 | 1.256167 | 613.36 | 23.84 | 2.9% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | f64 | 256 | col_major_contiguous | single | 1 | blas | 2048 | 1.310625 | 639.95 | 10.66 | 1.6% | — | ok |
| add_f64_concrete-shared_size4_single | add | concrete-shared | f64 | 4 | col_major_contiguous | single | 4 | blas | 2048 | 1.761458 | 860.09 | 19.33 | 1.9% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | f64 | 4 | col_major_contiguous | dependent10 | 4 | blas | 10240 | 8.649458 | 8446.74 | 167.28 | 1.1% | 844.67 | ok |
| add_f64_concrete_shared | add | concrete-shared | f64 | 16 | col_major_contiguous | single | 4 | blas | 1024 | 0.887042 | 866.25 | 39.35 | 3.6% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | f64 | 16 | col_major_contiguous | dependent10 | 4 | blas | 10240 | 8.564666 | 8363.93 | 118.88 | 1.2% | 836.39 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 0.912958 | 891.56 | 9.26 | 1.6% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | f64 | 256 | col_major_contiguous | dependent10 | 4 | blas | 10240 | 8.967708 | 8757.53 | 100.73 | 0.7% | 875.75 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.505125 | 2446.41 | 58.88 | 1.4% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 2.481833 | 2423.67 | 31.13 | 1.1% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 3.668166 | 3582.19 | 119.22 | 3.0% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | c64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.598041 | 2537.15 | 19.51 | 1.1% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | c64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 2.642458 | 2580.53 | 89.46 | 1.9% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | c64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 5.099167 | 4979.66 | 65.51 | 0.8% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.017084 | 1969.81 | 26.31 | 0.8% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 2.121708 | 2071.98 | 32.51 | 1.3% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 3.806834 | 3717.61 | 35.34 | 1.0% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 2.519709 | 2460.65 | 92.43 | 2.1% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | f64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 3.963750 | 3870.85 | 64.01 | 1.6% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | c64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 2.829917 | 2763.59 | 32.21 | 1.3% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | c64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 9.629375 | 9403.69 | 123.35 | 1.1% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | f64 | 4 | col_major_contiguous | single | 4 | blas | 2048 | 1.660292 | 810.69 | 44.22 | 3.2% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | f64 | 16 | col_major_contiguous | single | 4 | blas | 2048 | 1.674709 | 817.73 | 18.57 | 1.6% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 0.989875 | 966.67 | 15.60 | 2.1% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | f64 | 4 | col_major_contiguous | single | 4 | blas | 2048 | 1.392500 | 679.93 | 8.53 | 2.0% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | f64 | 16 | col_major_contiguous | single | 4 | blas | 2048 | 1.352209 | 660.26 | 10.35 | 2.0% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | f64 | 256 | col_major_contiguous | single | 4 | blas | 2048 | 1.605000 | 783.69 | 22.60 | 2.4% | — | ok |

## Per-route timing boundaries

Shared-session routes enter/exit the session once around warmup/calibration/all samples, outside timing. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions reuse the entered executor context. Earlier BLAS revisions included per-operation executor entry; consult the recorded tenferro commit. Fresh routes enter/exit for each operation. Prepared-repeat excludes plan preparation; compiled-repeat excludes tracing/compilation and runtime construction.

- **add / concrete-shared** — inside: add, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **einsum / prepared-repeat** — inside: prepared_execute, output_allocation; outside: backend_construction, input_construction, plan_preparation, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **gather / concrete-shared** — inside: gather, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **reduce_sum / concrete-shared** — inside: reduce_sum, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **solve / concrete-shared** — inside: solve, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.

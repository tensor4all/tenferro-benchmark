# CPU Small-Work Benchmark Results

- Suite: `cpu/small_work`
- Raw run: `data/results/mac-cpu/cpu/small_work/20260923_091753`

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

Full metadata: `data/results/mac-cpu/cpu/small_work/20260923_091753/run_t1.yaml`

```yaml
timestamp: '2026-09-23T09:17:53Z'
tenferro_rs:
  path: /Users/hiroshi/projects/tensor4all/tenferro-benchmark-coverage/extern/tenferro-rs
  commit: 412852a8cfb13b14246443fc09b1c0c34f5ce1cd
  dirty: false
  features:
  - system-accelerate
blas:
  implementation: accelerate
  version: unknown
  library: /System/Library/Frameworks/Accelerate.framework
environment:
  hostname: MacBook-Pro-9.local
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
    BENCHMARK_COMMIT: 9f63fb7704feec767fda97251865741714f3926f
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

Full metadata: `data/results/mac-cpu/cpu/small_work/20260923_091753/run_t4.yaml`

```yaml
timestamp: '2026-09-23T09:17:57Z'
tenferro_rs:
  path: /Users/hiroshi/projects/tensor4all/tenferro-benchmark-coverage/extern/tenferro-rs
  commit: 412852a8cfb13b14246443fc09b1c0c34f5ce1cd
  dirty: false
  features:
  - system-accelerate
blas:
  implementation: accelerate
  version: unknown
  library: /System/Library/Frameworks/Accelerate.framework
environment:
  hostname: MacBook-Pro-9.local
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
    BENCHMARK_COMMIT: 9f63fb7704feec767fda97251865741714f3926f
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
| add_f64_concrete-shared_size4_single | add | concrete-shared | f64 | 4 | col_major_contiguous | single | 1 | blas | 1024 | 1.372541 | 1340.37 | 273.13 | 11.9% | — | NOISY |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | f64 | 4 | col_major_contiguous | dependent10 | 1 | blas | 10240 | 7.378625 | 7205.69 | 576.66 | 8.9% | 720.57 | ok |
| add_f64_concrete_shared | add | concrete-shared | f64 | 16 | col_major_contiguous | single | 1 | blas | 2048 | 1.474792 | 720.11 | 12.18 | 2.3% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | f64 | 16 | col_major_contiguous | dependent10 | 1 | blas | 10240 | 7.421792 | 7247.84 | 304.00 | 2.3% | 724.78 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | f64 | 256 | col_major_contiguous | single | 1 | blas | 2048 | 1.560209 | 761.82 | 7.72 | 0.7% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | f64 | 256 | col_major_contiguous | dependent10 | 1 | blas | 10240 | 7.658541 | 7479.04 | 183.82 | 2.9% | 747.90 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 1.804042 | 1761.76 | 44.01 | 2.0% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 1.843791 | 1800.58 | 98.06 | 4.4% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.685667 | 2622.72 | 17.37 | 0.8% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | c64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 1.922166 | 1877.12 | 39.57 | 2.1% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | c64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 1.970166 | 1923.99 | 53.63 | 2.1% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | c64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 4.260667 | 4160.81 | 40.73 | 1.4% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 1.602291 | 1564.74 | 40.71 | 1.7% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 1.675708 | 1636.43 | 62.34 | 3.5% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.985625 | 2915.65 | 37.74 | 1.4% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 1.925958 | 1880.82 | 62.19 | 1.9% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | f64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 3.001208 | 2930.87 | 72.37 | 2.7% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | c64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 2.196292 | 2144.82 | 42.18 | 1.9% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | c64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 7.593333 | 7415.36 | 107.38 | 1.9% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | f64 | 4 | col_major_contiguous | single | 1 | blas | 2048 | 1.467125 | 716.37 | 34.34 | 2.6% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | f64 | 16 | col_major_contiguous | single | 1 | blas | 2048 | 1.479791 | 722.55 | 19.09 | 4.5% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | f64 | 256 | col_major_contiguous | single | 1 | blas | 1024 | 1.008208 | 984.58 | 100.28 | 6.2% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | f64 | 4 | col_major_contiguous | single | 1 | blas | 2048 | 1.240083 | 605.51 | 6.69 | 1.7% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | f64 | 16 | col_major_contiguous | single | 1 | blas | 2048 | 1.235250 | 603.15 | 9.78 | 1.9% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | f64 | 256 | col_major_contiguous | single | 1 | blas | 2048 | 1.316000 | 642.58 | 30.93 | 3.5% | — | ok |
| add_f64_concrete-shared_size4_single | add | concrete-shared | f64 | 4 | col_major_contiguous | single | 4 | blas | 2048 | 1.477084 | 721.23 | 8.63 | 0.9% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | f64 | 4 | col_major_contiguous | dependent10 | 4 | blas | 10240 | 7.299583 | 7128.50 | 116.96 | 1.3% | 712.85 | ok |
| add_f64_concrete_shared | add | concrete-shared | f64 | 16 | col_major_contiguous | single | 4 | blas | 2048 | 1.481625 | 723.45 | 13.46 | 2.1% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | f64 | 16 | col_major_contiguous | dependent10 | 4 | blas | 10240 | 7.242708 | 7072.96 | 106.04 | 1.2% | 707.30 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | f64 | 256 | col_major_contiguous | single | 4 | blas | 2048 | 1.612416 | 787.31 | 15.18 | 2.3% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | f64 | 256 | col_major_contiguous | dependent10 | 4 | blas | 10240 | 7.496959 | 7321.25 | 69.36 | 0.6% | 732.12 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.846000 | 1802.73 | 115.30 | 3.9% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 1.826042 | 1783.24 | 45.41 | 2.0% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.738917 | 2674.72 | 71.57 | 2.1% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | c64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.945417 | 1899.82 | 60.73 | 2.0% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | c64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 1.978000 | 1931.64 | 32.00 | 1.9% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | c64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 4.286834 | 4186.36 | 73.79 | 1.1% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.602375 | 1564.82 | 27.71 | 1.4% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 1.627333 | 1589.19 | 23.66 | 1.2% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.926875 | 2858.28 | 62.13 | 2.1% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 1.925833 | 1880.70 | 39.63 | 1.7% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | f64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 2.887042 | 2819.38 | 60.59 | 2.1% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | c64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 2.221875 | 2169.80 | 93.89 | 2.7% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | c64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 7.565292 | 7387.98 | 161.25 | 1.6% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | f64 | 4 | col_major_contiguous | single | 4 | blas | 2048 | 1.506542 | 735.62 | 30.99 | 3.9% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | f64 | 16 | col_major_contiguous | single | 4 | blas | 2048 | 1.492167 | 728.60 | 9.92 | 5.4% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 0.907333 | 886.07 | 14.40 | 1.4% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | f64 | 4 | col_major_contiguous | single | 4 | blas | 2048 | 1.251417 | 611.04 | 21.09 | 2.0% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | f64 | 16 | col_major_contiguous | single | 4 | blas | 2048 | 1.290083 | 629.92 | 13.98 | 1.7% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | f64 | 256 | col_major_contiguous | single | 4 | blas | 2048 | 1.296375 | 633.00 | 11.08 | 1.4% | — | ok |

## Per-route timing boundaries

Shared-session routes enter/exit the session once around warmup/calibration/all samples, outside timing. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions reuse the entered executor context. Earlier BLAS revisions included per-operation executor entry; consult the recorded tenferro commit. Fresh routes enter/exit for each operation. Prepared-repeat excludes plan preparation; compiled-repeat excludes tracing/compilation and runtime construction.

- **add / concrete-shared** — inside: add, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **einsum / prepared-repeat** — inside: prepared_execute, output_allocation; outside: backend_construction, input_construction, plan_preparation, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **gather / concrete-shared** — inside: gather, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **reduce_sum / concrete-shared** — inside: reduce_sum, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **solve / concrete-shared** — inside: solve, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.

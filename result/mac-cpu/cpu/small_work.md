# CPU Small-Work Benchmark Results

- Suite: `cpu/small_work`
- Raw run: `data/results/mac-cpu/cpu/small_work/20260923_120455`

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

Full metadata: `data/results/mac-cpu/cpu/small_work/20260923_120455/run_t1.yaml`

```yaml
timestamp: '2026-09-23T12:04:55Z'
tenferro_rs:
  path: /Users/hiroshi/projects/tensor4all/tenferro-benchmark-coverage/extern/tenferro-rs
  commit: 8a1839febeb3c868a502e26397ca10761bbc568d
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
    BENCHMARK_COMMIT: 5df994b8aee8e926b52e885ab89a5732e4e2ff8f
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

Full metadata: `data/results/mac-cpu/cpu/small_work/20260923_120455/run_t4.yaml`

```yaml
timestamp: '2026-09-23T12:04:59Z'
tenferro_rs:
  path: /Users/hiroshi/projects/tensor4all/tenferro-benchmark-coverage/extern/tenferro-rs
  commit: 8a1839febeb3c868a502e26397ca10761bbc568d
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
    BENCHMARK_COMMIT: 5df994b8aee8e926b52e885ab89a5732e4e2ff8f
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
| add_f64_concrete-shared_size4_single | add | concrete-shared | f64 | 4 | col_major_contiguous | single | 1 | blas | 1024 | 1.254875 | 1225.46 | 183.19 | 14.3% | — | NOISY |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | f64 | 4 | col_major_contiguous | dependent10 | 1 | blas | 10240 | 7.213500 | 7044.43 | 641.40 | 7.4% | 704.44 | ok |
| add_f64_concrete_shared | add | concrete-shared | f64 | 16 | col_major_contiguous | single | 1 | blas | 2048 | 1.537250 | 750.61 | 15.65 | 2.9% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | f64 | 16 | col_major_contiguous | dependent10 | 1 | blas | 10240 | 7.316750 | 7145.26 | 125.67 | 1.6% | 714.53 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | f64 | 256 | col_major_contiguous | single | 1 | blas | 2048 | 1.546833 | 755.29 | 25.91 | 3.0% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | f64 | 256 | col_major_contiguous | dependent10 | 1 | blas | 10240 | 7.447375 | 7272.83 | 279.58 | 2.3% | 727.28 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 1.842792 | 1799.60 | 64.15 | 2.3% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 1.892667 | 1848.31 | 77.03 | 2.7% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.708334 | 2644.86 | 90.09 | 2.5% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | c64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 1.975375 | 1929.08 | 74.42 | 4.1% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | c64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 2.017125 | 1969.85 | 82.36 | 2.6% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | c64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 4.300166 | 4199.38 | 115.07 | 2.2% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 1.561792 | 1525.19 | 71.43 | 2.8% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 1.672666 | 1633.46 | 42.34 | 2.0% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 3.025458 | 2954.55 | 70.92 | 2.0% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 1.960208 | 1914.27 | 29.26 | 2.4% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | f64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 3.007750 | 2937.26 | 80.38 | 2.6% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | c64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 2.204042 | 2152.38 | 31.60 | 1.4% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | c64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 7.872667 | 7688.15 | 152.87 | 1.6% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | f64 | 4 | col_major_contiguous | single | 1 | blas | 2048 | 1.529500 | 746.83 | 11.18 | 2.5% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | f64 | 16 | col_major_contiguous | single | 1 | blas | 2048 | 1.555500 | 759.52 | 32.27 | 2.6% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | f64 | 256 | col_major_contiguous | single | 1 | blas | 1024 | 0.899541 | 878.46 | 45.53 | 3.0% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | f64 | 4 | col_major_contiguous | single | 1 | blas | 2048 | 1.275500 | 622.80 | 17.56 | 2.0% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | f64 | 16 | col_major_contiguous | single | 1 | blas | 2048 | 1.258625 | 614.56 | 12.74 | 1.4% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | f64 | 256 | col_major_contiguous | single | 1 | blas | 2048 | 1.224542 | 597.92 | 15.30 | 5.1% | — | ok |
| add_f64_concrete-shared_size4_single | add | concrete-shared | f64 | 4 | col_major_contiguous | single | 4 | blas | 2048 | 1.466708 | 716.17 | 26.08 | 2.2% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | f64 | 4 | col_major_contiguous | dependent10 | 4 | blas | 10240 | 7.358000 | 7185.55 | 135.03 | 1.3% | 718.56 | ok |
| add_f64_concrete_shared | add | concrete-shared | f64 | 16 | col_major_contiguous | single | 4 | blas | 2048 | 1.454250 | 710.08 | 42.88 | 3.4% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | f64 | 16 | col_major_contiguous | dependent10 | 4 | blas | 10240 | 7.362584 | 7190.02 | 202.94 | 3.2% | 719.00 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | f64 | 256 | col_major_contiguous | single | 4 | blas | 2048 | 1.556833 | 760.17 | 38.67 | 3.2% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | f64 | 256 | col_major_contiguous | dependent10 | 4 | blas | 10240 | 7.774208 | 7592.00 | 215.64 | 2.1% | 759.20 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.857083 | 1813.56 | 35.93 | 1.7% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 1.917208 | 1872.27 | 15.89 | 0.9% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.752333 | 2687.83 | 41.20 | 1.6% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | c64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.003000 | 1956.05 | 51.25 | 1.4% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | c64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 1.973667 | 1927.41 | 56.29 | 1.6% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | c64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 4.227417 | 4128.34 | 101.99 | 2.1% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.587542 | 1550.33 | 50.66 | 1.8% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 1.655000 | 1616.21 | 51.47 | 2.2% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.928084 | 2859.46 | 93.65 | 2.3% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 1.957000 | 1911.13 | 15.58 | 1.1% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | f64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 2.948042 | 2878.95 | 64.09 | 1.6% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | c64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 2.213083 | 2161.21 | 67.73 | 2.4% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | c64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 7.607666 | 7429.36 | 125.75 | 1.2% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | f64 | 4 | col_major_contiguous | single | 4 | blas | 2048 | 1.574333 | 768.72 | 24.60 | 1.9% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | f64 | 16 | col_major_contiguous | single | 4 | blas | 2048 | 1.529833 | 746.99 | 18.50 | 2.3% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 0.925458 | 903.77 | 12.61 | 1.3% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | f64 | 4 | col_major_contiguous | single | 4 | blas | 2048 | 1.209500 | 590.58 | 30.62 | 3.2% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | f64 | 16 | col_major_contiguous | single | 4 | blas | 2048 | 1.267417 | 618.86 | 19.72 | 2.3% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | f64 | 256 | col_major_contiguous | single | 4 | blas | 2048 | 1.293916 | 631.79 | 19.67 | 1.9% | — | ok |

## Per-route timing boundaries

Shared-session routes enter/exit the session once around warmup/calibration/all samples, outside timing. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions reuse the entered executor context. Earlier BLAS revisions included per-operation executor entry; consult the recorded tenferro commit. Fresh routes enter/exit for each operation. Prepared-repeat excludes plan preparation; compiled-repeat excludes tracing/compilation and runtime construction.

- **add / concrete-shared** — inside: add, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **einsum / prepared-repeat** — inside: prepared_execute, output_allocation; outside: backend_construction, input_construction, plan_preparation, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **gather / concrete-shared** — inside: gather, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **reduce_sum / concrete-shared** — inside: reduce_sum, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **solve / concrete-shared** — inside: solve, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.

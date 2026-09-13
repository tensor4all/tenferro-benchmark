# CPU Small-Work Benchmark Results

- Suite: `cpu/small_work`
- Raw run: `data/results/mac-cpu/cpu/small_work/20260913_094526`

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

Full metadata: `data/results/mac-cpu/cpu/small_work/20260913_094526/run_t1.yaml`

```yaml
timestamp: '2026-09-13T09:45:26Z'
tenferro_rs:
  path: /Users/hiroshi/projects/tensor4all/tenferro-benchmark-m5-refresh/extern/tenferro-rs
  commit: 28dfc7e383a653e1461d38b73ced3d563b4efdad
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
    BENCHMARK_COMMIT: 4d4b94aca2e2bdc19b605180e4ce524dc67f1df0
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

Full metadata: `data/results/mac-cpu/cpu/small_work/20260913_094526/run_t4.yaml`

```yaml
timestamp: '2026-09-13T09:45:30Z'
tenferro_rs:
  path: /Users/hiroshi/projects/tensor4all/tenferro-benchmark-m5-refresh/extern/tenferro-rs
  commit: 28dfc7e383a653e1461d38b73ced3d563b4efdad
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
    BENCHMARK_COMMIT: 4d4b94aca2e2bdc19b605180e4ce524dc67f1df0
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
| add_f64_concrete-shared_size4_single | add | concrete-shared | f64 | 4 | col_major_contiguous | single | 1 | blas | 1024 | 1.379792 | 1347.45 | 207.60 | 11.1% | — | NOISY |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | f64 | 4 | col_major_contiguous | dependent10 | 1 | blas | 10240 | 8.437958 | 8240.19 | 486.61 | 7.0% | 824.02 | ok |
| add_f64_concrete_shared | add | concrete-shared | f64 | 16 | col_major_contiguous | single | 1 | blas | 2048 | 1.735000 | 847.17 | 49.12 | 3.3% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | f64 | 16 | col_major_contiguous | dependent10 | 1 | blas | 10240 | 8.564000 | 8363.28 | 166.34 | 1.6% | 836.33 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | f64 | 256 | col_major_contiguous | single | 1 | blas | 1024 | 0.906292 | 885.05 | 16.09 | 1.8% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | f64 | 256 | col_major_contiguous | dependent10 | 1 | blas | 10240 | 8.920833 | 8711.75 | 137.02 | 1.2% | 871.17 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.433042 | 2376.02 | 45.04 | 1.2% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 2.499916 | 2441.32 | 78.67 | 2.0% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 3.600459 | 3516.07 | 47.77 | 1.1% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | c64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.633167 | 2571.45 | 39.31 | 2.3% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | c64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 2.632209 | 2570.52 | 41.97 | 1.4% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | c64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 5.126750 | 5006.59 | 43.95 | 0.9% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.010583 | 1963.46 | 40.04 | 1.2% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 2.127791 | 2077.92 | 16.48 | 1.3% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 3.815666 | 3726.24 | 68.60 | 1.4% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 2.537708 | 2478.23 | 30.58 | 0.8% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | f64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 3.917166 | 3825.36 | 89.35 | 2.0% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | c64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 2.832333 | 2765.95 | 62.58 | 1.4% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | c64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 9.625209 | 9399.62 | 188.66 | 1.6% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | f64 | 4 | col_major_contiguous | single | 1 | blas | 2048 | 1.711833 | 835.86 | 46.93 | 3.8% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | f64 | 16 | col_major_contiguous | single | 1 | blas | 2048 | 1.682917 | 821.74 | 24.38 | 1.9% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | f64 | 256 | col_major_contiguous | single | 1 | blas | 1024 | 0.987333 | 964.19 | 14.53 | 1.6% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | f64 | 4 | col_major_contiguous | single | 1 | blas | 2048 | 1.353417 | 660.85 | 29.99 | 3.0% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | f64 | 16 | col_major_contiguous | single | 1 | blas | 2048 | 1.326417 | 647.66 | 5.68 | 2.0% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | f64 | 256 | col_major_contiguous | single | 1 | blas | 2048 | 1.410500 | 688.72 | 19.71 | 1.9% | — | ok |
| add_f64_concrete-shared_size4_single | add | concrete-shared | f64 | 4 | col_major_contiguous | single | 4 | blas | 1024 | 0.886875 | 866.09 | 16.05 | 2.2% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | f64 | 4 | col_major_contiguous | dependent10 | 4 | blas | 10240 | 8.431375 | 8233.76 | 35.01 | 1.1% | 823.38 | ok |
| add_f64_concrete_shared | add | concrete-shared | f64 | 16 | col_major_contiguous | single | 4 | blas | 2048 | 1.727792 | 843.65 | 15.26 | 1.6% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | f64 | 16 | col_major_contiguous | dependent10 | 4 | blas | 10240 | 8.466583 | 8268.15 | 80.18 | 1.2% | 826.81 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 0.916667 | 895.18 | 8.14 | 2.1% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | f64 | 256 | col_major_contiguous | dependent10 | 4 | blas | 10240 | 8.812416 | 8605.88 | 159.26 | 1.1% | 860.59 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.467208 | 2409.38 | 39.04 | 1.4% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 2.463166 | 2405.44 | 79.00 | 1.9% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 3.557958 | 3474.57 | 38.94 | 0.8% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | c64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.587166 | 2526.53 | 59.67 | 1.5% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | c64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 2.587666 | 2527.02 | 48.75 | 1.5% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | c64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 5.063291 | 4944.62 | 44.72 | 1.2% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.034459 | 1986.78 | 39.82 | 1.4% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 2.183125 | 2131.96 | 85.55 | 2.9% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 3.822167 | 3732.58 | 66.28 | 1.5% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 2.544750 | 2485.11 | 106.87 | 2.7% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | f64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 3.977833 | 3884.60 | 124.15 | 1.8% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | c64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 2.929500 | 2860.84 | 81.75 | 2.2% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | c64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 9.526833 | 9303.55 | 90.98 | 1.1% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | f64 | 4 | col_major_contiguous | single | 4 | blas | 2048 | 1.684000 | 822.27 | 11.96 | 1.4% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | f64 | 16 | col_major_contiguous | single | 4 | blas | 2048 | 1.693458 | 826.88 | 13.45 | 2.4% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 1.023708 | 999.71 | 28.12 | 2.3% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | f64 | 4 | col_major_contiguous | single | 4 | blas | 2048 | 1.324000 | 646.48 | 8.95 | 1.3% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | f64 | 16 | col_major_contiguous | single | 4 | blas | 2048 | 1.350625 | 659.48 | 11.18 | 2.3% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | f64 | 256 | col_major_contiguous | single | 4 | blas | 2048 | 1.589666 | 776.20 | 29.85 | 2.8% | — | ok |

## Per-route timing boundaries

Shared-session routes enter/exit the session once around warmup/calibration/all samples, outside timing. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions reuse the entered executor context. Earlier BLAS revisions included per-operation executor entry; consult the recorded tenferro commit. Fresh routes enter/exit for each operation. Prepared-repeat excludes plan preparation; compiled-repeat excludes tracing/compilation and runtime construction.

- **add / concrete-shared** — inside: add, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **einsum / prepared-repeat** — inside: prepared_execute, output_allocation; outside: backend_construction, input_construction, plan_preparation, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **gather / concrete-shared** — inside: gather, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **reduce_sum / concrete-shared** — inside: reduce_sum, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **solve / concrete-shared** — inside: solve, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.

# CPU Small-Work Benchmark Results

- Suite: `cpu/small_work`
- Raw run: `data/results/mac-cpu/cpu/small_work/20261024_182518`

## CPU Information

- Model: `Apple M1 (Virtual)`
- Vendor: `Apple`
- Logical CPUs: `3`
- Physical CPUs: `3`
- Sockets: `1`
- Cores per socket: `3`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Standard: 3 physical / 3 logical (L1i 192 KiB, L1d 128 KiB, L2 12 MiB, 3 CPUs/L2)`
- Python platform: `macOS-26.6.2-arm64-arm-64bit-Mach-O`

## run_t1

Full metadata: `data/results/mac-cpu/cpu/small_work/20261024_182518/run_t1.yaml`

```yaml
timestamp: '2026-10-24T18:25:19Z'
tenferro_rs:
  path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: fa39e8d78d9c58e36575084b67448a8112ec1d6c
  dirty: false
  features:
  - blas-accelerate
  resolved_path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: accelerate
  version: unknown
  library: /System/Library/Frameworks/Accelerate.framework
environment:
  hostname: iad20-fj922-9dc46341-0e5e-44ec-8c33-7c2622873f65-0A7060B00D9A.local
  os: macOS-26.6.2-arm64-arm-64bit-Mach-O
  arch: arm64
  cpu: arm
  env:
    OMP_NUM_THREADS: '1'
    OMP_THREAD_LIMIT: '1'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: mac-cpu
    BENCHMARK_COMMIT: 50a7dc9b3ea08f5dba14db07df1e09314cac56f0
    BENCH_COMMAND_BUDGET_SECONDS: '120'
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
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.1 (58cf8f9c3 2026-10-13)
    BENCH_INSTANCE: null
```

## run_t4

Full metadata: `data/results/mac-cpu/cpu/small_work/20261024_182518/run_t4.yaml`

```yaml
timestamp: '2026-10-24T18:25:28Z'
tenferro_rs:
  path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: fa39e8d78d9c58e36575084b67448a8112ec1d6c
  dirty: false
  features:
  - blas-accelerate
  resolved_path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: accelerate
  version: unknown
  library: /System/Library/Frameworks/Accelerate.framework
environment:
  hostname: iad20-fj922-9dc46341-0e5e-44ec-8c33-7c2622873f65-0A7060B00D9A.local
  os: macOS-26.6.2-arm64-arm-64bit-Mach-O
  arch: arm64
  cpu: arm
  env:
    OMP_NUM_THREADS: '4'
    OMP_THREAD_LIMIT: '4'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '4'
    BENCHMARK_TARGET_PROFILE: mac-cpu
    BENCHMARK_COMMIT: 50a7dc9b3ea08f5dba14db07df1e09314cac56f0
    BENCH_COMMAND_BUDGET_SECONDS: '120'
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
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.1 (58cf8f9c3 2026-10-13)
    BENCH_INSTANCE: null
```

## Case status

- Manifest: `cpu/small_work` version 3, coverage `quick`, effort `standard`
- Expected 144, selected 144, executed 144, unsupported 0, failed 0, missing 0, noisy 48.
- Complete: **yes** — only executed cases count as covered; unsupported, failed, missing and unselected cases never do.
- Noisy: `add_f64_concrete-shared_size4_dependent10@t1`, `add_f64_concrete_shared@t1`, `add_f64_concrete-shared_size256_dependent10@t1`, `einsum_f64_prepared-repeat_n2_single@t1`, `solve_f64_concrete-shared_n4_single@t1`, `einsum_f64_prepared-repeat_n32_single@t1`, `gather_f64_concrete-shared_size4_single@t1`, `einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single@t1`, `einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single@t1`, `nonad_einsum3_n2_col_major_contiguous_eager-shared@t1`, `nonad_qr_n2_col_major_contiguous_eager-no-ad@t1`, `nonad_svd_n2_col_major_contiguous_concrete-shared@t1`, `nonad_svd_n2_col_major_contiguous_eager-shared@t1`, `nonad_solve_n2_col_major_contiguous_concrete-fresh@t1`, `nonad_einsum_n16_col_major_contiguous_eager-shared@t1`, `nonad_einsum_n16_col_major_contiguous_concrete-shared@t1`, `nonad_einsum_n16_col_major_contiguous_eager-no-ad@t1`, `nonad_einsum_n16_col_major_contiguous_concrete-fresh@t1`, `nonad_qr_n16_col_major_contiguous_eager-shared@t1`, `nonad_solve_n16_col_major_contiguous_concrete-fresh@t1`, `nonad_qr_n16_col_major_contiguous_concrete-scoped@t1`, `nonad_solve_n16_col_major_contiguous_concrete-scoped@t1`, `add_f64_concrete-shared_size256_single@t4`, `solve_f64_concrete-shared_n2_single@t4`, `einsum_c64_prepared-repeat_n32_single@t4`, `gather_f64_concrete-shared_size256_single@t4`, `reduce_sum_f64_concrete-shared_size4_single@t4`, `reduce_sum_f64_concrete-shared_size256_single@t4`, `einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single@t4`, `einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single@t4`, `einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single@t4`, `einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single@t4`, `nonad_einsum_n2_col_major_contiguous_concrete-fresh@t4`, `nonad_einsum3_n2_col_major_contiguous_eager-no-ad@t4`, `nonad_einsum3_n2_col_major_contiguous_concrete-shared@t4`, `nonad_qr_n2_col_major_contiguous_eager-no-ad@t4`, `nonad_qr_n2_col_major_contiguous_concrete-shared@t4`, `nonad_qr_n2_col_major_contiguous_eager-shared@t4`, `nonad_svd_n2_col_major_contiguous_eager-no-ad@t4`, `nonad_svd_n2_col_major_contiguous_concrete-shared@t4` (+8 more)

Numerical checks: **144/144 recorded rows passed** (case × thread count).

## Timing

Sequential release runs; explicit initialization and warmups, then calibration toward the declared batch duration (2 ms by default), then the measured batches of the recorded effort (standard: 3 warmups and 15 batches; scan: a low-repetition screen that only marks suspects). Legacy single-call routes start calibration at 1024 workflows; matched non-AD and dependent-chain routes start at 16 and bound retained memory. Each batch uses one wall-clock interval; all outputs are retained until it stops. Statistics divide batch time by iterations. Median and inclusive IQR are in nanoseconds (ns); CoV is sample standard deviation / mean. NOISY means CoV > 10%, a descriptive label only. No noisy rows are excluded.

Workflow time is the total for one call or the complete dependent chain. The separate ns/op column divides the chain median by its operation count, not an independently measured call. Setup rows measure preparation only, not execution.

Inputs and backend construction, independent numerical checks and AD backward checks are outside timing. Returned outputs/plans and every dependent-chain intermediate are retained until after each batch interval; retention containers are prepared before the clock. value downloads/checks are outside. Eager AD rows measure forward execution/recording, not backward.

| Case ID | Operation | API route | Route (path / output / repr / session) | Dtype | Shape | Layout | Workflow | Threads | Provider | Operations/batch | Median batch ms | Median workflow ns | IQR ns | CoV | ns/op (chain only) | Status |
|---|---|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 4096 | 2.269125 | 553.99 | 21.04 | 3.6% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 1 | blas | 5120 | 2.769083 | 5408.37 | 265.75 | 33.0% | 540.84 | NOISY |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 4096 | 2.328417 | 568.46 | 50.59 | 16.6% | — | NOISY |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 1 | blas | 5120 | 2.730792 | 5333.58 | 128.87 | 2.1% | 533.36 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 2048 | 1.272750 | 621.46 | 27.44 | 4.8% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 1 | blas | 2560 | 1.534333 | 5993.49 | 109.86 | 14.8% | 599.35 | NOISY |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.263542 | 2210.49 | 183.09 | 20.7% | — | NOISY |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 2.303625 | 2249.63 | 106.87 | 6.2% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.984041 | 2914.10 | 34.20 | 1.3% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.500375 | 2441.77 | 60.12 | 3.2% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 2.515291 | 2456.34 | 45.47 | 1.8% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 4.170209 | 4072.47 | 146.81 | 3.2% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 4096 | 3.292667 | 803.87 | 54.61 | 9.5% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 0.904208 | 883.02 | 39.55 | 33.7% | — | NOISY |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.194125 | 2142.70 | 38.39 | 2.7% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 2.344875 | 2289.92 | 35.12 | 2.0% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 3.533167 | 3450.36 | 834.23 | 21.0% | — | NOISY |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 2.824375 | 2758.18 | 106.02 | 2.3% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 8.390583 | 8193.93 | 123.21 | 0.9% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 4096 | 3.076042 | 750.99 | 152.77 | 34.7% | — | NOISY |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 4096 | 2.885458 | 704.46 | 33.76 | 5.3% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 2048 | 1.985500 | 969.48 | 72.87 | 5.1% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 4096 | 2.227791 | 543.89 | 11.75 | 2.1% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 4096 | 2.228917 | 544.17 | 20.37 | 3.1% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 4096 | 2.344792 | 572.46 | 24.16 | 2.9% | — | ok |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 4.576375 | 4469.12 | 591.82 | 16.5% | — | NOISY |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 2.474250 | 2416.26 | 18.70 | 0.8% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 2.446583 | 2389.24 | 52.65 | 4.1% | — | ok |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 1024 | 2.446625 | 2389.28 | 184.67 | 4.6% | — | ok |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 2048 | 1.907958 | 931.62 | 18.69 | 1.5% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 2048 | 1.833625 | 895.32 | 28.53 | 15.3% | — | NOISY |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 1024 | 4.150625 | 4053.34 | 50.82 | 1.8% | — | ok |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 1024 | 2.578375 | 2517.94 | 82.89 | 8.0% | — | ok |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 1024 | 2.566458 | 2506.31 | 101.81 | 2.4% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 3.174000 | 6199.22 | 225.59 | 4.0% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.193000 | 2141.60 | 70.09 | 3.3% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 3.163000 | 6177.73 | 288.66 | 3.6% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 1.994250 | 1947.51 | 56.86 | 4.1% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 128 | 2.017000 | 15757.81 | 970.21 | 5.0% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 3.262250 | 6371.58 | 186.93 | 2.6% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 128 | 2.045458 | 15980.14 | 936.85 | 6.6% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 3.228625 | 6305.91 | 558.27 | 16.6% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 1.823125 | 890.20 | 25.43 | 1.8% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 1.070833 | 4182.94 | 5173.83 | 62.3% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 1.779625 | 868.96 | 40.51 | 3.2% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 1.766459 | 3450.12 | 124.75 | 3.8% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 1.584042 | 1546.92 | 38.80 | 2.9% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.360500 | 4610.35 | 319.38 | 3.9% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 1.644375 | 1605.83 | 408.59 | 40.1% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.293667 | 4479.82 | 88.66 | 19.4% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 1.565333 | 764.32 | 59.91 | 21.6% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 1.639125 | 3201.42 | 328.25 | 5.6% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 1.607250 | 784.79 | 58.45 | 7.0% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 3.105583 | 3032.80 | 114.38 | 3.2% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 3.127334 | 3054.04 | 456.52 | 23.6% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 2.229666 | 8709.63 | 1368.74 | 16.2% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 1.570875 | 3068.12 | 320.15 | 31.0% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 128 | 1.190250 | 9298.83 | 872.89 | 23.2% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 1.880750 | 7346.68 | 236.73 | 23.7% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.103167 | 4107.75 | 73.57 | 1.3% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 1.771250 | 6918.95 | 209.47 | 3.6% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.106583 | 4114.42 | 107.75 | 2.1% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.261833 | 4417.64 | 146.00 | 2.4% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.154625 | 2104.13 | 38.62 | 2.8% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.311041 | 4513.75 | 369.30 | 7.2% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.228666 | 2176.43 | 73.63 | 22.6% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 3.173625 | 6198.49 | 324.83 | 3.1% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 1.818959 | 888.16 | 55.71 | 3.6% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 1.714208 | 837.02 | 83.99 | 6.6% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 2.043583 | 7982.75 | 962.40 | 7.3% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.215666 | 4327.47 | 392.09 | 14.9% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.247042 | 2194.38 | 225.24 | 11.9% | — | NOISY |
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 4096 | 2.267750 | 553.65 | 5.12 | 4.8% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 4 | blas | 5120 | 2.658875 | 5193.12 | 123.54 | 5.2% | 519.31 | ok |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 4096 | 2.263333 | 552.57 | 3.97 | 1.2% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 4 | blas | 5120 | 2.675625 | 5225.83 | 147.74 | 3.6% | 522.58 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 2048 | 1.246334 | 608.56 | 57.14 | 10.5% | — | NOISY |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 4 | blas | 2560 | 1.466709 | 5729.33 | 103.44 | 6.3% | 572.93 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.143625 | 2093.38 | 51.39 | 4.2% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 2.151875 | 2101.44 | 35.75 | 5.6% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 3.025292 | 2954.39 | 356.49 | 9.4% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.525750 | 2466.55 | 178.81 | 6.6% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 2.716667 | 2653.00 | 175.74 | 8.1% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 4.428625 | 4324.83 | 282.69 | 5.3% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 1.841417 | 899.13 | 133.11 | 18.5% | — | NOISY |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 2048 | 1.987750 | 970.58 | 91.95 | 6.7% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.362167 | 2306.80 | 151.41 | 5.6% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 2.456875 | 2399.29 | 96.17 | 6.5% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 3.467000 | 3385.74 | 101.75 | 5.1% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 3.265333 | 3188.80 | 245.61 | 6.6% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 9.298166 | 9080.24 | 1357.65 | 19.0% | — | NOISY |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 4096 | 3.079958 | 751.94 | 58.48 | 4.9% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 2048 | 1.656208 | 808.70 | 63.92 | 7.4% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 1.177417 | 1149.82 | 132.53 | 11.4% | — | NOISY |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 2048 | 1.522292 | 743.31 | 182.43 | 25.2% | — | NOISY |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 2048 | 1.288458 | 629.13 | 31.26 | 4.3% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 4096 | 2.650250 | 647.03 | 69.65 | 18.7% | — | NOISY |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 4.481333 | 4376.30 | 441.59 | 8.9% | — | ok |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 2.367750 | 2312.26 | 199.79 | 5.8% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 2.399000 | 2342.77 | 256.59 | 6.5% | — | ok |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 1024 | 2.394500 | 2338.38 | 67.38 | 5.2% | — | ok |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 4096 | 3.697292 | 902.66 | 102.79 | 9.5% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 4096 | 4.239500 | 1035.03 | 59.57 | 11.4% | — | NOISY |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 1024 | 4.068167 | 3972.82 | 707.48 | 13.6% | — | NOISY |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 1024 | 2.992875 | 2922.73 | 356.12 | 12.8% | — | NOISY |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 1024 | 3.024250 | 2953.37 | 558.21 | 17.5% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 1.960750 | 7659.18 | 1239.99 | 10.4% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 1.278167 | 2496.42 | 296.96 | 7.5% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 2.011625 | 7857.91 | 809.24 | 7.3% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.456541 | 2398.97 | 253.72 | 5.7% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 128 | 2.548416 | 19909.50 | 1453.94 | 8.2% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 2.061917 | 8054.36 | 903.32 | 18.9% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 128 | 2.581459 | 20167.65 | 2336.43 | 10.3% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 3.364750 | 6571.78 | 811.56 | 6.7% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 1.916750 | 935.91 | 34.10 | 4.0% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.133167 | 4166.34 | 500.16 | 15.2% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 2.174334 | 1061.69 | 109.72 | 17.9% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.118708 | 4138.10 | 396.77 | 20.4% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.784542 | 1742.72 | 117.82 | 5.1% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.649333 | 5174.48 | 346.52 | 20.3% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.817041 | 1774.45 | 77.19 | 22.2% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.605417 | 5088.71 | 827.11 | 17.7% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 1.773709 | 866.07 | 68.41 | 5.1% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 1.876375 | 3664.79 | 232.79 | 5.2% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 1.756875 | 857.85 | 93.68 | 7.1% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 1.805041 | 3525.47 | 224.00 | 5.3% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 1.576500 | 3079.10 | 116.98 | 3.2% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 2.253542 | 8802.90 | 500.65 | 5.9% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 1.426208 | 2785.56 | 456.87 | 8.6% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 2.020417 | 7892.25 | 505.37 | 8.2% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 2.046958 | 7995.93 | 532.88 | 5.3% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.093042 | 4087.97 | 300.90 | 7.5% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 1.675334 | 6544.27 | 115.80 | 3.4% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 128 | 0.523334 | 4088.55 | 616.05 | 17.3% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.519667 | 4921.22 | 393.55 | 9.7% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.448208 | 2390.83 | 215.45 | 10.7% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.595833 | 5069.99 | 548.99 | 15.0% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.392334 | 2336.26 | 93.26 | 11.0% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 1.909625 | 7459.47 | 531.82 | 6.1% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 1.714500 | 837.16 | 35.27 | 3.0% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 4096 | 3.057667 | 746.50 | 31.33 | 4.8% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 2.360875 | 9222.17 | 857.10 | 19.3% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.441959 | 4769.45 | 356.45 | 12.7% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.531541 | 2472.21 | 399.35 | 17.1% | — | NOISY |

## Matched non-AD eager comparisons (#98)

All requested QR/SVD outputs are retained. Ordinary einsum uses the same parsed integer notation and default Auto strategy; planning/cache lookup stays inside both public calls. These are public-call diagnostics, not prepared execution or an isolated AD cost. Fresh pairs include session entry; shared pairs exclude it. Views are constructed untimed and consumed as views on both routes. No output to_tensor conversion is timed. Capture is inactive by construction in each fresh process, with no capture_trace guard; inputs and validation outputs are checked untracked.

| Operation | Shape | Layout | Session | Threads | Direct ns | Eager ns | Eager − direct ns | Eager/direct | Quality |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 1 | 6199.22 | 2141.60 | -4057.62 | 0.345 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | shared | 1 | 6177.73 | 1947.51 | -4230.22 | 0.315 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 4 | 7659.18 | 2496.42 | -5162.76 | 0.326 | inconclusive |
| einsum | 2×2 | col_major_contiguous | shared | 4 | 7857.91 | 2398.97 | -5458.94 | 0.305 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 1 | 9298.83 | 3068.12 | -6230.71 | 0.330 | inconclusive |
| einsum | 16×16 | col_major_contiguous | shared | 1 | 8709.63 | 3054.04 | -5655.59 | 0.351 | inconclusive |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 4 | 7892.25 | 2785.56 | -5106.69 | 0.353 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | shared | 4 | 8802.90 | 3079.10 | -5723.80 | 0.350 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 1 | 15757.81 | 6371.58 | -9386.23 | 0.404 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | shared | 1 | 15980.14 | 6305.91 | -9674.23 | 0.395 | inconclusive |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 4 | 19909.50 | 8054.36 | -11855.14 | 0.405 | inconclusive |
| einsum3 | 2×2 | col_major_contiguous | shared | 4 | 20167.65 | 6571.78 | -13595.87 | 0.326 | inconclusive |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 1 | 890.20 | 4182.94 | 3292.74 | 4.699 | inconclusive |
| qr | 2×2 | col_major_contiguous | shared | 1 | 868.96 | 3450.12 | 2581.16 | 3.970 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 4 | 935.91 | 4166.34 | 3230.43 | 4.452 | inconclusive |
| qr | 2×2 | col_major_contiguous | shared | 4 | 1061.69 | 4138.10 | 3076.41 | 3.898 | inconclusive |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 1 | 4114.42 | 6918.95 | 2804.53 | 1.682 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | shared | 1 | 4107.75 | 7346.68 | 3238.93 | 1.788 | inconclusive |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 4 | 4088.55 | 6544.27 | 2455.72 | 1.601 | inconclusive |
| qr | 16×16 | col_major_contiguous | shared | 4 | 4087.97 | 7995.93 | 3907.96 | 1.956 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 1 | 764.32 | 3201.42 | 2437.10 | 4.189 | inconclusive |
| solve | 2×2 | col_major_contiguous | shared | 1 | 784.79 | 3032.80 | 2248.01 | 3.864 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 4 | 866.07 | 3664.79 | 2798.72 | 4.232 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | shared | 4 | 857.85 | 3525.47 | 2667.62 | 4.110 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 1 | 2176.43 | 4513.75 | 2337.32 | 2.074 | inconclusive |
| solve | 16×16 | col_major_contiguous | shared | 1 | 2104.13 | 4417.64 | 2313.51 | 2.100 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 4 | 2336.26 | 5069.99 | 2733.73 | 2.170 | inconclusive |
| solve | 16×16 | col_major_contiguous | shared | 4 | 2390.83 | 4921.22 | 2530.39 | 2.058 | inconclusive |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1546.92 | 4610.35 | 3063.43 | 2.980 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | shared | 1 | 1605.83 | 4479.82 | 2873.99 | 2.790 | inconclusive |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 4 | 1742.72 | 5174.48 | 3431.76 | 2.969 | inconclusive |
| svd | 2×2 | col_major_contiguous | shared | 4 | 1774.45 | 5088.71 | 3314.26 | 2.868 | inconclusive |

## Separate allocation diagnostics

Instrumented executable is built separately after latency collection. Calling-thread Rust allocator requests only; excludes foreign BLAS and worker-thread allocations. Realloc counts its requested new size, not net retained bytes. Output destruction and instrumentation setup are excluded. Helper references and multi-output retention use stack storage. No allocation-instrumented run contributes latency samples.

| Corresponding case ID | Threads | Median allocation requests/op | Median requested bytes/op | Status |
|---|---:|---:|---:|---|
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | 1 | 116 | 12751 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | 1 | 15 | 4112 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | 1 | 116 | 12751 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-shared | 1 | 15 | 4112 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | 1 | 266 | 51445 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | 1 | 48 | 20936 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | 1 | 266 | 51445 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | 1 | 48 | 20936 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | 1 | 6 | 1760 | passed |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | 1 | 39 | 16003 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-shared | 1 | 6 | 1760 | passed |
| nonad_qr_n2_col_major_contiguous_eager-shared | 1 | 39 | 16003 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | 1 | 8 | 2952 | passed |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | 1 | 46 | 18531 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-shared | 1 | 8 | 2952 | passed |
| nonad_svd_n2_col_major_contiguous_eager-shared | 1 | 46 | 18531 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | 1 | 3 | 80 | passed |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | 1 | 35 | 16195 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-shared | 1 | 3 | 80 | passed |
| nonad_solve_n2_col_major_contiguous_eager-shared | 1 | 35 | 16195 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-shared | 1 | 15 | 4112 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | 1 | 116 | 12751 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | 1 | 15 | 4112 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | 1 | 116 | 12751 | passed |
| nonad_qr_n16_col_major_contiguous_eager-shared | 1 | 39 | 21603 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-shared | 1 | 6 | 7360 | passed |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | 1 | 39 | 21603 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | 1 | 6 | 7360 | passed |
| nonad_solve_n16_col_major_contiguous_eager-shared | 1 | 35 | 18211 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-shared | 1 | 3 | 4112 | passed |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | 1 | 35 | 18211 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | 1 | 3 | 4112 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | 1 | 116 | 12751 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | 1 | 6 | 1760 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | 1 | 3 | 80 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | 1 | 116 | 12751 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | 1 | 6 | 7360 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | 1 | 3 | 4112 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | 4 | 116 | 12751 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | 4 | 15 | 4112 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | 4 | 116 | 12751 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-shared | 4 | 15 | 4112 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | 4 | 266 | 51445 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | 4 | 48 | 20936 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | 4 | 266 | 51445 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | 4 | 48 | 20936 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | 4 | 6 | 1760 | passed |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | 4 | 39 | 16003 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-shared | 4 | 6 | 1760 | passed |
| nonad_qr_n2_col_major_contiguous_eager-shared | 4 | 39 | 16003 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | 4 | 8 | 2952 | passed |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | 4 | 46 | 18531 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-shared | 4 | 8 | 2952 | passed |
| nonad_svd_n2_col_major_contiguous_eager-shared | 4 | 46 | 18531 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | 4 | 3 | 80 | passed |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | 4 | 35 | 16195 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-shared | 4 | 3 | 80 | passed |
| nonad_solve_n2_col_major_contiguous_eager-shared | 4 | 35 | 16195 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-shared | 4 | 15 | 4112 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | 4 | 116 | 12751 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | 4 | 15 | 4112 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | 4 | 116 | 12751 | passed |
| nonad_qr_n16_col_major_contiguous_eager-shared | 4 | 39 | 21603 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-shared | 4 | 6 | 7360 | passed |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | 4 | 39 | 21603 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | 4 | 6 | 7360 | passed |
| nonad_solve_n16_col_major_contiguous_eager-shared | 4 | 35 | 18211 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-shared | 4 | 3 | 4112 | passed |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | 4 | 35 | 18211 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | 4 | 3 | 4112 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | 4 | 116 | 12751 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | 4 | 6 | 1760 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | 4 | 3 | 80 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | 4 | 116 | 12751 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | 4 | 6 | 7360 | passed |
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
  "collection_total_seconds": 0.22632404199976008,
  "case_process_seconds": 0.19890200099962385
}
```

Wall observations `wall_allocations_t4.json`:
```json
{
  "collection_total_seconds": 0.21539650000067923,
  "case_process_seconds": 0.18791500100178382
}
```

Wall observations `wall_command.json`:
```json
{
  "command_seconds_before_final_report_write": 324.84698375,
  "report_generation_seconds": 0.112220041,
  "scope": "command startup through report generation; excludes final report file writes",
  "budget_seconds": 120.0,
  "budget_status": "over_budget; diagnostic only"
}
```

Wall observations `wall_measure_t1.json`:
```json
{
  "collection_total_seconds": 3.8967689579999387,
  "case_process_seconds": 3.847584418000224
}
```

Wall observations `wall_measure_t4.json`:
```json
{
  "collection_total_seconds": 3.8663595830003032,
  "case_process_seconds": 3.817151670001749
}
```

Wall observations `wall_phases.json`:
```json
{
  "validation_and_latency_build_seconds": 7.42841625,
  "metadata_and_latency_collection_seconds": 17.263625084,
  "allocation_diagnostic_build_seconds": 299.435332166,
  "allocation_collection_seconds": 0.607390209,
  "command_before_report_seconds": 324.734763709
}
```

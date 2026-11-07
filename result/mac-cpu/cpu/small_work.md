# CPU Small-Work Benchmark Results

- Suite: `cpu/small_work`
- Raw run: `data/results/mac-cpu/cpu/small_work/20261107_183240`

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
- Python platform: `macOS-26.7.2-arm64-arm-64bit-Mach-O`
- Container: `none`

## run_t1

Full metadata: `data/results/mac-cpu/cpu/small_work/20261107_183240/run_t1.yaml`

```yaml
timestamp: '2026-11-07T18:32:40Z'
tenferro_rs:
  path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: 9a850b06155405acc99ecd0648829e9e6d3866ba
  dirty: false
  features:
  - blas-accelerate
  resolved_path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: accelerate
  version: unknown
  library: /System/Library/Frameworks/Accelerate.framework
environment:
  container: none
  hostname: sjc22-bm213-6f7e243e-d935-4cb5-98ec-e1cc5a37fa73-12839649360E.local
  os: macOS-26.7.2-arm64-arm-64bit-Mach-O
  arch: arm64
  cpu: arm
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '1'
    measured_cpus: null
    l3_domains: null
    cpu_boost: null
    scaling_governor: null
    scaling_driver: null
  env:
    OMP_NUM_THREADS: '1'
    OMP_THREAD_LIMIT: '1'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: mac-cpu
    BENCHMARK_COMMIT: d6655b567e09aa29c3201655e1de418716934876
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

Full metadata: `data/results/mac-cpu/cpu/small_work/20261107_183240/run_t4.yaml`

```yaml
timestamp: '2026-11-07T18:32:51Z'
tenferro_rs:
  path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: 9a850b06155405acc99ecd0648829e9e6d3866ba
  dirty: false
  features:
  - blas-accelerate
  resolved_path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: accelerate
  version: unknown
  library: /System/Library/Frameworks/Accelerate.framework
environment:
  container: none
  hostname: sjc22-bm213-6f7e243e-d935-4cb5-98ec-e1cc5a37fa73-12839649360E.local
  os: macOS-26.7.2-arm64-arm-64bit-Mach-O
  arch: arm64
  cpu: arm
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '4'
    measured_cpus: null
    l3_domains: null
    cpu_boost: null
    scaling_governor: null
    scaling_driver: null
  env:
    OMP_NUM_THREADS: '4'
    OMP_THREAD_LIMIT: '4'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '4'
    BENCHMARK_TARGET_PROFILE: mac-cpu
    BENCHMARK_COMMIT: d6655b567e09aa29c3201655e1de418716934876
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
- Expected 144, selected 144, executed 144, unsupported 0, failed 0, skipped 0, missing 0, noisy 120.
- Complete: **yes** — only executed cases count as covered; unsupported, skipped, failed, missing and unselected cases never do.

| expected | selected | executed | unsupported | failed | skipped | missing | noisy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 144 | 144 | 144 | 0 | 0 | 0 | 0 | 120 |

- Noisy: `add_f64_concrete-shared_size4_single@t1`, `add_f64_concrete-shared_size4_dependent10@t1`, `add_f64_concrete_shared@t1`, `add_f64_concrete-shared_size16_dependent10@t1`, `add_f64_concrete-shared_size256_single@t1`, `einsum_f64_prepared-repeat_n2_single@t1`, `einsum_f64_prepared-repeat_n4_single@t1`, `einsum_c64_prepared-repeat_n2_single@t1`, `einsum_c64_prepared-repeat_n4_single@t1`, `solve_f64_concrete-shared_n4_single@t1`, `einsum_c64_prepared-repeat_n8_single@t1`, `einsum_c64_prepared-repeat_n32_single@t1`, `gather_f64_concrete_shared@t1`, `gather_f64_concrete-shared_size256_single@t1`, `reduce_sum_f64_concrete-shared_size16_single@t1`, `reduce_sum_f64_concrete-shared_size256_single@t1`, `einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single@t1`, `einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single@t1`, `einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single@t1`, `einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single@t1`, `einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single@t1`, `einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single@t1`, `nonad_einsum_n2_col_major_contiguous_concrete-fresh@t1`, `nonad_einsum_n2_col_major_contiguous_eager-no-ad@t1`, `nonad_einsum_n2_col_major_contiguous_concrete-shared@t1`, `nonad_einsum_n2_col_major_contiguous_eager-shared@t1`, `nonad_einsum3_n2_col_major_contiguous_concrete-fresh@t1`, `nonad_einsum3_n2_col_major_contiguous_eager-no-ad@t1`, `nonad_einsum3_n2_col_major_contiguous_concrete-shared@t1`, `nonad_einsum3_n2_col_major_contiguous_eager-shared@t1`, `nonad_qr_n2_col_major_contiguous_concrete-fresh@t1`, `nonad_qr_n2_col_major_contiguous_eager-no-ad@t1`, `nonad_qr_n2_col_major_contiguous_concrete-shared@t1`, `nonad_qr_n2_col_major_contiguous_eager-shared@t1`, `nonad_svd_n2_col_major_contiguous_concrete-fresh@t1`, `nonad_svd_n2_col_major_contiguous_eager-no-ad@t1`, `nonad_svd_n2_col_major_contiguous_concrete-shared@t1`, `nonad_svd_n2_col_major_contiguous_eager-shared@t1`, `nonad_solve_n2_col_major_contiguous_concrete-fresh@t1`, `nonad_solve_n2_col_major_contiguous_eager-no-ad@t1` (+80 more)

Numerical checks: **144/144 recorded rows passed** (case × thread count).

## Timing

Sequential release runs; explicit initialization and warmups, then calibration toward the declared batch duration (2 ms by default), then the measured batches of the recorded effort (standard: 3 warmups and 15 batches; scan: a low-repetition screen that only marks suspects). Legacy single-call routes start calibration at 1024 workflows; matched non-AD and dependent-chain routes start at 16 and bound retained memory. Each batch uses one wall-clock interval; all outputs are retained until it stops. Statistics divide batch time by iterations. Median and inclusive IQR are in nanoseconds (ns); CoV is sample standard deviation / mean. NOISY means CoV > 10%, a descriptive label only. No noisy rows are excluded.

Workflow time is the total for one call or the complete dependent chain. The separate ns/op column divides the chain median by its operation count, not an independently measured call. Setup rows measure preparation only, not execution.

Inputs and backend construction, independent numerical checks and AD backward checks are outside timing. Returned outputs/plans and every dependent-chain intermediate are retained until after each batch interval; retention containers are prepared before the clock. value downloads/checks are outside. Eager AD rows measure forward execution/recording, not backward.

| Case ID | Operation | API route | Route (path / output / repr / session) | Dtype | Shape | Layout | Workflow | Threads | Provider | Operations/batch | Median batch ms | Median workflow ns | IQR ns | CoV | ns/op (chain only) | Status |
|---|---|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 4096 | 2.374500 | 579.71 | 33.63 | 22.8% | — | NOISY |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 1 | blas | 5120 | 2.952542 | 5766.68 | 241.70 | 17.5% | 576.67 | NOISY |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 4096 | 2.448375 | 597.75 | 26.88 | 11.5% | — | NOISY |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 1 | blas | 5120 | 2.921292 | 5705.65 | 559.16 | 14.7% | 570.56 | NOISY |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 2048 | 1.399250 | 683.23 | 88.80 | 23.1% | — | NOISY |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 1 | blas | 1280 | 0.820458 | 6409.83 | 640.46 | 6.0% | 640.98 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 1.895709 | 925.64 | 36.09 | 31.3% | — | NOISY |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 2048 | 1.886667 | 921.22 | 41.86 | 23.1% | — | NOISY |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 1.379875 | 1347.53 | 62.19 | 8.8% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 1.112083 | 1086.02 | 101.52 | 29.2% | — | NOISY |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 1 | blas | 2048 | 2.415084 | 1179.24 | 75.40 | 15.6% | — | NOISY |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.871000 | 2803.71 | 160.20 | 6.8% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 1.890791 | 923.24 | 45.52 | 4.8% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 2048 | 1.990250 | 971.80 | 103.95 | 31.3% | — | NOISY |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.335958 | 2281.21 | 134.83 | 4.4% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 2048 | 1.971667 | 962.73 | 40.43 | 4.1% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 1.656875 | 1618.04 | 22.81 | 1.8% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 1.481667 | 1446.94 | 80.61 | 11.6% | — | NOISY |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 7.811417 | 7628.34 | 2255.11 | 41.9% | — | NOISY |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 1024 | 0.812250 | 793.21 | 78.43 | 6.7% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 4096 | 3.011541 | 735.24 | 23.25 | 13.6% | — | NOISY |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 1024 | 1.037667 | 1013.35 | 602.78 | 54.6% | — | NOISY |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 4096 | 3.001167 | 732.71 | 20.80 | 9.1% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 4096 | 3.017125 | 736.60 | 20.06 | 20.4% | — | NOISY |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 2048 | 1.591792 | 777.24 | 59.25 | 16.3% | — | NOISY |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 2.706750 | 2643.31 | 97.19 | 4.4% | — | ok |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 2.993792 | 2923.62 | 320.66 | 21.1% | — | NOISY |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 3.034292 | 2963.18 | 579.28 | 12.3% | — | NOISY |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 2048 | 2.467292 | 1204.73 | 240.09 | 26.4% | — | NOISY |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 2048 | 2.421542 | 1182.39 | 115.88 | 25.7% | — | NOISY |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 2048 | 2.125625 | 1037.90 | 263.77 | 19.3% | — | NOISY |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 1024 | 3.277375 | 3200.56 | 203.88 | 7.2% | — | ok |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 1024 | 3.321709 | 3243.86 | 342.26 | 7.6% | — | ok |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 1024 | 3.303125 | 3225.71 | 597.17 | 15.6% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 1.488792 | 1453.90 | 322.47 | 17.7% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.652625 | 2590.45 | 333.33 | 16.2% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.167334 | 2116.54 | 1795.19 | 39.2% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 3.164125 | 3089.97 | 2113.83 | 37.8% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 64 | 1.547625 | 24181.64 | 29301.44 | 68.7% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 2.147958 | 8390.46 | 3679.36 | 35.0% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 128 | 3.095167 | 24180.99 | 11763.83 | 33.4% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 128 | 0.968333 | 7565.10 | 2012.04 | 54.9% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 4.370833 | 2134.20 | 1317.54 | 40.2% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 0.865000 | 3378.91 | 202.07 | 53.6% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 3.802584 | 1856.73 | 1453.07 | 47.7% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 1.848875 | 3611.08 | 2702.03 | 56.3% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 1.820417 | 1777.75 | 795.31 | 55.9% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 128 | 0.560500 | 4378.91 | 993.97 | 57.8% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 1.906083 | 1861.41 | 464.40 | 35.5% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.693500 | 5260.74 | 4937.70 | 51.2% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.305584 | 1125.77 | 443.80 | 47.8% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.192333 | 4281.90 | 2979.41 | 45.8% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 1.013000 | 989.26 | 365.68 | 32.6% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 1.959666 | 3827.47 | 3362.96 | 41.8% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 0.759625 | 2967.29 | 152.92 | 12.2% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 1.812834 | 1770.35 | 229.63 | 28.5% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 1.729875 | 3378.66 | 632.94 | 33.7% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 1.327583 | 2592.94 | 1901.29 | 40.8% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 2.158250 | 8430.66 | 4712.65 | 33.1% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 3.980958 | 7775.31 | 5220.66 | 42.3% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 128 | 1.202167 | 9391.93 | 3947.43 | 39.8% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 3.829041 | 7478.60 | 3553.79 | 29.1% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 1.378459 | 5384.61 | 1445.80 | 43.0% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 1.200042 | 2343.83 | 462.73 | 40.6% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 3.005458 | 5870.04 | 3457.40 | 38.0% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 1.198083 | 2340.01 | 127.65 | 3.6% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 1.293459 | 1263.14 | 539.59 | 23.8% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 0.574875 | 1122.80 | 76.54 | 12.2% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 1.600417 | 1562.91 | 1209.61 | 41.1% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 1.908625 | 1863.89 | 540.97 | 20.2% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 1.272167 | 4969.40 | 2849.29 | 35.3% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 1.274334 | 2488.93 | 551.39 | 30.3% | — | NOISY |
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 2048 | 2.216917 | 1082.48 | 194.73 | 24.9% | — | NOISY |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 4 | blas | 2560 | 2.844541 | 11111.49 | 6346.11 | 38.0% | 1111.15 | NOISY |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 2048 | 2.745334 | 1340.50 | 702.13 | 29.0% | — | NOISY |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 4 | blas | 2560 | 2.762166 | 10789.71 | 237.47 | 17.9% | 1078.97 | NOISY |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 1.160459 | 1133.26 | 35.95 | 3.5% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 4 | blas | 1280 | 1.412917 | 11038.41 | 249.35 | 48.3% | 1103.84 | NOISY |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.397792 | 1365.03 | 95.79 | 35.7% | — | NOISY |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 2048 | 2.877291 | 1404.93 | 69.31 | 19.1% | — | NOISY |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 1.980458 | 1934.04 | 31.09 | 13.8% | — | NOISY |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.645834 | 1607.26 | 387.06 | 35.2% | — | NOISY |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 1.703584 | 1663.66 | 78.82 | 17.3% | — | NOISY |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 3.696166 | 3609.54 | 304.40 | 13.8% | — | NOISY |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.465458 | 1431.11 | 84.64 | 39.1% | — | NOISY |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 2048 | 3.782292 | 1846.82 | 577.53 | 17.5% | — | NOISY |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.906625 | 2838.50 | 1176.03 | 22.3% | — | NOISY |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 1.527625 | 1491.82 | 87.75 | 22.1% | — | NOISY |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 2.592250 | 2531.49 | 805.81 | 20.2% | — | NOISY |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 2.053375 | 2005.25 | 435.28 | 41.7% | — | NOISY |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 8.790792 | 8584.76 | 1574.01 | 16.7% | — | NOISY |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 1024 | 1.447917 | 1413.98 | 82.30 | 7.9% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 2048 | 2.800084 | 1367.23 | 291.31 | 16.5% | — | NOISY |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 1.667042 | 1627.97 | 162.31 | 19.9% | — | NOISY |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 2048 | 2.797208 | 1365.82 | 81.32 | 12.6% | — | NOISY |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 2048 | 2.733166 | 1334.55 | 172.68 | 14.8% | — | NOISY |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 2048 | 3.013000 | 1471.19 | 433.51 | 24.7% | — | NOISY |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 3.583666 | 3499.67 | 870.97 | 27.9% | — | NOISY |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 3.651459 | 3565.88 | 373.31 | 5.9% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 3.741292 | 3653.61 | 417.01 | 14.8% | — | NOISY |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 2048 | 3.426000 | 1672.85 | 459.77 | 19.2% | — | NOISY |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 1024 | 1.562541 | 1525.92 | 48.87 | 12.0% | — | NOISY |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 2048 | 3.142125 | 1534.24 | 430.64 | 25.3% | — | NOISY |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 1024 | 3.546458 | 3463.34 | 301.25 | 7.3% | — | ok |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 1024 | 3.439708 | 3359.09 | 122.52 | 8.7% | — | ok |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 1024 | 3.762458 | 3674.28 | 823.57 | 16.3% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.792125 | 1750.12 | 130.09 | 34.8% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 1.542208 | 3012.12 | 624.19 | 11.7% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.882917 | 1838.79 | 376.34 | 13.9% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 3.105042 | 3032.27 | 587.81 | 17.1% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 128 | 2.632834 | 20569.02 | 3427.90 | 15.1% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 2.324833 | 9081.38 | 1153.65 | 13.9% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 128 | 2.467417 | 19276.70 | 3178.71 | 16.1% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 2.262167 | 8836.59 | 1028.73 | 22.1% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.703750 | 1663.82 | 107.52 | 8.4% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.201084 | 4298.99 | 531.45 | 22.8% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.652750 | 1614.01 | 88.48 | 6.4% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.391041 | 4670.00 | 2363.93 | 31.8% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.589333 | 2528.65 | 330.91 | 12.3% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.751459 | 5373.94 | 1317.83 | 26.5% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 1.231541 | 2405.35 | 147.79 | 6.0% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.805459 | 5479.41 | 655.64 | 11.2% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 0.783625 | 1530.52 | 131.35 | 14.5% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.251583 | 4397.62 | 541.02 | 31.0% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.607375 | 1569.70 | 153.91 | 31.7% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.473625 | 4831.30 | 872.07 | 21.3% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 1.949250 | 3807.13 | 398.56 | 12.8% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.539417 | 2479.90 | 280.21 | 15.9% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.006459 | 3918.87 | 574.18 | 13.9% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 3.106333 | 3033.53 | 2312.26 | 35.4% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 2.358917 | 9214.52 | 1944.91 | 19.4% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 1.716542 | 6705.24 | 5722.49 | 41.8% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 2.287083 | 8933.92 | 875.00 | 15.9% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 128 | 0.705208 | 5509.44 | 1207.68 | 18.5% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 3.366459 | 6575.12 | 741.58 | 18.9% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 1.496916 | 2923.66 | 389.24 | 16.5% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 3.421625 | 6682.86 | 1057.90 | 14.0% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 1.610208 | 3144.94 | 266.40 | 6.5% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.929209 | 1883.99 | 355.37 | 34.2% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.755000 | 1713.87 | 123.41 | 5.1% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 3.328542 | 1625.26 | 85.39 | 5.3% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.438458 | 2381.31 | 93.91 | 13.7% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 64 | 0.367042 | 5735.03 | 1908.20 | 19.4% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 1.455333 | 2842.45 | 166.18 | 23.0% | — | NOISY |

## Matched non-AD eager comparisons (#98)

All requested QR/SVD outputs are retained. Ordinary einsum uses the same parsed integer notation and default Auto strategy; planning/cache lookup stays inside both public calls. These are public-call diagnostics, not prepared execution or an isolated AD cost. Fresh pairs include session entry; shared pairs exclude it. Views are constructed untimed and consumed as views on both routes. No output to_tensor conversion is timed. Capture is inactive by construction in each fresh process, with no capture_trace guard; inputs and validation outputs are checked untracked.

| Operation | Shape | Layout | Session | Threads | Direct ns | Eager ns | Eager − direct ns | Eager/direct | Quality |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1453.90 | 2590.45 | 1136.55 | 1.782 | inconclusive |
| einsum | 2×2 | col_major_contiguous | shared | 1 | 2116.54 | 3089.97 | 973.43 | 1.460 | inconclusive |
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 4 | 1750.12 | 3012.12 | 1262.00 | 1.721 | inconclusive |
| einsum | 2×2 | col_major_contiguous | shared | 4 | 1838.79 | 3032.27 | 1193.48 | 1.649 | inconclusive |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 1 | 2592.94 | 3378.66 | 785.72 | 1.303 | inconclusive |
| einsum | 16×16 | col_major_contiguous | shared | 1 | 1770.35 | 2967.29 | 1196.94 | 1.676 | inconclusive |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 4 | 3033.53 | 3918.87 | 885.34 | 1.292 | inconclusive |
| einsum | 16×16 | col_major_contiguous | shared | 4 | 2479.90 | 3807.13 | 1327.23 | 1.535 | inconclusive |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 1 | 24181.64 | 8390.46 | -15791.18 | 0.347 | inconclusive |
| einsum3 | 2×2 | col_major_contiguous | shared | 1 | 24180.99 | 7565.10 | -16615.89 | 0.313 | inconclusive |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 4 | 20569.02 | 9081.38 | -11487.64 | 0.442 | inconclusive |
| einsum3 | 2×2 | col_major_contiguous | shared | 4 | 19276.70 | 8836.59 | -10440.11 | 0.458 | inconclusive |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 1 | 2134.20 | 3378.91 | 1244.71 | 1.583 | inconclusive |
| qr | 2×2 | col_major_contiguous | shared | 1 | 1856.73 | 3611.08 | 1754.35 | 1.945 | inconclusive |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 4 | 1663.82 | 4298.99 | 2635.17 | 2.584 | inconclusive |
| qr | 2×2 | col_major_contiguous | shared | 4 | 1614.01 | 4670.00 | 3055.99 | 2.893 | inconclusive |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 1 | 7478.60 | 9391.93 | 1913.33 | 1.256 | inconclusive |
| qr | 16×16 | col_major_contiguous | shared | 1 | 7775.31 | 8430.66 | 655.35 | 1.084 | inconclusive |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 4 | 5509.44 | 8933.92 | 3424.48 | 1.622 | inconclusive |
| qr | 16×16 | col_major_contiguous | shared | 4 | 6705.24 | 9214.52 | 2509.28 | 1.374 | inconclusive |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1125.77 | 4281.90 | 3156.13 | 3.804 | inconclusive |
| solve | 2×2 | col_major_contiguous | shared | 1 | 989.26 | 3827.47 | 2838.21 | 3.869 | inconclusive |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 4 | 1530.52 | 4397.62 | 2867.10 | 2.873 | inconclusive |
| solve | 2×2 | col_major_contiguous | shared | 4 | 1569.70 | 4831.30 | 3261.60 | 3.078 | inconclusive |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 1 | 2340.01 | 5870.04 | 3530.03 | 2.509 | inconclusive |
| solve | 16×16 | col_major_contiguous | shared | 1 | 2343.83 | 5384.61 | 3040.78 | 2.297 | inconclusive |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 4 | 3144.94 | 6682.86 | 3537.92 | 2.125 | inconclusive |
| solve | 16×16 | col_major_contiguous | shared | 4 | 2923.66 | 6575.12 | 3651.46 | 2.249 | inconclusive |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1777.75 | 4378.91 | 2601.16 | 2.463 | inconclusive |
| svd | 2×2 | col_major_contiguous | shared | 1 | 1861.41 | 5260.74 | 3399.33 | 2.826 | inconclusive |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 4 | 2528.65 | 5373.94 | 2845.29 | 2.125 | inconclusive |
| svd | 2×2 | col_major_contiguous | shared | 4 | 2405.35 | 5479.41 | 3074.06 | 2.278 | inconclusive |

## Separate allocation diagnostics

Instrumented executable is built separately after latency collection. Calling-thread Rust allocator requests only; excludes foreign BLAS and worker-thread allocations. Realloc counts its requested new size, not net retained bytes. Output destruction and instrumentation setup are excluded. Helper references and multi-output retention use stack storage. No allocation-instrumented run contributes latency samples.

| Corresponding case ID | Threads | Median allocation requests/op | Median requested bytes/op | Status |
|---|---:|---:|---:|---|
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | 1 | 8 | 120 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | 1 | 15 | 3976 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | 1 | 8 | 120 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-shared | 1 | 15 | 3976 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | 1 | 266 | 42709 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | 1 | 48 | 19712 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | 1 | 266 | 42709 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | 1 | 48 | 19712 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | 1 | 6 | 1760 | passed |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | 1 | 39 | 15859 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-shared | 1 | 6 | 1760 | passed |
| nonad_qr_n2_col_major_contiguous_eager-shared | 1 | 39 | 15859 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | 1 | 7 | 2112 | passed |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | 1 | 45 | 17299 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-shared | 1 | 7 | 2112 | passed |
| nonad_svd_n2_col_major_contiguous_eager-shared | 1 | 45 | 17299 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | 1 | 2 | 48 | passed |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | 1 | 35 | 16083 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-shared | 1 | 2 | 48 | passed |
| nonad_solve_n2_col_major_contiguous_eager-shared | 1 | 35 | 16083 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-shared | 1 | 15 | 3976 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | 1 | 8 | 120 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | 1 | 15 | 3976 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | 1 | 8 | 120 | passed |
| nonad_qr_n16_col_major_contiguous_eager-shared | 1 | 39 | 21459 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-shared | 1 | 6 | 7360 | passed |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | 1 | 39 | 21459 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | 1 | 6 | 7360 | passed |
| nonad_solve_n16_col_major_contiguous_eager-shared | 1 | 35 | 18099 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-shared | 1 | 2 | 2064 | passed |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | 1 | 35 | 18099 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | 1 | 2 | 2064 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | 1 | 8 | 120 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | 1 | 6 | 1760 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | 1 | 2 | 48 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | 1 | 8 | 120 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | 1 | 6 | 7360 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | 1 | 2 | 2064 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | 4 | 8 | 120 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | 4 | 15 | 3976 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | 4 | 8 | 120 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-shared | 4 | 15 | 3976 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | 4 | 266 | 42709 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | 4 | 48 | 19712 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | 4 | 266 | 42709 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | 4 | 48 | 19712 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | 4 | 6 | 1760 | passed |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | 4 | 39 | 15859 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-shared | 4 | 6 | 1760 | passed |
| nonad_qr_n2_col_major_contiguous_eager-shared | 4 | 39 | 15859 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | 4 | 7 | 2112 | passed |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | 4 | 45 | 17299 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-shared | 4 | 7 | 2112 | passed |
| nonad_svd_n2_col_major_contiguous_eager-shared | 4 | 45 | 17299 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | 4 | 2 | 48 | passed |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | 4 | 35 | 16083 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-shared | 4 | 2 | 48 | passed |
| nonad_solve_n2_col_major_contiguous_eager-shared | 4 | 35 | 16083 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-shared | 4 | 15 | 3976 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | 4 | 8 | 120 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | 4 | 15 | 3976 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | 4 | 8 | 120 | passed |
| nonad_qr_n16_col_major_contiguous_eager-shared | 4 | 39 | 21459 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-shared | 4 | 6 | 7360 | passed |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | 4 | 39 | 21459 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | 4 | 6 | 7360 | passed |
| nonad_solve_n16_col_major_contiguous_eager-shared | 4 | 35 | 18099 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-shared | 4 | 2 | 2064 | passed |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | 4 | 35 | 18099 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | 4 | 2 | 2064 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | 4 | 8 | 120 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | 4 | 6 | 1760 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | 4 | 2 | 48 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | 4 | 8 | 120 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | 4 | 6 | 7360 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | 4 | 2 | 2064 | passed |

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
  "collection_total_seconds": 0.3504850830004216,
  "case_process_seconds": 0.3189158380018853
}
```

Wall observations `wall_allocations_t4.json`:
```json
{
  "collection_total_seconds": 0.4034745409999232,
  "case_process_seconds": 0.37043729100059863
}
```

Wall observations `wall_command.json`:
```json
{
  "command_seconds_through_collection": 538.995184667,
  "scope": "command startup through collection; excludes report rendering and publication",
  "budget_seconds": 120.0,
  "budget_status": "over_budget; diagnostic only"
}
```

Wall observations `wall_measure_t1.json`:
```json
{
  "collection_total_seconds": 4.1179898339996726,
  "case_process_seconds": 4.052052539003853
}
```

Wall observations `wall_measure_t4.json`:
```json
{
  "collection_total_seconds": 4.1736812500003,
  "case_process_seconds": 4.111608164995232
}
```

Wall observations `wall_phases.json`:
```json
{
  "validation_and_latency_build_seconds": 10.803830833,
  "metadata_and_latency_collection_seconds": 20.254483042,
  "allocation_diagnostic_build_seconds": 506.793410208,
  "allocation_collection_seconds": 1.011589834,
  "command_before_report_seconds": 538.863313917
}
```

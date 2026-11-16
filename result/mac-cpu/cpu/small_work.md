# CPU Small-Work Benchmark Results

- Suite: `cpu/small_work`
- Raw run: `data/results/mac-cpu/cpu/small_work/20261116_180013`

## CPU Information

- Model: `Apple M2 Pro (Virtual)`
- Vendor: `Apple`
- Logical CPUs: `5`
- Physical CPUs: `5`
- Sockets: `1`
- Cores per socket: `5`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Standard: 5 physical / 5 logical (L1i 128 KiB, L1d 64 KiB, L2 4 MiB, 5 CPUs/L2)`
- Python platform: `macOS-26.7.2-arm64-arm-64bit-Mach-O`
- Container: `none`

## run_t1

Full metadata: `data/results/mac-cpu/cpu/small_work/20261116_180013/run_t1.yaml`

```yaml
timestamp: '2026-11-16T18:00:13Z'
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
  hostname: iad11-ei409-ba2c08c4-104f-4d0d-a740-6497af0be0c1-EE80FE6820DD.local
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
    BENCHMARK_COMMIT: 5c1d5a382dd88d20901598318730d850d3568ac8
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
    RUSTC_VERSION: rustc 1.100.0 (0ef1bc386 2026-11-09)
    BENCH_INSTANCE: null
```

## run_t4

Full metadata: `data/results/mac-cpu/cpu/small_work/20261116_180013/run_t4.yaml`

```yaml
timestamp: '2026-11-16T18:00:20Z'
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
  hostname: iad11-ei409-ba2c08c4-104f-4d0d-a740-6497af0be0c1-EE80FE6820DD.local
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
    BENCHMARK_COMMIT: 5c1d5a382dd88d20901598318730d850d3568ac8
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
    RUSTC_VERSION: rustc 1.100.0 (0ef1bc386 2026-11-09)
    BENCH_INSTANCE: null
```

## Case status

- Manifest: `cpu/small_work` version 3, coverage `quick`, effort `standard`
- Expected 144, selected 144, executed 144, unsupported 0, failed 0, skipped 0, missing 0, noisy 47.
- Complete: **yes** — only executed cases count as covered; unsupported, skipped, failed, missing and unselected cases never do.

| expected | selected | executed | unsupported | failed | skipped | missing | noisy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 144 | 144 | 144 | 0 | 0 | 0 | 0 | 47 |

- Noisy: `add_f64_concrete-shared_size256_dependent10@t1`, `einsum_f64_prepared-repeat_n2_single@t1`, `einsum_f64_prepared-repeat_n4_single@t1`, `nonad_einsum_n2_col_major_contiguous_concrete-fresh@t1`, `nonad_solve_n16_col_major_contiguous_concrete-fresh@t1`, `nonad_solve_n2_col_major_contiguous_concrete-scoped@t1`, `add_f64_concrete-shared_size4_single@t4`, `add_f64_concrete_shared@t4`, `add_f64_concrete-shared_size16_dependent10@t4`, `add_f64_concrete-shared_size256_dependent10@t4`, `solve_f64_concrete-shared_n16_single@t4`, `einsum_f64_prepared-repeat_n8_single@t4`, `einsum_c64_prepared-repeat_n8_single@t4`, `gather_f64_concrete-shared_size4_single@t4`, `gather_f64_concrete_shared@t4`, `gather_f64_concrete-shared_size256_single@t4`, `reduce_sum_f64_concrete-shared_size4_single@t4`, `reduce_sum_f64_concrete-shared_size16_single@t4`, `einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single@t4`, `einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single@t4`, `einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single@t4`, `einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single@t4`, `nonad_einsum_n2_col_major_contiguous_concrete-fresh@t4`, `nonad_einsum_n2_col_major_contiguous_eager-no-ad@t4`, `nonad_einsum_n2_col_major_contiguous_eager-shared@t4`, `nonad_einsum3_n2_col_major_contiguous_concrete-fresh@t4`, `nonad_einsum3_n2_col_major_contiguous_eager-no-ad@t4`, `nonad_qr_n2_col_major_contiguous_eager-no-ad@t4`, `nonad_qr_n2_col_major_contiguous_concrete-shared@t4`, `nonad_qr_n2_col_major_contiguous_eager-shared@t4`, `nonad_svd_n2_col_major_contiguous_eager-no-ad@t4`, `nonad_svd_n2_col_major_contiguous_concrete-shared@t4`, `nonad_solve_n2_col_major_contiguous_eager-no-ad@t4`, `nonad_solve_n2_col_major_contiguous_concrete-shared@t4`, `nonad_solve_n2_col_major_contiguous_eager-shared@t4`, `nonad_einsum_n16_col_major_contiguous_eager-shared@t4`, `nonad_einsum_n16_col_major_contiguous_concrete-shared@t4`, `nonad_einsum_n16_col_major_contiguous_eager-no-ad@t4`, `nonad_einsum_n16_col_major_contiguous_concrete-fresh@t4`, `nonad_qr_n16_col_major_contiguous_eager-shared@t4` (+7 more)

Numerical checks: **144/144 recorded rows passed** (case × thread count).

## Timing

Sequential release runs; explicit initialization and warmups, then calibration toward the declared batch duration (2 ms by default), then the measured batches of the recorded effort (standard: 3 warmups and 15 batches; scan: a low-repetition screen that only marks suspects). Legacy single-call routes start calibration at 1024 workflows; matched non-AD and dependent-chain routes start at 16 and bound retained memory. Each batch uses one wall-clock interval; all outputs are retained until it stops. Statistics divide batch time by iterations. Median and inclusive IQR are in nanoseconds (ns); CoV is sample standard deviation / mean. NOISY means CoV > 10%, a descriptive label only. No noisy rows are excluded.

Workflow time is the total for one call or the complete dependent chain. The separate ns/op column divides the chain median by its operation count, not an independently measured call. Setup rows measure preparation only, not execution.

Inputs and backend construction, independent numerical checks and AD backward checks are outside timing. Returned outputs/plans and every dependent-chain intermediate are retained until after each batch interval; retention containers are prepared before the clock. value downloads/checks are outside. Eager AD rows measure forward execution/recording, not backward.

| Case ID | Operation | API route | Route (path / output / repr / session) | Dtype | Shape | Layout | Workflow | Threads | Provider | Operations/batch | Median batch ms | Median workflow ns | IQR ns | CoV | ns/op (chain only) | Status |
|---|---|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 4096 | 1.984542 | 484.51 | 21.11 | 2.6% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 1 | blas | 5120 | 2.466208 | 4816.81 | 56.27 | 1.1% | 481.68 | ok |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 4096 | 2.026583 | 494.77 | 25.47 | 3.1% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 1 | blas | 5120 | 2.533459 | 4948.16 | 117.43 | 5.2% | 494.82 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 4096 | 2.250417 | 549.42 | 10.09 | 1.6% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 1 | blas | 5120 | 2.847625 | 5561.77 | 1192.18 | 12.5% | 556.18 | NOISY |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 4096 | 3.075333 | 750.81 | 43.17 | 10.2% | — | NOISY |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 4096 | 3.130500 | 764.28 | 21.80 | 12.8% | — | NOISY |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 2048 | 2.566333 | 1253.09 | 72.27 | 6.9% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 1.905416 | 930.38 | 9.40 | 1.2% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 1 | blas | 2048 | 1.914750 | 934.94 | 10.10 | 0.8% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.447584 | 2390.22 | 225.85 | 5.1% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 4096 | 2.987708 | 729.42 | 7.73 | 3.6% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 4096 | 3.320666 | 810.71 | 42.86 | 5.4% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.183000 | 2131.84 | 74.14 | 2.6% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 2048 | 1.912125 | 933.65 | 42.27 | 3.1% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 1.592209 | 1554.89 | 164.63 | 6.3% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 1 | blas | 2048 | 2.576292 | 1257.96 | 92.16 | 4.7% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 6.486917 | 6334.88 | 367.13 | 3.7% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 4096 | 2.796791 | 682.81 | 27.44 | 4.7% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 4096 | 2.848333 | 695.39 | 17.21 | 2.5% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 2048 | 1.953125 | 953.67 | 52.98 | 5.4% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 4096 | 3.125334 | 763.02 | 68.51 | 5.3% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 4096 | 3.024458 | 738.39 | 21.48 | 2.2% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 4096 | 3.129958 | 764.15 | 23.84 | 1.9% | — | ok |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 2.245208 | 2192.59 | 19.31 | 1.7% | — | ok |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 2.365208 | 2309.77 | 52.84 | 1.5% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 2.482875 | 2424.68 | 54.81 | 3.7% | — | ok |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 4096 | 3.604791 | 880.08 | 31.49 | 3.3% | — | ok |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 4096 | 3.793208 | 926.08 | 63.04 | 4.6% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 4096 | 3.501583 | 854.88 | 85.65 | 5.3% | — | ok |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 1024 | 3.051208 | 2979.70 | 36.66 | 3.9% | — | ok |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 1024 | 3.179084 | 3104.57 | 59.94 | 3.5% | — | ok |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 1024 | 3.129417 | 3056.07 | 78.47 | 4.7% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.357000 | 1150.88 | 178.98 | 10.6% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.201792 | 2150.19 | 233.34 | 8.6% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.357375 | 1151.06 | 78.01 | 7.9% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.247959 | 2195.27 | 137.47 | 3.2% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 128 | 2.399584 | 18746.75 | 373.70 | 3.8% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 3.251375 | 6350.34 | 213.46 | 5.3% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 128 | 2.160833 | 16881.51 | 2287.43 | 7.3% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.935458 | 5733.32 | 582.97 | 5.8% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.215667 | 1081.87 | 61.42 | 5.6% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 3.086291 | 3013.96 | 368.69 | 8.8% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.176583 | 1062.78 | 108.68 | 7.0% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 3.101459 | 3028.77 | 374.25 | 8.8% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 3.309583 | 1616.01 | 168.66 | 6.3% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.126209 | 4152.75 | 387.82 | 5.9% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 1.704375 | 1664.43 | 146.97 | 7.8% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 4.182167 | 4084.15 | 252.85 | 7.2% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 4096 | 3.854958 | 941.15 | 48.01 | 4.3% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 3.439375 | 3358.76 | 357.99 | 8.3% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 4096 | 3.928750 | 959.17 | 77.97 | 7.4% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 1.999250 | 3904.79 | 155.60 | 5.2% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 3.010250 | 2939.70 | 195.13 | 4.7% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 1.902208 | 1857.62 | 100.44 | 4.3% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 3.229042 | 3153.36 | 383.28 | 6.9% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 1.744958 | 1704.06 | 149.45 | 5.2% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 1.749833 | 6835.29 | 308.84 | 3.9% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.314917 | 4521.32 | 329.43 | 6.9% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 1.961917 | 7663.74 | 426.03 | 3.7% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.392333 | 4672.53 | 395.22 | 5.4% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.749875 | 5370.85 | 224.81 | 4.5% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.418125 | 2361.45 | 371.28 | 8.0% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.854750 | 5575.68 | 420.21 | 5.4% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.179209 | 2128.13 | 376.48 | 10.2% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.023416 | 988.00 | 71.87 | 4.5% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.156791 | 1053.12 | 39.29 | 4.4% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 1.871250 | 913.70 | 170.94 | 10.6% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 1.704542 | 1664.59 | 132.77 | 5.2% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.171459 | 4241.13 | 139.85 | 4.2% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 2.110125 | 2060.67 | 94.48 | 2.8% | — | ok |
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 2048 | 2.003250 | 978.15 | 25.54 | 12.1% | — | NOISY |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 4 | blas | 2560 | 2.464250 | 9625.98 | 1273.19 | 9.5% | 962.60 | ok |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 2048 | 2.029125 | 990.78 | 19.12 | 10.7% | — | NOISY |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 4 | blas | 2560 | 2.551625 | 9967.29 | 1777.75 | 28.0% | 996.73 | NOISY |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 2048 | 2.101209 | 1025.98 | 41.30 | 7.3% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 4 | blas | 1280 | 1.312792 | 10256.19 | 305.83 | 11.8% | 1025.62 | NOISY |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 2.466875 | 1204.53 | 38.63 | 2.9% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 2048 | 2.538834 | 1239.67 | 100.85 | 6.3% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 1.937750 | 1892.33 | 35.97 | 9.4% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.580417 | 1543.38 | 104.80 | 7.0% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 4 | blas | 2048 | 3.190417 | 1557.82 | 219.33 | 8.5% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 3.304917 | 3227.46 | 273.84 | 8.0% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 2.579667 | 1259.60 | 90.12 | 5.0% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 2048 | 2.681167 | 1309.16 | 100.46 | 4.4% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.533292 | 2473.92 | 272.20 | 19.4% | — | NOISY |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 1.446917 | 1413.00 | 188.15 | 15.4% | — | NOISY |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 2.226625 | 2174.44 | 36.23 | 2.2% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 1.861375 | 1817.75 | 237.41 | 18.1% | — | NOISY |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 7.540416 | 7363.69 | 118.51 | 1.2% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 2048 | 2.332875 | 1139.10 | 167.56 | 10.7% | — | NOISY |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 2048 | 2.372167 | 1158.28 | 189.52 | 34.7% | — | NOISY |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 1.458375 | 1424.19 | 249.61 | 13.0% | — | NOISY |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 2048 | 2.502834 | 1222.09 | 142.96 | 12.9% | — | NOISY |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 2048 | 2.740958 | 1338.36 | 234.54 | 16.7% | — | NOISY |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 2048 | 2.546750 | 1243.53 | 65.57 | 3.2% | — | ok |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 3.066959 | 2995.08 | 62.42 | 1.7% | — | ok |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 3.338333 | 3260.09 | 63.44 | 1.3% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 3.320041 | 3242.23 | 170.98 | 5.6% | — | ok |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 2048 | 2.725916 | 1331.01 | 319.75 | 21.8% | — | NOISY |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 1024 | 1.415791 | 1382.61 | 32.31 | 1.5% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 1024 | 1.451000 | 1416.99 | 407.76 | 31.8% | — | NOISY |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 1024 | 3.615292 | 3530.56 | 140.03 | 20.6% | — | NOISY |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 1024 | 3.669792 | 3583.78 | 76.92 | 4.5% | — | ok |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 1024 | 3.793750 | 3704.83 | 324.79 | 18.6% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 2.972083 | 1451.21 | 162.71 | 25.2% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.551375 | 2491.58 | 591.33 | 25.9% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 3.048250 | 1488.40 | 110.65 | 4.4% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.607458 | 2546.35 | 434.14 | 25.2% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 128 | 2.191541 | 17121.41 | 456.22 | 24.8% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 128 | 0.928375 | 7252.93 | 200.68 | 10.8% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 128 | 2.176375 | 17002.93 | 480.30 | 8.8% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 3.639958 | 7109.29 | 292.48 | 2.6% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 2.885125 | 1408.75 | 63.14 | 3.3% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 1.817333 | 3549.48 | 829.02 | 27.6% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.483500 | 1448.73 | 218.69 | 36.7% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 3.473167 | 3391.76 | 243.55 | 10.3% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.021292 | 1973.92 | 37.70 | 1.9% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.186917 | 4271.32 | 1731.08 | 37.1% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.104916 | 2055.58 | 349.98 | 40.8% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.188041 | 4273.52 | 260.99 | 9.9% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 2.639167 | 1288.66 | 87.80 | 6.5% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 1.900167 | 3711.26 | 646.61 | 38.6% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 2.662000 | 1299.80 | 260.06 | 24.7% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 0.993292 | 3880.05 | 78.78 | 32.5% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 1.880667 | 3673.18 | 403.04 | 15.9% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.296625 | 2242.80 | 278.36 | 15.6% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.034000 | 3972.66 | 585.21 | 10.8% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.325875 | 2271.36 | 236.43 | 25.0% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 1.793125 | 7004.39 | 960.12 | 25.6% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.433292 | 4752.52 | 945.07 | 25.8% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 1.772084 | 6922.20 | 146.16 | 3.0% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 1.214042 | 4742.35 | 165.45 | 23.6% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.536584 | 4954.27 | 1137.21 | 31.8% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.514042 | 2455.12 | 389.67 | 28.3% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.612834 | 5103.19 | 258.02 | 7.9% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.532791 | 2473.43 | 60.16 | 5.2% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 3.040458 | 1484.60 | 149.72 | 5.5% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.600042 | 1562.54 | 185.97 | 12.0% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.423875 | 1390.50 | 362.65 | 35.6% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.368125 | 2312.62 | 197.20 | 6.9% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 1.192250 | 4657.23 | 93.18 | 1.2% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 2.568292 | 2508.10 | 345.11 | 11.4% | — | NOISY |

## Matched non-AD eager comparisons (#98)

All requested QR/SVD outputs are retained. Ordinary einsum uses the same parsed integer notation and default Auto strategy; planning/cache lookup stays inside both public calls. These are public-call diagnostics, not prepared execution or an isolated AD cost. Fresh pairs include session entry; shared pairs exclude it. Views are constructed untimed and consumed as views on both routes. No output to_tensor conversion is timed. Capture is inactive by construction in each fresh process, with no capture_trace guard; inputs and validation outputs are checked untracked.

| Operation | Shape | Layout | Session | Threads | Direct ns | Eager ns | Eager − direct ns | Eager/direct | Quality |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1150.88 | 2150.19 | 999.31 | 1.868 | inconclusive |
| einsum | 2×2 | col_major_contiguous | shared | 1 | 1151.06 | 2195.27 | 1044.21 | 1.907 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 4 | 1451.21 | 2491.58 | 1040.37 | 1.717 | inconclusive |
| einsum | 2×2 | col_major_contiguous | shared | 4 | 1488.40 | 2546.35 | 1057.95 | 1.711 | inconclusive |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 1 | 1704.06 | 3153.36 | 1449.30 | 1.850 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | shared | 1 | 1857.62 | 2939.70 | 1082.08 | 1.583 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 4 | 2271.36 | 3972.66 | 1701.30 | 1.749 | inconclusive |
| einsum | 16×16 | col_major_contiguous | shared | 4 | 2242.80 | 3673.18 | 1430.38 | 1.638 | inconclusive |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 1 | 18746.75 | 6350.34 | -12396.41 | 0.339 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | shared | 1 | 16881.51 | 5733.32 | -11148.19 | 0.340 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 4 | 17121.41 | 7252.93 | -9868.48 | 0.424 | inconclusive |
| einsum3 | 2×2 | col_major_contiguous | shared | 4 | 17002.93 | 7109.29 | -9893.64 | 0.418 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1081.87 | 3013.96 | 1932.09 | 2.786 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | shared | 1 | 1062.78 | 3028.77 | 1965.99 | 2.850 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 4 | 1408.75 | 3549.48 | 2140.73 | 2.520 | inconclusive |
| qr | 2×2 | col_major_contiguous | shared | 4 | 1448.73 | 3391.76 | 1943.03 | 2.341 | inconclusive |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 1 | 4672.53 | 7663.74 | 2991.21 | 1.640 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | shared | 1 | 4521.32 | 6835.29 | 2313.97 | 1.512 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 4 | 4742.35 | 6922.20 | 2179.85 | 1.460 | inconclusive |
| qr | 16×16 | col_major_contiguous | shared | 4 | 4752.52 | 7004.39 | 2251.87 | 1.474 | inconclusive |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 1 | 941.15 | 3358.76 | 2417.61 | 3.569 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | shared | 1 | 959.17 | 3904.79 | 2945.62 | 4.071 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 4 | 1288.66 | 3711.26 | 2422.60 | 2.880 | inconclusive |
| solve | 2×2 | col_major_contiguous | shared | 4 | 1299.80 | 3880.05 | 2580.25 | 2.985 | inconclusive |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 1 | 2128.13 | 5575.68 | 3447.55 | 2.620 | inconclusive |
| solve | 16×16 | col_major_contiguous | shared | 1 | 2361.45 | 5370.85 | 3009.40 | 2.274 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 4 | 2473.43 | 5103.19 | 2629.76 | 2.063 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | shared | 4 | 2455.12 | 4954.27 | 2499.15 | 2.018 | inconclusive |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1616.01 | 4152.75 | 2536.74 | 2.570 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | shared | 1 | 1664.43 | 4084.15 | 2419.72 | 2.454 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 4 | 1973.92 | 4271.32 | 2297.40 | 2.164 | inconclusive |
| svd | 2×2 | col_major_contiguous | shared | 4 | 2055.58 | 4273.52 | 2217.94 | 2.079 | inconclusive |

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
  "collection_total_seconds": 0.2473182089997863,
  "case_process_seconds": 0.2199075010012166
}
```

Wall observations `wall_allocations_t4.json`:
```json
{
  "collection_total_seconds": 0.2579631249996055,
  "case_process_seconds": 0.2299222930000724
}
```

Wall observations `wall_command.json`:
```json
{
  "command_seconds_through_collection": 206.382287791,
  "scope": "command startup through collection; excludes report rendering and publication",
  "budget_seconds": 120.0,
  "budget_status": "over_budget; diagnostic only"
}
```

Wall observations `wall_measure_t1.json`:
```json
{
  "collection_total_seconds": 3.784609666999586,
  "case_process_seconds": 3.7439916209982584
}
```

Wall observations `wall_measure_t4.json`:
```json
{
  "collection_total_seconds": 3.7264926670000023,
  "case_process_seconds": 3.678923169997688
}
```

Wall observations `wall_phases.json`:
```json
{
  "validation_and_latency_build_seconds": 4.536464875,
  "metadata_and_latency_collection_seconds": 13.189064416,
  "allocation_diagnostic_build_seconds": 187.894595167,
  "allocation_collection_seconds": 0.678976375,
  "command_before_report_seconds": 206.299100833
}
```

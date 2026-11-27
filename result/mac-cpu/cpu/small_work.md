# CPU Small-Work Benchmark Results

- Suite: `cpu/small_work`
- Raw run: `data/results/mac-cpu/cpu/small_work/20261127_183506`

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

Full metadata: `data/results/mac-cpu/cpu/small_work/20261127_183506/run_t1.yaml`

```yaml
timestamp: '2026-11-27T18:35:07Z'
tenferro_rs:
  path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - native
  resolved_path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
environment:
  container: none
  hostname: iad20-fj919-5897566c-b56f-42d6-8a4f-ca512393d6c5-3EBFBE4F73D6.local
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
    BENCHMARK_COMMIT: 7da876a214acb65eb9ea099adef67b41ef8efc98
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

Full metadata: `data/results/mac-cpu/cpu/small_work/20261127_183506/run_t4.yaml`

```yaml
timestamp: '2026-11-27T18:35:16Z'
tenferro_rs:
  path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - native
  resolved_path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
environment:
  container: none
  hostname: iad20-fj919-5897566c-b56f-42d6-8a4f-ca512393d6c5-3EBFBE4F73D6.local
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
    BENCHMARK_COMMIT: 7da876a214acb65eb9ea099adef67b41ef8efc98
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
- Expected 144, selected 144, executed 144, unsupported 0, failed 0, skipped 0, missing 0, noisy 46.
- Complete: **yes** — only executed cases count as covered; unsupported, skipped, failed, missing and unselected cases never do.

| expected | selected | executed | unsupported | failed | skipped | missing | noisy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 144 | 144 | 144 | 0 | 0 | 0 | 0 | 46 |

- Noisy: `add_f64_concrete-shared_size256_single@t1`, `einsum_c64_prepared-repeat_n4_single@t1`, `solve_f64_concrete-shared_n4_single@t1`, `einsum_c64_prepared-repeat_n32_single@t1`, `gather_f64_concrete-shared_size4_single@t1`, `gather_f64_concrete-shared_size256_single@t1`, `nonad_einsum_n2_col_major_contiguous_eager-shared@t1`, `nonad_qr_n2_col_major_contiguous_eager-shared@t1`, `nonad_svd_n2_col_major_contiguous_eager-no-ad@t1`, `nonad_svd_n2_col_major_contiguous_concrete-shared@t1`, `nonad_svd_n2_col_major_contiguous_eager-shared@t1`, `nonad_solve_n2_col_major_contiguous_eager-no-ad@t1`, `nonad_solve_n2_col_major_contiguous_eager-shared@t1`, `nonad_einsum_n16_col_major_contiguous_concrete-fresh@t1`, `nonad_solve_n16_col_major_contiguous_eager-no-ad@t1`, `nonad_solve_n16_col_major_contiguous_concrete-fresh@t1`, `nonad_einsum_n2_col_major_contiguous_concrete-scoped@t1`, `nonad_einsum_n16_col_major_contiguous_concrete-scoped@t1`, `nonad_qr_n16_col_major_contiguous_concrete-scoped@t1`, `nonad_solve_n16_col_major_contiguous_concrete-scoped@t1`, `add_f64_concrete-shared_size4_dependent10@t4`, `add_f64_concrete-shared_size256_single@t4`, `einsum_f64_prepared-repeat_n16_single@t4`, `einsum_c64_prepared-repeat_n2_single@t4`, `einsum_c64_prepared-repeat_n16_single@t4`, `solve_f64_concrete-shared_n16_single@t4`, `einsum_f64_prepared-repeat_n32_single@t4`, `einsum_c64_prepared-repeat_n8_single@t4`, `einsum_c64_prepared-repeat_n32_single@t4`, `gather_f64_concrete-shared_size4_single@t4`, `einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single@t4`, `einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single@t4`, `einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single@t4`, `einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single@t4`, `nonad_einsum_n2_col_major_contiguous_concrete-shared@t4`, `nonad_einsum3_n2_col_major_contiguous_concrete-fresh@t4`, `nonad_einsum3_n2_col_major_contiguous_eager-no-ad@t4`, `nonad_einsum3_n2_col_major_contiguous_concrete-shared@t4`, `nonad_svd_n2_col_major_contiguous_eager-no-ad@t4`, `nonad_solve_n2_col_major_contiguous_concrete-shared@t4` (+6 more)

Numerical checks: **144/144 recorded rows passed** (case × thread count).

## Timing

Sequential release runs; explicit initialization and warmups, then calibration toward the declared batch duration (2 ms by default), then the measured batches of the recorded effort (standard: 3 warmups and 15 batches; scan: a low-repetition screen that only marks suspects). Legacy single-call routes start calibration at 1024 workflows; matched non-AD and dependent-chain routes start at 16 and bound retained memory. Each batch uses one wall-clock interval; all outputs are retained until it stops. Statistics divide batch time by iterations. Median and inclusive IQR are in nanoseconds (ns); CoV is sample standard deviation / mean. NOISY means CoV > 10%, a descriptive label only. No noisy rows are excluded.

Workflow time is the total for one call or the complete dependent chain. The separate ns/op column divides the chain median by its operation count, not an independently measured call. Setup rows measure preparation only, not execution.

Inputs and backend construction, independent numerical checks and AD backward checks are outside timing. Returned outputs/plans and every dependent-chain intermediate are retained until after each batch interval; retention containers are prepared before the clock. value downloads/checks are outside. Eager AD rows measure forward execution/recording, not backward.

| Case ID | Operation | API route | Route (path / output / repr / session) | Dtype | Shape | Layout | Workflow | Threads | Provider | Operations/batch | Median batch ms | Median workflow ns | IQR ns | CoV | ns/op (chain only) | Status |
|---|---|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | faer | 4096 | 2.334708 | 570.00 | 39.45 | 9.3% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 1 | faer | 5120 | 2.814875 | 5497.80 | 189.49 | 2.1% | 549.78 | ok |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | faer | 4096 | 2.387167 | 582.80 | 40.12 | 8.9% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 1 | faer | 5120 | 2.790500 | 5450.20 | 370.32 | 6.7% | 545.02 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | faer | 1024 | 0.696708 | 680.38 | 72.12 | 10.2% | — | NOISY |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 1 | faer | 640 | 0.356334 | 5567.72 | 58.27 | 6.0% | 556.77 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 1.805750 | 881.71 | 29.15 | 5.7% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | faer | 2048 | 1.847833 | 902.26 | 28.71 | 3.1% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 1.381458 | 1349.08 | 41.36 | 2.4% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 1.872292 | 914.21 | 34.70 | 4.0% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 1 | faer | 2048 | 2.056167 | 1003.99 | 164.87 | 18.7% | — | NOISY |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 2.126667 | 2076.82 | 87.14 | 3.0% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 1.879250 | 917.60 | 22.67 | 2.6% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | faer | 2048 | 2.129500 | 1039.79 | 67.55 | 18.5% | — | NOISY |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 2.535084 | 2475.67 | 47.67 | 2.2% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | faer | 2048 | 2.024333 | 988.44 | 67.94 | 5.4% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 1 | faer | 1024 | 3.115958 | 3042.93 | 67.73 | 2.1% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 1 | faer | 2048 | 2.284500 | 1115.48 | 121.18 | 5.8% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 1 | faer | 1024 | 11.323750 | 11058.35 | 1483.93 | 10.0% | — | NOISY |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | faer | 1024 | 0.875750 | 855.22 | 527.57 | 36.7% | — | NOISY |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | faer | 2048 | 1.593292 | 777.97 | 22.08 | 6.1% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | faer | 1024 | 1.171250 | 1143.80 | 63.84 | 15.4% | — | NOISY |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | faer | 2048 | 1.820833 | 889.08 | 23.20 | 2.0% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | faer | 4096 | 3.642667 | 889.32 | 14.38 | 1.1% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | faer | 2048 | 1.858083 | 907.27 | 39.32 | 3.6% | — | ok |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | faer | 1024 | 2.486417 | 2428.14 | 64.68 | 1.6% | — | ok |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | faer | 1024 | 2.607625 | 2546.51 | 152.71 | 7.2% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | faer | 1024 | 2.525625 | 2466.43 | 33.61 | 1.3% | — | ok |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | faer | 1024 | 1.202667 | 1174.48 | 54.73 | 4.4% | — | ok |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | faer | 2048 | 2.569333 | 1254.56 | 46.20 | 9.5% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | faer | 2048 | 2.155000 | 1052.25 | 19.11 | 9.7% | — | ok |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | faer | 1024 | 2.701000 | 2637.70 | 29.97 | 4.2% | — | ok |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | faer | 1024 | 2.797375 | 2731.81 | 50.07 | 1.6% | — | ok |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | faer | 1024 | 2.543916 | 2484.29 | 22.58 | 0.7% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 2.841625 | 1387.51 | 129.18 | 4.9% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 2.436333 | 2379.23 | 168.48 | 4.0% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 2.807791 | 1370.99 | 29.26 | 2.7% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 1.301542 | 2542.07 | 183.02 | 10.5% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 128 | 2.136750 | 16693.36 | 1493.49 | 6.9% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 3.121750 | 6097.17 | 385.46 | 7.5% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 128 | 1.822375 | 14237.30 | 541.66 | 4.0% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 256 | 1.589334 | 6208.34 | 34.02 | 1.2% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 1.996041 | 974.63 | 7.51 | 0.7% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 2.668125 | 2605.59 | 173.38 | 3.8% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 2.015458 | 984.11 | 6.17 | 0.4% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 256 | 0.714458 | 2790.85 | 59.41 | 15.5% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 1.603042 | 1565.47 | 16.54 | 1.4% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 1.894750 | 3700.68 | 406.25 | 12.2% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 1.694000 | 1654.30 | 193.03 | 24.8% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 2.195667 | 4288.41 | 99.77 | 14.5% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 2.282958 | 1114.73 | 18.70 | 1.3% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 3.244125 | 3168.09 | 102.13 | 12.1% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 1.866667 | 911.46 | 34.66 | 6.9% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 1.584500 | 3094.73 | 121.95 | 13.4% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 2.464125 | 2406.37 | 88.46 | 3.3% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 1.633791 | 1595.50 | 93.49 | 8.7% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 2.434167 | 2377.12 | 94.46 | 2.9% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 0.445708 | 1741.05 | 412.68 | 21.9% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 1.548000 | 6046.88 | 38.74 | 0.8% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 512 | 2.192625 | 4282.47 | 16.07 | 0.5% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 512 | 3.051000 | 5958.98 | 32.55 | 0.7% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 1.106875 | 4323.73 | 109.94 | 9.4% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 1.415000 | 5527.34 | 219.73 | 3.5% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 512 | 1.388125 | 2711.18 | 396.48 | 8.2% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 1.716417 | 6704.75 | 340.82 | 21.1% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 512 | 1.429500 | 2791.99 | 165.00 | 20.7% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 2.811000 | 1372.56 | 70.92 | 11.9% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 2.317750 | 1131.71 | 21.81 | 1.4% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 2.132584 | 1041.30 | 21.88 | 1.6% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 1.995209 | 1948.45 | 42.36 | 14.7% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 512 | 2.741625 | 5354.74 | 185.02 | 10.5% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 512 | 1.955542 | 3819.42 | 646.12 | 15.5% | — | NOISY |
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | faer | 4096 | 2.532458 | 618.28 | 24.14 | 4.7% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 4 | faer | 5120 | 3.049291 | 5955.65 | 414.30 | 16.5% | 595.56 | NOISY |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | faer | 1024 | 0.581291 | 567.67 | 5.96 | 0.9% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 4 | faer | 5120 | 2.907916 | 5679.52 | 346.64 | 8.9% | 567.95 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | faer | 1024 | 0.672083 | 656.33 | 104.02 | 11.7% | — | NOISY |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 4 | faer | 640 | 0.408209 | 6378.27 | 146.48 | 1.5% | 637.83 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 2048 | 1.926666 | 940.75 | 27.50 | 1.5% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | faer | 2048 | 2.002625 | 977.84 | 29.86 | 2.6% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 1024 | 1.536583 | 1500.57 | 216.04 | 16.9% | — | NOISY |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 4 | faer | 2048 | 2.593333 | 1266.28 | 270.66 | 11.6% | — | NOISY |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 4 | faer | 2048 | 2.188917 | 1068.81 | 40.19 | 9.7% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 4 | faer | 1024 | 2.346041 | 2291.06 | 164.98 | 11.3% | — | NOISY |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 2048 | 2.252375 | 1099.79 | 34.54 | 5.2% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | faer | 2048 | 2.441750 | 1192.26 | 24.36 | 1.9% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 1024 | 3.127917 | 3054.61 | 949.40 | 19.0% | — | NOISY |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | faer | 2048 | 2.362625 | 1153.63 | 31.77 | 3.5% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 4 | faer | 1024 | 3.696333 | 3609.70 | 925.98 | 36.7% | — | NOISY |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 4 | faer | 2048 | 2.547917 | 1244.10 | 92.53 | 33.8% | — | NOISY |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 4 | faer | 1024 | 12.679542 | 12382.37 | 2404.52 | 20.5% | — | NOISY |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | faer | 4096 | 3.307625 | 807.53 | 35.39 | 37.9% | — | NOISY |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | faer | 2048 | 1.736333 | 847.82 | 66.63 | 6.0% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | faer | 1024 | 1.202667 | 1174.48 | 54.34 | 4.6% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | faer | 2048 | 1.818875 | 888.12 | 10.51 | 2.0% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | faer | 2048 | 1.885916 | 920.86 | 35.11 | 3.8% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | faer | 2048 | 2.056083 | 1003.95 | 26.59 | 3.2% | — | ok |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | faer | 1024 | 2.433541 | 2376.50 | 93.93 | 27.5% | — | NOISY |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | faer | 1024 | 2.531292 | 2471.96 | 30.82 | 1.1% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | faer | 1024 | 3.043916 | 2972.57 | 543.58 | 22.0% | — | NOISY |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | faer | 2048 | 2.316917 | 1131.31 | 24.20 | 1.8% | — | ok |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | faer | 1024 | 1.266750 | 1237.06 | 23.80 | 1.9% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | faer | 1024 | 1.101167 | 1075.36 | 52.55 | 14.3% | — | NOISY |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | faer | 1024 | 2.593375 | 2532.59 | 41.57 | 1.2% | — | ok |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | faer | 1024 | 2.751541 | 2687.05 | 114.30 | 3.5% | — | ok |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | faer | 1024 | 2.455667 | 2398.11 | 61.63 | 11.7% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 2048 | 2.571209 | 1255.47 | 120.69 | 9.0% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 2.254958 | 2202.11 | 186.77 | 5.0% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 2048 | 2.740333 | 1338.05 | 95.89 | 30.3% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 2.310334 | 2256.19 | 128.07 | 4.0% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 128 | 2.222291 | 17361.65 | 2995.28 | 15.1% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 3.548333 | 6930.34 | 6483.81 | 51.6% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 64 | 1.488458 | 23257.16 | 3624.68 | 10.3% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 256 | 1.784084 | 6969.08 | 137.53 | 1.7% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 2048 | 2.434292 | 1188.62 | 30.66 | 2.6% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 3.183958 | 3109.33 | 166.02 | 3.5% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 2048 | 2.671250 | 1304.32 | 50.53 | 5.7% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 1.651417 | 3225.42 | 58.92 | 2.2% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 1.937041 | 1891.64 | 44.47 | 2.8% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 2.123333 | 4147.13 | 977.33 | 47.0% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 1.930333 | 1885.09 | 44.21 | 1.6% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 1.973708 | 3854.90 | 115.03 | 3.2% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 2048 | 2.102083 | 1026.41 | 11.67 | 1.6% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 1.805083 | 3525.55 | 48.22 | 1.1% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 2048 | 2.227667 | 1087.73 | 17.54 | 20.2% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 1.901667 | 3714.19 | 147.46 | 13.2% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 1024 | 3.172750 | 3098.39 | 159.61 | 3.6% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 256 | 0.565583 | 2209.31 | 116.62 | 4.5% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 512 | 1.675083 | 3271.65 | 77.47 | 2.1% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 1024 | 1.992209 | 1945.52 | 57.52 | 3.8% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 256 | 1.927500 | 7529.30 | 4939.78 | 60.2% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 512 | 2.433459 | 4752.85 | 208.54 | 12.2% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 256 | 1.643834 | 6421.23 | 925.05 | 35.0% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 512 | 2.410167 | 4707.36 | 139.85 | 3.6% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 256 | 1.573583 | 6146.81 | 347.66 | 10.4% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 1024 | 2.629542 | 2567.91 | 49.76 | 2.4% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 512 | 3.041541 | 5940.51 | 180.09 | 2.8% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 512 | 1.389541 | 2713.95 | 117.11 | 3.9% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 2048 | 2.842750 | 1388.06 | 133.77 | 6.6% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 1.241667 | 1212.57 | 45.11 | 2.0% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 2048 | 2.161708 | 1055.52 | 30.63 | 2.4% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 1024 | 1.950542 | 1904.83 | 56.15 | 2.1% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 512 | 2.544334 | 4969.40 | 119.51 | 13.8% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 1024 | 2.687750 | 2624.76 | 60.77 | 3.1% | — | ok |

## Matched non-AD eager comparisons (#98)

All requested QR/SVD outputs are retained. Ordinary einsum uses the same parsed integer notation and default Auto strategy; planning/cache lookup stays inside both public calls. These are public-call diagnostics, not prepared execution or an isolated AD cost. Fresh pairs include session entry; shared pairs exclude it. Views are constructed untimed and consumed as views on both routes. No output to_tensor conversion is timed. Capture is inactive by construction in each fresh process, with no capture_trace guard; inputs and validation outputs are checked untracked.

| Operation | Shape | Layout | Session | Threads | Direct ns | Eager ns | Eager − direct ns | Eager/direct | Quality |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1387.51 | 2379.23 | 991.72 | 1.715 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | shared | 1 | 1370.99 | 2542.07 | 1171.08 | 1.854 | inconclusive |
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 4 | 1255.47 | 2202.11 | 946.64 | 1.754 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | shared | 4 | 1338.05 | 2256.19 | 918.14 | 1.686 | inconclusive |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 1 | 1741.05 | 2377.12 | 636.07 | 1.365 | inconclusive |
| einsum | 16×16 | col_major_contiguous | shared | 1 | 1595.50 | 2406.37 | 810.87 | 1.508 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 4 | 1945.52 | 3271.65 | 1326.13 | 1.682 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | shared | 4 | 2209.31 | 3098.39 | 889.08 | 1.402 | inconclusive |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 1 | 16693.36 | 6097.17 | -10596.19 | 0.365 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | shared | 1 | 14237.30 | 6208.34 | -8028.96 | 0.436 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 4 | 17361.65 | 6930.34 | -10431.31 | 0.399 | inconclusive |
| einsum3 | 2×2 | col_major_contiguous | shared | 4 | 23257.16 | 6969.08 | -16288.08 | 0.300 | inconclusive |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 1 | 974.63 | 2605.59 | 1630.96 | 2.673 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | shared | 1 | 984.11 | 2790.85 | 1806.74 | 2.836 | inconclusive |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 4 | 1188.62 | 3109.33 | 1920.71 | 2.616 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | shared | 4 | 1304.32 | 3225.42 | 1921.10 | 2.473 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 1 | 4323.73 | 5958.98 | 1635.25 | 1.378 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | shared | 1 | 4282.47 | 6046.88 | 1764.41 | 1.412 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 4 | 4707.36 | 6421.23 | 1713.87 | 1.364 | inconclusive |
| qr | 16×16 | col_major_contiguous | shared | 4 | 4752.85 | 7529.30 | 2776.45 | 1.584 | inconclusive |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1114.73 | 3168.09 | 2053.36 | 2.842 | inconclusive |
| solve | 2×2 | col_major_contiguous | shared | 1 | 911.46 | 3094.73 | 2183.27 | 3.395 | inconclusive |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 4 | 1026.41 | 3525.55 | 2499.14 | 3.435 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | shared | 4 | 1087.73 | 3714.19 | 2626.46 | 3.415 | inconclusive |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 1 | 2791.99 | 6704.75 | 3912.76 | 2.401 | inconclusive |
| solve | 16×16 | col_major_contiguous | shared | 1 | 2711.18 | 5527.34 | 2816.16 | 2.039 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 4 | 2713.95 | 5940.51 | 3226.56 | 2.189 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | shared | 4 | 2567.91 | 6146.81 | 3578.90 | 2.394 | inconclusive |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1565.47 | 3700.68 | 2135.21 | 2.364 | inconclusive |
| svd | 2×2 | col_major_contiguous | shared | 1 | 1654.30 | 4288.41 | 2634.11 | 2.592 | inconclusive |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 4 | 1891.64 | 4147.13 | 2255.49 | 2.192 | inconclusive |
| svd | 2×2 | col_major_contiguous | shared | 4 | 1885.09 | 3854.90 | 1969.81 | 2.045 | observed; no speedup claim |

## Separate allocation diagnostics

Instrumented executable is built separately after latency collection. Calling-thread Rust allocator requests only; excludes foreign BLAS and worker-thread allocations. Realloc counts its requested new size, not net retained bytes. Output destruction and instrumentation setup are excluded. Helper references and multi-output retention use stack storage. No allocation-instrumented run contributes latency samples.

| Corresponding case ID | Threads | Median allocation requests/op | Median requested bytes/op | Status |
|---|---:|---:|---:|---|
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | 1 | 10 | 1528 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | 1 | 17 | 4152 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | 1 | 10 | 1528 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-shared | 1 | 17 | 4152 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | 1 | 268 | 32941 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | 1 | 51 | 11396 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | 1 | 268 | 32941 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | 1 | 51 | 11396 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | 1 | 1 | 624 | passed |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | 1 | 31 | 5867 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-shared | 1 | 1 | 624 | passed |
| nonad_qr_n2_col_major_contiguous_eager-shared | 1 | 31 | 5867 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | 1 | 1 | 936 | passed |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | 1 | 36 | 7267 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-shared | 1 | 1 | 936 | passed |
| nonad_svd_n2_col_major_contiguous_eager-shared | 1 | 36 | 7267 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | 1 | 0 | 0 | passed |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | 1 | 30 | 7179 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-shared | 1 | 0 | 0 | passed |
| nonad_solve_n2_col_major_contiguous_eager-shared | 1 | 30 | 7179 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-shared | 1 | 17 | 4152 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | 1 | 10 | 1528 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | 1 | 17 | 4152 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | 1 | 10 | 1528 | passed |
| nonad_qr_n16_col_major_contiguous_eager-shared | 1 | 31 | 5867 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-shared | 1 | 1 | 624 | passed |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | 1 | 31 | 5867 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | 1 | 1 | 624 | passed |
| nonad_solve_n16_col_major_contiguous_eager-shared | 1 | 30 | 7179 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-shared | 1 | 0 | 0 | passed |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | 1 | 30 | 7179 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | 1 | 0 | 0 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | 1 | 10 | 1528 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | 1 | 1 | 624 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | 1 | 0 | 0 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | 1 | 10 | 1528 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | 1 | 1 | 624 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | 1 | 0 | 0 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | 4 | 10 | 1528 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | 4 | 17 | 4152 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | 4 | 10 | 1528 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-shared | 4 | 17 | 4152 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | 4 | 268 | 32941 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | 4 | 51 | 11396 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | 4 | 268 | 32941 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | 4 | 51 | 11396 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | 4 | 1 | 624 | passed |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | 4 | 31 | 5867 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-shared | 4 | 1 | 624 | passed |
| nonad_qr_n2_col_major_contiguous_eager-shared | 4 | 31 | 5867 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | 4 | 1 | 936 | passed |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | 4 | 36 | 7267 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-shared | 4 | 1 | 936 | passed |
| nonad_svd_n2_col_major_contiguous_eager-shared | 4 | 36 | 7267 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | 4 | 0 | 0 | passed |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | 4 | 30 | 7179 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-shared | 4 | 0 | 0 | passed |
| nonad_solve_n2_col_major_contiguous_eager-shared | 4 | 30 | 7179 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-shared | 4 | 17 | 4152 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | 4 | 10 | 1528 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | 4 | 17 | 4152 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | 4 | 10 | 1528 | passed |
| nonad_qr_n16_col_major_contiguous_eager-shared | 4 | 31 | 5867 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-shared | 4 | 1 | 624 | passed |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | 4 | 31 | 5867 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | 4 | 1 | 624 | passed |
| nonad_solve_n16_col_major_contiguous_eager-shared | 4 | 30 | 7179 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-shared | 4 | 0 | 0 | passed |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | 4 | 30 | 7179 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | 4 | 0 | 0 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | 4 | 10 | 1528 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | 4 | 1 | 624 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | 4 | 0 | 0 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | 4 | 10 | 1528 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | 4 | 1 | 624 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | 4 | 0 | 0 | passed |

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
  "collection_total_seconds": 0.3867441249994954,
  "case_process_seconds": 0.34316025000134687
}
```

Wall observations `wall_allocations_t4.json`:
```json
{
  "collection_total_seconds": 0.3881805410001107,
  "case_process_seconds": 0.35097249899899907
}
```

Wall observations `wall_command.json`:
```json
{
  "command_seconds_through_collection": 462.924738166,
  "scope": "command startup through collection; excludes report rendering and publication",
  "budget_seconds": 120.0,
  "budget_status": "over_budget; diagnostic only"
}
```

Wall observations `wall_measure_t1.json`:
```json
{
  "collection_total_seconds": 3.4810944580003707,
  "case_process_seconds": 3.4310929150024094
}
```

Wall observations `wall_measure_t4.json`:
```json
{
  "collection_total_seconds": 3.720151249999617,
  "case_process_seconds": 3.668214334004915
}
```

Wall observations `wall_phases.json`:
```json
{
  "validation_and_latency_build_seconds": 8.773451875,
  "metadata_and_latency_collection_seconds": 17.845978291,
  "allocation_diagnostic_build_seconds": 435.046410792,
  "allocation_collection_seconds": 1.115708667,
  "command_before_report_seconds": 462.781549625
}
```

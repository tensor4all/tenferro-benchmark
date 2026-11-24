# CPU Small-Work Benchmark Results

- Suite: `cpu/small_work`
- Raw run: `data/results/mac-cpu/cpu/small_work/20261124_183516`

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

Full metadata: `data/results/mac-cpu/cpu/small_work/20261124_183516/run_t1.yaml`

```yaml
timestamp: '2026-11-24T18:35:17Z'
tenferro_rs:
  path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: ce9125d0cff4ee40841120759b5df45252c27f22
  dirty: false
  features:
  - native
  resolved_path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
environment:
  container: none
  hostname: sat12-bq163-e0ae879f-7bba-49c3-bbc8-29a90090b94e-AEA030E43CDC.local
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
    BENCHMARK_COMMIT: 9185e8ca26c3fc600458361d0f477495a717daf3
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

Full metadata: `data/results/mac-cpu/cpu/small_work/20261124_183516/run_t4.yaml`

```yaml
timestamp: '2026-11-24T18:35:26Z'
tenferro_rs:
  path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: ce9125d0cff4ee40841120759b5df45252c27f22
  dirty: false
  features:
  - native
  resolved_path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
environment:
  container: none
  hostname: sat12-bq163-e0ae879f-7bba-49c3-bbc8-29a90090b94e-AEA030E43CDC.local
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
    BENCHMARK_COMMIT: 9185e8ca26c3fc600458361d0f477495a717daf3
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
- Expected 144, selected 144, executed 144, unsupported 0, failed 0, skipped 0, missing 0, noisy 56.
- Complete: **yes** — only executed cases count as covered; unsupported, skipped, failed, missing and unselected cases never do.

| expected | selected | executed | unsupported | failed | skipped | missing | noisy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 144 | 144 | 144 | 0 | 0 | 0 | 0 | 56 |

- Noisy: `add_f64_concrete-shared_size4_dependent10@t1`, `add_f64_concrete-shared_size16_dependent10@t1`, `add_f64_concrete-shared_size256_single@t1`, `einsum_f64_prepared-repeat_n2_single@t1`, `einsum_c64_prepared-repeat_n32_single@t1`, `gather_f64_concrete-shared_size4_single@t1`, `reduce_sum_f64_concrete-shared_size4_single@t1`, `reduce_sum_f64_concrete-shared_size16_single@t1`, `reduce_sum_f64_concrete-shared_size256_single@t1`, `einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single@t1`, `nonad_einsum_n2_col_major_contiguous_concrete-fresh@t1`, `nonad_einsum3_n2_col_major_contiguous_concrete-shared@t1`, `nonad_einsum3_n2_col_major_contiguous_eager-shared@t1`, `nonad_solve_n2_col_major_contiguous_eager-no-ad@t1`, `nonad_einsum_n16_col_major_contiguous_eager-shared@t1`, `nonad_einsum_n16_col_major_contiguous_concrete-shared@t1`, `nonad_qr_n16_col_major_contiguous_eager-shared@t1`, `nonad_qr_n16_col_major_contiguous_concrete-fresh@t1`, `nonad_solve_n16_col_major_contiguous_concrete-fresh@t1`, `nonad_einsum_n2_col_major_contiguous_concrete-scoped@t1`, `nonad_solve_n2_col_major_contiguous_concrete-scoped@t1`, `add_f64_concrete-shared_size16_dependent10@t4`, `add_f64_concrete-shared_size256_dependent10@t4`, `einsum_f64_prepared-repeat_n2_single@t4`, `einsum_f64_prepared-repeat_n16_single@t4`, `einsum_c64_prepared-repeat_n2_single@t4`, `einsum_c64_prepared-repeat_n16_single@t4`, `solve_f64_concrete-shared_n16_single@t4`, `einsum_f64_prepared-repeat_n32_single@t4`, `gather_f64_concrete-shared_size4_single@t4`, `reduce_sum_f64_concrete-shared_size4_single@t4`, `einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single@t4`, `einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single@t4`, `einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single@t4`, `einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single@t4`, `nonad_einsum_n2_col_major_contiguous_concrete-fresh@t4`, `nonad_einsum_n2_col_major_contiguous_eager-no-ad@t4`, `nonad_einsum_n2_col_major_contiguous_concrete-shared@t4`, `nonad_einsum3_n2_col_major_contiguous_concrete-fresh@t4`, `nonad_einsum3_n2_col_major_contiguous_eager-no-ad@t4` (+16 more)

Numerical checks: **144/144 recorded rows passed** (case × thread count).

## Timing

Sequential release runs; explicit initialization and warmups, then calibration toward the declared batch duration (2 ms by default), then the measured batches of the recorded effort (standard: 3 warmups and 15 batches; scan: a low-repetition screen that only marks suspects). Legacy single-call routes start calibration at 1024 workflows; matched non-AD and dependent-chain routes start at 16 and bound retained memory. Each batch uses one wall-clock interval; all outputs are retained until it stops. Statistics divide batch time by iterations. Median and inclusive IQR are in nanoseconds (ns); CoV is sample standard deviation / mean. NOISY means CoV > 10%, a descriptive label only. No noisy rows are excluded.

Workflow time is the total for one call or the complete dependent chain. The separate ns/op column divides the chain median by its operation count, not an independently measured call. Setup rows measure preparation only, not execution.

Inputs and backend construction, independent numerical checks and AD backward checks are outside timing. Returned outputs/plans and every dependent-chain intermediate are retained until after each batch interval; retention containers are prepared before the clock. value downloads/checks are outside. Eager AD rows measure forward execution/recording, not backward.

| Case ID | Operation | API route | Route (path / output / repr / session) | Dtype | Shape | Layout | Workflow | Threads | Provider | Operations/batch | Median batch ms | Median workflow ns | IQR ns | CoV | ns/op (chain only) | Status |
|---|---|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | faer | 4096 | 2.245666 | 548.26 | 47.26 | 8.0% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 1 | faer | 5120 | 2.792500 | 5454.10 | 801.92 | 10.3% | 545.41 | NOISY |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | faer | 4096 | 2.334250 | 569.89 | 30.47 | 4.1% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 1 | faer | 5120 | 2.794416 | 5457.84 | 1365.19 | 14.3% | 545.78 | NOISY |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | faer | 1024 | 0.638917 | 623.94 | 129.62 | 14.9% | — | NOISY |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 1 | faer | 2560 | 1.591333 | 6216.14 | 649.66 | 8.4% | 621.61 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 2.041209 | 996.68 | 75.99 | 15.6% | — | NOISY |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | faer | 2048 | 2.017500 | 985.11 | 46.38 | 7.5% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 1.496209 | 1461.14 | 74.89 | 4.3% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 2.008666 | 980.79 | 39.04 | 4.0% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 1 | faer | 1024 | 1.042625 | 1018.19 | 67.61 | 6.4% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 2.113167 | 2063.64 | 191.65 | 5.0% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 0.895833 | 874.84 | 54.04 | 6.1% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | faer | 1024 | 1.096792 | 1071.09 | 18.11 | 2.3% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 2.486417 | 2428.14 | 129.88 | 6.0% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | faer | 1024 | 0.993625 | 970.34 | 23.86 | 2.7% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 1 | faer | 1024 | 3.285042 | 3208.05 | 332.80 | 8.6% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 1 | faer | 1024 | 1.142959 | 1116.17 | 52.53 | 2.5% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 1 | faer | 1024 | 11.354042 | 11087.93 | 2533.00 | 20.3% | — | NOISY |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | faer | 2048 | 1.587333 | 775.06 | 73.06 | 26.3% | — | NOISY |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | faer | 2048 | 1.527625 | 745.91 | 87.83 | 7.8% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | faer | 1024 | 1.046708 | 1022.18 | 63.58 | 4.6% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | faer | 4096 | 3.509709 | 856.86 | 511.69 | 40.5% | — | NOISY |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | faer | 2048 | 2.449417 | 1196.00 | 1017.23 | 37.2% | — | NOISY |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | faer | 2048 | 1.574208 | 768.66 | 78.48 | 10.7% | — | NOISY |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | faer | 1024 | 2.279625 | 2226.20 | 38.15 | 1.0% | — | ok |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | faer | 1024 | 2.276084 | 2222.74 | 137.13 | 4.9% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | faer | 1024 | 2.234583 | 2182.21 | 90.78 | 3.7% | — | ok |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | faer | 2048 | 2.111583 | 1031.05 | 48.46 | 2.9% | — | ok |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | faer | 2048 | 2.208292 | 1078.27 | 37.25 | 3.5% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | faer | 4096 | 4.267458 | 1041.86 | 102.80 | 7.9% | — | ok |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | faer | 1024 | 2.440833 | 2383.63 | 270.59 | 15.3% | — | NOISY |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | faer | 1024 | 2.713375 | 2649.78 | 280.90 | 6.3% | — | ok |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | faer | 1024 | 2.357625 | 2302.37 | 209.76 | 7.8% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 1.482083 | 1447.35 | 294.84 | 23.8% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 2.213209 | 2161.34 | 297.69 | 9.2% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 2.691167 | 1314.05 | 119.72 | 6.3% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 2.269375 | 2216.19 | 156.47 | 5.3% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 128 | 2.334625 | 18239.26 | 1472.17 | 7.7% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 3.495708 | 6827.55 | 274.54 | 4.5% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 128 | 2.277500 | 17792.97 | 1370.28 | 10.6% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 3.521000 | 6876.95 | 359.01 | 18.9% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 2.280833 | 1113.69 | 25.26 | 2.2% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 1.629000 | 3181.64 | 108.28 | 4.9% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 2.230917 | 1089.31 | 53.49 | 4.3% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 256 | 0.803375 | 3138.18 | 207.36 | 9.3% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 1.736666 | 1695.96 | 52.61 | 2.8% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 1.859125 | 3631.10 | 235.80 | 5.0% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 1.695333 | 1655.60 | 165.77 | 6.4% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 1.955125 | 3818.60 | 189.94 | 5.9% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 2.027542 | 990.01 | 29.08 | 3.3% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 1.902750 | 3716.31 | 276.04 | 20.8% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 2.053292 | 1002.58 | 13.85 | 2.1% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 3.283458 | 3206.50 | 156.82 | 4.5% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 3.133667 | 3060.22 | 349.16 | 23.5% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 2.019292 | 1971.96 | 507.14 | 33.0% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 512 | 1.547250 | 3021.97 | 217.45 | 6.6% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 2.016291 | 1969.03 | 110.25 | 5.0% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 1.887084 | 7371.42 | 689.94 | 17.2% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 512 | 2.475084 | 4834.15 | 476.40 | 9.7% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 1.839166 | 7184.24 | 333.50 | 3.6% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 512 | 2.592708 | 5063.88 | 195.80 | 10.3% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 1.625292 | 6348.80 | 202.56 | 3.9% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 512 | 1.349333 | 2635.42 | 123.01 | 3.3% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 1.702542 | 6650.55 | 296.96 | 7.7% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 0.762666 | 2979.16 | 675.05 | 12.8% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 1.543958 | 1507.77 | 130.86 | 12.6% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 2.257625 | 1102.36 | 47.01 | 3.0% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 1.120084 | 1093.83 | 1333.56 | 44.3% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 2.091875 | 2042.85 | 192.36 | 7.2% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 512 | 2.443167 | 4771.81 | 507.20 | 6.6% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 512 | 1.404125 | 2742.43 | 202.43 | 5.9% | — | ok |
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | faer | 2048 | 2.204708 | 1076.52 | 64.64 | 4.3% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 4 | faer | 2560 | 2.681791 | 10475.75 | 203.61 | 3.5% | 1047.58 | ok |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | faer | 2048 | 2.201458 | 1074.93 | 17.52 | 4.5% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 4 | faer | 2560 | 2.834333 | 11071.61 | 1139.32 | 25.8% | 1107.16 | NOISY |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | faer | 1024 | 1.159625 | 1132.45 | 58.59 | 4.0% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 4 | faer | 640 | 0.815625 | 12744.14 | 3367.84 | 20.1% | 1274.41 | NOISY |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 2048 | 3.207792 | 1566.30 | 202.01 | 30.5% | — | NOISY |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | faer | 1024 | 1.518917 | 1483.32 | 47.95 | 4.0% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 1024 | 1.949208 | 1903.52 | 68.75 | 22.0% | — | NOISY |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 4 | faer | 2048 | 3.163542 | 1544.70 | 130.20 | 29.8% | — | NOISY |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 4 | faer | 2048 | 3.158417 | 1542.20 | 70.47 | 4.9% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 4 | faer | 1024 | 2.768917 | 2704.02 | 224.57 | 38.7% | — | NOISY |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 2048 | 3.074709 | 1501.32 | 54.11 | 3.6% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | faer | 1024 | 1.653625 | 1614.87 | 56.60 | 2.3% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 1024 | 3.365000 | 3286.13 | 2178.20 | 40.3% | — | NOISY |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | faer | 2048 | 3.049542 | 1489.03 | 125.84 | 5.9% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 4 | faer | 1024 | 4.016667 | 3922.53 | 956.75 | 33.0% | — | NOISY |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 4 | faer | 1024 | 1.573541 | 1536.66 | 59.08 | 3.2% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 4 | faer | 1024 | 10.788042 | 10535.20 | 870.42 | 8.5% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | faer | 2048 | 2.704625 | 1320.62 | 475.63 | 63.3% | — | NOISY |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | faer | 2048 | 2.639583 | 1288.86 | 66.26 | 3.7% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | faer | 1024 | 1.645083 | 1606.53 | 153.58 | 7.2% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | faer | 2048 | 2.751792 | 1343.65 | 117.20 | 12.0% | — | NOISY |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | faer | 2048 | 2.658125 | 1297.91 | 84.29 | 5.9% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | faer | 2048 | 2.680083 | 1308.63 | 29.77 | 2.5% | — | ok |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | faer | 1024 | 2.890125 | 2822.39 | 214.60 | 35.1% | — | NOISY |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | faer | 1024 | 3.125000 | 3051.76 | 703.53 | 30.4% | — | NOISY |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | faer | 1024 | 2.951291 | 2882.12 | 346.99 | 28.8% | — | NOISY |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | faer | 1024 | 1.653250 | 1614.50 | 35.60 | 2.6% | — | ok |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | faer | 1024 | 1.842583 | 1799.40 | 61.91 | 2.1% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | faer | 1024 | 1.560042 | 1523.48 | 50.13 | 2.7% | — | ok |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | faer | 1024 | 3.283208 | 3206.26 | 226.62 | 4.0% | — | ok |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | faer | 1024 | 3.682042 | 3595.74 | 572.92 | 9.9% | — | ok |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | faer | 1024 | 3.238875 | 3162.96 | 1469.24 | 38.4% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 2.927917 | 2859.29 | 913.31 | 18.3% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 3.830750 | 3740.97 | 892.42 | 13.8% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 2.130667 | 2080.73 | 226.97 | 16.0% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 3.145625 | 3071.90 | 128.36 | 3.4% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 128 | 2.667417 | 20839.20 | 2816.73 | 11.6% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 256 | 2.352625 | 9189.94 | 2888.76 | 20.8% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 128 | 2.661750 | 20794.92 | 2972.98 | 11.1% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 256 | 2.398875 | 9370.61 | 1019.12 | 18.5% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 1.909166 | 1864.42 | 137.43 | 15.3% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 256 | 1.070125 | 4180.18 | 426.92 | 28.3% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 1.262833 | 2466.47 | 1484.99 | 36.1% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 2.282583 | 4458.17 | 1455.16 | 19.4% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 2.797666 | 2732.10 | 1130.07 | 30.1% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 2.780000 | 5429.69 | 1531.98 | 43.6% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 2.489417 | 2431.07 | 186.30 | 17.9% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 128 | 0.654333 | 5111.98 | 410.97 | 4.8% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 1.655708 | 1616.90 | 123.60 | 18.4% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 2.339667 | 4569.66 | 341.80 | 16.7% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 1.810958 | 1768.51 | 138.08 | 29.9% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 2.149875 | 4198.97 | 613.40 | 37.9% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 1024 | 3.438375 | 3357.79 | 244.75 | 4.8% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 1024 | 2.253791 | 2200.97 | 47.02 | 2.2% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 512 | 1.844958 | 3603.43 | 498.74 | 9.2% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 1024 | 2.342209 | 2287.31 | 103.11 | 3.6% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 256 | 1.960875 | 7659.67 | 680.83 | 6.1% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 512 | 2.606208 | 5090.25 | 452.19 | 7.1% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 256 | 1.858334 | 7259.12 | 532.39 | 5.8% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 512 | 2.553625 | 4987.55 | 182.09 | 3.1% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 256 | 1.799708 | 7030.11 | 929.04 | 7.3% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 512 | 1.530667 | 2989.58 | 53.59 | 1.3% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 256 | 1.658291 | 6477.70 | 44.43 | 1.9% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 512 | 1.530292 | 2988.85 | 57.98 | 13.0% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 1.842416 | 1799.23 | 61.81 | 3.9% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 1.605667 | 1568.03 | 30.42 | 1.9% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 2048 | 3.228834 | 1576.58 | 139.25 | 13.4% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 1024 | 2.395834 | 2339.68 | 162.72 | 4.4% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 16 | 0.123250 | 7703.12 | 1539.06 | 20.0% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 512 | 1.570000 | 3066.41 | 91.27 | 3.5% | — | ok |

## Matched non-AD eager comparisons (#98)

All requested QR/SVD outputs are retained. Ordinary einsum uses the same parsed integer notation and default Auto strategy; planning/cache lookup stays inside both public calls. These are public-call diagnostics, not prepared execution or an isolated AD cost. Fresh pairs include session entry; shared pairs exclude it. Views are constructed untimed and consumed as views on both routes. No output to_tensor conversion is timed. Capture is inactive by construction in each fresh process, with no capture_trace guard; inputs and validation outputs are checked untracked.

| Operation | Shape | Layout | Session | Threads | Direct ns | Eager ns | Eager − direct ns | Eager/direct | Quality |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1447.35 | 2161.34 | 713.99 | 1.493 | inconclusive |
| einsum | 2×2 | col_major_contiguous | shared | 1 | 1314.05 | 2216.19 | 902.14 | 1.687 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 4 | 2859.29 | 3740.97 | 881.68 | 1.308 | inconclusive |
| einsum | 2×2 | col_major_contiguous | shared | 4 | 2080.73 | 3071.90 | 991.17 | 1.476 | inconclusive |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 1 | 1969.03 | 3021.97 | 1052.94 | 1.535 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | shared | 1 | 1971.96 | 3060.22 | 1088.26 | 1.552 | inconclusive |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 4 | 2287.31 | 3603.43 | 1316.12 | 1.575 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | shared | 4 | 2200.97 | 3357.79 | 1156.82 | 1.526 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 1 | 18239.26 | 6827.55 | -11411.71 | 0.374 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | shared | 1 | 17792.97 | 6876.95 | -10916.02 | 0.386 | inconclusive |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 4 | 20839.20 | 9189.94 | -11649.26 | 0.441 | inconclusive |
| einsum3 | 2×2 | col_major_contiguous | shared | 4 | 20794.92 | 9370.61 | -11424.31 | 0.451 | inconclusive |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1113.69 | 3181.64 | 2067.95 | 2.857 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | shared | 1 | 1089.31 | 3138.18 | 2048.87 | 2.881 | inconclusive |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 4 | 1864.42 | 4180.18 | 2315.76 | 2.242 | inconclusive |
| qr | 2×2 | col_major_contiguous | shared | 4 | 2466.47 | 4458.17 | 1991.70 | 1.808 | inconclusive |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 1 | 5063.88 | 7184.24 | 2120.36 | 1.419 | inconclusive |
| qr | 16×16 | col_major_contiguous | shared | 1 | 4834.15 | 7371.42 | 2537.27 | 1.525 | inconclusive |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 4 | 4987.55 | 7259.12 | 2271.57 | 1.455 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | shared | 4 | 5090.25 | 7659.67 | 2569.42 | 1.505 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 1 | 990.01 | 3716.31 | 2726.30 | 3.754 | inconclusive |
| solve | 2×2 | col_major_contiguous | shared | 1 | 1002.58 | 3206.50 | 2203.92 | 3.198 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 4 | 1616.90 | 4569.66 | 2952.76 | 2.826 | inconclusive |
| solve | 2×2 | col_major_contiguous | shared | 4 | 1768.51 | 4198.97 | 2430.46 | 2.374 | inconclusive |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 1 | 2979.16 | 6650.55 | 3671.39 | 2.232 | inconclusive |
| solve | 16×16 | col_major_contiguous | shared | 1 | 2635.42 | 6348.80 | 3713.38 | 2.409 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 4 | 2988.85 | 6477.70 | 3488.85 | 2.167 | inconclusive |
| solve | 16×16 | col_major_contiguous | shared | 4 | 2989.58 | 7030.11 | 4040.53 | 2.352 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1695.96 | 3631.10 | 1935.14 | 2.141 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | shared | 1 | 1655.60 | 3818.60 | 2163.00 | 2.306 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 4 | 2732.10 | 5429.69 | 2697.59 | 1.987 | inconclusive |
| svd | 2×2 | col_major_contiguous | shared | 4 | 2431.07 | 5111.98 | 2680.91 | 2.103 | inconclusive |

## Separate allocation diagnostics

Instrumented executable is built separately after latency collection. Calling-thread Rust allocator requests only; excludes foreign BLAS and worker-thread allocations. Realloc counts its requested new size, not net retained bytes. Output destruction and instrumentation setup are excluded. Helper references and multi-output retention use stack storage. No allocation-instrumented run contributes latency samples.

| Corresponding case ID | Threads | Median allocation requests/op | Median requested bytes/op | Status |
|---|---:|---:|---:|---|
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | 1 | 10 | 1528 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | 1 | 17 | 4152 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | 1 | 10 | 1528 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-shared | 1 | 17 | 4152 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | 1 | 268 | 43933 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | 1 | 51 | 11396 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | 1 | 268 | 43933 | passed |
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
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | 4 | 268 | 43933 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | 4 | 51 | 11396 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | 4 | 268 | 43933 | passed |
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
  "collection_total_seconds": 0.6908603339998081,
  "case_process_seconds": 0.6204364620007254
}
```

Wall observations `wall_allocations_t4.json`:
```json
{
  "collection_total_seconds": 0.4681931249997433,
  "case_process_seconds": 0.39901096199719177
}
```

Wall observations `wall_command.json`:
```json
{
  "command_seconds_through_collection": 474.969712208,
  "scope": "command startup through collection; excludes report rendering and publication",
  "budget_seconds": 120.0,
  "budget_status": "over_budget; diagnostic only"
}
```

Wall observations `wall_measure_t1.json`:
```json
{
  "collection_total_seconds": 3.6486431659996015,
  "case_process_seconds": 3.592700416998923
}
```

Wall observations `wall_measure_t4.json`:
```json
{
  "collection_total_seconds": 3.89825295799983,
  "case_process_seconds": 3.844512922000831
}
```

Wall observations `wall_phases.json`:
```json
{
  "validation_and_latency_build_seconds": 8.2878145,
  "metadata_and_latency_collection_seconds": 18.453719792,
  "allocation_diagnostic_build_seconds": 446.435640791,
  "allocation_collection_seconds": 1.611007375,
  "command_before_report_seconds": 474.788182458
}
```

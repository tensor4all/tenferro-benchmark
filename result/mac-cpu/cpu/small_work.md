# CPU Small-Work Benchmark Results

- Suite: `cpu/small_work`
- Raw run: `data/results/mac-cpu/cpu/small_work/20261125_184020`

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

Full metadata: `data/results/mac-cpu/cpu/small_work/20261125_184020/run_t1.yaml`

```yaml
timestamp: '2026-11-25T18:40:20Z'
tenferro_rs:
  path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - native
  resolved_path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
environment:
  container: none
  hostname: sjc22-bt144-00b82c69-006f-4acd-b9f0-e769505d592b-46EAA3084D5A.local
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
    BENCHMARK_COMMIT: 74775e4a2ad3480fb9ee9f859e0037b8cac731f2
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

Full metadata: `data/results/mac-cpu/cpu/small_work/20261125_184020/run_t4.yaml`

```yaml
timestamp: '2026-11-25T18:40:31Z'
tenferro_rs:
  path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - native
  resolved_path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
environment:
  container: none
  hostname: sjc22-bt144-00b82c69-006f-4acd-b9f0-e769505d592b-46EAA3084D5A.local
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
    BENCHMARK_COMMIT: 74775e4a2ad3480fb9ee9f859e0037b8cac731f2
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
- Expected 144, selected 144, executed 144, unsupported 0, failed 0, skipped 0, missing 0, noisy 88.
- Complete: **yes** — only executed cases count as covered; unsupported, skipped, failed, missing and unselected cases never do.

| expected | selected | executed | unsupported | failed | skipped | missing | noisy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 144 | 144 | 144 | 0 | 0 | 0 | 0 | 88 |

- Noisy: `add_f64_concrete-shared_size4_dependent10@t1`, `add_f64_concrete_shared@t1`, `add_f64_concrete-shared_size256_single@t1`, `add_f64_concrete-shared_size256_dependent10@t1`, `einsum_c64_prepared-repeat_n16_single@t1`, `solve_f64_concrete-shared_n4_single@t1`, `solve_f64_concrete-shared_n16_single@t1`, `einsum_f64_prepared-repeat_n32_single@t1`, `einsum_c64_prepared-repeat_n8_single@t1`, `einsum_c64_prepared-repeat_n32_single@t1`, `gather_f64_concrete-shared_size4_single@t1`, `gather_f64_concrete_shared@t1`, `gather_f64_concrete-shared_size256_single@t1`, `reduce_sum_f64_concrete-shared_size4_single@t1`, `einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single@t1`, `einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single@t1`, `einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single@t1`, `einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single@t1`, `nonad_einsum_n2_col_major_contiguous_concrete-fresh@t1`, `nonad_einsum_n2_col_major_contiguous_concrete-shared@t1`, `nonad_einsum3_n2_col_major_contiguous_concrete-fresh@t1`, `nonad_einsum3_n2_col_major_contiguous_eager-no-ad@t1`, `nonad_einsum3_n2_col_major_contiguous_concrete-shared@t1`, `nonad_einsum3_n2_col_major_contiguous_eager-shared@t1`, `nonad_qr_n2_col_major_contiguous_concrete-fresh@t1`, `nonad_qr_n2_col_major_contiguous_eager-no-ad@t1`, `nonad_svd_n2_col_major_contiguous_concrete-fresh@t1`, `nonad_svd_n2_col_major_contiguous_concrete-shared@t1`, `nonad_svd_n2_col_major_contiguous_eager-shared@t1`, `nonad_solve_n2_col_major_contiguous_concrete-fresh@t1`, `nonad_solve_n2_col_major_contiguous_concrete-shared@t1`, `nonad_einsum_n16_col_major_contiguous_eager-shared@t1`, `nonad_einsum_n16_col_major_contiguous_eager-no-ad@t1`, `nonad_einsum_n16_col_major_contiguous_concrete-fresh@t1`, `nonad_qr_n16_col_major_contiguous_concrete-shared@t1`, `nonad_qr_n16_col_major_contiguous_concrete-fresh@t1`, `nonad_solve_n16_col_major_contiguous_eager-shared@t1`, `nonad_solve_n16_col_major_contiguous_concrete-shared@t1`, `nonad_solve_n16_col_major_contiguous_eager-no-ad@t1`, `nonad_einsum_n2_col_major_contiguous_concrete-scoped@t1` (+48 more)

Numerical checks: **144/144 recorded rows passed** (case × thread count).

## Timing

Sequential release runs; explicit initialization and warmups, then calibration toward the declared batch duration (2 ms by default), then the measured batches of the recorded effort (standard: 3 warmups and 15 batches; scan: a low-repetition screen that only marks suspects). Legacy single-call routes start calibration at 1024 workflows; matched non-AD and dependent-chain routes start at 16 and bound retained memory. Each batch uses one wall-clock interval; all outputs are retained until it stops. Statistics divide batch time by iterations. Median and inclusive IQR are in nanoseconds (ns); CoV is sample standard deviation / mean. NOISY means CoV > 10%, a descriptive label only. No noisy rows are excluded.

Workflow time is the total for one call or the complete dependent chain. The separate ns/op column divides the chain median by its operation count, not an independently measured call. Setup rows measure preparation only, not execution.

Inputs and backend construction, independent numerical checks and AD backward checks are outside timing. Returned outputs/plans and every dependent-chain intermediate are retained until after each batch interval; retention containers are prepared before the clock. value downloads/checks are outside. Eager AD rows measure forward execution/recording, not backward.

| Case ID | Operation | API route | Route (path / output / repr / session) | Dtype | Shape | Layout | Workflow | Threads | Provider | Operations/batch | Median batch ms | Median workflow ns | IQR ns | CoV | ns/op (chain only) | Status |
|---|---|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | faer | 2048 | 1.328625 | 648.74 | 39.05 | 5.2% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 1 | faer | 5120 | 2.986833 | 5833.66 | 413.41 | 12.9% | 583.37 | NOISY |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | faer | 4096 | 2.904833 | 709.19 | 179.40 | 29.3% | — | NOISY |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 1 | faer | 5120 | 3.231750 | 6312.01 | 345.38 | 3.5% | 631.20 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | faer | 2048 | 1.605542 | 783.96 | 87.78 | 32.3% | — | NOISY |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 1 | faer | 2560 | 1.800000 | 7031.25 | 1308.43 | 13.7% | 703.12 | NOISY |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 2.179500 | 1064.21 | 60.72 | 5.5% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | faer | 2048 | 2.302792 | 1124.41 | 87.67 | 5.7% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 1.702417 | 1662.52 | 158.47 | 8.7% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 2.231209 | 1089.46 | 78.79 | 5.8% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 1 | faer | 2048 | 2.231042 | 1089.38 | 75.37 | 4.3% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 2.530208 | 2470.91 | 283.37 | 19.7% | — | NOISY |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 2.221709 | 1084.82 | 133.47 | 9.6% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | faer | 1024 | 1.169000 | 1141.60 | 264.83 | 31.8% | — | NOISY |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 4.194500 | 4096.19 | 2753.05 | 37.4% | — | NOISY |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | faer | 1024 | 1.047375 | 1022.83 | 96.46 | 5.8% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 1 | faer | 1024 | 3.417000 | 3336.91 | 503.74 | 51.3% | — | NOISY |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 1 | faer | 1024 | 1.123291 | 1096.96 | 145.26 | 29.2% | — | NOISY |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 1 | faer | 1024 | 11.616834 | 11344.56 | 2587.00 | 22.1% | — | NOISY |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | faer | 2048 | 1.577167 | 770.10 | 107.22 | 25.8% | — | NOISY |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | faer | 2048 | 1.631875 | 796.81 | 142.14 | 28.7% | — | NOISY |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | faer | 1024 | 1.148625 | 1121.70 | 552.86 | 42.1% | — | NOISY |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | faer | 2048 | 1.855208 | 905.86 | 49.80 | 13.6% | — | NOISY |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | faer | 2048 | 1.853083 | 904.83 | 37.19 | 4.7% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | faer | 2048 | 1.919625 | 937.32 | 77.12 | 8.3% | — | ok |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | faer | 1024 | 2.686500 | 2623.54 | 272.07 | 31.8% | — | NOISY |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | faer | 1024 | 2.744458 | 2680.13 | 163.98 | 4.7% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | faer | 1024 | 2.521083 | 2462.00 | 85.08 | 2.4% | — | ok |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | faer | 2048 | 2.446458 | 1194.56 | 69.00 | 3.7% | — | ok |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | faer | 2048 | 2.713750 | 1325.07 | 92.05 | 7.7% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | faer | 2048 | 2.270084 | 1108.44 | 54.90 | 4.0% | — | ok |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | faer | 1024 | 2.927042 | 2858.44 | 181.03 | 15.8% | — | NOISY |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | faer | 1024 | 3.298833 | 3221.52 | 672.10 | 21.5% | — | NOISY |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | faer | 1024 | 2.725833 | 2661.95 | 358.93 | 11.8% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 3.186167 | 1555.75 | 124.91 | 13.7% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 2.688375 | 2625.37 | 128.38 | 6.2% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 1.625125 | 1587.04 | 256.75 | 11.0% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 2.561417 | 2501.38 | 156.05 | 5.7% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 128 | 2.264208 | 17689.12 | 1743.16 | 15.4% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 256 | 2.227458 | 8701.01 | 3662.43 | 37.4% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 64 | 1.631334 | 25489.59 | 14457.69 | 41.7% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 256 | 2.367625 | 9248.54 | 7610.19 | 46.7% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 2.276709 | 1111.67 | 123.74 | 15.5% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 3.228458 | 3152.79 | 186.20 | 23.2% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 2.171958 | 1060.53 | 21.30 | 1.7% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 1.547667 | 3022.79 | 60.95 | 6.7% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 1.789625 | 1747.68 | 51.39 | 16.1% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 1.911250 | 3732.91 | 85.12 | 6.2% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 1.741917 | 1701.09 | 460.94 | 40.3% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 2.086917 | 4076.01 | 252.04 | 12.2% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 2.010208 | 981.55 | 148.57 | 25.8% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 1.722125 | 3363.53 | 215.70 | 4.0% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 2.062959 | 1007.30 | 564.04 | 42.9% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 1.789125 | 3494.38 | 113.53 | 3.0% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 0.724625 | 2830.57 | 156.25 | 30.9% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 1.861375 | 1817.75 | 116.64 | 5.8% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 2.830209 | 2763.88 | 122.88 | 37.9% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 1.996708 | 1949.91 | 282.96 | 14.1% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 1.759416 | 6872.72 | 553.30 | 6.0% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 1.313000 | 5128.91 | 420.25 | 26.2% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 1.775958 | 6937.34 | 354.09 | 6.7% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 512 | 2.699291 | 5272.05 | 350.75 | 39.6% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 1.707625 | 6670.41 | 440.43 | 58.2% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 512 | 1.455625 | 2843.02 | 234.54 | 19.2% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 1.696834 | 6628.26 | 2738.77 | 50.6% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 512 | 1.407667 | 2749.35 | 151.16 | 8.8% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 3.018042 | 1473.65 | 298.89 | 33.5% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 2.432167 | 1187.58 | 61.37 | 19.6% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 2.100000 | 1025.39 | 81.07 | 5.7% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 2.019541 | 1972.21 | 111.21 | 10.8% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 512 | 2.615208 | 5107.83 | 205.44 | 3.5% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 512 | 1.453625 | 2839.11 | 123.62 | 2.5% | — | ok |
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | faer | 4096 | 2.885083 | 704.37 | 97.60 | 11.1% | — | NOISY |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 4 | faer | 5120 | 3.449292 | 6736.90 | 325.77 | 5.2% | 673.69 | ok |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | faer | 4096 | 2.985750 | 728.94 | 232.47 | 28.8% | — | NOISY |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 4 | faer | 5120 | 3.446250 | 6730.96 | 887.57 | 7.1% | 673.10 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | faer | 1024 | 0.698416 | 682.05 | 82.13 | 21.2% | — | NOISY |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 4 | faer | 2560 | 1.701167 | 6645.18 | 276.61 | 15.0% | 664.52 | NOISY |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 2048 | 2.070166 | 1010.82 | 75.99 | 5.2% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | faer | 2048 | 2.941417 | 1436.24 | 255.30 | 14.7% | — | NOISY |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 1024 | 1.599125 | 1561.65 | 151.29 | 25.3% | — | NOISY |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 1.087375 | 1061.89 | 89.11 | 5.3% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 4 | faer | 2048 | 2.203125 | 1075.74 | 52.49 | 3.0% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 4 | faer | 1024 | 2.448416 | 2391.03 | 847.57 | 36.1% | — | NOISY |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 2048 | 2.411458 | 1177.47 | 925.55 | 43.5% | — | NOISY |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | faer | 2048 | 2.468167 | 1205.16 | 159.13 | 37.5% | — | NOISY |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 1024 | 4.277625 | 4177.37 | 2943.73 | 36.6% | — | NOISY |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | faer | 2048 | 2.347292 | 1146.14 | 321.84 | 29.5% | — | NOISY |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 4 | faer | 1024 | 3.665417 | 3579.51 | 621.40 | 28.3% | — | NOISY |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 4 | faer | 1024 | 1.293500 | 1263.18 | 204.85 | 32.1% | — | NOISY |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 4 | faer | 1024 | 12.987334 | 12682.94 | 3111.37 | 19.1% | — | NOISY |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | faer | 1024 | 1.674917 | 1635.66 | 1204.10 | 38.5% | — | NOISY |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | faer | 2048 | 1.610458 | 786.36 | 264.20 | 42.1% | — | NOISY |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | faer | 1024 | 1.159292 | 1132.12 | 525.43 | 50.0% | — | NOISY |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | faer | 1024 | 0.911875 | 890.50 | 1066.16 | 86.6% | — | NOISY |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | faer | 2048 | 2.048791 | 1000.39 | 672.04 | 47.9% | — | NOISY |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | faer | 2048 | 1.950791 | 952.53 | 259.79 | 28.0% | — | NOISY |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | faer | 1024 | 3.876500 | 3785.64 | 2331.99 | 34.2% | — | NOISY |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | faer | 1024 | 2.817208 | 2751.18 | 2324.54 | 39.9% | — | NOISY |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | faer | 1024 | 2.977000 | 2907.23 | 1009.66 | 30.3% | — | NOISY |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | faer | 1024 | 1.224375 | 1195.68 | 174.44 | 23.9% | — | NOISY |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | faer | 1024 | 1.335208 | 1303.91 | 277.77 | 38.4% | — | NOISY |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | faer | 2048 | 2.302292 | 1124.17 | 265.00 | 24.0% | — | NOISY |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | faer | 1024 | 2.687917 | 2624.92 | 137.04 | 20.8% | — | NOISY |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | faer | 1024 | 2.766125 | 2701.29 | 367.21 | 43.9% | — | NOISY |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | faer | 1024 | 2.615625 | 2554.32 | 1300.29 | 28.7% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 1.591833 | 1554.52 | 219.87 | 44.0% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 2.483250 | 2425.05 | 320.41 | 21.1% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 1.556125 | 1519.65 | 65.80 | 13.0% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 2.376083 | 2320.39 | 106.85 | 3.4% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 128 | 2.135375 | 16682.62 | 590.98 | 4.3% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 3.586458 | 7004.80 | 196.49 | 14.4% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 128 | 2.155041 | 16836.26 | 1068.36 | 6.4% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 3.607042 | 7045.00 | 272.87 | 3.3% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 0.691584 | 1350.75 | 322.06 | 20.2% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 3.262000 | 3185.55 | 131.77 | 22.1% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 2048 | 2.587250 | 1263.31 | 27.12 | 2.1% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 1.686875 | 3294.68 | 175.70 | 3.5% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 2.015291 | 1968.06 | 69.27 | 2.2% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 2.121125 | 4142.82 | 252.93 | 4.0% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 1.997834 | 1951.01 | 43.52 | 2.8% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 2.126166 | 4152.67 | 212.44 | 3.4% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 2048 | 2.243458 | 1095.44 | 30.34 | 2.0% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 1.924542 | 3758.87 | 252.44 | 50.7% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 2048 | 2.270875 | 1108.83 | 52.80 | 2.8% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 256 | 0.984834 | 3847.01 | 238.28 | 4.9% | — | INCONCLUSIVE (<1 ms batch) |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 1024 | 2.900917 | 2832.93 | 113.67 | 2.4% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 1024 | 2.203125 | 2151.49 | 176.29 | 18.6% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 1024 | 2.924917 | 2856.36 | 142.44 | 3.6% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 1024 | 1.978958 | 1932.58 | 139.47 | 8.3% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 256 | 1.946250 | 7602.54 | 684.25 | 12.1% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 256 | 1.340708 | 5237.14 | 688.31 | 19.2% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 256 | 2.039959 | 7968.59 | 7066.00 | 44.9% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 256 | 1.245458 | 4865.07 | 246.50 | 60.5% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 256 | 1.674208 | 6539.88 | 424.80 | 11.5% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 512 | 1.446375 | 2824.95 | 188.68 | 6.7% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 256 | 1.715459 | 6701.01 | 486.33 | 8.0% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 512 | 1.511625 | 2952.39 | 533.90 | 19.0% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 1.652375 | 1613.65 | 179.30 | 6.0% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 2048 | 2.958167 | 1444.42 | 46.50 | 4.7% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 2048 | 2.593208 | 1266.21 | 93.40 | 14.7% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 1024 | 2.299333 | 2245.44 | 311.22 | 10.6% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 512 | 2.902083 | 5668.13 | 188.03 | 6.5% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 256 | 0.748458 | 2923.66 | 1773.52 | 48.7% | — | INCONCLUSIVE (<1 ms batch) |

## Matched non-AD eager comparisons (#98)

All requested QR/SVD outputs are retained. Ordinary einsum uses the same parsed integer notation and default Auto strategy; planning/cache lookup stays inside both public calls. These are public-call diagnostics, not prepared execution or an isolated AD cost. Fresh pairs include session entry; shared pairs exclude it. Views are constructed untimed and consumed as views on both routes. No output to_tensor conversion is timed. Capture is inactive by construction in each fresh process, with no capture_trace guard; inputs and validation outputs are checked untracked.

| Operation | Shape | Layout | Session | Threads | Direct ns | Eager ns | Eager − direct ns | Eager/direct | Quality |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1555.75 | 2625.37 | 1069.62 | 1.688 | inconclusive |
| einsum | 2×2 | col_major_contiguous | shared | 1 | 1587.04 | 2501.38 | 914.34 | 1.576 | inconclusive |
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 4 | 1554.52 | 2425.05 | 870.53 | 1.560 | inconclusive |
| einsum | 2×2 | col_major_contiguous | shared | 4 | 1519.65 | 2320.39 | 800.74 | 1.527 | inconclusive |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 1 | 1949.91 | 2763.88 | 813.97 | 1.417 | inconclusive |
| einsum | 16×16 | col_major_contiguous | shared | 1 | 1817.75 | 2830.57 | 1012.82 | 1.557 | inconclusive |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 4 | 1932.58 | 2856.36 | 923.78 | 1.478 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | shared | 4 | 2151.49 | 2832.93 | 681.44 | 1.317 | inconclusive |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 1 | 17689.12 | 8701.01 | -8988.11 | 0.492 | inconclusive |
| einsum3 | 2×2 | col_major_contiguous | shared | 1 | 25489.59 | 9248.54 | -16241.05 | 0.363 | inconclusive |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 4 | 16682.62 | 7004.80 | -9677.82 | 0.420 | inconclusive |
| einsum3 | 2×2 | col_major_contiguous | shared | 4 | 16836.26 | 7045.00 | -9791.26 | 0.418 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1111.67 | 3152.79 | 2041.12 | 2.836 | inconclusive |
| qr | 2×2 | col_major_contiguous | shared | 1 | 1060.53 | 3022.79 | 1962.26 | 2.850 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 4 | 1350.75 | 3185.55 | 1834.80 | 2.358 | inconclusive |
| qr | 2×2 | col_major_contiguous | shared | 4 | 1263.31 | 3294.68 | 2031.37 | 2.608 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 1 | 5272.05 | 6937.34 | 1665.29 | 1.316 | inconclusive |
| qr | 16×16 | col_major_contiguous | shared | 1 | 5128.91 | 6872.72 | 1743.81 | 1.340 | inconclusive |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 4 | 4865.07 | 7968.59 | 3103.52 | 1.638 | inconclusive |
| qr | 16×16 | col_major_contiguous | shared | 4 | 5237.14 | 7602.54 | 2365.40 | 1.452 | inconclusive |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 1 | 981.55 | 3363.53 | 2381.98 | 3.427 | inconclusive |
| solve | 2×2 | col_major_contiguous | shared | 1 | 1007.30 | 3494.38 | 2487.08 | 3.469 | inconclusive |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 4 | 1095.44 | 3758.87 | 2663.43 | 3.431 | inconclusive |
| solve | 2×2 | col_major_contiguous | shared | 4 | 1108.83 | 3847.01 | 2738.18 | 3.469 | inconclusive |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 1 | 2749.35 | 6628.26 | 3878.91 | 2.411 | inconclusive |
| solve | 16×16 | col_major_contiguous | shared | 1 | 2843.02 | 6670.41 | 3827.39 | 2.346 | inconclusive |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 4 | 2952.39 | 6701.01 | 3748.62 | 2.270 | inconclusive |
| solve | 16×16 | col_major_contiguous | shared | 4 | 2824.95 | 6539.88 | 3714.93 | 2.315 | inconclusive |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1747.68 | 3732.91 | 1985.23 | 2.136 | inconclusive |
| svd | 2×2 | col_major_contiguous | shared | 1 | 1701.09 | 4076.01 | 2374.92 | 2.396 | inconclusive |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 4 | 1968.06 | 4142.82 | 2174.76 | 2.105 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | shared | 4 | 1951.01 | 4152.67 | 2201.66 | 2.128 | observed; no speedup claim |

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
  "collection_total_seconds": 0.31370637500003795,
  "case_process_seconds": 0.278002582004774
}
```

Wall observations `wall_allocations_t4.json`:
```json
{
  "collection_total_seconds": 0.3456688749993191,
  "case_process_seconds": 0.31121020799946564
}
```

Wall observations `wall_command.json`:
```json
{
  "command_seconds_through_collection": 619.563238375,
  "scope": "command startup through collection; excludes report rendering and publication",
  "budget_seconds": 120.0,
  "budget_status": "over_budget; diagnostic only"
}
```

Wall observations `wall_measure_t1.json`:
```json
{
  "collection_total_seconds": 3.8715285829994173,
  "case_process_seconds": 3.809790953000629
}
```

Wall observations `wall_measure_t4.json`:
```json
{
  "collection_total_seconds": 3.9309938749993307,
  "case_process_seconds": 3.869585997003014
}
```

Wall observations `wall_phases.json`:
```json
{
  "validation_and_latency_build_seconds": 10.11135325,
  "metadata_and_latency_collection_seconds": 18.6989105,
  "allocation_diagnostic_build_seconds": 589.725485125,
  "allocation_collection_seconds": 0.920654542,
  "command_before_report_seconds": 619.456403417
}
```

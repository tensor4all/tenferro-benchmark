# CPU Small-Work Benchmark Results

- Suite: `cpu/small_work`
- Raw run: `data/results/amd-cpu/cpu/small_work/20261125_191419`

## CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1020-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`
- Container: `devcontainer`

## run_t1

Full metadata: `data/results/amd-cpu/cpu/small_work/20261125_191419/run_t1.yaml`

```yaml
timestamp: '2026-11-25T19:14:19Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - native
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
environment:
  container: devcontainer
  hostname: 40a2b640952a
  os: Linux-7.0.0-1020-azure-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '1'
    measured_cpus: 0-3
    l3_domains:
    - 0-3
    cpu_boost: null
    scaling_governor: null
    scaling_driver: null
  env:
    OMP_NUM_THREADS: '1'
    OMP_THREAD_LIMIT: '1'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: amd-cpu
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
    USE_CUDA: '0'
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.100.0 (0ef1bc386 2026-11-09)
    BENCH_INSTANCE: null
```

## run_t4

Full metadata: `data/results/amd-cpu/cpu/small_work/20261125_191419/run_t4.yaml`

```yaml
timestamp: '2026-11-25T19:14:26Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - native
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
environment:
  container: devcontainer
  hostname: 40a2b640952a
  os: Linux-7.0.0-1020-azure-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '4'
    measured_cpus: 0-3
    l3_domains:
    - 0-3
    cpu_boost: null
    scaling_governor: null
    scaling_driver: null
  env:
    OMP_NUM_THREADS: '4'
    OMP_THREAD_LIMIT: '4'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '4'
    BENCHMARK_TARGET_PROFILE: amd-cpu
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
    USE_CUDA: '0'
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

- Noisy: `add_f64_concrete-shared_size4_single@t4`, `add_f64_concrete-shared_size4_dependent10@t4`, `add_f64_concrete_shared@t4`, `add_f64_concrete-shared_size16_dependent10@t4`, `add_f64_concrete-shared_size256_single@t4`, `add_f64_concrete-shared_size256_dependent10@t4`, `einsum_f64_prepared-repeat_n16_single@t4`, `solve_f64_concrete-shared_n2_single@t4`, `solve_f64_concrete-shared_n4_single@t4`, `gather_f64_concrete-shared_size4_single@t4`, `gather_f64_concrete_shared@t4`, `gather_f64_concrete-shared_size256_single@t4`, `reduce_sum_f64_concrete-shared_size4_single@t4`, `reduce_sum_f64_concrete-shared_size16_single@t4`, `reduce_sum_f64_concrete-shared_size256_single@t4`, `einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single@t4`, `nonad_einsum_n2_col_major_contiguous_concrete-fresh@t4`, `nonad_einsum_n2_col_major_contiguous_concrete-shared@t4`, `nonad_einsum3_n2_col_major_contiguous_concrete-fresh@t4`, `nonad_einsum3_n2_col_major_contiguous_concrete-shared@t4`, `nonad_qr_n2_col_major_contiguous_concrete-fresh@t4`, `nonad_qr_n2_col_major_contiguous_eager-no-ad@t4`, `nonad_qr_n2_col_major_contiguous_concrete-shared@t4`, `nonad_qr_n2_col_major_contiguous_eager-shared@t4`, `nonad_svd_n2_col_major_contiguous_concrete-fresh@t4`, `nonad_svd_n2_col_major_contiguous_eager-no-ad@t4`, `nonad_svd_n2_col_major_contiguous_concrete-shared@t4`, `nonad_svd_n2_col_major_contiguous_eager-shared@t4`, `nonad_solve_n2_col_major_contiguous_concrete-fresh@t4`, `nonad_solve_n2_col_major_contiguous_eager-no-ad@t4`, `nonad_solve_n2_col_major_contiguous_concrete-shared@t4`, `nonad_solve_n2_col_major_contiguous_eager-shared@t4`, `nonad_einsum_n16_col_major_contiguous_eager-shared@t4`, `nonad_einsum_n16_col_major_contiguous_eager-no-ad@t4`, `nonad_qr_n16_col_major_contiguous_eager-shared@t4`, `nonad_qr_n16_col_major_contiguous_concrete-shared@t4`, `nonad_qr_n16_col_major_contiguous_eager-no-ad@t4`, `nonad_qr_n16_col_major_contiguous_concrete-fresh@t4`, `nonad_solve_n16_col_major_contiguous_eager-shared@t4`, `nonad_solve_n16_col_major_contiguous_concrete-shared@t4` (+6 more)

Numerical checks: **144/144 recorded rows passed** (case × thread count).

## Timing

Sequential release runs; explicit initialization and warmups, then calibration toward the declared batch duration (2 ms by default), then the measured batches of the recorded effort (standard: 3 warmups and 15 batches; scan: a low-repetition screen that only marks suspects). Legacy single-call routes start calibration at 1024 workflows; matched non-AD and dependent-chain routes start at 16 and bound retained memory. Each batch uses one wall-clock interval; all outputs are retained until it stops. Statistics divide batch time by iterations. Median and inclusive IQR are in nanoseconds (ns); CoV is sample standard deviation / mean. NOISY means CoV > 10%, a descriptive label only. No noisy rows are excluded.

Workflow time is the total for one call or the complete dependent chain. The separate ns/op column divides the chain median by its operation count, not an independently measured call. Setup rows measure preparation only, not execution.

Inputs and backend construction, independent numerical checks and AD backward checks are outside timing. Returned outputs/plans and every dependent-chain intermediate are retained until after each batch interval; retention containers are prepared before the clock. value downloads/checks are outside. Eager AD rows measure forward execution/recording, not backward.

| Case ID | Operation | API route | Route (path / output / repr / session) | Dtype | Shape | Layout | Workflow | Threads | Provider | Operations/batch | Median batch ms | Median workflow ns | IQR ns | CoV | ns/op (chain only) | Status |
|---|---|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | faer | 2048 | 1.785196 | 871.68 | 4.36 | 3.9% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 1 | faer | 2560 | 2.184540 | 8533.36 | 135.58 | 2.4% | 853.34 | ok |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | faer | 2048 | 1.784194 | 871.19 | 5.24 | 4.6% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 1 | faer | 2560 | 2.116013 | 8265.68 | 110.11 | 1.0% | 826.57 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | faer | 1024 | 1.004451 | 980.91 | 36.93 | 6.6% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 1 | faer | 2560 | 2.427923 | 9484.07 | 136.99 | 2.2% | 948.41 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 2048 | 3.549123 | 1732.97 | 9.84 | 1.9% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | faer | 1024 | 1.796036 | 1753.94 | 22.43 | 2.9% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 2.171235 | 2120.35 | 31.41 | 3.0% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 1.856800 | 1813.28 | 15.36 | 2.2% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 1 | faer | 1024 | 1.858773 | 1815.21 | 24.87 | 3.0% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 2.785610 | 2720.32 | 72.99 | 3.9% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 1.638803 | 1600.39 | 12.42 | 2.6% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | faer | 1024 | 1.757024 | 1715.84 | 12.02 | 3.0% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 4.310451 | 4209.42 | 33.81 | 2.4% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | faer | 1024 | 1.884721 | 1840.55 | 14.79 | 2.3% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 1 | faer | 1024 | 3.722525 | 3635.28 | 44.16 | 3.1% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 1 | faer | 1024 | 1.957327 | 1911.45 | 27.34 | 2.3% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 1 | faer | 1024 | 8.500377 | 8301.15 | 38.31 | 1.4% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | faer | 2048 | 2.300265 | 1123.18 | 10.80 | 3.5% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | faer | 2048 | 2.359445 | 1152.07 | 8.79 | 3.3% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | faer | 1024 | 1.433711 | 1400.11 | 19.69 | 3.0% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | faer | 2048 | 2.297020 | 1121.59 | 11.53 | 3.7% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | faer | 2048 | 2.320624 | 1133.12 | 9.06 | 3.5% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | faer | 2048 | 2.412104 | 1177.79 | 8.57 | 3.3% | — | ok |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | faer | 1024 | 3.263970 | 3187.47 | 40.69 | 2.1% | — | ok |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | faer | 1024 | 3.314325 | 3236.65 | 28.62 | 1.5% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | faer | 1024 | 2.971316 | 2901.68 | 20.30 | 2.0% | — | ok |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | faer | 1024 | 1.917683 | 1872.74 | 28.48 | 2.6% | — | ok |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | faer | 1024 | 1.880714 | 1836.63 | 19.51 | 2.3% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | faer | 2048 | 3.001763 | 1465.70 | 10.92 | 2.6% | — | ok |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | faer | 1024 | 3.005209 | 2934.77 | 24.83 | 1.6% | — | ok |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | faer | 1024 | 3.116727 | 3043.68 | 17.76 | 1.5% | — | ok |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | faer | 1024 | 3.042970 | 2971.65 | 39.24 | 1.7% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 2.176765 | 2125.75 | 33.72 | 5.0% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 3.583406 | 3499.42 | 15.28 | 1.7% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 2.187986 | 2136.71 | 21.97 | 4.9% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 1.781369 | 3479.24 | 24.07 | 0.8% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 128 | 3.060543 | 23910.49 | 263.69 | 1.0% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 256 | 2.478618 | 9682.10 | 75.49 | 0.9% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 128 | 3.059119 | 23899.37 | 174.74 | 0.6% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 256 | 2.510076 | 9804.98 | 120.16 | 2.7% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 1.673227 | 1634.01 | 36.49 | 6.1% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 2.231148 | 4357.71 | 58.80 | 1.4% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 1.676253 | 1636.97 | 25.85 | 6.9% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 2.154143 | 4207.31 | 91.07 | 6.9% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 2.826396 | 2760.15 | 24.09 | 4.7% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 2.994499 | 5848.63 | 54.12 | 1.1% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 2.850070 | 2783.27 | 55.55 | 6.1% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 2.874415 | 5614.09 | 49.58 | 0.9% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 1.641108 | 1602.64 | 29.57 | 6.7% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 2.700512 | 5274.44 | 98.51 | 1.5% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 1.610401 | 1572.66 | 23.06 | 6.8% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 2.614351 | 5106.15 | 51.10 | 0.9% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 512 | 1.923092 | 3756.04 | 43.81 | 1.2% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 2.552265 | 2492.45 | 18.68 | 4.8% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 512 | 1.955643 | 3819.62 | 16.20 | 0.6% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 2.512671 | 2453.78 | 18.32 | 5.1% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 3.047929 | 11905.97 | 76.76 | 1.0% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 2.371018 | 9261.79 | 67.51 | 1.5% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 2.997865 | 11710.41 | 91.05 | 0.9% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 2.323199 | 9075.00 | 136.15 | 1.6% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 2.334449 | 9118.94 | 58.00 | 0.8% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 512 | 2.142582 | 4184.73 | 26.92 | 3.5% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 2.374815 | 9276.62 | 150.96 | 1.3% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 512 | 2.156798 | 4212.50 | 37.30 | 3.8% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 1.074792 | 2099.20 | 30.20 | 8.4% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 1.677886 | 1638.56 | 13.84 | 6.3% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 1.614017 | 1576.19 | 10.95 | 6.6% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 2.490179 | 2431.82 | 7.90 | 4.4% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 2.313601 | 9037.50 | 160.38 | 2.0% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 512 | 2.181303 | 4260.36 | 74.27 | 3.8% | — | ok |
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | faer | 1024 | 1.477782 | 1443.15 | 128.78 | 17.4% | — | NOISY |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 4 | faer | 1280 | 1.763393 | 13776.51 | 156.50 | 15.7% | 1377.65 | NOISY |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | faer | 1024 | 1.471540 | 1437.05 | 135.59 | 17.6% | — | NOISY |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 4 | faer | 1280 | 1.860034 | 14531.52 | 257.86 | 10.7% | 1453.15 | NOISY |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | faer | 1024 | 1.586985 | 1549.79 | 13.94 | 17.0% | — | NOISY |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 4 | faer | 1280 | 1.928701 | 15067.98 | 359.81 | 10.0% | 1506.80 | NOISY |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 2.417361 | 2360.70 | 23.84 | 9.6% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | faer | 1024 | 2.398737 | 2342.52 | 22.92 | 9.5% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 1024 | 2.783974 | 2718.72 | 67.58 | 13.6% | — | NOISY |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 2.462024 | 2404.32 | 17.01 | 8.8% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 4 | faer | 1024 | 2.429404 | 2372.46 | 30.83 | 8.6% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 4 | faer | 1024 | 3.386106 | 3306.74 | 55.91 | 2.9% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 2.274246 | 2220.94 | 10.32 | 16.4% | — | NOISY |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | faer | 1024 | 2.383859 | 2327.99 | 25.31 | 15.9% | — | NOISY |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 1024 | 5.135172 | 5014.82 | 68.44 | 2.2% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | faer | 1024 | 2.439072 | 2381.91 | 48.53 | 9.5% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 4 | faer | 1024 | 4.355560 | 4253.48 | 37.30 | 2.8% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 4 | faer | 1024 | 2.570055 | 2509.82 | 7.37 | 2.6% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 4 | faer | 1024 | 17.860828 | 17442.21 | 202.56 | 1.0% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | faer | 1024 | 1.760271 | 1719.01 | 39.58 | 18.2% | — | NOISY |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | faer | 1024 | 1.840480 | 1797.34 | 24.59 | 16.4% | — | NOISY |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | faer | 1024 | 2.118559 | 2068.91 | 68.90 | 14.3% | — | NOISY |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | faer | 1024 | 2.137954 | 2087.85 | 736.23 | 18.1% | — | NOISY |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | faer | 1024 | 1.856681 | 1813.17 | 36.49 | 14.2% | — | NOISY |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | faer | 1024 | 1.814943 | 1772.41 | 25.89 | 16.3% | — | NOISY |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | faer | 1024 | 3.927100 | 3835.06 | 56.94 | 4.1% | — | ok |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | faer | 1024 | 4.029982 | 3935.53 | 34.22 | 5.7% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | faer | 1024 | 3.656415 | 3570.72 | 76.52 | 7.7% | — | ok |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | faer | 1024 | 2.602040 | 2541.05 | 16.54 | 7.3% | — | ok |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | faer | 1024 | 2.489029 | 2430.69 | 22.18 | 8.9% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | faer | 1024 | 2.111656 | 2062.16 | 22.47 | 12.1% | — | NOISY |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | faer | 1024 | 3.608866 | 3524.28 | 29.60 | 1.7% | — | ok |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | faer | 1024 | 3.796356 | 3707.38 | 37.74 | 1.5% | — | ok |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | faer | 1024 | 3.567409 | 3483.80 | 19.78 | 1.4% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 1.431388 | 2795.68 | 26.09 | 11.2% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 2.160667 | 4220.05 | 41.35 | 1.4% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 1.446616 | 2825.42 | 43.08 | 11.2% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 2.110023 | 4121.14 | 30.34 | 2.6% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 64 | 1.636190 | 25565.47 | 615.99 | 17.1% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 128 | 1.550821 | 12115.79 | 321.14 | 5.8% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 64 | 1.605021 | 25078.45 | 611.29 | 17.4% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 128 | 1.515875 | 11842.77 | 171.96 | 7.3% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 1.571218 | 3068.79 | 1253.03 | 21.2% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 256 | 1.372808 | 5362.53 | 95.26 | 17.6% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 1.225794 | 2394.13 | 34.00 | 21.5% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 256 | 1.304521 | 5095.79 | 103.16 | 18.0% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 1.829049 | 3572.36 | 56.44 | 16.8% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 256 | 1.738691 | 6791.76 | 160.42 | 13.4% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 1.842925 | 3599.46 | 22.51 | 16.5% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 256 | 1.731458 | 6763.51 | 191.47 | 14.0% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 1.165252 | 2275.88 | 65.54 | 22.2% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 256 | 1.535502 | 5998.05 | 128.74 | 16.8% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 1.120669 | 2188.81 | 135.36 | 22.9% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 256 | 1.524792 | 5956.22 | 138.58 | 16.8% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 256 | 1.148561 | 4486.57 | 74.12 | 13.5% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 512 | 1.629798 | 3183.20 | 82.97 | 9.1% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 256 | 1.168067 | 4562.76 | 49.00 | 13.2% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 512 | 1.647401 | 3217.58 | 69.94 | 5.7% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 128 | 1.654063 | 12922.37 | 238.77 | 14.1% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 128 | 1.260149 | 9844.91 | 164.88 | 17.8% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 128 | 1.635018 | 12773.58 | 252.94 | 14.4% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 128 | 1.285676 | 10044.34 | 189.50 | 16.7% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 128 | 1.296316 | 10127.47 | 218.81 | 18.4% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 256 | 1.286989 | 5027.30 | 120.54 | 19.4% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 128 | 1.304021 | 10187.66 | 410.88 | 18.7% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 256 | 1.241414 | 4849.27 | 106.64 | 20.6% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 1.468426 | 2868.02 | 46.48 | 9.7% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 1.279224 | 2498.48 | 62.43 | 20.2% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 1.131518 | 2210.00 | 68.44 | 23.5% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 512 | 1.616002 | 3156.25 | 21.81 | 6.0% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 128 | 1.269015 | 9914.18 | 157.76 | 17.8% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 256 | 1.289313 | 5036.38 | 269.00 | 19.0% | — | NOISY |

## Matched non-AD eager comparisons (#98)

All requested QR/SVD outputs are retained. Ordinary einsum uses the same parsed integer notation and default Auto strategy; planning/cache lookup stays inside both public calls. These are public-call diagnostics, not prepared execution or an isolated AD cost. Fresh pairs include session entry; shared pairs exclude it. Views are constructed untimed and consumed as views on both routes. No output to_tensor conversion is timed. Capture is inactive by construction in each fresh process, with no capture_trace guard; inputs and validation outputs are checked untracked.

| Operation | Shape | Layout | Session | Threads | Direct ns | Eager ns | Eager − direct ns | Eager/direct | Quality |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 1 | 2125.75 | 3499.42 | 1373.67 | 1.646 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | shared | 1 | 2136.71 | 3479.24 | 1342.53 | 1.628 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 4 | 2795.68 | 4220.05 | 1424.37 | 1.509 | inconclusive |
| einsum | 2×2 | col_major_contiguous | shared | 4 | 2825.42 | 4121.14 | 1295.72 | 1.459 | inconclusive |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 1 | 2453.78 | 3819.62 | 1365.84 | 1.557 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | shared | 1 | 2492.45 | 3756.04 | 1263.59 | 1.507 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 4 | 3217.58 | 4562.76 | 1345.18 | 1.418 | inconclusive |
| einsum | 16×16 | col_major_contiguous | shared | 4 | 3183.20 | 4486.57 | 1303.37 | 1.409 | inconclusive |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 1 | 23910.49 | 9682.10 | -14228.39 | 0.405 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | shared | 1 | 23899.37 | 9804.98 | -14094.39 | 0.410 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 4 | 25565.47 | 12115.79 | -13449.68 | 0.474 | inconclusive |
| einsum3 | 2×2 | col_major_contiguous | shared | 4 | 25078.45 | 11842.77 | -13235.68 | 0.472 | inconclusive |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1634.01 | 4357.71 | 2723.70 | 2.667 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | shared | 1 | 1636.97 | 4207.31 | 2570.34 | 2.570 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 4 | 3068.79 | 5362.53 | 2293.74 | 1.747 | inconclusive |
| qr | 2×2 | col_major_contiguous | shared | 4 | 2394.13 | 5095.79 | 2701.66 | 2.128 | inconclusive |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 1 | 9075.00 | 11710.41 | 2635.41 | 1.290 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | shared | 1 | 9261.79 | 11905.97 | 2644.18 | 1.285 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 4 | 10044.34 | 12773.58 | 2729.24 | 1.272 | inconclusive |
| qr | 16×16 | col_major_contiguous | shared | 4 | 9844.91 | 12922.37 | 3077.46 | 1.313 | inconclusive |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1602.64 | 5274.44 | 3671.80 | 3.291 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | shared | 1 | 1572.66 | 5106.15 | 3533.49 | 3.247 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 4 | 2275.88 | 5998.05 | 3722.17 | 2.635 | inconclusive |
| solve | 2×2 | col_major_contiguous | shared | 4 | 2188.81 | 5956.22 | 3767.41 | 2.721 | inconclusive |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 1 | 4212.50 | 9276.62 | 5064.12 | 2.202 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | shared | 1 | 4184.73 | 9118.94 | 4934.21 | 2.179 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 4 | 4849.27 | 10187.66 | 5338.39 | 2.101 | inconclusive |
| solve | 16×16 | col_major_contiguous | shared | 4 | 5027.30 | 10127.47 | 5100.17 | 2.014 | inconclusive |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 1 | 2760.15 | 5848.63 | 3088.48 | 2.119 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | shared | 1 | 2783.27 | 5614.09 | 2830.82 | 2.017 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 4 | 3572.36 | 6791.76 | 3219.40 | 1.901 | inconclusive |
| svd | 2×2 | col_major_contiguous | shared | 4 | 3599.46 | 6763.51 | 3164.05 | 1.879 | inconclusive |

## Separate allocation diagnostics

Instrumented executable is built separately after latency collection. Calling-thread Rust allocator requests only; excludes foreign BLAS and worker-thread allocations. Realloc counts its requested new size, not net retained bytes. Output destruction and instrumentation setup are excluded. Helper references and multi-output retention use stack storage. No allocation-instrumented run contributes latency samples.

| Corresponding case ID | Threads | Median allocation requests/op | Median requested bytes/op | Status |
|---|---:|---:|---:|---|
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | 1 | 10 | 1528 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | 1 | 17 | 4160 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | 1 | 10 | 1528 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-shared | 1 | 17 | 4160 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | 1 | 267 | 33325 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | 1 | 51 | 11428 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | 1 | 267 | 33325 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | 1 | 51 | 11428 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | 1 | 1 | 624 | passed |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | 1 | 31 | 5851 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-shared | 1 | 1 | 624 | passed |
| nonad_qr_n2_col_major_contiguous_eager-shared | 1 | 31 | 5851 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | 1 | 1 | 936 | passed |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | 1 | 36 | 7243 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-shared | 1 | 1 | 936 | passed |
| nonad_svd_n2_col_major_contiguous_eager-shared | 1 | 36 | 7243 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | 1 | 0 | 0 | passed |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | 1 | 30 | 7171 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-shared | 1 | 0 | 0 | passed |
| nonad_solve_n2_col_major_contiguous_eager-shared | 1 | 30 | 7171 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-shared | 1 | 17 | 4160 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | 1 | 10 | 1528 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | 1 | 17 | 4160 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | 1 | 10 | 1528 | passed |
| nonad_qr_n16_col_major_contiguous_eager-shared | 1 | 31 | 5851 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-shared | 1 | 1 | 624 | passed |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | 1 | 31 | 5851 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | 1 | 1 | 624 | passed |
| nonad_solve_n16_col_major_contiguous_eager-shared | 1 | 30 | 7171 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-shared | 1 | 0 | 0 | passed |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | 1 | 30 | 7171 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | 1 | 0 | 0 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | 1 | 10 | 1528 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | 1 | 1 | 624 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | 1 | 0 | 0 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | 1 | 10 | 1528 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | 1 | 1 | 624 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | 1 | 0 | 0 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | 4 | 10 | 1528 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | 4 | 17 | 4160 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | 4 | 10 | 1528 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-shared | 4 | 17 | 4160 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | 4 | 267 | 33325 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | 4 | 51 | 11428 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | 4 | 267 | 33325 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | 4 | 51 | 11428 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | 4 | 1 | 624 | passed |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | 4 | 31 | 5851 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-shared | 4 | 1 | 624 | passed |
| nonad_qr_n2_col_major_contiguous_eager-shared | 4 | 31 | 5851 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | 4 | 1 | 936 | passed |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | 4 | 36 | 7243 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-shared | 4 | 1 | 936 | passed |
| nonad_svd_n2_col_major_contiguous_eager-shared | 4 | 36 | 7243 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | 4 | 0 | 0 | passed |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | 4 | 30 | 7171 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-shared | 4 | 0 | 0 | passed |
| nonad_solve_n2_col_major_contiguous_eager-shared | 4 | 30 | 7171 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-shared | 4 | 17 | 4160 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | 4 | 10 | 1528 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | 4 | 17 | 4160 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | 4 | 10 | 1528 | passed |
| nonad_qr_n16_col_major_contiguous_eager-shared | 4 | 31 | 5851 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-shared | 4 | 1 | 624 | passed |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | 4 | 31 | 5851 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | 4 | 1 | 624 | passed |
| nonad_solve_n16_col_major_contiguous_eager-shared | 4 | 30 | 7171 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-shared | 4 | 0 | 0 | passed |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | 4 | 30 | 7171 | passed |
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
  "collection_total_seconds": 0.1733341270000892,
  "case_process_seconds": 0.1226102369992077
}
```

Wall observations `wall_allocations_t4.json`:
```json
{
  "collection_total_seconds": 0.26128910500028724,
  "case_process_seconds": 0.21084157599852915
}
```

Wall observations `wall_command.json`:
```json
{
  "command_seconds_through_collection": 447.321758802,
  "scope": "command startup through collection; excludes report rendering and publication",
  "budget_seconds": 120.0,
  "budget_status": "over_budget; diagnostic only"
}
```

Wall observations `wall_measure_t1.json`:
```json
{
  "collection_total_seconds": 3.4498151210000287,
  "case_process_seconds": 3.3842611910004052
}
```

Wall observations `wall_measure_t4.json`:
```json
{
  "collection_total_seconds": 3.467804387999422,
  "case_process_seconds": 3.402328906996445
}
```

Wall observations `wall_phases.json`:
```json
{
  "validation_and_latency_build_seconds": 4.684503712,
  "metadata_and_latency_collection_seconds": 13.745807268,
  "allocation_diagnostic_build_seconds": 428.224687007,
  "allocation_collection_seconds": 0.601395972,
  "command_before_report_seconds": 447.256393959
}
```

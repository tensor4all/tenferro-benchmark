# CPU Small-Work Benchmark Results

- Suite: `cpu/small_work`
- Raw run: `data/results/amd-cpu/cpu/small_work/20261126_191335`

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

Full metadata: `data/results/amd-cpu/cpu/small_work/20261126_191335/run_t1.yaml`

```yaml
timestamp: '2026-11-26T19:13:35Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - native
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
environment:
  container: devcontainer
  hostname: efb413200329
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
    BENCHMARK_COMMIT: 9b03dd718025b3350d93eaab13a8b93dcc66cab2
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

Full metadata: `data/results/amd-cpu/cpu/small_work/20261126_191335/run_t4.yaml`

```yaml
timestamp: '2026-11-26T19:13:42Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - native
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
environment:
  container: devcontainer
  hostname: efb413200329
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
    BENCHMARK_COMMIT: 9b03dd718025b3350d93eaab13a8b93dcc66cab2
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
- Expected 144, selected 144, executed 144, unsupported 0, failed 0, skipped 0, missing 0, noisy 44.
- Complete: **yes** — only executed cases count as covered; unsupported, skipped, failed, missing and unselected cases never do.

| expected | selected | executed | unsupported | failed | skipped | missing | noisy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 144 | 144 | 144 | 0 | 0 | 0 | 0 | 44 |

- Noisy: `add_f64_concrete-shared_size4_single@t4`, `add_f64_concrete-shared_size4_dependent10@t4`, `add_f64_concrete_shared@t4`, `add_f64_concrete-shared_size16_dependent10@t4`, `add_f64_concrete-shared_size256_single@t4`, `solve_f64_concrete-shared_n2_single@t4`, `solve_f64_concrete-shared_n4_single@t4`, `gather_f64_concrete-shared_size4_single@t4`, `gather_f64_concrete_shared@t4`, `gather_f64_concrete-shared_size256_single@t4`, `reduce_sum_f64_concrete-shared_size4_single@t4`, `reduce_sum_f64_concrete-shared_size16_single@t4`, `reduce_sum_f64_concrete-shared_size256_single@t4`, `einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single@t4`, `einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single@t4`, `nonad_einsum_n2_col_major_contiguous_concrete-shared@t4`, `nonad_einsum3_n2_col_major_contiguous_concrete-fresh@t4`, `nonad_einsum3_n2_col_major_contiguous_concrete-shared@t4`, `nonad_qr_n2_col_major_contiguous_concrete-fresh@t4`, `nonad_qr_n2_col_major_contiguous_eager-no-ad@t4`, `nonad_qr_n2_col_major_contiguous_concrete-shared@t4`, `nonad_qr_n2_col_major_contiguous_eager-shared@t4`, `nonad_svd_n2_col_major_contiguous_concrete-fresh@t4`, `nonad_svd_n2_col_major_contiguous_eager-no-ad@t4`, `nonad_svd_n2_col_major_contiguous_concrete-shared@t4`, `nonad_svd_n2_col_major_contiguous_eager-shared@t4`, `nonad_solve_n2_col_major_contiguous_concrete-fresh@t4`, `nonad_solve_n2_col_major_contiguous_eager-no-ad@t4`, `nonad_solve_n2_col_major_contiguous_concrete-shared@t4`, `nonad_solve_n2_col_major_contiguous_eager-shared@t4`, `nonad_einsum_n16_col_major_contiguous_eager-shared@t4`, `nonad_einsum_n16_col_major_contiguous_eager-no-ad@t4`, `nonad_qr_n16_col_major_contiguous_eager-shared@t4`, `nonad_qr_n16_col_major_contiguous_concrete-shared@t4`, `nonad_qr_n16_col_major_contiguous_eager-no-ad@t4`, `nonad_qr_n16_col_major_contiguous_concrete-fresh@t4`, `nonad_solve_n16_col_major_contiguous_eager-shared@t4`, `nonad_solve_n16_col_major_contiguous_concrete-shared@t4`, `nonad_solve_n16_col_major_contiguous_eager-no-ad@t4`, `nonad_solve_n16_col_major_contiguous_concrete-fresh@t4` (+4 more)

Numerical checks: **144/144 recorded rows passed** (case × thread count).

## Timing

Sequential release runs; explicit initialization and warmups, then calibration toward the declared batch duration (2 ms by default), then the measured batches of the recorded effort (standard: 3 warmups and 15 batches; scan: a low-repetition screen that only marks suspects). Legacy single-call routes start calibration at 1024 workflows; matched non-AD and dependent-chain routes start at 16 and bound retained memory. Each batch uses one wall-clock interval; all outputs are retained until it stops. Statistics divide batch time by iterations. Median and inclusive IQR are in nanoseconds (ns); CoV is sample standard deviation / mean. NOISY means CoV > 10%, a descriptive label only. No noisy rows are excluded.

Workflow time is the total for one call or the complete dependent chain. The separate ns/op column divides the chain median by its operation count, not an independently measured call. Setup rows measure preparation only, not execution.

Inputs and backend construction, independent numerical checks and AD backward checks are outside timing. Returned outputs/plans and every dependent-chain intermediate are retained until after each batch interval; retention containers are prepared before the clock. value downloads/checks are outside. Eager AD rows measure forward execution/recording, not backward.

| Case ID | Operation | API route | Route (path / output / repr / session) | Dtype | Shape | Layout | Workflow | Threads | Provider | Operations/batch | Median batch ms | Median workflow ns | IQR ns | CoV | ns/op (chain only) | Status |
|---|---|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | faer | 2048 | 1.854303 | 905.42 | 2.67 | 4.2% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 1 | faer | 2560 | 2.131792 | 8327.31 | 92.81 | 0.7% | 832.73 | ok |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | faer | 2048 | 1.884970 | 920.40 | 4.10 | 4.2% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 1 | faer | 2560 | 2.107958 | 8234.21 | 34.77 | 0.4% | 823.42 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | faer | 1024 | 0.976694 | 953.80 | 10.88 | 4.4% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 1 | faer | 2560 | 2.386798 | 9323.43 | 126.89 | 0.9% | 932.34 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 1.901832 | 1857.26 | 18.09 | 2.5% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | faer | 1024 | 1.867087 | 1823.33 | 3.82 | 2.6% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 2.245274 | 2192.65 | 13.90 | 1.8% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 1.908374 | 1863.65 | 22.50 | 2.3% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 1 | faer | 1024 | 1.893657 | 1849.27 | 11.41 | 2.8% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 2.863669 | 2796.55 | 28.52 | 4.5% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 1.672704 | 1633.50 | 22.42 | 2.5% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | faer | 1024 | 1.847290 | 1803.99 | 23.94 | 2.4% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 4.397623 | 4294.55 | 26.00 | 1.6% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | faer | 1024 | 1.921339 | 1876.31 | 9.00 | 2.3% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 1 | faer | 1024 | 3.638286 | 3553.01 | 17.39 | 2.0% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 1 | faer | 1024 | 2.048906 | 2000.88 | 14.04 | 2.2% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 1 | faer | 1024 | 8.545330 | 8345.05 | 109.52 | 2.4% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | faer | 2048 | 2.326275 | 1135.88 | 15.70 | 3.7% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | faer | 2048 | 2.398901 | 1171.34 | 16.38 | 3.3% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | faer | 1024 | 1.454847 | 1420.75 | 15.39 | 3.8% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | faer | 2048 | 2.399492 | 1171.63 | 7.85 | 3.4% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | faer | 2048 | 2.403529 | 1173.60 | 13.62 | 3.4% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | faer | 2048 | 2.461457 | 1201.88 | 15.97 | 4.9% | — | ok |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | faer | 1024 | 3.269105 | 3192.49 | 48.50 | 1.8% | — | ok |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | faer | 1024 | 3.282201 | 3205.27 | 22.88 | 1.8% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | faer | 1024 | 2.991678 | 2921.56 | 11.95 | 2.0% | — | ok |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | faer | 1024 | 1.947918 | 1902.26 | 15.81 | 2.4% | — | ok |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | faer | 1024 | 1.902353 | 1857.77 | 17.37 | 3.1% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | faer | 2048 | 3.205587 | 1565.23 | 12.37 | 2.4% | — | ok |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | faer | 1024 | 3.105941 | 3033.15 | 16.35 | 2.7% | — | ok |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | faer | 1024 | 3.232838 | 3157.07 | 41.47 | 3.4% | — | ok |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | faer | 1024 | 2.976860 | 2907.09 | 23.62 | 1.5% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 2.240665 | 2188.15 | 21.24 | 5.5% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 1.847831 | 3609.04 | 57.83 | 1.6% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 2.251395 | 2198.63 | 13.84 | 4.9% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 1.794461 | 3504.81 | 28.93 | 0.6% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 128 | 3.191781 | 24935.79 | 265.77 | 1.0% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 256 | 2.645420 | 10333.67 | 99.66 | 1.1% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 128 | 3.100631 | 24223.68 | 331.29 | 1.4% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 256 | 2.543230 | 9934.49 | 110.56 | 2.3% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 1.690437 | 1650.82 | 16.72 | 6.3% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 2.209026 | 4314.50 | 54.28 | 1.5% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 1.728208 | 1687.70 | 62.01 | 8.9% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 2.170905 | 4240.05 | 23.04 | 0.8% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 2.904384 | 2836.31 | 30.77 | 5.3% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 2.910957 | 5685.46 | 49.88 | 0.8% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 2.931886 | 2863.17 | 34.62 | 4.1% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 2.876623 | 5618.40 | 74.38 | 1.1% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 1.648329 | 1609.70 | 31.56 | 7.1% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 2.674055 | 5222.76 | 88.60 | 1.3% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 1.657315 | 1618.47 | 19.40 | 6.7% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 512 | 2.648366 | 5172.59 | 28.63 | 1.5% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 512 | 1.947357 | 3803.43 | 33.73 | 0.9% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 2.585840 | 2525.23 | 26.88 | 4.9% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 512 | 2.019903 | 3945.12 | 59.54 | 1.2% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 2.630913 | 2569.25 | 28.95 | 5.7% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 3.012968 | 11769.41 | 92.03 | 1.4% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 2.306608 | 9010.19 | 49.07 | 1.2% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 3.056529 | 11939.57 | 72.32 | 1.0% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 2.320044 | 9062.67 | 150.44 | 1.5% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 2.329831 | 9100.90 | 82.68 | 2.0% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 512 | 2.195651 | 4288.38 | 45.11 | 3.2% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 2.349778 | 9178.82 | 149.03 | 1.2% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 512 | 2.156498 | 4211.91 | 39.30 | 2.9% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 2.225677 | 2173.51 | 9.07 | 5.0% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 1.728989 | 1688.47 | 21.33 | 6.5% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | faer | 1024 | 1.666032 | 1626.98 | 17.14 | 6.3% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 1024 | 2.624923 | 2563.40 | 26.49 | 4.3% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 256 | 2.367212 | 9246.92 | 110.91 | 1.7% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | faer | 512 | 2.196633 | 4290.30 | 47.02 | 2.8% | — | ok |
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | faer | 1024 | 1.502167 | 1466.96 | 65.61 | 18.8% | — | NOISY |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 4 | faer | 1280 | 1.904848 | 14881.62 | 61.76 | 11.3% | 1488.16 | NOISY |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | faer | 1024 | 1.567788 | 1531.04 | 61.07 | 17.3% | — | NOISY |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 4 | faer | 1280 | 1.904658 | 14880.14 | 92.75 | 11.3% | 1488.01 | NOISY |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | faer | 1024 | 1.621298 | 1583.30 | 158.00 | 17.8% | — | NOISY |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 4 | faer | 1280 | 1.983596 | 15496.84 | 144.61 | 9.4% | 1549.68 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 2.566765 | 2506.61 | 15.89 | 7.8% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | faer | 1024 | 2.494049 | 2435.59 | 19.91 | 9.4% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 1024 | 2.943058 | 2874.08 | 26.17 | 2.5% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 2.518665 | 2459.63 | 22.97 | 8.7% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 4 | faer | 1024 | 2.567346 | 2507.17 | 15.05 | 6.7% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 4 | faer | 1024 | 3.591188 | 3507.02 | 25.09 | 2.5% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 1024 | 2.330774 | 2276.15 | 36.26 | 16.2% | — | NOISY |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | faer | 1024 | 2.519646 | 2460.59 | 32.48 | 15.0% | — | NOISY |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 1024 | 5.071723 | 4952.85 | 17.46 | 1.5% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | faer | 1024 | 2.544333 | 2484.70 | 19.83 | 7.4% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 4 | faer | 1024 | 4.441437 | 4337.34 | 51.70 | 2.4% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 4 | faer | 1024 | 2.610396 | 2549.21 | 13.16 | 5.9% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 4 | faer | 1024 | 17.724409 | 17308.99 | 56.52 | 1.0% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | faer | 1024 | 1.836892 | 1793.84 | 50.58 | 18.1% | — | NOISY |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | faer | 1024 | 1.821162 | 1778.48 | 20.93 | 18.9% | — | NOISY |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | faer | 1024 | 2.127153 | 2077.30 | 25.74 | 15.5% | — | NOISY |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | faer | 1024 | 1.900450 | 1855.91 | 37.48 | 15.1% | — | NOISY |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | faer | 1024 | 1.803469 | 1761.20 | 25.40 | 17.0% | — | NOISY |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | faer | 1024 | 1.864082 | 1820.39 | 5.08 | 16.0% | — | NOISY |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | faer | 1024 | 3.912579 | 3820.88 | 12.34 | 3.0% | — | ok |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | faer | 1024 | 4.047511 | 3952.65 | 36.58 | 5.8% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | faer | 1024 | 3.699310 | 3612.61 | 14.58 | 8.0% | — | ok |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | faer | 1024 | 2.676380 | 2613.65 | 29.79 | 11.9% | — | NOISY |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | faer | 1024 | 2.561505 | 2501.47 | 21.34 | 8.4% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | faer | 1024 | 2.216711 | 2164.76 | 475.00 | 20.0% | — | NOISY |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | faer | 1024 | 3.792194 | 3703.31 | 24.98 | 1.7% | — | ok |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | faer | 1024 | 3.901978 | 3810.53 | 55.75 | 1.4% | — | ok |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | faer | 1024 | 3.483998 | 3402.34 | 17.07 | 1.5% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 1.465237 | 2861.79 | 58.08 | 9.5% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 2.193588 | 4284.35 | 17.39 | 0.9% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 1.453204 | 2838.29 | 18.37 | 11.2% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 2.160696 | 4220.11 | 30.14 | 0.6% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 64 | 1.644903 | 25701.61 | 695.43 | 17.1% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 128 | 1.582947 | 12366.77 | 152.43 | 6.4% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 64 | 1.697261 | 26519.70 | 464.85 | 15.9% | — | NOISY |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 128 | 1.572447 | 12284.74 | 269.64 | 6.1% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 1.292745 | 2524.89 | 29.76 | 21.0% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 256 | 1.348049 | 5265.82 | 56.94 | 17.9% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 1.257579 | 2456.21 | 54.03 | 21.3% | — | NOISY |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 256 | 1.348499 | 5267.57 | 117.76 | 17.6% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 1.850206 | 3613.68 | 23.15 | 15.6% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 256 | 1.749077 | 6832.33 | 246.87 | 13.5% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 1.839687 | 3593.14 | 29.78 | 16.0% | — | NOISY |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 256 | 1.716146 | 6703.70 | 159.26 | 13.8% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 1.167051 | 2279.40 | 39.10 | 22.8% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 256 | 1.629274 | 6364.35 | 738.14 | 15.5% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 1.156491 | 2258.77 | 61.71 | 23.3% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 256 | 1.565915 | 6116.86 | 150.34 | 16.5% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 256 | 1.216893 | 4753.49 | 88.43 | 12.9% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 512 | 1.654992 | 3232.41 | 25.14 | 5.1% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 256 | 1.188751 | 4643.56 | 79.40 | 12.9% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 512 | 1.637870 | 3198.96 | 51.30 | 5.3% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 128 | 1.632509 | 12753.98 | 151.77 | 13.8% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 128 | 1.282556 | 10019.97 | 153.96 | 17.6% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 128 | 1.661002 | 12976.58 | 256.61 | 13.5% | — | NOISY |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 128 | 1.290371 | 10081.02 | 167.38 | 18.1% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 128 | 1.319705 | 10310.20 | 402.59 | 18.0% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 256 | 1.288376 | 5032.72 | 511.46 | 19.0% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 128 | 1.305769 | 10201.32 | 349.87 | 18.2% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 256 | 1.272597 | 4971.08 | 104.35 | 19.6% | — | NOISY |
| nonad_einsum_n2_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 1.507937 | 2945.19 | 20.62 | 9.4% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 1.301561 | 2542.11 | 60.82 | 20.8% | — | NOISY |
| nonad_solve_n2_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | faer | 512 | 1.159836 | 2265.30 | 26.07 | 23.5% | — | NOISY |
| nonad_einsum_n16_col_major_contiguous_concrete-scoped | einsum | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 512 | 1.622841 | 3169.61 | 42.61 | 4.1% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-scoped | qr | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 128 | 1.262087 | 9860.05 | 161.47 | 18.0% | — | NOISY |
| nonad_solve_n16_col_major_contiguous_concrete-scoped | solve | concrete-scoped | concrete / allocating / owned / execution-scope+session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | faer | 256 | 1.296251 | 5063.48 | 472.48 | 18.5% | — | NOISY |

## Matched non-AD eager comparisons (#98)

All requested QR/SVD outputs are retained. Ordinary einsum uses the same parsed integer notation and default Auto strategy; planning/cache lookup stays inside both public calls. These are public-call diagnostics, not prepared execution or an isolated AD cost. Fresh pairs include session entry; shared pairs exclude it. Views are constructed untimed and consumed as views on both routes. No output to_tensor conversion is timed. Capture is inactive by construction in each fresh process, with no capture_trace guard; inputs and validation outputs are checked untracked.

| Operation | Shape | Layout | Session | Threads | Direct ns | Eager ns | Eager − direct ns | Eager/direct | Quality |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 1 | 2188.15 | 3609.04 | 1420.89 | 1.649 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | shared | 1 | 2198.63 | 3504.81 | 1306.18 | 1.594 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 4 | 2861.79 | 4284.35 | 1422.56 | 1.497 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | shared | 4 | 2838.29 | 4220.11 | 1381.82 | 1.487 | inconclusive |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 1 | 2569.25 | 3945.12 | 1375.87 | 1.536 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | shared | 1 | 2525.23 | 3803.43 | 1278.20 | 1.506 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 4 | 3198.96 | 4643.56 | 1444.60 | 1.452 | inconclusive |
| einsum | 16×16 | col_major_contiguous | shared | 4 | 3232.41 | 4753.49 | 1521.08 | 1.471 | inconclusive |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 1 | 24935.79 | 10333.67 | -14602.12 | 0.414 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | shared | 1 | 24223.68 | 9934.49 | -14289.19 | 0.410 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 4 | 25701.61 | 12366.77 | -13334.84 | 0.481 | inconclusive |
| einsum3 | 2×2 | col_major_contiguous | shared | 4 | 26519.70 | 12284.74 | -14234.96 | 0.463 | inconclusive |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1650.82 | 4314.50 | 2663.68 | 2.614 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | shared | 1 | 1687.70 | 4240.05 | 2552.35 | 2.512 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 4 | 2524.89 | 5265.82 | 2740.93 | 2.086 | inconclusive |
| qr | 2×2 | col_major_contiguous | shared | 4 | 2456.21 | 5267.57 | 2811.36 | 2.145 | inconclusive |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 1 | 9062.67 | 11939.57 | 2876.90 | 1.317 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | shared | 1 | 9010.19 | 11769.41 | 2759.22 | 1.306 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 4 | 10081.02 | 12976.58 | 2895.56 | 1.287 | inconclusive |
| qr | 16×16 | col_major_contiguous | shared | 4 | 10019.97 | 12753.98 | 2734.01 | 1.273 | inconclusive |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1609.70 | 5222.76 | 3613.06 | 3.245 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | shared | 1 | 1618.47 | 5172.59 | 3554.12 | 3.196 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 4 | 2279.40 | 6364.35 | 4084.95 | 2.792 | inconclusive |
| solve | 2×2 | col_major_contiguous | shared | 4 | 2258.77 | 6116.86 | 3858.09 | 2.708 | inconclusive |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 1 | 4211.91 | 9178.82 | 4966.91 | 2.179 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | shared | 1 | 4288.38 | 9100.90 | 4812.52 | 2.122 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 4 | 4971.08 | 10201.32 | 5230.24 | 2.052 | inconclusive |
| solve | 16×16 | col_major_contiguous | shared | 4 | 5032.72 | 10310.20 | 5277.48 | 2.049 | inconclusive |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 1 | 2836.31 | 5685.46 | 2849.15 | 2.005 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | shared | 1 | 2863.17 | 5618.40 | 2755.23 | 1.962 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 4 | 3613.68 | 6832.33 | 3218.65 | 1.891 | inconclusive |
| svd | 2×2 | col_major_contiguous | shared | 4 | 3593.14 | 6703.70 | 3110.56 | 1.866 | inconclusive |

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
  "collection_total_seconds": 0.1883194120000553,
  "case_process_seconds": 0.13791392999701202
}
```

Wall observations `wall_allocations_t4.json`:
```json
{
  "collection_total_seconds": 0.34969755099973554,
  "case_process_seconds": 0.297423144998902
}
```

Wall observations `wall_command.json`:
```json
{
  "command_seconds_through_collection": 444.626063075,
  "scope": "command startup through collection; excludes report rendering and publication",
  "budget_seconds": 120.0,
  "budget_status": "over_budget; diagnostic only"
}
```

Wall observations `wall_measure_t1.json`:
```json
{
  "collection_total_seconds": 3.438392084000043,
  "case_process_seconds": 3.3733332420024453
}
```

Wall observations `wall_measure_t4.json`:
```json
{
  "collection_total_seconds": 3.4853894360003324,
  "case_process_seconds": 3.420907057002296
}
```

Wall observations `wall_phases.json`:
```json
{
  "validation_and_latency_build_seconds": 4.803469643,
  "metadata_and_latency_collection_seconds": 13.808392164,
  "allocation_diagnostic_build_seconds": 425.237710849,
  "allocation_collection_seconds": 0.708402201,
  "command_before_report_seconds": 444.557974857
}
```

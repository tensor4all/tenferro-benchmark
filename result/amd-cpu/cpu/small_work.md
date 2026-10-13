# CPU Small-Work Benchmark Results

- Suite: `cpu/small_work`
- Raw run: `data/results/amd-cpu/cpu/small_work/20261013_085616`

## CPU Information

- Model: `Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `16`
- Sockets: `1`
- Cores per socket: `16`
- Threads per core: `1`
- NUMA nodes: `unknown`
- Python platform: `Linux-7.0.14-linuxkit-x86_64-with-glibc2.39`

## run_t1

Full metadata: `data/results/amd-cpu/cpu/small_work/20261013_085616/run_t1.yaml`

```yaml
timestamp: '2026-10-13T08:56:16Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
  commit: 8b13220c1e0397a29a27be65557c24105352a35b
  dirty: false
  features:
  - blas-mkl
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: mkl
  version: 2026.0.1
  root: /opt/intel/oneapi/mkl/latest
  library: /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so
environment:
  hostname: f88adb25e9bd
  os: Linux-7.0.14-linuxkit-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  env:
    OMP_NUM_THREADS: '1'
    OMP_THREAD_LIMIT: '1'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: amd-cpu
    BENCHMARK_COMMIT: 852967968f6f25a2c94b5dcdc8572ff0584858f3
    OPENBLAS_ROOT: /opt/openblas
    MKLROOT: /opt/intel/oneapi/mkl/latest
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
    CARGO_TARGET_DIR: /tmp/tenferro-candidates-target
    USE_CUDA: '0'
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.0 (b940084d7 2026-09-28)
    BENCH_INSTANCE: null
```

## run_t4

Full metadata: `data/results/amd-cpu/cpu/small_work/20261013_085616/run_t4.yaml`

```yaml
timestamp: '2026-10-13T08:57:29Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
  commit: 8b13220c1e0397a29a27be65557c24105352a35b
  dirty: false
  features:
  - blas-mkl
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: mkl
  version: 2026.0.1
  root: /opt/intel/oneapi/mkl/latest
  library: /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so
environment:
  hostname: f88adb25e9bd
  os: Linux-7.0.14-linuxkit-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  env:
    OMP_NUM_THREADS: '4'
    OMP_THREAD_LIMIT: '4'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '4'
    BENCHMARK_TARGET_PROFILE: amd-cpu
    BENCHMARK_COMMIT: 852967968f6f25a2c94b5dcdc8572ff0584858f3
    OPENBLAS_ROOT: /opt/openblas
    MKLROOT: /opt/intel/oneapi/mkl/latest
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
    CARGO_TARGET_DIR: /tmp/tenferro-candidates-target
    USE_CUDA: '0'
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.0 (b940084d7 2026-09-28)
    BENCH_INSTANCE: null
```

## Case status

- Manifest: `cpu/small_work` version 2, coverage `full`, effort `standard`
- Expected 572, selected 572, executed 572, unsupported 0, failed 0, missing 0, noisy 2.
- Complete: **yes** — only executed cases count as covered; unsupported, failed, missing and unselected cases never do.
- Noisy: `einsum_f64_compiled-setup_n4_single@t4`, `einsum_f64_compiled-setup_n16_single@t4`

Numerical checks: **572/572 recorded rows passed** (case × thread count).

## Timing

Sequential release runs; warmups, then calibration from at least 1024 operations toward the declared batch duration (2 ms by default; memory bounded for non-AD cases), then the measured batches of the recorded effort (standard: 3 warmups and 15 batches; scan: a low-repetition screen that only marks suspects). Each batch uses one wall-clock interval; all outputs are retained until it stops. Statistics divide batch time by iterations. Median and inclusive IQR are in nanoseconds (ns); CoV is sample standard deviation / mean. NOISY means CoV > 10%, a descriptive label only. No noisy rows are excluded.

Workflow time is the total for one call or the complete dependent chain. The separate ns/op column divides the chain median by its operation count, not an independently measured call. Setup rows measure preparation only, not execution.

Inputs and backend construction, independent numerical checks and AD backward checks are outside timing. Returned outputs/plans are retained until after each batch interval; value downloads/checks are outside. Eager AD rows measure forward execution/recording, not backward.

| Case ID | Operation | API route | Route (path / output / repr / session) | Dtype | Shape | Layout | Workflow | Threads | Provider | Operations/batch | Median batch ms | Median workflow ns | IQR ns | CoV | ns/op (chain only) | Status |
|---|---|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 2048 | 2.129878 | 1039.98 | 17.27 | 2.2% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 1 | blas | 10240 | 10.483835 | 10238.12 | 85.00 | 0.6% | 1023.81 | ok |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 2048 | 2.149233 | 1049.43 | 27.13 | 3.7% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 1 | blas | 10240 | 10.655922 | 10406.17 | 104.69 | 0.7% | 1040.62 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 2048 | 2.234834 | 1091.23 | 25.95 | 2.7% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 1 | blas | 10240 | 11.135116 | 10874.14 | 131.98 | 0.8% | 1087.41 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 11.939989 | 11660.15 | 155.60 | 0.9% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 12.021161 | 11739.42 | 163.73 | 1.0% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 13.156980 | 12848.61 | 164.35 | 0.9% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 14.496641 | 14156.88 | 181.92 | 1.0% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 14.697587 | 14353.11 | 145.46 | 1.4% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 15.623818 | 15257.63 | 201.22 | 1.3% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.222004 | 1084.96 | 22.25 | 2.3% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 2048 | 2.382590 | 1163.37 | 11.58 | 0.8% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 3.959250 | 3866.46 | 29.91 | 1.8% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 11.634875 | 11362.18 | 80.75 | 0.8% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 14.046746 | 13717.53 | 183.89 | 1.1% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 14.484568 | 14145.09 | 187.12 | 1.2% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 20.817518 | 20329.61 | 231.05 | 1.1% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 2048 | 2.486778 | 1214.25 | 55.38 | 3.9% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 2048 | 2.444007 | 1193.36 | 17.15 | 2.6% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 1024 | 1.458667 | 1424.48 | 41.62 | 2.4% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 2048 | 1.878182 | 917.08 | 13.15 | 2.5% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 2048 | 1.849457 | 903.06 | 11.11 | 2.4% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 2048 | 1.981965 | 967.76 | 20.87 | 2.4% | — | ok |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 18.960695 | 18516.30 | 288.03 | 1.1% | — | ok |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 14.086925 | 13756.76 | 129.90 | 0.8% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 13.912635 | 13586.56 | 165.41 | 1.0% | — | ok |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 1024 | 12.527479 | 12233.87 | 244.47 | 3.2% | — | ok |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 1024 | 8.762216 | 8556.85 | 340.85 | 4.1% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 1024 | 8.361126 | 8165.16 | 152.26 | 2.1% | — | ok |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 1024 | 13.708972 | 13387.67 | 261.78 | 1.5% | — | ok |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 1024 | 9.979221 | 9745.33 | 149.57 | 1.3% | — | ok |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 1024 | 9.806753 | 9576.91 | 128.73 | 1.7% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 128 | 3.031626 | 23684.58 | 153.74 | 0.6% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 128 | 1.951312 | 15244.62 | 200.11 | 0.9% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 128 | 2.799246 | 21869.11 | 271.02 | 1.0% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 3.238917 | 12652.02 | 167.96 | 0.8% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 64 | 1.835511 | 28679.86 | 760.11 | 2.6% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 64 | 2.246413 | 35100.20 | 359.52 | 1.1% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 128 | 3.429955 | 26796.52 | 349.10 | 1.0% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 64 | 2.017867 | 31529.17 | 897.25 | 2.6% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.217831 | 2165.85 | 31.81 | 1.1% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 2.141843 | 8366.57 | 111.13 | 2.4% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.510026 | 1225.60 | 24.81 | 1.5% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 3.310281 | 6465.39 | 152.98 | 2.1% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.007053 | 3920.03 | 51.25 | 1.9% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 3.019999 | 11796.87 | 253.59 | 1.6% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.639433 | 2577.57 | 68.40 | 1.7% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 256 | 2.500907 | 9769.17 | 212.01 | 2.6% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 1.947518 | 1901.87 | 34.13 | 1.6% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 3.521884 | 6878.68 | 80.15 | 0.9% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 2048 | 2.193889 | 1071.23 | 8.08 | 0.9% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 512 | 2.843447 | 5553.61 | 91.22 | 1.1% | — | ok |
| nonad_einsum_n4_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 256 | 3.281063 | 12816.65 | 360.71 | 1.7% | — | ok |
| nonad_einsum_n4_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 128 | 2.876619 | 22473.59 | 136.95 | 1.0% | — | ok |
| nonad_einsum_n4_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 128 | 1.956120 | 15282.19 | 282.86 | 2.4% | — | ok |
| nonad_einsum_n4_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 128 | 3.363181 | 26274.85 | 303.98 | 0.9% | — | ok |
| nonad_einsum3_n4_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 64 | 2.051940 | 32061.56 | 477.01 | 2.5% | — | ok |
| nonad_einsum3_n4_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 128 | 3.372255 | 26345.74 | 294.85 | 1.3% | — | ok |
| nonad_einsum3_n4_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 64 | 2.292848 | 35825.75 | 867.77 | 2.6% | — | ok |
| nonad_einsum3_n4_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 128 | 3.695940 | 28874.53 | 361.34 | 1.1% | — | ok |
| nonad_qr_n4_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 512 | 3.472783 | 6782.78 | 114.65 | 1.7% | — | ok |
| nonad_qr_n4_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 2048 | 3.145235 | 1535.76 | 6.30 | 1.0% | — | ok |
| nonad_qr_n4_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 256 | 2.309894 | 9023.02 | 178.99 | 1.6% | — | ok |
| nonad_qr_n4_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 2.425684 | 2368.83 | 30.02 | 0.9% | — | ok |
| nonad_svd_n4_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 256 | 3.399965 | 13281.11 | 202.07 | 1.8% | — | ok |
| nonad_svd_n4_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 512 | 2.894285 | 5652.90 | 20.06 | 1.1% | — | ok |
| nonad_svd_n4_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 128 | 2.025304 | 15822.69 | 414.52 | 2.3% | — | ok |
| nonad_svd_n4_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 512 | 3.671377 | 7170.66 | 98.74 | 1.2% | — | ok |
| nonad_solve_n4_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 512 | 2.883216 | 5631.28 | 34.52 | 0.5% | — | ok |
| nonad_solve_n4_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 2048 | 2.380783 | 1162.49 | 8.26 | 1.3% | — | ok |
| nonad_solve_n4_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 512 | 3.630566 | 7090.95 | 94.65 | 1.2% | — | ok |
| nonad_solve_n4_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 2.046731 | 1998.76 | 26.73 | 0.9% | — | ok |
| nonad_einsum_n8_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 128 | 3.086649 | 24114.45 | 668.58 | 2.3% | — | ok |
| nonad_einsum_n8_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 128 | 1.889204 | 14759.41 | 229.25 | 2.2% | — | ok |
| nonad_einsum_n8_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 128 | 2.847426 | 22245.52 | 229.91 | 0.8% | — | ok |
| nonad_einsum_n8_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 256 | 3.382609 | 13213.32 | 192.38 | 1.3% | — | ok |
| nonad_einsum3_n8_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 128 | 3.873593 | 30262.45 | 435.78 | 1.1% | — | ok |
| nonad_einsum3_n8_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 64 | 2.273803 | 35528.17 | 503.91 | 1.2% | — | ok |
| nonad_einsum3_n8_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 128 | 3.568241 | 27876.88 | 545.74 | 1.3% | — | ok |
| nonad_einsum3_n8_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 64 | 2.096367 | 32755.73 | 313.48 | 1.2% | — | ok |
| nonad_qr_n8_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 3.206576 | 3131.42 | 30.92 | 0.8% | — | ok |
| nonad_qr_n8_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 256 | 2.417305 | 9442.60 | 160.83 | 1.1% | — | ok |
| nonad_qr_n8_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 2.297737 | 2243.88 | 24.86 | 1.3% | — | ok |
| nonad_qr_n8_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 256 | 1.938557 | 7572.49 | 172.86 | 3.0% | — | ok |
| nonad_svd_n8_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 128 | 1.983933 | 15499.48 | 83.44 | 0.9% | — | ok |
| nonad_svd_n8_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 128 | 3.139566 | 24527.86 | 188.12 | 1.8% | — | ok |
| nonad_svd_n8_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 256 | 3.535635 | 13811.07 | 67.69 | 0.7% | — | ok |
| nonad_svd_n8_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 128 | 2.761382 | 21573.30 | 174.95 | 1.4% | — | ok |
| nonad_solve_n8_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 2.738824 | 2674.63 | 52.82 | 1.4% | — | ok |
| nonad_solve_n8_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 256 | 1.994287 | 7790.18 | 91.17 | 1.1% | — | ok |
| nonad_solve_n8_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 1.859819 | 1816.23 | 25.37 | 1.3% | — | ok |
| nonad_solve_n8_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 512 | 3.277810 | 6401.97 | 101.32 | 1.4% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 3.419746 | 13358.38 | 219.06 | 1.2% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 128 | 2.860513 | 22347.76 | 310.92 | 1.4% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 128 | 2.007332 | 15682.28 | 242.04 | 1.8% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 128 | 3.130328 | 24455.69 | 130.71 | 1.4% | — | ok |
| nonad_einsum_n16_row_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 256 | 3.642826 | 14229.79 | 262.03 | 1.9% | — | ok |
| nonad_einsum_n16_row_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 128 | 3.183275 | 24869.34 | 251.02 | 1.6% | — | ok |
| nonad_einsum_n16_row_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 128 | 2.107235 | 16462.77 | 231.23 | 1.2% | — | ok |
| nonad_einsum_n16_row_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 128 | 3.443530 | 26902.58 | 458.22 | 1.2% | — | ok |
| nonad_einsum3_n16_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 64 | 2.072801 | 32387.52 | 311.66 | 1.2% | — | ok |
| nonad_einsum3_n16_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 128 | 3.497354 | 27323.08 | 466.07 | 1.8% | — | ok |
| nonad_einsum3_n16_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 64 | 2.295200 | 35862.50 | 701.25 | 1.5% | — | ok |
| nonad_einsum3_n16_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 64 | 1.912853 | 29888.33 | 656.73 | 2.2% | — | ok |
| nonad_einsum3_n16_row_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 64 | 2.120265 | 33129.14 | 274.79 | 1.3% | — | ok |
| nonad_einsum3_n16_row_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 64 | 1.912295 | 29879.61 | 493.23 | 1.5% | — | ok |
| nonad_einsum3_n16_row_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 64 | 2.284489 | 35695.14 | 347.61 | 1.9% | — | ok |
| nonad_einsum3_n16_row_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 64 | 2.059658 | 32182.16 | 855.28 | 2.6% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 2.578960 | 10074.06 | 213.77 | 1.6% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.876273 | 5617.72 | 61.70 | 1.6% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 3.072469 | 12001.83 | 183.64 | 1.1% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 3.395727 | 6632.28 | 37.54 | 1.0% | — | ok |
| nonad_qr_n16_row_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 256 | 2.792234 | 10907.16 | 335.69 | 2.1% | — | ok |
| nonad_qr_n16_row_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 512 | 3.904397 | 7625.78 | 126.08 | 1.3% | — | ok |
| nonad_qr_n16_row_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 256 | 3.285374 | 12833.49 | 157.55 | 1.4% | — | ok |
| nonad_qr_n16_row_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 256 | 1.876556 | 7330.30 | 78.11 | 0.9% | — | ok |
| nonad_svd_n16_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 64 | 2.737368 | 42771.38 | 711.64 | 1.8% | — | ok |
| nonad_svd_n16_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 64 | 2.162420 | 33787.81 | 354.73 | 1.7% | — | ok |
| nonad_svd_n16_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 64 | 2.880583 | 45009.11 | 794.58 | 1.3% | — | ok |
| nonad_svd_n16_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 64 | 2.279090 | 35610.78 | 565.94 | 1.9% | — | ok |
| nonad_svd_n16_row_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 64 | 2.804448 | 43819.50 | 667.58 | 2.3% | — | ok |
| nonad_svd_n16_row_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 64 | 2.338943 | 36545.98 | 636.33 | 2.4% | — | ok |
| nonad_svd_n16_row_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 64 | 3.026920 | 47295.62 | 627.34 | 2.4% | — | ok |
| nonad_svd_n16_row_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 64 | 2.464827 | 38512.92 | 539.19 | 1.1% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 2.124959 | 8300.62 | 97.19 | 1.3% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 4.174150 | 4076.32 | 34.96 | 1.1% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 256 | 2.572897 | 10050.38 | 179.84 | 1.8% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 512 | 2.159451 | 4217.68 | 30.00 | 1.1% | — | ok |
| nonad_solve_n16_row_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 256 | 2.277107 | 8894.95 | 130.10 | 1.4% | — | ok |
| nonad_solve_n16_row_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 512 | 1.831391 | 3576.94 | 27.49 | 1.7% | — | ok |
| nonad_solve_n16_row_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 256 | 2.735277 | 10684.68 | 360.14 | 2.3% | — | ok |
| nonad_solve_n16_row_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 1 | blas | 512 | 2.326156 | 4543.27 | 59.18 | 2.0% | — | ok |
| nonad_einsum_n64_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 64 | 2.212580 | 34571.56 | 1429.73 | 2.6% | — | ok |
| nonad_einsum_n64_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 128 | 3.222605 | 25176.60 | 962.76 | 2.4% | — | ok |
| nonad_einsum_n64_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 64 | 2.150676 | 33604.31 | 787.32 | 2.6% | — | ok |
| nonad_einsum_n64_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 128 | 2.931745 | 22904.26 | 637.94 | 1.7% | — | ok |
| nonad_einsum3_n64_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 64 | 3.231973 | 50499.58 | 1490.25 | 1.9% | — | ok |
| nonad_einsum3_n64_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 32 | 1.875432 | 58607.25 | 1072.88 | 2.4% | — | ok |
| nonad_einsum3_n64_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 64 | 3.142132 | 49095.81 | 1035.02 | 1.5% | — | ok |
| nonad_einsum3_n64_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 64 | 3.589737 | 56089.64 | 1498.09 | 2.4% | — | ok |
| nonad_qr_n64_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 32 | 1.736977 | 54280.53 | 813.47 | 1.3% | — | ok |
| nonad_qr_n64_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 32 | 2.125943 | 66435.72 | 964.92 | 1.2% | — | ok |
| nonad_qr_n64_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 32 | 1.666862 | 52089.44 | 989.72 | 1.9% | — | ok |
| nonad_qr_n64_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 32 | 2.060952 | 64404.75 | 1265.55 | 2.7% | — | ok |
| nonad_svd_n64_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 16 | 6.527046 | 407940.38 | 4280.59 | 1.0% | — | ok |
| nonad_svd_n64_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 16 | 6.792179 | 424511.19 | 2524.78 | 0.9% | — | ok |
| nonad_svd_n64_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 16 | 6.465427 | 404089.19 | 4361.00 | 1.1% | — | ok |
| nonad_svd_n64_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 16 | 6.687965 | 417997.81 | 2695.66 | 0.8% | — | ok |
| nonad_solve_n64_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 64 | 2.741317 | 42833.08 | 973.93 | 1.9% | — | ok |
| nonad_solve_n64_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 64 | 2.556970 | 39952.66 | 999.94 | 1.9% | — | ok |
| nonad_solve_n64_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 64 | 2.631499 | 41117.17 | 624.63 | 1.4% | — | ok |
| nonad_solve_n64_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 1 | blas | 64 | 2.408650 | 37635.16 | 732.08 | 3.1% | — | ok |
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 1024 | 85.903237 | 83889.88 | 2843.40 | 2.3% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 4 | blas | 10240 | 870.606501 | 850201.66 | 5697.43 | 1.1% | 85020.17 | ok |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 1024 | 85.656701 | 83649.12 | 2025.08 | 1.9% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 4 | blas | 10240 | 862.882039 | 842658.24 | 8105.43 | 0.6% | 84265.82 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 86.174106 | 84154.40 | 967.32 | 1.0% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 4 | blas | 10240 | 862.187228 | 841979.71 | 6646.51 | 0.7% | 84197.97 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 12.409788 | 12118.93 | 619.61 | 3.7% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 12.076018 | 11792.99 | 143.86 | 2.9% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 12.604592 | 12309.17 | 237.07 | 2.0% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 15.111192 | 14757.02 | 136.64 | 2.8% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 14.704539 | 14359.90 | 242.61 | 1.9% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 15.598802 | 15233.21 | 215.35 | 2.1% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 2.274017 | 1110.36 | 14.56 | 1.4% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 2048 | 2.417658 | 1180.50 | 20.11 | 1.2% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 4.151652 | 4054.35 | 86.34 | 2.2% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 12.064933 | 11782.16 | 286.11 | 2.7% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 17.593993 | 17181.63 | 814.66 | 3.9% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 14.380681 | 14043.63 | 217.75 | 3.4% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 20.752759 | 20266.37 | 428.62 | 1.9% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 1024 | 83.711937 | 81749.94 | 1451.57 | 1.6% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 1024 | 83.793168 | 81829.27 | 1663.70 | 1.7% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 83.727054 | 81764.70 | 1565.97 | 1.5% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 1024 | 83.640537 | 81680.21 | 1220.51 | 2.1% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 1024 | 83.107992 | 81160.15 | 1097.03 | 1.3% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 83.126091 | 81177.82 | 1047.86 | 0.9% | — | ok |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 18.537256 | 18102.79 | 243.43 | 1.1% | — | ok |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 14.252750 | 13918.70 | 217.82 | 1.0% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 13.773836 | 13451.01 | 125.92 | 0.9% | — | ok |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 1024 | 12.636148 | 12339.99 | 183.16 | 1.0% | — | ok |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 1024 | 8.831001 | 8624.02 | 110.25 | 0.9% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 1024 | 8.544637 | 8344.37 | 146.10 | 1.4% | — | ok |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 1024 | 13.123549 | 12815.97 | 200.54 | 1.2% | — | ok |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 1024 | 10.008298 | 9773.73 | 168.59 | 1.3% | — | ok |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 1024 | 9.622142 | 9396.62 | 173.61 | 1.3% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 128 | 3.131375 | 24463.87 | 240.13 | 0.9% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 128 | 1.913477 | 14949.04 | 238.80 | 2.7% | — | ok |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 128 | 2.866358 | 22393.42 | 314.50 | 1.4% | — | ok |
| nonad_einsum_n2_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 256 | 3.261779 | 12741.32 | 252.47 | 2.1% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 128 | 3.762137 | 29391.70 | 880.54 | 2.7% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 16 | 2.668684 | 166792.75 | 7794.97 | 3.2% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 128 | 3.431449 | 26808.20 | 484.64 | 1.2% | — | ok |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 16 | 2.525476 | 157842.25 | 4012.69 | 2.1% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.193288 | 2141.88 | 33.11 | 1.4% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 16 | 2.232196 | 139512.25 | 7537.72 | 4.3% | — | ok |
| nonad_qr_n2_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 2.599684 | 1269.38 | 18.20 | 1.8% | — | ok |
| nonad_qr_n2_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 16 | 2.100952 | 131309.50 | 5517.59 | 3.0% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.108034 | 4117.25 | 36.09 | 1.5% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 16 | 2.276920 | 142307.50 | 5441.47 | 2.8% | — | ok |
| nonad_svd_n2_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.802406 | 2736.72 | 22.83 | 0.8% | — | ok |
| nonad_svd_n2_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 16 | 2.193944 | 137121.50 | 3730.91 | 3.8% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.925212 | 1880.09 | 18.58 | 1.0% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 3.488929 | 6814.31 | 123.29 | 1.4% | — | ok |
| nonad_solve_n2_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 2048 | 2.205356 | 1076.83 | 22.09 | 1.3% | — | ok |
| nonad_solve_n2_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 512 | 2.814251 | 5496.58 | 102.27 | 1.6% | — | ok |
| nonad_einsum_n4_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 256 | 3.272445 | 12782.99 | 341.06 | 1.7% | — | ok |
| nonad_einsum_n4_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 128 | 2.901139 | 22665.15 | 580.84 | 2.7% | — | ok |
| nonad_einsum_n4_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 128 | 1.903569 | 14871.63 | 182.35 | 1.3% | — | ok |
| nonad_einsum_n4_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 128 | 3.266756 | 25521.53 | 326.25 | 1.6% | — | ok |
| nonad_einsum3_n4_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 16 | 2.641559 | 165097.44 | 6073.97 | 3.3% | — | ok |
| nonad_einsum3_n4_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 128 | 3.498977 | 27335.76 | 366.54 | 1.9% | — | ok |
| nonad_einsum3_n4_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 16 | 2.658914 | 166182.12 | 8750.75 | 3.7% | — | ok |
| nonad_einsum3_n4_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 128 | 3.818399 | 29831.24 | 271.37 | 1.5% | — | ok |
| nonad_qr_n4_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 16 | 2.126556 | 132909.75 | 7560.28 | 4.1% | — | ok |
| nonad_qr_n4_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 2048 | 3.197440 | 1561.25 | 41.78 | 1.8% | — | ok |
| nonad_qr_n4_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 16 | 2.135031 | 133439.44 | 9335.12 | 5.6% | — | ok |
| nonad_qr_n4_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 2.475485 | 2417.47 | 30.99 | 1.8% | — | ok |
| nonad_svd_n4_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 16 | 2.282250 | 142640.62 | 6754.56 | 4.4% | — | ok |
| nonad_svd_n4_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 512 | 2.835701 | 5538.48 | 57.27 | 1.4% | — | ok |
| nonad_svd_n4_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 16 | 2.318923 | 144932.69 | 5563.84 | 3.1% | — | ok |
| nonad_svd_n4_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 512 | 3.629193 | 7088.27 | 161.70 | 1.7% | — | ok |
| nonad_solve_n4_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 512 | 2.866899 | 5599.41 | 50.38 | 1.3% | — | ok |
| nonad_solve_n4_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 2048 | 2.343641 | 1144.36 | 24.68 | 1.5% | — | ok |
| nonad_solve_n4_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 512 | 3.641222 | 7111.76 | 64.96 | 1.9% | — | ok |
| nonad_solve_n4_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 2.015458 | 1968.22 | 38.03 | 1.4% | — | ok |
| nonad_einsum_n8_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 128 | 3.179107 | 24836.77 | 690.04 | 1.9% | — | ok |
| nonad_einsum_n8_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 128 | 1.935839 | 15123.74 | 265.66 | 1.9% | — | ok |
| nonad_einsum_n8_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 128 | 2.840857 | 22194.20 | 427.05 | 2.0% | — | ok |
| nonad_einsum_n8_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 256 | 3.261950 | 12741.99 | 147.62 | 1.8% | — | ok |
| nonad_einsum3_n8_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 128 | 3.990376 | 31174.81 | 788.71 | 1.9% | — | ok |
| nonad_einsum3_n8_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 16 | 2.633804 | 164612.75 | 7913.78 | 3.8% | — | ok |
| nonad_einsum3_n8_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 128 | 3.636996 | 28414.03 | 243.86 | 1.1% | — | ok |
| nonad_einsum3_n8_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 16 | 2.566206 | 160387.88 | 7565.84 | 4.6% | — | ok |
| nonad_qr_n8_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 3.236381 | 3160.53 | 39.59 | 1.6% | — | ok |
| nonad_qr_n8_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 16 | 2.177942 | 136121.38 | 4020.16 | 3.4% | — | ok |
| nonad_qr_n8_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 2.276073 | 2222.73 | 32.66 | 1.5% | — | ok |
| nonad_qr_n8_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 16 | 2.121008 | 132563.00 | 12311.72 | 6.1% | — | ok |
| nonad_svd_n8_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 128 | 1.954681 | 15270.95 | 401.16 | 2.0% | — | ok |
| nonad_svd_n8_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 16 | 2.492159 | 155759.94 | 3791.25 | 2.9% | — | ok |
| nonad_svd_n8_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 256 | 3.547625 | 13857.91 | 111.83 | 1.7% | — | ok |
| nonad_svd_n8_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 16 | 2.406575 | 150410.94 | 4173.62 | 3.8% | — | ok |
| nonad_solve_n8_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 2.736641 | 2672.50 | 18.61 | 1.0% | — | ok |
| nonad_solve_n8_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 256 | 1.928745 | 7534.16 | 59.37 | 1.2% | — | ok |
| nonad_solve_n8_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 1.890431 | 1846.12 | 16.40 | 1.3% | — | ok |
| nonad_solve_n8_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 512 | 3.174638 | 6200.46 | 34.94 | 0.9% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 3.336537 | 13033.35 | 224.70 | 1.4% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 128 | 2.888247 | 22564.43 | 301.92 | 1.3% | — | ok |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 128 | 2.005687 | 15669.43 | 343.37 | 2.9% | — | ok |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 128 | 3.156257 | 24658.26 | 360.60 | 1.4% | — | ok |
| nonad_einsum_n16_row_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 256 | 3.515617 | 13732.88 | 226.47 | 1.2% | — | ok |
| nonad_einsum_n16_row_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 128 | 3.138640 | 24520.62 | 1061.29 | 2.7% | — | ok |
| nonad_einsum_n16_row_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 128 | 2.075497 | 16214.82 | 206.34 | 2.1% | — | ok |
| nonad_einsum_n16_row_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 128 | 3.396602 | 26535.95 | 706.69 | 2.0% | — | ok |
| nonad_einsum3_n16_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 16 | 2.598454 | 162403.38 | 8277.47 | 3.4% | — | ok |
| nonad_einsum3_n16_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 128 | 3.538520 | 27644.69 | 983.32 | 3.5% | — | ok |
| nonad_einsum3_n16_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 16 | 2.649588 | 165599.25 | 6693.47 | 2.9% | — | ok |
| nonad_einsum3_n16_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 64 | 1.914958 | 29921.22 | 932.11 | 3.4% | — | ok |
| nonad_einsum3_n16_row_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 16 | 2.681060 | 167566.25 | 5010.94 | 3.5% | — | ok |
| nonad_einsum3_n16_row_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 64 | 1.910805 | 29856.33 | 449.79 | 2.4% | — | ok |
| nonad_einsum3_n16_row_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 16 | 2.645245 | 165327.81 | 7221.81 | 2.9% | — | ok |
| nonad_einsum3_n16_row_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 64 | 2.064205 | 32253.20 | 1153.29 | 2.4% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 16 | 2.188397 | 136774.81 | 5908.22 | 2.7% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.846035 | 5558.66 | 78.35 | 1.0% | — | ok |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 16 | 2.263851 | 141490.69 | 8706.38 | 4.5% | — | ok |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 3.369169 | 6580.41 | 71.03 | 1.1% | — | ok |
| nonad_qr_n16_row_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 16 | 2.218188 | 138636.75 | 10369.06 | 4.3% | — | ok |
| nonad_qr_n16_row_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 16 | 2.031753 | 126984.56 | 4519.59 | 2.9% | — | ok |
| nonad_qr_n16_row_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 16 | 2.228756 | 139297.25 | 8120.66 | 3.8% | — | ok |
| nonad_qr_n16_row_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 16 | 2.073117 | 129569.81 | 7285.62 | 4.1% | — | ok |
| nonad_svd_n16_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 16 | 2.772557 | 173284.81 | 7930.22 | 3.5% | — | ok |
| nonad_svd_n16_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 64 | 2.168745 | 33886.64 | 932.27 | 2.5% | — | ok |
| nonad_svd_n16_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 16 | 2.843582 | 177723.88 | 9369.19 | 3.4% | — | ok |
| nonad_svd_n16_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 64 | 2.252214 | 35190.84 | 560.20 | 1.1% | — | ok |
| nonad_svd_n16_row_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 16 | 2.805682 | 175355.12 | 5975.12 | 2.3% | — | ok |
| nonad_svd_n16_row_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 16 | 2.678752 | 167422.00 | 10050.62 | 4.8% | — | ok |
| nonad_svd_n16_row_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 16 | 2.911447 | 181965.44 | 9299.19 | 3.6% | — | ok |
| nonad_svd_n16_row_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 16 | 2.714489 | 169655.56 | 9474.25 | 4.0% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 2.080684 | 8127.67 | 101.55 | 1.7% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 1.648798 | 3220.31 | 40.05 | 1.7% | — | ok |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 256 | 2.537918 | 9913.74 | 184.82 | 1.6% | — | ok |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 512 | 2.175708 | 4249.43 | 102.70 | 1.3% | — | ok |
| nonad_solve_n16_row_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 256 | 2.209807 | 8632.06 | 190.51 | 1.8% | — | ok |
| nonad_solve_n16_row_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 512 | 1.782971 | 3482.37 | 57.01 | 1.5% | — | ok |
| nonad_solve_n16_row_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 256 | 2.663558 | 10404.52 | 220.62 | 1.1% | — | ok |
| nonad_solve_n16_row_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 16×16 | row_major_contiguous | single | 4 | blas | 512 | 2.339753 | 4569.83 | 80.38 | 1.4% | — | ok |
| nonad_einsum_n64_col_major_contiguous_concrete-fresh | einsum | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 64 | 2.150162 | 33596.28 | 1789.64 | 4.3% | — | ok |
| nonad_einsum_n64_col_major_contiguous_eager-no-ad | einsum | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 128 | 3.032018 | 23687.64 | 511.34 | 4.4% | — | ok |
| nonad_einsum_n64_col_major_contiguous_concrete-shared | einsum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 64 | 2.091960 | 32686.88 | 1630.38 | 4.8% | — | ok |
| nonad_einsum_n64_col_major_contiguous_eager-shared | einsum | eager-shared | eager / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 128 | 2.650928 | 20710.38 | 1871.79 | 9.1% | — | ok |
| nonad_einsum3_n64_col_major_contiguous_concrete-fresh | einsum3 | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 64 | 3.157415 | 49334.61 | 987.17 | 2.7% | — | ok |
| nonad_einsum3_n64_col_major_contiguous_eager-no-ad | einsum3 | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 16 | 3.526272 | 220392.00 | 30375.12 | 9.5% | — | ok |
| nonad_einsum3_n64_col_major_contiguous_concrete-shared | einsum3 | concrete-shared | concrete / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 64 | 2.810659 | 43916.55 | 1611.55 | 5.0% | — | ok |
| nonad_einsum3_n64_col_major_contiguous_eager-shared | einsum3 | eager-shared | eager / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 16 | 3.499316 | 218707.25 | 14719.53 | 7.0% | — | ok |
| nonad_qr_n64_col_major_contiguous_concrete-fresh | qr | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 32 | 2.387355 | 74604.84 | 4407.50 | 4.8% | — | ok |
| nonad_qr_n64_col_major_contiguous_eager-no-ad | qr | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 16 | 4.616543 | 288533.94 | 23919.50 | 5.8% | — | ok |
| nonad_qr_n64_col_major_contiguous_concrete-shared | qr | concrete-shared | concrete / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 32 | 2.271733 | 70991.66 | 3596.17 | 5.3% | — | ok |
| nonad_qr_n64_col_major_contiguous_eager-shared | qr | eager-shared | eager / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 16 | 4.515596 | 282224.75 | 18740.41 | 4.9% | — | ok |
| nonad_svd_n64_col_major_contiguous_concrete-fresh | svd | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 16 | 6.905728 | 431608.00 | 14041.72 | 2.8% | — | ok |
| nonad_svd_n64_col_major_contiguous_eager-no-ad | svd | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 16 | 10.503939 | 656496.19 | 26252.56 | 3.6% | — | ok |
| nonad_svd_n64_col_major_contiguous_concrete-shared | svd | concrete-shared | concrete / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 16 | 6.772621 | 423288.81 | 11240.97 | 3.8% | — | ok |
| nonad_svd_n64_col_major_contiguous_eager-shared | svd | eager-shared | eager / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 16 | 10.651425 | 665714.06 | 37032.84 | 3.4% | — | ok |
| nonad_solve_n64_col_major_contiguous_concrete-fresh | solve | concrete-fresh | concrete / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 64 | 2.866571 | 44790.17 | 2580.53 | 5.8% | — | ok |
| nonad_solve_n64_col_major_contiguous_eager-no-ad | solve | eager-no-ad | eager / allocating / owned / session-per-call | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 64 | 2.769487 | 43273.23 | 3689.51 | 6.0% | — | ok |
| nonad_solve_n64_col_major_contiguous_concrete-shared | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 64 | 2.794197 | 43659.33 | 2235.76 | 5.3% | — | ok |
| nonad_solve_n64_col_major_contiguous_eager-shared | solve | eager-shared | eager / allocating / owned / shared-session | f64 | 64×64 | col_major_contiguous | single | 4 | blas | 64 | 2.542675 | 39729.30 | 3374.51 | 8.7% | — | ok |

## Matched non-AD eager comparisons (#98)

All requested QR/SVD outputs are retained. Ordinary einsum uses the same parsed integer notation and default Auto strategy; planning/cache lookup stays inside both public calls. These are public-call diagnostics, not prepared execution or an isolated AD cost. Fresh pairs include session entry; shared pairs exclude it. Views are constructed untimed and consumed as views on both routes. No output to_tensor conversion is timed. Capture is inactive by construction in each fresh process, with no capture_trace guard; inputs and validation outputs are checked untracked.

| Operation | Shape | Layout | Session | Threads | Direct ns | Eager ns | Eager − direct ns | Eager/direct | Quality |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 1 | 23684.58 | 15244.62 | -8439.96 | 0.644 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | shared | 1 | 21869.11 | 12652.02 | -9217.09 | 0.579 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | per call diagnostic | 4 | 24463.87 | 14949.04 | -9514.83 | 0.611 | observed; no speedup claim |
| einsum | 2×2 | col_major_contiguous | shared | 4 | 22393.42 | 12741.32 | -9652.10 | 0.569 | observed; no speedup claim |
| einsum | 4×4 | col_major_contiguous | per call diagnostic | 1 | 26274.85 | 15282.19 | -10992.66 | 0.582 | observed; no speedup claim |
| einsum | 4×4 | col_major_contiguous | shared | 1 | 22473.59 | 12816.65 | -9656.94 | 0.570 | observed; no speedup claim |
| einsum | 4×4 | col_major_contiguous | per call diagnostic | 4 | 25521.53 | 14871.63 | -10649.90 | 0.583 | observed; no speedup claim |
| einsum | 4×4 | col_major_contiguous | shared | 4 | 22665.15 | 12782.99 | -9882.16 | 0.564 | observed; no speedup claim |
| einsum | 8×8 | col_major_contiguous | per call diagnostic | 1 | 24114.45 | 14759.41 | -9355.04 | 0.612 | observed; no speedup claim |
| einsum | 8×8 | col_major_contiguous | shared | 1 | 22245.52 | 13213.32 | -9032.20 | 0.594 | observed; no speedup claim |
| einsum | 8×8 | col_major_contiguous | per call diagnostic | 4 | 24836.77 | 15123.74 | -9713.03 | 0.609 | observed; no speedup claim |
| einsum | 8×8 | col_major_contiguous | shared | 4 | 22194.20 | 12741.99 | -9452.21 | 0.574 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 1 | 24455.69 | 15682.28 | -8773.41 | 0.641 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | shared | 1 | 22347.76 | 13358.38 | -8989.38 | 0.598 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | per call diagnostic | 4 | 24658.26 | 15669.43 | -8988.83 | 0.635 | observed; no speedup claim |
| einsum | 16×16 | col_major_contiguous | shared | 4 | 22564.43 | 13033.35 | -9531.08 | 0.578 | observed; no speedup claim |
| einsum | 16×16 | row_major_contiguous | per call diagnostic | 1 | 26902.58 | 16462.77 | -10439.81 | 0.612 | observed; no speedup claim |
| einsum | 16×16 | row_major_contiguous | shared | 1 | 24869.34 | 14229.79 | -10639.55 | 0.572 | observed; no speedup claim |
| einsum | 16×16 | row_major_contiguous | per call diagnostic | 4 | 26535.95 | 16214.82 | -10321.13 | 0.611 | observed; no speedup claim |
| einsum | 16×16 | row_major_contiguous | shared | 4 | 24520.62 | 13732.88 | -10787.74 | 0.560 | observed; no speedup claim |
| einsum | 64×64 | col_major_contiguous | per call diagnostic | 1 | 34571.56 | 25176.60 | -9394.96 | 0.728 | observed; no speedup claim |
| einsum | 64×64 | col_major_contiguous | shared | 1 | 33604.31 | 22904.26 | -10700.05 | 0.682 | observed; no speedup claim |
| einsum | 64×64 | col_major_contiguous | per call diagnostic | 4 | 33596.28 | 23687.64 | -9908.64 | 0.705 | observed; no speedup claim |
| einsum | 64×64 | col_major_contiguous | shared | 4 | 32686.88 | 20710.38 | -11976.50 | 0.634 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 1 | 28679.86 | 35100.20 | 6420.34 | 1.224 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | shared | 1 | 26796.52 | 31529.17 | 4732.65 | 1.177 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | per call diagnostic | 4 | 29391.70 | 166792.75 | 137401.05 | 5.675 | observed; no speedup claim |
| einsum3 | 2×2 | col_major_contiguous | shared | 4 | 26808.20 | 157842.25 | 131034.05 | 5.888 | observed; no speedup claim |
| einsum3 | 4×4 | col_major_contiguous | per call diagnostic | 1 | 28874.53 | 35825.75 | 6951.22 | 1.241 | observed; no speedup claim |
| einsum3 | 4×4 | col_major_contiguous | shared | 1 | 26345.74 | 32061.56 | 5715.82 | 1.217 | observed; no speedup claim |
| einsum3 | 4×4 | col_major_contiguous | per call diagnostic | 4 | 29831.24 | 166182.12 | 136350.88 | 5.571 | observed; no speedup claim |
| einsum3 | 4×4 | col_major_contiguous | shared | 4 | 27335.76 | 165097.44 | 137761.68 | 6.040 | observed; no speedup claim |
| einsum3 | 8×8 | col_major_contiguous | per call diagnostic | 1 | 30262.45 | 35528.17 | 5265.72 | 1.174 | observed; no speedup claim |
| einsum3 | 8×8 | col_major_contiguous | shared | 1 | 27876.88 | 32755.73 | 4878.85 | 1.175 | observed; no speedup claim |
| einsum3 | 8×8 | col_major_contiguous | per call diagnostic | 4 | 31174.81 | 164612.75 | 133437.94 | 5.280 | observed; no speedup claim |
| einsum3 | 8×8 | col_major_contiguous | shared | 4 | 28414.03 | 160387.88 | 131973.85 | 5.645 | observed; no speedup claim |
| einsum3 | 16×16 | col_major_contiguous | per call diagnostic | 1 | 29888.33 | 35862.50 | 5974.17 | 1.200 | observed; no speedup claim |
| einsum3 | 16×16 | col_major_contiguous | shared | 1 | 27323.08 | 32387.52 | 5064.44 | 1.185 | observed; no speedup claim |
| einsum3 | 16×16 | col_major_contiguous | per call diagnostic | 4 | 29921.22 | 165599.25 | 135678.03 | 5.535 | observed; no speedup claim |
| einsum3 | 16×16 | col_major_contiguous | shared | 4 | 27644.69 | 162403.38 | 134758.69 | 5.875 | observed; no speedup claim |
| einsum3 | 16×16 | row_major_contiguous | per call diagnostic | 1 | 32182.16 | 35695.14 | 3512.98 | 1.109 | observed; no speedup claim |
| einsum3 | 16×16 | row_major_contiguous | shared | 1 | 29879.61 | 33129.14 | 3249.53 | 1.109 | observed; no speedup claim |
| einsum3 | 16×16 | row_major_contiguous | per call diagnostic | 4 | 32253.20 | 165327.81 | 133074.61 | 5.126 | observed; no speedup claim |
| einsum3 | 16×16 | row_major_contiguous | shared | 4 | 29856.33 | 167566.25 | 137709.92 | 5.612 | observed; no speedup claim |
| einsum3 | 64×64 | col_major_contiguous | per call diagnostic | 1 | 50499.58 | 58607.25 | 8107.67 | 1.161 | observed; no speedup claim |
| einsum3 | 64×64 | col_major_contiguous | shared | 1 | 49095.81 | 56089.64 | 6993.83 | 1.142 | observed; no speedup claim |
| einsum3 | 64×64 | col_major_contiguous | per call diagnostic | 4 | 49334.61 | 220392.00 | 171057.39 | 4.467 | observed; no speedup claim |
| einsum3 | 64×64 | col_major_contiguous | shared | 4 | 43916.55 | 218707.25 | 174790.70 | 4.980 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 1 | 2165.85 | 8366.57 | 6200.72 | 3.863 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | shared | 1 | 1225.60 | 6465.39 | 5239.79 | 5.275 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | per call diagnostic | 4 | 2141.88 | 139512.25 | 137370.37 | 65.135 | observed; no speedup claim |
| qr | 2×2 | col_major_contiguous | shared | 4 | 1269.38 | 131309.50 | 130040.12 | 103.444 | observed; no speedup claim |
| qr | 4×4 | col_major_contiguous | per call diagnostic | 1 | 2368.83 | 9023.02 | 6654.19 | 3.809 | observed; no speedup claim |
| qr | 4×4 | col_major_contiguous | shared | 1 | 1535.76 | 6782.78 | 5247.02 | 4.417 | observed; no speedup claim |
| qr | 4×4 | col_major_contiguous | per call diagnostic | 4 | 2417.47 | 133439.44 | 131021.97 | 55.198 | observed; no speedup claim |
| qr | 4×4 | col_major_contiguous | shared | 4 | 1561.25 | 132909.75 | 131348.50 | 85.130 | observed; no speedup claim |
| qr | 8×8 | col_major_contiguous | per call diagnostic | 1 | 3131.42 | 9442.60 | 6311.18 | 3.015 | observed; no speedup claim |
| qr | 8×8 | col_major_contiguous | shared | 1 | 2243.88 | 7572.49 | 5328.61 | 3.375 | observed; no speedup claim |
| qr | 8×8 | col_major_contiguous | per call diagnostic | 4 | 3160.53 | 136121.38 | 132960.85 | 43.069 | observed; no speedup claim |
| qr | 8×8 | col_major_contiguous | shared | 4 | 2222.73 | 132563.00 | 130340.27 | 59.640 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 1 | 6632.28 | 12001.83 | 5369.55 | 1.810 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | shared | 1 | 5617.72 | 10074.06 | 4456.34 | 1.793 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | per call diagnostic | 4 | 6580.41 | 141490.69 | 134910.28 | 21.502 | observed; no speedup claim |
| qr | 16×16 | col_major_contiguous | shared | 4 | 5558.66 | 136774.81 | 131216.15 | 24.606 | observed; no speedup claim |
| qr | 16×16 | row_major_contiguous | per call diagnostic | 1 | 7330.30 | 12833.49 | 5503.19 | 1.751 | observed; no speedup claim |
| qr | 16×16 | row_major_contiguous | shared | 1 | 7625.78 | 10907.16 | 3281.38 | 1.430 | observed; no speedup claim |
| qr | 16×16 | row_major_contiguous | per call diagnostic | 4 | 129569.81 | 139297.25 | 9727.44 | 1.075 | observed; no speedup claim |
| qr | 16×16 | row_major_contiguous | shared | 4 | 126984.56 | 138636.75 | 11652.19 | 1.092 | observed; no speedup claim |
| qr | 64×64 | col_major_contiguous | per call diagnostic | 1 | 54280.53 | 66435.72 | 12155.19 | 1.224 | observed; no speedup claim |
| qr | 64×64 | col_major_contiguous | shared | 1 | 52089.44 | 64404.75 | 12315.31 | 1.236 | observed; no speedup claim |
| qr | 64×64 | col_major_contiguous | per call diagnostic | 4 | 74604.84 | 288533.94 | 213929.10 | 3.867 | observed; no speedup claim |
| qr | 64×64 | col_major_contiguous | shared | 4 | 70991.66 | 282224.75 | 211233.09 | 3.975 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 1 | 1901.87 | 6878.68 | 4976.81 | 3.617 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | shared | 1 | 1071.23 | 5553.61 | 4482.38 | 5.184 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | per call diagnostic | 4 | 1880.09 | 6814.31 | 4934.22 | 3.624 | observed; no speedup claim |
| solve | 2×2 | col_major_contiguous | shared | 4 | 1076.83 | 5496.58 | 4419.75 | 5.104 | observed; no speedup claim |
| solve | 4×4 | col_major_contiguous | per call diagnostic | 1 | 1998.76 | 7090.95 | 5092.19 | 3.548 | observed; no speedup claim |
| solve | 4×4 | col_major_contiguous | shared | 1 | 1162.49 | 5631.28 | 4468.79 | 4.844 | observed; no speedup claim |
| solve | 4×4 | col_major_contiguous | per call diagnostic | 4 | 1968.22 | 7111.76 | 5143.54 | 3.613 | observed; no speedup claim |
| solve | 4×4 | col_major_contiguous | shared | 4 | 1144.36 | 5599.41 | 4455.05 | 4.893 | observed; no speedup claim |
| solve | 8×8 | col_major_contiguous | per call diagnostic | 1 | 2674.63 | 7790.18 | 5115.55 | 2.913 | observed; no speedup claim |
| solve | 8×8 | col_major_contiguous | shared | 1 | 1816.23 | 6401.97 | 4585.74 | 3.525 | observed; no speedup claim |
| solve | 8×8 | col_major_contiguous | per call diagnostic | 4 | 2672.50 | 7534.16 | 4861.66 | 2.819 | observed; no speedup claim |
| solve | 8×8 | col_major_contiguous | shared | 4 | 1846.12 | 6200.46 | 4354.34 | 3.359 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 1 | 4217.68 | 10050.38 | 5832.70 | 2.383 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | shared | 1 | 4076.32 | 8300.62 | 4224.30 | 2.036 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | per call diagnostic | 4 | 4249.43 | 9913.74 | 5664.31 | 2.333 | observed; no speedup claim |
| solve | 16×16 | col_major_contiguous | shared | 4 | 3220.31 | 8127.67 | 4907.36 | 2.524 | observed; no speedup claim |
| solve | 16×16 | row_major_contiguous | per call diagnostic | 1 | 4543.27 | 10684.68 | 6141.41 | 2.352 | observed; no speedup claim |
| solve | 16×16 | row_major_contiguous | shared | 1 | 3576.94 | 8894.95 | 5318.01 | 2.487 | observed; no speedup claim |
| solve | 16×16 | row_major_contiguous | per call diagnostic | 4 | 4569.83 | 10404.52 | 5834.69 | 2.277 | observed; no speedup claim |
| solve | 16×16 | row_major_contiguous | shared | 4 | 3482.37 | 8632.06 | 5149.69 | 2.479 | observed; no speedup claim |
| solve | 64×64 | col_major_contiguous | per call diagnostic | 1 | 42833.08 | 39952.66 | -2880.42 | 0.933 | observed; no speedup claim |
| solve | 64×64 | col_major_contiguous | shared | 1 | 41117.17 | 37635.16 | -3482.01 | 0.915 | observed; no speedup claim |
| solve | 64×64 | col_major_contiguous | per call diagnostic | 4 | 44790.17 | 43273.23 | -1516.94 | 0.966 | observed; no speedup claim |
| solve | 64×64 | col_major_contiguous | shared | 4 | 43659.33 | 39729.30 | -3930.03 | 0.910 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 1 | 3920.03 | 11796.87 | 7876.84 | 3.009 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | shared | 1 | 2577.57 | 9769.17 | 7191.60 | 3.790 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | per call diagnostic | 4 | 4117.25 | 142307.50 | 138190.25 | 34.564 | observed; no speedup claim |
| svd | 2×2 | col_major_contiguous | shared | 4 | 2736.72 | 137121.50 | 134384.78 | 50.104 | observed; no speedup claim |
| svd | 4×4 | col_major_contiguous | per call diagnostic | 1 | 7170.66 | 15822.69 | 8652.03 | 2.207 | observed; no speedup claim |
| svd | 4×4 | col_major_contiguous | shared | 1 | 5652.90 | 13281.11 | 7628.21 | 2.349 | observed; no speedup claim |
| svd | 4×4 | col_major_contiguous | per call diagnostic | 4 | 7088.27 | 144932.69 | 137844.42 | 20.447 | observed; no speedup claim |
| svd | 4×4 | col_major_contiguous | shared | 4 | 5538.48 | 142640.62 | 137102.14 | 25.754 | observed; no speedup claim |
| svd | 8×8 | col_major_contiguous | per call diagnostic | 1 | 15499.48 | 24527.86 | 9028.38 | 1.582 | observed; no speedup claim |
| svd | 8×8 | col_major_contiguous | shared | 1 | 13811.07 | 21573.30 | 7762.23 | 1.562 | observed; no speedup claim |
| svd | 8×8 | col_major_contiguous | per call diagnostic | 4 | 15270.95 | 155759.94 | 140488.99 | 10.200 | observed; no speedup claim |
| svd | 8×8 | col_major_contiguous | shared | 4 | 13857.91 | 150410.94 | 136553.03 | 10.854 | observed; no speedup claim |
| svd | 16×16 | col_major_contiguous | per call diagnostic | 1 | 35610.78 | 45009.11 | 9398.33 | 1.264 | observed; no speedup claim |
| svd | 16×16 | col_major_contiguous | shared | 1 | 33787.81 | 42771.38 | 8983.57 | 1.266 | observed; no speedup claim |
| svd | 16×16 | col_major_contiguous | per call diagnostic | 4 | 35190.84 | 177723.88 | 142533.04 | 5.050 | observed; no speedup claim |
| svd | 16×16 | col_major_contiguous | shared | 4 | 33886.64 | 173284.81 | 139398.17 | 5.114 | observed; no speedup claim |
| svd | 16×16 | row_major_contiguous | per call diagnostic | 1 | 38512.92 | 47295.62 | 8782.70 | 1.228 | observed; no speedup claim |
| svd | 16×16 | row_major_contiguous | shared | 1 | 36545.98 | 43819.50 | 7273.52 | 1.199 | observed; no speedup claim |
| svd | 16×16 | row_major_contiguous | per call diagnostic | 4 | 169655.56 | 181965.44 | 12309.88 | 1.073 | observed; no speedup claim |
| svd | 16×16 | row_major_contiguous | shared | 4 | 167422.00 | 175355.12 | 7933.12 | 1.047 | observed; no speedup claim |
| svd | 64×64 | col_major_contiguous | per call diagnostic | 1 | 407940.38 | 424511.19 | 16570.81 | 1.041 | observed; no speedup claim |
| svd | 64×64 | col_major_contiguous | shared | 1 | 404089.19 | 417997.81 | 13908.62 | 1.034 | observed; no speedup claim |
| svd | 64×64 | col_major_contiguous | per call diagnostic | 4 | 431608.00 | 656496.19 | 224888.19 | 1.521 | observed; no speedup claim |
| svd | 64×64 | col_major_contiguous | shared | 4 | 423288.81 | 665714.06 | 242425.25 | 1.573 | observed; no speedup claim |

## Separate allocation diagnostics

Instrumented executable is built separately after latency collection. Calling-thread Rust allocator requests only; excludes foreign BLAS and worker-thread allocations. Realloc counts its requested new size, not net retained bytes. Output destruction and instrumentation setup are excluded. Identical result packaging is included. No allocation-instrumented run contributes latency samples.

| Corresponding case ID | Threads | Median allocation requests/op | Median requested bytes/op | Status |
|---|---:|---:|---:|---|
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | 1 | 221 | 18452 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | 1 | 122 | 14885 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | 1 | 219 | 18188 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-shared | 1 | 120 | 14621 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | 1 | 314 | 26989 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | 1 | 276 | 30882 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | 1 | 312 | 26725 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | 1 | 274 | 30618 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | 1 | 9 | 1928 | passed |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | 1 | 48 | 12483 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-shared | 1 | 7 | 1664 | passed |
| nonad_qr_n2_col_major_contiguous_eager-shared | 1 | 46 | 12219 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | 1 | 12 | 3120 | passed |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | 1 | 56 | 14907 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-shared | 1 | 9 | 2664 | passed |
| nonad_svd_n2_col_major_contiguous_eager-shared | 1 | 53 | 14451 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | 1 | 7 | 488 | passed |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | 1 | 42 | 18011 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-shared | 1 | 6 | 416 | passed |
| nonad_solve_n2_col_major_contiguous_eager-shared | 1 | 41 | 17939 | passed |
| nonad_einsum_n4_col_major_contiguous_eager-shared | 1 | 120 | 14749 | passed |
| nonad_einsum_n4_col_major_contiguous_concrete-shared | 1 | 219 | 18316 | passed |
| nonad_einsum_n4_col_major_contiguous_eager-no-ad | 1 | 122 | 15013 | passed |
| nonad_einsum_n4_col_major_contiguous_concrete-fresh | 1 | 221 | 18580 | passed |
| nonad_einsum3_n4_col_major_contiguous_eager-shared | 1 | 274 | 30874 | passed |
| nonad_einsum3_n4_col_major_contiguous_concrete-shared | 1 | 312 | 26725 | passed |
| nonad_einsum3_n4_col_major_contiguous_eager-no-ad | 1 | 276 | 31138 | passed |
| nonad_einsum3_n4_col_major_contiguous_concrete-fresh | 1 | 314 | 26989 | passed |
| nonad_qr_n4_col_major_contiguous_eager-shared | 1 | 46 | 12411 | passed |
| nonad_qr_n4_col_major_contiguous_concrete-shared | 1 | 7 | 1856 | passed |
| nonad_qr_n4_col_major_contiguous_eager-no-ad | 1 | 48 | 12675 | passed |
| nonad_qr_n4_col_major_contiguous_concrete-fresh | 1 | 9 | 2120 | passed |
| nonad_svd_n4_col_major_contiguous_eager-shared | 1 | 53 | 15043 | passed |
| nonad_svd_n4_col_major_contiguous_concrete-shared | 1 | 9 | 3256 | passed |
| nonad_svd_n4_col_major_contiguous_eager-no-ad | 1 | 56 | 15499 | passed |
| nonad_svd_n4_col_major_contiguous_concrete-fresh | 1 | 12 | 3712 | passed |
| nonad_solve_n4_col_major_contiguous_eager-shared | 1 | 41 | 18139 | passed |
| nonad_solve_n4_col_major_contiguous_concrete-shared | 1 | 6 | 712 | passed |
| nonad_solve_n4_col_major_contiguous_eager-no-ad | 1 | 42 | 18211 | passed |
| nonad_solve_n4_col_major_contiguous_concrete-fresh | 1 | 7 | 784 | passed |
| nonad_einsum_n8_col_major_contiguous_concrete-fresh | 1 | 221 | 18836 | passed |
| nonad_einsum_n8_col_major_contiguous_eager-no-ad | 1 | 122 | 15269 | passed |
| nonad_einsum_n8_col_major_contiguous_concrete-shared | 1 | 219 | 18572 | passed |
| nonad_einsum_n8_col_major_contiguous_eager-shared | 1 | 120 | 15005 | passed |
| nonad_einsum3_n8_col_major_contiguous_concrete-fresh | 1 | 314 | 26989 | passed |
| nonad_einsum3_n8_col_major_contiguous_eager-no-ad | 1 | 275 | 31642 | passed |
| nonad_einsum3_n8_col_major_contiguous_concrete-shared | 1 | 312 | 26725 | passed |
| nonad_einsum3_n8_col_major_contiguous_eager-shared | 1 | 273 | 31378 | passed |
| nonad_qr_n8_col_major_contiguous_concrete-fresh | 1 | 9 | 2888 | passed |
| nonad_qr_n8_col_major_contiguous_eager-no-ad | 1 | 48 | 13443 | passed |
| nonad_qr_n8_col_major_contiguous_concrete-shared | 1 | 7 | 2624 | passed |
| nonad_qr_n8_col_major_contiguous_eager-shared | 1 | 46 | 13179 | passed |
| nonad_svd_n8_col_major_contiguous_concrete-fresh | 1 | 12 | 5856 | passed |
| nonad_svd_n8_col_major_contiguous_eager-no-ad | 1 | 56 | 17643 | passed |
| nonad_svd_n8_col_major_contiguous_concrete-shared | 1 | 9 | 5400 | passed |
| nonad_svd_n8_col_major_contiguous_eager-shared | 1 | 53 | 17187 | passed |
| nonad_solve_n8_col_major_contiguous_concrete-fresh | 1 | 7 | 1952 | passed |
| nonad_solve_n8_col_major_contiguous_eager-no-ad | 1 | 42 | 18995 | passed |
| nonad_solve_n8_col_major_contiguous_concrete-shared | 1 | 6 | 1880 | passed |
| nonad_solve_n8_col_major_contiguous_eager-shared | 1 | 41 | 18923 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-shared | 1 | 120 | 15517 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | 1 | 219 | 19084 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | 1 | 122 | 15781 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | 1 | 221 | 19348 | passed |
| nonad_einsum_n16_row_major_contiguous_eager-shared | 1 | 130 | 17837 | passed |
| nonad_einsum_n16_row_major_contiguous_concrete-shared | 1 | 219 | 20380 | passed |
| nonad_einsum_n16_row_major_contiguous_eager-no-ad | 1 | 132 | 18101 | passed |
| nonad_einsum_n16_row_major_contiguous_concrete-fresh | 1 | 221 | 20644 | passed |
| nonad_einsum3_n16_col_major_contiguous_eager-shared | 1 | 268 | 32326 | passed |
| nonad_einsum3_n16_col_major_contiguous_concrete-shared | 1 | 312 | 26725 | passed |
| nonad_einsum3_n16_col_major_contiguous_eager-no-ad | 1 | 270 | 32590 | passed |
| nonad_einsum3_n16_col_major_contiguous_concrete-fresh | 1 | 314 | 26989 | passed |
| nonad_einsum3_n16_row_major_contiguous_eager-shared | 1 | 283 | 35806 | passed |
| nonad_einsum3_n16_row_major_contiguous_concrete-shared | 1 | 311 | 26701 | passed |
| nonad_einsum3_n16_row_major_contiguous_eager-no-ad | 1 | 285 | 36070 | passed |
| nonad_einsum3_n16_row_major_contiguous_concrete-fresh | 1 | 313 | 26965 | passed |
| nonad_qr_n16_col_major_contiguous_eager-shared | 1 | 46 | 16251 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-shared | 1 | 7 | 5696 | passed |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | 1 | 48 | 16515 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | 1 | 9 | 5960 | passed |
| nonad_qr_n16_row_major_contiguous_eager-shared | 1 | 56 | 17507 | passed |
| nonad_qr_n16_row_major_contiguous_concrete-shared | 1 | 18 | 6608 | passed |
| nonad_qr_n16_row_major_contiguous_eager-no-ad | 1 | 58 | 17771 | passed |
| nonad_qr_n16_row_major_contiguous_concrete-fresh | 1 | 20 | 6872 | passed |
| nonad_svd_n16_col_major_contiguous_eager-shared | 1 | 53 | 25315 | passed |
| nonad_svd_n16_col_major_contiguous_concrete-shared | 1 | 9 | 13528 | passed |
| nonad_svd_n16_col_major_contiguous_eager-no-ad | 1 | 56 | 25771 | passed |
| nonad_svd_n16_col_major_contiguous_concrete-fresh | 1 | 12 | 13984 | passed |
| nonad_svd_n16_row_major_contiguous_eager-shared | 1 | 63 | 26571 | passed |
| nonad_svd_n16_row_major_contiguous_concrete-shared | 1 | 20 | 14440 | passed |
| nonad_svd_n16_row_major_contiguous_eager-no-ad | 1 | 66 | 27027 | passed |
| nonad_svd_n16_row_major_contiguous_concrete-fresh | 1 | 23 | 14896 | passed |
| nonad_solve_n16_col_major_contiguous_eager-shared | 1 | 41 | 22027 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-shared | 1 | 6 | 6520 | passed |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | 1 | 42 | 22099 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | 1 | 7 | 6592 | passed |
| nonad_solve_n16_row_major_contiguous_eager-shared | 1 | 51 | 24315 | passed |
| nonad_solve_n16_row_major_contiguous_concrete-shared | 1 | 6 | 5784 | passed |
| nonad_solve_n16_row_major_contiguous_eager-no-ad | 1 | 52 | 24387 | passed |
| nonad_solve_n16_row_major_contiguous_concrete-fresh | 1 | 7 | 5856 | passed |
| nonad_einsum_n64_col_major_contiguous_concrete-fresh | 1 | 221 | 22420 | passed |
| nonad_einsum_n64_col_major_contiguous_eager-no-ad | 1 | 122 | 18853 | passed |
| nonad_einsum_n64_col_major_contiguous_concrete-shared | 1 | 219 | 22156 | passed |
| nonad_einsum_n64_col_major_contiguous_eager-shared | 1 | 120 | 18589 | passed |
| nonad_einsum3_n64_col_major_contiguous_concrete-fresh | 1 | 336 | 27357 | passed |
| nonad_einsum3_n64_col_major_contiguous_eager-no-ad | 1 | 270 | 38734 | passed |
| nonad_einsum3_n64_col_major_contiguous_concrete-shared | 1 | 334 | 27093 | passed |
| nonad_einsum3_n64_col_major_contiguous_eager-shared | 1 | 268 | 38470 | passed |
| nonad_qr_n64_col_major_contiguous_concrete-fresh | 1 | 9 | 67400 | passed |
| nonad_qr_n64_col_major_contiguous_eager-no-ad | 1 | 48 | 77955 | passed |
| nonad_qr_n64_col_major_contiguous_concrete-shared | 1 | 7 | 67136 | passed |
| nonad_qr_n64_col_major_contiguous_eager-shared | 1 | 46 | 77691 | passed |
| nonad_svd_n64_col_major_contiguous_concrete-fresh | 1 | 12 | 170272 | passed |
| nonad_svd_n64_col_major_contiguous_eager-no-ad | 1 | 56 | 182059 | passed |
| nonad_svd_n64_col_major_contiguous_concrete-shared | 1 | 9 | 169816 | passed |
| nonad_svd_n64_col_major_contiguous_eager-shared | 1 | 53 | 181603 | passed |
| nonad_solve_n64_col_major_contiguous_concrete-fresh | 1 | 7 | 98944 | passed |
| nonad_solve_n64_col_major_contiguous_eager-no-ad | 1 | 42 | 83731 | passed |
| nonad_solve_n64_col_major_contiguous_concrete-shared | 1 | 6 | 98872 | passed |
| nonad_solve_n64_col_major_contiguous_eager-shared | 1 | 41 | 83659 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-fresh | 4 | 221 | 18452 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-no-ad | 4 | 122 | 14885 | passed |
| nonad_einsum_n2_col_major_contiguous_concrete-shared | 4 | 219 | 18188 | passed |
| nonad_einsum_n2_col_major_contiguous_eager-shared | 4 | 120 | 14621 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-fresh | 4 | 314 | 26989 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-no-ad | 4 | 254 | 30442 | passed |
| nonad_einsum3_n2_col_major_contiguous_concrete-shared | 4 | 312 | 26725 | passed |
| nonad_einsum3_n2_col_major_contiguous_eager-shared | 4 | 252 | 30178 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-fresh | 4 | 9 | 1928 | passed |
| nonad_qr_n2_col_major_contiguous_eager-no-ad | 4 | 42 | 12147 | passed |
| nonad_qr_n2_col_major_contiguous_concrete-shared | 4 | 7 | 1664 | passed |
| nonad_qr_n2_col_major_contiguous_eager-shared | 4 | 41 | 12075 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-fresh | 4 | 12 | 3120 | passed |
| nonad_svd_n2_col_major_contiguous_eager-no-ad | 4 | 50 | 14571 | passed |
| nonad_svd_n2_col_major_contiguous_concrete-shared | 4 | 9 | 2664 | passed |
| nonad_svd_n2_col_major_contiguous_eager-shared | 4 | 48 | 14307 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-fresh | 4 | 7 | 488 | passed |
| nonad_solve_n2_col_major_contiguous_eager-no-ad | 4 | 42 | 18011 | passed |
| nonad_solve_n2_col_major_contiguous_concrete-shared | 4 | 6 | 416 | passed |
| nonad_solve_n2_col_major_contiguous_eager-shared | 4 | 41 | 17939 | passed |
| nonad_einsum_n4_col_major_contiguous_eager-shared | 4 | 120 | 14749 | passed |
| nonad_einsum_n4_col_major_contiguous_concrete-shared | 4 | 219 | 18316 | passed |
| nonad_einsum_n4_col_major_contiguous_eager-no-ad | 4 | 122 | 15013 | passed |
| nonad_einsum_n4_col_major_contiguous_concrete-fresh | 4 | 221 | 18580 | passed |
| nonad_einsum3_n4_col_major_contiguous_eager-shared | 4 | 252 | 30434 | passed |
| nonad_einsum3_n4_col_major_contiguous_concrete-shared | 4 | 312 | 26725 | passed |
| nonad_einsum3_n4_col_major_contiguous_eager-no-ad | 4 | 254 | 30698 | passed |
| nonad_einsum3_n4_col_major_contiguous_concrete-fresh | 4 | 314 | 26989 | passed |
| nonad_qr_n4_col_major_contiguous_eager-shared | 4 | 41 | 12267 | passed |
| nonad_qr_n4_col_major_contiguous_concrete-shared | 4 | 7 | 1856 | passed |
| nonad_qr_n4_col_major_contiguous_eager-no-ad | 4 | 42 | 12339 | passed |
| nonad_qr_n4_col_major_contiguous_concrete-fresh | 4 | 9 | 2120 | passed |
| nonad_svd_n4_col_major_contiguous_eager-shared | 4 | 48 | 14899 | passed |
| nonad_svd_n4_col_major_contiguous_concrete-shared | 4 | 9 | 3256 | passed |
| nonad_svd_n4_col_major_contiguous_eager-no-ad | 4 | 50 | 15163 | passed |
| nonad_svd_n4_col_major_contiguous_concrete-fresh | 4 | 12 | 3712 | passed |
| nonad_solve_n4_col_major_contiguous_eager-shared | 4 | 41 | 18139 | passed |
| nonad_solve_n4_col_major_contiguous_concrete-shared | 4 | 6 | 712 | passed |
| nonad_solve_n4_col_major_contiguous_eager-no-ad | 4 | 42 | 18211 | passed |
| nonad_solve_n4_col_major_contiguous_concrete-fresh | 4 | 7 | 784 | passed |
| nonad_einsum_n8_col_major_contiguous_concrete-fresh | 4 | 221 | 18836 | passed |
| nonad_einsum_n8_col_major_contiguous_eager-no-ad | 4 | 122 | 15269 | passed |
| nonad_einsum_n8_col_major_contiguous_concrete-shared | 4 | 219 | 18572 | passed |
| nonad_einsum_n8_col_major_contiguous_eager-shared | 4 | 120 | 15005 | passed |
| nonad_einsum3_n8_col_major_contiguous_concrete-fresh | 4 | 314 | 26989 | passed |
| nonad_einsum3_n8_col_major_contiguous_eager-no-ad | 4 | 254 | 31210 | passed |
| nonad_einsum3_n8_col_major_contiguous_concrete-shared | 4 | 312 | 26725 | passed |
| nonad_einsum3_n8_col_major_contiguous_eager-shared | 4 | 252 | 30946 | passed |
| nonad_qr_n8_col_major_contiguous_concrete-fresh | 4 | 9 | 2888 | passed |
| nonad_qr_n8_col_major_contiguous_eager-no-ad | 4 | 42 | 13107 | passed |
| nonad_qr_n8_col_major_contiguous_concrete-shared | 4 | 7 | 2624 | passed |
| nonad_qr_n8_col_major_contiguous_eager-shared | 4 | 41 | 13035 | passed |
| nonad_svd_n8_col_major_contiguous_concrete-fresh | 4 | 12 | 5856 | passed |
| nonad_svd_n8_col_major_contiguous_eager-no-ad | 4 | 50 | 17307 | passed |
| nonad_svd_n8_col_major_contiguous_concrete-shared | 4 | 9 | 5400 | passed |
| nonad_svd_n8_col_major_contiguous_eager-shared | 4 | 48 | 17043 | passed |
| nonad_solve_n8_col_major_contiguous_concrete-fresh | 4 | 7 | 1952 | passed |
| nonad_solve_n8_col_major_contiguous_eager-no-ad | 4 | 42 | 18995 | passed |
| nonad_solve_n8_col_major_contiguous_concrete-shared | 4 | 6 | 1880 | passed |
| nonad_solve_n8_col_major_contiguous_eager-shared | 4 | 41 | 18923 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-shared | 4 | 120 | 15517 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-shared | 4 | 219 | 19084 | passed |
| nonad_einsum_n16_col_major_contiguous_eager-no-ad | 4 | 122 | 15781 | passed |
| nonad_einsum_n16_col_major_contiguous_concrete-fresh | 4 | 221 | 19348 | passed |
| nonad_einsum_n16_row_major_contiguous_eager-shared | 4 | 130 | 17837 | passed |
| nonad_einsum_n16_row_major_contiguous_concrete-shared | 4 | 219 | 20380 | passed |
| nonad_einsum_n16_row_major_contiguous_eager-no-ad | 4 | 132 | 18101 | passed |
| nonad_einsum_n16_row_major_contiguous_concrete-fresh | 4 | 221 | 20644 | passed |
| nonad_einsum3_n16_col_major_contiguous_eager-shared | 4 | 252 | 31970 | passed |
| nonad_einsum3_n16_col_major_contiguous_concrete-shared | 4 | 312 | 26725 | passed |
| nonad_einsum3_n16_col_major_contiguous_eager-no-ad | 4 | 254 | 32234 | passed |
| nonad_einsum3_n16_col_major_contiguous_concrete-fresh | 4 | 314 | 26989 | passed |
| nonad_einsum3_n16_row_major_contiguous_eager-shared | 4 | 267 | 35450 | passed |
| nonad_einsum3_n16_row_major_contiguous_concrete-shared | 4 | 311 | 26701 | passed |
| nonad_einsum3_n16_row_major_contiguous_eager-no-ad | 4 | 269 | 35714 | passed |
| nonad_einsum3_n16_row_major_contiguous_concrete-fresh | 4 | 313 | 26965 | passed |
| nonad_qr_n16_col_major_contiguous_eager-shared | 4 | 41 | 16107 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-shared | 4 | 7 | 5696 | passed |
| nonad_qr_n16_col_major_contiguous_eager-no-ad | 4 | 42 | 16179 | passed |
| nonad_qr_n16_col_major_contiguous_concrete-fresh | 4 | 9 | 5960 | passed |
| nonad_qr_n16_row_major_contiguous_eager-shared | 4 | 46 | 17251 | passed |
| nonad_qr_n16_row_major_contiguous_concrete-shared | 4 | 8 | 6352 | passed |
| nonad_qr_n16_row_major_contiguous_eager-no-ad | 4 | 47 | 17323 | passed |
| nonad_qr_n16_row_major_contiguous_concrete-fresh | 4 | 9 | 6424 | passed |
| nonad_svd_n16_col_major_contiguous_eager-shared | 4 | 48 | 25171 | passed |
| nonad_svd_n16_col_major_contiguous_concrete-shared | 4 | 9 | 13528 | passed |
| nonad_svd_n16_col_major_contiguous_eager-no-ad | 4 | 50 | 25435 | passed |
| nonad_svd_n16_col_major_contiguous_concrete-fresh | 4 | 12 | 13984 | passed |
| nonad_svd_n16_row_major_contiguous_eager-shared | 4 | 53 | 26315 | passed |
| nonad_svd_n16_row_major_contiguous_concrete-shared | 4 | 10 | 14184 | passed |
| nonad_svd_n16_row_major_contiguous_eager-no-ad | 4 | 55 | 26579 | passed |
| nonad_svd_n16_row_major_contiguous_concrete-fresh | 4 | 12 | 14448 | passed |
| nonad_solve_n16_col_major_contiguous_eager-shared | 4 | 41 | 22027 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-shared | 4 | 6 | 6520 | passed |
| nonad_solve_n16_col_major_contiguous_eager-no-ad | 4 | 42 | 22099 | passed |
| nonad_solve_n16_col_major_contiguous_concrete-fresh | 4 | 7 | 6592 | passed |
| nonad_solve_n16_row_major_contiguous_eager-shared | 4 | 51 | 24315 | passed |
| nonad_solve_n16_row_major_contiguous_concrete-shared | 4 | 6 | 5784 | passed |
| nonad_solve_n16_row_major_contiguous_eager-no-ad | 4 | 52 | 24387 | passed |
| nonad_solve_n16_row_major_contiguous_concrete-fresh | 4 | 7 | 5856 | passed |
| nonad_einsum_n64_col_major_contiguous_concrete-fresh | 4 | 221 | 22420 | passed |
| nonad_einsum_n64_col_major_contiguous_eager-no-ad | 4 | 122 | 18853 | passed |
| nonad_einsum_n64_col_major_contiguous_concrete-shared | 4 | 219 | 22156 | passed |
| nonad_einsum_n64_col_major_contiguous_eager-shared | 4 | 120 | 18589 | passed |
| nonad_einsum3_n64_col_major_contiguous_concrete-fresh | 4 | 336 | 27357 | passed |
| nonad_einsum3_n64_col_major_contiguous_eager-no-ad | 4 | 254 | 38378 | passed |
| nonad_einsum3_n64_col_major_contiguous_concrete-shared | 4 | 334 | 27093 | passed |
| nonad_einsum3_n64_col_major_contiguous_eager-shared | 4 | 252 | 38114 | passed |
| nonad_qr_n64_col_major_contiguous_concrete-fresh | 4 | 9 | 67400 | passed |
| nonad_qr_n64_col_major_contiguous_eager-no-ad | 4 | 42 | 77619 | passed |
| nonad_qr_n64_col_major_contiguous_concrete-shared | 4 | 7 | 67136 | passed |
| nonad_qr_n64_col_major_contiguous_eager-shared | 4 | 41 | 77547 | passed |
| nonad_svd_n64_col_major_contiguous_concrete-fresh | 4 | 12 | 170272 | passed |
| nonad_svd_n64_col_major_contiguous_eager-no-ad | 4 | 50 | 181723 | passed |
| nonad_svd_n64_col_major_contiguous_concrete-shared | 4 | 9 | 169816 | passed |
| nonad_svd_n64_col_major_contiguous_eager-shared | 4 | 48 | 181459 | passed |
| nonad_solve_n64_col_major_contiguous_concrete-fresh | 4 | 7 | 98944 | passed |
| nonad_solve_n64_col_major_contiguous_eager-no-ad | 4 | 42 | 83731 | passed |
| nonad_solve_n64_col_major_contiguous_concrete-shared | 4 | 6 | 98872 | passed |
| nonad_solve_n64_col_major_contiguous_eager-shared | 4 | 41 | 83659 | passed |

## Explicit setup diagnostics (not operation comparisons)

| Case | Median ns | IQR ns | Status |
|---|---:|---:|---|
| add_f64_concrete_fresh | 1887.94 | 31.32 | ok |
| add_f64_concrete-fresh_size4_dependent10 | 18761.16 | 97.27 | ok |
| add_f64_concrete-fresh_size16_single | 1866.08 | 36.93 | ok |
| add_f64_concrete-fresh_size16_dependent10 | 18753.78 | 85.27 | ok |
| add_f64_concrete-fresh_size256_single | 1953.91 | 18.48 | ok |
| add_f64_concrete-fresh_size256_dependent10 | 19553.52 | 112.71 | ok |
| add_f64_eager-no-ad_size4_single | 3987.58 | 72.38 | ok |
| add_f64_eager-no-ad_size4_dependent10 | 37283.31 | 318.35 | ok |
| add_f64_eager_no_ad | 4145.30 | 98.65 | ok |
| add_f64_eager-no-ad_size16_dependent10 | 37466.93 | 159.67 | ok |
| add_f64_eager-no-ad_size256_single | 4258.36 | 114.08 | ok |
| add_f64_eager-no-ad_size256_dependent10 | 38846.70 | 274.36 | ok |
| add_f64_eager_active_ad | 9556.23 | 180.73 | ok |
| add_f64_eager-ad_size4_dependent10 | 96163.42 | 1136.05 | ok |
| add_f64_eager-ad_size16_single | 10048.60 | 231.90 | ok |
| add_f64_eager-ad_size16_dependent10 | 94058.84 | 929.02 | ok |
| add_f64_eager-ad_size256_single | 10285.65 | 261.52 | ok |
| add_f64_eager-ad_size256_dependent10 | 99564.54 | 1005.30 | ok |
| einsum_f64_concrete-fresh_n2_single | 26441.64 | 311.54 | ok |
| einsum_f64_concrete-fresh_n4_single | 26966.58 | 287.44 | ok |
| einsum_f64_concrete-fresh_n16_single | 27033.32 | 370.11 | ok |
| einsum_f64_concrete-shared_n2_single | 24006.89 | 209.05 | ok |
| einsum_f64_concrete-shared_n4_single | 24296.89 | 247.30 | ok |
| einsum_f64_concrete-shared_n16_single | 24222.97 | 225.55 | ok |
| einsum_f64_eager-no-ad_n2_single | 14490.32 | 290.32 | ok |
| einsum_f64_eager-no-ad_n4_single | 15334.52 | 300.13 | ok |
| einsum_f64_eager-no-ad_n16_single | 15311.58 | 337.05 | ok |
| einsum_f64_eager-ad_n2_single | 24614.59 | 355.36 | ok |
| einsum_f64_eager-ad_n4_single | 24667.56 | 439.38 | ok |
| einsum_f64_eager-ad_n16_single | 24834.73 | 387.17 | ok |
| einsum_f64_prepared-setup_n2_single | 6237.59 | 41.78 | ok |
| einsum_f64_prepared-setup_n4_single | 6189.11 | 37.59 | ok |
| einsum_f64_prepared-setup_n16_single | 6229.71 | 69.70 | ok |
| einsum_f64_borrowed-fresh_n2_col_major_contiguous_single | 27022.70 | 521.82 | ok |
| einsum_f64_borrowed-fresh_n2_row_major_contiguous_single | 26679.43 | 496.94 | ok |
| einsum_f64_borrowed-fresh_n2_strided_single | 29693.90 | 316.42 | ok |
| einsum_f64_borrowed-shared_n2_col_major_contiguous_single | 24885.83 | 481.62 | ok |
| einsum_f64_borrowed-shared_n2_row_major_contiguous_single | 23693.40 | 421.81 | ok |
| einsum_f64_borrowed-shared_n2_strided_single | 27915.20 | 285.61 | ok |
| einsum_f64_borrowed-fresh_n4_col_major_contiguous_single | 26216.21 | 214.11 | ok |
| einsum_f64_borrowed-fresh_n4_row_major_contiguous_single | 26502.82 | 277.47 | ok |
| einsum_f64_borrowed-fresh_n4_strided_single | 29757.32 | 544.18 | ok |
| einsum_f64_borrowed-shared_n4_col_major_contiguous_single | 24128.09 | 282.60 | ok |
| einsum_f64_borrowed-shared_n4_row_major_contiguous_single | 24060.91 | 392.10 | ok |
| einsum_f64_borrowed-shared_n4_strided_single | 27341.33 | 354.39 | ok |
| einsum_f64_borrowed-fresh_n16_col_major_contiguous_single | 27477.24 | 318.91 | ok |
| einsum_f64_borrowed-fresh_n16_row_major_contiguous_single | 27546.28 | 424.43 | ok |
| einsum_f64_borrowed-fresh_n16_strided_single | 30515.38 | 354.22 | ok |
| einsum_f64_borrowed-shared_n16_col_major_contiguous_single | 24235.33 | 210.94 | ok |
| einsum_f64_borrowed-shared_n16_row_major_contiguous_single | 25023.19 | 929.81 | ok |
| einsum_f64_borrowed-shared_n16_strided_single | 28673.75 | 270.82 | ok |
| einsum_f64_compiled-repeat_n2_single | 23385.17 | 429.43 | ok |
| einsum_f64_compiled-repeat_n4_single | 23340.85 | 229.41 | ok |
| einsum_f64_compiled-repeat_n16_single | 24150.30 | 653.59 | ok |
| einsum_c64_concrete-fresh_n2_single | 28559.05 | 359.24 | ok |
| einsum_c64_concrete-shared_n2_single | 26974.01 | 572.56 | ok |
| einsum_c64_concrete-fresh_n4_single | 30232.81 | 585.39 | ok |
| einsum_c64_concrete-shared_n4_single | 26580.03 | 321.02 | ok |
| einsum_c64_concrete-fresh_n16_single | 31007.92 | 696.00 | ok |
| einsum_c64_concrete-shared_n16_single | 28690.22 | 459.09 | ok |
| einsum_c64_prepared-setup_n2_single | 6242.27 | 17.78 | ok |
| einsum_c64_prepared-setup_n4_single | 6200.02 | 37.20 | ok |
| einsum_c64_prepared-setup_n16_single | 6192.45 | 25.90 | ok |
| einsum_c64_eager-no-ad_n2_single | 17745.43 | 347.78 | ok |
| einsum_c64_compiled-repeat_n2_single | 26800.52 | 467.75 | ok |
| einsum_c64_eager-no-ad_n4_single | 17648.33 | 587.06 | ok |
| einsum_c64_compiled-repeat_n4_single | 26663.40 | 494.73 | ok |
| einsum_c64_eager-no-ad_n16_single | 19153.48 | 264.21 | ok |
| einsum_c64_compiled-repeat_n16_single | 27764.63 | 367.50 | ok |
| einsum_c64_eager-ad_n2_single | 27241.03 | 846.54 | ok |
| einsum_c64_eager-ad_n4_single | 27498.74 | 382.17 | ok |
| einsum_c64_eager-ad_n16_single | 29099.86 | 610.09 | ok |
| einsum_f64_compiled-setup_n2_single | 5383.79 | 42.33 | ok |
| einsum_f64_compiled-setup_n4_single | 5379.09 | 36.01 | ok |
| einsum_f64_compiled-setup_n16_single | 5415.28 | 50.97 | ok |
| einsum_c64_compiled-setup_n2_single | 5399.03 | 52.05 | ok |
| einsum_c64_compiled-setup_n4_single | 5363.45 | 59.03 | ok |
| einsum_c64_compiled-setup_n16_single | 5343.54 | 38.70 | ok |
| solve_f64_concrete-fresh_n2_single | 1908.25 | 23.36 | ok |
| solve_f64_concrete-fresh_n4_single | 2087.21 | 27.82 | ok |
| solve_f64_concrete-fresh_n16_single | 5081.55 | 46.58 | ok |
| einsum_f64_borrowed-fresh_n2_broadcast_single | 29539.73 | 390.52 | ok |
| einsum_f64_borrowed-shared_n2_broadcast_single | 26779.47 | 322.43 | ok |
| einsum_f64_borrowed-fresh_n4_broadcast_single | 30140.55 | 375.15 | ok |
| einsum_f64_borrowed-shared_n4_broadcast_single | 26723.36 | 245.36 | ok |
| einsum_f64_borrowed-fresh_n16_broadcast_single | 30483.26 | 309.99 | ok |
| einsum_f64_borrowed-shared_n16_broadcast_single | 28734.85 | 284.16 | ok |
| einsum_f64_concrete-fresh_n8_single | 27617.99 | 543.69 | ok |
| einsum_f64_concrete-shared_n8_single | 24782.27 | 555.44 | ok |
| einsum_f64_prepared-setup_n8_single | 6206.43 | 36.90 | ok |
| einsum_f64_concrete-fresh_n32_single | 28518.06 | 549.32 | ok |
| einsum_f64_concrete-shared_n32_single | 25391.35 | 382.11 | ok |
| einsum_f64_prepared-setup_n32_single | 6181.54 | 35.11 | ok |
| einsum_c64_concrete-fresh_n8_single | 30122.34 | 574.58 | ok |
| einsum_c64_concrete-shared_n8_single | 27210.58 | 588.84 | ok |
| einsum_c64_prepared-setup_n8_single | 6187.70 | 47.99 | ok |
| einsum_c64_concrete-fresh_n32_single | 35194.24 | 410.51 | ok |
| einsum_c64_concrete-shared_n32_single | 33045.68 | 513.11 | ok |
| einsum_c64_prepared-setup_n32_single | 6242.42 | 69.09 | ok |
| gather_f64_concrete_fresh | 2004.38 | 89.47 | ok |
| gather_f64_concrete-fresh_size16_single | 2039.61 | 27.10 | ok |
| gather_f64_concrete-fresh_size256_single | 2297.91 | 39.92 | ok |
| einsum_c64_borrowed-fresh_n2_col_major_contiguous_single | 29195.65 | 483.83 | ok |
| einsum_c64_borrowed-fresh_n2_row_major_contiguous_single | 29796.01 | 496.00 | ok |
| einsum_c64_borrowed-fresh_n2_strided_single | 32416.69 | 501.36 | ok |
| einsum_c64_borrowed-shared_n2_col_major_contiguous_single | 26925.58 | 400.76 | ok |
| einsum_c64_borrowed-shared_n2_row_major_contiguous_single | 27455.28 | 523.21 | ok |
| einsum_c64_borrowed-shared_n2_strided_single | 29461.06 | 349.74 | ok |
| einsum_c64_borrowed-fresh_n4_col_major_contiguous_single | 30156.25 | 225.36 | ok |
| einsum_c64_borrowed-fresh_n4_row_major_contiguous_single | 30170.63 | 225.01 | ok |
| einsum_c64_borrowed-fresh_n4_strided_single | 33212.66 | 328.37 | ok |
| einsum_c64_borrowed-shared_n4_col_major_contiguous_single | 27370.87 | 489.22 | ok |
| einsum_c64_borrowed-shared_n4_row_major_contiguous_single | 27200.89 | 415.55 | ok |
| einsum_c64_borrowed-shared_n4_strided_single | 30957.24 | 461.79 | ok |
| einsum_c64_borrowed-fresh_n16_col_major_contiguous_single | 31688.70 | 475.94 | ok |
| einsum_c64_borrowed-fresh_n16_row_major_contiguous_single | 31598.30 | 448.94 | ok |
| einsum_c64_borrowed-fresh_n16_strided_single | 34606.02 | 1137.47 | ok |
| einsum_c64_borrowed-shared_n16_col_major_contiguous_single | 28348.02 | 475.10 | ok |
| einsum_c64_borrowed-shared_n16_row_major_contiguous_single | 28586.92 | 523.37 | ok |
| einsum_c64_borrowed-shared_n16_strided_single | 32873.48 | 560.20 | ok |
| einsum_c64_borrowed-fresh_n2_broadcast_single | 32938.15 | 244.49 | ok |
| einsum_c64_borrowed-shared_n2_broadcast_single | 29791.95 | 421.50 | ok |
| einsum_c64_borrowed-fresh_n4_broadcast_single | 33086.04 | 396.96 | ok |
| einsum_c64_borrowed-shared_n4_broadcast_single | 30658.39 | 290.97 | ok |
| einsum_c64_borrowed-fresh_n16_broadcast_single | 34291.92 | 768.24 | ok |
| einsum_c64_borrowed-shared_n16_broadcast_single | 32204.25 | 258.72 | ok |
| reduce_sum_f64_concrete-fresh_size4_single | 1697.15 | 17.81 | ok |
| reduce_sum_f64_concrete-fresh_size16_single | 1702.32 | 14.77 | ok |
| reduce_sum_f64_concrete-fresh_size256_single | 1754.08 | 25.61 | ok |
| einsum_f64_concrete-shared_abcd-dbef-acef_d4_single | 33452.55 | 372.50 | ok |
| einsum_f64_concrete-shared_ax-asb-xsb_d4s2_single | 25984.72 | 660.75 | ok |
| einsum_f64_concrete-shared_ij-jk-ik_m95k95n1_single | 24744.04 | 358.43 | ok |
| add_f64_concrete_fresh | 107351.30 | 2969.10 | ok |
| add_f64_concrete-fresh_size4_dependent10 | 1083306.09 | 9702.65 | ok |
| add_f64_concrete-fresh_size16_single | 106940.53 | 2994.36 | ok |
| add_f64_concrete-fresh_size16_dependent10 | 1062161.79 | 9124.87 | ok |
| add_f64_concrete-fresh_size256_single | 105818.66 | 1600.04 | ok |
| add_f64_concrete-fresh_size256_dependent10 | 1084028.55 | 6147.23 | ok |
| add_f64_eager-no-ad_size4_single | 119581.70 | 903.22 | ok |
| add_f64_eager-no-ad_size4_dependent10 | 1198497.69 | 3059.67 | ok |
| add_f64_eager_no_ad | 119459.96 | 815.33 | ok |
| add_f64_eager-no-ad_size16_dependent10 | 1202032.40 | 3562.46 | ok |
| add_f64_eager-no-ad_size256_single | 120459.35 | 650.92 | ok |
| add_f64_eager-no-ad_size256_dependent10 | 1207283.13 | 7292.70 | ok |
| add_f64_eager_active_ad | 131593.68 | 576.10 | ok |
| add_f64_eager-ad_size4_dependent10 | 1314315.73 | 6895.40 | ok |
| add_f64_eager-ad_size16_single | 131058.43 | 979.99 | ok |
| add_f64_eager-ad_size16_dependent10 | 1332215.52 | 7283.07 | ok |
| add_f64_eager-ad_size256_single | 132789.06 | 1252.28 | ok |
| add_f64_eager-ad_size256_dependent10 | 1312457.22 | 21624.51 | ok |
| einsum_f64_concrete-fresh_n2_single | 26360.63 | 338.18 | ok |
| einsum_f64_concrete-fresh_n4_single | 26450.32 | 271.09 | ok |
| einsum_f64_concrete-fresh_n16_single | 26820.14 | 487.19 | ok |
| einsum_f64_concrete-shared_n2_single | 23866.68 | 265.66 | ok |
| einsum_f64_concrete-shared_n4_single | 24004.20 | 430.02 | ok |
| einsum_f64_concrete-shared_n16_single | 24276.73 | 477.46 | ok |
| einsum_f64_eager-no-ad_n2_single | 14910.23 | 415.93 | ok |
| einsum_f64_eager-no-ad_n4_single | 14833.19 | 228.57 | ok |
| einsum_f64_eager-no-ad_n16_single | 15627.60 | 252.16 | ok |
| einsum_f64_eager-ad_n2_single | 24516.01 | 892.35 | ok |
| einsum_f64_eager-ad_n4_single | 25362.81 | 819.88 | ok |
| einsum_f64_eager-ad_n16_single | 26334.54 | 778.03 | ok |
| einsum_f64_prepared-setup_n2_single | 6234.22 | 116.48 | ok |
| einsum_f64_prepared-setup_n4_single | 6307.80 | 113.66 | ok |
| einsum_f64_prepared-setup_n16_single | 6257.11 | 78.23 | ok |
| einsum_f64_borrowed-fresh_n2_col_major_contiguous_single | 27254.65 | 443.06 | ok |
| einsum_f64_borrowed-fresh_n2_row_major_contiguous_single | 26854.74 | 376.30 | ok |
| einsum_f64_borrowed-fresh_n2_strided_single | 29813.41 | 193.03 | ok |
| einsum_f64_borrowed-shared_n2_col_major_contiguous_single | 24093.35 | 402.04 | ok |
| einsum_f64_borrowed-shared_n2_row_major_contiguous_single | 24170.73 | 474.00 | ok |
| einsum_f64_borrowed-shared_n2_strided_single | 27374.53 | 275.11 | ok |
| einsum_f64_borrowed-fresh_n4_col_major_contiguous_single | 26833.48 | 426.39 | ok |
| einsum_f64_borrowed-fresh_n4_row_major_contiguous_single | 26811.80 | 515.20 | ok |
| einsum_f64_borrowed-fresh_n4_strided_single | 30269.24 | 724.68 | ok |
| einsum_f64_borrowed-shared_n4_col_major_contiguous_single | 24818.75 | 798.54 | ok |
| einsum_f64_borrowed-shared_n4_row_major_contiguous_single | 24459.66 | 526.32 | ok |
| einsum_f64_borrowed-shared_n4_strided_single | 26659.62 | 287.97 | ok |
| einsum_f64_borrowed-fresh_n16_col_major_contiguous_single | 28549.30 | 712.44 | ok |
| einsum_f64_borrowed-fresh_n16_row_major_contiguous_single | 28725.64 | 434.70 | ok |
| einsum_f64_borrowed-fresh_n16_strided_single | 31513.99 | 985.72 | ok |
| einsum_f64_borrowed-shared_n16_col_major_contiguous_single | 25222.35 | 492.51 | ok |
| einsum_f64_borrowed-shared_n16_row_major_contiguous_single | 24689.07 | 720.48 | ok |
| einsum_f64_borrowed-shared_n16_strided_single | 27708.47 | 658.77 | ok |
| einsum_f64_compiled-repeat_n2_single | 23522.07 | 336.41 | ok |
| einsum_f64_compiled-repeat_n4_single | 23940.73 | 325.90 | ok |
| einsum_f64_compiled-repeat_n16_single | 24111.20 | 498.28 | ok |
| einsum_c64_concrete-fresh_n2_single | 29152.13 | 443.57 | ok |
| einsum_c64_concrete-shared_n2_single | 26818.21 | 642.90 | ok |
| einsum_c64_concrete-fresh_n4_single | 29632.48 | 595.85 | ok |
| einsum_c64_concrete-shared_n4_single | 27539.01 | 781.90 | ok |
| einsum_c64_concrete-fresh_n16_single | 31234.20 | 536.02 | ok |
| einsum_c64_concrete-shared_n16_single | 28616.59 | 679.90 | ok |
| einsum_c64_prepared-setup_n2_single | 6294.47 | 69.13 | ok |
| einsum_c64_prepared-setup_n4_single | 6352.45 | 156.68 | ok |
| einsum_c64_prepared-setup_n16_single | 6253.70 | 138.74 | ok |
| einsum_c64_eager-no-ad_n2_single | 17466.55 | 583.49 | ok |
| einsum_c64_compiled-repeat_n2_single | 26594.48 | 361.75 | ok |
| einsum_c64_eager-no-ad_n4_single | 17512.34 | 432.84 | ok |
| einsum_c64_compiled-repeat_n4_single | 26865.65 | 518.51 | ok |
| einsum_c64_eager-no-ad_n16_single | 18901.33 | 302.85 | ok |
| einsum_c64_compiled-repeat_n16_single | 27824.51 | 676.73 | ok |
| einsum_c64_eager-ad_n2_single | 27368.03 | 396.08 | ok |
| einsum_c64_eager-ad_n4_single | 28170.28 | 382.54 | ok |
| einsum_c64_eager-ad_n16_single | 29849.84 | 1006.06 | ok |
| einsum_f64_compiled-setup_n2_single | 5392.42 | 122.52 | ok |
| einsum_f64_compiled-setup_n4_single | 5397.07 | 80.98 | NOISY |
| einsum_f64_compiled-setup_n16_single | 5292.05 | 88.13 | NOISY |
| einsum_c64_compiled-setup_n2_single | 5494.28 | 83.31 | ok |
| einsum_c64_compiled-setup_n4_single | 5446.03 | 55.22 | ok |
| einsum_c64_compiled-setup_n16_single | 5608.73 | 86.59 | ok |
| solve_f64_concrete-fresh_n2_single | 1968.58 | 33.32 | ok |
| solve_f64_concrete-fresh_n4_single | 2078.62 | 30.45 | ok |
| solve_f64_concrete-fresh_n16_single | 5035.81 | 72.54 | ok |
| einsum_f64_borrowed-fresh_n2_broadcast_single | 30252.71 | 591.34 | ok |
| einsum_f64_borrowed-shared_n2_broadcast_single | 27211.33 | 297.05 | ok |
| einsum_f64_borrowed-fresh_n4_broadcast_single | 30526.21 | 412.27 | ok |
| einsum_f64_borrowed-shared_n4_broadcast_single | 27320.49 | 364.38 | ok |
| einsum_f64_borrowed-fresh_n16_broadcast_single | 31697.58 | 513.19 | ok |
| einsum_f64_borrowed-shared_n16_broadcast_single | 28592.50 | 309.12 | ok |
| einsum_f64_concrete-fresh_n8_single | 27656.24 | 1130.95 | ok |
| einsum_f64_concrete-shared_n8_single | 24832.51 | 466.55 | ok |
| einsum_f64_prepared-setup_n8_single | 6288.66 | 69.81 | ok |
| einsum_f64_concrete-fresh_n32_single | 33414.81 | 1210.32 | ok |
| einsum_f64_concrete-shared_n32_single | 30592.32 | 779.54 | ok |
| einsum_f64_prepared-setup_n32_single | 6493.41 | 160.84 | ok |
| einsum_c64_concrete-fresh_n8_single | 30918.46 | 833.78 | ok |
| einsum_c64_concrete-shared_n8_single | 27932.79 | 628.53 | ok |
| einsum_c64_prepared-setup_n8_single | 6272.40 | 93.76 | ok |
| einsum_c64_concrete-fresh_n32_single | 35912.24 | 489.75 | ok |
| einsum_c64_concrete-shared_n32_single | 33517.05 | 672.43 | ok |
| einsum_c64_prepared-setup_n32_single | 6425.35 | 210.96 | ok |
| gather_f64_concrete_fresh | 100057.37 | 3590.95 | ok |
| gather_f64_concrete-fresh_size16_single | 95026.16 | 1925.67 | ok |
| gather_f64_concrete-fresh_size256_single | 92899.88 | 2707.91 | ok |
| einsum_c64_borrowed-fresh_n2_col_major_contiguous_single | 29389.53 | 428.23 | ok |
| einsum_c64_borrowed-fresh_n2_row_major_contiguous_single | 29875.61 | 504.94 | ok |
| einsum_c64_borrowed-fresh_n2_strided_single | 32631.56 | 264.69 | ok |
| einsum_c64_borrowed-shared_n2_col_major_contiguous_single | 26965.41 | 350.87 | ok |
| einsum_c64_borrowed-shared_n2_row_major_contiguous_single | 26837.67 | 477.55 | ok |
| einsum_c64_borrowed-shared_n2_strided_single | 29869.28 | 413.92 | ok |
| einsum_c64_borrowed-fresh_n4_col_major_contiguous_single | 30046.65 | 517.61 | ok |
| einsum_c64_borrowed-fresh_n4_row_major_contiguous_single | 30447.21 | 481.90 | ok |
| einsum_c64_borrowed-fresh_n4_strided_single | 33687.56 | 580.90 | ok |
| einsum_c64_borrowed-shared_n4_col_major_contiguous_single | 28004.58 | 257.27 | ok |
| einsum_c64_borrowed-shared_n4_row_major_contiguous_single | 28029.41 | 393.21 | ok |
| einsum_c64_borrowed-shared_n4_strided_single | 30205.19 | 695.74 | ok |
| einsum_c64_borrowed-fresh_n16_col_major_contiguous_single | 31229.21 | 344.96 | ok |
| einsum_c64_borrowed-fresh_n16_row_major_contiguous_single | 31857.24 | 653.45 | ok |
| einsum_c64_borrowed-fresh_n16_strided_single | 35053.62 | 652.12 | ok |
| einsum_c64_borrowed-shared_n16_col_major_contiguous_single | 29134.08 | 875.53 | ok |
| einsum_c64_borrowed-shared_n16_row_major_contiguous_single | 28714.63 | 811.80 | ok |
| einsum_c64_borrowed-shared_n16_strided_single | 30944.50 | 742.50 | ok |
| einsum_c64_borrowed-fresh_n2_broadcast_single | 32081.44 | 208.63 | ok |
| einsum_c64_borrowed-shared_n2_broadcast_single | 29583.39 | 647.16 | ok |
| einsum_c64_borrowed-fresh_n4_broadcast_single | 33729.42 | 243.31 | ok |
| einsum_c64_borrowed-shared_n4_broadcast_single | 31078.29 | 320.70 | ok |
| einsum_c64_borrowed-fresh_n16_broadcast_single | 34009.36 | 547.67 | ok |
| einsum_c64_borrowed-shared_n16_broadcast_single | 31792.82 | 514.02 | ok |
| reduce_sum_f64_concrete-fresh_size4_single | 94231.72 | 2523.41 | ok |
| reduce_sum_f64_concrete-fresh_size16_single | 91407.14 | 1408.29 | ok |
| reduce_sum_f64_concrete-fresh_size256_single | 96206.78 | 3672.26 | ok |
| einsum_f64_concrete-shared_abcd-dbef-acef_d4_single | 33815.21 | 601.24 | ok |
| einsum_f64_concrete-shared_ax-asb-xsb_d4s2_single | 25992.13 | 281.15 | ok |
| einsum_f64_concrete-shared_ij-jk-ik_m95k95n1_single | 25669.20 | 239.37 | ok |

## Per-route timing boundaries

Shared-session routes enter/exit the session once around warmup/calibration/all samples, outside timing. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions reuse the entered executor context. Earlier BLAS revisions included per-operation executor entry; consult the recorded tenferro commit. Fresh routes enter/exit for each operation. Prepared-repeat excludes plan preparation; compiled-repeat excludes tracing/compilation and runtime construction. Prepared-into-repeat and dot-general-into-shared write into one destination preallocated outside timing (no per-call output allocation) and re-check it after the last batch.

- **add / concrete-fresh** — inside: session_entry_exit, add, output_allocation; outside: backend_construction, input_construction, correctness_check, output_destruction, operation_config_construction.
- **add / concrete-shared** — inside: add, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **add / eager-ad** — inside: add_forward_recording, eager_dispatch, output_allocation; outside: backend_construction, input_construction, eager_runtime_construction, backward, correctness_check, output_destruction, operation_config_construction.
- **add / eager-no-ad** — inside: add, eager_dispatch, output_allocation; outside: backend_construction, input_construction, eager_runtime_construction, correctness_check, output_destruction, operation_config_construction.
- **einsum / borrowed-fresh** — inside: session_entry_exit, einsum, output_allocation; outside: backend_construction, input_construction, correctness_check, output_destruction, operation_config_construction.
- **einsum / borrowed-shared** — inside: einsum, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **einsum / compiled-repeat** — inside: runtime_admission_execution, output_allocation; outside: backend_construction, input_construction, trace_compile, runtime_construction, input_bindings, correctness_check, output_destruction, operation_config_construction.
- **einsum / compiled-setup** — inside: trace_compile, program_allocation; outside: backend_construction, input_construction, runtime_construction, input_bindings, correctness_check, output_destruction, operation_config_construction.
- **einsum / concrete-fresh** — inside: public extension call, intrinsic output allocation and result packaging, per-call session entry/exit diagnostic, ordinary einsum default Auto planning/cache lookup included equally; parsed integer notation; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **einsum / concrete-shared** — inside: public extension call, intrinsic output allocation and result packaging, session entry excluded, ordinary einsum default Auto planning/cache lookup included equally; parsed integer notation; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **einsum / dot-general-into-shared** — inside: dot_general_read_into; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, dot_general_config_construction, output_preallocation, output_destruction, operation_config_construction.
- **einsum / eager-ad** — inside: einsum_forward_recording, eager_dispatch, output_allocation; outside: backend_construction, input_construction, eager_runtime_construction, backward, correctness_check, output_destruction, operation_config_construction.
- **einsum / eager-no-ad** — inside: public extension call, intrinsic output allocation and result packaging, per-call session entry/exit diagnostic, ordinary einsum default Auto planning/cache lookup included equally; parsed integer notation; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **einsum / eager-shared** — inside: public extension call, intrinsic output allocation and result packaging, session entry excluded, ordinary einsum default Auto planning/cache lookup included equally; parsed integer notation; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **einsum / prepared-into-repeat** — inside: prepared_execute_into; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, plan_preparation, output_preallocation, output_destruction, operation_config_construction.
- **einsum / prepared-repeat** — inside: prepared_execute, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, plan_preparation, output_destruction, operation_config_construction.
- **einsum / prepared-setup** — inside: prepare, plan_allocation; outside: backend_construction, input_construction, correctness_check, output_destruction, operation_config_construction.
- **einsum3 / concrete-fresh** — inside: public extension call, intrinsic output allocation and result packaging, per-call session entry/exit diagnostic, ordinary einsum default Auto planning/cache lookup included equally; parsed integer notation; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **einsum3 / concrete-shared** — inside: public extension call, intrinsic output allocation and result packaging, session entry excluded, ordinary einsum default Auto planning/cache lookup included equally; parsed integer notation; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **einsum3 / eager-no-ad** — inside: public extension call, intrinsic output allocation and result packaging, per-call session entry/exit diagnostic, ordinary einsum default Auto planning/cache lookup included equally; parsed integer notation; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **einsum3 / eager-shared** — inside: public extension call, intrinsic output allocation and result packaging, session entry excluded, ordinary einsum default Auto planning/cache lookup included equally; parsed integer notation; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **gather / concrete-fresh** — inside: session_entry_exit, gather, output_allocation; outside: backend_construction, input_construction, correctness_check, output_destruction, operation_config_construction.
- **gather / concrete-shared** — inside: gather, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **qr / concrete-fresh** — inside: public extension call, intrinsic output allocation and result packaging, per-call session entry/exit diagnostic, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **qr / concrete-shared** — inside: public extension call, intrinsic output allocation and result packaging, session entry excluded, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **qr / eager-no-ad** — inside: public extension call, intrinsic output allocation and result packaging, per-call session entry/exit diagnostic, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **qr / eager-shared** — inside: public extension call, intrinsic output allocation and result packaging, session entry excluded, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **reduce_sum / concrete-fresh** — inside: session_entry_exit, reduce_sum, output_allocation; outside: backend_construction, input_construction, correctness_check, output_destruction, operation_config_construction.
- **reduce_sum / concrete-shared** — inside: reduce_sum, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **solve / concrete-fresh** — inside: public extension call, intrinsic output allocation and result packaging, per-call session entry/exit diagnostic, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **solve / concrete-shared** — inside: public extension call, intrinsic output allocation and result packaging, session entry excluded, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **solve / eager-no-ad** — inside: public extension call, intrinsic output allocation and result packaging, per-call session entry/exit diagnostic, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **solve / eager-shared** — inside: public extension call, intrinsic output allocation and result packaging, session entry excluded, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **svd / concrete-fresh** — inside: public extension call, intrinsic output allocation and result packaging, per-call session entry/exit diagnostic, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **svd / concrete-shared** — inside: public extension call, intrinsic output allocation and result packaging, session entry excluded, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **svd / eager-no-ad** — inside: public extension call, intrinsic output allocation and result packaging, per-call session entry/exit diagnostic, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.
- **svd / eager-shared** — inside: public extension call, intrinsic output allocation and result packaging, session entry excluded, default linalg options; outside: fixtures and view creation, backend/runtime construction, shared session entry/exit, initialization and warmups, validation/materialization, output destruction.

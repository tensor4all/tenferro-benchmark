# CPU Small-Work Benchmark Results

- Suite: `cpu/small_work`
- Raw run: `data/results/amd-cpu/cpu/small_work/20261008_062614`

## CPU Information

- Model: `AMD Ryzen 9 9955HX 16-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `32`
- Sockets: `1`
- Cores per socket: `16`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`

## run_t1

Full metadata: `data/results/amd-cpu/cpu/small_work/20261008_062614/run_t1.yaml`

```yaml
timestamp: '2026-10-08T06:26:14Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
  commit: 5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102
  dirty: false
  features:
  - system-mkl
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: mkl
  version: 2026.0.1
  root: /opt/intel/oneapi/mkl/latest
  library: /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so
environment:
  hostname: 14099c68ff8e
  os: Linux-7.0.0-38-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  env:
    OMP_NUM_THREADS: '1'
    OMP_THREAD_LIMIT: '1'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '1'
    TENFERRO_CPU_BACKEND_KIND: blas
    BENCHMARK_TARGET_PROFILE: amd-cpu
    BENCHMARK_COMMIT: 2cd0fd6445dcd9aad98990f76c17e33e773c4a7d
    OPENBLAS_ROOT: /opt/openblas
    MKLROOT: /opt/intel/oneapi/mkl/latest
    OPENBLAS_NUM_THREADS: '1'
    GOTO_NUM_THREADS: '1'
    MKL_NUM_THREADS: '1'
    VECLIB_MAXIMUM_THREADS: '1'
    VECLIB_NUM_THREADS: '1'
    NUMEXPR_NUM_THREADS: '1'
    BLIS_NUM_THREADS: '1'
    PJRT_NPROC: '1'
    XLA_FLAGS: --xla_cpu_multi_thread_eigen=false --xla_cpu_experimental_ynn_fusion_type=
      intra_op_parallelism_threads=1
    JULIA_NUM_THREADS: '1'
    USE_CUDA: '0'
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.0 (b940084d7 2026-09-28)
    BENCH_INSTANCE: null
```

## run_t4

Full metadata: `data/results/amd-cpu/cpu/small_work/20261008_062614/run_t4.yaml`

```yaml
timestamp: '2026-10-08T06:27:00Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
  commit: 5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102
  dirty: false
  features:
  - system-mkl
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: mkl
  version: 2026.0.1
  root: /opt/intel/oneapi/mkl/latest
  library: /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so
environment:
  hostname: 14099c68ff8e
  os: Linux-7.0.0-38-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  env:
    OMP_NUM_THREADS: '4'
    OMP_THREAD_LIMIT: '4'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '4'
    TENFERRO_CPU_BACKEND_KIND: blas
    BENCHMARK_TARGET_PROFILE: amd-cpu
    BENCHMARK_COMMIT: 2cd0fd6445dcd9aad98990f76c17e33e773c4a7d
    OPENBLAS_ROOT: /opt/openblas
    MKLROOT: /opt/intel/oneapi/mkl/latest
    OPENBLAS_NUM_THREADS: '4'
    GOTO_NUM_THREADS: '4'
    MKL_NUM_THREADS: '4'
    VECLIB_MAXIMUM_THREADS: '4'
    VECLIB_NUM_THREADS: '4'
    NUMEXPR_NUM_THREADS: '4'
    BLIS_NUM_THREADS: '4'
    PJRT_NPROC: '4'
    XLA_FLAGS: --xla_cpu_multi_thread_eigen=true --xla_cpu_experimental_ynn_fusion_type=
      intra_op_parallelism_threads=4
    JULIA_NUM_THREADS: '4'
    USE_CUDA: '0'
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.0 (b940084d7 2026-09-28)
    BENCH_INSTANCE: null
```

## Case status

- Manifest: `cpu/small_work` version 1, coverage `full`, effort `standard`
- Expected 332, selected 332, executed 332, unsupported 0, failed 0, missing 0, noisy 45.
- Complete: **yes** — only executed cases count as covered; unsupported, failed, missing and unselected cases never do.
- Noisy: `add_f64_concrete-fresh_size16_single@t1`, `add_f64_eager-no-ad_size4_dependent10@t1`, `add_f64_eager-no-ad_size256_single@t1`, `add_f64_eager-no-ad_size256_dependent10@t1`, `add_f64_eager_active_ad@t1`, `add_f64_eager-ad_size16_single@t1`, `add_f64_eager-ad_size16_dependent10@t1`, `add_f64_eager-ad_size256_single@t1`, `einsum_f64_eager-no-ad_n2_single@t1`, `einsum_f64_eager-no-ad_n4_single@t1`, `einsum_f64_eager-ad_n4_single@t1`, `einsum_f64_compiled-repeat_n4_single@t1`, `einsum_f64_compiled-repeat_n16_single@t1`, `einsum_c64_concrete-fresh_n4_single@t1`, `einsum_c64_compiled-repeat_n16_single@t1`, `einsum_c64_eager-ad_n2_single@t1`, `einsum_c64_eager-ad_n4_single@t1`, `einsum_c64_eager-ad_n16_single@t1`, `solve_f64_concrete-fresh_n16_single@t1`, `gather_f64_concrete_fresh@t1`, `reduce_sum_f64_concrete-fresh_size4_single@t1`, `reduce_sum_f64_concrete-fresh_size16_single@t1`, `add_f64_concrete_fresh@t4`, `add_f64_concrete-fresh_size16_single@t4`, `add_f64_concrete_shared@t4`, `add_f64_concrete-shared_size256_single@t4`, `add_f64_eager_no_ad@t4`, `add_f64_eager-no-ad_size16_dependent10@t4`, `add_f64_eager_active_ad@t4`, `add_f64_eager-ad_size16_single@t4`, `einsum_f64_eager-ad_n2_single@t4`, `einsum_f64_compiled-repeat_n2_single@t4`, `einsum_f64_compiled-repeat_n16_single@t4`, `einsum_c64_concrete-fresh_n16_single@t4`, `einsum_c64_eager-no-ad_n4_single@t4`, `einsum_c64_compiled-repeat_n4_single@t4`, `einsum_c64_eager-ad_n16_single@t4`, `solve_f64_concrete-fresh_n4_single@t4`, `einsum_f64_concrete-shared_n32_single@t4`, `einsum_f64_prepared-repeat_n32_single@t4` (+5 more)

Numerical checks: **332/332 recorded rows passed** (case × thread count).

## Timing

Sequential release runs; warmups, then calibration from at least 1024 operations toward a 1 ms batch, then the measured batches of the recorded effort (standard: 3 warmups and 15 batches; scan: a low-repetition screen that only marks suspects). Each batch uses one wall-clock interval; all outputs are retained until it stops. Statistics divide batch time by iterations. Median and inclusive IQR are in nanoseconds (ns); CoV is sample standard deviation / mean. NOISY means CoV > 10%, a descriptive label only. No noisy rows are excluded.

Workflow time is the total for one call or the complete dependent chain. The separate ns/op column divides the chain median by its operation count, not an independently measured call. Setup rows measure preparation only, not execution.

Inputs and backend construction, independent numerical checks and AD backward checks are outside timing. Returned outputs/plans are retained until after each batch interval; value downloads/checks are outside. Eager AD rows measure forward execution/recording, not backward.

| Case ID | Operation | API route | Route (path / output / repr / session) | Dtype | Shape | Layout | Workflow | Threads | Provider | Operations/batch | Median batch ms | Median workflow ns | IQR ns | CoV | ns/op (chain only) | Status |
|---|---|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 1024 | 1.042783 | 1018.34 | 11.77 | 2.1% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 1 | blas | 10240 | 9.642888 | 9416.88 | 60.19 | 0.7% | 941.69 | ok |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 1024 | 1.050357 | 1025.74 | 15.74 | 2.7% | — | ok |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 1 | blas | 10240 | 10.278936 | 10038.02 | 47.25 | 1.6% | 1003.80 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 1024 | 1.536794 | 1500.78 | 8.18 | 2.2% | — | ok |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 1 | blas | 10240 | 11.099130 | 10838.99 | 51.15 | 0.5% | 1083.90 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 2.933715 | 2864.96 | 22.14 | 6.7% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 3.002324 | 2931.96 | 46.90 | 4.5% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 4.305107 | 4204.21 | 22.22 | 5.0% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 3.064571 | 2992.75 | 17.68 | 0.9% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 3.135214 | 3061.73 | 19.55 | 1.5% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 5.325388 | 5200.57 | 49.20 | 1.0% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 1 | blas | 1024 | 1.331307 | 1300.10 | 10.23 | 6.7% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 1 | blas | 1024 | 1.410095 | 1377.05 | 13.32 | 1.5% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 1 | blas | 1024 | 4.827640 | 4714.49 | 33.59 | 2.3% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 2.987266 | 2917.25 | 17.10 | 2.0% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 8.366455 | 8170.37 | 24.36 | 2.2% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 1 | blas | 1024 | 3.883062 | 3792.05 | 27.05 | 2.8% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 1 | blas | 1024 | 15.297677 | 14939.14 | 90.65 | 0.9% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 1024 | 1.244383 | 1215.22 | 11.72 | 1.7% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 1024 | 1.269030 | 1239.29 | 16.61 | 1.9% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 1024 | 2.374531 | 2318.88 | 11.74 | 1.3% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 1 | blas | 1024 | 1.200481 | 1172.34 | 5.48 | 1.6% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 1 | blas | 1024 | 1.200992 | 1172.84 | 6.24 | 5.4% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 1 | blas | 1024 | 1.212593 | 1184.17 | 5.64 | 1.5% | — | ok |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 10.518777 | 10272.24 | 43.19 | 0.5% | — | ok |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 7.596805 | 7418.75 | 23.56 | 2.7% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 1 | blas | 1024 | 7.524660 | 7348.30 | 24.72 | 3.0% | — | ok |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 1024 | 3.858175 | 3767.75 | 12.67 | 1.0% | — | ok |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 1024 | 1.833423 | 1790.45 | 6.92 | 7.3% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 1 | blas | 1024 | 1.734737 | 1694.08 | 4.04 | 0.9% | — | ok |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 1024 | 4.480247 | 4375.24 | 24.95 | 0.4% | — | ok |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 1024 | 2.466545 | 2408.74 | 19.58 | 3.4% | — | ok |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 1 | blas | 1024 | 2.389640 | 2333.63 | 3.95 | 0.8% | — | ok |
| add_f64_concrete-shared_size4_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 1024 | 1.034488 | 1010.24 | 5.76 | 2.6% | — | ok |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | dependent10 | 4 | blas | 10240 | 9.624463 | 9398.89 | 41.06 | 1.3% | 939.89 | ok |
| add_f64_concrete_shared | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 1024 | 1.008849 | 985.20 | 6.19 | 14.2% | — | NOISY |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | dependent10 | 4 | blas | 10240 | 9.925009 | 9692.39 | 48.24 | 0.4% | 969.24 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 1.521304 | 1485.65 | 38.82 | 12.8% | — | NOISY |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | dependent10 | 4 | blas | 10240 | 11.122534 | 10861.85 | 45.13 | 0.4% | 1086.18 | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 2.940296 | 2871.38 | 83.72 | 6.6% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 3.006491 | 2936.03 | 15.21 | 1.1% | — | ok |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 4.388855 | 4285.99 | 64.55 | 2.3% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 3.080701 | 3008.50 | 17.78 | 3.1% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 3.012513 | 2941.91 | 21.04 | 1.2% | — | ok |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 5.327011 | 5202.16 | 23.94 | 1.0% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 2×2 | col_major_contiguous | single | 4 | blas | 1024 | 1.330094 | 1298.92 | 6.95 | 1.3% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4×4 | col_major_contiguous | single | 4 | blas | 1024 | 1.411207 | 1378.13 | 7.99 | 1.1% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16×16 | col_major_contiguous | single | 4 | blas | 1024 | 4.809847 | 4697.12 | 14.20 | 1.0% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 3.027120 | 2956.17 | 55.95 | 1.9% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 9.336190 | 9117.37 | 273.31 | 18.2% | — | NOISY |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 8×8 | col_major_contiguous | single | 4 | blas | 1024 | 3.791249 | 3702.39 | 44.84 | 4.6% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | c64 | 32×32 | col_major_contiguous | single | 4 | blas | 1024 | 16.931373 | 16534.54 | 214.24 | 10.9% | — | NOISY |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 1024 | 1.262938 | 1233.34 | 6.55 | 1.5% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 1024 | 1.287334 | 1257.16 | 6.86 | 2.2% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 2.413044 | 2356.49 | 13.06 | 1.3% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 4 | col_major_contiguous | single | 4 | blas | 1024 | 1.202474 | 1174.29 | 4.41 | 1.6% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 16 | col_major_contiguous | single | 4 | blas | 1024 | 1.206141 | 1177.87 | 8.86 | 1.7% | — | ok |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | concrete / allocating / owned / shared-session | f64 | 256 | col_major_contiguous | single | 4 | blas | 1024 | 1.214107 | 1185.65 | 12.86 | 2.1% | — | ok |
| einsum_f64_prepared-repeat_abcd-dbef-acef_d4_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 10.483691 | 10237.98 | 131.47 | 2.3% | — | ok |
| einsum_f64_prepared-into-repeat_abcd-dbef-acef_d4_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 7.594461 | 7416.47 | 55.85 | 1.5% | — | ok |
| einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4 | col_major_contiguous | single | 4 | blas | 1024 | 7.526182 | 7349.79 | 44.00 | 0.5% | — | ok |
| einsum_f64_prepared-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 1024 | 3.875388 | 3784.56 | 37.82 | 2.1% | — | ok |
| einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 1024 | 1.837921 | 1794.84 | 12.51 | 1.1% | — | ok |
| einsum_f64_dot-general-into-shared_ax-asb-xsb_d4s2_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ax,asb->xsb` 4×4, 4×2×4 → 4×2×4 | col_major_contiguous | single | 4 | blas | 1024 | 1.753472 | 1712.38 | 12.75 | 6.3% | — | ok |
| einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-repeat | concrete-prepared / allocating / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 1024 | 4.490976 | 4385.72 | 20.42 | 0.8% | — | ok |
| einsum_f64_prepared-into-repeat_ij-jk-ik_m95k95n1_single | einsum | prepared-into-repeat | concrete-prepared / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 1024 | 2.456566 | 2398.99 | 9.86 | 0.6% | — | ok |
| einsum_f64_dot-general-into-shared_ij-jk-ik_m95k95n1_single | einsum | dot-general-into-shared | concrete / reused / owned / shared-session | f64 | `ij,jk->ik` 95×95, 95×1 → 95×1 | col_major_contiguous | single | 4 | blas | 1024 | 2.381986 | 2326.16 | 21.66 | 0.8% | — | ok |

## Explicit setup diagnostics (not operation comparisons)

| Case | Median ns | IQR ns | Status |
|---|---:|---:|---|
| add_f64_concrete_fresh | 3731.98 | 91.35 | ok |
| add_f64_concrete-fresh_size4_dependent10 | 69469.65 | 838.71 | ok |
| add_f64_concrete-fresh_size16_single | 3627.62 | 344.79 | NOISY |
| add_f64_concrete-fresh_size16_dependent10 | 67857.83 | 1011.07 | ok |
| add_f64_concrete-fresh_size256_single | 6793.38 | 505.46 | ok |
| add_f64_concrete-fresh_size256_dependent10 | 37904.91 | 304.71 | ok |
| add_f64_eager-no-ad_size4_single | 9330.12 | 227.82 | ok |
| add_f64_eager-no-ad_size4_dependent10 | 62364.50 | 24015.51 | NOISY |
| add_f64_eager_no_ad | 8948.85 | 92.58 | ok |
| add_f64_eager-no-ad_size16_dependent10 | 65933.20 | 1240.89 | ok |
| add_f64_eager-no-ad_size256_single | 6826.47 | 2632.65 | NOISY |
| add_f64_eager-no-ad_size256_dependent10 | 64687.67 | 6526.02 | NOISY |
| add_f64_eager_active_ad | 12291.17 | 3461.98 | NOISY |
| add_f64_eager-ad_size4_dependent10 | 110583.69 | 2055.29 | ok |
| add_f64_eager-ad_size16_single | 11392.40 | 1368.83 | NOISY |
| add_f64_eager-ad_size16_dependent10 | 107922.95 | 5511.12 | NOISY |
| add_f64_eager-ad_size256_single | 11715.04 | 791.96 | NOISY |
| add_f64_eager-ad_size256_dependent10 | 118123.37 | 2766.09 | ok |
| einsum_f64_concrete-fresh_n2_single | 15438.32 | 109.89 | ok |
| einsum_f64_concrete-fresh_n4_single | 12554.30 | 423.03 | ok |
| einsum_f64_concrete-fresh_n16_single | 14213.26 | 406.80 | ok |
| einsum_f64_concrete-shared_n2_single | 9643.62 | 61.25 | ok |
| einsum_f64_concrete-shared_n4_single | 9423.63 | 112.82 | ok |
| einsum_f64_concrete-shared_n16_single | 10743.35 | 69.56 | ok |
| einsum_f64_eager-no-ad_n2_single | 7340.17 | 386.48 | NOISY |
| einsum_f64_eager-no-ad_n4_single | 7237.83 | 693.61 | NOISY |
| einsum_f64_eager-no-ad_n16_single | 10892.65 | 196.50 | ok |
| einsum_f64_eager-ad_n2_single | 12063.48 | 344.16 | ok |
| einsum_f64_eager-ad_n4_single | 12335.51 | 1008.08 | NOISY |
| einsum_f64_eager-ad_n16_single | 13191.12 | 315.83 | ok |
| einsum_f64_prepared-setup_n2_single | 5635.68 | 101.60 | ok |
| einsum_f64_prepared-setup_n4_single | 5569.54 | 22.13 | ok |
| einsum_f64_prepared-setup_n16_single | 5570.25 | 15.29 | ok |
| einsum_f64_borrowed-fresh_n2_col_major_contiguous_single | 13665.23 | 89.34 | ok |
| einsum_f64_borrowed-fresh_n2_row_major_contiguous_single | 16269.63 | 243.36 | ok |
| einsum_f64_borrowed-fresh_n2_strided_single | 17779.63 | 106.23 | ok |
| einsum_f64_borrowed-shared_n2_col_major_contiguous_single | 10221.79 | 54.93 | ok |
| einsum_f64_borrowed-shared_n2_row_major_contiguous_single | 10228.65 | 118.51 | ok |
| einsum_f64_borrowed-shared_n2_strided_single | 14146.34 | 124.84 | ok |
| einsum_f64_borrowed-fresh_n4_col_major_contiguous_single | 16318.20 | 2545.00 | ok |
| einsum_f64_borrowed-fresh_n4_row_major_contiguous_single | 16696.28 | 254.73 | ok |
| einsum_f64_borrowed-fresh_n4_strided_single | 17954.68 | 106.31 | ok |
| einsum_f64_borrowed-shared_n4_col_major_contiguous_single | 10317.29 | 167.92 | ok |
| einsum_f64_borrowed-shared_n4_row_major_contiguous_single | 10420.12 | 191.11 | ok |
| einsum_f64_borrowed-shared_n4_strided_single | 14114.21 | 63.03 | ok |
| einsum_f64_borrowed-fresh_n16_col_major_contiguous_single | 17401.97 | 3117.14 | ok |
| einsum_f64_borrowed-fresh_n16_row_major_contiguous_single | 15999.27 | 1766.59 | ok |
| einsum_f64_borrowed-fresh_n16_strided_single | 22520.75 | 237.77 | ok |
| einsum_f64_borrowed-shared_n16_col_major_contiguous_single | 11966.14 | 118.17 | ok |
| einsum_f64_borrowed-shared_n16_row_major_contiguous_single | 12076.89 | 208.80 | ok |
| einsum_f64_borrowed-shared_n16_strided_single | 16190.73 | 164.18 | ok |
| einsum_f64_compiled-repeat_n2_single | 20391.35 | 350.42 | ok |
| einsum_f64_compiled-repeat_n4_single | 14923.39 | 1106.58 | NOISY |
| einsum_f64_compiled-repeat_n16_single | 16107.96 | 2998.99 | NOISY |
| einsum_c64_concrete-fresh_n2_single | 15758.17 | 201.50 | ok |
| einsum_c64_concrete-shared_n2_single | 9318.36 | 64.03 | ok |
| einsum_c64_concrete-fresh_n4_single | 13552.94 | 2872.51 | NOISY |
| einsum_c64_concrete-shared_n4_single | 9676.81 | 109.44 | ok |
| einsum_c64_concrete-fresh_n16_single | 15587.94 | 1566.40 | ok |
| einsum_c64_concrete-shared_n16_single | 11674.04 | 375.02 | ok |
| einsum_c64_prepared-setup_n2_single | 5565.67 | 14.22 | ok |
| einsum_c64_prepared-setup_n4_single | 5612.83 | 7.30 | ok |
| einsum_c64_prepared-setup_n16_single | 5604.06 | 24.94 | ok |
| einsum_c64_eager-no-ad_n2_single | 7234.99 | 40.19 | ok |
| einsum_c64_compiled-repeat_n2_single | 21576.35 | 374.99 | ok |
| einsum_c64_eager-no-ad_n4_single | 7280.35 | 759.18 | ok |
| einsum_c64_compiled-repeat_n4_single | 14680.75 | 643.20 | ok |
| einsum_c64_eager-no-ad_n16_single | 9664.38 | 196.90 | ok |
| einsum_c64_compiled-repeat_n16_single | 15808.59 | 210.62 | NOISY |
| einsum_c64_eager-ad_n2_single | 12686.51 | 4147.10 | NOISY |
| einsum_c64_eager-ad_n4_single | 12415.14 | 597.77 | NOISY |
| einsum_c64_eager-ad_n16_single | 14040.12 | 983.89 | NOISY |
| einsum_f64_compiled-setup_n2_single | 5144.98 | 26.83 | ok |
| einsum_f64_compiled-setup_n4_single | 5195.65 | 34.64 | ok |
| einsum_f64_compiled-setup_n16_single | 5171.26 | 13.72 | ok |
| einsum_c64_compiled-setup_n2_single | 5127.68 | 9.35 | ok |
| einsum_c64_compiled-setup_n4_single | 5132.62 | 36.91 | ok |
| einsum_c64_compiled-setup_n16_single | 5179.21 | 24.95 | ok |
| solve_f64_concrete-fresh_n2_single | 7013.65 | 533.95 | ok |
| solve_f64_concrete-fresh_n4_single | 7055.12 | 450.42 | ok |
| solve_f64_concrete-fresh_n16_single | 7552.21 | 647.51 | NOISY |
| einsum_f64_borrowed-fresh_n2_broadcast_single | 17747.68 | 93.28 | ok |
| einsum_f64_borrowed-shared_n2_broadcast_single | 14120.65 | 214.83 | ok |
| einsum_f64_borrowed-fresh_n4_broadcast_single | 17625.32 | 116.70 | ok |
| einsum_f64_borrowed-shared_n4_broadcast_single | 14079.07 | 140.78 | ok |
| einsum_f64_borrowed-fresh_n16_broadcast_single | 19690.04 | 108.89 | ok |
| einsum_f64_borrowed-shared_n16_broadcast_single | 16237.16 | 110.48 | ok |
| einsum_f64_concrete-fresh_n8_single | 15524.36 | 46.45 | ok |
| einsum_f64_concrete-shared_n8_single | 9608.83 | 48.28 | ok |
| einsum_f64_prepared-setup_n8_single | 5580.30 | 132.70 | ok |
| einsum_f64_concrete-fresh_n32_single | 18774.51 | 2624.43 | ok |
| einsum_f64_concrete-shared_n32_single | 14542.84 | 84.28 | ok |
| einsum_f64_prepared-setup_n32_single | 5583.42 | 31.82 | ok |
| einsum_c64_concrete-fresh_n8_single | 15019.31 | 797.27 | ok |
| einsum_c64_concrete-shared_n8_single | 10150.38 | 52.66 | ok |
| einsum_c64_prepared-setup_n8_single | 5609.89 | 17.76 | ok |
| einsum_c64_concrete-fresh_n32_single | 25697.24 | 288.29 | ok |
| einsum_c64_concrete-shared_n32_single | 21324.97 | 68.21 | ok |
| einsum_c64_prepared-setup_n32_single | 5598.20 | 73.42 | ok |
| gather_f64_concrete_fresh | 3682.29 | 288.54 | NOISY |
| gather_f64_concrete-fresh_size16_single | 3796.63 | 240.73 | ok |
| gather_f64_concrete-fresh_size256_single | 5077.96 | 745.68 | ok |
| einsum_c64_borrowed-fresh_n2_col_major_contiguous_single | 13589.02 | 360.04 | ok |
| einsum_c64_borrowed-fresh_n2_row_major_contiguous_single | 13843.62 | 162.75 | ok |
| einsum_c64_borrowed-fresh_n2_strided_single | 17977.62 | 135.39 | ok |
| einsum_c64_borrowed-shared_n2_col_major_contiguous_single | 10139.00 | 144.80 | ok |
| einsum_c64_borrowed-shared_n2_row_major_contiguous_single | 10256.39 | 190.20 | ok |
| einsum_c64_borrowed-shared_n2_strided_single | 14397.83 | 88.47 | ok |
| einsum_c64_borrowed-fresh_n4_col_major_contiguous_single | 14362.19 | 151.76 | ok |
| einsum_c64_borrowed-fresh_n4_row_major_contiguous_single | 14876.48 | 2751.15 | ok |
| einsum_c64_borrowed-fresh_n4_strided_single | 20804.29 | 152.59 | ok |
| einsum_c64_borrowed-shared_n4_col_major_contiguous_single | 10751.20 | 201.01 | ok |
| einsum_c64_borrowed-shared_n4_row_major_contiguous_single | 10908.01 | 81.64 | ok |
| einsum_c64_borrowed-shared_n4_strided_single | 14458.29 | 114.40 | ok |
| einsum_c64_borrowed-fresh_n16_col_major_contiguous_single | 16407.50 | 343.82 | ok |
| einsum_c64_borrowed-fresh_n16_row_major_contiguous_single | 16892.60 | 2773.84 | ok |
| einsum_c64_borrowed-fresh_n16_strided_single | 23839.28 | 153.68 | ok |
| einsum_c64_borrowed-shared_n16_col_major_contiguous_single | 12709.40 | 46.56 | ok |
| einsum_c64_borrowed-shared_n16_row_major_contiguous_single | 12877.70 | 176.10 | ok |
| einsum_c64_borrowed-shared_n16_strided_single | 17818.52 | 189.11 | ok |
| einsum_c64_borrowed-fresh_n2_broadcast_single | 20521.07 | 138.67 | ok |
| einsum_c64_borrowed-shared_n2_broadcast_single | 14614.65 | 66.20 | ok |
| einsum_c64_borrowed-fresh_n4_broadcast_single | 18171.53 | 102.08 | ok |
| einsum_c64_borrowed-shared_n4_broadcast_single | 14594.95 | 80.69 | ok |
| einsum_c64_borrowed-fresh_n16_broadcast_single | 20955.72 | 119.28 | ok |
| einsum_c64_borrowed-shared_n16_broadcast_single | 17393.64 | 96.25 | ok |
| reduce_sum_f64_concrete-fresh_size4_single | 3905.77 | 2778.12 | NOISY |
| reduce_sum_f64_concrete-fresh_size16_single | 6023.94 | 2723.12 | NOISY |
| reduce_sum_f64_concrete-fresh_size256_single | 6796.96 | 43.54 | ok |
| einsum_f64_concrete-shared_abcd-dbef-acef_d4_single | 20331.09 | 124.40 | ok |
| einsum_f64_concrete-shared_ax-asb-xsb_d4s2_single | 11045.76 | 61.04 | ok |
| einsum_f64_concrete-shared_ij-jk-ik_m95k95n1_single | 10810.33 | 47.81 | ok |
| add_f64_concrete_fresh | 4912.75 | 2532.09 | NOISY |
| add_f64_concrete-fresh_size4_dependent10 | 43154.44 | 991.83 | ok |
| add_f64_concrete-fresh_size16_single | 4297.19 | 164.13 | NOISY |
| add_f64_concrete-fresh_size16_dependent10 | 43910.97 | 1441.85 | ok |
| add_f64_concrete-fresh_size256_single | 4381.29 | 248.45 | ok |
| add_f64_concrete-fresh_size256_dependent10 | 47261.39 | 2945.68 | ok |
| add_f64_eager-no-ad_size4_single | 7702.80 | 320.90 | ok |
| add_f64_eager-no-ad_size4_dependent10 | 78212.89 | 1773.58 | ok |
| add_f64_eager_no_ad | 8426.81 | 334.43 | NOISY |
| add_f64_eager-no-ad_size16_dependent10 | 78975.44 | 3710.04 | NOISY |
| add_f64_eager-no-ad_size256_single | 8046.22 | 270.85 | ok |
| add_f64_eager-no-ad_size256_dependent10 | 75452.89 | 2842.92 | ok |
| add_f64_eager_active_ad | 13597.97 | 2804.35 | NOISY |
| add_f64_eager-ad_size4_dependent10 | 132084.49 | 4545.95 | ok |
| add_f64_eager-ad_size16_single | 16558.13 | 2871.40 | NOISY |
| add_f64_eager-ad_size16_dependent10 | 135546.44 | 3486.82 | ok |
| add_f64_eager-ad_size256_single | 13573.43 | 456.19 | ok |
| add_f64_eager-ad_size256_dependent10 | 139050.53 | 5940.93 | ok |
| einsum_f64_concrete-fresh_n2_single | 15237.63 | 2677.11 | ok |
| einsum_f64_concrete-fresh_n4_single | 14633.90 | 1076.69 | ok |
| einsum_f64_concrete-fresh_n16_single | 18479.84 | 1875.69 | ok |
| einsum_f64_concrete-shared_n2_single | 9549.36 | 69.21 | ok |
| einsum_f64_concrete-shared_n4_single | 9492.71 | 91.36 | ok |
| einsum_f64_concrete-shared_n16_single | 10644.44 | 61.18 | ok |
| einsum_f64_eager-no-ad_n2_single | 8265.00 | 254.48 | ok |
| einsum_f64_eager-no-ad_n4_single | 8689.92 | 465.92 | ok |
| einsum_f64_eager-no-ad_n16_single | 9712.41 | 592.89 | ok |
| einsum_f64_eager-ad_n2_single | 13758.04 | 1028.41 | NOISY |
| einsum_f64_eager-ad_n4_single | 13977.67 | 615.80 | ok |
| einsum_f64_eager-ad_n16_single | 15592.19 | 1317.51 | ok |
| einsum_f64_prepared-setup_n2_single | 5575.41 | 183.27 | ok |
| einsum_f64_prepared-setup_n4_single | 5538.64 | 26.58 | ok |
| einsum_f64_prepared-setup_n16_single | 5592.15 | 15.57 | ok |
| einsum_f64_borrowed-fresh_n2_col_major_contiguous_single | 15768.67 | 1085.58 | ok |
| einsum_f64_borrowed-fresh_n2_row_major_contiguous_single | 17967.98 | 1038.78 | ok |
| einsum_f64_borrowed-fresh_n2_strided_single | 19189.20 | 582.82 | ok |
| einsum_f64_borrowed-shared_n2_col_major_contiguous_single | 10339.33 | 77.28 | ok |
| einsum_f64_borrowed-shared_n2_row_major_contiguous_single | 10153.14 | 120.14 | ok |
| einsum_f64_borrowed-shared_n2_strided_single | 14231.91 | 186.23 | ok |
| einsum_f64_borrowed-fresh_n4_col_major_contiguous_single | 17219.58 | 1381.15 | ok |
| einsum_f64_borrowed-fresh_n4_row_major_contiguous_single | 19207.08 | 816.85 | ok |
| einsum_f64_borrowed-fresh_n4_strided_single | 20083.15 | 1537.60 | ok |
| einsum_f64_borrowed-shared_n4_col_major_contiguous_single | 10345.88 | 121.81 | ok |
| einsum_f64_borrowed-shared_n4_row_major_contiguous_single | 10653.06 | 157.52 | ok |
| einsum_f64_borrowed-shared_n4_strided_single | 14018.13 | 90.94 | ok |
| einsum_f64_borrowed-fresh_n16_col_major_contiguous_single | 17324.18 | 1286.15 | ok |
| einsum_f64_borrowed-fresh_n16_row_major_contiguous_single | 17106.97 | 1090.72 | ok |
| einsum_f64_borrowed-fresh_n16_strided_single | 24317.20 | 2739.41 | ok |
| einsum_f64_borrowed-shared_n16_col_major_contiguous_single | 12088.11 | 45.04 | ok |
| einsum_f64_borrowed-shared_n16_row_major_contiguous_single | 12117.74 | 93.71 | ok |
| einsum_f64_borrowed-shared_n16_strided_single | 16264.71 | 71.31 | ok |
| einsum_f64_compiled-repeat_n2_single | 17748.05 | 2280.21 | NOISY |
| einsum_f64_compiled-repeat_n4_single | 18102.39 | 1302.01 | ok |
| einsum_f64_compiled-repeat_n16_single | 19707.48 | 6920.79 | NOISY |
| einsum_c64_concrete-fresh_n2_single | 17559.11 | 1530.55 | ok |
| einsum_c64_concrete-shared_n2_single | 9340.40 | 47.90 | ok |
| einsum_c64_concrete-fresh_n4_single | 14613.41 | 1015.80 | ok |
| einsum_c64_concrete-shared_n4_single | 9518.48 | 55.77 | ok |
| einsum_c64_concrete-fresh_n16_single | 17255.75 | 2387.68 | NOISY |
| einsum_c64_concrete-shared_n16_single | 11657.09 | 85.07 | ok |
| einsum_c64_prepared-setup_n2_single | 5587.75 | 30.89 | ok |
| einsum_c64_prepared-setup_n4_single | 5570.11 | 67.71 | ok |
| einsum_c64_prepared-setup_n16_single | 5577.84 | 41.90 | ok |
| einsum_c64_eager-no-ad_n2_single | 8903.17 | 1196.10 | ok |
| einsum_c64_compiled-repeat_n2_single | 17750.69 | 930.00 | ok |
| einsum_c64_eager-no-ad_n4_single | 8544.42 | 2077.33 | NOISY |
| einsum_c64_compiled-repeat_n4_single | 18457.17 | 6187.29 | NOISY |
| einsum_c64_eager-no-ad_n16_single | 13347.16 | 1228.70 | ok |
| einsum_c64_compiled-repeat_n16_single | 19782.66 | 1049.47 | ok |
| einsum_c64_eager-ad_n2_single | 13868.81 | 1567.48 | ok |
| einsum_c64_eager-ad_n4_single | 13934.98 | 862.81 | ok |
| einsum_c64_eager-ad_n16_single | 17656.54 | 3282.66 | NOISY |
| einsum_f64_compiled-setup_n2_single | 5227.95 | 50.75 | ok |
| einsum_f64_compiled-setup_n4_single | 5132.13 | 53.13 | ok |
| einsum_f64_compiled-setup_n16_single | 5156.53 | 20.05 | ok |
| einsum_c64_compiled-setup_n2_single | 5110.77 | 30.07 | ok |
| einsum_c64_compiled-setup_n4_single | 5155.03 | 82.97 | ok |
| einsum_c64_compiled-setup_n16_single | 5162.92 | 15.00 | ok |
| solve_f64_concrete-fresh_n2_single | 5080.11 | 149.72 | ok |
| solve_f64_concrete-fresh_n4_single | 4564.80 | 328.94 | NOISY |
| solve_f64_concrete-fresh_n16_single | 9449.64 | 346.52 | ok |
| einsum_f64_borrowed-fresh_n2_broadcast_single | 22005.76 | 277.01 | ok |
| einsum_f64_borrowed-shared_n2_broadcast_single | 14069.46 | 89.55 | ok |
| einsum_f64_borrowed-fresh_n4_broadcast_single | 22406.03 | 812.70 | ok |
| einsum_f64_borrowed-shared_n4_broadcast_single | 14239.81 | 108.41 | ok |
| einsum_f64_borrowed-fresh_n16_broadcast_single | 24599.32 | 871.37 | ok |
| einsum_f64_borrowed-shared_n16_broadcast_single | 16025.46 | 83.46 | ok |
| einsum_f64_concrete-fresh_n8_single | 14387.91 | 326.49 | ok |
| einsum_f64_concrete-shared_n8_single | 9460.70 | 165.46 | ok |
| einsum_f64_prepared-setup_n8_single | 5526.75 | 43.24 | ok |
| einsum_f64_concrete-fresh_n32_single | 27017.28 | 4235.70 | ok |
| einsum_f64_concrete-shared_n32_single | 15280.00 | 119.39 | NOISY |
| einsum_f64_prepared-setup_n32_single | 5684.29 | 356.93 | ok |
| einsum_c64_concrete-fresh_n8_single | 15587.04 | 1284.75 | ok |
| einsum_c64_concrete-shared_n8_single | 10160.43 | 82.74 | ok |
| einsum_c64_prepared-setup_n8_single | 5539.00 | 68.17 | ok |
| einsum_c64_concrete-fresh_n32_single | 30444.69 | 4566.16 | ok |
| einsum_c64_concrete-shared_n32_single | 22733.80 | 68.02 | ok |
| einsum_c64_prepared-setup_n32_single | 5549.10 | 10.93 | ok |
| gather_f64_concrete_fresh | 6312.93 | 1243.27 | NOISY |
| gather_f64_concrete-fresh_size16_single | 5100.73 | 383.21 | ok |
| gather_f64_concrete-fresh_size256_single | 6878.74 | 1084.07 | ok |
| einsum_c64_borrowed-fresh_n2_col_major_contiguous_single | 15529.76 | 642.98 | ok |
| einsum_c64_borrowed-fresh_n2_row_major_contiguous_single | 15244.53 | 1589.24 | ok |
| einsum_c64_borrowed-fresh_n2_strided_single | 19994.44 | 1047.56 | ok |
| einsum_c64_borrowed-shared_n2_col_major_contiguous_single | 10195.02 | 239.98 | ok |
| einsum_c64_borrowed-shared_n2_row_major_contiguous_single | 10178.59 | 241.82 | ok |
| einsum_c64_borrowed-shared_n2_strided_single | 14454.88 | 335.68 | ok |
| einsum_c64_borrowed-fresh_n4_col_major_contiguous_single | 16383.20 | 1057.57 | ok |
| einsum_c64_borrowed-fresh_n4_row_major_contiguous_single | 18836.14 | 845.35 | ok |
| einsum_c64_borrowed-fresh_n4_strided_single | 19592.22 | 1083.70 | ok |
| einsum_c64_borrowed-shared_n4_col_major_contiguous_single | 10792.35 | 95.16 | ok |
| einsum_c64_borrowed-shared_n4_row_major_contiguous_single | 10670.18 | 188.97 | ok |
| einsum_c64_borrowed-shared_n4_strided_single | 14385.04 | 113.76 | ok |
| einsum_c64_borrowed-fresh_n16_col_major_contiguous_single | 17609.91 | 1060.63 | ok |
| einsum_c64_borrowed-fresh_n16_row_major_contiguous_single | 17955.02 | 846.83 | ok |
| einsum_c64_borrowed-fresh_n16_strided_single | 23411.85 | 740.22 | ok |
| einsum_c64_borrowed-shared_n16_col_major_contiguous_single | 12442.19 | 554.54 | ok |
| einsum_c64_borrowed-shared_n16_row_major_contiguous_single | 12841.42 | 74.06 | ok |
| einsum_c64_borrowed-shared_n16_strided_single | 17157.39 | 382.49 | ok |
| einsum_c64_borrowed-fresh_n2_broadcast_single | 19798.58 | 3986.66 | NOISY |
| einsum_c64_borrowed-shared_n2_broadcast_single | 14374.61 | 62.87 | ok |
| einsum_c64_borrowed-fresh_n4_broadcast_single | 19486.94 | 537.26 | ok |
| einsum_c64_borrowed-shared_n4_broadcast_single | 14528.33 | 36.87 | ok |
| einsum_c64_borrowed-fresh_n16_broadcast_single | 23652.98 | 915.67 | ok |
| einsum_c64_borrowed-shared_n16_broadcast_single | 17320.62 | 50.57 | ok |
| reduce_sum_f64_concrete-fresh_size4_single | 4497.26 | 42.90 | ok |
| reduce_sum_f64_concrete-fresh_size16_single | 5814.35 | 1630.10 | NOISY |
| reduce_sum_f64_concrete-fresh_size256_single | 5000.95 | 507.69 | NOISY |
| einsum_f64_concrete-shared_abcd-dbef-acef_d4_single | 20344.63 | 112.44 | ok |
| einsum_f64_concrete-shared_ax-asb-xsb_d4s2_single | 10886.95 | 84.99 | ok |
| einsum_f64_concrete-shared_ij-jk-ik_m95k95n1_single | 10759.77 | 135.39 | ok |

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
- **einsum / concrete-fresh** — inside: session_entry_exit, einsum, output_allocation; outside: backend_construction, input_construction, correctness_check, output_destruction, operation_config_construction.
- **einsum / concrete-shared** — inside: einsum, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **einsum / dot-general-into-shared** — inside: dot_general_read_into; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, dot_general_config_construction, output_preallocation, output_destruction, operation_config_construction.
- **einsum / eager-ad** — inside: einsum_forward_recording, eager_dispatch, output_allocation; outside: backend_construction, input_construction, eager_runtime_construction, backward, correctness_check, output_destruction, operation_config_construction.
- **einsum / eager-no-ad** — inside: einsum, eager_dispatch, output_allocation; outside: backend_construction, input_construction, eager_runtime_construction, correctness_check, output_destruction, operation_config_construction.
- **einsum / prepared-into-repeat** — inside: prepared_execute_into; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, plan_preparation, output_preallocation, output_destruction, operation_config_construction.
- **einsum / prepared-repeat** — inside: prepared_execute, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, plan_preparation, output_destruction, operation_config_construction.
- **einsum / prepared-setup** — inside: prepare, plan_allocation; outside: backend_construction, input_construction, correctness_check, output_destruction, operation_config_construction.
- **gather / concrete-fresh** — inside: session_entry_exit, gather, output_allocation; outside: backend_construction, input_construction, correctness_check, output_destruction, operation_config_construction.
- **gather / concrete-shared** — inside: gather, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **reduce_sum / concrete-fresh** — inside: session_entry_exit, reduce_sum, output_allocation; outside: backend_construction, input_construction, correctness_check, output_destruction, operation_config_construction.
- **reduce_sum / concrete-shared** — inside: reduce_sum, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.
- **solve / concrete-fresh** — inside: session_entry_exit, solve, output_allocation; outside: backend_construction, input_construction, correctness_check, output_destruction, operation_config_construction.
- **solve / concrete-shared** — inside: solve, output_allocation; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check, output_destruction, operation_config_construction.

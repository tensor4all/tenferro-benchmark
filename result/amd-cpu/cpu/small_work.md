# CPU Small-Work Benchmark Results

- Suite: `cpu/small_work`
- Raw run: `data/results/amd-cpu/cpu/small_work/20260906_060238`

## CPU Information

- Model: `AMD EPYC 7713P 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `64`
- Sockets: `1`
- Cores per socket: `64`
- Threads per core: `1`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-101-generic-x86_64-with-glibc2.39`

## Measurement notes

Collected in a newly built Linux CPU devcontainer on the shared AMD EPYC host.
The 1-thread collection completed before the 4-thread collection began.
No CPU selection, affinity settings, idle-host gate, or performance acceptance
threshold was added. Other host activity was not controlled; no cause is
assigned to the observed variation.

All 154 cases passed numerical checks at each thread count. There are 51/154
NOISY rows at 1 thread and 117/154 at 4 threads (CoV > 10%); maximum CoV is
83.4% and 120.6%, respectively. These measurements and all 4,620 raw timing
samples are retained as collected. Do not infer speedups or thread scaling
from these noisy rows. No rerun was requested merely to obtain quieter data.

The measured benchmark revision is `d5faef9` and tenferro-rs is `181cbadf`.
The later report-only commit adds these notes and the generated results;
it does not change the case execution or timing code.

Collection command, run from the benchmark worktree on the Linux host:

```bash
npx --yes @devcontainers/cli exec --workspace-folder . \
  --remote-env BENCHMARK_COMMIT="$(git rev-parse HEAD)" bash -lc '
  export CARGO_HOME=/tmp/benchmark95-cargo UV_CACHE_DIR=/tmp/benchmark95-uv
  TENFERRO_CPU_FEATURES=system-mkl TENFERRO_CPU_BACKEND_KIND=blas \
  BENCHMARK_TARGET_PROFILE=amd-cpu ./scripts/run_small_work.sh 1 4'
```

The temporary cache paths avoid permission conflicts in existing shared Docker
cache volumes; those shared volumes were not modified to fix permissions.

## run_t1

Full metadata: `data/results/amd-cpu/cpu/small_work/20260906_060238/run_t1.yaml`

```yaml
timestamp: '2026-09-06T06:02:38Z'
tenferro_rs:
  path: /workspaces/benchmark-95-suite/extern/tenferro-rs
  commit: 181cbadf050a3908ca6dfee821ce3c213e2c66f9
  dirty: false
  features:
  - system-mkl
blas:
  implementation: mkl
  version: 2026.0.1
  root: /opt/intel/oneapi/mkl/latest
  library: /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so
environment:
  hostname: 082181a7ebbf
  os: Linux-6.8.0-101-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  env:
    OMP_NUM_THREADS: '1'
    OMP_THREAD_LIMIT: '1'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '1'
    TENFERRO_CPU_BACKEND_KIND: blas
    BENCHMARK_TARGET_PROFILE: amd-cpu
    OPENBLAS_ROOT: /opt/openblas
    MKLROOT: /opt/intel/oneapi/mkl/latest
    OPENBLAS_NUM_THREADS: '1'
    GOTO_NUM_THREADS: '1'
    MKL_NUM_THREADS: '1'
    VECLIB_MAXIMUM_THREADS: '1'
    VECLIB_NUM_THREADS: '1'
    NUMEXPR_NUM_THREADS: '1'
    BLIS_NUM_THREADS: '1'
    XLA_FLAGS: --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1
    JULIA_NUM_THREADS: '1'
    USE_CUDA: '0'
    BENCHMARK_COMMIT: d5faef953daaa4c1a1fcd956fa811fe538fdeff8
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.98.1 (48a229cea 2026-09-01)
    BENCH_INSTANCE: all 154
```

## run_t4

Full metadata: `data/results/amd-cpu/cpu/small_work/20260906_060238/run_t4.yaml`

```yaml
timestamp: '2026-09-06T06:02:49Z'
tenferro_rs:
  path: /workspaces/benchmark-95-suite/extern/tenferro-rs
  commit: 181cbadf050a3908ca6dfee821ce3c213e2c66f9
  dirty: false
  features:
  - system-mkl
blas:
  implementation: mkl
  version: 2026.0.1
  root: /opt/intel/oneapi/mkl/latest
  library: /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so
environment:
  hostname: 082181a7ebbf
  os: Linux-6.8.0-101-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  env:
    OMP_NUM_THREADS: '4'
    OMP_THREAD_LIMIT: '4'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '4'
    TENFERRO_CPU_BACKEND_KIND: blas
    BENCHMARK_TARGET_PROFILE: amd-cpu
    OPENBLAS_ROOT: /opt/openblas
    MKLROOT: /opt/intel/oneapi/mkl/latest
    OPENBLAS_NUM_THREADS: '4'
    GOTO_NUM_THREADS: '4'
    MKL_NUM_THREADS: '4'
    VECLIB_MAXIMUM_THREADS: '4'
    VECLIB_NUM_THREADS: '4'
    NUMEXPR_NUM_THREADS: '4'
    BLIS_NUM_THREADS: '4'
    XLA_FLAGS: --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4
    JULIA_NUM_THREADS: '4'
    USE_CUDA: '0'
    BENCHMARK_COMMIT: d5faef953daaa4c1a1fcd956fa811fe538fdeff8
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.98.1 (48a229cea 2026-09-01)
    BENCH_INSTANCE: all 154
```

Numerical checks: **308/308 recorded rows passed** (case × thread count).

## Timing

Sequential release runs; 3 warmups, then calibration to a 1 ms batch and 15 measured batches per case. Statistics describe batch elapsed time / iterations, not individually timed calls. Median and inclusive IQR are in nanoseconds (ns); CoV is sample standard deviation / mean. NOISY means CoV > 10%, a descriptive label only. No noisy rows are excluded.

Workflow time is the total for one call or the complete dependent chain. The separate ns/op column divides the chain median by its operation count, not an independently measured call. Setup rows measure preparation only, not execution.

Inputs and backend construction, independent numerical checks and AD backward checks are outside timing. Returned outputs/plans are consumed with black_box and dropped inside timing; value downloads/checks are outside. Eager AD rows measure forward execution/recording, not backward.

| Case ID | Operation | API route | Dtype | Shape | Layout | Workflow | Threads | Median workflow ns | IQR ns | CoV | ns/op (chain only) | Status |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| add_f64_concrete_fresh | add | concrete-fresh | f64 | 4 | col_major_contiguous | single | 1 | 14865.20 | 5359.07 | 43.6% | — | NOISY |
| add_f64_concrete-fresh_size4_dependent10 | add | concrete-fresh | f64 | 4 | col_major_contiguous | dependent10 | 1 | 138817.12 | 15264.19 | 52.3% | 13881.71 | NOISY |
| add_f64_concrete-fresh_size16_single | add | concrete-fresh | f64 | 16 | col_major_contiguous | single | 1 | 11084.49 | 2378.80 | 75.2% | — | NOISY |
| add_f64_concrete-fresh_size16_dependent10 | add | concrete-fresh | f64 | 16 | col_major_contiguous | dependent10 | 1 | 154018.88 | 127890.62 | 71.8% | 15401.89 | NOISY |
| add_f64_concrete-fresh_size256_single | add | concrete-fresh | f64 | 256 | col_major_contiguous | single | 1 | 20983.80 | 10108.30 | 43.2% | — | NOISY |
| add_f64_concrete-fresh_size256_dependent10 | add | concrete-fresh | f64 | 256 | col_major_contiguous | dependent10 | 1 | 167961.62 | 107964.50 | 54.7% | 16796.16 | NOISY |
| add_f64_concrete-shared_size4_single | add | concrete-shared | f64 | 4 | col_major_contiguous | single | 1 | 15171.70 | 11456.53 | 35.0% | — | NOISY |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | f64 | 4 | col_major_contiguous | dependent10 | 1 | 137097.12 | 18874.25 | 51.8% | 13709.71 | NOISY |
| add_f64_concrete_shared | add | concrete-shared | f64 | 16 | col_major_contiguous | single | 1 | 12562.11 | 10403.11 | 45.2% | — | NOISY |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | f64 | 16 | col_major_contiguous | dependent10 | 1 | 150071.00 | 22804.38 | 66.1% | 15007.10 | NOISY |
| add_f64_concrete-shared_size256_single | add | concrete-shared | f64 | 256 | col_major_contiguous | single | 1 | 17921.15 | 1491.48 | 15.5% | — | NOISY |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | f64 | 256 | col_major_contiguous | dependent10 | 1 | 234904.50 | 134265.75 | 35.1% | 23490.45 | NOISY |
| add_f64_eager-no-ad_size4_single | add | eager-no-ad | f64 | 4 | col_major_contiguous | single | 1 | 38981.28 | 23834.95 | 33.9% | — | NOISY |
| add_f64_eager-no-ad_size4_dependent10 | add | eager-no-ad | f64 | 4 | col_major_contiguous | dependent10 | 1 | 470786.50 | 150363.75 | 33.3% | 47078.65 | NOISY |
| add_f64_eager_no_ad | add | eager-no-ad | f64 | 16 | col_major_contiguous | single | 1 | 25990.17 | 10526.20 | 36.1% | — | NOISY |
| add_f64_eager-no-ad_size16_dependent10 | add | eager-no-ad | f64 | 16 | col_major_contiguous | dependent10 | 1 | 537650.75 | 162912.75 | 25.6% | 53765.07 | NOISY |
| add_f64_eager-no-ad_size256_single | add | eager-no-ad | f64 | 256 | col_major_contiguous | single | 1 | 37174.69 | 255.66 | 4.4% | — | ok |
| add_f64_eager-no-ad_size256_dependent10 | add | eager-no-ad | f64 | 256 | col_major_contiguous | dependent10 | 1 | 528585.50 | 197096.12 | 21.3% | 52858.55 | NOISY |
| add_f64_eager_active_ad | add | eager-ad | f64 | 4 | col_major_contiguous | single | 1 | 20692.84 | 10474.88 | 28.7% | — | NOISY |
| add_f64_eager-ad_size4_dependent10 | add | eager-ad | f64 | 4 | col_major_contiguous | dependent10 | 1 | 244493.50 | 100181.81 | 25.2% | 24449.35 | NOISY |
| add_f64_eager-ad_size16_single | add | eager-ad | f64 | 16 | col_major_contiguous | single | 1 | 20528.94 | 1816.77 | 62.6% | — | NOISY |
| add_f64_eager-ad_size16_dependent10 | add | eager-ad | f64 | 16 | col_major_contiguous | dependent10 | 1 | 288549.75 | 62052.88 | 17.0% | 28854.97 | NOISY |
| add_f64_eager-ad_size256_single | add | eager-ad | f64 | 256 | col_major_contiguous | single | 1 | 22123.06 | 1651.94 | 83.4% | — | NOISY |
| add_f64_eager-ad_size256_dependent10 | add | eager-ad | f64 | 256 | col_major_contiguous | dependent10 | 1 | 163055.25 | 10390.19 | 6.3% | 16305.52 | ok |
| einsum_f64_concrete-fresh_n2_single | einsum | concrete-fresh | f64 | 2×2 | col_major_contiguous | single | 1 | 46630.53 | 11740.59 | 18.6% | — | NOISY |
| einsum_f64_concrete-fresh_n4_single | einsum | concrete-fresh | f64 | 4×4 | col_major_contiguous | single | 1 | 49669.66 | 10788.72 | 25.7% | — | NOISY |
| einsum_f64_concrete-fresh_n16_single | einsum | concrete-fresh | f64 | 16×16 | col_major_contiguous | single | 1 | 51486.25 | 26420.62 | 63.5% | — | NOISY |
| einsum_f64_concrete-shared_n2_single | einsum | concrete-shared | f64 | 2×2 | col_major_contiguous | single | 1 | 42796.69 | 11682.33 | 14.6% | — | NOISY |
| einsum_f64_concrete-shared_n4_single | einsum | concrete-shared | f64 | 4×4 | col_major_contiguous | single | 1 | 55016.31 | 18549.81 | 22.2% | — | NOISY |
| einsum_f64_concrete-shared_n16_single | einsum | concrete-shared | f64 | 16×16 | col_major_contiguous | single | 1 | 43163.25 | 2820.55 | 33.8% | — | NOISY |
| einsum_f64_eager-no-ad_n2_single | einsum | eager-no-ad | f64 | 2×2 | col_major_contiguous | single | 1 | 58165.50 | 24624.83 | 32.4% | — | NOISY |
| einsum_f64_eager-no-ad_n4_single | einsum | eager-no-ad | f64 | 4×4 | col_major_contiguous | single | 1 | 42467.91 | 328.77 | 0.6% | — | ok |
| einsum_f64_eager-no-ad_n16_single | einsum | eager-no-ad | f64 | 16×16 | col_major_contiguous | single | 1 | 45533.94 | 1608.47 | 26.1% | — | NOISY |
| einsum_f64_eager-ad_n2_single | einsum | eager-ad | f64 | 2×2 | col_major_contiguous | single | 1 | 31041.06 | 9975.72 | 47.2% | — | NOISY |
| einsum_f64_eager-ad_n4_single | einsum | eager-ad | f64 | 4×4 | col_major_contiguous | single | 1 | 31572.97 | 7729.41 | 16.8% | — | NOISY |
| einsum_f64_eager-ad_n16_single | einsum | eager-ad | f64 | 16×16 | col_major_contiguous | single | 1 | 41361.34 | 3117.11 | 19.3% | — | NOISY |
| einsum_f64_prepared-setup_n2_single | einsum | prepared-setup | f64 | 2×2 | col_major_contiguous | single | 1 | 12334.29 | 1757.31 | 19.1% | — | NOISY |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | f64 | 2×2 | col_major_contiguous | single | 1 | 16616.03 | 540.95 | 35.2% | — | NOISY |
| einsum_f64_prepared-setup_n4_single | einsum | prepared-setup | f64 | 4×4 | col_major_contiguous | single | 1 | 12551.09 | 1362.65 | 18.7% | — | NOISY |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | f64 | 4×4 | col_major_contiguous | single | 1 | 16889.48 | 991.97 | 31.5% | — | NOISY |
| einsum_f64_prepared-setup_n16_single | einsum | prepared-setup | f64 | 16×16 | col_major_contiguous | single | 1 | 12948.91 | 1730.36 | 25.5% | — | NOISY |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | f64 | 16×16 | col_major_contiguous | single | 1 | 19192.98 | 8365.76 | 38.2% | — | NOISY |
| einsum_f64_borrowed-fresh_n2_col_major_contiguous_single | einsum | borrowed-fresh | f64 | 2×2 | col_major_contiguous | single | 1 | 50789.38 | 17331.84 | 39.6% | — | NOISY |
| einsum_f64_borrowed-fresh_n2_row_major_contiguous_single | einsum | borrowed-fresh | f64 | 2×2 | row_major_contiguous | single | 1 | 44650.44 | 9212.72 | 43.8% | — | NOISY |
| einsum_f64_borrowed-fresh_n2_strided_single | einsum | borrowed-fresh | f64 | 2×2 | strided | single | 1 | 71356.44 | 20658.02 | 23.1% | — | NOISY |
| einsum_f64_borrowed-shared_n2_col_major_contiguous_single | einsum | borrowed-shared | f64 | 2×2 | col_major_contiguous | single | 1 | 41517.28 | 19160.50 | 43.3% | — | NOISY |
| einsum_f64_borrowed-shared_n2_row_major_contiguous_single | einsum | borrowed-shared | f64 | 2×2 | row_major_contiguous | single | 1 | 58155.81 | 10403.69 | 14.8% | — | NOISY |
| einsum_f64_borrowed-shared_n2_strided_single | einsum | borrowed-shared | f64 | 2×2 | strided | single | 1 | 51926.00 | 2732.25 | 9.5% | — | ok |
| einsum_f64_borrowed-fresh_n4_col_major_contiguous_single | einsum | borrowed-fresh | f64 | 4×4 | col_major_contiguous | single | 1 | 43887.66 | 402.97 | 0.6% | — | ok |
| einsum_f64_borrowed-fresh_n4_row_major_contiguous_single | einsum | borrowed-fresh | f64 | 4×4 | row_major_contiguous | single | 1 | 44584.84 | 317.95 | 2.4% | — | ok |
| einsum_f64_borrowed-fresh_n4_strided_single | einsum | borrowed-fresh | f64 | 4×4 | strided | single | 1 | 54771.34 | 407.95 | 0.5% | — | ok |
| einsum_f64_borrowed-shared_n4_col_major_contiguous_single | einsum | borrowed-shared | f64 | 4×4 | col_major_contiguous | single | 1 | 42763.53 | 540.97 | 1.0% | — | ok |
| einsum_f64_borrowed-shared_n4_row_major_contiguous_single | einsum | borrowed-shared | f64 | 4×4 | row_major_contiguous | single | 1 | 42454.78 | 252.69 | 0.9% | — | ok |
| einsum_f64_borrowed-shared_n4_strided_single | einsum | borrowed-shared | f64 | 4×4 | strided | single | 1 | 54663.22 | 562.36 | 1.1% | — | ok |
| einsum_f64_borrowed-fresh_n16_col_major_contiguous_single | einsum | borrowed-fresh | f64 | 16×16 | col_major_contiguous | single | 1 | 46471.12 | 355.31 | 0.5% | — | ok |
| einsum_f64_borrowed-fresh_n16_row_major_contiguous_single | einsum | borrowed-fresh | f64 | 16×16 | row_major_contiguous | single | 1 | 45608.94 | 6849.23 | 8.7% | — | ok |
| einsum_f64_borrowed-fresh_n16_strided_single | einsum | borrowed-fresh | f64 | 16×16 | strided | single | 1 | 58299.88 | 2274.27 | 2.7% | — | ok |
| einsum_f64_borrowed-shared_n16_col_major_contiguous_single | einsum | borrowed-shared | f64 | 16×16 | col_major_contiguous | single | 1 | 44432.97 | 560.78 | 0.8% | — | ok |
| einsum_f64_borrowed-shared_n16_row_major_contiguous_single | einsum | borrowed-shared | f64 | 16×16 | row_major_contiguous | single | 1 | 37716.88 | 245.62 | 0.6% | — | ok |
| einsum_f64_borrowed-shared_n16_strided_single | einsum | borrowed-shared | f64 | 16×16 | strided | single | 1 | 57002.34 | 946.44 | 1.3% | — | ok |
| einsum_f64_compiled-repeat_n2_single | einsum | compiled-repeat | f64 | 2×2 | col_major_contiguous | single | 1 | 64641.00 | 1065.91 | 1.2% | — | ok |
| einsum_f64_compiled-repeat_n4_single | einsum | compiled-repeat | f64 | 4×4 | col_major_contiguous | single | 1 | 70039.19 | 237.84 | 0.8% | — | ok |
| einsum_f64_compiled-repeat_n16_single | einsum | compiled-repeat | f64 | 16×16 | col_major_contiguous | single | 1 | 70363.62 | 1079.44 | 3.0% | — | ok |
| einsum_c64_concrete-fresh_n2_single | einsum | concrete-fresh | c64 | 2×2 | col_major_contiguous | single | 1 | 34842.09 | 400.48 | 1.0% | — | ok |
| einsum_c64_concrete-shared_n2_single | einsum | concrete-shared | c64 | 2×2 | col_major_contiguous | single | 1 | 40451.62 | 905.19 | 1.6% | — | ok |
| einsum_c64_concrete-fresh_n4_single | einsum | concrete-fresh | c64 | 4×4 | col_major_contiguous | single | 1 | 41964.47 | 1875.36 | 3.4% | — | ok |
| einsum_c64_concrete-shared_n4_single | einsum | concrete-shared | c64 | 4×4 | col_major_contiguous | single | 1 | 41152.59 | 454.84 | 0.9% | — | ok |
| einsum_c64_concrete-fresh_n16_single | einsum | concrete-fresh | c64 | 16×16 | col_major_contiguous | single | 1 | 45588.59 | 322.03 | 0.7% | — | ok |
| einsum_c64_concrete-shared_n16_single | einsum | concrete-shared | c64 | 16×16 | col_major_contiguous | single | 1 | 47080.25 | 291.56 | 2.1% | — | ok |
| einsum_c64_prepared-setup_n2_single | einsum | prepared-setup | c64 | 2×2 | col_major_contiguous | single | 1 | 12315.15 | 88.99 | 0.5% | — | ok |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | c64 | 2×2 | col_major_contiguous | single | 1 | 16312.75 | 128.05 | 0.9% | — | ok |
| einsum_c64_prepared-setup_n4_single | einsum | prepared-setup | c64 | 4×4 | col_major_contiguous | single | 1 | 12435.54 | 61.17 | 0.6% | — | ok |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | c64 | 4×4 | col_major_contiguous | single | 1 | 16450.56 | 107.27 | 0.8% | — | ok |
| einsum_c64_prepared-setup_n16_single | einsum | prepared-setup | c64 | 16×16 | col_major_contiguous | single | 1 | 12465.70 | 93.05 | 44.0% | — | NOISY |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | c64 | 16×16 | col_major_contiguous | single | 1 | 20325.66 | 421.43 | 4.4% | — | ok |
| einsum_c64_eager-no-ad_n2_single | einsum | eager-no-ad | c64 | 2×2 | col_major_contiguous | single | 1 | 30872.62 | 836.44 | 2.5% | — | ok |
| einsum_c64_compiled-repeat_n2_single | einsum | compiled-repeat | c64 | 2×2 | col_major_contiguous | single | 1 | 69254.19 | 696.84 | 0.8% | — | ok |
| einsum_c64_eager-no-ad_n4_single | einsum | eager-no-ad | c64 | 4×4 | col_major_contiguous | single | 1 | 33169.28 | 1222.98 | 4.5% | — | ok |
| einsum_c64_compiled-repeat_n4_single | einsum | compiled-repeat | c64 | 4×4 | col_major_contiguous | single | 1 | 71991.12 | 698.69 | 0.7% | — | ok |
| einsum_c64_eager-no-ad_n16_single | einsum | eager-no-ad | c64 | 16×16 | col_major_contiguous | single | 1 | 45789.25 | 561.11 | 0.8% | — | ok |
| einsum_c64_compiled-repeat_n16_single | einsum | compiled-repeat | c64 | 16×16 | col_major_contiguous | single | 1 | 64792.25 | 502.19 | 0.7% | — | ok |
| einsum_c64_eager-ad_n2_single | einsum | eager-ad | c64 | 2×2 | col_major_contiguous | single | 1 | 31803.28 | 4752.30 | 9.3% | — | ok |
| einsum_c64_eager-ad_n4_single | einsum | eager-ad | c64 | 4×4 | col_major_contiguous | single | 1 | 38324.06 | 945.33 | 2.6% | — | ok |
| einsum_c64_eager-ad_n16_single | einsum | eager-ad | c64 | 16×16 | col_major_contiguous | single | 1 | 39213.47 | 1539.41 | 4.3% | — | ok |
| einsum_f64_compiled-setup_n2_single | einsum | compiled-setup | f64 | 2×2 | col_major_contiguous | single | 1 | 5114.58 | 15.23 | 0.5% | — | ok |
| einsum_f64_compiled-setup_n4_single | einsum | compiled-setup | f64 | 4×4 | col_major_contiguous | single | 1 | 5024.23 | 49.65 | 2.6% | — | ok |
| einsum_f64_compiled-setup_n16_single | einsum | compiled-setup | f64 | 16×16 | col_major_contiguous | single | 1 | 4818.13 | 24.82 | 0.3% | — | ok |
| einsum_c64_compiled-setup_n2_single | einsum | compiled-setup | c64 | 2×2 | col_major_contiguous | single | 1 | 5087.16 | 33.63 | 1.3% | — | ok |
| einsum_c64_compiled-setup_n4_single | einsum | compiled-setup | c64 | 4×4 | col_major_contiguous | single | 1 | 4879.22 | 8.59 | 0.2% | — | ok |
| einsum_c64_compiled-setup_n16_single | einsum | compiled-setup | c64 | 16×16 | col_major_contiguous | single | 1 | 5096.84 | 12.48 | 0.3% | — | ok |
| solve_f64_concrete-fresh_n2_single | solve | concrete-fresh | f64 | 2×2 | col_major_contiguous | single | 1 | 10583.30 | 49.22 | 1.4% | — | ok |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | f64 | 2×2 | col_major_contiguous | single | 1 | 13032.20 | 93.48 | 0.4% | — | ok |
| solve_f64_concrete-fresh_n4_single | solve | concrete-fresh | f64 | 4×4 | col_major_contiguous | single | 1 | 12249.21 | 52.19 | 1.2% | — | ok |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | f64 | 4×4 | col_major_contiguous | single | 1 | 13395.48 | 186.96 | 4.5% | — | ok |
| solve_f64_concrete-fresh_n16_single | solve | concrete-fresh | f64 | 16×16 | col_major_contiguous | single | 1 | 16434.47 | 134.30 | 1.0% | — | ok |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | f64 | 16×16 | col_major_contiguous | single | 1 | 19466.73 | 81.55 | 1.1% | — | ok |
| einsum_f64_borrowed-fresh_n2_broadcast_single | einsum | borrowed-fresh | f64 | 2×2 | broadcast | single | 1 | 55423.88 | 282.36 | 0.7% | — | ok |
| einsum_f64_borrowed-shared_n2_broadcast_single | einsum | borrowed-shared | f64 | 2×2 | broadcast | single | 1 | 51384.41 | 328.12 | 0.9% | — | ok |
| einsum_f64_borrowed-fresh_n4_broadcast_single | einsum | borrowed-fresh | f64 | 4×4 | broadcast | single | 1 | 52601.94 | 1478.62 | 8.2% | — | ok |
| einsum_f64_borrowed-shared_n4_broadcast_single | einsum | borrowed-shared | f64 | 4×4 | broadcast | single | 1 | 44744.53 | 3629.59 | 9.7% | — | ok |
| einsum_f64_borrowed-fresh_n16_broadcast_single | einsum | borrowed-fresh | f64 | 16×16 | broadcast | single | 1 | 55704.50 | 490.02 | 0.7% | — | ok |
| einsum_f64_borrowed-shared_n16_broadcast_single | einsum | borrowed-shared | f64 | 16×16 | broadcast | single | 1 | 54375.72 | 1024.56 | 4.2% | — | ok |
| einsum_f64_concrete-fresh_n8_single | einsum | concrete-fresh | f64 | 8×8 | col_major_contiguous | single | 1 | 36928.41 | 590.48 | 1.1% | — | ok |
| einsum_f64_concrete-shared_n8_single | einsum | concrete-shared | f64 | 8×8 | col_major_contiguous | single | 1 | 42760.75 | 490.03 | 1.1% | — | ok |
| einsum_f64_prepared-setup_n8_single | einsum | prepared-setup | f64 | 8×8 | col_major_contiguous | single | 1 | 12291.71 | 465.29 | 21.2% | — | NOISY |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | f64 | 8×8 | col_major_contiguous | single | 1 | 17254.33 | 180.00 | 0.6% | — | ok |
| einsum_f64_concrete-fresh_n32_single | einsum | concrete-fresh | f64 | 32×32 | col_major_contiguous | single | 1 | 49004.97 | 2006.94 | 4.4% | — | ok |
| einsum_f64_concrete-shared_n32_single | einsum | concrete-shared | f64 | 32×32 | col_major_contiguous | single | 1 | 44771.72 | 5227.78 | 7.9% | — | ok |
| einsum_f64_prepared-setup_n32_single | einsum | prepared-setup | f64 | 32×32 | col_major_contiguous | single | 1 | 11416.53 | 151.64 | 3.1% | — | ok |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | f64 | 32×32 | col_major_contiguous | single | 1 | 22290.55 | 133.45 | 0.6% | — | ok |
| einsum_c64_concrete-fresh_n8_single | einsum | concrete-fresh | c64 | 8×8 | col_major_contiguous | single | 1 | 36085.91 | 612.50 | 3.6% | — | ok |
| einsum_c64_concrete-shared_n8_single | einsum | concrete-shared | c64 | 8×8 | col_major_contiguous | single | 1 | 43572.34 | 617.84 | 1.1% | — | ok |
| einsum_c64_prepared-setup_n8_single | einsum | prepared-setup | c64 | 8×8 | col_major_contiguous | single | 1 | 12174.68 | 66.48 | 0.7% | — | ok |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | c64 | 8×8 | col_major_contiguous | single | 1 | 17142.92 | 73.13 | 0.4% | — | ok |
| einsum_c64_concrete-fresh_n32_single | einsum | concrete-fresh | c64 | 32×32 | col_major_contiguous | single | 1 | 52636.94 | 152.06 | 0.5% | — | ok |
| einsum_c64_concrete-shared_n32_single | einsum | concrete-shared | c64 | 32×32 | col_major_contiguous | single | 1 | 54585.12 | 473.28 | 2.1% | — | ok |
| einsum_c64_prepared-setup_n32_single | einsum | prepared-setup | c64 | 32×32 | col_major_contiguous | single | 1 | 11241.91 | 66.99 | 0.6% | — | ok |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | c64 | 32×32 | col_major_contiguous | single | 1 | 26961.30 | 117.11 | 1.8% | — | ok |
| gather_f64_concrete_fresh | gather | concrete-fresh | f64 | 4 | col_major_contiguous | single | 1 | 12406.48 | 997.95 | 5.0% | — | ok |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | f64 | 4 | col_major_contiguous | single | 1 | 10490.65 | 75.35 | 0.8% | — | ok |
| gather_f64_concrete-fresh_size16_single | gather | concrete-fresh | f64 | 16 | col_major_contiguous | single | 1 | 12214.21 | 61.60 | 1.5% | — | ok |
| gather_f64_concrete_shared | gather | concrete-shared | f64 | 16 | col_major_contiguous | single | 1 | 11566.45 | 60.52 | 1.0% | — | ok |
| gather_f64_concrete-fresh_size256_single | gather | concrete-fresh | f64 | 256 | col_major_contiguous | single | 1 | 8407.47 | 143.05 | 5.0% | — | ok |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | f64 | 256 | col_major_contiguous | single | 1 | 11928.73 | 109.81 | 0.9% | — | ok |
| einsum_c64_borrowed-fresh_n2_col_major_contiguous_single | einsum | borrowed-fresh | c64 | 2×2 | col_major_contiguous | single | 1 | 44280.47 | 389.06 | 1.0% | — | ok |
| einsum_c64_borrowed-fresh_n2_row_major_contiguous_single | einsum | borrowed-fresh | c64 | 2×2 | row_major_contiguous | single | 1 | 47540.56 | 606.41 | 2.0% | — | ok |
| einsum_c64_borrowed-fresh_n2_strided_single | einsum | borrowed-fresh | c64 | 2×2 | strided | single | 1 | 45373.94 | 7237.34 | 7.9% | — | ok |
| einsum_c64_borrowed-shared_n2_col_major_contiguous_single | einsum | borrowed-shared | c64 | 2×2 | col_major_contiguous | single | 1 | 41815.72 | 437.98 | 1.0% | — | ok |
| einsum_c64_borrowed-shared_n2_row_major_contiguous_single | einsum | borrowed-shared | c64 | 2×2 | row_major_contiguous | single | 1 | 35279.00 | 142.97 | 4.3% | — | ok |
| einsum_c64_borrowed-shared_n2_strided_single | einsum | borrowed-shared | c64 | 2×2 | strided | single | 1 | 52045.34 | 392.05 | 0.5% | — | ok |
| einsum_c64_borrowed-fresh_n4_col_major_contiguous_single | einsum | borrowed-fresh | c64 | 4×4 | col_major_contiguous | single | 1 | 38964.41 | 7235.67 | 8.9% | — | ok |
| einsum_c64_borrowed-fresh_n4_row_major_contiguous_single | einsum | borrowed-fresh | c64 | 4×4 | row_major_contiguous | single | 1 | 47020.84 | 433.30 | 4.3% | — | ok |
| einsum_c64_borrowed-fresh_n4_strided_single | einsum | borrowed-fresh | c64 | 4×4 | strided | single | 1 | 53928.84 | 439.55 | 0.5% | — | ok |
| einsum_c64_borrowed-shared_n4_col_major_contiguous_single | einsum | borrowed-shared | c64 | 4×4 | col_major_contiguous | single | 1 | 43327.62 | 301.12 | 0.5% | — | ok |
| einsum_c64_borrowed-shared_n4_row_major_contiguous_single | einsum | borrowed-shared | c64 | 4×4 | row_major_contiguous | single | 1 | 42169.50 | 5517.78 | 8.9% | — | ok |
| einsum_c64_borrowed-shared_n4_strided_single | einsum | borrowed-shared | c64 | 4×4 | strided | single | 1 | 52295.03 | 321.58 | 0.5% | — | ok |
| einsum_c64_borrowed-fresh_n16_col_major_contiguous_single | einsum | borrowed-fresh | c64 | 16×16 | col_major_contiguous | single | 1 | 45884.25 | 255.48 | 0.8% | — | ok |
| einsum_c64_borrowed-fresh_n16_row_major_contiguous_single | einsum | borrowed-fresh | c64 | 16×16 | row_major_contiguous | single | 1 | 46873.03 | 7269.55 | 8.2% | — | ok |
| einsum_c64_borrowed-fresh_n16_strided_single | einsum | borrowed-fresh | c64 | 16×16 | strided | single | 1 | 58473.66 | 1420.84 | 21.3% | — | NOISY |
| einsum_c64_borrowed-shared_n16_col_major_contiguous_single | einsum | borrowed-shared | c64 | 16×16 | col_major_contiguous | single | 1 | 45282.69 | 2350.52 | 3.9% | — | ok |
| einsum_c64_borrowed-shared_n16_row_major_contiguous_single | einsum | borrowed-shared | c64 | 16×16 | row_major_contiguous | single | 1 | 45004.53 | 512.97 | 1.4% | — | ok |
| einsum_c64_borrowed-shared_n16_strided_single | einsum | borrowed-shared | c64 | 16×16 | strided | single | 1 | 56281.38 | 268.61 | 2.7% | — | ok |
| einsum_c64_borrowed-fresh_n2_broadcast_single | einsum | borrowed-fresh | c64 | 2×2 | broadcast | single | 1 | 55754.84 | 461.88 | 0.7% | — | ok |
| einsum_c64_borrowed-shared_n2_broadcast_single | einsum | borrowed-shared | c64 | 2×2 | broadcast | single | 1 | 51092.19 | 569.25 | 1.0% | — | ok |
| einsum_c64_borrowed-fresh_n4_broadcast_single | einsum | borrowed-fresh | c64 | 4×4 | broadcast | single | 1 | 52865.69 | 436.11 | 0.7% | — | ok |
| einsum_c64_borrowed-shared_n4_broadcast_single | einsum | borrowed-shared | c64 | 4×4 | broadcast | single | 1 | 54029.16 | 1608.80 | 2.4% | — | ok |
| einsum_c64_borrowed-fresh_n16_broadcast_single | einsum | borrowed-fresh | c64 | 16×16 | broadcast | single | 1 | 59881.19 | 2500.38 | 3.1% | — | ok |
| einsum_c64_borrowed-shared_n16_broadcast_single | einsum | borrowed-shared | c64 | 16×16 | broadcast | single | 1 | 55244.78 | 274.25 | 0.7% | — | ok |
| reduce_sum_f64_concrete-fresh_size4_single | reduce_sum | concrete-fresh | f64 | 4 | col_major_contiguous | single | 1 | 10285.48 | 104.34 | 1.2% | — | ok |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | f64 | 4 | col_major_contiguous | single | 1 | 6509.57 | 220.02 | 3.4% | — | ok |
| reduce_sum_f64_concrete-fresh_size16_single | reduce_sum | concrete-fresh | f64 | 16 | col_major_contiguous | single | 1 | 10367.60 | 6121.44 | 43.9% | — | NOISY |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | f64 | 16 | col_major_contiguous | single | 1 | 9977.59 | 897.75 | 48.9% | — | NOISY |
| reduce_sum_f64_concrete-fresh_size256_single | reduce_sum | concrete-fresh | f64 | 256 | col_major_contiguous | single | 1 | 17873.48 | 5556.70 | 22.0% | — | NOISY |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | f64 | 256 | col_major_contiguous | single | 1 | 9606.96 | 847.41 | 33.5% | — | NOISY |
| add_f64_concrete_fresh | add | concrete-fresh | f64 | 4 | col_major_contiguous | single | 4 | 15952.89 | 692.35 | 37.8% | — | NOISY |
| add_f64_concrete-fresh_size4_dependent10 | add | concrete-fresh | f64 | 4 | col_major_contiguous | dependent10 | 4 | 162514.00 | 4352.62 | 57.5% | 16251.40 | NOISY |
| add_f64_concrete-fresh_size16_single | add | concrete-fresh | f64 | 16 | col_major_contiguous | single | 4 | 14545.21 | 3389.22 | 43.8% | — | NOISY |
| add_f64_concrete-fresh_size16_dependent10 | add | concrete-fresh | f64 | 16 | col_major_contiguous | dependent10 | 4 | 148161.25 | 12172.75 | 5.6% | 14816.12 | ok |
| add_f64_concrete-fresh_size256_single | add | concrete-fresh | f64 | 256 | col_major_contiguous | single | 4 | 25758.45 | 6269.37 | 23.4% | — | NOISY |
| add_f64_concrete-fresh_size256_dependent10 | add | concrete-fresh | f64 | 256 | col_major_contiguous | dependent10 | 4 | 189544.75 | 5591.31 | 33.1% | 18954.47 | NOISY |
| add_f64_concrete-shared_size4_single | add | concrete-shared | f64 | 4 | col_major_contiguous | single | 4 | 19301.97 | 6946.00 | 26.6% | — | NOISY |
| add_f64_concrete-shared_size4_dependent10 | add | concrete-shared | f64 | 4 | col_major_contiguous | dependent10 | 4 | 194184.75 | 56137.75 | 45.3% | 19418.47 | NOISY |
| add_f64_concrete_shared | add | concrete-shared | f64 | 16 | col_major_contiguous | single | 4 | 22706.66 | 11433.18 | 78.2% | — | NOISY |
| add_f64_concrete-shared_size16_dependent10 | add | concrete-shared | f64 | 16 | col_major_contiguous | dependent10 | 4 | 140473.50 | 8323.00 | 8.5% | 14047.35 | ok |
| add_f64_concrete-shared_size256_single | add | concrete-shared | f64 | 256 | col_major_contiguous | single | 4 | 17900.44 | 7255.10 | 37.2% | — | NOISY |
| add_f64_concrete-shared_size256_dependent10 | add | concrete-shared | f64 | 256 | col_major_contiguous | dependent10 | 4 | 235109.62 | 101098.12 | 31.2% | 23510.96 | NOISY |
| add_f64_eager-no-ad_size4_single | add | eager-no-ad | f64 | 4 | col_major_contiguous | single | 4 | 44188.59 | 25073.91 | 43.8% | — | NOISY |
| add_f64_eager-no-ad_size4_dependent10 | add | eager-no-ad | f64 | 4 | col_major_contiguous | dependent10 | 4 | 473906.50 | 149876.00 | 26.0% | 47390.65 | NOISY |
| add_f64_eager_no_ad | add | eager-no-ad | f64 | 16 | col_major_contiguous | single | 4 | 41872.28 | 3991.98 | 36.2% | — | NOISY |
| add_f64_eager-no-ad_size16_dependent10 | add | eager-no-ad | f64 | 16 | col_major_contiguous | dependent10 | 4 | 402780.00 | 180560.62 | 43.0% | 40278.00 | NOISY |
| add_f64_eager-no-ad_size256_single | add | eager-no-ad | f64 | 256 | col_major_contiguous | single | 4 | 37674.06 | 1736.25 | 3.0% | — | ok |
| add_f64_eager-no-ad_size256_dependent10 | add | eager-no-ad | f64 | 256 | col_major_contiguous | dependent10 | 4 | 477392.00 | 249283.75 | 43.3% | 47739.20 | NOISY |
| add_f64_eager_active_ad | add | eager-ad | f64 | 4 | col_major_contiguous | single | 4 | 22121.48 | 19886.11 | 38.4% | — | NOISY |
| add_f64_eager-ad_size4_dependent10 | add | eager-ad | f64 | 4 | col_major_contiguous | dependent10 | 4 | 219869.25 | 9369.56 | 33.1% | 21986.92 | NOISY |
| add_f64_eager-ad_size16_single | add | eager-ad | f64 | 16 | col_major_contiguous | single | 4 | 22768.22 | 11085.68 | 26.3% | — | NOISY |
| add_f64_eager-ad_size16_dependent10 | add | eager-ad | f64 | 16 | col_major_contiguous | dependent10 | 4 | 225585.75 | 14017.88 | 38.4% | 22558.58 | NOISY |
| add_f64_eager-ad_size256_single | add | eager-ad | f64 | 256 | col_major_contiguous | single | 4 | 24316.53 | 2703.58 | 29.8% | — | NOISY |
| add_f64_eager-ad_size256_dependent10 | add | eager-ad | f64 | 256 | col_major_contiguous | dependent10 | 4 | 251158.62 | 175256.88 | 36.9% | 25115.86 | NOISY |
| einsum_f64_concrete-fresh_n2_single | einsum | concrete-fresh | f64 | 2×2 | col_major_contiguous | single | 4 | 47769.00 | 22740.09 | 34.6% | — | NOISY |
| einsum_f64_concrete-fresh_n4_single | einsum | concrete-fresh | f64 | 4×4 | col_major_contiguous | single | 4 | 46211.81 | 3681.97 | 44.2% | — | NOISY |
| einsum_f64_concrete-fresh_n16_single | einsum | concrete-fresh | f64 | 16×16 | col_major_contiguous | single | 4 | 66509.47 | 46789.92 | 40.8% | — | NOISY |
| einsum_f64_concrete-shared_n2_single | einsum | concrete-shared | f64 | 2×2 | col_major_contiguous | single | 4 | 64056.56 | 37664.38 | 47.9% | — | NOISY |
| einsum_f64_concrete-shared_n4_single | einsum | concrete-shared | f64 | 4×4 | col_major_contiguous | single | 4 | 44845.50 | 3927.91 | 39.0% | — | NOISY |
| einsum_f64_concrete-shared_n16_single | einsum | concrete-shared | f64 | 16×16 | col_major_contiguous | single | 4 | 46108.69 | 70955.50 | 65.3% | — | NOISY |
| einsum_f64_eager-no-ad_n2_single | einsum | eager-no-ad | f64 | 2×2 | col_major_contiguous | single | 4 | 47254.31 | 743.94 | 1.2% | — | ok |
| einsum_f64_eager-no-ad_n4_single | einsum | eager-no-ad | f64 | 4×4 | col_major_contiguous | single | 4 | 44643.91 | 2962.73 | 4.7% | — | ok |
| einsum_f64_eager-no-ad_n16_single | einsum | eager-no-ad | f64 | 16×16 | col_major_contiguous | single | 4 | 44050.78 | 1244.88 | 3.7% | — | ok |
| einsum_f64_eager-ad_n2_single | einsum | eager-ad | f64 | 2×2 | col_major_contiguous | single | 4 | 29567.31 | 6635.00 | 13.2% | — | NOISY |
| einsum_f64_eager-ad_n4_single | einsum | eager-ad | f64 | 4×4 | col_major_contiguous | single | 4 | 29175.88 | 1655.12 | 4.2% | — | ok |
| einsum_f64_eager-ad_n16_single | einsum | eager-ad | f64 | 16×16 | col_major_contiguous | single | 4 | 34397.89 | 6228.59 | 9.9% | — | ok |
| einsum_f64_prepared-setup_n2_single | einsum | prepared-setup | f64 | 2×2 | col_major_contiguous | single | 4 | 12341.09 | 84.41 | 0.6% | — | ok |
| einsum_f64_prepared-repeat_n2_single | einsum | prepared-repeat | f64 | 2×2 | col_major_contiguous | single | 4 | 19062.81 | 304.07 | 1.2% | — | ok |
| einsum_f64_prepared-setup_n4_single | einsum | prepared-setup | f64 | 4×4 | col_major_contiguous | single | 4 | 12403.59 | 128.12 | 0.7% | — | ok |
| einsum_f64_prepared-repeat_n4_single | einsum | prepared-repeat | f64 | 4×4 | col_major_contiguous | single | 4 | 16010.23 | 679.07 | 3.0% | — | ok |
| einsum_f64_prepared-setup_n16_single | einsum | prepared-setup | f64 | 16×16 | col_major_contiguous | single | 4 | 14208.32 | 2839.99 | 16.0% | — | NOISY |
| einsum_f64_prepared-repeat_n16_single | einsum | prepared-repeat | f64 | 16×16 | col_major_contiguous | single | 4 | 18900.62 | 309.53 | 4.5% | — | ok |
| einsum_f64_borrowed-fresh_n2_col_major_contiguous_single | einsum | borrowed-fresh | f64 | 2×2 | col_major_contiguous | single | 4 | 44797.38 | 2221.94 | 4.5% | — | ok |
| einsum_f64_borrowed-fresh_n2_row_major_contiguous_single | einsum | borrowed-fresh | f64 | 2×2 | row_major_contiguous | single | 4 | 46478.34 | 1526.59 | 3.2% | — | ok |
| einsum_f64_borrowed-fresh_n2_strided_single | einsum | borrowed-fresh | f64 | 2×2 | strided | single | 4 | 54659.00 | 2272.75 | 12.6% | — | NOISY |
| einsum_f64_borrowed-shared_n2_col_major_contiguous_single | einsum | borrowed-shared | f64 | 2×2 | col_major_contiguous | single | 4 | 39862.25 | 3242.50 | 19.9% | — | NOISY |
| einsum_f64_borrowed-shared_n2_row_major_contiguous_single | einsum | borrowed-shared | f64 | 2×2 | row_major_contiguous | single | 4 | 46776.12 | 2084.44 | 8.0% | — | ok |
| einsum_f64_borrowed-shared_n2_strided_single | einsum | borrowed-shared | f64 | 2×2 | strided | single | 4 | 59960.25 | 3546.19 | 7.2% | — | ok |
| einsum_f64_borrowed-fresh_n4_col_major_contiguous_single | einsum | borrowed-fresh | f64 | 4×4 | col_major_contiguous | single | 4 | 45318.62 | 1276.28 | 2.2% | — | ok |
| einsum_f64_borrowed-fresh_n4_row_major_contiguous_single | einsum | borrowed-fresh | f64 | 4×4 | row_major_contiguous | single | 4 | 45851.75 | 992.22 | 1.8% | — | ok |
| einsum_f64_borrowed-fresh_n4_strided_single | einsum | borrowed-fresh | f64 | 4×4 | strided | single | 4 | 61085.91 | 1068.00 | 3.1% | — | ok |
| einsum_f64_borrowed-shared_n4_col_major_contiguous_single | einsum | borrowed-shared | f64 | 4×4 | col_major_contiguous | single | 4 | 41486.00 | 11301.50 | 64.5% | — | NOISY |
| einsum_f64_borrowed-shared_n4_row_major_contiguous_single | einsum | borrowed-shared | f64 | 4×4 | row_major_contiguous | single | 4 | 42892.00 | 2550.22 | 4.5% | — | ok |
| einsum_f64_borrowed-shared_n4_strided_single | einsum | borrowed-shared | f64 | 4×4 | strided | single | 4 | 60920.25 | 2039.47 | 3.3% | — | ok |
| einsum_f64_borrowed-fresh_n16_col_major_contiguous_single | einsum | borrowed-fresh | f64 | 16×16 | col_major_contiguous | single | 4 | 47592.75 | 1972.70 | 2.4% | — | ok |
| einsum_f64_borrowed-fresh_n16_row_major_contiguous_single | einsum | borrowed-fresh | f64 | 16×16 | row_major_contiguous | single | 4 | 45607.06 | 1029.73 | 5.6% | — | ok |
| einsum_f64_borrowed-fresh_n16_strided_single | einsum | borrowed-fresh | f64 | 16×16 | strided | single | 4 | 62742.19 | 2143.19 | 4.2% | — | ok |
| einsum_f64_borrowed-shared_n16_col_major_contiguous_single | einsum | borrowed-shared | f64 | 16×16 | col_major_contiguous | single | 4 | 46308.03 | 935.83 | 1.6% | — | ok |
| einsum_f64_borrowed-shared_n16_row_major_contiguous_single | einsum | borrowed-shared | f64 | 16×16 | row_major_contiguous | single | 4 | 46425.56 | 1497.22 | 3.8% | — | ok |
| einsum_f64_borrowed-shared_n16_strided_single | einsum | borrowed-shared | f64 | 16×16 | strided | single | 4 | 54617.62 | 3278.36 | 12.2% | — | NOISY |
| einsum_f64_compiled-repeat_n2_single | einsum | compiled-repeat | f64 | 2×2 | col_major_contiguous | single | 4 | 69289.25 | 4656.44 | 4.4% | — | ok |
| einsum_f64_compiled-repeat_n4_single | einsum | compiled-repeat | f64 | 4×4 | col_major_contiguous | single | 4 | 71301.75 | 1831.28 | 2.6% | — | ok |
| einsum_f64_compiled-repeat_n16_single | einsum | compiled-repeat | f64 | 16×16 | col_major_contiguous | single | 4 | 75816.88 | 2895.09 | 2.5% | — | ok |
| einsum_c64_concrete-fresh_n2_single | einsum | concrete-fresh | c64 | 2×2 | col_major_contiguous | single | 4 | 44381.72 | 1610.52 | 2.2% | — | ok |
| einsum_c64_concrete-shared_n2_single | einsum | concrete-shared | c64 | 2×2 | col_major_contiguous | single | 4 | 43175.50 | 49349.06 | 75.3% | — | NOISY |
| einsum_c64_concrete-fresh_n4_single | einsum | concrete-fresh | c64 | 4×4 | col_major_contiguous | single | 4 | 45271.75 | 1323.12 | 6.2% | — | ok |
| einsum_c64_concrete-shared_n4_single | einsum | concrete-shared | c64 | 4×4 | col_major_contiguous | single | 4 | 38523.16 | 4361.20 | 40.5% | — | NOISY |
| einsum_c64_concrete-fresh_n16_single | einsum | concrete-fresh | c64 | 16×16 | col_major_contiguous | single | 4 | 52930.38 | 28612.61 | 37.2% | — | NOISY |
| einsum_c64_concrete-shared_n16_single | einsum | concrete-shared | c64 | 16×16 | col_major_contiguous | single | 4 | 46011.78 | 857.69 | 36.6% | — | NOISY |
| einsum_c64_prepared-setup_n2_single | einsum | prepared-setup | c64 | 2×2 | col_major_contiguous | single | 4 | 12284.37 | 525.91 | 28.5% | — | NOISY |
| einsum_c64_prepared-repeat_n2_single | einsum | prepared-repeat | c64 | 2×2 | col_major_contiguous | single | 4 | 16669.95 | 775.41 | 33.2% | — | NOISY |
| einsum_c64_prepared-setup_n4_single | einsum | prepared-setup | c64 | 4×4 | col_major_contiguous | single | 4 | 12373.83 | 77.46 | 22.5% | — | NOISY |
| einsum_c64_prepared-repeat_n4_single | einsum | prepared-repeat | c64 | 4×4 | col_major_contiguous | single | 4 | 20028.78 | 12767.20 | 51.7% | — | NOISY |
| einsum_c64_prepared-setup_n16_single | einsum | prepared-setup | c64 | 16×16 | col_major_contiguous | single | 4 | 11257.31 | 1711.09 | 36.9% | — | NOISY |
| einsum_c64_prepared-repeat_n16_single | einsum | prepared-repeat | c64 | 16×16 | col_major_contiguous | single | 4 | 24170.28 | 11413.80 | 45.4% | — | NOISY |
| einsum_c64_eager-no-ad_n2_single | einsum | eager-no-ad | c64 | 2×2 | col_major_contiguous | single | 4 | 48468.06 | 5714.84 | 57.1% | — | NOISY |
| einsum_c64_compiled-repeat_n2_single | einsum | compiled-repeat | c64 | 2×2 | col_major_contiguous | single | 4 | 74745.62 | 12901.31 | 43.6% | — | NOISY |
| einsum_c64_eager-no-ad_n4_single | einsum | eager-no-ad | c64 | 4×4 | col_major_contiguous | single | 4 | 44562.03 | 4843.58 | 44.8% | — | NOISY |
| einsum_c64_compiled-repeat_n4_single | einsum | compiled-repeat | c64 | 4×4 | col_major_contiguous | single | 4 | 77116.31 | 6280.78 | 35.5% | — | NOISY |
| einsum_c64_eager-no-ad_n16_single | einsum | eager-no-ad | c64 | 16×16 | col_major_contiguous | single | 4 | 52492.00 | 5904.50 | 65.2% | — | NOISY |
| einsum_c64_compiled-repeat_n16_single | einsum | compiled-repeat | c64 | 16×16 | col_major_contiguous | single | 4 | 88802.25 | 16693.81 | 83.4% | — | NOISY |
| einsum_c64_eager-ad_n2_single | einsum | eager-ad | c64 | 2×2 | col_major_contiguous | single | 4 | 31922.34 | 1265.50 | 3.3% | — | ok |
| einsum_c64_eager-ad_n4_single | einsum | eager-ad | c64 | 4×4 | col_major_contiguous | single | 4 | 32709.56 | 2332.84 | 60.1% | — | NOISY |
| einsum_c64_eager-ad_n16_single | einsum | eager-ad | c64 | 16×16 | col_major_contiguous | single | 4 | 38477.53 | 6628.28 | 40.6% | — | NOISY |
| einsum_f64_compiled-setup_n2_single | einsum | compiled-setup | f64 | 2×2 | col_major_contiguous | single | 4 | 5023.21 | 934.40 | 39.2% | — | NOISY |
| einsum_f64_compiled-setup_n4_single | einsum | compiled-setup | f64 | 4×4 | col_major_contiguous | single | 4 | 4958.44 | 1471.40 | 19.1% | — | NOISY |
| einsum_f64_compiled-setup_n16_single | einsum | compiled-setup | f64 | 16×16 | col_major_contiguous | single | 4 | 5036.41 | 14.22 | 28.8% | — | NOISY |
| einsum_c64_compiled-setup_n2_single | einsum | compiled-setup | c64 | 2×2 | col_major_contiguous | single | 4 | 5007.51 | 1500.37 | 17.7% | — | NOISY |
| einsum_c64_compiled-setup_n4_single | einsum | compiled-setup | c64 | 4×4 | col_major_contiguous | single | 4 | 5016.22 | 362.15 | 26.2% | — | NOISY |
| einsum_c64_compiled-setup_n16_single | einsum | compiled-setup | c64 | 16×16 | col_major_contiguous | single | 4 | 5107.82 | 1481.21 | 21.7% | — | NOISY |
| solve_f64_concrete-fresh_n2_single | solve | concrete-fresh | f64 | 2×2 | col_major_contiguous | single | 4 | 15909.38 | 4460.07 | 22.2% | — | NOISY |
| solve_f64_concrete-shared_n2_single | solve | concrete-shared | f64 | 2×2 | col_major_contiguous | single | 4 | 17413.72 | 3326.25 | 19.1% | — | NOISY |
| solve_f64_concrete-fresh_n4_single | solve | concrete-fresh | f64 | 4×4 | col_major_contiguous | single | 4 | 15098.11 | 2464.44 | 21.7% | — | NOISY |
| solve_f64_concrete-shared_n4_single | solve | concrete-shared | f64 | 4×4 | col_major_contiguous | single | 4 | 18414.67 | 8317.24 | 22.5% | — | NOISY |
| solve_f64_concrete-fresh_n16_single | solve | concrete-fresh | f64 | 16×16 | col_major_contiguous | single | 4 | 21833.67 | 3731.34 | 25.2% | — | NOISY |
| solve_f64_concrete-shared_n16_single | solve | concrete-shared | f64 | 16×16 | col_major_contiguous | single | 4 | 23896.06 | 1656.06 | 11.0% | — | NOISY |
| einsum_f64_borrowed-fresh_n2_broadcast_single | einsum | borrowed-fresh | f64 | 2×2 | broadcast | single | 4 | 54781.38 | 3836.66 | 42.6% | — | NOISY |
| einsum_f64_borrowed-shared_n2_broadcast_single | einsum | borrowed-shared | f64 | 2×2 | broadcast | single | 4 | 54963.88 | 18909.31 | 81.5% | — | NOISY |
| einsum_f64_borrowed-fresh_n4_broadcast_single | einsum | borrowed-fresh | f64 | 4×4 | broadcast | single | 4 | 73248.38 | 40075.20 | 33.8% | — | NOISY |
| einsum_f64_borrowed-shared_n4_broadcast_single | einsum | borrowed-shared | f64 | 4×4 | broadcast | single | 4 | 52799.44 | 24081.69 | 33.1% | — | NOISY |
| einsum_f64_borrowed-fresh_n16_broadcast_single | einsum | borrowed-fresh | f64 | 16×16 | broadcast | single | 4 | 74296.53 | 32414.72 | 32.4% | — | NOISY |
| einsum_f64_borrowed-shared_n16_broadcast_single | einsum | borrowed-shared | f64 | 16×16 | broadcast | single | 4 | 56574.00 | 4687.50 | 22.6% | — | NOISY |
| einsum_f64_concrete-fresh_n8_single | einsum | concrete-fresh | f64 | 8×8 | col_major_contiguous | single | 4 | 62666.56 | 21475.84 | 28.4% | — | NOISY |
| einsum_f64_concrete-shared_n8_single | einsum | concrete-shared | f64 | 8×8 | col_major_contiguous | single | 4 | 60314.94 | 28733.83 | 26.9% | — | NOISY |
| einsum_f64_prepared-setup_n8_single | einsum | prepared-setup | f64 | 8×8 | col_major_contiguous | single | 4 | 12341.01 | 1076.83 | 9.4% | — | ok |
| einsum_f64_prepared-repeat_n8_single | einsum | prepared-repeat | f64 | 8×8 | col_major_contiguous | single | 4 | 18739.22 | 6881.10 | 31.6% | — | NOISY |
| einsum_f64_concrete-fresh_n32_single | einsum | concrete-fresh | f64 | 32×32 | col_major_contiguous | single | 4 | 88733.44 | 13222.53 | 31.7% | — | NOISY |
| einsum_f64_concrete-shared_n32_single | einsum | concrete-shared | f64 | 32×32 | col_major_contiguous | single | 4 | 87654.75 | 4560.12 | 104.0% | — | NOISY |
| einsum_f64_prepared-setup_n32_single | einsum | prepared-setup | f64 | 32×32 | col_major_contiguous | single | 4 | 11022.39 | 1543.99 | 26.6% | — | NOISY |
| einsum_f64_prepared-repeat_n32_single | einsum | prepared-repeat | f64 | 32×32 | col_major_contiguous | single | 4 | 126156.88 | 25826.19 | 32.9% | — | NOISY |
| einsum_c64_concrete-fresh_n8_single | einsum | concrete-fresh | c64 | 8×8 | col_major_contiguous | single | 4 | 53346.31 | 25047.78 | 31.0% | — | NOISY |
| einsum_c64_concrete-shared_n8_single | einsum | concrete-shared | c64 | 8×8 | col_major_contiguous | single | 4 | 45861.00 | 1551.12 | 21.7% | — | NOISY |
| einsum_c64_prepared-setup_n8_single | einsum | prepared-setup | c64 | 8×8 | col_major_contiguous | single | 4 | 11280.05 | 1140.26 | 28.0% | — | NOISY |
| einsum_c64_prepared-repeat_n8_single | einsum | prepared-repeat | c64 | 8×8 | col_major_contiguous | single | 4 | 22106.16 | 5676.08 | 26.4% | — | NOISY |
| einsum_c64_concrete-fresh_n32_single | einsum | concrete-fresh | c64 | 32×32 | col_major_contiguous | single | 4 | 105577.62 | 33018.94 | 23.6% | — | NOISY |
| einsum_c64_concrete-shared_n32_single | einsum | concrete-shared | c64 | 32×32 | col_major_contiguous | single | 4 | 110632.75 | 15711.50 | 83.3% | — | NOISY |
| einsum_c64_prepared-setup_n32_single | einsum | prepared-setup | c64 | 32×32 | col_major_contiguous | single | 4 | 12308.12 | 424.70 | 21.4% | — | NOISY |
| einsum_c64_prepared-repeat_n32_single | einsum | prepared-repeat | c64 | 32×32 | col_major_contiguous | single | 4 | 81450.75 | 3044.78 | 29.5% | — | NOISY |
| gather_f64_concrete_fresh | gather | concrete-fresh | f64 | 4 | col_major_contiguous | single | 4 | 13069.78 | 382.32 | 30.3% | — | NOISY |
| gather_f64_concrete-shared_size4_single | gather | concrete-shared | f64 | 4 | col_major_contiguous | single | 4 | 12402.97 | 1166.98 | 49.7% | — | NOISY |
| gather_f64_concrete-fresh_size16_single | gather | concrete-fresh | f64 | 16 | col_major_contiguous | single | 4 | 13832.22 | 1674.12 | 37.6% | — | NOISY |
| gather_f64_concrete_shared | gather | concrete-shared | f64 | 16 | col_major_contiguous | single | 4 | 13564.88 | 179.46 | 39.5% | — | NOISY |
| gather_f64_concrete-fresh_size256_single | gather | concrete-fresh | f64 | 256 | col_major_contiguous | single | 4 | 14740.76 | 1104.36 | 43.2% | — | NOISY |
| gather_f64_concrete-shared_size256_single | gather | concrete-shared | f64 | 256 | col_major_contiguous | single | 4 | 14956.00 | 33192.16 | 65.6% | — | NOISY |
| einsum_c64_borrowed-fresh_n2_col_major_contiguous_single | einsum | borrowed-fresh | c64 | 2×2 | col_major_contiguous | single | 4 | 49525.59 | 7596.91 | 34.4% | — | NOISY |
| einsum_c64_borrowed-fresh_n2_row_major_contiguous_single | einsum | borrowed-fresh | c64 | 2×2 | row_major_contiguous | single | 4 | 47808.69 | 10549.62 | 72.9% | — | NOISY |
| einsum_c64_borrowed-fresh_n2_strided_single | einsum | borrowed-fresh | c64 | 2×2 | strided | single | 4 | 62051.56 | 49790.14 | 44.1% | — | NOISY |
| einsum_c64_borrowed-shared_n2_col_major_contiguous_single | einsum | borrowed-shared | c64 | 2×2 | col_major_contiguous | single | 4 | 52621.94 | 35876.22 | 45.6% | — | NOISY |
| einsum_c64_borrowed-shared_n2_row_major_contiguous_single | einsum | borrowed-shared | c64 | 2×2 | row_major_contiguous | single | 4 | 71041.75 | 43041.00 | 80.2% | — | NOISY |
| einsum_c64_borrowed-shared_n2_strided_single | einsum | borrowed-shared | c64 | 2×2 | strided | single | 4 | 47417.72 | 4106.81 | 64.7% | — | NOISY |
| einsum_c64_borrowed-fresh_n4_col_major_contiguous_single | einsum | borrowed-fresh | c64 | 4×4 | col_major_contiguous | single | 4 | 48596.50 | 1910.00 | 4.7% | — | ok |
| einsum_c64_borrowed-fresh_n4_row_major_contiguous_single | einsum | borrowed-fresh | c64 | 4×4 | row_major_contiguous | single | 4 | 52928.75 | 18146.88 | 115.8% | — | NOISY |
| einsum_c64_borrowed-fresh_n4_strided_single | einsum | borrowed-fresh | c64 | 4×4 | strided | single | 4 | 59172.41 | 7647.83 | 32.2% | — | NOISY |
| einsum_c64_borrowed-shared_n4_col_major_contiguous_single | einsum | borrowed-shared | c64 | 4×4 | col_major_contiguous | single | 4 | 45796.00 | 1102.75 | 4.0% | — | ok |
| einsum_c64_borrowed-shared_n4_row_major_contiguous_single | einsum | borrowed-shared | c64 | 4×4 | row_major_contiguous | single | 4 | 45374.56 | 8003.48 | 43.8% | — | NOISY |
| einsum_c64_borrowed-shared_n4_strided_single | einsum | borrowed-shared | c64 | 4×4 | strided | single | 4 | 56811.41 | 45708.31 | 32.9% | — | NOISY |
| einsum_c64_borrowed-fresh_n16_col_major_contiguous_single | einsum | borrowed-fresh | c64 | 16×16 | col_major_contiguous | single | 4 | 51784.41 | 33470.22 | 38.5% | — | NOISY |
| einsum_c64_borrowed-fresh_n16_row_major_contiguous_single | einsum | borrowed-fresh | c64 | 16×16 | row_major_contiguous | single | 4 | 50658.12 | 33204.91 | 58.6% | — | NOISY |
| einsum_c64_borrowed-fresh_n16_strided_single | einsum | borrowed-fresh | c64 | 16×16 | strided | single | 4 | 62002.00 | 7859.50 | 14.4% | — | NOISY |
| einsum_c64_borrowed-shared_n16_col_major_contiguous_single | einsum | borrowed-shared | c64 | 16×16 | col_major_contiguous | single | 4 | 51818.75 | 8156.50 | 120.6% | — | NOISY |
| einsum_c64_borrowed-shared_n16_row_major_contiguous_single | einsum | borrowed-shared | c64 | 16×16 | row_major_contiguous | single | 4 | 48778.72 | 15220.69 | 41.1% | — | NOISY |
| einsum_c64_borrowed-shared_n16_strided_single | einsum | borrowed-shared | c64 | 16×16 | strided | single | 4 | 57663.31 | 1259.69 | 98.8% | — | NOISY |
| einsum_c64_borrowed-fresh_n2_broadcast_single | einsum | borrowed-fresh | c64 | 2×2 | broadcast | single | 4 | 55836.09 | 10008.55 | 31.0% | — | NOISY |
| einsum_c64_borrowed-shared_n2_broadcast_single | einsum | borrowed-shared | c64 | 2×2 | broadcast | single | 4 | 56745.47 | 13123.14 | 34.5% | — | NOISY |
| einsum_c64_borrowed-fresh_n4_broadcast_single | einsum | borrowed-fresh | c64 | 4×4 | broadcast | single | 4 | 59076.50 | 8027.62 | 10.6% | — | NOISY |
| einsum_c64_borrowed-shared_n4_broadcast_single | einsum | borrowed-shared | c64 | 4×4 | broadcast | single | 4 | 56186.41 | 34906.33 | 43.2% | — | NOISY |
| einsum_c64_borrowed-fresh_n16_broadcast_single | einsum | borrowed-fresh | c64 | 16×16 | broadcast | single | 4 | 85720.88 | 18242.94 | 40.4% | — | NOISY |
| einsum_c64_borrowed-shared_n16_broadcast_single | einsum | borrowed-shared | c64 | 16×16 | broadcast | single | 4 | 56845.16 | 36896.09 | 29.1% | — | NOISY |
| reduce_sum_f64_concrete-fresh_size4_single | reduce_sum | concrete-fresh | f64 | 4 | col_major_contiguous | single | 4 | 11930.61 | 1539.64 | 43.6% | — | NOISY |
| reduce_sum_f64_concrete-shared_size4_single | reduce_sum | concrete-shared | f64 | 4 | col_major_contiguous | single | 4 | 12057.64 | 2528.04 | 10.9% | — | NOISY |
| reduce_sum_f64_concrete-fresh_size16_single | reduce_sum | concrete-fresh | f64 | 16 | col_major_contiguous | single | 4 | 12105.25 | 712.38 | 7.2% | — | ok |
| reduce_sum_f64_concrete-shared_size16_single | reduce_sum | concrete-shared | f64 | 16 | col_major_contiguous | single | 4 | 10666.36 | 2963.75 | 15.4% | — | NOISY |
| reduce_sum_f64_concrete-fresh_size256_single | reduce_sum | concrete-fresh | f64 | 256 | col_major_contiguous | single | 4 | 13571.83 | 3284.03 | 23.7% | — | NOISY |
| reduce_sum_f64_concrete-shared_size256_single | reduce_sum | concrete-shared | f64 | 256 | col_major_contiguous | single | 4 | 9430.39 | 1186.55 | 37.9% | — | NOISY |

## Per-route timing boundaries

Shared-session routes enter/exit the session once around warmup/calibration/all samples, outside timing. Fresh routes enter/exit for each operation. Prepared-repeat excludes plan preparation; compiled-repeat excludes tracing/compilation and runtime construction.

- **add / concrete-fresh** — inside: session_entry_exit, add, output_lifetime; outside: backend_construction, input_construction, correctness_check.
- **add / concrete-shared** — inside: add, output_lifetime; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check.
- **add / eager-ad** — inside: add_forward_recording, eager_dispatch, output_lifetime; outside: backend_construction, input_construction, eager_runtime_construction, backward, correctness_check.
- **add / eager-no-ad** — inside: add, eager_dispatch, output_lifetime; outside: backend_construction, input_construction, eager_runtime_construction, correctness_check.
- **einsum / borrowed-fresh** — inside: session_entry_exit, einsum, output_lifetime; outside: backend_construction, input_construction, correctness_check.
- **einsum / borrowed-shared** — inside: einsum, output_lifetime; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check.
- **einsum / compiled-repeat** — inside: runtime_admission_execution, output_lifetime; outside: backend_construction, input_construction, trace_compile, runtime_construction, input_bindings, correctness_check.
- **einsum / compiled-setup** — inside: trace_compile, program_lifetime; outside: backend_construction, input_construction, runtime_construction, input_bindings, correctness_check.
- **einsum / concrete-fresh** — inside: session_entry_exit, einsum, output_lifetime; outside: backend_construction, input_construction, correctness_check.
- **einsum / concrete-shared** — inside: einsum, output_lifetime; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check.
- **einsum / eager-ad** — inside: einsum_forward_recording, eager_dispatch, output_lifetime; outside: backend_construction, input_construction, eager_runtime_construction, backward, correctness_check.
- **einsum / eager-no-ad** — inside: einsum, eager_dispatch, output_lifetime; outside: backend_construction, input_construction, eager_runtime_construction, correctness_check.
- **einsum / prepared-repeat** — inside: prepared_execute, output_lifetime; outside: backend_construction, input_construction, plan_preparation, session_entry_exit, shared_admission_exit, correctness_check.
- **einsum / prepared-setup** — inside: prepare, plan_lifetime; outside: backend_construction, input_construction, correctness_check.
- **gather / concrete-fresh** — inside: index_config_construction, session_entry_exit, gather, output_lifetime; outside: backend_construction, input_construction, correctness_check.
- **gather / concrete-shared** — inside: index_config_construction, gather, output_lifetime; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check.
- **reduce_sum / concrete-fresh** — inside: session_entry_exit, reduce_sum, output_lifetime; outside: backend_construction, input_construction, correctness_check.
- **reduce_sum / concrete-shared** — inside: reduce_sum, output_lifetime; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check.
- **solve / concrete-fresh** — inside: session_entry_exit, solve, output_lifetime; outside: backend_construction, input_construction, correctness_check.
- **solve / concrete-shared** — inside: solve, output_lifetime; outside: backend_construction, input_construction, session_entry_exit, shared_admission_exit, correctness_check.

# CPU FFT Benchmark Results

- Suite: `cpu/fft`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/fft.yaml`
- Run metadata: `data/results/mac-cpu/cpu/fft/20260728_175304/run.yaml`
- Timestamp: `20260728_175304`

Latest run: `./scripts/run_cpu_fft.sh 1 4`.

This file is generated from sequential CPU FFT runs under `data/results/mac-cpu/cpu/fft/20260728_175304`.

- tenferro-rs commit: `80ebcc38ce11fb93385e8b6a1a49b613bc17452f`

## CPU Information

- Model: `Apple M4`
- Vendor: `Apple`
- Logical CPUs: `10`
- Physical CPUs: `10`
- Sockets: `1`
- Cores per socket: `10`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Performance: 4 physical / 4 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 4 CPUs/L2); Efficiency: 6 physical / 6 logical (L1i 128 KiB, L1d 64 KiB, L2 4 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5.2-arm64-arm-64bit`

## Thread Environments

### Threads: 1

- Run metadata: `data/results/mac-cpu/cpu/fft/20260728_175304/run_t1.yaml`
- OMP_NUM_THREADS: `1`
- OMP_THREAD_LIMIT: `1`
- OMP_DYNAMIC: `FALSE`
- RAYON_NUM_THREADS: `1`
- OPENBLAS_NUM_THREADS: `1`
- GOTO_NUM_THREADS: `1`
- MKL_NUM_THREADS: `1`
- VECLIB_MAXIMUM_THREADS: `1`
- VECLIB_NUM_THREADS: `1`
- NUMEXPR_NUM_THREADS: `1`
- BLIS_NUM_THREADS: `1`
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`

### Threads: 4

- Run metadata: `data/results/mac-cpu/cpu/fft/20260728_175304/run_t4.yaml`
- OMP_NUM_THREADS: `4`
- OMP_THREAD_LIMIT: `4`
- OMP_DYNAMIC: `FALSE`
- RAYON_NUM_THREADS: `4`
- OPENBLAS_NUM_THREADS: `4`
- GOTO_NUM_THREADS: `4`
- MKL_NUM_THREADS: `4`
- VECLIB_MAXIMUM_THREADS: `4`
- VECLIB_NUM_THREADS: `4`
- NUMEXPR_NUM_THREADS: `4`
- BLIS_NUM_THREADS: `4`
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4`

## Timing Discipline

- Input tensors are created outside the timed region.
- Each timed call creates the output tensor.
- tenferro-rs immediate rows use one-shot `TensorFftExt` calls.
- tenferro-rs cached rows reuse a caller-owned `FftExecutor` across warmups and timed runs.
- PyTorch rows use `torch.fft` after warmup, allowing PyTorch internal planning/cache behavior.
- The primary fair comparison is cached `FftExecutor` versus warmed `torch.fft`; immediate rows are retained only as one-shot diagnostics.
- This initial suite only measures 1D transforms to avoid row-major/column-major batched-axis layout artifacts.

## Threads: 1 4

- CSV: `data/results/mac-cpu/cpu/fft/20260728_175304/cpu_fft_t1_20260728_175304.csv`
- CSV: `data/results/mac-cpu/cpu/fft/20260728_175304/cpu_fft_t4_20260728_175304.csv`
- Source table: `data/results/mac-cpu/cpu/fft/20260728_175304/cpu_fft_20260728_175304.md`

## CPU FFT Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

Timing scope: input tensors are created outside the timed region; each timed call creates the FFT output tensor. The primary comparison is tenferro-rs FftExecutor cached versus warmed PyTorch torch.fft; one-shot tenferro-rs rows are diagnostic. Rows are limited to 1D transforms so tenferro-rs column-major layout and PyTorch row-major layout do not change the measured transform axis.

| suite | benchmark | dtype | threads | shape | tenferro-rs one-shot diagnostic (ms) | tenferro-rs cached primary (ms) | PyTorch torch.fft (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| cpu/fft | `fft` | c32 | 1 | `1d_n1048576` | 8.810 ± 0.874 | 4.349 ± 0.210 | 10.360 ± 0.500 |
| cpu/fft | `fft` | c32 | 4 | `1d_n1048576` | 8.965 ± 0.498 | 4.500 ± 0.096 | 10.663 ± 0.260 |
| cpu/fft | `fft` | c64 | 1 | `1d_n1048576` | 14.070 ± 0.146 | 8.176 ± 0.588 | 13.776 ± 1.353 |
| cpu/fft | `fft` | c64 | 4 | `1d_n1048576` | 13.920 ± 0.459 | 8.099 ± 0.230 | 12.751 ± 0.592 |
| cpu/fft | `ifft` | c32 | 1 | `1d_n1048576` | 8.945 ± 0.185 | 4.573 ± 0.210 | 11.693 ± 0.968 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n1048576` | 8.989 ± 0.449 | 4.466 ± 0.138 | 16.364 ± 1.488 |
| cpu/fft | `ifft` | c64 | 1 | `1d_n1048576` | 18.346 ± 5.555 | 11.078 ± 0.626 | 12.196 ± 0.746 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n1048576` | 14.988 ± 0.644 | 9.020 ± 3.035 | 23.034 ± 17.912 |
| cpu/fft | `irfft` | c32 | 1 | `1d_n1048576` | 9.925 ± 0.494 | 4.912 ± 0.262 | 3.869 ± 0.134 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n1048576` | 8.847 ± 0.288 | 4.507 ± 0.257 | 4.605 ± 0.255 |
| cpu/fft | `irfft` | c64 | 1 | `1d_n1048576` | 15.150 ± 1.842 | 8.476 ± 0.546 | 5.653 ± 0.569 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n1048576` | 14.227 ± 0.343 | 9.973 ± 1.792 | 18.327 ± 4.277 |
| cpu/fft | `rfft` | f32 | 1 | `1d_n1048576` | 10.989 ± 2.083 | 5.056 ± 0.518 | 4.096 ± 0.155 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n1048576` | 8.707 ± 0.538 | 4.473 ± 0.094 | 8.825 ± 0.967 |
| cpu/fft | `rfft` | f64 | 1 | `1d_n1048576` | 14.925 ± 0.355 | 8.078 ± 0.848 | 5.406 ± 0.280 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n1048576` | 15.132 ± 0.793 | 8.284 ± 0.594 | 6.605 ± 2.072 |

Notes:

- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs
- tenferro-fft-immediate: one-shot TensorFftExt call; no caller-owned plan cache

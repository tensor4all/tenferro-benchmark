# CPU FFT Benchmark Results

- Suite: `cpu/fft`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/fft.yaml`
- Run metadata: `data/results/mac-cpu/cpu/fft/20260728_195614/run.yaml`
- Timestamp: `20260728_195614`

Latest run: `./scripts/run_cpu_fft.sh 1 4`.

This file is generated from sequential CPU FFT runs under `data/results/mac-cpu/cpu/fft/20260728_195614`.

- tenferro-rs commit: `4ff6b8e1d56fc8d3f6b772f6053c09c745429585`

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

- Run metadata: `data/results/mac-cpu/cpu/fft/20260728_195614/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/fft/20260728_195614/run_t4.yaml`
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
- tenferro-rs eager rows reuse one `EagerRuntime` and input `EagerTensor`; setup is outside timing.
- tenferro-rs trace rows construct and compile `TracedTensorFftExt` graphs outside timing and reuse the compiled program.
- PyTorch rows use `torch.fft` after warmup, allowing PyTorch internal planning/cache behavior.
- The primary fair comparison is cached `FftExecutor` versus warmed `torch.fft`; immediate rows are retained only as one-shot diagnostics.
- This initial suite only measures 1D transforms to avoid row-major/column-major batched-axis layout artifacts.

## Threads: 1 4

- CSV: `data/results/mac-cpu/cpu/fft/20260728_195614/cpu_fft_t1_20260728_195614.csv`
- CSV: `data/results/mac-cpu/cpu/fft/20260728_195614/cpu_fft_t4_20260728_195614.csv`
- Source table: `data/results/mac-cpu/cpu/fft/20260728_195614/cpu_fft_20260728_195614.md`

## CPU FFT Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

Timing scope: input tensors are created outside the timed region; each timed call creates the FFT output tensor. The primary eager comparison is tenferro-rs FftExecutor cached versus warmed PyTorch torch.fft; traced rows reuse one compiled graph and one-shot rows are diagnostic. Rows are limited to 1D transforms so tenferro-rs column-major layout and PyTorch row-major layout do not change the measured transform axis.

| suite | benchmark | dtype | threads | shape | tenferro-rs one-shot diagnostic (ms) | tenferro-rs cached primary (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch torch.fft (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| cpu/fft | `fft` | c32 | 1 | `1d_n1048576` | 8.585 ± 0.492 | 4.339 ± 0.116 | 4.494 ± 0.072 | 4.625 ± 0.137 | 10.017 ± 0.193 |
| cpu/fft | `fft` | c32 | 4 | `1d_n1048576` | 8.656 ± 0.988 | 4.196 ± 0.244 | 4.575 ± 0.084 | 4.505 ± 0.041 | 10.215 ± 0.384 |
| cpu/fft | `fft` | c64 | 1 | `1d_n1048576` | 14.136 ± 0.254 | 8.112 ± 0.685 | 9.493 ± 1.154 | 8.830 ± 0.938 | 14.452 ± 1.216 |
| cpu/fft | `fft` | c64 | 4 | `1d_n1048576` | 13.546 ± 0.285 | 7.626 ± 0.545 | 8.251 ± 0.436 | 8.152 ± 0.978 | 12.523 ± 0.174 |
| cpu/fft | `ifft` | c32 | 1 | `1d_n1048576` | 8.907 ± 0.348 | 4.905 ± 1.072 | 4.665 ± 0.110 | 4.636 ± 0.061 | 11.559 ± 2.128 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n1048576` | 8.727 ± 0.261 | 4.579 ± 0.153 | 5.004 ± 0.399 | 4.824 ± 0.097 | 10.317 ± 0.295 |
| cpu/fft | `ifft` | c64 | 1 | `1d_n1048576` | 14.467 ± 0.673 | 8.717 ± 0.778 | 9.316 ± 0.796 | 9.360 ± 0.819 | 12.902 ± 1.208 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n1048576` | 14.907 ± 0.300 | 8.009 ± 0.241 | 8.712 ± 0.210 | 8.566 ± 0.188 | 11.955 ± 0.887 |
| cpu/fft | `irfft` | c32 | 1 | `1d_n1048576` | 8.678 ± 0.200 | 4.280 ± 0.223 | 4.382 ± 0.100 | 4.346 ± 0.077 | 3.786 ± 0.080 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n1048576` | 8.552 ± 0.214 | 4.418 ± 0.275 | 4.391 ± 0.108 | 4.408 ± 0.053 | 3.900 ± 0.304 |
| cpu/fft | `irfft` | c64 | 1 | `1d_n1048576` | 14.085 ± 0.900 | 8.034 ± 0.285 | 7.983 ± 0.737 | 8.038 ± 0.906 | 7.022 ± 6.285 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n1048576` | 14.321 ± 0.719 | 7.881 ± 0.714 | 8.366 ± 0.760 | 7.917 ± 0.744 | 5.120 ± 0.220 |
| cpu/fft | `rfft` | f32 | 1 | `1d_n1048576` | 8.402 ± 0.283 | 4.220 ± 0.257 | 4.244 ± 0.233 | 4.360 ± 0.213 | 4.128 ± 0.356 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n1048576` | 8.288 ± 0.126 | 4.175 ± 0.196 | 4.460 ± 0.079 | 4.457 ± 0.371 | 3.945 ± 0.053 |
| cpu/fft | `rfft` | f64 | 1 | `1d_n1048576` | 13.824 ± 0.380 | 7.655 ± 0.429 | 7.776 ± 0.352 | 7.846 ± 0.206 | 5.373 ± 0.513 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n1048576` | 14.184 ± 0.563 | 7.960 ± 0.256 | 7.894 ± 0.499 | 7.690 ± 0.237 | 5.136 ± 0.358 |

Notes:

- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs
- tenferro-fft-immediate: one-shot TensorFftExt call; no caller-owned plan cache
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs

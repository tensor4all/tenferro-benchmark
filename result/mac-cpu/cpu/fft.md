# CPU FFT Benchmark Results

- Suite: `cpu/fft`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/fft.yaml`
- Run metadata: `data/results/mac-cpu/cpu/fft/20260822_091026/run.yaml`
- Timestamp: `20260822_091026`

Latest run: `./scripts/run_cpu_fft.sh 1 4`.

This file is generated from sequential CPU FFT runs under `data/results/mac-cpu/cpu/fft/20260822_091026`.

- tenferro-rs commit: `a21a4c602fc6700b9bc0c3f1b14ebd19b9d7ec45`

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

- Run metadata: `data/results/mac-cpu/cpu/fft/20260822_091026/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/fft/20260822_091026/run_t4.yaml`
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
- tenferro-rs read rows call `TensorReadFftExt` on an owned contiguous `TensorRead`; its documented materialization and one-shot plan are inside timing.
- tenferro-rs cached rows reuse a caller-owned `FftExecutor` across warmups and timed runs.
- tenferro-rs eager rows reuse one `EagerRuntime` and input `EagerTensor`; setup is outside timing.
- tenferro-rs trace rows construct and compile `TracedTensorFftExt` graphs outside timing and reuse the compiled program.
- PyTorch rows use `torch.fft` after warmup, allowing PyTorch internal planning/cache behavior.
- The primary fair comparison is cached `FftExecutor` versus warmed `torch.fft`; immediate rows are retained only as one-shot diagnostics.
- This initial suite only measures 1D transforms to avoid row-major/column-major batched-axis layout artifacts.

## Threads: 1 4

- CSV: `data/results/mac-cpu/cpu/fft/20260822_091026/cpu_fft_t1_20260822_091026.csv`
- CSV: `data/results/mac-cpu/cpu/fft/20260822_091026/cpu_fft_t4_20260822_091026.csv`
- Source table: `data/results/mac-cpu/cpu/fft/20260822_091026/cpu_fft_20260822_091026.md`

## CPU FFT Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

Timing scope: input tensors are created outside the timed region; each timed call creates the FFT output tensor. The primary eager comparison is tenferro-rs FftExecutor cached versus warmed PyTorch torch.fft; traced rows reuse one compiled graph and one-shot rows are diagnostic. Rows are limited to 1D transforms so tenferro-rs column-major layout and PyTorch row-major layout do not change the measured transform axis.

| suite | benchmark | dtype | threads | shape | tenferro-rs one-shot diagnostic (ms) | tenferro-rs TensorRead API (ms) | tenferro-rs cached primary (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch torch.fft (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| cpu/fft | `fft` | c32 | 1 | `1d_n1048576` | 8.569 ± 0.780 | 8.547 ± 0.079 | 4.232 ± 0.403 | 4.487 ± 0.103 | 4.531 ± 0.136 | 9.914 ± 0.128 |
| cpu/fft | `fft` | c32 | 4 | `1d_n1048576` | 8.291 ± 0.212 | 8.423 ± 0.094 | 4.370 ± 0.424 | 4.491 ± 0.056 | 4.541 ± 0.135 | 9.902 ± 0.164 |
| cpu/fft | `fft` | c64 | 1 | `1d_n1048576` | 12.788 ± 0.157 | 13.219 ± 0.089 | 7.857 ± 0.752 | 7.919 ± 0.229 | 7.983 ± 0.131 | 12.663 ± 0.353 |
| cpu/fft | `fft` | c64 | 4 | `1d_n1048576` | 12.768 ± 0.196 | 13.453 ± 0.892 | 7.510 ± 0.379 | 8.354 ± 0.177 | 8.245 ± 0.580 | 12.413 ± 0.836 |
| cpu/fft | `ifft` | c32 | 1 | `1d_n1048576` | 8.565 ± 0.127 | 8.804 ± 0.164 | 4.253 ± 0.218 | 4.634 ± 0.049 | 4.630 ± 0.060 | 9.982 ± 0.138 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n1048576` | 8.610 ± 0.188 | 8.953 ± 0.514 | 4.472 ± 0.331 | 4.610 ± 0.082 | 4.675 ± 0.068 | 10.176 ± 0.154 |
| cpu/fft | `ifft` | c64 | 1 | `1d_n1048576` | 13.340 ± 0.278 | 13.695 ± 0.154 | 7.970 ± 0.118 | 8.445 ± 0.124 | 8.642 ± 0.598 | 11.428 ± 0.262 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n1048576` | 13.415 ± 0.263 | 13.668 ± 0.314 | 8.438 ± 0.738 | 8.611 ± 0.604 | 8.385 ± 0.495 | 11.451 ± 0.296 |
| cpu/fft | `irfft` | c32 | 1 | `1d_n1048576` | 8.336 ± 0.229 | 8.699 ± 0.166 | 4.231 ± 0.214 | 4.295 ± 0.073 | 4.384 ± 0.075 | 3.678 ± 0.023 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n1048576` | 8.543 ± 0.205 | 8.745 ± 0.238 | 4.358 ± 0.432 | 4.337 ± 0.041 | 4.402 ± 0.102 | 3.778 ± 0.042 |
| cpu/fft | `irfft` | c64 | 1 | `1d_n1048576` | 12.822 ± 0.452 | 13.119 ± 0.228 | 7.586 ± 0.206 | 7.744 ± 0.324 | 7.614 ± 0.151 | 4.945 ± 0.085 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n1048576` | 13.044 ± 0.812 | 13.076 ± 0.385 | 7.508 ± 0.590 | 7.843 ± 0.275 | 7.732 ± 0.211 | 5.040 ± 0.158 |
| cpu/fft | `rfft` | f32 | 1 | `1d_n1048576` | 8.242 ± 0.188 | 8.366 ± 0.098 | 4.244 ± 0.190 | 4.193 ± 0.443 | 4.241 ± 0.063 | 3.940 ± 0.095 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n1048576` | 8.388 ± 0.155 | 8.442 ± 0.096 | 4.266 ± 0.051 | 4.111 ± 0.098 | 4.271 ± 0.253 | 3.943 ± 0.158 |
| cpu/fft | `rfft` | f64 | 1 | `1d_n1048576` | 12.664 ± 0.384 | 12.809 ± 0.139 | 7.245 ± 0.179 | 7.573 ± 0.209 | 7.487 ± 0.170 | 4.959 ± 0.063 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n1048576` | 12.674 ± 0.523 | 12.998 ± 0.346 | 7.467 ± 0.995 | 7.615 ± 0.543 | 7.630 ± 0.275 | 4.869 ± 0.093 |

Notes:

- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs
- tenferro-fft-immediate: one-shot TensorFftExt call; no caller-owned plan cache
- tenferro-fft-read: TensorReadFftExt on an owned contiguous TensorRead; materialization and one-shot planning included
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs

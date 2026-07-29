# CPU FFT Benchmark Results

- Suite: `cpu/fft`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/fft.yaml`
- Run metadata: `data/results/mac-cpu/cpu/fft/20260729_200115/run.yaml`
- Timestamp: `20260729_200115`

Latest run: `./scripts/run_cpu_fft.sh 1 4`.

This file is generated from sequential CPU FFT runs under `data/results/mac-cpu/cpu/fft/20260729_200115`.

- tenferro-rs commit: `57c41c218356e871532e4dea4465a6bcf2d6c8fb`

## CPU Information

- Model: `Apple M5 Max`
- Vendor: `Apple`
- Logical CPUs: `18`
- Physical CPUs: `18`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Super: 6 physical / 6 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 6 CPUs/L2); Performance: 12 physical / 12 logical (L1i 128 KiB, L1d 64 KiB, L2 8 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5.1-arm64-arm-64bit`

## Thread Environments

### Threads: 1

- Run metadata: `data/results/mac-cpu/cpu/fft/20260729_200115/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/fft/20260729_200115/run_t4.yaml`
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

- CSV: `data/results/mac-cpu/cpu/fft/20260729_200115/cpu_fft_t1_20260729_200115.csv`
- CSV: `data/results/mac-cpu/cpu/fft/20260729_200115/cpu_fft_t4_20260729_200115.csv`
- Source table: `data/results/mac-cpu/cpu/fft/20260729_200115/cpu_fft_20260729_200115.md`

## CPU FFT Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

Timing scope: input tensors are created outside the timed region; each timed call creates the FFT output tensor. The primary eager comparison is tenferro-rs FftExecutor cached versus warmed PyTorch torch.fft; traced rows reuse one compiled graph and one-shot rows are diagnostic. Rows are limited to 1D transforms so tenferro-rs column-major layout and PyTorch row-major layout do not change the measured transform axis.

| suite | benchmark | dtype | threads | shape | tenferro-rs one-shot diagnostic (ms) | tenferro-rs TensorRead API (ms) | tenferro-rs cached primary (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch torch.fft (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| cpu/fft | `fft` | c32 | 1 | `1d_n1048576` | 8.385 ± 0.848 | 8.000 ± 0.509 | 4.405 ± 0.127 | 4.253 ± 0.143 | 4.540 ± 0.112 | 10.453 ± 0.292 |
| cpu/fft | `fft` | c32 | 4 | `1d_n1048576` | 8.167 ± 0.976 | 8.096 ± 0.306 | 4.385 ± 0.130 | 4.689 ± 0.231 | 4.548 ± 0.044 | 10.424 ± 0.283 |
| cpu/fft | `fft` | c64 | 1 | `1d_n1048576` | 11.603 ± 0.887 | 12.135 ± 0.676 | 7.031 ± 0.245 | 7.494 ± 0.361 | 6.678 ± 0.293 | 12.030 ± 0.479 |
| cpu/fft | `fft` | c64 | 4 | `1d_n1048576` | 12.273 ± 0.644 | 12.604 ± 0.434 | 6.931 ± 0.925 | 7.201 ± 0.107 | 6.534 ± 0.083 | 12.128 ± 0.474 |
| cpu/fft | `ifft` | c32 | 1 | `1d_n1048576` | 8.375 ± 0.408 | 8.411 ± 0.125 | 4.138 ± 0.170 | 4.870 ± 0.144 | 4.465 ± 0.230 | 9.886 ± 0.256 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n1048576` | 8.505 ± 0.473 | 8.661 ± 0.194 | 4.565 ± 0.059 | 4.876 ± 0.307 | 4.758 ± 0.090 | 10.643 ± 0.277 |
| cpu/fft | `ifft` | c64 | 1 | `1d_n1048576` | 11.945 ± 1.136 | 12.087 ± 0.644 | 6.596 ± 0.183 | 7.314 ± 0.353 | 7.885 ± 0.426 | 12.078 ± 3.335 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n1048576` | 12.537 ± 0.542 | 12.567 ± 0.608 | 6.644 ± 0.155 | 6.877 ± 0.240 | 7.748 ± 0.206 | 11.646 ± 0.939 |
| cpu/fft | `irfft` | c32 | 1 | `1d_n1048576` | 7.904 ± 0.774 | 8.085 ± 0.341 | 4.214 ± 0.100 | 4.546 ± 0.197 | 4.292 ± 0.106 | 3.720 ± 0.536 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n1048576` | 7.818 ± 0.321 | 8.116 ± 0.279 | 4.211 ± 0.164 | 4.271 ± 0.052 | 4.219 ± 0.031 | 3.512 ± 0.063 |
| cpu/fft | `irfft` | c64 | 1 | `1d_n1048576` | 11.471 ± 1.154 | 12.369 ± 0.203 | 6.626 ± 0.261 | 6.774 ± 0.230 | 7.580 ± 0.284 | 4.752 ± 0.288 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n1048576` | 11.409 ± 0.999 | 11.932 ± 1.064 | 6.000 ± 0.097 | 6.192 ± 0.116 | 7.027 ± 0.157 | 4.533 ± 0.032 |
| cpu/fft | `rfft` | f32 | 1 | `1d_n1048576` | 7.834 ± 0.455 | 8.111 ± 0.266 | 4.050 ± 0.168 | 4.131 ± 0.090 | 4.119 ± 0.138 | 3.452 ± 0.062 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n1048576` | 7.682 ± 0.491 | 7.982 ± 0.277 | 4.086 ± 0.058 | 4.212 ± 0.308 | 4.082 ± 0.055 | 3.407 ± 0.022 |
| cpu/fft | `rfft` | f64 | 1 | `1d_n1048576` | 11.998 ± 0.769 | 11.858 ± 0.938 | 6.939 ± 0.242 | 7.054 ± 0.304 | 6.398 ± 0.061 | 4.474 ± 0.071 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n1048576` | 11.875 ± 0.205 | 11.450 ± 0.397 | 6.943 ± 0.645 | 6.808 ± 0.109 | 6.711 ± 0.149 | 4.474 ± 0.224 |

Notes:

- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs
- tenferro-fft-immediate: one-shot TensorFftExt call; no caller-owned plan cache
- tenferro-fft-read: TensorReadFftExt on an owned contiguous TensorRead; materialization and one-shot planning included
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs

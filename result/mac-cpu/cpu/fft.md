# CPU FFT Benchmark Results

- Suite: `cpu/fft`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/fft.yaml`
- Run metadata: `data/results/mac-cpu/cpu/fft/20260913_124630/run.yaml`
- Timestamp: `20260913_124630`

Latest run: `./scripts/run_cpu_fft.sh 1 4`.

This file is generated from sequential CPU FFT runs under `data/results/mac-cpu/cpu/fft/20260913_124630`.

- tenferro-rs commit: `a48866b1a0bb52e6f9712c485925105b14ea30b9`

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
- Python platform: `macOS-26.5.1-arm64-arm-64bit-Mach-O`

## Thread Environments

### Threads: 1

- Run metadata: `data/results/mac-cpu/cpu/fft/20260913_124630/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/fft/20260913_124630/run_t4.yaml`
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

- CSV: `data/results/mac-cpu/cpu/fft/20260913_124630/cpu_fft_t1_20260913_124630.csv`
- CSV: `data/results/mac-cpu/cpu/fft/20260913_124630/cpu_fft_t4_20260913_124630.csv`
- Source table: `data/results/mac-cpu/cpu/fft/20260913_124630/cpu_fft_20260913_124630.md`

## CPU FFT Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

Timing scope: input tensors are created outside the timed region; each timed call creates the FFT output tensor. The primary eager comparison is tenferro-rs FftExecutor cached versus warmed PyTorch torch.fft; traced rows reuse one compiled graph and one-shot rows are diagnostic. Rows are limited to 1D transforms so tenferro-rs column-major layout and PyTorch row-major layout do not change the measured transform axis.

| suite | benchmark | dtype | threads | shape | tenferro-rs one-shot diagnostic (ms) | tenferro-rs TensorRead API (ms) | tenferro-rs cached primary (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch torch.fft (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| cpu/fft | `fft` | c32 | 1 | `1d_n1048576` | 8.195 ± 0.532 | 8.060 ± 0.189 | 4.349 ± 0.206 | 4.356 ± 0.062 | 4.437 ± 0.135 | 10.625 ± 0.270 |
| cpu/fft | `fft` | c32 | 4 | `1d_n1048576` | 8.219 ± 0.722 | 8.124 ± 0.261 | 4.394 ± 0.186 | 4.435 ± 0.094 | 4.385 ± 0.035 | 10.559 ± 0.250 |
| cpu/fft | `fft` | c64 | 1 | `1d_n1048576` | 11.821 ± 0.554 | 11.884 ± 0.316 | 6.849 ± 0.203 | 6.936 ± 0.299 | 7.569 ± 0.166 | 12.061 ± 0.363 |
| cpu/fft | `fft` | c64 | 4 | `1d_n1048576` | 12.038 ± 0.310 | 11.997 ± 0.299 | 7.501 ± 0.151 | 7.507 ± 0.192 | 7.530 ± 0.117 | 12.074 ± 0.209 |
| cpu/fft | `ifft` | c32 | 1 | `1d_n1048576` | 8.238 ± 0.245 | 8.223 ± 0.525 | 4.450 ± 0.065 | 4.060 ± 0.067 | 4.530 ± 0.100 | 11.912 ± 2.204 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n1048576` | 8.334 ± 0.239 | 8.299 ± 0.286 | 4.488 ± 0.154 | 4.562 ± 0.215 | 4.482 ± 0.110 | 12.377 ± 2.063 |
| cpu/fft | `ifft` | c64 | 1 | `1d_n1048576` | 11.993 ± 0.694 | 12.148 ± 0.587 | 7.450 ± 0.130 | 7.453 ± 0.164 | 7.032 ± 0.263 | 11.618 ± 0.377 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n1048576` | 12.159 ± 0.473 | 12.213 ± 0.645 | 7.752 ± 0.169 | 7.754 ± 0.198 | 6.875 ± 0.295 | 11.507 ± 0.091 |
| cpu/fft | `irfft` | c32 | 1 | `1d_n1048576` | 8.240 ± 0.420 | 8.275 ± 0.391 | 4.339 ± 0.104 | 4.418 ± 0.119 | 3.953 ± 0.161 | 3.631 ± 0.043 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n1048576` | 8.316 ± 0.431 | 8.357 ± 0.259 | 4.416 ± 0.175 | 4.485 ± 0.113 | 4.065 ± 0.113 | 3.610 ± 0.085 |
| cpu/fft | `irfft` | c64 | 1 | `1d_n1048576` | 12.044 ± 0.700 | 11.980 ± 0.419 | 7.624 ± 0.165 | 7.641 ± 0.143 | 7.812 ± 0.214 | 4.731 ± 0.124 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n1048576` | 12.057 ± 0.769 | 12.271 ± 0.221 | 7.678 ± 0.293 | 7.716 ± 0.243 | 7.816 ± 0.346 | 4.648 ± 0.160 |
| cpu/fft | `rfft` | f32 | 1 | `1d_n1048576` | 8.008 ± 0.157 | 8.266 ± 0.155 | 3.753 ± 0.189 | 4.257 ± 0.292 | 4.263 ± 0.197 | 3.499 ± 0.082 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n1048576` | 8.148 ± 0.259 | 8.204 ± 0.315 | 3.659 ± 0.157 | 3.707 ± 0.138 | 3.903 ± 0.133 | 3.494 ± 0.074 |
| cpu/fft | `rfft` | f64 | 1 | `1d_n1048576` | 11.668 ± 1.046 | 11.670 ± 0.426 | 6.526 ± 0.169 | 6.661 ± 0.311 | 7.389 ± 0.237 | 4.465 ± 0.076 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n1048576` | 11.798 ± 0.749 | 11.784 ± 0.452 | 7.389 ± 0.143 | 6.530 ± 0.277 | 7.301 ± 0.306 | 4.411 ± 0.067 |

Notes:

- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs
- tenferro-fft-immediate: one-shot TensorFftExt call; no caller-owned plan cache
- tenferro-fft-read: TensorReadFftExt on an owned contiguous TensorRead; materialization and one-shot planning included
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs

# CPU FFT Benchmark Results

- Suite: `cpu/fft`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/fft.yaml`
- Run metadata: `data/results/mac-cpu/cpu/fft/20260728_201407/run.yaml`
- Timestamp: `20260728_201407`

Latest run: `./scripts/run_cpu_fft.sh 1 4`.

This file is generated from sequential CPU FFT runs under `data/results/mac-cpu/cpu/fft/20260728_201407`.

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

- Run metadata: `data/results/mac-cpu/cpu/fft/20260728_201407/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/fft/20260728_201407/run_t4.yaml`
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

- CSV: `data/results/mac-cpu/cpu/fft/20260728_201407/cpu_fft_t1_20260728_201407.csv`
- CSV: `data/results/mac-cpu/cpu/fft/20260728_201407/cpu_fft_t4_20260728_201407.csv`
- Source table: `data/results/mac-cpu/cpu/fft/20260728_201407/cpu_fft_20260728_201407.md`

## CPU FFT Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

Timing scope: input tensors are created outside the timed region; each timed call creates the FFT output tensor. The primary eager comparison is tenferro-rs FftExecutor cached versus warmed PyTorch torch.fft; traced rows reuse one compiled graph and one-shot rows are diagnostic. Rows are limited to 1D transforms so tenferro-rs column-major layout and PyTorch row-major layout do not change the measured transform axis.

| suite | benchmark | dtype | threads | shape | tenferro-rs one-shot diagnostic (ms) | tenferro-rs TensorRead API (ms) | tenferro-rs cached primary (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch torch.fft (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| cpu/fft | `fft` | c32 | 1 | `1d_n1048576` | 8.642 ± 0.899 | 8.640 ± 0.161 | 4.276 ± 0.406 | 4.417 ± 0.116 | 4.505 ± 0.276 | 10.414 ± 0.294 |
| cpu/fft | `fft` | c32 | 4 | `1d_n1048576` | 9.019 ± 0.876 | 8.882 ± 0.260 | 4.393 ± 0.243 | 5.453 ± 0.904 | 4.704 ± 0.563 | 10.388 ± 0.304 |
| cpu/fft | `fft` | c64 | 1 | `1d_n1048576` | 13.787 ± 0.277 | 14.159 ± 0.555 | 8.011 ± 0.159 | 8.504 ± 0.100 | 8.799 ± 0.177 | 12.618 ± 0.336 |
| cpu/fft | `fft` | c64 | 4 | `1d_n1048576` | 14.222 ± 2.946 | 14.449 ± 0.401 | 7.881 ± 0.476 | 8.476 ± 0.606 | 8.553 ± 0.301 | 12.557 ± 0.344 |
| cpu/fft | `ifft` | c32 | 1 | `1d_n1048576` | 8.943 ± 0.361 | 9.005 ± 0.184 | 4.490 ± 0.129 | 4.750 ± 0.170 | 4.627 ± 0.135 | 9.874 ± 0.096 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n1048576` | 9.027 ± 0.462 | 9.101 ± 0.089 | 4.436 ± 0.039 | 4.675 ± 0.094 | 4.629 ± 0.118 | 10.106 ± 0.112 |
| cpu/fft | `ifft` | c64 | 1 | `1d_n1048576` | 14.573 ± 0.725 | 14.947 ± 0.492 | 8.354 ± 0.600 | 8.842 ± 0.120 | 8.747 ± 0.438 | 12.029 ± 0.648 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n1048576` | 14.526 ± 0.501 | 14.804 ± 0.584 | 8.178 ± 0.671 | 8.922 ± 0.536 | 8.765 ± 0.250 | 12.070 ± 0.241 |
| cpu/fft | `irfft` | c32 | 1 | `1d_n1048576` | 8.786 ± 0.141 | 8.839 ± 0.204 | 4.397 ± 0.249 | 4.446 ± 0.104 | 4.369 ± 0.202 | 3.732 ± 0.087 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n1048576` | 8.668 ± 0.135 | 8.814 ± 0.082 | 4.306 ± 0.397 | 4.374 ± 0.112 | 4.442 ± 0.236 | 3.759 ± 0.040 |
| cpu/fft | `irfft` | c64 | 1 | `1d_n1048576` | 13.967 ± 0.377 | 14.169 ± 0.372 | 7.704 ± 0.480 | 8.184 ± 0.598 | 8.305 ± 0.480 | 4.949 ± 0.111 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n1048576` | 14.218 ± 0.532 | 14.038 ± 0.678 | 7.806 ± 0.497 | 8.164 ± 0.369 | 8.076 ± 0.822 | 5.101 ± 0.095 |
| cpu/fft | `rfft` | f32 | 1 | `1d_n1048576` | 8.580 ± 0.264 | 8.731 ± 0.557 | 4.273 ± 0.053 | 4.335 ± 0.054 | 4.236 ± 0.196 | 3.963 ± 0.019 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n1048576` | 8.665 ± 0.355 | 8.880 ± 0.411 | 4.244 ± 0.149 | 4.371 ± 0.109 | 4.396 ± 0.065 | 3.947 ± 0.019 |
| cpu/fft | `rfft` | f64 | 1 | `1d_n1048576` | 13.666 ± 0.340 | 13.607 ± 0.665 | 7.760 ± 0.377 | 7.933 ± 0.290 | 7.864 ± 0.230 | 4.962 ± 0.046 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n1048576` | 13.933 ± 0.259 | 14.145 ± 0.315 | 7.630 ± 0.485 | 8.267 ± 0.637 | 7.825 ± 0.455 | 4.984 ± 0.098 |

Notes:

- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs
- tenferro-fft-immediate: one-shot TensorFftExt call; no caller-owned plan cache
- tenferro-fft-read: TensorReadFftExt on an owned contiguous TensorRead; materialization and one-shot planning included
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs

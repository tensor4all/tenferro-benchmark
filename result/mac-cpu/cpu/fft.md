# CPU FFT Benchmark Results

- Suite: `cpu/fft`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/fft.yaml`
- Run metadata: `data/results/mac-cpu/cpu/fft/20260806_154304/run.yaml`
- Timestamp: `20260806_154304`

Latest run: `./scripts/run_cpu_fft.sh 1 4`.

This file is generated from sequential CPU FFT runs under `data/results/mac-cpu/cpu/fft/20260806_154304`.

- tenferro-rs commit: `11a5b5a3c30b6258919557b4b69b429b6e686d75`

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

- Run metadata: `data/results/mac-cpu/cpu/fft/20260806_154304/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/fft/20260806_154304/run_t4.yaml`
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

- CSV: `data/results/mac-cpu/cpu/fft/20260806_154304/cpu_fft_t1_20260806_154304.csv`
- CSV: `data/results/mac-cpu/cpu/fft/20260806_154304/cpu_fft_t4_20260806_154304.csv`
- Source table: `data/results/mac-cpu/cpu/fft/20260806_154304/cpu_fft_20260806_154304.md`

## CPU FFT Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

Timing scope: input tensors are created outside the timed region; each timed call creates the FFT output tensor. The primary eager comparison is tenferro-rs FftExecutor cached versus warmed PyTorch torch.fft; traced rows reuse one compiled graph and one-shot rows are diagnostic. Rows are limited to 1D transforms so tenferro-rs column-major layout and PyTorch row-major layout do not change the measured transform axis.

| suite | benchmark | dtype | threads | shape | tenferro-rs one-shot diagnostic (ms) | tenferro-rs TensorRead API (ms) | tenferro-rs cached primary (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch torch.fft (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| cpu/fft | `fft` | c32 | 1 | `1d_n1048576` | 8.597 ± 0.848 | 8.567 ± 0.121 | 4.355 ± 0.407 | 4.488 ± 0.046 | 4.543 ± 0.121 | 9.951 ± 0.128 |
| cpu/fft | `fft` | c32 | 4 | `1d_n1048576` | 8.748 ± 0.833 | 8.608 ± 0.104 | 4.384 ± 0.058 | 4.498 ± 0.085 | 4.549 ± 0.111 | 9.889 ± 0.589 |
| cpu/fft | `fft` | c64 | 1 | `1d_n1048576` | 13.570 ± 0.404 | 13.929 ± 0.418 | 7.721 ± 0.172 | 8.190 ± 0.069 | 8.285 ± 1.330 | 12.810 ± 0.764 |
| cpu/fft | `fft` | c64 | 4 | `1d_n1048576` | 13.418 ± 0.123 | 14.385 ± 0.797 | 7.674 ± 0.175 | 8.251 ± 0.086 | 8.122 ± 0.467 | 12.056 ± 0.234 |
| cpu/fft | `ifft` | c32 | 1 | `1d_n1048576` | 8.705 ± 0.325 | 8.896 ± 0.125 | 4.454 ± 0.035 | 4.552 ± 0.105 | 4.708 ± 0.050 | 10.260 ± 0.406 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n1048576` | 8.634 ± 0.149 | 8.789 ± 0.122 | 4.498 ± 0.015 | 4.632 ± 0.062 | 4.680 ± 0.080 | 10.144 ± 0.386 |
| cpu/fft | `ifft` | c64 | 1 | `1d_n1048576` | 14.500 ± 1.049 | 14.493 ± 0.601 | 8.689 ± 0.524 | 9.164 ± 0.543 | 9.428 ± 0.538 | 11.621 ± 0.161 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n1048576` | 14.045 ± 0.594 | 14.150 ± 0.165 | 7.966 ± 0.479 | 8.509 ± 0.168 | 8.545 ± 0.306 | 11.570 ± 0.404 |
| cpu/fft | `irfft` | c32 | 1 | `1d_n1048576` | 9.057 ± 1.113 | 8.720 ± 0.209 | 4.404 ± 0.096 | 4.521 ± 0.184 | 4.756 ± 0.338 | 3.747 ± 0.043 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n1048576` | 8.575 ± 0.690 | 8.998 ± 0.170 | 4.440 ± 0.237 | 4.311 ± 0.093 | 4.351 ± 0.168 | 3.763 ± 0.060 |
| cpu/fft | `irfft` | c64 | 1 | `1d_n1048576` | 13.837 ± 0.489 | 16.388 ± 1.165 | 8.508 ± 0.633 | 8.512 ± 0.223 | 8.426 ± 1.040 | 5.014 ± 0.053 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n1048576` | 13.469 ± 0.241 | 13.782 ± 0.242 | 7.481 ± 0.359 | 8.330 ± 0.942 | 7.986 ± 0.125 | 4.965 ± 0.067 |
| cpu/fft | `rfft` | f32 | 1 | `1d_n1048576` | 8.596 ± 0.256 | 8.797 ± 0.249 | 4.231 ± 0.029 | 4.260 ± 0.044 | 4.253 ± 0.283 | 3.919 ± 0.027 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n1048576` | 8.653 ± 0.220 | 8.525 ± 0.134 | 4.225 ± 0.033 | 4.133 ± 0.064 | 4.087 ± 0.155 | 3.940 ± 0.023 |
| cpu/fft | `rfft` | f64 | 1 | `1d_n1048576` | 13.518 ± 0.171 | 13.895 ± 0.594 | 7.566 ± 0.172 | 8.105 ± 0.332 | 7.774 ± 0.174 | 4.885 ± 0.031 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n1048576` | 13.322 ± 0.456 | 14.026 ± 0.347 | 8.070 ± 0.453 | 8.271 ± 1.116 | 7.897 ± 0.483 | 4.864 ± 0.019 |

Notes:

- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs
- tenferro-fft-immediate: one-shot TensorFftExt call; no caller-owned plan cache
- tenferro-fft-read: TensorReadFftExt on an owned contiguous TensorRead; materialization and one-shot planning included
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs

# CPU FFT Benchmark Results

- Suite: `cpu/fft`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/fft.yaml`
- Run metadata: `data/results/mac-cpu/cpu/fft/20260913_142656/run.yaml`
- Timestamp: `20260913_142656`

Latest run: `./scripts/run_cpu_fft.sh 1 4`.

This file is generated from sequential CPU FFT runs under `data/results/mac-cpu/cpu/fft/20260913_142656`.

- tenferro-rs commit: `a793c2e95693f053722fbff8d0db25722336c21f`

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

- Run metadata: `data/results/mac-cpu/cpu/fft/20260913_142656/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/fft/20260913_142656/run_t4.yaml`
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
- tenferro-rs immediate rows require BENCH_INCLUDE_SETUP_DIAGNOSTICS=1 and use the separate cpu/fft_setup suite.
- tenferro-rs read rows require BENCH_INCLUDE_SETUP_DIAGNOSTICS=1 because the API performs one-shot planning.
- tenferro-rs cached rows reuse a caller-owned `FftExecutor` across warmups and timed runs.
- tenferro-rs eager rows reuse one `EagerRuntime` and input `EagerTensor`; setup is outside timing.
- tenferro-rs trace rows construct and compile `TracedTensorFftExt` graphs outside timing and reuse the compiled program.
- PyTorch rows use `torch.fft` after warmup, allowing PyTorch internal planning/cache behavior.
- The primary fair comparison is cached `FftExecutor` versus warmed `torch.fft`; one-shot diagnostics are opt-in and never operation comparisons.
- This initial suite only measures 1D transforms to avoid row-major/column-major batched-axis layout artifacts.

## Threads: 1 4

- CSV: `data/results/mac-cpu/cpu/fft/20260913_142656/cpu_fft_t1_20260913_142656.csv`
- CSV: `data/results/mac-cpu/cpu/fft/20260913_142656/cpu_fft_t4_20260913_142656.csv`
- Source table: `data/results/mac-cpu/cpu/fft/20260913_142656/cpu_fft_20260913_142656.md`

## CPU FFT Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

Timing scope: input tensors are created outside the timed region; each timed call creates the FFT output tensor. The primary eager comparison is tenferro-rs FftExecutor cached versus warmed PyTorch torch.fft; traced rows reuse one compiled graph and one-shot rows are diagnostic. Rows are limited to 1D transforms so tenferro-rs column-major layout and PyTorch row-major layout do not change the measured transform axis.

### Operation execution

| suite | benchmark | dtype | threads | shape | tenferro-rs one-shot diagnostic (ms) | tenferro-rs TensorRead API (ms) | tenferro-rs cached primary (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch torch.fft (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| cpu/fft | `fft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 4.636 ± 0.744 | 4.135 ± 0.121 | 4.074 ± 0.097 | 10.291 ± 0.248 |
| cpu/fft | `fft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 4.222 ± 0.162 | 4.251 ± 0.097 | 4.238 ± 0.138 | 10.253 ± 0.187 |
| cpu/fft | `fft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 6.963 ± 0.288 | 6.988 ± 0.177 | 6.313 ± 2.640 | 11.787 ± 0.263 |
| cpu/fft | `fft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 7.241 ± 0.310 | 7.222 ± 0.350 | 6.348 ± 0.228 | 11.691 ± 0.217 |
| cpu/fft | `ifft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 4.325 ± 0.178 | 3.990 ± 0.106 | 4.337 ± 0.134 | 10.412 ± 0.267 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 4.284 ± 0.134 | 4.335 ± 0.141 | 4.405 ± 0.098 | 10.399 ± 0.237 |
| cpu/fft | `ifft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 6.359 ± 0.299 | 6.490 ± 0.280 | 6.699 ± 0.204 | 11.226 ± 0.248 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 6.475 ± 0.280 | 6.481 ± 0.260 | 7.421 ± 0.335 | 11.021 ± 0.318 |
| cpu/fft | `irfft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 4.196 ± 0.101 | 3.797 ± 0.120 | 4.245 ± 0.102 | 3.534 ± 0.128 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 4.267 ± 0.158 | 3.858 ± 0.061 | 4.362 ± 0.071 | 3.517 ± 0.044 |
| cpu/fft | `irfft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 6.530 ± 0.275 | 6.472 ± 0.144 | 7.233 ± 0.206 | 4.537 ± 0.090 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 6.570 ± 0.125 | 6.554 ± 0.129 | 7.332 ± 0.229 | 4.560 ± 0.203 |
| cpu/fft | `rfft` | f32 | 1 | `1d_n1048576` | skipped | skipped | 4.012 ± 0.128 | 3.651 ± 0.078 | 3.553 ± 0.111 | 3.462 ± 0.081 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n1048576` | skipped | skipped | 4.002 ± 0.179 | 4.037 ± 0.111 | 3.632 ± 0.090 | 3.456 ± 0.055 |
| cpu/fft | `rfft` | f64 | 1 | `1d_n1048576` | skipped | skipped | 6.766 ± 0.234 | 6.261 ± 0.232 | 6.885 ± 0.237 | 4.343 ± 0.102 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n1048576` | skipped | skipped | 7.064 ± 0.306 | 6.321 ± 0.176 | 6.923 ± 0.385 | 4.347 ± 0.082 |

Notes:

- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs
- tenferro-fft-immediate: one-shot API includes planning; use cached executor for operation timing
- tenferro-fft-read: one-shot API includes planning; use cached executor for operation timing
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs

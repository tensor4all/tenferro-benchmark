# CPU FFT Benchmark Results

- Suite: `cpu/fft`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/fft.yaml`
- Run metadata: `data/results/mac-cpu/cpu/fft/20260728_174549/run.yaml`
- Timestamp: `20260728_174549`

Latest run: `./scripts/run_cpu_fft.sh 1 4`.

This file is generated from sequential CPU FFT runs under `data/results/mac-cpu/cpu/fft/20260728_174549`.

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

- Run metadata: `data/results/mac-cpu/cpu/fft/20260728_174549/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/fft/20260728_174549/run_t4.yaml`
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
- This initial suite only measures 1D transforms to avoid row-major/column-major batched-axis layout artifacts.

## Threads: 1 4

- CSV: `data/results/mac-cpu/cpu/fft/20260728_174549/cpu_fft_t1_20260728_174549.csv`
- CSV: `data/results/mac-cpu/cpu/fft/20260728_174549/cpu_fft_t4_20260728_174549.csv`
- Source table: `data/results/mac-cpu/cpu/fft/20260728_174549/cpu_fft_20260728_174549.md`

## CPU FFT Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

Timing scope: input tensors are created outside the timed region; each timed call creates the FFT output tensor. Rows are limited to 1D transforms so tenferro-rs column-major layout and PyTorch row-major layout do not change the measured transform axis.

| suite | benchmark | dtype | threads | shape | tenferro-rs immediate (ms) | tenferro-rs FftExecutor cached (ms) | PyTorch torch.fft (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| cpu/fft | `fft` | c32 | 1 | `1d_n1048576` | 8.473 ± 0.856 | 4.263 ± 0.161 | 10.266 ± 0.190 |
| cpu/fft | `fft` | c32 | 4 | `1d_n1048576` | 8.787 ± 1.018 | 4.285 ± 0.301 | 9.920 ± 0.102 |
| cpu/fft | `fft` | c64 | 1 | `1d_n1048576` | 13.364 ± 0.295 | 7.666 ± 0.221 | 12.140 ± 0.179 |
| cpu/fft | `fft` | c64 | 4 | `1d_n1048576` | 13.349 ± 0.157 | 7.476 ± 0.583 | 12.484 ± 0.256 |
| cpu/fft | `ifft` | c32 | 1 | `1d_n1048576` | 8.649 ± 0.133 | 4.468 ± 0.091 | 10.108 ± 0.143 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n1048576` | 8.647 ± 0.158 | 4.452 ± 0.288 | 10.094 ± 0.065 |
| cpu/fft | `ifft` | c64 | 1 | `1d_n1048576` | 13.953 ± 0.277 | 8.090 ± 0.298 | 11.678 ± 0.250 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n1048576` | 14.610 ± 0.560 | 8.229 ± 0.093 | 11.562 ± 0.203 |
| cpu/fft | `irfft` | c32 | 1 | `1d_n1048576` | 8.898 ± 0.324 | 4.429 ± 0.317 | 3.711 ± 0.088 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n1048576` | 9.051 ± 0.506 | 4.320 ± 0.191 | 3.696 ± 0.077 |
| cpu/fft | `irfft` | c64 | 1 | `1d_n1048576` | 13.662 ± 0.636 | 7.475 ± 0.171 | 4.932 ± 0.064 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n1048576` | 13.510 ± 0.786 | 7.506 ± 0.367 | 4.888 ± 0.081 |
| cpu/fft | `rfft` | f32 | 1 | `1d_n1048576` | 8.240 ± 0.132 | 3.854 ± 0.426 | 3.932 ± 0.015 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n1048576` | 8.247 ± 0.138 | 4.248 ± 0.358 | 3.937 ± 0.017 |
| cpu/fft | `rfft` | f64 | 1 | `1d_n1048576` | 13.357 ± 0.246 | 7.436 ± 0.173 | 4.913 ± 0.018 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n1048576` | 13.329 ± 0.551 | 7.223 ± 0.245 | 4.870 ± 0.133 |

Notes:

- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs
- tenferro-fft-immediate: one-shot TensorFftExt call; no caller-owned plan cache

# CPU FFT Benchmark Results

- Suite: `cpu/fft`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/fft.yaml`
- Run metadata: `data/results/mac-cpu/cpu/fft/20260728_164219/run.yaml`
- Timestamp: `20260728_164219`

Latest run: `./scripts/run_cpu_fft.sh 4`.

This file is generated from one CPU FFT run under `data/results/mac-cpu/cpu/fft/20260728_164219`.

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

## Thread Environment

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

## Threads: 4

- CSV: `data/results/mac-cpu/cpu/fft/20260728_164219/cpu_fft_t4_20260728_164219.csv`
- Source table: `data/results/mac-cpu/cpu/fft/20260728_164219/cpu_fft_t4_20260728_164219.md`

## CPU FFT Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

Timing scope: input tensors are created outside the timed region; each timed call creates the FFT output tensor. Rows are limited to 1D transforms so tenferro-rs column-major layout and PyTorch row-major layout do not change the measured transform axis.

| suite | benchmark | dtype | threads | shape | tenferro-rs immediate (ms) | tenferro-rs FftExecutor cached (ms) | PyTorch torch.fft (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| cpu/fft | `fft` | c32 | 4 | `1d_n1024` | 0.007 ± 0.002 | 0.003 ± 0.001 | 0.004 ± 0.001 |
| cpu/fft | `fft` | c32 | 4 | `1d_n65536` | 0.450 ± 0.002 | 0.218 ± 0.010 | 0.431 ± 0.043 |
| cpu/fft | `fft` | c64 | 4 | `1d_n1024` | 0.008 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.000 |
| cpu/fft | `fft` | c64 | 4 | `1d_n65536` | 0.596 ± 0.010 | 0.365 ± 0.008 | 0.502 ± 0.263 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n1024` | 0.007 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.000 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n65536` | 0.515 ± 0.053 | 0.259 ± 0.036 | 0.451 ± 0.037 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n1024` | 0.008 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.000 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n65536` | 0.629 ± 0.057 | 0.416 ± 0.002 | 0.486 ± 0.030 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n1024` | 0.007 ± 0.000 | 0.003 ± 0.000 | 0.003 ± 0.000 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n65536` | 0.515 ± 0.008 | 0.253 ± 0.013 | 0.193 ± 0.000 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n1024` | 0.008 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n65536` | 0.685 ± 0.075 | 0.471 ± 0.050 | 0.274 ± 0.032 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n1024` | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.003 ± 0.000 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n65536` | 0.490 ± 0.023 | 0.231 ± 0.004 | 0.223 ± 0.037 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n1024` | 0.008 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n65536` | 0.678 ± 0.027 | 0.429 ± 0.005 | 0.259 ± 0.109 |

Notes:

- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs
- tenferro-fft-immediate: one-shot TensorFftExt call; no caller-owned plan cache

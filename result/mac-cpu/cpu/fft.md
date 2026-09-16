# CPU FFT Benchmark Results

- Suite: `cpu/fft`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/fft.yaml`
- Run metadata: `data/results/mac-cpu/cpu/fft/20260916_105338/run.yaml`
- Timestamp: `20260916_105338`

Latest run: `./scripts/run_cpu_fft.sh 1 4`.

This file is generated from sequential CPU FFT runs under `data/results/mac-cpu/cpu/fft/20260916_105338`.

- tenferro-rs commit: `bca2d54a5884586b1aeae8006caa9b8c0ba44a26`

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

- Run metadata: `data/results/mac-cpu/cpu/fft/20260916_105338/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/fft/20260916_105338/run_t4.yaml`
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

- CSV: `data/results/mac-cpu/cpu/fft/20260916_105338/cpu_fft_t1_20260916_105338.csv`
- CSV: `data/results/mac-cpu/cpu/fft/20260916_105338/cpu_fft_t4_20260916_105338.csv`
- Source table: `data/results/mac-cpu/cpu/fft/20260916_105338/cpu_fft_20260916_105338.md`

## CPU FFT Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

Timing scope: input tensors are created outside the timed region; each timed call creates the FFT output tensor. The primary eager comparison is tenferro-rs FftExecutor cached versus warmed PyTorch torch.fft; traced rows reuse one compiled graph and one-shot rows are diagnostic. Rows are limited to 1D transforms so tenferro-rs column-major layout and PyTorch row-major layout do not change the measured transform axis.

### Operation execution

| suite | benchmark | dtype | threads | shape | tenferro-rs one-shot diagnostic (ms) | tenferro-rs TensorRead API (ms) | tenferro-rs cached primary (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch torch.fft (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| cpu/fft | `fft` | c32 | 1 | `1d_n1024` | skipped | skipped | 0.006 ± 0.001 | skipped | skipped | 0.004 ± 0.001 |
| cpu/fft | `fft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 4.024 ± 0.021 | 4.071 ± 0.025 | 4.114 ± 0.056 | 10.519 ± 0.105 |
| cpu/fft | `fft` | c32 | 1 | `1d_n16384` | skipped | skipped | 0.047 ± 0.007 | skipped | skipped | 0.082 ± 0.001 |
| cpu/fft | `fft` | c32 | 4 | `1d_n1024` | skipped | skipped | 0.005 ± 0.002 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `fft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 4.214 ± 0.050 | 4.282 ± 0.022 | 4.273 ± 0.050 | 10.024 ± 0.291 |
| cpu/fft | `fft` | c32 | 4 | `1d_n16384` | skipped | skipped | 0.046 ± 0.006 | skipped | skipped | 0.082 ± 0.001 |
| cpu/fft | `fft` | c64 | 1 | `1d_n1024` | skipped | skipped | 0.007 ± 0.001 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `fft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 7.067 ± 0.401 | 6.577 ± 0.112 | 5.846 ± 0.067 | 11.765 ± 0.290 |
| cpu/fft | `fft` | c64 | 1 | `1d_n16384` | skipped | skipped | 0.070 ± 0.002 | skipped | skipped | 0.104 ± 0.001 |
| cpu/fft | `fft` | c64 | 4 | `1d_n1024` | skipped | skipped | 0.007 ± 0.002 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `fft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 6.607 ± 0.280 | 6.632 ± 0.084 | 5.863 ± 0.106 | 11.644 ± 0.321 |
| cpu/fft | `fft` | c64 | 4 | `1d_n16384` | skipped | skipped | 0.071 ± 0.001 | skipped | skipped | 0.105 ± 0.002 |
| cpu/fft | `ifft` | c32 | 1 | `1d_n1024` | skipped | skipped | 0.004 ± 0.001 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `ifft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 4.174 ± 0.036 | 4.050 ± 0.095 | 4.288 ± 0.036 | 10.125 ± 0.052 |
| cpu/fft | `ifft` | c32 | 1 | `1d_n16384` | skipped | skipped | 0.042 ± 0.001 | skipped | skipped | 0.084 ± 0.001 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n1024` | skipped | skipped | 0.004 ± 0.001 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 4.199 ± 0.073 | 3.773 ± 0.095 | 3.950 ± 0.023 | 10.169 ± 0.346 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n16384` | skipped | skipped | 0.043 ± 0.001 | skipped | skipped | 0.083 ± 0.001 |
| cpu/fft | `ifft` | c64 | 1 | `1d_n1024` | skipped | skipped | 0.004 ± 0.001 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `ifft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 6.056 ± 0.066 | 6.225 ± 0.356 | 7.065 ± 0.136 | 10.850 ± 0.691 |
| cpu/fft | `ifft` | c64 | 1 | `1d_n16384` | skipped | skipped | 0.072 ± 0.001 | skipped | skipped | 0.103 ± 0.001 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n1024` | skipped | skipped | 0.006 ± 0.001 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 6.194 ± 0.138 | 6.126 ± 0.083 | 6.259 ± 0.148 | 10.830 ± 0.101 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n16384` | skipped | skipped | 0.074 ± 0.001 | skipped | skipped | 0.102 ± 0.002 |
| cpu/fft | `irfft` | c32 | 1 | `1d_n1024` | skipped | skipped | 0.004 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `irfft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 4.471 ± 0.055 | 4.375 ± 0.180 | 4.252 ± 0.076 | 3.497 ± 0.015 |
| cpu/fft | `irfft` | c32 | 1 | `1d_n16384` | skipped | skipped | 0.042 ± 0.001 | skipped | skipped | 0.037 ± 0.000 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n1024` | skipped | skipped | 0.004 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 4.184 ± 0.085 | 3.688 ± 0.072 | 3.633 ± 0.073 | 3.493 ± 0.074 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n16384` | skipped | skipped | 0.042 ± 0.001 | skipped | skipped | 0.036 ± 0.000 |
| cpu/fft | `irfft` | c64 | 1 | `1d_n1024` | skipped | skipped | 0.005 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `irfft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 6.306 ± 0.359 | 6.309 ± 0.166 | 6.028 ± 0.185 | 4.486 ± 0.091 |
| cpu/fft | `irfft` | c64 | 1 | `1d_n16384` | skipped | skipped | 0.072 ± 0.001 | skipped | skipped | 0.050 ± 0.002 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n1024` | skipped | skipped | 0.005 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 6.070 ± 0.303 | 6.009 ± 0.429 | 5.888 ± 0.155 | 4.476 ± 0.071 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n16384` | skipped | skipped | 0.075 ± 0.001 | skipped | skipped | 0.049 ± 0.001 |
| cpu/fft | `rfft` | f32 | 1 | `1d_n1024` | skipped | skipped | 0.003 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `rfft` | f32 | 1 | `1d_n1048576` | skipped | skipped | 4.077 ± 0.099 | 4.001 ± 0.147 | 4.029 ± 0.110 | 3.483 ± 0.029 |
| cpu/fft | `rfft` | f32 | 1 | `1d_n16384` | skipped | skipped | 0.039 ± 0.001 | skipped | skipped | 0.037 ± 0.001 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n1024` | skipped | skipped | 0.004 ± 0.001 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n1048576` | skipped | skipped | 3.923 ± 0.065 | 3.429 ± 0.144 | 3.468 ± 0.033 | 3.427 ± 0.052 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n16384` | skipped | skipped | 0.038 ± 0.001 | skipped | skipped | 0.037 ± 0.000 |
| cpu/fft | `rfft` | f64 | 1 | `1d_n1024` | skipped | skipped | 0.005 ± 0.001 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `rfft` | f64 | 1 | `1d_n1048576` | skipped | skipped | 5.693 ± 0.273 | 6.048 ± 0.327 | 6.242 ± 0.261 | 4.305 ± 0.053 |
| cpu/fft | `rfft` | f64 | 1 | `1d_n16384` | skipped | skipped | 0.069 ± 0.002 | skipped | skipped | 0.049 ± 0.000 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n1024` | skipped | skipped | 0.005 ± 0.001 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n1048576` | skipped | skipped | 6.055 ± 0.246 | 5.694 ± 0.294 | 6.455 ± 0.216 | 4.341 ± 0.058 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n16384` | skipped | skipped | 0.067 ± 0.001 | skipped | skipped | 0.049 ± 0.001 |

Notes:

- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.320750
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.321292
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.325041
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.335000
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.336875
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.338375
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.338666
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.353333
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.464250
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.481375
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.484292
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.485875
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.494625
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.495458
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.496709
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.510500
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=10.475500
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=10.550166
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=10.673666
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=10.705875
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=13.073500
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=13.166584
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=13.334708
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=13.395250
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=4.654125
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=4.684500
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=4.741667
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=4.782958
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=6.211208
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=6.232208
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=6.274500
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=6.341709
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.024000
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.125125
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.169042
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.518792
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.830083
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.849500
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=11.643667
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=11.765500
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=3.426875
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=3.482542
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=3.492792
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=3.497125
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=4.304709
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=4.341167
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=4.476375
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=4.485875
- tenferro-fft-eager: API includes per-call planning or session entry; use shared cached executor for short operation timing
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.428875
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.688083
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.772958
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=4.001125
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=4.050125
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=4.071334
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=4.281792
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=4.375500
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=5.694000
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.009250
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.048416
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.126083
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.224917
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.308917
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.576541
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.631791
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.403292
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.448125
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.472167
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.479333
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.483583
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.508584
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.531250
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.596625
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.598916
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.649084
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.649625
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.666208
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.706250
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.784333
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.836666
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.859083
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=4.917333
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=4.929791
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.337708
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.381125
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.428833
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.488042
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.849500
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.954958
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=8.567833
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=8.792833
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=8.939875
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=9.094625
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=9.264083
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=9.266125
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=9.520083
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=9.574708
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.922917
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=4.023750
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=4.076625
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=4.174083
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=4.184291
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=4.199375
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=4.213583
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=4.471084
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=5.692625
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.054625
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.056000
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.070084
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.194166
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.306333
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.607458
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=7.067000
- tenferro-fft-immediate: API includes per-call planning or session entry; use shared cached executor for short operation timing
- tenferro-fft-read: API includes per-call planning or session entry; use shared cached executor for short operation timing
- tenferro-fft-trace: API includes per-call planning or session entry; use shared cached executor for short operation timing
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=3.468250
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=3.633209
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=3.950458
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.029208
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.114333
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.251584
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.272709
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.288458
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=5.845833
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=5.862958
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=5.887958
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.027958
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.241875
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.258875
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.455167
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=7.065125

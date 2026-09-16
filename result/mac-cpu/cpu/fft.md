# CPU FFT Benchmark Results

- Suite: `cpu/fft`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/fft.yaml`
- Run metadata: `data/results/mac-cpu/cpu/fft/20260916_165724/run.yaml`
- Timestamp: `20260916_165724`

Latest run: `./scripts/run_cpu_fft.sh 1 4`.

This file is generated from sequential CPU FFT runs under `data/results/mac-cpu/cpu/fft/20260916_165724`.

- tenferro-rs commit: `d8759f4320a337d2399f4a87dfec55af51d2ebf1`

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

- Run metadata: `data/results/mac-cpu/cpu/fft/20260916_165724/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/fft/20260916_165724/run_t4.yaml`
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

- CSV: `data/results/mac-cpu/cpu/fft/20260916_165724/cpu_fft_t1_20260916_165724.csv`
- CSV: `data/results/mac-cpu/cpu/fft/20260916_165724/cpu_fft_t4_20260916_165724.csv`
- Source table: `data/results/mac-cpu/cpu/fft/20260916_165724/cpu_fft_20260916_165724.md`

## CPU FFT Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

Timing scope: input tensors are created outside the timed region; each timed call creates the FFT output tensor. The primary eager comparison is tenferro-rs FftExecutor cached versus warmed PyTorch torch.fft; traced rows reuse one compiled graph and one-shot rows are diagnostic. Rows are limited to 1D transforms so tenferro-rs column-major layout and PyTorch row-major layout do not change the measured transform axis.

### Operation execution

| suite | benchmark | dtype | threads | shape | tenferro-rs one-shot diagnostic (ms) | tenferro-rs TensorRead API (ms) | tenferro-rs cached primary (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch torch.fft (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| cpu/fft | `fft` | c32 | 1 | `1d_n1024` | skipped | skipped | 0.005 ± 0.002 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `fft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 4.299 ± 0.175 | 3.955 ± 0.117 | 4.210 ± 0.126 | 10.317 ± 0.173 |
| cpu/fft | `fft` | c32 | 1 | `1d_n16384` | skipped | skipped | 0.041 ± 0.001 | skipped | skipped | 0.087 ± 0.004 |
| cpu/fft | `fft` | c32 | 4 | `1d_n1024` | skipped | skipped | 0.005 ± 0.002 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `fft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 4.182 ± 0.117 | 4.209 ± 0.075 | 4.173 ± 0.087 | 10.295 ± 0.164 |
| cpu/fft | `fft` | c32 | 4 | `1d_n16384` | skipped | skipped | 0.042 ± 0.001 | skipped | skipped | 0.084 ± 0.001 |
| cpu/fft | `fft` | c64 | 1 | `1d_n1024` | skipped | skipped | 0.006 ± 0.001 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `fft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 7.728 ± 0.309 | 7.205 ± 0.197 | 6.133 ± 0.237 | 11.941 ± 0.295 |
| cpu/fft | `fft` | c64 | 1 | `1d_n16384` | skipped | skipped | 0.070 ± 0.001 | skipped | skipped | 0.107 ± 0.003 |
| cpu/fft | `fft` | c64 | 4 | `1d_n1024` | skipped | skipped | 0.006 ± 0.001 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `fft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 6.895 ± 0.210 | 6.914 ± 0.190 | 6.086 ± 0.133 | 12.049 ± 0.251 |
| cpu/fft | `fft` | c64 | 4 | `1d_n16384` | skipped | skipped | 0.072 ± 0.002 | skipped | skipped | 0.106 ± 0.001 |
| cpu/fft | `ifft` | c32 | 1 | `1d_n1024` | skipped | skipped | 0.004 ± 0.001 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `ifft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 4.255 ± 0.069 | 4.438 ± 0.125 | 4.107 ± 0.097 | 10.768 ± 1.149 |
| cpu/fft | `ifft` | c32 | 1 | `1d_n16384` | skipped | skipped | 0.042 ± 0.000 | skipped | skipped | 0.084 ± 0.001 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n1024` | skipped | skipped | 0.004 ± 0.001 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 3.769 ± 0.122 | 4.375 ± 0.111 | 4.374 ± 0.115 | 10.229 ± 0.215 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n16384` | skipped | skipped | 0.043 ± 0.001 | skipped | skipped | 0.087 ± 0.002 |
| cpu/fft | `ifft` | c64 | 1 | `1d_n1024` | skipped | skipped | 0.005 ± 0.000 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `ifft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 6.395 ± 0.170 | 6.504 ± 0.257 | 7.075 ± 0.285 | 11.249 ± 0.333 |
| cpu/fft | `ifft` | c64 | 1 | `1d_n16384` | skipped | skipped | 0.074 ± 0.001 | skipped | skipped | 0.103 ± 0.001 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n1024` | skipped | skipped | 0.005 ± 0.000 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 6.425 ± 0.195 | 6.543 ± 0.172 | 7.307 ± 0.190 | 11.244 ± 0.438 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n16384` | skipped | skipped | 0.075 ± 0.001 | skipped | skipped | 0.106 ± 0.002 |
| cpu/fft | `irfft` | c32 | 1 | `1d_n1024` | skipped | skipped | 0.003 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `irfft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 4.006 ± 1.497 | 5.638 ± 0.434 | 5.761 ± 0.923 | 3.533 ± 0.045 |
| cpu/fft | `irfft` | c32 | 1 | `1d_n16384` | skipped | skipped | 0.043 ± 0.001 | skipped | skipped | 0.037 ± 0.000 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n1024` | skipped | skipped | 0.003 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 4.264 ± 0.078 | 3.946 ± 0.129 | 3.869 ± 0.234 | 3.484 ± 0.044 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n16384` | skipped | skipped | 0.043 ± 0.001 | skipped | skipped | 0.037 ± 0.001 |
| cpu/fft | `irfft` | c64 | 1 | `1d_n1024` | skipped | skipped | 0.004 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `irfft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 10.208 ± 1.673 | 6.834 ± 0.313 | 7.467 ± 0.259 | 6.333 ± 0.795 |
| cpu/fft | `irfft` | c64 | 1 | `1d_n16384` | skipped | skipped | 0.075 ± 0.001 | skipped | skipped | 0.049 ± 0.001 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n1024` | skipped | skipped | 0.004 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 6.329 ± 0.222 | 6.342 ± 0.120 | 6.348 ± 0.085 | 4.523 ± 0.114 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n16384` | skipped | skipped | 0.075 ± 0.000 | skipped | skipped | 0.050 ± 0.000 |
| cpu/fft | `rfft` | f32 | 1 | `1d_n1024` | skipped | skipped | 0.003 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `rfft` | f32 | 1 | `1d_n1048576` | skipped | skipped | 4.305 ± 0.311 | 4.095 ± 0.109 | 3.603 ± 0.094 | 3.424 ± 0.064 |
| cpu/fft | `rfft` | f32 | 1 | `1d_n16384` | skipped | skipped | 0.039 ± 0.001 | skipped | skipped | 0.038 ± 0.000 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n1024` | skipped | skipped | 0.003 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n1048576` | skipped | skipped | 3.992 ± 0.142 | 3.493 ± 0.087 | 3.569 ± 0.122 | 3.566 ± 0.133 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n16384` | skipped | skipped | 0.040 ± 0.001 | skipped | skipped | 0.038 ± 0.001 |
| cpu/fft | `rfft` | f64 | 1 | `1d_n1024` | skipped | skipped | 0.004 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `rfft` | f64 | 1 | `1d_n1048576` | skipped | skipped | 6.821 ± 0.187 | 6.081 ± 0.123 | 7.098 ± 0.153 | 4.615 ± 0.322 |
| cpu/fft | `rfft` | f64 | 1 | `1d_n16384` | skipped | skipped | 0.069 ± 0.001 | skipped | skipped | 0.049 ± 0.000 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n1024` | skipped | skipped | 0.004 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n1048576` | skipped | skipped | 6.888 ± 0.259 | 6.060 ± 0.135 | 6.286 ± 0.163 | 4.476 ± 0.087 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n16384` | skipped | skipped | 0.069 ± 0.001 | skipped | skipped | 0.049 ± 0.001 |

Notes:

- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.322083
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.330000
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.330584
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.335792
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.343959
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.346583
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.351333
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.364459
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.477250
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.482208
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.483125
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.494458
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.499541
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.505959
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.511417
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.531375
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=10.691667
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=10.806084
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=11.081125
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=11.097500
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=13.223041
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=13.579417
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=13.597792
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=13.669375
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=4.690667
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=4.713750
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=4.850208
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=4.893125
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=6.248708
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=6.254833
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=6.313916
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=6.339000
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.229250
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.294791
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.316584
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.767834
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=11.244458
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=11.249375
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=11.940667
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=12.048958
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=3.423750
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=3.483625
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=3.532917
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=3.566084
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=4.476375
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=4.523000
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=4.614958
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=6.333167
- tenferro-fft-eager: API includes per-call planning or session entry; use shared cached executor for short operation timing
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.493208
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.946458
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.954875
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=4.095042
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=4.209125
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=4.374833
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=4.437708
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=5.637917
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.059750
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.080917
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.342333
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.503833
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.542959
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.833584
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.913709
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=7.205500
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.409083
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.426458
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.428375
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.436209
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.529250
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.534583
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.539375
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.548667
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.551375
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.556042
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.643458
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.652625
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.653166
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.658083
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.747709
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.763291
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=4.958416
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.088625
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.249625
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.341333
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.404625
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.521708
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.560667
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.567084
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=8.850500
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=8.865500
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=9.019250
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=9.174042
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=9.464125
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=9.568625
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=9.610625
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=9.618417
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=10.208125
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.769167
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.992375
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=4.006125
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=4.182084
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=4.254833
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=4.264416
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=4.299125
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=4.304625
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.328500
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.395167
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.425125
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.821167
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.888083
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.894750
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=7.727917
- tenferro-fft-immediate: API includes per-call planning or session entry; use shared cached executor for short operation timing
- tenferro-fft-read: API includes per-call planning or session entry; use shared cached executor for short operation timing
- tenferro-fft-trace: API includes per-call planning or session entry; use shared cached executor for short operation timing
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=3.568959
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=3.603167
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=3.869000
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.106708
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.172833
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.210334
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.373625
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=5.760666
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.086334
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.133208
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.285667
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.348500
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=7.074750
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=7.097833
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=7.307084
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=7.467292

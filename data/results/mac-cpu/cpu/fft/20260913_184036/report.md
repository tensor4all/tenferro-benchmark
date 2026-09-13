# CPU FFT Benchmark Results

- Suite: `cpu/fft`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/fft.yaml`
- Run metadata: `data/results/mac-cpu/cpu/fft/20260913_184036/run.yaml`
- Timestamp: `20260913_184036`

Latest run: `./scripts/run_cpu_fft.sh 1 4`.

This file is generated from sequential CPU FFT runs under `data/results/mac-cpu/cpu/fft/20260913_184036`.

- tenferro-rs commit: `28dfc7e383a653e1461d38b73ced3d563b4efdad`

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

- Run metadata: `data/results/mac-cpu/cpu/fft/20260913_184036/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/fft/20260913_184036/run_t4.yaml`
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

- CSV: `data/results/mac-cpu/cpu/fft/20260913_184036/cpu_fft_t1_20260913_184036.csv`
- CSV: `data/results/mac-cpu/cpu/fft/20260913_184036/cpu_fft_t4_20260913_184036.csv`
- Source table: `data/results/mac-cpu/cpu/fft/20260913_184036/cpu_fft_20260913_184036.md`

## CPU FFT Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

Timing scope: input tensors are created outside the timed region; each timed call creates the FFT output tensor. The primary eager comparison is tenferro-rs FftExecutor cached versus warmed PyTorch torch.fft; traced rows reuse one compiled graph and one-shot rows are diagnostic. Rows are limited to 1D transforms so tenferro-rs column-major layout and PyTorch row-major layout do not change the measured transform axis.

### Operation execution

| suite | benchmark | dtype | threads | shape | tenferro-rs one-shot diagnostic (ms) | tenferro-rs TensorRead API (ms) | tenferro-rs cached primary (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch torch.fft (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| cpu/fft | `fft` | c32 | 1 | `1d_n1024` | skipped | skipped | 0.005 ± 0.002 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `fft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 3.998 ± 0.055 | 4.197 ± 0.101 | 4.206 ± 0.103 | 10.477 ± 0.204 |
| cpu/fft | `fft` | c32 | 1 | `1d_n16384` | skipped | skipped | 0.042 ± 0.001 | skipped | skipped | 0.085 ± 0.002 |
| cpu/fft | `fft` | c32 | 4 | `1d_n1024` | skipped | skipped | 0.005 ± 0.001 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `fft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 4.236 ± 0.083 | 4.234 ± 0.107 | 4.176 ± 0.114 | 10.448 ± 0.198 |
| cpu/fft | `fft` | c32 | 4 | `1d_n16384` | skipped | skipped | 0.041 ± 0.001 | skipped | skipped | 0.083 ± 0.001 |
| cpu/fft | `fft` | c64 | 1 | `1d_n1024` | skipped | skipped | 0.006 ± 0.001 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `fft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 6.916 ± 0.260 | 6.922 ± 0.206 | 6.126 ± 0.167 | 11.765 ± 0.330 |
| cpu/fft | `fft` | c64 | 1 | `1d_n16384` | skipped | skipped | 0.071 ± 0.001 | skipped | skipped | 0.107 ± 0.001 |
| cpu/fft | `fft` | c64 | 4 | `1d_n1024` | skipped | skipped | 0.005 ± 0.001 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `fft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 7.094 ± 0.187 | 7.082 ± 0.177 | 6.189 ± 0.221 | 11.870 ± 0.289 |
| cpu/fft | `fft` | c64 | 4 | `1d_n16384` | skipped | skipped | 0.070 ± 0.000 | skipped | skipped | 0.106 ± 0.001 |
| cpu/fft | `ifft` | c32 | 1 | `1d_n1024` | skipped | skipped | 0.004 ± 0.000 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `ifft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 4.256 ± 0.076 | 3.839 ± 0.101 | 4.188 ± 0.061 | 10.242 ± 0.194 |
| cpu/fft | `ifft` | c32 | 1 | `1d_n16384` | skipped | skipped | 0.042 ± 0.001 | skipped | skipped | 0.086 ± 0.000 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n1024` | skipped | skipped | 0.004 ± 0.001 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 4.315 ± 0.095 | 3.865 ± 0.043 | 4.287 ± 0.074 | 10.221 ± 0.180 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n16384` | skipped | skipped | 0.043 ± 0.001 | skipped | skipped | 0.084 ± 0.001 |
| cpu/fft | `ifft` | c64 | 1 | `1d_n1024` | skipped | skipped | 0.005 ± 0.000 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `ifft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 6.252 ± 0.162 | 6.471 ± 0.186 | 6.415 ± 0.131 | 11.113 ± 0.493 |
| cpu/fft | `ifft` | c64 | 1 | `1d_n16384` | skipped | skipped | 0.074 ± 0.000 | skipped | skipped | 0.106 ± 0.001 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n1024` | skipped | skipped | 0.005 ± 0.000 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 6.262 ± 0.144 | 6.369 ± 0.135 | 7.277 ± 0.222 | 11.361 ± 0.250 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n16384` | skipped | skipped | 0.075 ± 0.001 | skipped | skipped | 0.104 ± 0.001 |
| cpu/fft | `irfft` | c32 | 1 | `1d_n1024` | skipped | skipped | 0.003 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `irfft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 4.264 ± 0.082 | 4.155 ± 0.059 | 4.291 ± 0.126 | 3.524 ± 0.074 |
| cpu/fft | `irfft` | c32 | 1 | `1d_n16384` | skipped | skipped | 0.043 ± 0.001 | skipped | skipped | 0.037 ± 0.000 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n1024` | skipped | skipped | 0.003 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 4.287 ± 0.079 | 3.800 ± 0.076 | 4.268 ± 0.066 | 3.517 ± 0.037 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n16384` | skipped | skipped | 0.042 ± 0.001 | skipped | skipped | 0.037 ± 0.000 |
| cpu/fft | `irfft` | c64 | 1 | `1d_n1024` | skipped | skipped | 0.004 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `irfft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 6.562 ± 0.248 | 6.378 ± 0.203 | 7.300 ± 0.334 | 4.549 ± 0.064 |
| cpu/fft | `irfft` | c64 | 1 | `1d_n16384` | skipped | skipped | 0.075 ± 0.001 | skipped | skipped | 0.050 ± 0.000 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n1024` | skipped | skipped | 0.004 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 6.490 ± 0.078 | 6.480 ± 0.112 | 7.242 ± 0.176 | 4.568 ± 0.038 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n16384` | skipped | skipped | 0.075 ± 0.001 | skipped | skipped | 0.049 ± 0.000 |
| cpu/fft | `rfft` | f32 | 1 | `1d_n1024` | skipped | skipped | 0.003 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `rfft` | f32 | 1 | `1d_n1048576` | skipped | skipped | 4.053 ± 0.100 | 4.052 ± 0.085 | 3.863 ± 0.066 | 3.443 ± 0.034 |
| cpu/fft | `rfft` | f32 | 1 | `1d_n16384` | skipped | skipped | 0.040 ± 0.002 | skipped | skipped | 0.038 ± 0.000 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n1024` | skipped | skipped | 0.003 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n1048576` | skipped | skipped | 4.052 ± 0.090 | 4.015 ± 0.104 | 3.548 ± 0.066 | 3.486 ± 0.063 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n16384` | skipped | skipped | 0.039 ± 0.000 | skipped | skipped | 0.038 ± 0.000 |
| cpu/fft | `rfft` | f64 | 1 | `1d_n1024` | skipped | skipped | 0.004 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `rfft` | f64 | 1 | `1d_n1048576` | skipped | skipped | 6.749 ± 0.140 | 6.179 ± 0.141 | 6.018 ± 0.094 | 4.369 ± 0.068 |
| cpu/fft | `rfft` | f64 | 1 | `1d_n16384` | skipped | skipped | 0.069 ± 0.000 | skipped | skipped | 0.049 ± 0.000 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n1024` | skipped | skipped | 0.004 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n1048576` | skipped | skipped | 6.769 ± 0.139 | 6.159 ± 0.172 | 6.801 ± 0.276 | 4.347 ± 0.068 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n16384` | skipped | skipped | 0.069 ± 0.001 | skipped | skipped | 0.049 ± 0.000 |

Notes:

- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.322083
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.327209
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.333542
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.342042
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.346125
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.348833
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.352917
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.360166
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.481000
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.483375
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.485083
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.485292
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.497250
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.500083
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.512834
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.516458
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=10.661833
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=10.814125
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=10.818000
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=10.973750
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=13.290625
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=13.536042
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=13.613625
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=13.746542
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=4.694917
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=4.695625
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=4.817333
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=4.825583
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=6.235833
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=6.249500
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=6.335792
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=6.352041
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.221167
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.241625
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.448333
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.477042
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=11.112625
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=11.360667
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=11.764750
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=11.869584
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=3.443042
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=3.485625
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=3.517375
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=3.523666
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=4.346625
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=4.368500
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=4.548875
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=4.567542
- tenferro-fft-eager: API includes per-call planning or session entry; use shared cached executor for short operation timing
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.800166
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.838541
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.864542
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=4.014791
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=4.052042
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=4.154875
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=4.196958
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=4.233583
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.159208
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.179458
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.368959
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.378459
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.471125
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.479542
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.921875
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=7.082458
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.407333
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.410334
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.412083
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.431250
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.497666
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.515583
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.528084
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.528917
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.533750
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.538584
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.633750
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.636250
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.637333
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.654500
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.675791
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.743458
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=4.971334
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.179875
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.244833
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.355792
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.419041
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.439542
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.446041
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.460459
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=8.773584
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=8.803833
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=8.915000
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=9.040417
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=9.464292
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=9.542667
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=9.551833
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=9.563166
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.998291
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=4.051708
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=4.052917
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=4.236250
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=4.255833
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=4.264250
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=4.287000
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=4.314833
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.251958
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.261875
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.489667
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.561542
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.748792
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.768917
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.916083
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=7.094000
- tenferro-fft-immediate: API includes per-call planning or session entry; use shared cached executor for short operation timing
- tenferro-fft-read: API includes per-call planning or session entry; use shared cached executor for short operation timing
- tenferro-fft-trace: API includes per-call planning or session entry; use shared cached executor for short operation timing
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=3.548125
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=3.862625
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.175625
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.188458
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.206083
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.268000
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.287166
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.291334
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.017583
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.126459
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.189292
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.415000
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.801375
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=7.241666
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=7.277500
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=7.300083

# CPU FFT Benchmark Results

- Suite: `cpu/fft`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/fft.yaml`
- Run metadata: `data/results/mac-cpu/cpu/fft/20260923_205931/run.yaml`
- Timestamp: `20260923_205931`

Latest run: `./scripts/run_cpu_fft.sh 1 4`.

This file is generated from sequential CPU FFT runs under `data/results/mac-cpu/cpu/fft/20260923_205931`.

- tenferro-rs commit: `8a1839febeb3c868a502e26397ca10761bbc568d`

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

- Run metadata: `data/results/mac-cpu/cpu/fft/20260923_205931/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/fft/20260923_205931/run_t4.yaml`
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

- CSV: `data/results/mac-cpu/cpu/fft/20260923_205931/cpu_fft_t1_20260923_205931.csv`
- CSV: `data/results/mac-cpu/cpu/fft/20260923_205931/cpu_fft_t4_20260923_205931.csv`
- Source table: `data/results/mac-cpu/cpu/fft/20260923_205931/cpu_fft_20260923_205931.md`

## CPU FFT Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

Timing scope: input tensors are created outside the timed region; each timed call creates the FFT output tensor. The primary eager comparison is tenferro-rs FftExecutor cached versus warmed PyTorch torch.fft; traced rows reuse one compiled graph and one-shot rows are diagnostic. Rows are limited to 1D transforms so tenferro-rs column-major layout and PyTorch row-major layout do not change the measured transform axis.

### Operation execution

| suite | benchmark | dtype | threads | shape | tenferro-rs one-shot diagnostic (ms) | tenferro-rs TensorRead API (ms) | tenferro-rs cached primary (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch torch.fft (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| cpu/fft | `fft` | c32 | 1 | `1d_n1024` | skipped | skipped | 0.002 ± 0.000 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `fft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 4.094 ± 0.114 | 4.148 ± 0.102 | 4.125 ± 0.101 | 10.358 ± 0.172 |
| cpu/fft | `fft` | c32 | 1 | `1d_n16384` | skipped | skipped | 0.041 ± 0.001 | skipped | skipped | 0.082 ± 0.001 |
| cpu/fft | `fft` | c32 | 4 | `1d_n1024` | skipped | skipped | 0.005 ± 0.001 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `fft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 4.210 ± 0.077 | 4.194 ± 0.056 | 4.124 ± 0.046 | 10.285 ± 0.143 |
| cpu/fft | `fft` | c32 | 4 | `1d_n16384` | skipped | skipped | 0.040 ± 0.003 | skipped | skipped | 0.083 ± 0.001 |
| cpu/fft | `fft` | c64 | 1 | `1d_n1024` | skipped | skipped | 0.003 ± 0.000 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `fft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 6.621 ± 0.172 | 6.600 ± 0.082 | 5.861 ± 0.062 | 11.412 ± 0.196 |
| cpu/fft | `fft` | c64 | 1 | `1d_n16384` | skipped | skipped | 0.069 ± 0.001 | skipped | skipped | 0.105 ± 0.000 |
| cpu/fft | `fft` | c64 | 4 | `1d_n1024` | skipped | skipped | 0.005 ± 0.001 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `fft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 5.868 ± 0.093 | 5.858 ± 0.334 | 5.802 ± 0.050 | 11.589 ± 0.358 |
| cpu/fft | `fft` | c64 | 4 | `1d_n16384` | skipped | skipped | 0.069 ± 0.006 | skipped | skipped | 0.105 ± 0.001 |
| cpu/fft | `ifft` | c32 | 1 | `1d_n1024` | skipped | skipped | 0.002 ± 0.000 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `ifft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 3.753 ± 0.033 | 3.753 ± 0.078 | 3.865 ± 0.060 | 10.069 ± 0.122 |
| cpu/fft | `ifft` | c32 | 1 | `1d_n16384` | skipped | skipped | 0.042 ± 0.001 | skipped | skipped | 0.084 ± 0.001 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n1024` | skipped | skipped | 0.004 ± 0.000 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 4.248 ± 0.184 | 4.218 ± 0.079 | 4.178 ± 0.039 | 10.201 ± 0.321 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n16384` | skipped | skipped | 0.042 ± 0.000 | skipped | skipped | 0.083 ± 0.001 |
| cpu/fft | `ifft` | c64 | 1 | `1d_n1024` | skipped | skipped | 0.003 ± 0.000 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `ifft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 6.049 ± 0.111 | 6.052 ± 0.153 | 6.811 ± 0.078 | 11.239 ± 0.506 |
| cpu/fft | `ifft` | c64 | 1 | `1d_n16384` | skipped | skipped | 0.073 ± 0.000 | skipped | skipped | 0.103 ± 0.001 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n1024` | skipped | skipped | 0.005 ± 0.001 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 5.955 ± 0.136 | 5.994 ± 0.035 | 6.177 ± 0.200 | 10.865 ± 0.861 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n16384` | skipped | skipped | 0.072 ± 0.000 | skipped | skipped | 0.102 ± 0.001 |
| cpu/fft | `irfft` | c32 | 1 | `1d_n1024` | skipped | skipped | 0.002 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `irfft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 3.578 ± 0.069 | 3.618 ± 0.052 | 4.207 ± 0.065 | 3.547 ± 0.066 |
| cpu/fft | `irfft` | c32 | 1 | `1d_n16384` | skipped | skipped | 0.042 ± 0.000 | skipped | skipped | 0.037 ± 0.000 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n1024` | skipped | skipped | 0.003 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 3.626 ± 0.048 | 3.633 ± 0.031 | 4.311 ± 0.052 | 3.484 ± 0.028 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n16384` | skipped | skipped | 0.042 ± 0.001 | skipped | skipped | 0.037 ± 0.000 |
| cpu/fft | `irfft` | c64 | 1 | `1d_n1024` | skipped | skipped | 0.003 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `irfft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 5.923 ± 0.119 | 5.944 ± 0.119 | 5.951 ± 0.052 | 4.510 ± 0.090 |
| cpu/fft | `irfft` | c64 | 1 | `1d_n16384` | skipped | skipped | 0.073 ± 0.000 | skipped | skipped | 0.049 ± 0.000 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n1024` | skipped | skipped | 0.004 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 5.900 ± 0.085 | 5.956 ± 0.320 | 6.238 ± 0.303 | 4.594 ± 0.117 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n16384` | skipped | skipped | 0.072 ± 0.001 | skipped | skipped | 0.049 ± 0.000 |
| cpu/fft | `rfft` | f32 | 1 | `1d_n1024` | skipped | skipped | 0.002 ± 0.000 | skipped | skipped | 0.002 ± 0.000 |
| cpu/fft | `rfft` | f32 | 1 | `1d_n1048576` | skipped | skipped | 3.946 ± 0.065 | 3.917 ± 0.055 | 3.421 ± 0.085 | 3.472 ± 0.171 |
| cpu/fft | `rfft` | f32 | 1 | `1d_n16384` | skipped | skipped | 0.039 ± 0.000 | skipped | skipped | 0.037 ± 0.000 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n1024` | skipped | skipped | 0.003 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n1048576` | skipped | skipped | 3.959 ± 0.040 | 3.930 ± 0.105 | 3.397 ± 0.043 | 3.432 ± 0.070 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n16384` | skipped | skipped | 0.039 ± 0.000 | skipped | skipped | 0.037 ± 0.000 |
| cpu/fft | `rfft` | f64 | 1 | `1d_n1024` | skipped | skipped | 0.003 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `rfft` | f64 | 1 | `1d_n1048576` | skipped | skipped | 6.449 ± 0.108 | 5.616 ± 0.093 | 5.773 ± 0.086 | 4.333 ± 0.091 |
| cpu/fft | `rfft` | f64 | 1 | `1d_n16384` | skipped | skipped | 0.067 ± 0.001 | skipped | skipped | 0.048 ± 0.000 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n1024` | skipped | skipped | 0.004 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n1048576` | skipped | skipped | 6.405 ± 0.125 | 5.600 ± 0.238 | 5.734 ± 0.071 | 4.392 ± 0.019 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n16384` | skipped | skipped | 0.068 ± 0.001 | skipped | skipped | 0.048 ± 0.000 |

Notes:

- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.318126
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.328916
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.334083
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.335250
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.339084
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.339333
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.349834
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.353166
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.470625
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.479250
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.482707
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.484083
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.493666
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.497125
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.510333
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.514292
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=10.479625
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=10.629417
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=10.667041
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=10.741876
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=13.106625
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=13.176875
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=13.386958
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=13.406833
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=4.682666
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=4.713959
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=4.770625
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=4.773417
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=6.157543
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=6.191875
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=6.270333
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=6.273917
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.068834
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.201417
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.284542
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.357710
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.864583
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=11.239375
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=11.411667
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=11.588667
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=3.432042
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=3.472376
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=3.483709
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=3.546959
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=4.333333
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=4.392083
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=4.509917
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=4.594208
- tenferro-fft-eager: API includes per-call planning or session entry; use shared cached executor for short operation timing
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.618000
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.632667
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.752666
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.917416
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.929792
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=4.148417
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=4.193833
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=4.218417
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=5.600208
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=5.616292
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=5.858084
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=5.943625
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=5.956250
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=5.994250
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.052417
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.600041
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.283875
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.306334
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.307542
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.316083
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.341584
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.378042
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.391667
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.421625
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.428041
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.436459
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.470250
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.488416
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.490792
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.581208
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.618791
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.679500
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=4.977292
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.002500
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.134208
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.290875
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.338584
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.368792
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.378250
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.387541
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=8.579750
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=8.743083
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=8.819084
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=8.857167
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=9.216875
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=9.253500
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=9.286875
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=9.286916
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.578292
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.625792
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.752625
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.945750
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.959250
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=4.094333
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=4.209958
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=4.247709
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=5.867708
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=5.900250
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=5.923042
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=5.954792
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.049333
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.405250
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.449458
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.621333
- tenferro-fft-immediate: API includes per-call planning or session entry; use shared cached executor for short operation timing
- tenferro-fft-read: API includes per-call planning or session entry; use shared cached executor for short operation timing
- tenferro-fft-trace: API includes per-call planning or session entry; use shared cached executor for short operation timing
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=3.397334
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=3.420875
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=3.865500
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.123750
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.125375
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.177583
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.207083
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.310750
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=5.733667
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=5.773292
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=5.801834
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=5.861250
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=5.951125
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.177333
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.237875
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.810583

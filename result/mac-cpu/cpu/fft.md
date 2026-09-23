# CPU FFT Benchmark Results

- Suite: `cpu/fft`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/fft.yaml`
- Run metadata: `data/results/mac-cpu/cpu/fft/20260923_181222/run.yaml`
- Timestamp: `20260923_181222`

Latest run: `./scripts/run_cpu_fft.sh 1 4`.

This file is generated from sequential CPU FFT runs under `data/results/mac-cpu/cpu/fft/20260923_181222`.

- tenferro-rs commit: `412852a8cfb13b14246443fc09b1c0c34f5ce1cd`

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

- Run metadata: `data/results/mac-cpu/cpu/fft/20260923_181222/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/fft/20260923_181222/run_t4.yaml`
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

- CSV: `data/results/mac-cpu/cpu/fft/20260923_181222/cpu_fft_t1_20260923_181222.csv`
- CSV: `data/results/mac-cpu/cpu/fft/20260923_181222/cpu_fft_t4_20260923_181222.csv`
- Source table: `data/results/mac-cpu/cpu/fft/20260923_181222/cpu_fft_20260923_181222.md`

## CPU FFT Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

Timing scope: input tensors are created outside the timed region; each timed call creates the FFT output tensor. The primary eager comparison is tenferro-rs FftExecutor cached versus warmed PyTorch torch.fft; traced rows reuse one compiled graph and one-shot rows are diagnostic. Rows are limited to 1D transforms so tenferro-rs column-major layout and PyTorch row-major layout do not change the measured transform axis.

### Operation execution

| suite | benchmark | dtype | threads | shape | tenferro-rs one-shot diagnostic (ms) | tenferro-rs TensorRead API (ms) | tenferro-rs cached primary (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch torch.fft (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| cpu/fft | `fft` | c32 | 1 | `1d_n1024` | skipped | skipped | 0.004 ± 0.000 | skipped | skipped | 0.004 ± 0.001 |
| cpu/fft | `fft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 4.077 ± 0.049 | 4.087 ± 0.053 | 4.082 ± 0.050 | 10.492 ± 0.245 |
| cpu/fft | `fft` | c32 | 1 | `1d_n16384` | skipped | skipped | 0.040 ± 0.000 | skipped | skipped | 0.086 ± 0.007 |
| cpu/fft | `fft` | c32 | 4 | `1d_n1024` | skipped | skipped | 0.002 ± 0.000 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `fft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 4.182 ± 0.145 | 4.274 ± 0.265 | 4.160 ± 0.106 | 10.260 ± 0.149 |
| cpu/fft | `fft` | c32 | 4 | `1d_n16384` | skipped | skipped | 0.041 ± 0.002 | skipped | skipped | 0.084 ± 0.008 |
| cpu/fft | `fft` | c64 | 1 | `1d_n1024` | skipped | skipped | 0.004 ± 0.000 | skipped | skipped | 0.004 ± 0.001 |
| cpu/fft | `fft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 6.543 ± 0.078 | 6.598 ± 0.088 | 5.887 ± 0.095 | 11.683 ± 0.576 |
| cpu/fft | `fft` | c64 | 1 | `1d_n16384` | skipped | skipped | 0.068 ± 0.000 | skipped | skipped | 0.107 ± 0.004 |
| cpu/fft | `fft` | c64 | 4 | `1d_n1024` | skipped | skipped | 0.004 ± 0.003 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `fft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 6.789 ± 0.237 | 7.069 ± 0.240 | 6.174 ± 0.153 | 11.661 ± 0.155 |
| cpu/fft | `fft` | c64 | 4 | `1d_n16384` | skipped | skipped | 0.070 ± 0.003 | skipped | skipped | 0.107 ± 0.001 |
| cpu/fft | `ifft` | c32 | 1 | `1d_n1024` | skipped | skipped | 0.003 ± 0.000 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `ifft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 4.169 ± 0.084 | 4.186 ± 0.065 | 4.198 ± 0.043 | 10.108 ± 0.106 |
| cpu/fft | `ifft` | c32 | 1 | `1d_n16384` | skipped | skipped | 0.042 ± 0.001 | skipped | skipped | 0.085 ± 0.002 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n1024` | skipped | skipped | 0.003 ± 0.001 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 3.734 ± 0.069 | 3.820 ± 0.130 | 3.782 ± 0.115 | 10.353 ± 0.295 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n16384` | skipped | skipped | 0.042 ± 0.001 | skipped | skipped | 0.086 ± 0.001 |
| cpu/fft | `ifft` | c64 | 1 | `1d_n1024` | skipped | skipped | 0.004 ± 0.000 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `ifft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 6.116 ± 0.079 | 6.162 ± 0.082 | 6.925 ± 0.205 | 11.110 ± 0.628 |
| cpu/fft | `ifft` | c64 | 1 | `1d_n16384` | skipped | skipped | 0.072 ± 0.001 | skipped | skipped | 0.103 ± 0.003 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n1024` | skipped | skipped | 0.004 ± 0.000 | skipped | skipped | 0.004 ± 0.000 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 6.294 ± 0.229 | 6.361 ± 0.189 | 6.337 ± 0.253 | 11.985 ± 1.087 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n16384` | skipped | skipped | 0.072 ± 0.001 | skipped | skipped | 0.105 ± 0.001 |
| cpu/fft | `irfft` | c32 | 1 | `1d_n1024` | skipped | skipped | 0.003 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `irfft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 3.624 ± 0.057 | 3.756 ± 0.087 | 4.165 ± 0.056 | 3.511 ± 0.052 |
| cpu/fft | `irfft` | c32 | 1 | `1d_n16384` | skipped | skipped | 0.042 ± 0.000 | skipped | skipped | 0.037 ± 0.001 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n1024` | skipped | skipped | 0.003 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 4.341 ± 0.050 | 4.273 ± 0.080 | 4.241 ± 0.153 | 3.655 ± 0.164 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n16384` | skipped | skipped | 0.042 ± 0.001 | skipped | skipped | 0.037 ± 0.001 |
| cpu/fft | `irfft` | c64 | 1 | `1d_n1024` | skipped | skipped | 0.004 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `irfft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 6.198 ± 0.088 | 6.219 ± 0.090 | 6.809 ± 0.109 | 4.487 ± 0.132 |
| cpu/fft | `irfft` | c64 | 1 | `1d_n16384` | skipped | skipped | 0.072 ± 0.001 | skipped | skipped | 0.050 ± 0.000 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n1024` | skipped | skipped | 0.004 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 6.381 ± 0.159 | 6.387 ± 0.124 | 6.941 ± 0.172 | 4.626 ± 0.197 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n16384` | skipped | skipped | 0.074 ± 0.005 | skipped | skipped | 0.050 ± 0.001 |
| cpu/fft | `rfft` | f32 | 1 | `1d_n1024` | skipped | skipped | 0.003 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `rfft` | f32 | 1 | `1d_n1048576` | skipped | skipped | 3.953 ± 0.064 | 3.922 ± 0.049 | 3.569 ± 0.098 | 3.437 ± 0.066 |
| cpu/fft | `rfft` | f32 | 1 | `1d_n16384` | skipped | skipped | 0.039 ± 0.001 | skipped | skipped | 0.037 ± 0.000 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n1024` | skipped | skipped | 0.002 ± 0.000 | skipped | skipped | 0.002 ± 0.000 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n1048576` | skipped | skipped | 4.083 ± 0.110 | 4.049 ± 0.098 | 4.061 ± 0.118 | 3.451 ± 0.028 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n16384` | skipped | skipped | 0.039 ± 0.000 | skipped | skipped | 0.038 ± 0.001 |
| cpu/fft | `rfft` | f64 | 1 | `1d_n1024` | skipped | skipped | 0.004 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `rfft` | f64 | 1 | `1d_n1048576` | skipped | skipped | 6.300 ± 0.117 | 5.903 ± 0.110 | 6.048 ± 0.093 | 4.438 ± 0.179 |
| cpu/fft | `rfft` | f64 | 1 | `1d_n16384` | skipped | skipped | 0.067 ± 0.001 | skipped | skipped | 0.049 ± 0.003 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n1024` | skipped | skipped | 0.004 ± 0.000 | skipped | skipped | 0.003 ± 0.000 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n1048576` | skipped | skipped | 5.854 ± 0.146 | 6.034 ± 0.081 | 5.951 ± 0.230 | 4.456 ± 0.098 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n16384` | skipped | skipped | 0.067 ± 0.001 | skipped | skipped | 0.049 ± 0.001 |

Notes:

- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.316459
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.323250
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.331041
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.333000
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.333874
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.337083
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.346124
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.349625
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.473917
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.477542
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.494624
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.495791
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.500458
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.501125
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.505666
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=0.516417
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=10.793666
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=10.861000
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=10.945708
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=11.063250
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=13.216875
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=13.420958
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=13.633708
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=13.642708
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=4.720625
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=4.729958
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=4.792917
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=4.866250
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=6.236334
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=6.295042
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=6.376084
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=128; median_batch_ms=6.377541
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.108500
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.260209
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.352791
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.492500
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=11.109500
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=11.661208
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=11.682834
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=11.985083
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=3.437333
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=3.451375
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=3.511082
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=3.654624
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=4.438375
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=4.455958
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=4.486708
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=4.625959
- tenferro-fft-eager: API includes per-call planning or session entry; use shared cached executor for short operation timing
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.756500
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.820000
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.922000
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=4.048541
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=4.086625
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=4.185625
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=4.273333
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=4.273792
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=5.903458
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.034416
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.162292
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.218666
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.361333
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.387375
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.598375
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=7.069417
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.306875
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.318084
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.336291
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.342916
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.349334
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.384541
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.391792
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.454291
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.455875
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.462333
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.465542
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.500708
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.526583
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.533166
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=0.546292
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=4.993292
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.005958
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.097042
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.290542
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.314667
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.315166
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.326917
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=5.426875
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=8.548209
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=8.622708
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=8.756667
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=8.980959
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=9.166583
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=9.227959
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=9.230166
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=128; median_batch_ms=9.419667
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.623958
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.734125
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.952875
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=4.077500
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=4.082916
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=4.169208
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=4.181791
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=4.341083
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=5.853708
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.116000
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.198000
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.293834
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.300041
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.380542
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.543500
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.789000
- tenferro-fft-immediate: API includes per-call planning or session entry; use shared cached executor for short operation timing
- tenferro-fft-read: API includes per-call planning or session entry; use shared cached executor for short operation timing
- tenferro-fft-trace: API includes per-call planning or session entry; use shared cached executor for short operation timing
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=3.568917
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=3.781625
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.060667
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.081833
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.160167
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.164625
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.197583
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.241125
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=5.887375
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=5.951458
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.047583
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.173875
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.337416
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.808542
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.924875
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.940834

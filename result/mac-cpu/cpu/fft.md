# CPU FFT Benchmark Results

- Suite: `cpu/fft`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/fft.yaml`
- Run metadata: `data/results/mac-cpu/cpu/fft/20261113_182559/run.yaml`
- Timestamp: `2026-11-13T18:25:59.842453+00:00`

Latest run: `./scripts/run_cpu_fft.sh 1 4`.

This file is generated from sequential CPU FFT runs under `data/results/mac-cpu/cpu/fft/20261113_182559`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

## CPU Information

- Model: `Apple M1 (Virtual)`
- Vendor: `Apple`
- Logical CPUs: `3`
- Physical CPUs: `3`
- Sockets: `1`
- Cores per socket: `3`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Standard: 3 physical / 3 logical (L1i 192 KiB, L1d 128 KiB, L2 12 MiB, 3 CPUs/L2)`
- Python platform: `macOS-26.7.2-arm64-arm-64bit-Mach-O`
- Container: `none`

## Thread Environments

### Threads: 1

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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=false --xla_cpu_experimental_ynn_fusion_type= intra_op_parallelism_threads=1`

### Threads: 4

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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true --xla_cpu_experimental_ynn_fusion_type= intra_op_parallelism_threads=4`

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

- CSV: `data/results/mac-cpu/cpu/fft/20261113_182559/cpu_fft_t1_20261113_182559.csv`
- CSV: `data/results/mac-cpu/cpu/fft/20261113_182559/cpu_fft_t4_20261113_182559.csv`

## CPU FFT Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

Timing audit: PyTorch 2.12 CPU torch.fft creates and commits a DFTI descriptor per call even after warmup ([upstream source](https://github.com/pytorch/pytorch/blob/7661cd9c6b841b62b7f411aa52ec51f05457263b/aten/src/ATen/native/mkl/SpectralOps.cpp#L490)). Its rows include setup and are noncompliant as operation-only performance evidence; warmed execution does not remove that setup. Input tensors are untimed and allocation-returning calls retain output allocation. Traced calls also include internal session work. No historical timing values have been edited.

### Operation execution

| suite | benchmark | dtype | threads | shape | tenferro-rs one-shot diagnostic (ms) | tenferro-rs TensorRead API (ms) | tenferro-rs cached primary (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch torch.fft (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| cpu/fft | `fft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 9.039 ± 1.362 | 8.705 ± 0.248 | 9.353 ± 1.164 | 22.042 ± 1.222 |
| cpu/fft | `fft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 9.042 ± 3.513 | 7.855 ± 1.140 | 8.932 ± 4.132 | 21.450 ± 2.643 |
| cpu/fft | `fft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 17.920 ± 0.548 | 17.107 ± 1.394 | 17.945 ± 1.087 | 31.273 ± 12.224 |
| cpu/fft | `fft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 15.837 ± 1.388 | 15.188 ± 0.548 | 15.805 ± 1.657 | 22.291 ± 0.638 |
| cpu/fft | `ifft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 9.283 ± 1.845 | 8.144 ± 0.434 | 8.478 ± 0.948 | 25.495 ± 15.934 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 8.827 ± 1.351 | 9.091 ± 0.972 | 7.782 ± 1.992 | 20.343 ± 1.070 |
| cpu/fft | `ifft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 18.455 ± 4.211 | 15.515 ± 4.478 | 16.884 ± 2.335 | 22.467 ± 5.088 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 15.447 ± 2.832 | 15.651 ± 1.013 | 23.223 ± 4.753 | 19.327 ± 0.362 |
| cpu/fft | `irfft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 3.985 ± 0.314 | 4.367 ± 0.805 | 3.908 ± 0.098 | 6.917 ± 0.265 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 5.363 ± 2.521 | 6.340 ± 3.233 | 4.417 ± 0.843 | 6.651 ± 0.364 |
| cpu/fft | `irfft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 6.960 ± 1.324 | 6.899 ± 0.221 | 6.986 ± 0.432 | 10.166 ± 6.127 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 7.863 ± 1.885 | 8.246 ± 1.990 | 7.472 ± 1.415 | 10.266 ± 1.116 |
| cpu/fft | `rfft` | f32 | 1 | `1d_n1048576` | skipped | skipped | 3.769 ± 0.142 | 3.769 ± 0.206 | 3.860 ± 0.160 | 7.078 ± 0.543 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n1048576` | skipped | skipped | 4.266 ± 0.581 | 4.260 ± 0.834 | 4.106 ± 0.966 | 6.362 ± 0.374 |
| cpu/fft | `rfft` | f64 | 1 | `1d_n1048576` | skipped | skipped | 6.727 ± 0.346 | 6.752 ± 0.278 | 6.800 ± 0.139 | 11.123 ± 1.098 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n1048576` | skipped | skipped | 7.462 ± 5.322 | 14.078 ± 7.934 | 13.263 ± 8.560 | 11.430 ± 1.256 |

Notes:

- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.166334
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.265708
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=11.123333
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=11.430209
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=19.327041
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=20.343291
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=21.449625
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=22.042250
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=22.290666
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=22.466792
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=25.495333
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=31.273458
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=6.361625
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=6.651167
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=6.917125
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=7.077834
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=14.078292
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=15.188166
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=15.514917
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=15.651000
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=17.107042
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.769167
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=4.259625
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=4.366709
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.340292
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.752250
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.898667
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=7.855208
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=8.143834
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=8.245958
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=8.705084
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=9.090750
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=15.447375
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=15.837375
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=17.920125
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=18.455042
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.769167
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.984875
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=4.265500
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=5.362625
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.727041
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.959584
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=7.462167
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=7.863000
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=8.827292
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=9.039250
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=9.041875
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=9.282542
- tenferro-fft-immediate: API includes per-call planning or session entry; use shared cached executor for short operation timing
- tenferro-fft-read: API includes per-call planning or session entry; use shared cached executor for short operation timing
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=13.263083
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=15.805334
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=16.884166
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=17.944834
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=23.222542
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=3.859917
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=3.907542
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.106083
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.417166
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.800000
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.985625
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=7.471833
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=7.782250
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=8.478084
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=8.932375
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=9.353084

# CPU FFT Benchmark Results

- Suite: `cpu/fft`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/fft.yaml`
- Run metadata: `data/results/mac-cpu/cpu/fft/20261108_182132/run.yaml`
- Timestamp: `2026-11-08T18:21:33.026749+00:00`

Latest run: `./scripts/run_cpu_fft.sh 1 4`.

This file is generated from sequential CPU FFT runs under `data/results/mac-cpu/cpu/fft/20261108_182132`.

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

- CSV: `data/results/mac-cpu/cpu/fft/20261108_182132/cpu_fft_t1_20261108_182132.csv`
- CSV: `data/results/mac-cpu/cpu/fft/20261108_182132/cpu_fft_t4_20261108_182132.csv`

## CPU FFT Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

Timing audit: PyTorch 2.12 CPU torch.fft creates and commits a DFTI descriptor per call even after warmup ([upstream source](https://github.com/pytorch/pytorch/blob/7661cd9c6b841b62b7f411aa52ec51f05457263b/aten/src/ATen/native/mkl/SpectralOps.cpp#L490)). Its rows include setup and are noncompliant as operation-only performance evidence; warmed execution does not remove that setup. Input tensors are untimed and allocation-returning calls retain output allocation. Traced calls also include internal session work. No historical timing values have been edited.

### Operation execution

| suite | benchmark | dtype | threads | shape | tenferro-rs one-shot diagnostic (ms) | tenferro-rs TensorRead API (ms) | tenferro-rs cached primary (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch torch.fft (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| cpu/fft | `fft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 7.923 ± 1.362 | 8.501 ± 1.316 | 7.274 ± 0.193 | 16.361 ± 0.344 |
| cpu/fft | `fft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 8.274 ± 1.056 | 8.271 ± 1.204 | 6.944 ± 1.209 | 18.087 ± 0.990 |
| cpu/fft | `fft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 17.275 ± 2.801 | 16.756 ± 5.391 | 12.702 ± 2.119 | 17.182 ± 0.488 |
| cpu/fft | `fft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 14.915 ± 1.859 | 14.586 ± 0.768 | 15.154 ± 0.668 | 16.851 ± 5.460 |
| cpu/fft | `ifft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 6.118 ± 0.352 | 7.496 ± 1.230 | 8.119 ± 1.021 | 16.653 ± 0.204 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 7.060 ± 0.888 | 7.430 ± 0.450 | 7.511 ± 1.280 | 15.898 ± 2.188 |
| cpu/fft | `ifft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 11.769 ± 0.273 | 11.877 ± 0.622 | 12.925 ± 0.960 | 17.384 ± 0.474 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 14.158 ± 0.628 | 14.097 ± 0.687 | 12.506 ± 0.697 | 17.084 ± 0.515 |
| cpu/fft | `irfft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 3.376 ± 0.086 | 3.313 ± 0.134 | 3.296 ± 0.057 | 5.680 ± 0.233 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 3.166 ± 0.014 | 3.172 ± 0.037 | 3.203 ± 0.031 | 5.975 ± 1.062 |
| cpu/fft | `irfft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 5.931 ± 0.192 | 6.113 ± 0.167 | 5.980 ± 0.217 | 7.909 ± 0.052 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 5.713 ± 0.038 | 5.738 ± 0.056 | 5.724 ± 0.026 | 8.002 ± 0.424 |
| cpu/fft | `rfft` | f32 | 1 | `1d_n1048576` | skipped | skipped | 3.164 ± 0.203 | 3.351 ± 0.424 | 3.289 ± 0.442 | 5.746 ± 0.099 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n1048576` | skipped | skipped | 3.073 ± 0.026 | 3.132 ± 0.036 | 3.126 ± 0.050 | 5.842 ± 0.077 |
| cpu/fft | `rfft` | f64 | 1 | `1d_n1048576` | skipped | skipped | 5.803 ± 0.164 | 5.895 ± 0.304 | 5.936 ± 0.368 | 8.633 ± 0.037 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n1048576` | skipped | skipped | 5.529 ± 0.027 | 5.563 ± 0.033 | 5.634 ± 0.241 | 9.492 ± 0.382 |

Notes:

- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=15.898250
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=16.361500
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=16.653083
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=16.850875
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=17.084417
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=17.181625
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=17.384417
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=18.087209
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=5.679625
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=5.746417
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=5.842375
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=5.975250
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=7.908584
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=8.001958
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=8.633500
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=9.492167
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=11.876917
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=14.096708
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=14.585584
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=16.756083
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.132250
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.171750
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.312542
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.350875
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=5.562750
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=5.737917
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=5.895000
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.113417
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=7.430208
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=7.496291
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=8.271084
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=8.500959
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=11.769375
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=14.158000
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=14.914500
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=17.274917
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.073416
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.164167
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.165917
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.376167
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=5.529084
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=5.712959
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=5.803334
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=5.931084
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.118292
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=7.060375
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=7.922625
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=8.273542
- tenferro-fft-immediate: API includes per-call planning or session entry; use shared cached executor for short operation timing
- tenferro-fft-read: API includes per-call planning or session entry; use shared cached executor for short operation timing
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=12.505542
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=12.702167
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=12.924583
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=15.153542
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=3.126083
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=3.202542
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=3.288666
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=3.295833
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=5.634416
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=5.724375
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=5.936333
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=5.979708
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.943750
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=7.274250
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=7.511292
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=8.118750

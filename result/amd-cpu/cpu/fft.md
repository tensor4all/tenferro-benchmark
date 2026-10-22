# CPU FFT Benchmark Results

- Suite: `cpu/fft`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/fft.yaml`
- Run metadata: `data/results/amd-cpu/cpu/fft/20261021_185713/run.yaml`
- Timestamp: `20261021_185713`

Latest run: `./scripts/run_cpu_fft.sh 1 4`.

This file is generated from sequential CPU FFT runs under `data/results/amd-cpu/cpu/fft/20261021_185713`.

- tenferro-rs commit: `f1d4d8d961ff6b937768111888ad2a493314892a`

## CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`

## Thread Environments

### Threads: 1

- Run metadata: `data/results/amd-cpu/cpu/fft/20261021_185713/run_t1.yaml`
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

- Run metadata: `data/results/amd-cpu/cpu/fft/20261021_185713/run_t4.yaml`
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

- CSV: `data/results/amd-cpu/cpu/fft/20261021_185713/cpu_fft_t1_20261021_185713.csv`
- CSV: `data/results/amd-cpu/cpu/fft/20261021_185713/cpu_fft_t4_20261021_185713.csv`
- Source table: `data/results/amd-cpu/cpu/fft/20261021_185713/cpu_fft_20261021_185713.md`

## CPU FFT Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

Timing audit: PyTorch 2.12 CPU torch.fft creates and commits a DFTI descriptor per call even after warmup ([upstream source](https://github.com/pytorch/pytorch/blob/7661cd9c6b841b62b7f411aa52ec51f05457263b/aten/src/ATen/native/mkl/SpectralOps.cpp#L490)). Its rows include setup and are noncompliant as operation-only performance evidence; warmed execution does not remove that setup. Input tensors are untimed and allocation-returning calls retain output allocation. Traced calls also include internal session work. Use the [cached native oneMKL follow-up](https://github.com/tensor4all/tenferro-benchmark/blob/main/result/amd-cpu/cpu/followup_12x.md) for audited current-main operation evidence. No historical timing values have been edited.

### Operation execution

| suite | benchmark | dtype | threads | shape | tenferro-rs one-shot diagnostic (ms) | tenferro-rs TensorRead API (ms) | tenferro-rs cached primary (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch torch.fft (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| cpu/fft | `fft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 6.824 ± 0.183 | 6.881 ± 0.129 | 6.758 ± 1.707 | 10.036 ± 0.337 |
| cpu/fft | `fft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 6.989 ± 0.414 | 6.730 ± 0.140 | 6.311 ± 0.057 | 5.383 ± 0.141 |
| cpu/fft | `fft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 12.347 ± 0.152 | 12.124 ± 0.189 | 11.202 ± 0.186 | 16.544 ± 2.285 |
| cpu/fft | `fft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 12.126 ± 0.125 | 11.591 ± 0.082 | 10.818 ± 0.222 | 16.571 ± 2.574 |
| cpu/fft | `ifft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 6.865 ± 0.089 | 6.783 ± 0.127 | 6.524 ± 0.177 | 9.993 ± 0.257 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 6.711 ± 0.079 | 6.639 ± 0.065 | 6.398 ± 0.190 | 3.310 ± 0.183 |
| cpu/fft | `ifft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 12.525 ± 0.241 | 11.762 ± 0.179 | 11.039 ± 0.483 | 17.037 ± 0.685 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 11.514 ± 0.125 | 11.594 ± 0.197 | 10.983 ± 0.314 | 16.378 ± 0.841 |
| cpu/fft | `irfft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 3.556 ± 0.053 | 3.494 ± 0.048 | 3.654 ± 0.094 | 15.496 ± 0.100 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 3.493 ± 0.134 | 3.497 ± 0.073 | 3.724 ± 0.063 | 4.264 ± 0.018 |
| cpu/fft | `irfft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 6.300 ± 0.138 | 6.274 ± 0.238 | 6.736 ± 0.387 | 17.758 ± 0.218 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 6.264 ± 0.219 | 6.195 ± 0.219 | 6.498 ± 0.188 | 16.480 ± 0.169 |
| cpu/fft | `rfft` | f32 | 1 | `1d_n1048576` | skipped | skipped | 3.948 ± 0.026 | 3.878 ± 0.074 | 3.767 ± 0.079 | 14.993 ± 0.114 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n1048576` | skipped | skipped | 3.839 ± 0.086 | 3.766 ± 0.065 | 3.781 ± 0.045 | 4.136 ± 0.057 |
| cpu/fft | `rfft` | f64 | 1 | `1d_n1048576` | skipped | skipped | 5.879 ± 0.249 | 6.055 ± 0.165 | 5.799 ± 0.070 | 18.495 ± 0.212 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n1048576` | skipped | skipped | 5.876 ± 0.302 | 5.859 ± 0.195 | 5.860 ± 0.914 | 16.598 ± 0.604 |

Notes:

- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.035518
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=14.992874
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=15.495540
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=16.378451
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=16.479860
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=16.543820
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=16.570912
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=16.598020
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=17.037218
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=17.758475
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=18.494571
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=3.309625
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=4.136085
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=4.263552
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=5.382598
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=9.993479
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=11.591406
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=11.593670
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=11.761561
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=12.123857
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.494103
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.497265
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.766076
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.877988
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=5.858945
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.055440
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.194711
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.274067
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.638578
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.729868
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.783378
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=6.880920
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=11.513520
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=12.126252
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=12.347153
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=12.524794
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.492757
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.556168
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.838562
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.948428
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=5.875537
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=5.878751
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.264351
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.299825
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.710672
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.824214
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.864820
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=6.989031
- tenferro-fft-immediate: API includes per-call planning or session entry; use shared cached executor for short operation timing
- tenferro-fft-read: API includes per-call planning or session entry; use shared cached executor for short operation timing
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=10.818275
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=10.983012
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=11.039496
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=11.201759
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=3.653700
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=3.724478
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=3.767141
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=3.780854
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=5.799093
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=5.860117
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.310728
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.398291
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.498266
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.524366
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.736107
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.757719

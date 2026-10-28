# CPU FFT Benchmark Results

- Suite: `cpu/fft`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/fft.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/fft/20261028_004719/run.yaml`
- Timestamp: `20261028_004719`

Latest run: `./scripts/run_cpu_fft.sh 1 4 8`.

This file is generated from sequential CPU FFT runs under `data/results/zen5-cpu/cpu/fft/20261028_004719`.

- tenferro-rs commit: `8fec1ca70852b930c092e7177c30bb8af7594a57`

## CPU Information

- Model: `AMD Ryzen AI 9 HX 470 w/ Radeon 890M`
- Vendor: `AuthenticAMD`
- Logical CPUs: `24`
- Sockets: `1`
- Cores per socket: `12`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3,12-15; 4-11,16-23`
- CPU boost: `enabled`

## Thread Environments

### Threads: 1

- Run metadata: `data/results/zen5-cpu/cpu/fft/20261028_004719/run_t1.yaml`
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

- Run metadata: `data/results/zen5-cpu/cpu/fft/20261028_004719/run_t4.yaml`
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

### Threads: 8

- Run metadata: `data/results/zen5-cpu/cpu/fft/20261028_004719/run_t8.yaml`
- OMP_NUM_THREADS: `8`
- OMP_THREAD_LIMIT: `8`
- OMP_DYNAMIC: `FALSE`
- RAYON_NUM_THREADS: `8`
- OPENBLAS_NUM_THREADS: `8`
- GOTO_NUM_THREADS: `8`
- MKL_NUM_THREADS: `8`
- VECLIB_MAXIMUM_THREADS: `8`
- VECLIB_NUM_THREADS: `8`
- NUMEXPR_NUM_THREADS: `8`
- BLIS_NUM_THREADS: `8`
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true --xla_cpu_experimental_ynn_fusion_type= intra_op_parallelism_threads=8`

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

## Threads: 1 4 8

- CSV: `data/results/zen5-cpu/cpu/fft/20261028_004719/cpu_fft_t1_20261028_004719.csv`
- CSV: `data/results/zen5-cpu/cpu/fft/20261028_004719/cpu_fft_t4_20261028_004719.csv`
- CSV: `data/results/zen5-cpu/cpu/fft/20261028_004719/cpu_fft_t8_20261028_004719.csv`
- Source table: `data/results/zen5-cpu/cpu/fft/20261028_004719/cpu_fft_20261028_004719.md`

## CPU FFT Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

Timing audit: PyTorch 2.12 CPU torch.fft creates and commits a DFTI descriptor per call even after warmup ([upstream source](https://github.com/pytorch/pytorch/blob/7661cd9c6b841b62b7f411aa52ec51f05457263b/aten/src/ATen/native/mkl/SpectralOps.cpp#L490)). Its rows include setup and are noncompliant as operation-only performance evidence; warmed execution does not remove that setup. Input tensors are untimed and allocation-returning calls retain output allocation. Traced calls also include internal session work. No historical timing values have been edited.

### Operation execution

| suite | benchmark | dtype | threads | shape | tenferro-rs one-shot diagnostic (ms) | tenferro-rs TensorRead API (ms) | tenferro-rs cached primary (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch torch.fft (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| cpu/fft | `fft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 3.703 ± 0.038 | 4.014 ± 0.019 | 3.512 ± 0.042 | 5.800 ± 0.099 |
| cpu/fft | `fft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 3.798 ± 0.053 | 3.984 ± 0.044 | 3.559 ± 0.023 | 3.138 ± 0.065 |
| cpu/fft | `fft` | c32 | 8 | `1d_n1048576` | skipped | skipped | 5.732 ± 0.032 | 5.457 ± 0.036 | 5.428 ± 0.052 | 5.470 ± 0.083 |
| cpu/fft | `fft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 8.432 ± 0.179 | 8.558 ± 0.143 | 8.374 ± 0.037 | 10.178 ± 0.247 |
| cpu/fft | `fft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 8.433 ± 0.051 | 8.582 ± 0.122 | 8.451 ± 0.120 | 10.273 ± 0.037 |
| cpu/fft | `fft` | c64 | 8 | `1d_n1048576` | skipped | skipped | 10.724 ± 0.075 | 10.792 ± 0.101 | 10.586 ± 0.079 | 14.148 ± 0.184 |
| cpu/fft | `ifft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 3.691 ± 0.066 | 4.031 ± 0.047 | 3.511 ± 0.027 | 5.964 ± 0.053 |
| cpu/fft | `ifft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 3.774 ± 0.022 | 3.987 ± 0.042 | 3.572 ± 0.031 | 1.808 ± 0.149 |
| cpu/fft | `ifft` | c32 | 8 | `1d_n1048576` | skipped | skipped | 5.753 ± 0.037 | 5.457 ± 0.023 | 5.400 ± 0.070 | 2.272 ± 0.034 |
| cpu/fft | `ifft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 8.398 ± 0.058 | 8.533 ± 0.143 | 8.362 ± 0.052 | 10.627 ± 0.222 |
| cpu/fft | `ifft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 8.625 ± 0.094 | 8.528 ± 0.025 | 8.552 ± 0.037 | 10.642 ± 0.124 |
| cpu/fft | `ifft` | c64 | 8 | `1d_n1048576` | skipped | skipped | 10.645 ± 0.116 | 10.652 ± 0.111 | 10.521 ± 0.120 | 14.564 ± 0.079 |
| cpu/fft | `irfft` | c32 | 1 | `1d_n1048576` | skipped | skipped | 2.147 ± 0.067 | 1.966 ± 0.011 | 1.966 ± 0.057 | 8.677 ± 0.024 |
| cpu/fft | `irfft` | c32 | 4 | `1d_n1048576` | skipped | skipped | 2.225 ± 0.099 | 2.019 ± 0.049 | 2.039 ± 0.033 | 2.110 ± 0.018 |
| cpu/fft | `irfft` | c32 | 8 | `1d_n1048576` | skipped | skipped | 3.228 ± 0.147 | 2.976 ± 0.043 | 2.931 ± 0.027 | 1.658 ± 0.023 |
| cpu/fft | `irfft` | c64 | 1 | `1d_n1048576` | skipped | skipped | 3.764 ± 0.029 | 4.099 ± 0.028 | 4.116 ± 0.039 | 10.039 ± 0.091 |
| cpu/fft | `irfft` | c64 | 4 | `1d_n1048576` | skipped | skipped | 3.814 ± 0.050 | 4.217 ± 0.063 | 4.216 ± 0.033 | 8.009 ± 0.043 |
| cpu/fft | `irfft` | c64 | 8 | `1d_n1048576` | skipped | skipped | 5.326 ± 0.059 | 5.968 ± 0.094 | 6.136 ± 0.051 | 10.960 ± 0.038 |
| cpu/fft | `rfft` | f32 | 1 | `1d_n1048576` | skipped | skipped | 1.897 ± 0.055 | 1.907 ± 0.030 | 1.895 ± 0.024 | 8.706 ± 0.049 |
| cpu/fft | `rfft` | f32 | 4 | `1d_n1048576` | skipped | skipped | 1.965 ± 0.035 | 1.960 ± 0.033 | 1.951 ± 0.024 | 1.856 ± 0.012 |
| cpu/fft | `rfft` | f32 | 8 | `1d_n1048576` | skipped | skipped | 2.817 ± 0.097 | 2.853 ± 0.032 | 2.834 ± 0.010 | 1.847 ± 0.038 |
| cpu/fft | `rfft` | f64 | 1 | `1d_n1048576` | skipped | skipped | 3.844 ± 0.069 | 3.832 ± 0.030 | 3.864 ± 0.039 | 10.630 ± 0.065 |
| cpu/fft | `rfft` | f64 | 4 | `1d_n1048576` | skipped | skipped | 3.855 ± 0.017 | 3.872 ± 0.029 | 3.905 ± 0.077 | 9.769 ± 0.056 |
| cpu/fft | `rfft` | f64 | 8 | `1d_n1048576` | skipped | skipped | 5.401 ± 0.059 | 5.419 ± 0.044 | 5.385 ± 0.063 | 14.424 ± 0.092 |

Notes:

- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=1.657582
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=1.807652
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=1.846874
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=1.856131
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.038884
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.178183
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.272911
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.626760
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.630006
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.641528
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=10.960112
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=14.148298
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=14.424142
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=14.564234
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=2.110126
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=2.271506
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=3.138474
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=5.470463
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=5.800047
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=5.963813
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=8.008617
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=8.676984
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=8.705596
- pytorch-cpu: torch.fft warmup before measured runs; input allocation outside timed region; operations_per_sample=1; median_batch_ms=9.769011
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=1.906757
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=1.960326
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=1.965816
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=10.652347
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=10.792490
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=2.019035
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=2.852570
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=2.976251
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.832308
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.872032
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.984411
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=3.986846
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=4.013515
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=4.030758
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=4.099376
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=4.216574
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=5.419107
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=5.456637
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=5.457368
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=5.968351
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=8.527615
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=8.532895
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=8.558071
- tenferro-fft-eager: EagerTensorFftExt with one reused eager runtime and input; operations_per_sample=1; median_batch_ms=8.581846
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=1.896918
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=1.964684
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=10.644914
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=10.723871
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=2.147405
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=2.224700
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=2.817154
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.227901
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.690804
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.702646
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.764261
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.773789
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.798094
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.813993
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.844189
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=3.854750
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=5.325643
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=5.400693
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=5.732351
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=5.752920
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=8.397723
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=8.431846
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=8.433259
- tenferro-fft-executor-cached: FftExecutor reused across warmups and measured runs; operations_per_sample=1; median_batch_ms=8.624695
- tenferro-fft-immediate: API includes per-call planning or session entry; use shared cached executor for short operation timing
- tenferro-fft-read: API includes per-call planning or session entry; use shared cached executor for short operation timing
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=1.895154
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=1.951179
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=1.966067
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=10.521223
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=10.586224
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=2.039023
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=2.833625
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=2.930806
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=3.511480
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=3.512090
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=3.558988
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=3.572002
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=3.864197
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=3.905384
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.115516
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=4.215944
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=5.385254
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=5.399981
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=5.427653
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=6.135653
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=8.361926
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=8.374028
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=8.451443
- tenferro-fft-trace: TracedTensorFftExt graph compiled once; compiled program reused across runs; operations_per_sample=1; median_batch_ms=8.551970

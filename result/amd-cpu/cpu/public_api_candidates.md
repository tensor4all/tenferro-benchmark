# CPU Public API Benchmark Results

- Suite: `cpu/public_api_candidates`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/public_api_candidates.yaml`
- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`
- Run metadata: `data/results/amd-cpu/cpu/public_api_candidates/20261013_075621/run.yaml`
- Timestamp: `20261013_075621`
- Selection: initial six API candidates only; this report does not replace the complete Public API report. Julia columns are unselected.
- Invocation prefix: `PUBLIC_API_CANDIDATES_ONLY=1 TENFERRO_CPU_FEATURES=blas-mkl BENCHMARK_TARGET_PROFILE=amd-cpu`

Latest run: `PUBLICATION_GATE_PROFILE=full ./scripts/run_cpu_public_api.sh 1 4`.

- Sampling: `full` profile, 15 measured runs, 3 warmups per row

This file is generated from sequential CPU public API runs under `data/results/amd-cpu/cpu/public_api_candidates/20261013_075621`.

- tenferro-rs commit: `8b13220c1e0397a29a27be65557c24105352a35b`

## CPU Information

- Model: `Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `16`
- Sockets: `1`
- Cores per socket: `16`
- Threads per core: `1`
- NUMA nodes: `unknown`
- Python platform: `Linux-7.0.14-linuxkit-x86_64-with-glibc2.39`

## Thread Environments

### Threads: 1

- Run metadata: `data/results/amd-cpu/cpu/public_api_candidates/20261013_075621/run_t1.yaml`
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
- JULIA_NUM_THREADS: `1`

### Threads: 4

- Run metadata: `data/results/amd-cpu/cpu/public_api_candidates/20261013_075621/run_t4.yaml`
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
- JULIA_NUM_THREADS: `4`

## Timing Discipline

- Fixtures, Tensor/EagerTensor construction, session entry, input copies, destinations, library initialization and JAX compilation are completed explicitly before sampling, including zero-warmup runs.
- Direct and eager paths reuse one entered session. Outputs remain alive until the clock stops; numerical validation and destruction occur outside timing.
- Compact Householder QR measures compact-state initialization. Q/R extraction, reconstruction and orthogonality checks are untimed.
- RRQR requests Q, R, column permutation and numerical rank, with Raw gauge, rtol=1e-12 and atol=0. Full-rank and deficient fixtures are separate cases. PyTorch has no pivoted QR; JAX's three-output pivoted QR is not a matching four-output operation.
- Caller-output solve preallocates its destination and includes factorization plus solve. PyTorch uses torch.linalg.solve(out=...). JAX and eager/trace caller-output solve are unsupported.
- Masked log-softmax compares the public tenferro operation with jax.nn.log_softmax(where=mask). PyTorch's mask-plus-log-softmax composition is excluded. Axis-wise sum-of-squares compositions are also excluded.
- Selection compares torch.where/jnp.where with identical logical values in each framework's native layout. The existing select row covers the same backend kernel; this row explicitly measures the public session spelling.
- Candidate operations use batches of sixteen retained outputs, normalized per operation. Selection is memory-limited to one operation because each output is 256 MiB. Batch counts and durations are preserved in the CSV notes and table below.
- Compiled trace execution has no public borrowed BackendSession API. Its per-call session initialization is excluded from these standard operation rows; compact-state trace also cannot return the opaque state without extra operations.
- Provider and linked-library observations are recorded in run_t1.yaml and run_t4.yaml. The CPU model above is the actual Docker host; the amd-cpu path is the repository's standard Linux CPU collection profile.

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.353 ± 0.008 | 0.362 ± 0.005 | unsupported | unsupported | 0.672 ± 0.031 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 2.280 ± 0.110 | 1.800 ± 0.045 | unsupported | unsupported | 0.560 ± 0.025 | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 1.421 ± 0.076 | 1.431 ± 0.036 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 1.031 ± 0.070 | 1.065 ± 0.037 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 78.770 ± 0.807 | 94.165 ± 0.958 | unsupported | 42.644 ± 0.381 | 45.874 ± 1.940 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 31.109 ± 0.151 | 34.783 ± 0.851 | unsupported | 29.282 ± 0.489 | 29.477 ± 0.308 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.320 ± 0.003 | 0.363 ± 0.003 | unsupported | 0.314 ± 0.002 | 0.582 ± 0.007 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 5.066 ± 0.180 | 5.227 ± 0.038 | unsupported | 5.138 ± 0.041 | 7.464 ± 0.164 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.212 ± 0.010 | 0.469 ± 0.033 | unsupported | 0.267 ± 0.013 | 1.284 ± 0.033 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 3.021 ± 0.082 | 3.527 ± 0.063 | unsupported | 2.659 ± 0.058 | 6.544 ± 0.105 | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 0.760 ± 0.004 | 0.792 ± 0.012 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 15.287 ± 0.819 | 15.650 ± 0.416 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 0.656 ± 0.017 | 0.952 ± 0.029 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 7.764 ± 0.078 | 8.345 ± 0.128 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 0.720 ± 0.004 | 0.758 ± 0.042 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 14.552 ± 0.836 | 14.649 ± 0.463 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 0.612 ± 0.013 | 0.900 ± 0.026 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 7.100 ± 0.115 | 7.736 ± 0.174 | unsupported | unsupported | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.395 ± 0.005 | unsupported | unsupported | 0.530 ± 0.003 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 6.612 ± 0.189 | unsupported | unsupported | 7.320 ± 0.251 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.228 ± 0.015 | unsupported | unsupported | 0.311 ± 0.013 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 2.416 ± 0.072 | unsupported | unsupported | 2.663 ± 0.060 | unsupported | - | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 10.751401 | 671962.56 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 5.653554 | 353347.12 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 5.787744 | 361734.00 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 8.963362 | 560210.12 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 36.477787 | 2279861.69 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 28.807961 | 1800497.56 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 22.733400 | 1420837.50 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 22.891016 | 1430688.50 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 16.493388 | 1030836.75 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 17.045606 | 1065350.38 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 9.317670 | 582354.38 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 5.017403 | 313587.69 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 5.114237 | 319639.81 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 5.810942 | 363183.88 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 119.427380 | 7464211.25 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 82.204618 | 5137788.62 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 81.048773 | 5065548.31 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 83.627188 | 5226699.25 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 20.547819 | 1284238.69 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 4.278058 | 267378.62 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 3.392045 | 212002.81 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 7.510157 | 469384.81 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 104.706865 | 6544179.06 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 42.536781 | 2658548.81 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 48.332392 | 3020774.50 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 56.430770 | 3526923.12 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 12.165131 | 760320.69 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 12.676840 | 792302.50 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 244.598459 | 15287403.69 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 250.399895 | 15649993.44 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 10.495699 | 655981.19 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 15.228770 | 951798.12 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 124.217363 | 7763585.19 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 133.526141 | 8345383.81 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 11.517341 | 719833.81 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 12.135841 | 758490.06 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 232.830490 | 14551905.62 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 234.384696 | 14649043.50 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 9.787126 | 611695.38 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 14.392559 | 899534.94 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 113.592664 | 7099541.50 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 123.775458 | 7735966.12 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 8.485611 | 530350.69 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 6.317733 | 394858.31 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 117.119950 | 7319996.88 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 105.796388 | 6612274.25 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 4.976768 | 311048.00 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 3.647634 | 227977.12 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 42.603500 | 2662718.75 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 38.660443 | 2416277.69 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

No cross-backend spread above 10x was detected.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/amd-cpu/cpu/public_api/20261126_185520/run.yaml`
- Timestamp: `2026-11-26T18:55:20.865088+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/amd-cpu/cpu/public_api/20261126_185520`.

- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

## CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1020-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`
- Container: `devcontainer`

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
- JULIA_NUM_THREADS: `1`

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
- JULIA_NUM_THREADS: `4`

## Timing Discipline

- Input fixture tensors are created during warmup and outside the measured region for tenferro-rs, PyTorch, and JAX.
- The tenferro-rs direct column measures immediate public operations (normally concrete `Tensor + CpuBackend`; `lstsq`/`svd_full` use `EagerTensor` because no concrete spelling exists). It is not labeled as the `EagerTensor` call layer.
- tenferro-rs trace graphs are constructed and compiled outside the measured region; each compiled graph is reused for every warmup and timed run.
- JAX functions are compiled with `jax.jit` during warmup, outside the measured region; timed calls include dispatch through `jax.block_until_ready`.
- Allocation-returning API rows create their output tensor inside each timed call.
- `cpu/output_reuse` rows allocate the destination during warmup and reuse it; tenferro-rs `*_into` is compared with PyTorch `out=`/`copy_`.
- Trace mode is `unsupported` for caller-output rows because compiled tenferro-rs graphs own their output tensors; JAX is shown as missing because it has no equivalent mutable `out=` API.
- PyTorch view-producing indexing operations are cloned inside the timed region to match tenferro-rs owned, materialized outputs.
- `reshape` compares materialized outputs: PyTorch clones its reshape view inside timing to match tenferro-rs' output-sized write; JAX has value semantics and no public strided-view contract.
- `cpu/view_metadata` separately compares concrete tenferro-rs, PyTorch, and Julia (`julia-base`) view creation without an output-sized copy; trace mode is unsupported and JAX is missing because neither exposes the same concrete strided-view contract.
- Julia `cpu/view_metadata` rows are `reshape`/`transpose`/`@view` lazy wrappers with no output-sized copy; `broadcast_in_dim_view` has no natural Base spelling and stays missing for Julia.
- Julia `cpu/output_reuse` rows use broadcast-into (`.=`), `mul!`, and `copyto!` with destinations allocated during warmup and reused, the same reuse discipline as the tenferro-rs `_into`/PyTorch `out=` rows above.
- Rust, PyTorch, and JAX fixtures contain identical logical values. Python/JAX reconstruct the Rust column-major fixture in each framework's native layout before timing.
- Compact Householder QR and RRQR use explicit fixture/session setup; QR extraction and reconstruction/orthogonality checks are untimed. RRQR retains Q, R, permutation and numerical rank; JAX's three-output pivoted QR and PyTorch's unpivoted QR are not equivalent and are unsupported. Compact trace state and caller-output trace solve are unsupported.
- The gap cases use explicit setup and numerical validation outside sampling. Axis-wise sum-of-squares and PyTorch/Julia masked log-softmax compositions are excluded from direct operation comparisons.
- PyTorch complex conjugation uses `torch.conj_physical` to match tenferro-rs physical output rather than the lazy conjugate view from `torch.conj`.
- `dot_general_with_conj` instead uses PyTorch's lazy conjugate view so conjugation can be handled by the contraction, matching tenferro-rs' conjugation flags; trace is unsupported because there is no equivalent public traced API. Julia's `dot_general_with_conj` row materializes the conjugate before the GEMM instead, since Julia has no lazy conj-without-transpose spelling that BLAS can fuse.
- `pad` has no natural Base spelling and stays missing for Julia.
- `dynamic_update_slice` reports trace mode as `unsupported` because tenferro-rs does not currently expose a corresponding `TracedTensor` API.
- `full_piv_lu` and `full_piv_lu_solve` are excluded because PyTorch has no direct public full-pivot equivalent; substituting `torch.linalg.solve` would compare different algorithms.
- `cpu/linalg_batched` rows use a batch of 1024 matrices with one rhs column. `batched_lu_factor` times the packed LU factorization alone. `batched_lu_solve` reuses LU factors prepared during warmup, outside the measured region, and times only the solve (tenferro-rs `LinalgBackend::lu_solve_prepared`, PyTorch `torch.linalg.lu_solve`, JAX `jax.scipy.linalg.lu_solve`, Julia `LAPACK.getrs!`). `batched_triangular_solve` times a lower-triangular solve.
- Trace mode is `unsupported` for `batched_lu_factor` and `batched_lu_solve` because `LinalgOp` is not public, so a trace cannot hold a bare LU factor or prepared solve. The traced solve (LU factor plus prepared solve) and its backward are measured by the `cpu/cpu_ops` `batched_solve` and `grad_sum_batched_solve_backward` rows.
- Julia has no batched LAPACK entry point, so the Julia `cpu/linalg_batched` rows loop the per-matrix LAPACK call over the batch axis and allocate their outputs inside the timed call.
- The tenferro-rs runner builds its direct `CpuBackend`, the trace runtime, and the eager runtime with exactly `--num-threads` workers, and it fails at startup when a thread environment variable disagrees with that count or when the backend execution scope reports a different Rayon thread count.
- `svd_full` remains in the table even when the selected tenferro-rs provider reports it as unsupported.
- Julia is column-major, like tenferro-rs, so the `julia-base`/`strided-jl` columns need no PyTorch/JAX-style layout reconstruction to keep the same logical fixture values.
- Julia warmup runs move JIT compilation outside the measured region, the same way PyTorch/JAX warmups do.
- Julia factorization rows materialize their factors inside the timed call (`cholesky` returns the factor matrix, `lu` returns P/L/U, `qr` returns Q/R). A Julia `Factorization` keeps its factors packed in LAPACK's working storage, so timing the compact object would compare strictly less work than the tenferro-rs and PyTorch columns, which return separate materialized tensors.
- Julia dense linalg rows run through Julia's own BLAS/LAPACK (libblastrampoline, by default OpenBLAS), recorded as `julia.blas_provider` in the run metadata. When that differs from the provider tenferro-rs and PyTorch link against (Accelerate on macOS), those rows partly compare BLAS implementations rather than framework overhead; read them together with the recorded providers.
- The Julia `lstsq` row uses `qr(a) \ rhs` rather than `a \ rhs`: the bare backslash runs a column-pivoted, rank-revealing QR (the LAPACK `gelsy` algorithm), while the PyTorch row selects the `gels` driver, so the unpivoted spelling is the like-for-like comparison.
- `julia-base` uses the natural Base/LinearAlgebra spelling and `strided-jl` the natural Strided.jl (`@strided`) spelling; each is populated only for rows where that spelling naturally applies (reductions and dense linalg have no natural Strided.jl spelling, so `strided-jl` covers elementwise/chain/transpose rows, the elementwise `cpu/output_reuse` `_into` rows, and the elementwise `cpu/complex` rows conj/mul/div/exp/log).
- Strided.jl (https://github.com/Jutho/Strided.jl) is prior art for tenferro-rs' strided-rs kernel layer; the `strided-jl` column credits that lineage directly in the report.

## Threads: 1 4

- CSV: `data/results/amd-cpu/cpu/public_api/20261126_185520/cpu_public_api_t1_20261126_185520.csv`
- CSV: `data/results/amd-cpu/cpu/public_api/20261126_185520/cpu_public_api_t4_20261126_185520.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 3.736 ± 0.097 | - | 3.680 ± 0.019 | 4.530 ± 0.023 | 3.884 ± 0.042 | 4.009 ± 1.461 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.496 ± 0.016 | - | 2.557 ± 0.018 | 2.279 ± 0.059 | 3.233 ± 0.049 | 3.276 ± 2.973 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 123.355 ± 2.519 | - | 125.495 ± 3.531 | 125.414 ± 1.546 | 127.191 ± 0.858 | 145.752 ± 1.740 | 148.956 ± 3.779 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 41.052 ± 0.260 | - | 40.765 ± 0.321 | 41.713 ± 0.550 | 66.110 ± 1.166 | 148.445 ± 1.306 | 61.831 ± 27.611 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 105.103 ± 1.562 | - | 111.269 ± 1.557 | 93.361 ± 0.881 | 93.798 ± 0.911 | 141.222 ± 9.260 | 147.513 ± 3.908 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 44.056 ± 0.778 | - | 43.716 ± 0.245 | 31.055 ± 0.165 | 47.327 ± 1.296 | 130.206 ± 5.031 | 61.115 ± 2.084 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 47.139 ± 0.223 | - | 47.130 ± 0.162 | 42.443 ± 0.309 | 68.382 ± 0.211 | 47.737 ± 0.420 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 23.514 ± 0.214 | - | 23.291 ± 0.113 | 29.057 ± 0.454 | 29.550 ± 0.602 | 23.770 ± 0.668 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 47.206 ± 0.276 | - | unsupported | 43.011 ± 0.223 | 68.922 ± 0.452 | 51.460 ± 19.100 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 23.451 ± 0.178 | - | unsupported | 29.168 ± 0.582 | 30.614 ± 0.985 | 26.962 ± 14.246 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 7.429 ± 0.032 | - | 7.466 ± 0.025 | 11.553 ± 0.030 | 11.900 ± 0.043 | 12.016 ± 0.177 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 10.091 ± 0.061 | - | 10.097 ± 0.024 | 14.003 ± 0.077 | 17.402 ± 0.071 | 17.787 ± 0.123 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 70.306 ± 0.334 | - | 70.453 ± 0.200 | 108.941 ± 0.860 | 103.833 ± 2.587 | 89.469 ± 13.072 | 85.372 ± 1.016 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 31.083 ± 0.404 | - | 31.105 ± 1.273 | 43.151 ± 0.074 | 52.559 ± 0.619 | 83.299 ± 11.571 | 36.688 ± 0.781 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 116.852 ± 0.343 | - | 116.900 ± 0.222 | 153.462 ± 0.876 | 176.440 ± 0.883 | 151.886 ± 6.337 | 163.075 ± 19.033 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 49.553 ± 3.489 | - | 49.531 ± 0.158 | 63.976 ± 1.990 | 72.972 ± 2.348 | 149.170 ± 5.374 | 64.098 ± 18.199 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 72.483 ± 1.751 | - | 72.231 ± 2.013 | 69.709 ± 0.609 | 74.849 ± 1.017 | 88.833 ± 3.642 | 90.239 ± 4.879 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 24.622 ± 3.404 | - | 23.956 ± 0.595 | 23.439 ± 0.382 | 35.847 ± 0.925 | 77.070 ± 0.277 | 33.579 ± 0.265 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 2.287 ± 0.041 | - | 20.773 ± 0.153 | 24.124 ± 0.232 | 16.010 ± 0.090 | 4.919 ± 0.154 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 0.944 ± 0.045 | - | 9.208 ± 0.112 | 11.285 ± 0.055 | 2.594 ± 0.525 | 4.228 ± 0.026 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 7.395 ± 0.012 | - | 7.510 ± 0.084 | 9.965 ± 0.256 | 7.563 ± 0.067 | 9.320 ± 0.253 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 4.391 ± 0.019 | - | 4.431 ± 0.022 | 5.726 ± 0.029 | 6.859 ± 0.075 | 9.990 ± 7.237 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 4.550 ± 0.029 | - | 4.521 ± 0.022 | 5.608 ± 0.952 | 5.120 ± 0.071 | 5.141 ± 0.111 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.488 ± 0.111 | - | 3.298 ± 0.080 | 3.596 ± 0.036 | 4.697 ± 2.510 | 4.398 ± 0.112 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 6.434 ± 0.050 | - | 6.460 ± 0.054 | 7.689 ± 0.074 | 5.439 ± 0.032 | 5.595 ± 0.160 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 6.446 ± 0.024 | - | 6.457 ± 0.046 | 5.853 ± 0.214 | 5.936 ± 0.066 | 6.218 ± 1.356 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 47.256 ± 0.137 | - | 47.068 ± 0.218 | 43.295 ± 0.529 | 68.319 ± 0.438 | 48.181 ± 0.516 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 23.491 ± 0.088 | - | 23.521 ± 0.215 | 28.950 ± 0.954 | 29.431 ± 1.134 | 23.502 ± 1.242 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 48.602 ± 0.150 | - | unsupported | 50.341 ± 0.932 | 62.329 ± 0.456 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 22.877 ± 0.174 | - | unsupported | 25.323 ± 0.234 | 35.270 ± 0.548 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 122.784 ± 0.909 | - | 124.856 ± 3.489 | 123.233 ± 1.694 | 125.687 ± 2.080 | 142.748 ± 16.655 | 154.593 ± 0.829 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 40.966 ± 0.984 | - | 75.697 ± 3.632 | 70.732 ± 8.104 | 64.500 ± 0.875 | 143.502 ± 3.469 | 63.966 ± 0.698 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 133.838 ± 1.767 | - | 131.733 ± 4.598 | 132.649 ± 2.637 | 141.093 ± 3.535 | 150.387 ± 11.963 | 165.188 ± 4.878 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 51.866 ± 0.786 | - | 51.423 ± 1.827 | 44.726 ± 1.453 | 70.956 ± 2.152 | 150.348 ± 8.521 | 70.182 ± 4.626 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 27.652 ± 0.414 | - | 58.433 ± 0.245 | 75.600 ± 0.696 | 51.160 ± 0.319 | 80.336 ± 8.483 | 85.518 ± 9.400 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 12.344 ± 0.548 | - | 25.373 ± 0.583 | 32.802 ± 0.398 | 26.064 ± 0.138 | 81.888 ± 0.308 | 33.162 ± 16.621 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 9.856 ± 0.184 | - | 9.069 ± 0.196 | 36.407 ± 0.210 | 37.675 ± 0.583 | 36.488 ± 9.656 | 41.578 ± 10.355 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 6.737 ± 0.118 | - | 6.871 ± 0.057 | 12.690 ± 0.449 | 20.356 ± 0.338 | 36.108 ± 9.765 | 14.257 ± 17.051 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 21.863 ± 0.197 | - | 21.855 ± 0.419 | 42.618 ± 0.406 | 41.707 ± 0.529 | 20.198 ± 0.299 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 14.454 ± 0.434 | - | 14.409 ± 0.075 | 19.785 ± 0.607 | 23.858 ± 0.253 | 22.898 ± 0.314 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 28.203 ± 0.050 | - | 28.349 ± 0.044 | 56.915 ± 0.421 | 108.603 ± 0.889 | 67.441 ± 9.613 | 73.042 ± 10.087 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 11.885 ± 2.809 | - | 12.008 ± 0.037 | 22.014 ± 0.458 | 54.407 ± 0.365 | 66.762 ± 14.493 | 44.583 ± 20.147 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 133.544 ± 2.087 | - | 133.460 ± 2.052 | 133.551 ± 2.713 | 144.404 ± 0.648 | 152.893 ± 4.493 | 163.925 ± 4.923 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 45.138 ± 0.330 | - | 45.092 ± 0.567 | 44.992 ± 1.042 | 71.055 ± 1.016 | 153.811 ± 3.532 | 67.975 ± 0.701 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 20.585 ± 0.040 | - | 20.917 ± 0.131 | 44.292 ± 0.253 | 41.559 ± 0.580 | 75.249 ± 15.128 | 74.083 ± 13.077 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 8.460 ± 0.800 | - | 8.474 ± 0.050 | 16.691 ± 2.521 | 21.605 ± 0.409 | 63.091 ± 13.867 | 40.318 ± 26.641 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 20.393 ± 0.020 | - | 20.623 ± 0.042 | 31.969 ± 0.621 | 27.870 ± 0.290 | 38.153 ± 11.261 | 41.146 ± 9.989 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 8.088 ± 0.008 | - | 8.097 ± 0.020 | 11.566 ± 0.485 | 14.021 ± 0.050 | 38.168 ± 0.245 | 17.560 ± 10.457 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 25.841 ± 0.172 | - | 25.596 ± 0.060 | 49.981 ± 0.066 | 65.637 ± 0.530 | 83.610 ± 13.253 | 89.338 ± 11.998 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 10.514 ± 0.027 | - | 10.654 ± 3.888 | 19.157 ± 0.266 | 32.952 ± 0.144 | 84.556 ± 14.058 | 42.950 ± 23.835 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 13.190 ± 0.019 | - | 13.312 ± 0.035 | 32.567 ± 0.547 | 37.198 ± 0.259 | 46.283 ± 8.805 | 50.248 ± 9.069 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 5.637 ± 0.013 | - | 5.651 ± 0.017 | 11.944 ± 0.581 | 18.731 ± 0.084 | 46.624 ± 8.120 | 20.886 ± 11.813 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.459 ± 0.013 | 0.478 ± 0.014 | unsupported | unsupported | 0.492 ± 0.026 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.461 ± 0.005 | 0.494 ± 0.008 | unsupported | unsupported | 0.372 ± 0.013 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 137.519 ± 1.696 | - | 137.611 ± 2.902 | 133.015 ± 1.862 | 137.865 ± 1.006 | 152.376 ± 27.080 | 164.022 ± 1.653 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 45.506 ± 0.317 | - | 45.288 ± 1.411 | 44.990 ± 0.431 | 71.752 ± 0.642 | 156.774 ± 2.608 | 69.416 ± 2.557 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 136.056 ± 2.770 | - | 137.578 ± 3.036 | 131.020 ± 1.999 | 139.419 ± 0.908 | 152.437 ± 21.704 | 166.928 ± 1.233 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 45.714 ± 1.812 | - | 45.170 ± 1.450 | 45.669 ± 0.472 | 71.638 ± 0.946 | 154.785 ± 6.892 | 69.170 ± 0.534 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 134.329 ± 1.835 | - | 133.740 ± 4.280 | 132.392 ± 2.356 | 142.387 ± 5.675 | 150.652 ± 4.869 | 164.264 ± 0.533 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 49.007 ± 3.346 | - | 45.951 ± 0.965 | 45.600 ± 1.019 | 72.033 ± 1.137 | 151.135 ± 44.531 | 68.821 ± 2.776 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 123.832 ± 0.551 | - | 124.436 ± 5.945 | 125.245 ± 1.615 | 125.420 ± 0.281 | 141.352 ± 36.509 | 155.751 ± 2.552 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 40.437 ± 0.479 | - | 40.697 ± 0.521 | 72.762 ± 9.560 | 64.245 ± 1.765 | 140.490 ± 1.637 | 65.551 ± 0.747 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 57.690 ± 0.132 | - | 58.782 ± 0.169 | 62.741 ± 0.582 | 74.270 ± 0.272 | 125.458 ± 1.045 | 124.513 ± 16.521 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 23.744 ± 0.822 | - | 23.673 ± 0.114 | 22.066 ± 0.407 | 37.330 ± 0.830 | 125.609 ± 0.672 | 50.297 ± 16.326 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 16.165 ± 0.130 | - | 16.290 ± 0.191 | 11.989 ± 0.151 | 26.046 ± 1.408 | 13.897 ± 0.319 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 7.268 ± 0.187 | - | 7.030 ± 0.659 | 6.682 ± 0.206 | 13.747 ± 0.070 | 13.072 ± 0.155 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.847 ± 0.051 | - | 1.913 ± 0.044 | 24.434 ± 0.197 | 2.495 ± 0.056 | 1.462 ± 0.142 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.776 ± 0.023 | - | 0.802 ± 0.007 | 11.125 ± 0.349 | 1.467 ± 0.025 | 1.276 ± 0.098 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 2.513 ± 0.035 | - | 2.495 ± 0.056 | 4.051 ± 0.039 | 2.422 ± 0.387 | 1.241 ± 0.190 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 1.278 ± 0.141 | - | 1.272 ± 0.045 | 2.314 ± 0.029 | 1.508 ± 0.309 | 1.468 ± 0.175 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 15.973 ± 0.108 | - | 16.073 ± 0.200 | 12.498 ± 0.295 | 26.117 ± 0.175 | 13.923 ± 0.312 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 7.114 ± 0.266 | - | 7.496 ± 0.234 | 6.773 ± 0.146 | 13.844 ± 0.090 | 12.503 ± 0.118 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.823 ± 0.046 | - | 1.834 ± 0.094 | 23.158 ± 0.479 | 2.435 ± 0.026 | 1.227 ± 0.097 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.820 ± 0.045 | - | 0.809 ± 0.026 | 10.421 ± 0.582 | 1.337 ± 0.033 | 1.694 ± 0.065 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 10.247 ± 0.031 | - | 10.193 ± 0.026 | 16.237 ± 0.314 | 11.936 ± 0.347 | 6.610 ± 0.072 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 5.160 ± 0.329 | - | 5.139 ± 0.062 | 8.797 ± 0.043 | 6.155 ± 0.756 | 6.555 ± 0.116 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 10.920 ± 0.150 | - | 11.254 ± 0.066 | 10.875 ± 0.138 | 20.179 ± 0.691 | 11.554 ± 0.292 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 6.541 ± 0.099 | - | 6.504 ± 0.139 | 6.441 ± 0.102 | 10.977 ± 0.213 | 11.232 ± 0.164 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 1.026 ± 0.160 | - | 1.049 ± 0.173 | 2.075 ± 0.049 | 1.985 ± 0.018 | 0.908 ± 0.120 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.427 ± 0.093 | - | 0.440 ± 0.058 | 0.947 ± 0.029 | 1.147 ± 0.033 | 1.065 ± 0.100 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.207 ± 0.141 | - | 1.153 ± 0.156 | 0.845 ± 0.029 | 3.209 ± 0.258 | 1.036 ± 0.058 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.719 ± 0.068 | - | 0.658 ± 0.062 | 0.353 ± 0.035 | 1.743 ± 0.236 | 1.259 ± 0.096 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 10.877 ± 0.236 | - | 11.073 ± 0.185 | 10.973 ± 0.354 | 20.422 ± 0.365 | 11.722 ± 0.380 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 6.959 ± 0.205 | - | 6.448 ± 0.053 | 6.413 ± 0.072 | 10.830 ± 0.092 | 11.127 ± 0.323 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 1.021 ± 0.016 | - | 1.023 ± 0.261 | 2.005 ± 0.104 | 1.975 ± 0.028 | 0.915 ± 0.087 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.397 ± 0.061 | - | 0.402 ± 0.062 | 0.907 ± 0.049 | 1.120 ± 0.003 | 1.045 ± 0.132 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.177 ± 0.062 | - | 1.224 ± 0.209 | 0.810 ± 0.026 | 3.308 ± 0.304 | 1.078 ± 0.049 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.588 ± 0.035 | - | 0.635 ± 0.091 | 0.351 ± 0.041 | 1.614 ± 0.235 | 1.303 ± 0.061 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 1.109 ± 0.047 | 1.097 ± 0.043 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.376 ± 0.021 | 0.383 ± 0.034 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 39.820 ± 0.982 | - | 38.415 ± 0.105 | 55.744 ± 0.635 | 71.075 ± 0.275 | 80.248 ± 20.900 | 78.961 ± 8.782 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 18.579 ± 0.116 | - | 18.459 ± 0.111 | 20.703 ± 16.897 | 37.224 ± 0.941 | 70.405 ± 11.720 | 38.290 ± 12.138 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 144.607 ± 1.174 | - | 145.517 ± 3.457 | 145.865 ± 1.049 | 248.239 ± 2.132 | 265.050 ± 3.099 | 264.477 ± 4.623 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 46.745 ± 0.135 | - | 46.452 ± 0.318 | 47.471 ± 0.552 | 125.798 ± 1.094 | 263.947 ± 1.561 | 108.657 ± 17.149 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 138.319 ± 3.177 | - | 138.753 ± 1.922 | 143.884 ± 4.176 | 151.923 ± 1.372 | 154.119 ± 4.828 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 48.193 ± 0.801 | - | 47.997 ± 0.703 | 52.264 ± 1.625 | 79.599 ± 1.472 | 150.608 ± 6.642 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 130.009 ± 1.319 | - | 128.820 ± 4.727 | 127.012 ± 0.609 | 129.006 ± 3.220 | 142.817 ± 0.749 | 151.007 ± 0.637 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 42.679 ± 30.235 | - | 40.796 ± 0.250 | 41.226 ± 0.899 | 67.906 ± 1.097 | 143.237 ± 24.176 | 62.979 ± 0.752 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 24.486 ± 0.081 | - | 24.507 ± 0.094 | 58.306 ± 0.291 | 108.834 ± 0.431 | 66.551 ± 13.636 | 70.628 ± 17.142 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 10.204 ± 0.075 | - | 10.382 ± 0.047 | 22.516 ± 1.673 | 54.615 ± 0.520 | 75.473 ± 9.349 | 40.656 ± 11.224 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 134.842 ± 0.831 | - | 134.794 ± 4.178 | 187.862 ± 0.815 | 136.199 ± 2.812 | 215.227 ± 0.473 | 223.701 ± 4.224 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 44.212 ± 0.172 | - | 44.285 ± 1.972 | 68.991 ± 0.489 | 68.896 ± 0.429 | 217.879 ± 5.231 | 93.627 ± 32.630 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 131.456 ± 4.414 | - | 132.128 ± 4.629 | 132.936 ± 2.191 | 139.359 ± 0.716 | 148.769 ± 3.543 | 166.026 ± 2.750 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 49.749 ± 2.743 | - | 48.537 ± 1.132 | 45.571 ± 1.105 | 71.018 ± 2.854 | 150.126 ± 4.068 | 69.198 ± 2.311 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 51.237 ± 0.048 | - | 51.247 ± 0.044 | 76.807 ± 0.246 | 39.716 ± 0.404 | 82.732 ± 13.243 | 87.853 ± 13.453 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 20.405 ± 0.072 | - | 20.508 ± 0.049 | 30.425 ± 0.461 | 20.444 ± 0.505 | 79.886 ± 13.784 | 37.974 ± 16.632 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 137.303 ± 2.982 | 137.862 ± 3.463 | unsupported | 140.873 ± 3.383 | 152.439 ± 1.442 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 47.733 ± 0.510 | 46.281 ± 0.620 | unsupported | 52.385 ± 1.032 | 79.457 ± 0.744 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.795 ± 0.015 | - | 0.800 ± 0.093 | 0.840 ± 0.061 | 0.964 ± 0.122 | 7.580 ± 3.870 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.318 ± 0.019 | - | 0.341 ± 0.068 | 0.378 ± 0.026 | 0.992 ± 0.151 | 7.472 ± 3.962 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.874 ± 0.452 | - | 0.841 ± 0.206 | 0.875 ± 0.060 | 0.979 ± 0.364 | 1.643 ± 1.054 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.446 ± 0.185 | - | 0.367 ± 0.091 | 0.383 ± 0.026 | 0.720 ± 0.198 | 7.786 ± 4.054 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.236 ± 0.060 | - | unsupported | 1.280 ± 0.087 | 1.309 ± 0.206 | 7.967 ± 3.022 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.596 ± 0.040 | - | unsupported | 0.647 ± 0.052 | 1.004 ± 0.129 | 6.707 ± 3.959 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.396 ± 0.030 | - | 0.480 ± 0.028 | 0.714 ± 0.057 | 0.435 ± 0.028 | 1.265 ± 0.066 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.356 ± 0.086 | - | 0.410 ± 0.020 | 0.364 ± 0.054 | 0.399 ± 0.038 | 1.195 ± 0.082 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.966 ± 0.331 | - | 0.800 ± 0.095 | 1.379 ± 0.188 | 1.014 ± 0.149 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.415 ± 0.232 | - | 0.382 ± 0.043 | 0.625 ± 0.166 | 1.085 ± 0.042 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.839 ± 0.043 | - | 0.862 ± 0.029 | 0.784 ± 0.040 | 1.078 ± 0.289 | 1.490 ± 10.893 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.311 ± 0.046 | - | 0.333 ± 0.067 | 0.379 ± 0.063 | 0.571 ± 0.116 | 8.191 ± 8.538 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.790 ± 0.007 | - | 0.912 ± 0.014 | 0.456 ± 0.037 | 0.489 ± 0.054 | 1.405 ± 0.109 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 1.190 ± 0.015 | - | 1.328 ± 0.011 | 0.624 ± 0.020 | 0.479 ± 0.041 | 1.462 ± 0.072 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.623 ± 0.062 | - | 0.002 ± 0.000 | 1.751 ± 0.110 | 1.445 ± 0.090 | 9.221 ± 5.596 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.680 ± 0.068 | - | 0.003 ± 0.000 | 0.822 ± 0.174 | 1.063 ± 0.086 | 9.223 ± 4.925 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.075 ± 0.001 | - | 0.077 ± 0.006 | 0.075 ± 0.001 | 0.098 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.133 ± 0.001 | - | 0.135 ± 0.007 | 0.123 ± 0.002 | 0.149 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.259 ± 0.009 | - | 0.258 ± 0.010 | 0.279 ± 0.011 | 0.326 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.050 ± 0.000 | - | 0.053 ± 0.000 | 0.144 ± 0.001 | 0.120 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.001 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.001 | 0.036 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.027 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.402 ± 0.008 | - | 0.391 ± 0.010 | 1.161 ± 0.009 | 0.676 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.019 ± 0.000 | - | 0.020 ± 0.000 | 0.040 ± 0.003 | 0.061 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.030 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.077 ± 0.000 | - | 0.119 ± 0.001 | 0.085 ± 0.002 | 0.096 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.134 ± 0.000 | - | 0.192 ± 0.013 | 0.124 ± 0.002 | 0.160 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.256 ± 0.021 | - | 0.383 ± 0.010 | 0.340 ± 0.012 | 0.347 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.057 ± 0.003 | - | 0.099 ± 0.001 | 0.178 ± 0.002 | 0.122 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.016 ± 0.001 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.010 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.009 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.003 ± 0.000 | - | 0.007 ± 0.001 | 0.011 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.024 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.003 ± 0.000 | - | 0.007 ± 0.001 | 0.011 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.242 ± 0.007 | - | 0.234 ± 0.017 | 1.364 ± 0.029 | 1.149 ± 0.149 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.019 ± 0.000 | - | 0.037 ± 0.002 | 0.059 ± 0.001 | 0.054 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.001 | 0.010 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.004 ± 0.000 | - | 0.008 ± 0.001 | 0.015 ± 0.016 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.545 ± 0.013 | - | 0.552 ± 0.009 | 0.720 ± 0.013 | 0.880 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 2.158 ± 0.021 | - | 2.164 ± 0.052 | 2.170 ± 0.013 | 2.693 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 6.648 ± 0.012 | - | 6.700 ± 0.041 | 6.535 ± 0.030 | 8.913 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 2.413 ± 0.022 | - | 2.420 ± 0.016 | 3.217 ± 0.005 | 2.631 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.001 | 0.026 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.103 ± 0.001 | - | 0.113 ± 0.035 | 0.121 ± 0.002 | 0.192 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.010 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.017 ± 0.005 | 0.039 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.001 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.016 ± 0.000 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.021 ± 0.000 | - | 0.023 ± 0.000 | 0.030 ± 0.001 | 0.097 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.024 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.010 ± 0.001 | - | 0.012 ± 0.000 | 0.018 ± 0.001 | 0.041 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.028 ± 0.000 | - | 0.030 ± 0.000 | 0.036 ± 0.004 | 0.103 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 19.305 ± 0.089 | - | 19.290 ± 0.066 | 25.685 ± 0.154 | 20.545 ± 0.132 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.001 | 0.014 ± 0.000 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.821 ± 0.005 | - | 0.825 ± 0.006 | 0.877 ± 0.032 | 1.007 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.019 ± 0.000 | - | 0.021 ± 0.001 | 0.027 ± 0.001 | 0.043 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.054 ± 0.001 | - | 0.056 ± 0.000 | 0.061 ± 0.001 | 0.141 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.540 ± 0.016 | - | 0.873 ± 0.007 | 0.728 ± 0.431 | 0.919 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 0.860 ± 0.005 | - | 0.865 ± 0.002 | 2.132 ± 0.011 | 2.961 ± 0.372 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 2.490 ± 0.011 | - | 2.499 ± 0.018 | 9.304 ± 0.021 | 7.204 ± 1.525 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 2.400 ± 0.038 | - | 2.413 ± 0.014 | 2.250 ± 0.046 | 3.402 ± 0.611 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.029 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.105 ± 0.003 | - | 0.158 ± 0.002 | 0.177 ± 0.002 | 0.192 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.010 ± 0.000 | 0.012 ± 0.001 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.009 ± 0.000 | - | 0.016 ± 0.000 | 0.016 ± 0.001 | 0.034 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.004 ± 0.001 | - | 0.009 ± 0.001 | 0.016 ± 0.002 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.014 ± 0.008 | - | 0.016 ± 0.000 | 0.016 ± 0.001 | 0.030 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.022 ± 0.000 | - | 0.034 ± 0.002 | 0.030 ± 0.001 | 0.101 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.002 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.011 ± 0.000 | - | 0.019 ± 0.004 | 0.018 ± 0.001 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.028 ± 0.000 | - | 0.045 ± 0.003 | 0.036 ± 0.000 | 0.093 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 8.156 ± 0.037 | - | 8.131 ± 0.033 | 17.667 ± 0.193 | 28.448 ± 11.596 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.013 ± 0.001 | 0.022 ± 0.001 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.355 ± 0.015 | - | 0.356 ± 0.013 | 1.250 ± 0.008 | 1.995 ± 0.323 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.019 ± 0.000 | - | 0.031 ± 0.003 | 0.026 ± 0.001 | 0.066 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.054 ± 0.001 | - | 0.081 ± 0.003 | 0.061 ± 0.001 | 0.131 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.492 ± 0.027 | - | 0.483 ± 0.011 | 0.911 ± 0.027 | 0.445 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.956 ± 0.029 | - | 1.946 ± 0.010 | 1.959 ± 0.010 | 1.994 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 6.418 ± 0.057 | - | 6.415 ± 0.029 | 5.702 ± 0.012 | 6.441 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.890 ± 0.016 | - | 1.672 ± 0.014 | 1.333 ± 0.020 | 1.464 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.012 | 0.022 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.084 ± 0.018 | - | 0.091 ± 0.023 | 0.093 ± 0.005 | 0.129 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.016 ± 0.001 | 0.029 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.015 ± 0.001 | 0.031 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.001 | 0.021 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.021 ± 0.004 | 0.033 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.020 ± 0.000 | - | 0.022 ± 0.000 | 0.027 ± 0.001 | 0.041 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.026 ± 0.001 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.026 ± 0.000 | - | 0.028 ± 0.000 | 0.032 ± 0.002 | 0.047 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 14.518 ± 0.082 | - | 14.362 ± 0.135 | 10.430 ± 0.018 | 11.287 ± 1.093 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.001 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.696 ± 0.004 | - | 0.696 ± 0.012 | 0.626 ± 0.015 | 0.744 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.017 ± 0.000 | - | 0.019 ± 0.000 | 0.037 ± 0.003 | 0.036 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.052 ± 0.001 | - | 0.053 ± 0.001 | 0.053 ± 0.001 | 0.119 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.480 ± 0.011 | - | 0.787 ± 0.014 | 0.801 ± 0.095 | 0.447 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.908 ± 0.008 | - | 1.915 ± 0.007 | 1.920 ± 0.006 | 1.271 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 2.292 ± 0.012 | - | 2.302 ± 0.009 | 8.512 ± 0.023 | 3.678 ± 0.123 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.642 ± 0.017 | - | 1.805 ± 0.032 | 1.196 ± 0.044 | 1.794 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.012 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.085 ± 0.008 | - | 0.126 ± 0.005 | 0.136 ± 0.005 | 0.147 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.001 | 0.028 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.009 ± 0.000 | - | 0.015 ± 0.001 | 0.015 ± 0.001 | 0.035 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.014 ± 0.002 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.008 ± 0.000 | - | 0.015 ± 0.000 | 0.014 ± 0.001 | 0.028 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.021 ± 0.001 | - | 0.032 ± 0.004 | 0.026 ± 0.001 | 0.043 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.010 ± 0.001 | 0.015 ± 0.003 | 0.022 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.010 ± 0.000 | - | 0.018 ± 0.003 | 0.017 ± 0.001 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.026 ± 0.000 | - | 0.041 ± 0.002 | 0.031 ± 0.001 | 0.050 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 5.632 ± 0.125 | - | 5.258 ± 0.018 | 9.122 ± 0.071 | 14.053 ± 0.212 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | 0.013 ± 0.000 | 0.018 ± 0.001 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.274 ± 0.017 | - | 0.285 ± 0.018 | 0.926 ± 0.004 | 0.455 ± 0.059 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.017 ± 0.000 | - | 0.028 ± 0.001 | 0.023 ± 0.001 | 0.046 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.051 ± 0.000 | - | 0.074 ± 0.003 | 0.052 ± 0.001 | 0.091 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.347 ± 0.017 | - | 0.338 ± 0.010 | 0.238 ± 0.013 | 0.463 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 1.564 ± 0.043 | - | 1.571 ± 0.013 | 0.948 ± 0.005 | 1.938 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 4.931 ± 0.028 | - | 4.944 ± 0.008 | 2.877 ± 0.013 | 6.425 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.899 ± 0.004 | - | 0.908 ± 0.016 | 0.551 ± 0.014 | 1.431 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.010 ± 0.001 | 0.015 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.062 ± 0.000 | - | 0.063 ± 0.001 | 0.040 ± 0.002 | 0.124 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.001 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.012 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.015 ± 0.000 | - | 0.018 ± 0.000 | 0.020 ± 0.001 | 0.042 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.011 ± 0.001 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.001 | 0.014 ± 0.000 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.020 ± 0.000 | - | 0.022 ± 0.000 | 0.022 ± 0.001 | 0.040 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 7.200 ± 0.022 | - | 7.204 ± 0.049 | 4.268 ± 0.010 | 11.072 ± 0.068 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.490 ± 0.008 | - | 0.491 ± 0.010 | 0.244 ± 0.015 | 0.730 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.017 ± 0.000 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.039 ± 0.000 | - | 0.041 ± 0.000 | 0.032 ± 0.000 | 0.092 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.339 ± 0.018 | - | 0.585 ± 0.010 | 0.400 ± 0.013 | 0.437 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 1.555 ± 0.008 | - | 1.562 ± 0.334 | 0.946 ± 0.002 | 1.284 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 1.726 ± 0.012 | - | 1.730 ± 0.016 | 3.655 ± 0.010 | 3.620 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.910 ± 0.003 | - | 1.145 ± 0.070 | 0.705 ± 0.017 | 1.767 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.014 ± 0.001 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.063 ± 0.001 | - | 0.084 ± 0.002 | 0.055 ± 0.004 | 0.143 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.001 | 0.015 ± 0.002 | 0.018 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.007 ± 0.000 | - | 0.012 ± 0.000 | 0.013 ± 0.001 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.015 ± 0.000 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.011 ± 0.000 | - | 0.012 ± 0.000 | 0.019 ± 0.002 | 0.025 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.016 ± 0.001 | - | 0.024 ± 0.001 | 0.019 ± 0.001 | 0.042 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.015 ± 0.001 | 0.018 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.014 ± 0.004 | - | 0.014 ± 0.000 | 0.019 ± 0.003 | 0.027 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.020 ± 0.000 | - | 0.031 ± 0.002 | 0.022 ± 0.001 | 0.043 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 2.447 ± 0.038 | - | 2.451 ± 0.014 | 5.239 ± 0.065 | 16.985 ± 5.060 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.009 ± 0.000 | - | 0.010 ± 0.000 | 0.017 ± 0.003 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.177 ± 0.014 | - | 0.181 ± 0.024 | 0.322 ± 0.017 | 0.416 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.014 ± 0.010 | - | 0.023 ± 0.000 | 0.024 ± 0.001 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.039 ± 0.000 | - | 0.055 ± 0.003 | 0.033 ± 0.001 | 0.084 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.130 ± 0.009 | - | 0.128 ± 0.003 | 0.200 ± 0.012 | 0.146 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.269 ± 0.024 | - | 0.271 ± 0.016 | 0.272 ± 0.012 | 0.213 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.715 ± 0.009 | - | 0.716 ± 0.004 | 0.555 ± 0.016 | 0.580 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.131 ± 0.005 | - | 0.133 ± 0.007 | 0.174 ± 0.016 | 0.220 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.013 ± 0.001 | 0.025 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.025 ± 0.001 | 0.052 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.013 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.028 ± 0.004 | 0.058 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.013 ± 0.001 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.001 | 0.020 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.017 ± 0.002 | 0.023 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.005 | 0.017 ± 0.001 | 0.036 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.005 | - | 0.005 ± 0.000 | 0.016 ± 0.001 | 0.023 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.000 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.000 | 0.044 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 1.002 ± 0.019 | - | 0.992 ± 0.005 | 1.287 ± 0.007 | 1.281 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.063 ± 0.003 | - | 0.065 ± 0.001 | 0.087 ± 0.001 | 0.137 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.016 ± 0.001 | 0.029 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.020 ± 0.000 | 0.042 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.127 ± 0.001 | - | 0.244 ± 0.013 | 0.174 ± 0.002 | 0.138 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.255 ± 0.022 | - | 0.474 ± 0.012 | 0.235 ± 0.017 | 0.215 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.328 ± 0.014 | - | 0.340 ± 0.857 | 0.401 ± 0.010 | 0.679 ± 0.074 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.211 ± 0.001 | - | 0.215 ± 0.009 | 0.227 ± 0.019 | 0.204 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.024 ± 0.002 | 0.021 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.012 ± 0.000 | - | 0.021 ± 0.003 | 0.039 ± 0.001 | 0.062 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.025 ± 0.004 | 0.031 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.013 ± 0.000 | - | 0.022 ± 0.001 | 0.042 ± 0.001 | 0.057 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.023 ± 0.003 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.025 ± 0.001 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.026 ± 0.000 | 0.023 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.006 ± 0.000 | - | 0.011 ± 0.000 | 0.028 ± 0.004 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.026 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.026 ± 0.002 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.006 ± 0.000 | - | 0.012 ± 0.000 | 0.028 ± 0.001 | 0.019 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.519 ± 0.018 | - | 0.525 ± 0.015 | 0.721 ± 0.028 | 0.967 ± 0.059 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.027 ± 0.000 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.062 ± 0.000 | - | 0.109 ± 0.001 | 0.080 ± 0.001 | 0.215 ± 0.065 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.005 ± 0.000 | - | 0.011 ± 0.001 | 0.028 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.009 ± 0.000 | - | 0.017 ± 0.000 | 0.031 ± 0.006 | 0.022 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.249 ± 0.007 | - | 0.252 ± 0.003 | 0.222 ± 0.003 | 0.347 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.695 ± 0.009 | - | 0.696 ± 0.010 | 0.526 ± 0.009 | 0.857 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 2.169 ± 0.009 | - | 2.143 ± 0.006 | 1.361 ± 0.008 | 2.351 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.475 ± 0.016 | - | 0.472 ± 0.017 | 0.564 ± 0.011 | 0.664 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.029 ± 0.000 | - | 0.032 ± 0.003 | 0.026 ± 0.001 | 0.103 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.026 ± 0.003 | - | 0.029 ± 0.000 | 0.030 ± 0.001 | 0.061 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.027 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.033 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.010 ± 0.000 | 0.049 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 3.812 ± 0.062 | - | 3.737 ± 0.051 | 4.437 ± 0.040 | 4.736 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.220 ± 0.009 | - | 0.224 ± 0.002 | 0.175 ± 0.010 | 0.240 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.033 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.019 ± 0.000 | - | 0.020 ± 0.001 | 0.016 ± 0.002 | 0.038 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.248 ± 0.020 | - | 0.383 ± 0.012 | 0.322 ± 0.014 | 0.322 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.673 ± 0.012 | - | 1.001 ± 0.003 | 0.760 ± 0.014 | 0.883 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 0.852 ± 0.004 | - | 0.861 ± 0.008 | 2.024 ± 0.012 | 1.347 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.466 ± 0.016 | - | 0.756 ± 0.020 | 0.513 ± 0.027 | 0.937 ± 0.077 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.001 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.030 ± 0.000 | - | 0.053 ± 0.003 | 0.044 ± 0.001 | 0.079 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.004 | 0.011 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.027 ± 0.000 | - | 0.046 ± 0.002 | 0.051 ± 0.007 | 0.060 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | 0.010 ± 0.001 | 0.012 ± 0.000 | 0.030 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.001 | 0.022 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.001 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.014 ± 0.005 | - | 0.015 ± 0.000 | 0.016 ± 0.000 | 0.033 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.001 | 0.011 ± 0.000 | 0.027 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.009 ± 0.000 | - | 0.010 ± 0.004 | 0.013 ± 0.000 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.017 ± 0.007 | - | 0.018 ± 0.000 | 0.018 ± 0.000 | 0.031 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 1.839 ± 0.015 | - | 1.820 ± 0.018 | 3.758 ± 0.048 | 7.765 ± 0.768 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.005 ± 0.000 | - | 0.009 ± 0.001 | 0.013 ± 0.004 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.118 ± 0.001 | - | 0.123 ± 0.004 | 0.289 ± 0.012 | 0.293 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.013 ± 0.004 | - | 0.014 ± 0.000 | 0.015 ± 0.001 | 0.032 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.019 ± 0.001 | - | 0.030 ± 0.003 | 0.026 ± 0.000 | 0.039 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.214 ± 0.037 | - | 0.138 ± 0.004 | 0.099 ± 0.001 | 0.255 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.302 ± 0.013 | - | 0.259 ± 0.003 | 0.141 ± 0.004 | 0.383 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.789 ± 0.006 | - | 0.747 ± 0.003 | 0.405 ± 0.012 | 0.814 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.095 ± 0.001 | - | 0.123 ± 0.042 | 0.147 ± 0.007 | 0.200 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.017 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.008 ± 0.001 | - | 0.013 ± 0.002 | 0.031 ± 0.003 | 0.078 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.001 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.021 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.021 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.020 ± 0.001 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.010 ± 0.000 | 0.020 ± 0.006 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.730 ± 0.013 | - | 0.781 ± 0.022 | 1.023 ± 0.009 | 1.034 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.049 ± 0.000 | - | 0.055 ± 0.000 | 0.081 ± 0.013 | 0.147 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.018 ± 0.000 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.008 ± 0.000 | - | 0.013 ± 0.000 | 0.021 ± 0.000 | 0.026 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.189 ± 0.025 | - | 0.251 ± 0.012 | 0.133 ± 0.003 | 0.262 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.294 ± 0.012 | - | 0.417 ± 0.013 | 0.172 ± 0.001 | 0.372 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.366 ± 0.012 | - | 0.375 ± 0.020 | 0.392 ± 0.017 | 0.838 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.145 ± 0.004 | - | 0.174 ± 0.080 | 0.204 ± 0.004 | 0.170 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.012 ± 0.000 | 0.026 ± 0.005 | 0.017 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.009 ± 0.000 | - | 0.023 ± 0.004 | 0.047 ± 0.006 | 0.054 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.005 ± 0.000 | - | 0.013 ± 0.000 | 0.026 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.013 ± 0.003 | 0.026 ± 0.002 | 0.018 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.013 ± 0.000 | 0.026 ± 0.003 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.014 ± 0.000 | 0.027 ± 0.004 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.015 ± 0.000 | 0.030 ± 0.004 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.013 ± 0.000 | 0.028 ± 0.004 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.014 ± 0.000 | 0.026 ± 0.001 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.016 ± 0.000 | 0.032 ± 0.004 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.359 ± 0.014 | - | 0.452 ± 0.014 | 0.582 ± 0.020 | 0.747 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.014 ± 0.000 | 0.029 ± 0.000 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.049 ± 0.000 | - | 0.087 ± 0.003 | 0.074 ± 0.005 | 0.152 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.005 ± 0.000 | - | 0.015 ± 0.000 | 0.029 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.009 ± 0.000 | - | 0.022 ± 0.002 | 0.032 ± 0.001 | 0.028 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.190 ± 0.009 | - | 1.213 ± 0.006 | 1.556 ± 0.012 | 2.021 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 3.560 ± 0.005 | - | 3.820 ± 0.017 | 5.156 ± 0.081 | 5.117 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 11.033 ± 0.047 | - | 12.203 ± 0.041 | 16.035 ± 0.088 | 14.733 ± 0.092 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 4.215 ± 0.045 | - | 4.240 ± 0.052 | 5.600 ± 0.033 | 4.231 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.001 | 0.014 ± 0.001 | 0.035 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.180 ± 0.001 | - | 0.199 ± 0.014 | 0.241 ± 0.020 | 0.270 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.020 ± 0.001 | 0.036 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.151 ± 0.002 | - | 0.165 ± 0.003 | 0.208 ± 0.017 | 0.220 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.013 ± 0.001 | - | 0.016 ± 0.001 | 0.032 ± 0.001 | 0.056 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.033 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.013 ± 0.000 | - | 0.016 ± 0.000 | 0.030 ± 0.001 | 0.045 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.034 ± 0.000 | - | 0.039 ± 0.000 | 0.064 ± 0.005 | 0.072 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.019 ± 0.001 | 0.080 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.016 ± 0.000 | - | 0.019 ± 0.000 | 0.035 ± 0.001 | 0.048 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.044 ± 0.000 | - | 0.051 ± 0.000 | 0.079 ± 0.001 | 0.089 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 33.603 ± 0.256 | - | 33.863 ± 0.173 | 44.060 ± 0.206 | 33.367 ± 0.167 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.012 ± 0.007 | - | 0.014 ± 0.000 | 0.025 ± 0.001 | 0.051 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.406 ± 0.012 | - | 1.526 ± 0.017 | 1.649 ± 0.024 | 1.588 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.030 ± 0.000 | - | 0.034 ± 0.000 | 0.057 ± 0.003 | 0.068 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.086 ± 0.001 | - | 0.097 ± 0.002 | 0.141 ± 0.010 | 0.163 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.185 ± 0.013 | - | 1.204 ± 0.764 | 1.520 ± 0.008 | 2.054 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 1.430 ± 0.011 | - | 1.432 ± 0.019 | 5.097 ± 0.007 | 2.772 ± 0.105 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 4.221 ± 0.012 | - | 4.239 ± 0.029 | 18.565 ± 0.190 | 7.550 ± 0.058 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 4.189 ± 0.005 | - | 4.217 ± 0.046 | 4.690 ± 0.053 | 5.110 ± 0.936 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.005 ± 0.000 | - | 0.010 ± 0.000 | 0.019 ± 0.007 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.178 ± 0.028 | - | 0.269 ± 0.013 | 0.333 ± 0.018 | 0.256 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.007 ± 0.000 | - | 0.014 ± 0.003 | 0.020 ± 0.002 | 0.038 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.151 ± 0.005 | - | 0.229 ± 0.018 | 0.286 ± 0.015 | 0.219 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.014 ± 0.000 | - | 0.023 ± 0.001 | 0.031 ± 0.001 | 0.044 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.007 ± 0.000 | - | 0.014 ± 0.000 | 0.026 ± 0.004 | 0.031 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.014 ± 0.000 | - | 0.023 ± 0.001 | 0.030 ± 0.002 | 0.045 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.034 ± 0.000 | - | 0.054 ± 0.002 | 0.064 ± 0.003 | 0.076 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.008 ± 0.001 | - | 0.015 ± 0.002 | 0.029 ± 0.008 | 0.033 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.017 ± 0.000 | - | 0.029 ± 0.003 | 0.035 ± 0.002 | 0.048 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.045 ± 0.001 | - | 0.071 ± 0.002 | 0.080 ± 0.004 | 0.087 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 15.774 ± 0.227 | - | 15.603 ± 0.057 | 36.477 ± 0.074 | 61.770 ± 27.061 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.012 ± 0.000 | - | 0.023 ± 0.004 | 0.037 ± 0.005 | 0.052 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 0.592 ± 0.026 | - | 0.604 ± 0.009 | 2.337 ± 0.013 | 0.898 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.031 ± 0.000 | - | 0.050 ± 0.002 | 0.056 ± 0.005 | 0.068 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.087 ± 0.002 | - | 0.131 ± 0.003 | 0.141 ± 0.005 | 0.160 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.968 ± 0.002 | - | 0.975 ± 0.004 | 1.094 ± 0.004 | 1.356 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 3.215 ± 0.007 | - | 3.273 ± 0.004 | 4.356 ± 0.012 | 4.083 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 10.592 ± 0.036 | - | 10.775 ± 0.137 | 13.849 ± 0.023 | 12.296 ± 0.084 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 2.287 ± 0.025 | - | 2.242 ± 0.022 | 2.463 ± 0.021 | 2.450 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.013 ± 0.001 | 0.029 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.133 ± 0.002 | - | 0.136 ± 0.004 | 0.164 ± 0.017 | 0.220 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.006 | 0.034 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.096 ± 0.005 | - | 0.097 ± 0.004 | 0.135 ± 0.005 | 0.159 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.013 ± 0.001 | - | 0.015 ± 0.000 | 0.027 ± 0.001 | 0.038 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.016 ± 0.001 | 0.033 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.028 ± 0.001 | 0.038 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.032 ± 0.000 | - | 0.035 ± 0.000 | 0.052 ± 0.001 | 0.067 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.007 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.015 ± 0.000 | - | 0.018 ± 0.000 | 0.031 ± 0.002 | 0.044 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.042 ± 0.000 | - | 0.045 ± 0.000 | 0.065 ± 0.005 | 0.073 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 18.154 ± 0.087 | - | 17.776 ± 0.174 | 19.529 ± 0.229 | 18.656 ± 0.068 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.020 ± 0.001 | 0.051 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 1.033 ± 0.005 | - | 1.068 ± 0.019 | 1.158 ± 0.018 | 1.175 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.028 ± 0.001 | - | 0.030 ± 0.000 | 0.047 ± 0.000 | 0.057 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.083 ± 0.005 | - | 0.086 ± 0.001 | 0.117 ± 0.004 | 0.165 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.972 ± 0.002 | - | 1.236 ± 0.591 | 1.055 ± 0.020 | 1.307 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 3.163 ± 0.015 | - | 3.172 ± 0.012 | 4.289 ± 0.018 | 2.441 ± 0.053 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 3.865 ± 0.010 | - | 3.863 ± 0.008 | 16.651 ± 0.014 | 6.291 ± 0.135 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 2.218 ± 0.014 | - | 2.219 ± 0.023 | 2.788 ± 0.019 | 3.077 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.001 | 0.032 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.134 ± 0.004 | - | 0.199 ± 0.011 | 0.254 ± 0.008 | 0.208 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | 0.013 ± 0.000 | 0.018 ± 0.001 | 0.037 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.097 ± 0.002 | - | 0.151 ± 0.009 | 0.199 ± 0.006 | 0.192 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.013 ± 0.000 | - | 0.022 ± 0.003 | 0.028 ± 0.001 | 0.039 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.012 ± 0.006 | - | 0.013 ± 0.002 | 0.023 ± 0.005 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.013 ± 0.009 | - | 0.022 ± 0.001 | 0.028 ± 0.001 | 0.046 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.033 ± 0.000 | - | 0.049 ± 0.002 | 0.052 ± 0.001 | 0.078 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.013 ± 0.006 | - | 0.014 ± 0.000 | 0.023 ± 0.003 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.016 ± 0.001 | - | 0.027 ± 0.003 | 0.032 ± 0.008 | 0.043 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.043 ± 0.005 | - | 0.064 ± 0.002 | 0.067 ± 0.003 | 0.076 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 8.124 ± 0.103 | - | 8.228 ± 0.147 | 21.863 ± 0.158 | 25.341 ± 1.730 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.011 ± 0.011 | - | 0.020 ± 0.001 | 0.031 ± 0.003 | 0.043 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 0.417 ± 0.016 | - | 0.434 ± 0.711 | 1.758 ± 0.005 | 0.669 ± 0.126 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.028 ± 0.000 | - | 0.046 ± 0.003 | 0.047 ± 0.000 | 0.058 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.083 ± 0.001 | - | 0.122 ± 0.001 | 0.118 ± 0.003 | 0.154 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.583 ± 0.024 | - | unsupported | 0.353 ± 0.011 | 0.763 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.959 ± 0.003 | - | unsupported | 1.938 ± 0.010 | 2.706 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 5.455 ± 0.045 | - | unsupported | 5.333 ± 0.039 | 6.798 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 1.226 ± 0.013 | - | unsupported | 1.141 ± 0.018 | 1.108 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.009 ± 0.001 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.059 ± 0.005 | - | unsupported | 0.064 ± 0.008 | 0.073 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.012 ± 0.001 | 0.021 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.048 ± 0.007 | - | unsupported | 0.055 ± 0.003 | 0.071 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | unsupported | 0.015 ± 0.001 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.003 ± 0.001 | - | unsupported | 0.011 ± 0.001 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.007 ± 0.000 | - | unsupported | 0.016 ± 0.001 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.017 ± 0.000 | - | unsupported | 0.026 ± 0.002 | 0.055 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.019 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.009 ± 0.000 | - | unsupported | 0.017 ± 0.000 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.023 ± 0.000 | - | unsupported | 0.032 ± 0.001 | 0.043 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 9.806 ± 0.013 | - | unsupported | 8.929 ± 0.116 | 8.621 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | unsupported | 0.012 ± 0.001 | 0.020 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.461 ± 0.011 | - | unsupported | 0.422 ± 0.013 | 0.494 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.016 ± 0.000 | - | unsupported | 0.025 ± 0.003 | 0.038 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.043 ± 0.000 | - | unsupported | 0.051 ± 0.001 | 0.069 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.583 ± 0.009 | - | unsupported | 0.584 ± 0.014 | 0.753 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 1.942 ± 0.002 | - | unsupported | 1.928 ± 0.018 | 1.436 ± 0.095 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 1.937 ± 0.007 | - | unsupported | 7.672 ± 0.007 | 3.524 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.428 ± 0.244 | - | unsupported | 1.445 ± 0.014 | 1.434 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | unsupported | 0.013 ± 0.005 | 0.022 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.077 ± 0.002 | - | unsupported | 0.089 ± 0.020 | 0.080 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | unsupported | 0.016 ± 0.001 | 0.026 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.068 ± 0.002 | - | unsupported | 0.079 ± 0.004 | 0.090 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.008 ± 0.000 | - | unsupported | 0.015 ± 0.001 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | unsupported | 0.015 ± 0.003 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.008 ± 0.000 | - | unsupported | 0.023 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.018 ± 0.000 | - | unsupported | 0.026 ± 0.000 | 0.040 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | unsupported | 0.015 ± 0.002 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.009 ± 0.000 | - | unsupported | 0.025 ± 0.001 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.024 ± 0.000 | - | unsupported | 0.032 ± 0.001 | 0.054 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 3.202 ± 0.018 | - | unsupported | 11.193 ± 0.213 | 11.482 ± 3.514 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | unsupported | 0.018 ± 0.004 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.168 ± 0.003 | - | unsupported | 0.425 ± 0.013 | 0.260 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.017 ± 0.000 | - | unsupported | 0.025 ± 0.001 | 0.049 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.044 ± 0.000 | - | unsupported | 0.053 ± 0.003 | 0.071 ± 0.003 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 1.424 ± 0.014 | - | unsupported | 0.953 ± 0.004 | 3.388 ± 0.067 | 2.480 ± 0.092 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.108 ± 0.003 | - | unsupported | 0.059 ± 0.004 | 0.125 ± 0.021 | 0.092 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.199 ± 0.003 | - | unsupported | 0.084 ± 0.003 | 0.184 ± 0.021 | 0.171 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.498 ± 0.020 | - | unsupported | 0.243 ± 0.002 | 0.493 ± 0.013 | 0.667 ± 0.016 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.681 ± 0.014 | - | unsupported | 0.486 ± 0.012 | 1.139 ± 0.105 | 1.879 ± 0.755 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.194 ± 0.014 | - | unsupported | 0.063 ± 0.005 | 0.111 ± 0.004 | 0.093 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.327 ± 0.013 | - | unsupported | 0.080 ± 0.004 | 0.187 ± 0.019 | 0.172 ± 0.006 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.233 ± 0.017 | - | unsupported | 0.160 ± 0.004 | 0.545 ± 0.078 | 0.693 ± 0.042 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.558 ± 0.009 | - | unsupported | 0.375 ± 0.014 | 0.827 ± 0.030 | 0.415 ± 0.016 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.055 ± 0.001 | - | unsupported | 0.045 ± 0.001 | 0.188 ± 0.017 | 0.072 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.067 ± 0.001 | - | unsupported | 0.061 ± 0.001 | 0.224 ± 0.012 | 0.110 ± 0.005 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.210 ± 0.005 | - | unsupported | 0.167 ± 0.004 | 0.376 ± 0.028 | 0.204 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.320 ± 0.013 | - | unsupported | 0.560 ± 0.014 | 0.820 ± 0.070 | 0.417 ± 0.008 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.097 ± 0.002 | - | unsupported | 0.075 ± 0.004 | 0.307 ± 0.016 | 0.066 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.120 ± 0.009 | - | unsupported | 0.098 ± 0.003 | 0.237 ± 0.013 | 0.109 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.358 ± 0.013 | - | unsupported | 0.225 ± 0.003 | 0.405 ± 0.017 | 0.202 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.305 ± 0.020 | - | 0.295 ± 0.020 | 0.194 ± 0.012 | 0.405 ± 0.020 | 0.316 ± 0.021 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.039 ± 0.000 | - | 0.041 ± 0.000 | 0.073 ± 0.013 | 0.079 ± 0.008 | 0.060 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.047 ± 0.000 | - | 0.050 ± 0.000 | 0.088 ± 0.004 | 0.105 ± 0.006 | 0.087 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.126 ± 0.001 | - | 0.126 ± 0.000 | 0.123 ± 0.013 | 0.179 ± 0.009 | 0.141 ± 0.005 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.141 ± 0.001 | - | 0.145 ± 0.018 | 0.257 ± 0.012 | 0.450 ± 0.028 | 0.248 ± 0.008 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.067 ± 0.002 | - | 0.072 ± 0.002 | 0.127 ± 0.015 | 0.083 ± 0.009 | 0.051 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.047 ± 0.000 | - | 0.086 ± 0.003 | 0.142 ± 0.010 | 0.104 ± 0.006 | 0.078 ± 0.007 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.127 ± 0.002 | - | 0.212 ± 0.017 | 0.180 ± 0.014 | 0.224 ± 0.027 | 0.131 ± 0.001 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 39.717 ± 0.228 | - | 44.988 ± 0.440 | 63.139 ± 0.672 | 70.099 ± 0.563 | 52.082 ± 18.234 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 24.314 ± 0.308 | - | 22.388 ± 0.294 | 25.091 ± 0.470 | 65.043 ± 14.991 | 37.767 ± 4.203 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 22.207 ± 0.193 | - | 21.000 ± 0.766 | 28.292 ± 0.733 | 22.129 ± 0.173 | 21.159 ± 5.842 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 17.233 ± 0.087 | - | 12.562 ± 0.231 | 11.430 ± 0.223 | 14.972 ± 4.997 | 12.574 ± 0.183 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 7.188 ± 0.301 | - | 7.200 ± 0.026 | 7.926 ± 0.033 | 7.972 ± 0.028 | 8.074 ± 0.173 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 9.935 ± 0.028 | - | 9.925 ± 0.026 | 10.518 ± 0.142 | 11.600 ± 0.106 | 11.811 ± 0.243 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 8.776 ± 0.099 | - | 8.809 ± 0.022 | 8.862 ± 0.134 | 9.602 ± 0.066 | 9.127 ± 0.021 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 12.674 ± 0.025 | - | 12.678 ± 0.023 | 11.801 ± 0.021 | 14.220 ± 0.114 | 13.762 ± 0.095 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 19.168 ± 0.128 | - | 19.470 ± 0.263 | 19.216 ± 0.077 | 38.280 ± 0.104 | 13.784 ± 0.181 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 19.377 ± 0.124 | - | 19.672 ± 0.140 | 15.015 ± 0.251 | 32.770 ± 1.997 | 12.333 ± 0.921 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.330 ± 0.003 | 0.348 ± 0.008 | unsupported | 0.618 ± 0.010 | 0.765 ± 0.030 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 6.360 ± 0.039 | 6.466 ± 0.022 | unsupported | 8.891 ± 0.068 | 9.779 ± 0.114 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.358 ± 0.008 | 0.378 ± 0.013 | unsupported | 0.512 ± 0.006 | 1.496 ± 0.824 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 4.064 ± 0.071 | 4.177 ± 0.802 | unsupported | 5.694 ± 0.066 | 9.613 ± 0.141 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 32.406 ± 0.305 | - | 33.347 ± 0.090 | 43.617 ± 1.339 | 32.070 ± 0.109 | 27.667 ± 0.170 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 18.999 ± 0.102 | - | 18.234 ± 0.371 | 21.768 ± 0.094 | 19.964 ± 0.215 | 16.595 ± 0.093 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 15.804 ± 0.133 | 15.511 ± 0.113 | 9.289 ± 0.082 | 55.875 ± 0.344 | 9.622 ± 0.081 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 10.259 ± 0.206 | 10.113 ± 0.146 | 6.345 ± 0.210 | 53.377 ± 1.379 | 8.930 ± 0.129 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 22.418 ± 0.722 | - | 22.424 ± 0.318 | 32.929 ± 1.102 | 32.434 ± 0.152 | 27.617 ± 6.598 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 17.181 ± 0.227 | - | 17.200 ± 0.059 | 15.750 ± 3.254 | 24.875 ± 1.198 | 26.635 ± 10.657 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 1.199 ± 0.087 | - | 1.117 ± 0.175 | 1.473 ± 0.040 | 3.938 ± 0.187 | 2.712 ± 0.028 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.516 ± 0.068 | - | 0.487 ± 0.047 | 1.317 ± 0.085 | 14.696 ± 3.117 | 2.689 ± 0.022 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 16.474 ± 0.281 | - | 17.615 ± 0.672 | 21.067 ± 0.306 | 20.181 ± 0.331 | 19.993 ± 1.641 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 18.719 ± 0.094 | - | 18.428 ± 0.092 | 16.707 ± 0.177 | 20.301 ± 1.124 | 19.651 ± 2.959 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 16.703 ± 0.350 | - | 17.114 ± 0.147 | 21.138 ± 0.379 | 20.494 ± 0.202 | 19.274 ± 0.608 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 18.957 ± 0.378 | - | 18.836 ± 0.394 | 16.463 ± 0.226 | 21.125 ± 1.463 | 19.930 ± 0.545 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 0.997 ± 0.023 | 1.143 ± 0.014 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 23.201 ± 1.367 | 20.831 ± 0.192 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 0.995 ± 0.013 | 1.091 ± 0.011 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 13.668 ± 0.866 | 13.890 ± 0.831 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.017 ± 0.013 | 1.120 ± 0.005 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 22.182 ± 0.200 | 21.559 ± 0.608 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 0.983 ± 0.015 | 1.106 ± 0.010 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 13.322 ± 0.137 | 14.162 ± 0.824 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 22.215 ± 0.296 | - | 20.901 ± 0.436 | 27.812 ± 0.463 | 22.108 ± 0.171 | 21.182 ± 0.485 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 17.185 ± 0.095 | - | 12.614 ± 0.436 | 11.338 ± 0.200 | 13.935 ± 0.203 | 12.586 ± 1.479 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 55.165 ± 0.262 | 54.566 ± 0.989 | 62.804 ± 0.535 | 63.151 ± 0.566 | 63.227 ± 1.557 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 55.743 ± 0.549 | 55.294 ± 0.985 | 38.498 ± 0.734 | 51.986 ± 1.435 | 54.484 ± 7.869 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 36.806 ± 0.570 | - | 36.856 ± 0.086 | 30.565 ± 0.208 | 130.869 ± 3.755 | 29.550 ± 0.333 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 18.792 ± 0.061 | - | 18.425 ± 0.270 | 27.188 ± 6.783 | 118.804 ± 12.179 | 18.733 ± 0.414 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.139 ± 0.001 | 0.140 ± 0.002 | unsupported | 0.162 ± 0.030 | 0.145 ± 0.011 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.224 ± 0.012 | 2.214 ± 0.009 | unsupported | 2.476 ± 0.015 | 0.409 ± 0.005 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.081 ± 0.000 | 0.083 ± 0.001 | unsupported | 0.077 ± 0.002 | 0.130 ± 0.017 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.909 ± 0.002 | 0.910 ± 0.007 | unsupported | 0.754 ± 0.059 | 0.233 ± 0.348 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.078 ± 0.000 | 0.079 ± 0.001 | unsupported | 0.136 ± 0.013 | 0.254 ± 0.003 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.257 ± 0.003 | 1.253 ± 0.006 | unsupported | 1.753 ± 0.026 | 2.623 ± 0.020 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.047 ± 0.000 | 0.047 ± 0.000 | unsupported | 0.099 ± 0.002 | 0.157 ± 0.009 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.568 ± 0.002 | 0.570 ± 0.007 | unsupported | 0.727 ± 0.014 | 1.201 ± 0.219 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.196 ± 0.000 | 0.197 ± 0.001 | unsupported | 0.284 ± 0.002 | 0.188 ± 0.003 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 3.158 ± 0.006 | 3.125 ± 0.046 | unsupported | 4.443 ± 0.018 | 1.849 ± 0.070 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.086 ± 0.001 | 0.086 ± 0.000 | unsupported | 0.107 ± 0.009 | 0.147 ± 0.002 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 1.186 ± 0.001 | 1.188 ± 0.006 | unsupported | 1.500 ± 0.015 | 0.990 ± 0.005 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.058 ± 0.006 | 0.077 ± 0.009 | unsupported | 0.161 ± 0.002 | 0.198 ± 0.003 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.614 ± 0.010 | 0.690 ± 0.054 | unsupported | 0.912 ± 0.029 | 1.857 ± 0.054 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.135 ± 0.004 | 0.170 ± 0.004 | unsupported | 0.068 ± 0.002 | 0.181 ± 0.009 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.452 ± 0.014 | 0.524 ± 0.031 | unsupported | 0.362 ± 0.005 | 1.197 ± 0.143 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.129 ± 0.006 | 0.146 ± 0.003 | unsupported | 0.136 ± 0.001 | 0.240 ± 0.002 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.691 ± 0.010 | 1.802 ± 0.106 | unsupported | 1.231 ± 0.056 | 2.351 ± 0.052 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.200 ± 0.002 | 0.231 ± 0.003 | unsupported | 0.062 ± 0.001 | 0.207 ± 0.003 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.955 ± 0.409 | 1.017 ± 0.560 | unsupported | 0.488 ± 0.008 | 1.374 ± 0.224 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.300 ± 0.004 | 0.316 ± 0.004 | unsupported | unsupported | 0.258 ± 0.002 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 4.382 ± 0.121 | 4.413 ± 0.012 | unsupported | unsupported | 3.140 ± 0.121 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.356 ± 0.013 | 0.397 ± 0.056 | unsupported | unsupported | 0.242 ± 0.002 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 2.039 ± 0.046 | 2.093 ± 0.016 | unsupported | unsupported | 1.823 ± 0.074 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.012 ± 0.000 | 0.016 ± 0.001 | unsupported | 0.013 ± 0.000 | 0.052 ± 0.014 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.096 ± 0.017 | 0.104 ± 0.005 | unsupported | 0.090 ± 0.009 | 0.252 ± 0.007 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.027 ± 0.000 | 0.038 ± 0.020 | unsupported | 0.015 ± 0.000 | 0.038 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.128 ± 0.036 | 0.099 ± 0.020 | unsupported | 0.076 ± 0.028 | 0.212 ± 0.005 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.031 ± 0.005 | 0.043 ± 0.005 | unsupported | 0.103 ± 0.004 | 0.187 ± 0.002 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.341 ± 0.017 | 0.410 ± 0.012 | unsupported | 0.589 ± 0.023 | 1.681 ± 0.032 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.068 ± 0.002 | 0.091 ± 0.002 | unsupported | 0.099 ± 0.002 | 0.144 ± 0.006 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.245 ± 0.036 | 0.297 ± 0.028 | unsupported | 0.248 ± 0.031 | 1.035 ± 0.122 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.054 ± 0.001 | 0.055 ± 0.000 | unsupported | 0.118 ± 0.001 | 0.196 ± 0.007 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.865 ± 0.004 | 0.874 ± 0.005 | unsupported | 1.818 ± 0.044 | 1.724 ± 0.011 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.036 ± 0.001 | 0.037 ± 0.001 | unsupported | 0.096 ± 0.001 | 0.134 ± 0.003 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.398 ± 0.003 | 0.402 ± 0.003 | unsupported | 0.680 ± 0.011 | 0.950 ± 0.028 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.056 ± 0.001 | 0.057 ± 0.002 | unsupported | 0.120 ± 0.002 | 0.196 ± 0.006 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.899 ± 0.004 | 0.916 ± 0.013 | unsupported | 1.857 ± 0.028 | 1.731 ± 0.079 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.037 ± 0.000 | 0.038 ± 0.003 | unsupported | 0.100 ± 0.002 | 0.139 ± 0.003 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.413 ± 0.004 | 0.418 ± 0.124 | unsupported | 0.721 ± 0.006 | 0.977 ± 0.294 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.129 ± 0.005 | 0.146 ± 0.006 | unsupported | 0.140 ± 0.001 | 0.237 ± 0.001 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.675 ± 0.066 | 1.786 ± 0.075 | unsupported | 1.239 ± 0.023 | 2.210 ± 0.015 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.200 ± 0.002 | 0.231 ± 0.004 | unsupported | 0.062 ± 0.001 | 0.203 ± 0.005 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 1.013 ± 0.061 | 1.051 ± 0.041 | unsupported | 0.469 ± 0.003 | 1.300 ± 0.017 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.183 ± 0.000 | 0.184 ± 0.001 | unsupported | 0.274 ± 0.002 | 0.341 ± 0.003 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 2.929 ± 0.003 | 2.910 ± 0.015 | unsupported | 4.299 ± 0.004 | 3.962 ± 0.037 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.082 ± 0.001 | 0.083 ± 0.017 | unsupported | 0.201 ± 0.003 | 0.203 ± 0.005 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 1.135 ± 0.006 | 1.144 ± 0.032 | unsupported | 1.554 ± 0.043 | 2.081 ± 0.279 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.744 ± 0.012 | 1.315 ± 0.026 | unsupported | 0.266 ± 0.003 | 0.238 ± 0.031 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 18.664 ± 0.419 | 28.941 ± 0.266 | unsupported | 5.576 ± 0.040 | 2.297 ± 0.025 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 1.006 ± 0.024 | 1.750 ± 0.132 | unsupported | 0.296 ± 0.017 | 0.166 ± 0.002 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 9.333 ± 0.628 | 21.629 ± 1.476 | unsupported | 2.314 ± 0.074 | 1.212 ± 0.034 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 28.343 ± 0.424 | - | unsupported | 26.592 ± 0.481 | - | 31.964 ± 0.299 | 38.413 ± 0.208 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 21.857 ± 0.150 | - | unsupported | 21.896 ± 1.560 | - | 32.357 ± 0.397 | 22.240 ± 0.268 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 22.154 ± 0.282 | - | unsupported | 27.095 ± 0.286 | - | 22.476 ± 0.255 | 24.046 ± 0.988 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 15.050 ± 0.164 | - | unsupported | 15.334 ± 0.358 | - | 21.602 ± 0.532 | 15.139 ± 0.277 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 14.478 ± 0.203 | - | unsupported | 23.359 ± 0.274 | - | 14.823 ± 0.470 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 15.096 ± 0.193 | - | unsupported | 15.069 ± 0.170 | - | 14.048 ± 0.188 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 27.811 ± 0.180 | - | unsupported | 28.826 ± 0.579 | - | 31.891 ± 0.091 | 37.519 ± 0.575 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 21.600 ± 1.070 | - | unsupported | 21.383 ± 0.093 | - | 32.272 ± 0.105 | 21.776 ± 0.499 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 48.367 ± 0.338 | - | unsupported | 50.130 ± 0.170 | - | 48.453 ± 0.162 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 22.924 ± 0.098 | - | unsupported | 25.221 ± 0.120 | - | 23.598 ± 0.517 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 48.694 ± 0.484 | - | unsupported | 49.916 ± 0.596 | - | 48.202 ± 0.557 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 23.000 ± 0.090 | - | unsupported | 25.177 ± 0.657 | - | 52.011 ± 20.197 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 28.026 ± 0.155 | - | unsupported | 27.571 ± 1.304 | - | 32.249 ± 0.348 | 39.364 ± 0.750 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 21.361 ± 0.878 | - | unsupported | 21.730 ± 0.202 | - | 32.476 ± 0.253 | 23.331 ± 0.630 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 21.261 ± 0.122 | - | unsupported | 23.077 ± 0.153 | - | 25.893 ± 0.124 | 27.416 ± 0.337 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 15.642 ± 0.518 | - | unsupported | 14.873 ± 0.182 | - | 26.567 ± 0.170 | 15.748 ± 2.430 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.549 ± 0.002 | unsupported | unsupported | 0.909 ± 0.006 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 9.133 ± 0.089 | unsupported | unsupported | 12.990 ± 0.048 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.623 ± 0.003 | unsupported | unsupported | 0.613 ± 0.002 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 6.573 ± 0.052 | unsupported | unsupported | 6.770 ± 0.193 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 28.060 ± 0.209 | - | unsupported | 26.766 ± 0.202 | - | 31.811 ± 0.189 | 38.516 ± 0.524 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 22.098 ± 0.578 | - | unsupported | 22.366 ± 3.253 | - | 32.627 ± 0.176 | 21.768 ± 0.882 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 110.583 ± 3.385 | - | 0.002 ± 0.000 | 108.415 ± 0.486 | 110.094 ± 4.395 | 136.707 ± 5.013 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 44.100 ± 0.413 | - | 0.003 ± 0.000 | 35.771 ± 1.257 | 56.193 ± 0.987 | 135.937 ± 4.676 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 66.528 ± 1.636 | - | 67.888 ± 2.077 | 67.994 ± 1.653 | 69.961 ± 2.750 | 77.169 ± 1.901 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 23.287 ± 0.389 | - | 22.824 ± 0.286 | 23.007 ± 0.195 | 35.900 ± 0.289 | 77.142 ± 1.571 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 218.623 ± 3.783 | - | 219.168 ± 3.867 | 223.312 ± 1.616 | 224.983 ± 2.476 | 253.788 ± 4.847 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 72.936 ± 3.017 | - | 71.587 ± 0.487 | 70.880 ± 0.673 | 113.754 ± 1.584 | 255.870 ± 10.513 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 69.007 ± 0.211 | - | 68.448 ± 2.012 | 70.544 ± 1.822 | 69.694 ± 0.992 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 42.828 ± 0.922 | - | 42.676 ± 0.682 | 24.344 ± 0.270 | 36.291 ± 0.418 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 136.394 ± 3.624 | - | 0.002 ± 0.000 | 122.006 ± 1.895 | 121.570 ± 1.911 | 155.587 ± 1.178 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 41.250 ± 0.740 | - | 0.003 ± 0.000 | 40.631 ± 0.233 | 51.471 ± 1.142 | 154.626 ± 2.774 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 84.234 ± 1.461 | - | 0.002 ± 0.000 | 78.771 ± 1.074 | 84.386 ± 0.844 | 214.629 ± 9.049 | 130.135 ± 2.593 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 29.993 ± 0.512 | - | 0.003 ± 0.000 | 77.544 ± 1.094 | 43.204 ± 0.072 | 213.890 ± 3.784 | 51.975 ± 1.460 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 60.337 ± 1.197 | - | 60.689 ± 0.588 | 65.409 ± 1.109 | 65.352 ± 1.957 | 80.892 ± 1.877 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 21.246 ± 0.215 | - | 21.148 ± 0.101 | 23.912 ± 0.850 | 33.835 ± 0.338 | 81.523 ± 1.576 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 59.630 ± 0.811 | - | 60.270 ± 1.632 | 63.151 ± 0.688 | 64.841 ± 1.549 | 82.124 ± 1.972 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 20.718 ± 0.232 | - | 20.481 ± 0.555 | 23.169 ± 0.386 | 34.297 ± 1.020 | 82.735 ± 1.455 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 7.868894 | 491805.88 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 7.343584 | 458974.00 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 7.642462 | 477653.88 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 5.951193 | 371949.56 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 7.370780 | 460673.75 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 7.904045 | 494002.81 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 17.750645 | 1109415.31 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 17.552826 | 1097051.62 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 6.020531 | 376283.19 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 6.120417 | 382526.06 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 12.235373 | 764710.81 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 9.895755 | 618484.69 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 5.274784 | 329674.00 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 5.568251 | 348015.69 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 156.458329 | 9778645.56 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 142.250758 | 8890672.38 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 101.765703 | 6360356.44 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 103.460476 | 6466279.75 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 23.929387 | 1495586.69 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 8.198246 | 512390.37 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 5.728779 | 358048.69 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 6.050779 | 378173.69 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 153.811986 | 9613249.12 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 91.102818 | 5693926.12 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 65.020725 | 4063795.31 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 66.833367 | 4177085.44 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 15.949945 | 996871.56 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 18.286687 | 1142917.94 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 371.213090 | 23200818.12 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 333.291619 | 20830726.19 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 15.923830 | 995239.38 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 17.459486 | 1091217.88 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 218.683028 | 13667689.25 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 222.243641 | 13890227.56 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 16.272147 | 1017009.19 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 17.923438 | 1120214.88 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 354.918590 | 22182411.88 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 344.951699 | 21559481.19 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 15.730870 | 983179.38 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 17.691900 | 1105743.75 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 213.146437 | 13321652.31 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 226.596672 | 14162292.00 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.283873 | 145060.52 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 10.349571 | 161712.05 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.908635 | 139197.42 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.966143 | 140095.98 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 3.271685 | 408960.62 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 19.808561 | 2476070.12 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 17.789569 | 2223696.12 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 17.712976 | 2214122.00 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.326406 | 130100.09 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.925229 | 76956.70 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.208087 | 81376.36 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.295670 | 82744.84 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.867224 | 233403.00 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.029220 | 753652.50 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.275685 | 909460.62 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.277148 | 909643.50 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 16.236326 | 253692.59 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.727014 | 136359.59 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.983951 | 77874.23 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.035768 | 78683.88 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 20.983143 | 2622892.88 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.024237 | 1753029.62 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.058792 | 1257349.00 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.021944 | 1252743.00 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.059929 | 157186.39 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.306697 | 98542.14 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.977085 | 46516.95 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3.035765 | 47433.83 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.604251 | 1200531.38 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.812264 | 726533.00 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.544439 | 568054.88 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.556031 | 569503.88 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.057469 | 188397.95 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 18.171636 | 283931.81 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 12.571541 | 196430.33 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 12.600005 | 196875.08 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.793374 | 1849171.75 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 35.542149 | 4442768.62 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 25.264359 | 3158044.88 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 25.001649 | 3125206.12 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.376645 | 146510.08 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.877312 | 107458.00 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.486587 | 85727.92 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.525489 | 86335.77 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.921890 | 990236.25 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.001682 | 1500210.25 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.490277 | 1186284.62 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 9.500827 | 1187603.38 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.700589 | 198446.70 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 10.284279 | 160691.86 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.717228 | 58081.69 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.939348 | 77177.31 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.859557 | 1857444.62 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.298899 | 912362.38 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.908510 | 613563.75 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.516977 | 689622.12 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.605863 | 181341.61 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.353433 | 68022.39 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.649248 | 135144.50 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 10.859932 | 169686.44 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.576467 | 1197058.38 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.898638 | 362329.75 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.613282 | 451660.25 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.195468 | 524433.50 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 15.350093 | 239845.20 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.686147 | 135721.05 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.262350 | 129099.22 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.315846 | 145560.09 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.809538 | 2351192.25 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.844619 | 1230577.38 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.525973 | 1690746.62 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 14.419230 | 1802403.75 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.258868 | 207169.81 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.968855 | 62013.36 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 12.800994 | 200015.53 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 14.779776 | 230934.00 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.989784 | 1373723.00 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.904545 | 488068.12 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.639735 | 954966.88 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 8.134327 | 1016790.88 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 16.542567 | 258477.61 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 19.177279 | 299644.98 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 20.246725 | 316355.08 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 25.124000 | 3140500.00 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 35.053070 | 4381633.75 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 35.306343 | 4413292.88 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 15.496323 | 242130.05 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 22.807114 | 356361.16 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 25.437511 | 397461.11 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 14.581225 | 1822653.12 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 16.309580 | 2038697.50 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 16.741025 | 2092628.12 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 3.305804 | 51653.19 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.822624 | 12853.50 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.798701 | 12479.70 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.003633 | 15681.77 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 2.016993 | 252124.12 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.721687 | 90210.88 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.765379 | 95672.38 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.828316 | 103539.50 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.413012 | 37703.31 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.945223 | 14769.11 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.759283 | 27488.80 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.401251 | 37519.55 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.699170 | 212396.25 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.611030 | 76378.75 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.021396 | 127674.50 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 0.791527 | 98940.88 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.944528 | 186633.25 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.566423 | 102600.36 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.978682 | 30916.91 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.735374 | 42740.22 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.446881 | 1680860.12 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.713585 | 589198.12 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.729645 | 341205.62 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.279029 | 409878.62 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.189014 | 143578.34 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.323008 | 98797.00 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.327093 | 67610.83 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.817474 | 90898.03 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.283725 | 1035465.62 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.984162 | 248020.25 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.956621 | 244577.62 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.373959 | 296744.88 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.539428 | 195928.56 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.534389 | 117724.83 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.468383 | 54193.48 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.505522 | 54773.78 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.791875 | 1723984.38 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.545681 | 1818210.12 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.922019 | 865252.38 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.988442 | 873555.25 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.592251 | 134253.92 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.131741 | 95808.45 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.320189 | 36252.95 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.342922 | 36608.16 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.599198 | 949899.75 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.438367 | 679795.88 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.186735 | 398341.88 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.214006 | 401750.75 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.571618 | 196431.53 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.711188 | 120487.31 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.604206 | 56315.72 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.669758 | 57339.97 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.845435 | 1730679.38 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.859426 | 1857428.25 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.190579 | 898822.38 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.329758 | 916219.75 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.878506 | 138726.66 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.378010 | 99656.41 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.368369 | 37005.77 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.421658 | 37838.41 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.816193 | 977024.12 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.766218 | 720777.25 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.307722 | 413465.25 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.347276 | 418409.50 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 15.188681 | 237323.14 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.962533 | 140039.58 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.224679 | 128510.61 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.322839 | 145669.36 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 17.681643 | 2210205.38 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.909871 | 1238733.88 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.401140 | 1675142.50 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 14.287895 | 1785986.88 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.992801 | 203012.52 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.944099 | 61626.55 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 12.816703 | 200260.98 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 14.795816 | 231184.62 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.398160 | 1299770.00 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.752000 | 469000.00 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.106716 | 1013339.50 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 8.406104 | 1050763.00 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 21.841015 | 341265.86 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.525871 | 273841.73 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 11.689395 | 182646.80 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 11.749568 | 183587.00 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 31.695253 | 3961906.62 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 34.389187 | 4298648.38 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 23.428393 | 2928549.12 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 23.282942 | 2910367.75 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.003131 | 203173.92 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 12.887876 | 201373.06 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.255946 | 82124.16 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.304668 | 82885.44 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 16.650977 | 2081372.13 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.432627 | 1554078.38 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.077888 | 1134736.00 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 9.150543 | 1143817.88 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 15.247897 | 238248.39 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.010800 | 265793.75 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 47.586377 | 743537.14 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 84.173999 | 1315218.73 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.373545 | 2296693.12 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 44.608475 | 5576059.38 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 149.313087 | 18664135.88 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 231.531854 | 28941481.75 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.614274 | 165848.03 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 18.954665 | 296166.64 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 64.391401 | 1006115.64 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 112.007393 | 1750115.52 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.693625 | 1211703.12 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 18.515976 | 2314497.00 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 74.665451 | 9333181.38 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 173.028045 | 21628505.62 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 14.544960 | 909060.00 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 8.785991 | 549124.44 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 207.832624 | 12989539.00 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 146.132622 | 9133288.88 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 9.803611 | 612725.69 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 9.971084 | 623192.75 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 108.323628 | 6770226.75 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 105.167792 | 6572987.00 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.024075 | 1504.69 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002434 | 152.12 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.023634 | 1477.12 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002435 | 152.19 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000120 | 7.50 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.022262 | 1391.38 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.007474 | 467.12 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000121 | 7.56 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.022152 | 1384.50 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.007824 | 489.00 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000110 | 6.88 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.023033 | 1439.56 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.005840 | 365.00 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000190 | 11.88 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.022622 | 1413.88 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.005991 | 374.44 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000110 | 6.88 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.020928 | 1308.00 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003507 | 219.19 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000101 | 6.31 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.020748 | 1296.75 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003286 | 205.38 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/norm_fro` (c64, threads=1, shape=`2048x1536`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 24.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 11.285 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 16.7x faster than the slowest successful cell (`pytorch-cpu`, 24.434 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 13.2x faster than the slowest successful cell (`pytorch-cpu`, 24.434 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 12.8x faster than the slowest successful cell (`pytorch-cpu`, 24.434 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 14.3x faster than the slowest successful cell (`pytorch-cpu`, 11.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 13.9x faster than the slowest successful cell (`pytorch-cpu`, 11.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 18.9x faster than the slowest successful cell (`pytorch-cpu`, 23.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 23.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 23.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 10.421 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 10.421 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `pytorch-cpu` is 19.8x faster than the slowest successful cell (`julia-base`, 7.472 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `tenferro-direct` is 23.5x faster than the slowest successful cell (`julia-base`, 7.472 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `tenferro-trace` is 21.9x faster than the slowest successful cell (`julia-base`, 7.472 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `jax-cpu` is 10.8x faster than the slowest successful cell (`julia-base`, 7.786 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `pytorch-cpu` is 20.3x faster than the slowest successful cell (`julia-base`, 7.786 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `tenferro-direct` is 17.4x faster than the slowest successful cell (`julia-base`, 7.786 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `tenferro-trace` is 21.2x faster than the slowest successful cell (`julia-base`, 7.786 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`julia-base`, 6.707 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 11.2x faster than the slowest successful cell (`julia-base`, 6.707 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `jax-cpu` is 14.3x faster than the slowest successful cell (`julia-base`, 8.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 21.6x faster than the slowest successful cell (`julia-base`, 8.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 26.3x faster than the slowest successful cell (`julia-base`, 8.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-trace` is 24.6x faster than the slowest successful cell (`julia-base`, 8.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 5083.4x faster than the slowest successful cell (`julia-base`, 9.221 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `pytorch-cpu` is 11.2x faster than the slowest successful cell (`julia-base`, 9.223 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-direct` is 13.6x faster than the slowest successful cell (`julia-base`, 9.223 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 3323.5x faster than the slowest successful cell (`julia-base`, 9.223 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 12.4x faster than the slowest successful cell (`jax-cpu`, 0.027 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.024 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=1, shape=`1x2x2`): `tenferro-direct` is 11.0x faster than the slowest successful cell (`jax-cpu`, 0.026 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (c64, threads=1, shape=`4x2x2`): `tenferro-direct` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 11.2x faster than the slowest successful cell (`jax-cpu`, 14.696 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 28.5x faster than the slowest successful cell (`jax-cpu`, 14.696 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 30.2x faster than the slowest successful cell (`jax-cpu`, 14.696 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 12.6x faster than the slowest successful cell (`tenferro-eager`, 28.941 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 10.6x faster than the slowest successful cell (`tenferro-eager`, 1.750 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 17.8x faster than the slowest successful cell (`tenferro-eager`, 21.629 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 74136.1x faster than the slowest successful cell (`julia-base`, 136.707 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 50817.4x faster than the slowest successful cell (`julia-base`, 135.937 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 87753.6x faster than the slowest successful cell (`julia-base`, 155.587 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 56330.2x faster than the slowest successful cell (`julia-base`, 154.626 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 126029.7x faster than the slowest successful cell (`julia-base`, 214.629 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 85419.2x faster than the slowest successful cell (`julia-base`, 213.890 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 173.9x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 173.1x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 205.7x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 117.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 186.9x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 216.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 9244.2 GB/s at 0.002 ms. The conservative physical lower bound is 0.168 ms, so the cell is 92.4x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 6042.9 GB/s at 0.003 ms. The conservative physical lower bound is 0.042 ms, so the cell is 15.1x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 145572.4 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1455.7x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 100349.7 GB/s at 0.003 ms. The conservative physical lower bound is 0.671 ms, so the cell is 250.9x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 151401.8 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1514.0x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 97790.7 GB/s at 0.003 ms. The conservative physical lower bound is 0.671 ms, so the cell is 244.5x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 78812.5 GB/s at 0.002 ms. The conservative physical lower bound is 1.342 ms, so the cell is 788.1x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 53601.3 GB/s at 0.003 ms. The conservative physical lower bound is 0.336 ms, so the cell is 134.0x faster than the bound.

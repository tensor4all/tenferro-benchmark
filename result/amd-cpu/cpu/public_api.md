# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/amd-cpu/cpu/public_api/20261116_201021/run.yaml`
- Timestamp: `2026-11-16T20:10:21.965060+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/amd-cpu/cpu/public_api/20261116_201021`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

## CPU Information

- Model: `Intel(R) Xeon(R) 6975P-C`
- Vendor: `GenuineIntel`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1016-aws-x86_64-with-glibc2.39`
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

- CSV: `data/results/amd-cpu/cpu/public_api/20261116_201021/cpu_public_api_t1_20261116_201021.csv`
- CSV: `data/results/amd-cpu/cpu/public_api/20261116_201021/cpu_public_api_t4_20261116_201021.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 1.826 ± 0.027 | - | 1.919 ± 0.069 | 2.445 ± 0.040 | 3.210 ± 0.106 | 3.036 ± 0.036 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.026 ± 0.036 | - | 1.929 ± 0.022 | 1.370 ± 0.018 | 2.629 ± 0.469 | 2.327 ± 1.678 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 88.164 ± 0.449 | - | 88.061 ± 0.780 | 86.311 ± 0.478 | 87.137 ± 0.442 | 91.349 ± 0.470 | 102.516 ± 1.758 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 33.398 ± 0.834 | - | 101.975 ± 4.045 | 32.847 ± 0.332 | 48.503 ± 0.769 | 92.855 ± 0.262 | 49.175 ± 10.913 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 128.958 ± 1.630 | - | 129.921 ± 2.041 | 121.480 ± 1.649 | 120.544 ± 2.270 | 125.398 ± 0.969 | 133.501 ± 0.901 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 30.233 ± 0.162 | - | 127.386 ± 2.194 | 24.747 ± 0.855 | 63.870 ± 0.906 | 124.843 ± 1.886 | 42.569 ± 0.901 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 19.603 ± 0.688 | - | 20.122 ± 0.427 | 19.019 ± 0.033 | 58.018 ± 0.321 | 22.733 ± 0.244 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 20.145 ± 0.112 | - | 20.209 ± 0.060 | 9.862 ± 0.095 | 33.259 ± 0.398 | 12.581 ± 0.248 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 20.690 ± 0.392 | - | unsupported | 19.697 ± 0.031 | 64.271 ± 0.492 | 23.645 ± 2.741 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 21.132 ± 0.271 | - | unsupported | 10.147 ± 0.090 | 34.710 ± 1.209 | 15.067 ± 1.827 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 9.092 ± 0.399 | - | 9.058 ± 0.300 | 8.101 ± 0.127 | 8.811 ± 0.179 | 8.645 ± 0.096 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 8.913 ± 0.044 | - | 8.911 ± 0.037 | 7.975 ± 0.189 | 12.994 ± 0.553 | 12.549 ± 0.203 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 40.764 ± 0.463 | - | 41.737 ± 1.630 | 84.808 ± 0.491 | 80.407 ± 0.671 | 63.112 ± 10.668 | 60.694 ± 1.033 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 19.181 ± 0.067 | - | 58.175 ± 1.188 | 34.110 ± 3.069 | 42.704 ± 0.779 | 69.182 ± 9.883 | 27.329 ± 1.131 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 74.634 ± 1.209 | - | 75.088 ± 0.959 | 107.687 ± 2.167 | 156.801 ± 0.462 | 89.186 ± 0.401 | 106.907 ± 10.343 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 31.759 ± 0.209 | - | 92.999 ± 3.790 | 43.175 ± 0.568 | 61.051 ± 0.320 | 92.482 ± 3.100 | 45.569 ± 12.101 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 51.393 ± 0.620 | - | 51.129 ± 1.803 | 51.554 ± 0.983 | 52.992 ± 0.666 | 55.167 ± 1.014 | 57.002 ± 0.184 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 19.939 ± 0.181 | - | 67.401 ± 0.654 | 19.877 ± 0.314 | 29.339 ± 0.529 | 54.172 ± 0.462 | 27.702 ± 0.776 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.785 ± 0.232 | - | 24.327 ± 1.223 | 16.114 ± 0.649 | 11.518 ± 0.559 | 2.400 ± 0.169 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 0.885 ± 0.018 | - | 62.993 ± 6.308 | 5.867 ± 0.105 | 2.684 ± 1.347 | 5.627 ± 1.069 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 3.352 ± 0.135 | - | 3.321 ± 0.042 | 3.717 ± 0.066 | 4.713 ± 0.076 | 5.168 ± 0.100 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 4.059 ± 27.012 | - | 4.123 ± 0.080 | 2.646 ± 0.047 | 6.086 ± 0.194 | 4.405 ± 0.754 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 2.266 ± 0.032 | - | 2.439 ± 0.381 | 2.547 ± 0.058 | 3.675 ± 0.129 | 3.275 ± 0.129 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 324.975 ± 33.009 | - | 337.014 ± 49.014 | 1.823 ± 0.082 | 2.990 ± 0.003 | 2.612 ± 0.067 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 2.888 ± 0.247 | - | 2.766 ± 0.055 | 2.935 ± 0.071 | 3.850 ± 0.017 | 3.785 ± 0.065 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 5.166 ± 0.092 | - | 5.151 ± 0.127 | 3.099 ± 0.672 | 5.762 ± 0.347 | 5.554 ± 0.031 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 19.390 ± 0.089 | - | 21.505 ± 1.071 | 19.031 ± 0.045 | 57.916 ± 0.070 | 21.150 ± 0.110 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 20.131 ± 0.073 | - | 20.142 ± 0.048 | 9.867 ± 0.097 | 33.615 ± 0.323 | 11.592 ± 0.509 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 20.879 ± 0.082 | - | unsupported | 20.270 ± 0.077 | 52.265 ± 0.493 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 22.520 ± 0.202 | - | unsupported | 10.026 ± 0.053 | 26.153 ± 0.174 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 85.391 ± 0.792 | - | 84.630 ± 0.215 | 86.628 ± 0.616 | 95.132 ± 3.275 | 89.847 ± 10.376 | 104.499 ± 1.074 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 34.463 ± 0.181 | - | 103.037 ± 4.089 | 32.689 ± 0.574 | 51.358 ± 0.897 | 93.855 ± 6.386 | 43.843 ± 1.129 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 94.515 ± 0.442 | - | 94.547 ± 0.497 | 100.977 ± 1.496 | 110.563 ± 0.742 | 100.086 ± 7.580 | 112.924 ± 2.237 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 39.227 ± 0.442 | - | 106.184 ± 7.641 | 38.949 ± 0.395 | 59.988 ± 0.422 | 99.758 ± 2.875 | 53.006 ± 0.944 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 36.880 ± 1.695 | - | 44.216 ± 1.363 | 48.455 ± 1.087 | 44.389 ± 0.433 | 48.371 ± 3.106 | 53.671 ± 5.095 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 6.944 ± 0.173 | - | 100.000 ± 3.752 | 19.554 ± 0.517 | 22.311 ± 0.243 | 48.984 ± 5.484 | 21.507 ± 5.616 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 14.306 ± 0.243 | - | 14.434 ± 0.173 | 28.235 ± 0.256 | 32.006 ± 0.499 | 29.785 ± 3.038 | 33.615 ± 2.980 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 7.333 ± 0.236 | - | 31.081 ± 10.114 | 11.368 ± 0.112 | 17.461 ± 0.780 | 33.562 ± 5.405 | 11.933 ± 9.275 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 31.947 ± 0.193 | - | 31.839 ± 0.196 | 51.146 ± 0.296 | 40.182 ± 0.384 | 29.355 ± 0.238 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 16.166 ± 0.085 | - | 47.998 ± 0.004 | 18.955 ± 0.516 | 21.962 ± 0.421 | 28.154 ± 0.880 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 16.140 ± 0.873 | - | 15.852 ± 0.587 | 26.208 ± 0.285 | 62.859 ± 0.477 | 52.145 ± 4.428 | 51.144 ± 3.538 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 5.734 ± 0.046 | - | 34.998 ± 4.070 | 9.373 ± 0.047 | 33.640 ± 0.069 | 50.653 ± 4.888 | 20.362 ± 9.886 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 96.813 ± 0.909 | - | 95.305 ± 0.909 | 97.497 ± 1.240 | 131.746 ± 0.552 | 102.870 ± 0.902 | 115.215 ± 0.991 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 39.466 ± 0.771 | - | 108.450 ± 5.871 | 38.252 ± 0.693 | 68.881 ± 0.293 | 103.345 ± 1.422 | 51.153 ± 0.455 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 11.905 ± 0.735 | - | 12.260 ± 1.014 | 23.660 ± 0.162 | 34.840 ± 1.397 | 46.713 ± 4.905 | 45.361 ± 4.553 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 3.939 ± 0.061 | - | 23.998 ± 0.006 | 8.638 ± 0.290 | 20.533 ± 1.660 | 50.233 ± 7.229 | 19.864 ± 7.448 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 25.406 ± 0.409 | - | 25.307 ± 6.975 | 26.648 ± 0.272 | 24.017 ± 1.851 | 28.209 ± 3.941 | 29.935 ± 5.288 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 4.538 ± 0.493 | - | 45.035 ± 11.300 | 7.466 ± 0.402 | 13.689 ± 0.169 | 26.574 ± 5.261 | 11.108 ± 4.849 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 39.291 ± 0.709 | - | 43.567 ± 1.140 | 25.455 ± 0.225 | 60.423 ± 0.304 | 60.150 ± 5.422 | 81.493 ± 7.359 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 6.087 ± 0.107 | - | 59.996 ± 0.005 | 9.583 ± 0.086 | 32.143 ± 0.507 | 62.576 ± 6.901 | 24.619 ± 8.391 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 9.200 ± 5.741 | - | 21.102 ± 0.562 | 22.652 ± 0.572 | 34.664 ± 0.018 | 41.795 ± 4.478 | 32.992 ± 4.852 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 2.772 ± 0.084 | - | 35.998 ± 0.005 | 6.375 ± 0.154 | 19.895 ± 0.299 | 32.719 ± 5.355 | 13.104 ± 5.152 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.277 ± 0.017 | 0.292 ± 0.021 | unsupported | unsupported | 0.358 ± 0.008 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.216 ± 0.108 | 91.750 ± 0.562 | unsupported | unsupported | 0.314 ± 0.028 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 96.326 ± 1.103 | - | 97.496 ± 1.792 | 100.614 ± 1.051 | 117.959 ± 3.209 | 105.092 ± 21.013 | 123.786 ± 4.785 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 40.036 ± 0.549 | - | 106.073 ± 1.226 | 38.836 ± 0.541 | 63.124 ± 0.765 | 106.974 ± 0.482 | 50.796 ± 7.958 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 94.436 ± 0.345 | - | 93.688 ± 0.100 | 100.680 ± 1.581 | 115.361 ± 4.188 | 106.457 ± 1.566 | 116.966 ± 0.605 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 39.667 ± 0.518 | - | 106.653 ± 6.588 | 39.043 ± 0.388 | 63.398 ± 0.177 | 103.612 ± 2.178 | 51.384 ± 5.049 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 94.153 ± 0.734 | - | 95.641 ± 0.521 | 98.200 ± 2.891 | 113.659 ± 3.725 | 101.666 ± 12.578 | 113.519 ± 0.598 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 39.707 ± 0.406 | - | 105.685 ± 9.018 | 38.822 ± 0.489 | 61.067 ± 1.080 | 105.664 ± 3.721 | 48.669 ± 0.430 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 84.898 ± 0.462 | - | 85.238 ± 0.368 | 86.328 ± 2.021 | 92.360 ± 4.721 | 89.213 ± 1.160 | 102.432 ± 1.622 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 34.422 ± 0.588 | - | 104.346 ± 1.703 | 32.952 ± 0.761 | 49.800 ± 0.463 | 93.060 ± 0.850 | 43.165 ± 1.322 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 52.114 ± 0.182 | - | 52.818 ± 0.183 | 29.111 ± 0.113 | 62.176 ± 0.169 | 95.343 ± 0.131 | 69.048 ± 9.602 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 24.796 ± 0.701 | - | 70.694 ± 0.875 | 10.740 ± 0.149 | 31.508 ± 0.381 | 95.529 ± 0.217 | 33.664 ± 10.726 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 16.953 ± 0.344 | - | 17.011 ± 0.479 | 13.900 ± 0.094 | 73.599 ± 0.223 | 25.488 ± 0.537 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 6.900 ± 1.317 | - | 34.999 ± 0.995 | 5.751 ± 0.152 | 37.400 ± 0.437 | 25.994 ± 0.247 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.705 ± 0.079 | - | 1.701 ± 0.045 | 25.634 ± 0.342 | 1.556 ± 0.107 | 1.374 ± 0.019 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.763 ± 0.049 | - | 20.089 ± 3.868 | 13.097 ± 0.144 | 0.920 ± 0.016 | 1.258 ± 0.340 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 1.155 ± 0.021 | - | 1.160 ± 0.020 | 1.876 ± 0.017 | 4.287 ± 0.062 | 1.158 ± 0.027 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 0.594 ± 0.015 | - | 21.918 ± 2.945 | 0.948 ± 0.017 | 2.276 ± 1.198 | 1.149 ± 0.015 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 21.258 ± 1.291 | - | 21.379 ± 1.401 | 13.956 ± 0.064 | 73.674 ± 0.196 | 21.757 ± 0.231 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 7.015 ± 0.047 | - | 37.996 ± 6.006 | 5.610 ± 0.014 | 36.954 ± 0.544 | 21.822 ± 0.330 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.689 ± 0.046 | - | 1.699 ± 0.087 | 25.009 ± 1.002 | 1.515 ± 0.013 | 1.366 ± 0.038 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.753 ± 0.018 | - | 21.882 ± 4.081 | 11.862 ± 0.189 | 0.860 ± 0.021 | 1.265 ± 0.169 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 10.967 ± 0.076 | - | 10.857 ± 0.340 | 15.949 ± 0.873 | 35.538 ± 0.459 | 10.891 ± 1.595 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 3.213 ± 0.475 | - | 23.999 ± 0.005 | 4.708 ± 0.227 | 17.115 ± 1.679 | 8.539 ± 0.080 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 15.964 ± 0.317 | - | 16.068 ± 0.111 | 14.253 ± 0.430 | 28.648 ± 0.103 | 15.418 ± 0.418 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 8.128 ± 0.293 | - | 33.979 ± 5.015 | 6.437 ± 0.489 | 13.718 ± 0.052 | 16.483 ± 0.165 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 1.140 ± 0.012 | - | 1.144 ± 0.010 | 1.629 ± 0.015 | 1.254 ± 0.017 | 1.134 ± 0.018 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.588 ± 0.016 | - | 20.998 ± 2.849 | 0.860 ± 0.014 | 0.730 ± 0.716 | 1.133 ± 0.017 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.147 ± 0.024 | - | 1.156 ± 0.015 | 1.161 ± 0.009 | 1.269 ± 0.028 | 1.135 ± 0.016 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.560 ± 0.026 | - | 20.870 ± 1.874 | 0.598 ± 0.019 | 0.676 ± 0.035 | 1.144 ± 0.019 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 16.059 ± 0.212 | - | 15.657 ± 0.181 | 14.520 ± 0.380 | 19.387 ± 0.161 | 15.609 ± 0.428 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 8.451 ± 0.148 | - | 33.991 ± 1.003 | 6.907 ± 0.335 | 10.024 ± 0.241 | 16.434 ± 0.163 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 1.137 ± 0.012 | - | 1.145 ± 0.009 | 1.637 ± 0.019 | 1.257 ± 0.021 | 1.130 ± 0.008 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.574 ± 0.017 | - | 19.093 ± 3.826 | 0.878 ± 0.011 | 0.688 ± 0.041 | 1.143 ± 0.016 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.148 ± 0.014 | - | 1.148 ± 0.022 | 1.164 ± 0.008 | 1.264 ± 0.016 | 1.134 ± 0.008 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.539 ± 0.019 | - | 21.089 ± 1.942 | 0.598 ± 0.014 | 1.286 ± 0.546 | 1.140 ± 0.021 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 1.144 ± 0.006 | 1.144 ± 0.002 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.589 ± 0.004 | 20.233 ± 0.378 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 29.354 ± 0.103 | - | 29.394 ± 0.075 | 58.540 ± 0.962 | 55.580 ± 0.595 | 63.092 ± 9.305 | 60.897 ± 2.787 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 10.728 ± 0.113 | - | 45.997 ± 3.787 | 17.161 ± 0.432 | 38.856 ± 2.822 | 65.839 ± 9.384 | 26.561 ± 6.010 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 114.247 ± 0.834 | - | 112.860 ± 0.479 | 114.796 ± 2.066 | 234.529 ± 2.671 | 183.220 ± 0.935 | 183.532 ± 1.738 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 38.880 ± 0.301 | - | 129.239 ± 1.070 | 36.528 ± 0.587 | 133.049 ± 2.198 | 188.366 ± 0.808 | 76.863 ± 1.575 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 98.937 ± 0.356 | - | 97.668 ± 0.925 | 110.458 ± 1.447 | 106.903 ± 5.249 | 106.514 ± 0.849 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 42.576 ± 0.470 | - | 115.690 ± 3.694 | 41.400 ± 0.721 | 58.221 ± 1.641 | 105.418 ± 1.661 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 88.539 ± 0.191 | - | 86.965 ± 0.286 | 88.704 ± 10.558 | 103.396 ± 5.547 | 94.817 ± 0.950 | 105.883 ± 1.211 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 35.111 ± 0.156 | - | 103.386 ± 1.055 | 33.589 ± 0.262 | 53.622 ± 0.483 | 91.880 ± 0.383 | 47.539 ± 0.884 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 19.648 ± 0.954 | - | 19.478 ± 1.190 | 27.007 ± 0.290 | 63.274 ± 0.192 | 48.285 ± 6.202 | 52.943 ± 3.359 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 4.826 ± 0.078 | - | 35.998 ± 0.002 | 9.537 ± 0.193 | 33.947 ± 0.870 | 56.588 ± 5.496 | 19.809 ± 7.155 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 99.578 ± 2.812 | - | 96.576 ± 0.342 | 100.872 ± 5.110 | 99.596 ± 0.652 | 159.041 ± 1.573 | 165.521 ± 1.438 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 36.293 ± 0.650 | - | 113.545 ± 9.522 | 34.746 ± 0.551 | 56.405 ± 0.990 | 163.848 ± 1.446 | 62.152 ± 2.655 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 94.410 ± 0.864 | - | 93.889 ± 0.167 | 101.347 ± 0.502 | 113.660 ± 3.162 | 101.040 ± 0.447 | 118.376 ± 3.098 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 39.194 ± 0.726 | - | 106.022 ± 4.429 | 39.042 ± 0.247 | 59.849 ± 0.802 | 100.725 ± 0.753 | 52.627 ± 0.982 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 49.923 ± 0.620 | - | 50.150 ± 0.698 | 28.784 ± 0.240 | 30.914 ± 0.870 | 55.692 ± 7.003 | 60.181 ± 5.183 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 12.197 ± 0.141 | - | 66.426 ± 2.265 | 10.838 ± 0.618 | 17.605 ± 1.055 | 57.992 ± 7.648 | 22.717 ± 8.700 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 98.545 ± 0.706 | 97.382 ± 0.532 | unsupported | 112.373 ± 1.433 | 105.653 ± 1.410 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 41.681 ± 1.218 | 107.172 ± 2.286 | unsupported | 41.592 ± 0.588 | 58.398 ± 0.651 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 1.141 ± 0.008 | - | 1.149 ± 0.013 | 1.157 ± 0.014 | 1.155 ± 0.013 | 6.068 ± 3.686 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.584 ± 0.016 | - | 11.998 ± 0.004 | 0.608 ± 0.020 | 1.161 ± 0.010 | 6.011 ± 3.648 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 1.138 ± 0.013 | - | 1.150 ± 0.007 | 1.157 ± 0.009 | 1.164 ± 0.010 | 6.203 ± 3.536 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.582 ± 0.016 | - | 11.998 ± 0.004 | 0.593 ± 0.011 | 0.637 ± 0.027 | 6.222 ± 3.670 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.707 ± 0.010 | - | unsupported | 1.754 ± 0.015 | 1.753 ± 0.005 | 4.003 ± 11.186 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.869 ± 0.007 | - | unsupported | 0.919 ± 0.024 | 1.341 ± 0.116 | 6.440 ± 2.651 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.502 ± 0.031 | - | 0.642 ± 0.044 | 0.717 ± 0.025 | 0.329 ± 0.026 | 0.958 ± 0.130 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.358 ± 0.033 | - | 40.432 ± 2.247 | 0.434 ± 0.027 | 0.231 ± 0.028 | 0.962 ± 0.691 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 1.139 ± 0.014 | - | 1.275 ± 0.011 | 1.748 ± 0.013 | 1.162 ± 0.009 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.582 ± 0.013 | - | 11.999 ± 0.005 | 0.903 ± 0.009 | 0.637 ± 0.019 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 1.141 ± 0.011 | - | 1.147 ± 0.015 | 1.158 ± 0.010 | 1.161 ± 0.015 | 1.849 ± 11.240 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.580 ± 0.015 | - | 11.999 ± 0.004 | 0.600 ± 0.019 | 1.168 ± 0.018 | 5.574 ± 7.704 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.756 ± 0.025 | - | 0.961 ± 0.039 | 0.806 ± 0.031 | 0.637 ± 0.017 | 1.354 ± 0.113 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.884 ± 0.015 | - | 32.939 ± 1.075 | 0.672 ± 0.028 | 0.815 ± 0.058 | 1.338 ± 0.174 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.710 ± 0.011 | - | 0.001 ± 0.000 | 1.725 ± 0.020 | 1.731 ± 0.011 | 6.514 ± 1.114 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.866 ± 0.013 | - | 0.001 ± 0.000 | 0.887 ± 0.016 | 0.923 ± 0.010 | 6.397 ± 3.596 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.035 ± 0.002 | - | 0.035 ± 0.000 | 0.055 ± 0.000 | 0.089 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.060 ± 0.001 | - | 0.061 ± 0.001 | 0.069 ± 0.001 | 0.112 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.211 ± 0.009 | - | 0.211 ± 0.014 | 0.253 ± 0.017 | 0.251 ± 0.120 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.041 ± 0.001 | - | 0.044 ± 0.005 | 0.113 ± 0.002 | 0.078 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.039 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.048 ± 0.183 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.351 ± 0.022 | - | 0.350 ± 0.040 | 0.895 ± 0.021 | 0.553 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.025 ± 0.000 | 0.059 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.049 ± 0.226 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.034 ± 0.001 | - | 0.035 ± 0.001 | 0.046 ± 0.000 | 0.068 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.063 ± 0.005 | - | 0.061 ± 0.001 | 0.090 ± 0.003 | 0.121 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.210 ± 0.010 | - | 0.212 ± 0.004 | 0.249 ± 0.015 | 0.259 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.046 ± 0.007 | - | 0.044 ± 0.000 | 0.093 ± 0.002 | 0.102 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.036 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.006 ± 0.001 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.350 ± 0.033 | - | 0.383 ± 0.050 | 0.674 ± 0.027 | 0.998 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.036 ± 0.002 | 0.063 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.025 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.359 ± 0.010 | - | 0.346 ± 0.010 | 0.483 ± 0.008 | 0.575 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.641 ± 0.015 | - | 1.642 ± 0.023 | 1.669 ± 0.019 | 1.876 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 5.450 ± 0.093 | - | 5.375 ± 0.038 | 5.338 ± 0.021 | 6.684 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 1.627 ± 0.031 | - | 1.602 ± 0.032 | 1.314 ± 0.010 | 2.057 ± 0.123 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.074 ± 0.013 | - | 0.075 ± 0.003 | 0.077 ± 0.003 | 0.130 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.012 ± 0.001 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.010 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.017 ± 0.000 | - | 0.019 ± 0.002 | 0.022 ± 0.001 | 0.040 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.001 | 0.012 ± 0.008 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.022 ± 0.000 | - | 0.024 ± 0.000 | 0.026 ± 0.000 | 0.040 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 13.061 ± 0.028 | - | 13.014 ± 0.061 | 10.333 ± 0.111 | 16.845 ± 0.180 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.507 ± 0.016 | - | 0.565 ± 0.020 | 0.563 ± 0.027 | 0.701 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.018 ± 0.000 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.042 ± 0.007 | - | 0.044 ± 0.000 | 0.046 ± 0.000 | 0.098 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.350 ± 0.013 | - | 0.352 ± 0.012 | 0.512 ± 0.128 | 0.559 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 1.690 ± 0.055 | - | 1.654 ± 0.023 | 1.645 ± 0.009 | 2.092 ± 0.087 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 5.418 ± 0.088 | - | 5.409 ± 0.029 | 6.080 ± 0.066 | 4.662 ± 0.867 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 2.618 ± 0.054 | - | 2.664 ± 0.025 | 1.172 ± 0.012 | 2.996 ± 1.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.073 ± 0.001 | - | 0.072 ± 0.002 | 0.090 ± 0.002 | 0.131 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.006 ± 0.001 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.017 ± 0.001 | - | 0.019 ± 0.000 | 0.021 ± 0.003 | 0.033 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.023 ± 0.000 | - | 0.023 ± 0.000 | 0.026 ± 0.000 | 0.042 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 21.030 ± 0.084 | - | 20.698 ± 0.034 | 9.163 ± 0.157 | 22.208 ± 6.534 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.013 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.560 ± 0.006 | - | 0.569 ± 0.023 | 0.666 ± 0.019 | 1.056 ± 0.974 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.015 ± 0.000 | - | 0.015 ± 0.000 | 0.018 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.044 ± 0.001 | - | 0.045 ± 0.000 | 0.047 ± 0.001 | 0.065 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.383 ± 0.012 | - | 0.384 ± 0.006 | 0.330 ± 0.012 | 0.268 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.511 ± 0.024 | - | 1.492 ± 0.030 | 1.316 ± 0.011 | 1.305 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 4.643 ± 0.019 | - | 4.707 ± 0.026 | 3.933 ± 0.018 | 4.498 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 0.742 ± 0.018 | - | 0.747 ± 0.011 | 0.720 ± 0.022 | 0.970 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.002 ± 0.001 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.050 ± 0.001 | - | 0.051 ± 0.001 | 0.050 ± 0.002 | 0.078 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.013 ± 0.000 | - | 0.016 ± 0.001 | 0.017 ± 0.001 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.001 | 0.013 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.020 ± 0.002 | - | 0.021 ± 0.000 | 0.020 ± 0.000 | 0.027 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 5.866 ± 0.075 | - | 5.926 ± 0.039 | 5.639 ± 0.096 | 7.555 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.394 ± 0.032 | - | 0.396 ± 0.104 | 0.352 ± 0.010 | 0.506 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.037 ± 0.003 | - | 0.039 ± 0.000 | 0.035 ± 0.000 | 0.060 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.369 ± 0.013 | - | 0.370 ± 0.013 | 0.419 ± 0.083 | 0.266 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.479 ± 0.031 | - | 1.500 ± 0.030 | 1.302 ± 0.017 | 0.898 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 4.618 ± 0.052 | - | 4.603 ± 0.042 | 4.510 ± 0.009 | 2.954 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.372 ± 0.021 | - | 1.357 ± 0.026 | 0.748 ± 0.029 | 1.213 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.002 ± 0.001 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.066 ± 0.005 | - | 0.063 ± 0.005 | 0.060 ± 0.001 | 0.113 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.002 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.015 ± 0.002 | - | 0.017 ± 0.001 | 0.016 ± 0.000 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.019 ± 0.002 | - | 0.021 ± 0.000 | 0.020 ± 0.000 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 10.724 ± 0.160 | - | 10.803 ± 0.158 | 5.789 ± 0.038 | 10.247 ± 2.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.438 ± 0.041 | - | 0.478 ± 0.015 | 0.420 ± 0.025 | 0.330 ± 0.082 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.037 ± 0.003 | - | 0.035 ± 0.002 | 0.035 ± 0.000 | 0.085 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.166 ± 0.002 | - | 0.168 ± 0.002 | 0.144 ± 0.014 | 0.294 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 0.808 ± 0.008 | - | 0.810 ± 0.015 | 0.733 ± 0.017 | 1.493 ± 0.168 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 2.370 ± 0.024 | - | 2.365 ± 0.008 | 2.254 ± 0.042 | 4.574 ± 0.090 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.335 ± 0.010 | - | 0.337 ± 0.015 | 0.357 ± 0.012 | 0.977 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.006 ± 0.003 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.023 ± 0.000 | - | 0.024 ± 0.003 | 0.029 ± 0.001 | 0.104 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.001 | 0.006 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.013 ± 0.001 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.001 | 0.006 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.010 ± 0.000 | - | 0.011 ± 0.002 | 0.014 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 2.639 ± 0.019 | - | 2.642 ± 0.043 | 2.843 ± 0.024 | 7.610 ± 0.181 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.169 ± 0.009 | - | 0.169 ± 0.002 | 0.174 ± 0.007 | 0.551 ± 0.113 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.019 ± 0.000 | - | 0.020 ± 0.000 | 0.023 ± 0.002 | 0.119 ± 0.179 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.169 ± 0.002 | - | 0.173 ± 0.009 | 0.205 ± 0.013 | 0.264 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 0.811 ± 0.014 | - | 0.826 ± 0.018 | 0.735 ± 0.014 | 1.140 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 2.372 ± 0.046 | - | 2.390 ± 0.012 | 2.429 ± 0.009 | 2.831 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.826 ± 0.034 | - | 0.834 ± 0.032 | 0.511 ± 0.016 | 1.199 ± 0.069 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.026 ± 0.006 | - | 0.025 ± 0.001 | 0.032 ± 0.006 | 0.090 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.011 ± 0.000 | - | 0.012 ± 0.000 | 0.015 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 6.699 ± 0.110 | - | 6.690 ± 0.139 | 3.955 ± 0.028 | 9.993 ± 2.525 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.171 ± 0.005 | - | 0.176 ± 0.011 | 0.191 ± 0.011 | 0.320 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.014 ± 0.002 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.023 ± 0.000 | 0.079 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.100 ± 0.001 | - | 0.102 ± 0.001 | 0.126 ± 0.005 | 0.105 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.133 ± 0.010 | - | 0.126 ± 0.001 | 0.162 ± 0.001 | 0.167 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.462 ± 0.032 | - | 0.480 ± 0.008 | 0.579 ± 0.043 | 0.433 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.073 ± 0.001 | - | 0.072 ± 0.003 | 0.100 ± 0.005 | 0.142 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.009 ± 0.000 | - | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.059 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.003 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.010 ± 0.000 | - | 0.011 ± 0.000 | 0.017 ± 0.000 | 0.027 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.044 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.003 | 0.008 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.052 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.645 ± 0.026 | - | 0.658 ± 0.028 | 0.881 ± 0.034 | 0.991 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.050 ± 0.001 | - | 0.051 ± 0.000 | 0.052 ± 0.001 | 0.092 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.052 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.002 | 0.056 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.103 ± 0.005 | - | 0.104 ± 0.002 | 0.100 ± 0.005 | 0.083 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.127 ± 0.002 | - | 0.129 ± 0.007 | 0.123 ± 0.004 | 0.169 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.497 ± 0.016 | - | 0.493 ± 0.030 | 0.302 ± 0.030 | 0.641 ± 0.086 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.079 ± 0.005 | - | 0.081 ± 0.004 | 0.107 ± 0.003 | 0.162 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.012 ± 0.001 | 0.010 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.009 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.001 | 0.027 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.012 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.021 ± 0.001 | 0.022 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.012 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.013 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.014 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.015 ± 0.000 | 0.058 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.014 ± 0.000 | 0.011 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.014 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.637 ± 0.028 | - | 0.683 ± 0.041 | 0.456 ± 0.034 | 0.834 ± 0.076 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.016 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.053 ± 0.008 | - | 0.051 ± 0.001 | 0.048 ± 0.003 | 0.115 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.016 ± 0.001 | 0.054 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.019 ± 0.000 | 0.060 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.153 ± 0.005 | - | 0.152 ± 0.006 | 0.140 ± 0.002 | 0.194 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.357 ± 0.008 | - | 0.355 ± 0.010 | 0.507 ± 0.019 | 0.537 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 0.936 ± 0.013 | - | 0.936 ± 0.027 | 1.222 ± 0.030 | 1.498 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.210 ± 0.014 | - | 0.221 ± 0.007 | 0.291 ± 0.020 | 0.453 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.033 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.009 ± 0.001 | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.027 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 1.772 ± 0.040 | - | 2.097 ± 0.038 | 2.370 ± 0.048 | 3.550 ± 0.062 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.063 ± 0.001 | - | 0.066 ± 0.001 | 0.072 ± 0.001 | 0.139 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.022 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.154 ± 0.008 | - | 0.160 ± 0.013 | 0.198 ± 0.005 | 0.227 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.354 ± 0.018 | - | 0.358 ± 0.016 | 0.641 ± 0.046 | 0.600 ± 0.141 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 0.925 ± 0.011 | - | 0.930 ± 0.019 | 1.371 ± 0.064 | 0.997 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.306 ± 0.016 | - | 0.309 ± 0.016 | 0.267 ± 0.016 | 1.101 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.018 ± 0.000 | 0.069 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.011 ± 0.001 | - | 0.011 ± 0.001 | 0.020 ± 0.000 | 0.059 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.002 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 2.453 ± 0.022 | - | 2.837 ± 0.108 | 2.061 ± 0.017 | 10.727 ± 2.280 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.070 ± 0.005 | - | 0.070 ± 0.007 | 0.101 ± 0.003 | 0.231 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.009 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.001 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.112 ± 0.005 | - | 0.105 ± 0.009 | 0.059 ± 0.016 | 0.214 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.132 ± 0.005 | - | 0.118 ± 0.011 | 0.076 ± 0.001 | 0.280 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.457 ± 0.013 | - | 0.461 ± 0.006 | 0.553 ± 0.015 | 0.557 ± 0.106 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.054 ± 0.003 | - | 0.060 ± 0.002 | 0.080 ± 0.001 | 0.137 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.016 ± 0.000 | 0.081 ± 0.183 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.425 ± 0.008 | - | 0.524 ± 0.022 | 0.619 ± 0.012 | 0.804 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.045 ± 0.001 | - | 0.050 ± 0.002 | 0.049 ± 0.001 | 0.095 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.002 ± 0.001 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.014 ± 0.001 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.113 ± 0.002 | - | 0.103 ± 0.001 | 0.087 ± 0.004 | 0.184 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.133 ± 0.005 | - | 6.000 ± 0.005 | 0.136 ± 0.015 | 0.255 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.467 ± 0.013 | - | 20.298 ± 2.722 | 0.379 ± 0.016 | 0.770 ± 0.062 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.055 ± 0.009 | - | 6.002 ± 0.013 | 0.085 ± 0.001 | 0.109 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.012 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.019 ± 0.001 | 0.039 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.012 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.002 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.013 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.002 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.015 ± 0.001 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.013 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.002 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.015 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.436 ± 0.026 | - | 19.993 ± 2.730 | 0.323 ± 0.027 | 0.718 ± 0.115 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.015 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.046 ± 0.001 | - | 0.051 ± 0.004 | 0.041 ± 0.002 | 0.111 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.015 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.019 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.129 ± 0.013 | - | 1.134 ± 0.013 | 0.985 ± 0.020 | 1.309 ± 0.100 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 4.313 ± 0.016 | - | 4.357 ± 0.024 | 3.792 ± 0.015 | 3.609 ± 0.145 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 12.863 ± 0.045 | - | 12.822 ± 0.043 | 12.268 ± 0.065 | 10.590 ± 0.102 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 2.171 ± 0.044 | - | 2.180 ± 0.032 | 2.233 ± 0.058 | 2.929 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.121 ± 0.008 | - | 0.119 ± 0.004 | 0.144 ± 0.012 | 0.192 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.106 ± 0.012 | - | 0.107 ± 0.018 | 0.120 ± 0.003 | 0.175 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.014 ± 0.001 | - | 0.016 ± 0.002 | 0.019 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.015 ± 0.000 | - | 0.017 ± 0.001 | 0.017 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.039 ± 0.000 | - | 0.040 ± 0.002 | 0.042 ± 0.001 | 0.042 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.021 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.051 ± 0.001 | - | 0.053 ± 0.002 | 0.054 ± 0.002 | 0.052 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 17.321 ± 0.102 | - | 17.403 ± 0.123 | 17.631 ± 0.165 | 23.529 ± 0.119 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.014 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 0.906 ± 0.021 | - | 0.923 ± 0.098 | 1.050 ± 0.022 | 1.093 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.037 ± 0.001 | - | 0.038 ± 0.000 | 0.036 ± 0.008 | 0.038 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.103 ± 0.010 | - | 0.100 ± 0.002 | 0.104 ± 0.012 | 0.126 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.140 ± 0.012 | - | 1.147 ± 0.018 | 0.964 ± 0.011 | 1.328 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 4.339 ± 0.037 | - | 4.368 ± 0.042 | 3.871 ± 0.015 | 1.781 ± 0.091 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 12.263 ± 0.062 | - | 12.666 ± 0.037 | 13.078 ± 0.129 | 6.451 ± 0.149 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 4.056 ± 0.038 | - | 4.119 ± 0.112 | 2.632 ± 0.059 | 4.636 ± 0.115 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.121 ± 0.010 | - | 0.121 ± 0.005 | 0.139 ± 0.025 | 0.149 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.117 ± 0.021 | - | 0.116 ± 0.009 | 0.144 ± 0.012 | 0.140 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.015 ± 0.001 | - | 0.016 ± 0.000 | 0.018 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.003 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.015 ± 0.000 | - | 0.016 ± 0.000 | 0.017 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.039 ± 0.002 | - | 0.039 ± 0.002 | 0.042 ± 0.001 | 0.042 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.021 ± 0.000 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.052 ± 0.001 | - | 0.053 ± 0.002 | 0.054 ± 0.005 | 0.052 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 32.136 ± 0.044 | - | 32.392 ± 0.354 | 20.584 ± 0.090 | 73.990 ± 26.998 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.017 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 0.894 ± 0.029 | - | 0.926 ± 0.013 | 1.053 ± 0.075 | 0.672 ± 0.138 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.038 ± 0.008 | - | 0.038 ± 0.000 | 0.036 ± 0.003 | 0.038 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.104 ± 0.005 | - | 0.101 ± 0.007 | 0.100 ± 0.006 | 0.127 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.764 ± 0.011 | - | 0.768 ± 0.010 | 0.739 ± 0.005 | 0.872 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 3.547 ± 0.030 | - | 3.522 ± 0.037 | 3.074 ± 0.037 | 2.888 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 10.984 ± 0.038 | - | 12.328 ± 0.839 | 9.064 ± 0.021 | 8.742 ± 0.135 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.340 ± 0.118 | - | 1.339 ± 0.032 | 1.348 ± 0.019 | 1.648 ± 0.083 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.004 ± 0.001 | 0.006 ± 0.001 | 0.011 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.098 ± 0.021 | - | 0.090 ± 0.003 | 0.099 ± 0.010 | 0.142 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.074 ± 0.008 | - | 0.072 ± 0.006 | 0.067 ± 0.007 | 0.129 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.013 ± 0.001 | - | 0.014 ± 0.000 | 0.015 ± 0.001 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.013 ± 0.001 | - | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.031 ± 0.000 | - | 0.033 ± 0.000 | 0.030 ± 0.000 | 0.034 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.016 ± 0.000 | - | 0.017 ± 0.000 | 0.018 ± 0.000 | 0.022 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.041 ± 0.001 | - | 0.042 ± 0.001 | 0.038 ± 0.000 | 0.043 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 10.574 ± 0.070 | - | 10.450 ± 0.087 | 10.636 ± 0.034 | 13.185 ± 0.147 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 0.675 ± 0.199 | - | 0.898 ± 0.040 | 0.684 ± 0.026 | 0.784 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.030 ± 0.001 | - | 0.031 ± 0.001 | 0.029 ± 0.003 | 0.033 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.082 ± 0.007 | - | 0.085 ± 0.007 | 0.074 ± 0.013 | 0.115 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.747 ± 0.012 | - | 0.747 ± 0.021 | 0.717 ± 0.006 | 0.891 ± 0.134 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 3.533 ± 0.028 | - | 3.550 ± 0.026 | 3.055 ± 0.010 | 1.390 ± 0.087 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 10.984 ± 0.066 | - | 10.858 ± 0.037 | 9.879 ± 0.177 | 4.572 ± 0.649 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 1.509 ± 0.068 | - | 1.489 ± 0.047 | 1.452 ± 0.020 | 2.558 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.097 ± 0.008 | - | 0.096 ± 0.011 | 0.092 ± 0.006 | 0.151 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.077 ± 0.010 | - | 0.074 ± 0.002 | 0.078 ± 0.017 | 0.119 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.013 ± 0.003 | - | 0.014 ± 0.000 | 0.015 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.013 ± 0.000 | - | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.031 ± 0.000 | - | 0.033 ± 0.001 | 0.030 ± 0.000 | 0.038 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.017 ± 0.000 | - | 0.018 ± 0.000 | 0.018 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.043 ± 0.001 | - | 0.043 ± 0.001 | 0.039 ± 0.001 | 0.043 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 11.682 ± 0.069 | - | 11.814 ± 0.069 | 11.332 ± 0.083 | 20.997 ± 6.024 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.016 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 0.689 ± 0.017 | - | 0.709 ± 0.033 | 0.679 ± 0.021 | 0.457 ± 0.093 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.030 ± 0.000 | - | 0.032 ± 0.005 | 0.030 ± 0.001 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.080 ± 0.004 | - | 0.085 ± 0.003 | 0.072 ± 0.002 | 0.101 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.284 ± 0.009 | - | unsupported | 0.223 ± 0.015 | 0.478 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.407 ± 0.009 | - | unsupported | 1.278 ± 0.024 | 1.705 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 3.539 ± 0.043 | - | unsupported | 3.339 ± 0.063 | 4.555 ± 0.108 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.681 ± 0.024 | - | unsupported | 0.710 ± 0.013 | 0.849 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.032 ± 0.002 | - | unsupported | 0.037 ± 0.001 | 0.046 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.024 ± 0.001 | - | unsupported | 0.031 ± 0.001 | 0.038 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.007 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.012 ± 0.000 | - | unsupported | 0.016 ± 0.006 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.003 ± 0.001 | - | unsupported | 0.006 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.007 ± 0.002 | - | unsupported | 0.010 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.016 ± 0.000 | - | unsupported | 0.018 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 5.409 ± 0.049 | - | unsupported | 5.604 ± 0.023 | 6.769 ± 0.101 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.247 ± 0.012 | - | unsupported | 0.249 ± 0.015 | 0.326 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.012 ± 0.000 | - | unsupported | 0.014 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.031 ± 0.005 | - | unsupported | 0.031 ± 0.000 | 0.040 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.278 ± 0.008 | - | unsupported | 0.317 ± 0.012 | 0.505 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 1.410 ± 0.026 | - | unsupported | 1.257 ± 0.016 | 0.897 ± 0.157 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 3.555 ± 0.063 | - | unsupported | 4.161 ± 0.033 | 2.299 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 0.725 ± 0.009 | - | unsupported | 0.802 ± 0.023 | 1.492 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.007 ± 0.001 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.032 ± 0.001 | - | unsupported | 0.035 ± 0.003 | 0.047 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.026 ± 0.001 | - | unsupported | 0.034 ± 0.001 | 0.048 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.006 ± 0.000 | - | unsupported | 0.012 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.012 ± 0.001 | - | unsupported | 0.015 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.007 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.016 ± 0.000 | - | unsupported | 0.018 ± 0.001 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 5.763 ± 0.049 | - | unsupported | 6.237 ± 0.042 | 12.128 ± 2.482 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.247 ± 0.012 | - | unsupported | 0.243 ± 0.014 | 0.201 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.012 ± 0.000 | - | unsupported | 0.015 ± 0.000 | 0.020 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.030 ± 0.001 | - | unsupported | 0.031 ± 0.000 | 0.040 ± 0.008 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 1.087 ± 0.020 | - | unsupported | 1.216 ± 0.044 | 2.530 ± 0.080 | 1.912 ± 0.061 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.089 ± 0.002 | - | unsupported | 0.037 ± 0.001 | 0.072 ± 0.010 | 0.088 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.092 ± 0.002 | - | unsupported | 0.048 ± 0.001 | 0.120 ± 0.009 | 0.143 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.327 ± 0.016 | - | unsupported | 0.335 ± 0.024 | 0.336 ± 0.026 | 0.493 ± 0.157 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 1.086 ± 0.016 | - | unsupported | 0.463 ± 0.035 | 1.154 ± 0.056 | 2.009 ± 0.339 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.081 ± 0.001 | - | unsupported | 0.039 ± 0.003 | 0.068 ± 0.012 | 0.074 ± 0.015 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.097 ± 0.001 | - | unsupported | 0.059 ± 0.007 | 0.111 ± 0.009 | 0.117 ± 0.007 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.330 ± 0.011 | - | unsupported | 0.129 ± 0.009 | 0.529 ± 0.029 | 0.524 ± 0.038 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.228 ± 0.021 | - | unsupported | 0.444 ± 0.029 | 0.724 ± 0.026 | 0.264 ± 0.018 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.021 ± 0.000 | - | unsupported | 0.026 ± 0.005 | 0.128 ± 0.012 | 0.042 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.039 ± 0.000 | - | unsupported | 0.032 ± 0.001 | 0.153 ± 0.024 | 0.066 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.121 ± 0.005 | - | unsupported | 0.213 ± 0.010 | 0.273 ± 0.023 | 0.120 ± 0.007 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.224 ± 0.022 | - | unsupported | 0.485 ± 0.034 | 0.842 ± 0.020 | 0.266 ± 0.026 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.021 ± 0.000 | - | unsupported | 0.037 ± 0.001 | 0.158 ± 0.011 | 0.044 ± 0.008 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.039 ± 0.000 | - | unsupported | 0.051 ± 0.001 | 0.169 ± 0.015 | 0.065 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.120 ± 0.005 | - | unsupported | 0.244 ± 0.013 | 0.291 ± 0.009 | 0.120 ± 0.008 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.134 ± 0.006 | - | 0.130 ± 0.017 | 0.212 ± 0.017 | 0.365 ± 0.040 | 0.194 ± 0.010 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.053 ± 0.005 | - | 0.055 ± 0.010 | 0.048 ± 0.000 | 0.056 ± 0.003 | 0.036 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.053 ± 0.001 | - | 0.055 ± 0.002 | 0.061 ± 0.000 | 0.064 ± 0.001 | 0.051 ± 0.003 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.074 ± 0.000 | - | 0.076 ± 0.000 | 0.116 ± 0.014 | 0.113 ± 0.026 | 0.082 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.125 ± 0.016 | - | 0.129 ± 0.005 | 0.250 ± 0.028 | 0.472 ± 0.026 | 0.151 ± 0.020 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.052 ± 0.002 | - | 0.055 ± 0.001 | 0.064 ± 0.002 | 0.056 ± 0.001 | 0.037 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.053 ± 0.002 | - | 0.055 ± 0.003 | 0.085 ± 0.001 | 0.065 ± 0.001 | 0.052 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.075 ± 0.002 | - | 0.076 ± 0.002 | 0.132 ± 0.015 | 0.194 ± 0.031 | 0.085 ± 0.003 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 17.505 ± 0.084 | - | 17.240 ± 0.145 | 41.714 ± 0.202 | 53.563 ± 1.626 | 38.810 ± 9.358 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 17.288 ± 0.096 | - | 17.260 ± 0.101 | 17.549 ± 0.105 | 53.254 ± 6.837 | 32.006 ± 3.768 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 11.684 ± 0.218 | - | 10.983 ± 0.536 | 11.685 ± 0.099 | 14.796 ± 0.107 | 14.426 ± 0.438 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 722.056 ± 63.793 | - | 816.147 ± 41.560 | 5.779 ± 0.148 | 10.341 ± 2.003 | 9.129 ± 0.131 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 4.810 ± 0.491 | - | 4.828 ± 0.018 | 4.689 ± 0.023 | 5.516 ± 0.166 | 5.301 ± 0.126 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 6.100 ± 0.063 | - | 6.093 ± 0.105 | 6.452 ± 0.058 | 8.233 ± 0.159 | 7.852 ± 0.087 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 6.083 ± 0.212 | - | 6.073 ± 0.097 | 5.739 ± 0.056 | 6.707 ± 0.067 | 6.303 ± 0.032 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 7.038 ± 0.110 | - | 6.981 ± 0.020 | 7.197 ± 0.030 | 9.337 ± 0.072 | 9.008 ± 0.065 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 9.199 ± 0.042 | - | 9.321 ± 0.075 | 9.313 ± 0.013 | 22.248 ± 0.309 | 9.440 ± 0.277 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 15.658 ± 0.081 | - | 15.771 ± 0.082 | 9.295 ± 0.071 | 20.055 ± 0.540 | 8.832 ± 0.330 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.211 ± 0.012 | 0.228 ± 0.007 | unsupported | 0.233 ± 0.001 | 0.543 ± 0.024 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 4.543 ± 0.052 | 4.697 ± 0.012 | unsupported | 4.656 ± 0.036 | 6.238 ± 0.028 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.202 ± 0.013 | 20.997 ± 0.856 | unsupported | 0.185 ± 0.005 | 1.582 ± 0.067 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 423.864 ± 28.714 | 446.062 ± 27.306 | unsupported | 2.611 ± 0.030 | 8.957 ± 0.701 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 14.514 ± 0.245 | - | 15.262 ± 0.320 | 15.442 ± 0.125 | 30.984 ± 0.360 | 19.500 ± 2.918 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 453.648 ± 35.569 | - | 479.668 ± 18.154 | 8.292 ± 0.048 | 18.287 ± 0.150 | 12.715 ± 0.191 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 8.464 ± 0.259 | 8.100 ± 0.063 | 4.288 ± 0.032 | 34.649 ± 0.276 | 4.574 ± 0.059 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 460.193 ± 32.746 | 355.049 ± 87.112 | 2.914 ± 0.026 | 40.500 ± 1.214 | 4.229 ± 5.263 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 11.817 ± 0.498 | - | 11.454 ± 0.110 | 20.698 ± 0.271 | 29.697 ± 0.163 | 21.164 ± 16.617 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 726.232 ± 49.387 | - | 716.696 ± 19.193 | 10.890 ± 0.247 | 24.968 ± 2.309 | 19.314 ± 15.192 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 1.155 ± 0.025 | - | 1.147 ± 0.022 | 1.162 ± 0.016 | 5.253 ± 0.338 | 1.587 ± 0.027 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.582 ± 0.022 | - | 20.979 ± 0.999 | 1.159 ± 0.012 | 9.984 ± 1.184 | 1.561 ± 0.376 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 15.616 ± 0.143 | - | 15.372 ± 0.093 | 15.771 ± 0.114 | 11.390 ± 0.103 | 12.964 ± 0.149 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 198.326 ± 14.744 | - | 241.304 ± 15.766 | 40.797 ± 3.818 | 16.101 ± 0.480 | 15.844 ± 1.560 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 15.390 ± 0.449 | - | 15.313 ± 0.117 | 15.655 ± 0.144 | 11.394 ± 0.071 | 12.796 ± 0.326 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 194.324 ± 9.831 | - | 235.704 ± 5.018 | 40.722 ± 0.088 | 16.514 ± 0.352 | 15.528 ± 2.395 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 0.591 ± 0.006 | 0.592 ± 0.003 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 13.081 ± 0.150 | 13.017 ± 0.225 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 0.960 ± 0.011 | 0.973 ± 0.006 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 18.996 ± 0.095 | 18.865 ± 0.128 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 0.547 ± 0.007 | 0.551 ± 0.005 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 12.527 ± 0.480 | 12.483 ± 0.149 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 0.913 ± 0.003 | 0.925 ± 0.008 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 18.190 ± 0.132 | 18.384 ± 0.172 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 11.643 ± 0.198 | - | 11.056 ± 0.525 | 11.689 ± 0.108 | 14.727 ± 0.128 | 14.894 ± 2.808 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 735.019 ± 66.634 | - | 781.898 ± 41.869 | 5.798 ± 0.137 | 10.782 ± 0.875 | 9.104 ± 0.215 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 70.865 ± 0.183 | 71.955 ± 1.381 | 67.629 ± 0.275 | 38.833 ± 0.554 | 41.911 ± 9.386 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 641.246 ± 57.021 | 650.195 ± 68.298 | 150.358 ± 1.916 | 46.886 ± 1.873 | 47.634 ± 6.048 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 15.799 ± 0.145 | - | 16.540 ± 0.309 | 13.880 ± 0.711 | 106.849 ± 3.210 | 18.229 ± 0.470 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 21.939 ± 0.142 | - | 23.797 ± 0.301 | 7.253 ± 0.184 | 100.286 ± 8.201 | 13.943 ± 0.875 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.063 ± 0.003 | 0.063 ± 0.001 | unsupported | 0.069 ± 0.016 | 0.023 ± 0.008 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 1.044 ± 0.025 | 1.059 ± 0.027 | unsupported | 1.077 ± 0.045 | 0.794 ± 0.104 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.033 ± 0.001 | 19.999 ± 0.136 | unsupported | 0.040 ± 0.050 | 0.081 ± 0.007 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.471 ± 0.001 | 20.549 ± 0.247 | unsupported | 0.493 ± 0.024 | 0.545 ± 0.166 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.031 ± 0.001 | 0.031 ± 0.001 | unsupported | 0.065 ± 0.003 | 0.147 ± 0.019 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 0.512 ± 0.031 | 0.532 ± 0.025 | unsupported | 0.964 ± 0.086 | 1.929 ± 0.026 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.018 ± 0.001 | 12.000 ± 0.016 | unsupported | 0.051 ± 0.004 | 0.097 ± 0.007 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.256 ± 0.003 | 11.999 ± 0.000 | unsupported | 0.438 ± 0.012 | 0.823 ± 0.015 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.104 ± 0.002 | 0.104 ± 0.002 | unsupported | 0.135 ± 0.001 | 0.108 ± 0.005 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 1.764 ± 0.014 | 1.689 ± 0.012 | unsupported | 2.388 ± 0.015 | 1.425 ± 0.041 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.045 ± 0.000 | 12.000 ± 0.000 | unsupported | 0.064 ± 0.042 | 0.095 ± 0.011 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 0.717 ± 0.009 | 11.999 ± 0.001 | unsupported | 0.964 ± 0.018 | 0.895 ± 0.142 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.043 ± 0.010 | 0.061 ± 0.009 | unsupported | 0.105 ± 0.001 | 0.115 ± 0.006 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.940 ± 0.084 | 0.979 ± 0.076 | unsupported | 1.191 ± 0.099 | 1.768 ± 0.065 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.069 ± 0.009 | 52.015 ± 0.125 | unsupported | 0.049 ± 0.001 | 0.156 ± 0.013 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.542 ± 0.036 | 80.319 ± 0.504 | unsupported | 0.507 ± 0.017 | 1.182 ± 0.051 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.060 ± 0.008 | 0.070 ± 0.004 | unsupported | 0.077 ± 0.001 | 0.193 ± 0.016 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.290 ± 0.055 | 1.317 ± 0.097 | unsupported | 1.209 ± 0.067 | 2.869 ± 0.032 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.075 ± 0.009 | 71.890 ± 0.110 | unsupported | 0.041 ± 0.002 | 0.177 ± 0.011 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.727 ± 0.053 | 100.174 ± 0.327 | unsupported | 0.507 ± 0.012 | 2.243 ± 0.069 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.196 ± 0.008 | 0.218 ± 0.008 | unsupported | unsupported | 0.231 ± 0.002 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 3.635 ± 0.128 | 3.671 ± 0.124 | unsupported | unsupported | 4.167 ± 0.183 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.176 ± 0.004 | 91.828 ± 0.188 | unsupported | unsupported | 0.265 ± 0.011 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 1.568 ± 0.036 | 120.871 ± 0.812 | unsupported | unsupported | 2.332 ± 0.109 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.009 ± 0.000 | 0.013 ± 0.000 | unsupported | 0.009 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.159 ± 0.003 | 0.168 ± 0.003 | unsupported | 0.154 ± 0.004 | 0.228 ± 0.097 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.009 ± 0.000 | 19.993 ± 0.042 | unsupported | 0.010 ± 0.000 | 0.019 ± 0.000 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.079 ± 0.002 | 19.999 ± 0.501 | unsupported | 0.050 ± 0.006 | 0.156 ± 0.006 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.026 ± 0.007 | 0.036 ± 0.002 | unsupported | 0.056 ± 0.001 | 0.131 ± 0.016 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.471 ± 0.095 | 0.507 ± 0.100 | unsupported | 1.146 ± 0.073 | 1.584 ± 0.094 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.028 ± 0.012 | 25.968 ± 0.094 | unsupported | 0.064 ± 0.002 | 0.108 ± 0.014 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.265 ± 0.035 | 40.559 ± 0.302 | unsupported | 0.591 ± 0.015 | 1.420 ± 0.101 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.024 ± 0.004 | 0.025 ± 0.001 | unsupported | 0.055 ± 0.001 | 0.110 ± 0.006 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.374 ± 0.057 | 0.398 ± 0.050 | unsupported | 1.058 ± 0.050 | 1.296 ± 0.033 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.014 ± 0.001 | 12.000 ± 0.000 | unsupported | 0.051 ± 0.000 | 0.091 ± 0.007 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.189 ± 0.013 | 11.999 ± 0.000 | unsupported | 0.436 ± 0.021 | 0.737 ± 0.021 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.024 ± 0.001 | 0.025 ± 0.001 | unsupported | 0.054 ± 0.002 | 0.122 ± 0.026 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.397 ± 0.058 | 0.425 ± 0.049 | unsupported | 1.070 ± 0.129 | 1.360 ± 0.045 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.015 ± 0.001 | 12.000 ± 0.015 | unsupported | 0.051 ± 0.000 | 0.091 ± 0.010 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.198 ± 0.008 | 11.999 ± 0.000 | unsupported | 0.436 ± 0.009 | 0.844 ± 0.027 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.062 ± 0.007 | 0.075 ± 0.005 | unsupported | 0.071 ± 0.003 | 0.191 ± 0.029 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.306 ± 0.019 | 1.284 ± 0.083 | unsupported | 0.887 ± 0.003 | 2.660 ± 0.032 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.089 ± 0.001 | 71.797 ± 0.172 | unsupported | 0.039 ± 0.001 | 0.188 ± 0.034 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 0.712 ± 0.006 | 101.121 ± 1.243 | unsupported | 0.310 ± 0.005 | 2.209 ± 0.057 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.074 ± 0.003 | 0.076 ± 0.002 | unsupported | 0.120 ± 0.001 | 0.231 ± 0.003 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 1.202 ± 0.016 | 1.226 ± 0.024 | unsupported | 2.123 ± 0.031 | 2.740 ± 0.010 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.036 ± 0.000 | 12.000 ± 0.016 | unsupported | 0.097 ± 0.001 | 0.130 ± 0.006 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 0.529 ± 0.007 | 11.999 ± 0.250 | unsupported | 0.822 ± 0.018 | 1.507 ± 0.034 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.611 ± 0.004 | 1.046 ± 0.007 | unsupported | 0.259 ± 0.001 | 0.124 ± 0.021 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 10.890 ± 0.414 | 19.578 ± 0.632 | unsupported | 5.867 ± 0.117 | 1.419 ± 0.049 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.412 ± 0.003 | 86.069 ± 0.217 | unsupported | 0.253 ± 0.003 | 0.105 ± 0.002 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 5.523 ± 0.025 | 113.614 ± 2.249 | unsupported | 2.350 ± 0.134 | 0.831 ± 0.035 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 44.875 ± 0.530 | - | unsupported | 43.409 ± 0.190 | - | 43.921 ± 0.224 | 43.810 ± 0.477 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 22.627 ± 0.173 | - | unsupported | 23.884 ± 1.484 | - | 43.942 ± 0.706 | 22.973 ± 0.416 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 34.212 ± 0.720 | - | unsupported | 28.808 ± 0.478 | - | 30.824 ± 0.204 | 30.566 ± 0.306 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 15.081 ± 0.316 | - | unsupported | 15.069 ± 0.400 | - | 31.530 ± 0.209 | 14.850 ± 0.632 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 20.504 ± 0.051 | - | unsupported | 28.969 ± 0.203 | - | 18.915 ± 0.130 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 15.287 ± 0.185 | - | unsupported | 15.017 ± 0.525 | - | 18.764 ± 0.172 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 45.214 ± 0.433 | - | unsupported | 42.816 ± 0.386 | - | 43.756 ± 0.398 | 44.068 ± 0.306 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 21.928 ± 0.336 | - | unsupported | 22.021 ± 0.680 | - | 43.726 ± 0.483 | 22.951 ± 0.335 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 20.865 ± 0.343 | - | unsupported | 20.296 ± 0.112 | - | 22.701 ± 0.145 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 22.561 ± 0.054 | - | unsupported | 9.982 ± 0.027 | - | 11.876 ± 0.070 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 20.799 ± 0.113 | - | unsupported | 20.313 ± 0.030 | - | 22.457 ± 0.154 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 22.579 ± 0.079 | - | unsupported | 10.012 ± 0.351 | - | 11.778 ± 0.160 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 45.010 ± 0.413 | - | unsupported | 43.704 ± 0.667 | - | 43.900 ± 0.332 | 43.805 ± 0.223 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 22.014 ± 0.071 | - | unsupported | 21.871 ± 0.128 | - | 44.243 ± 0.096 | 22.981 ± 0.190 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 32.343 ± 0.459 | - | unsupported | 28.837 ± 0.121 | - | 28.717 ± 0.115 | 29.602 ± 0.296 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 14.697 ± 0.123 | - | unsupported | 14.471 ± 0.361 | - | 28.981 ± 0.441 | 14.645 ± 0.308 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.268 ± 0.002 | unsupported | unsupported | 0.359 ± 0.005 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 4.781 ± 0.018 | unsupported | unsupported | 6.371 ± 0.038 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 211.503 ± 6.568 | unsupported | unsupported | 0.355 ± 0.007 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 424.875 ± 7.433 | unsupported | unsupported | 3.476 ± 0.042 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 45.021 ± 0.246 | - | unsupported | 43.746 ± 0.234 | - | 44.197 ± 0.507 | 43.843 ± 0.103 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 22.226 ± 0.154 | - | unsupported | 22.259 ± 0.193 | - | 44.410 ± 0.271 | 23.225 ± 1.808 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 76.278 ± 1.708 | - | 0.001 ± 0.000 | 73.885 ± 1.490 | 74.307 ± 2.245 | 87.824 ± 1.384 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 34.266 ± 0.368 | - | 0.001 ± 0.000 | 26.995 ± 0.750 | 44.798 ± 0.642 | 89.251 ± 5.385 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 52.120 ± 0.555 | - | 145.106 ± 1.331 | 53.397 ± 0.949 | 55.445 ± 0.497 | 58.125 ± 1.130 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 21.236 ± 0.278 | - | 179.449 ± 6.742 | 21.042 ± 0.959 | 30.482 ± 0.384 | 59.559 ± 1.365 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 152.833 ± 0.727 | - | 153.361 ± 0.324 | 151.237 ± 0.749 | 155.432 ± 2.196 | 170.678 ± 1.902 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 54.454 ± 0.709 | - | 155.077 ± 0.553 | 54.378 ± 0.806 | 86.583 ± 3.075 | 173.080 ± 1.307 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 52.719 ± 0.203 | - | 52.550 ± 0.678 | 52.663 ± 0.748 | 46.445 ± 0.252 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 28.842 ± 0.231 | - | 53.960 ± 1.003 | 22.947 ± 0.424 | 28.025 ± 0.434 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 146.634 ± 0.646 | - | 0.001 ± 0.000 | 87.575 ± 2.061 | 98.203 ± 2.136 | 152.937 ± 1.689 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 33.709 ± 0.390 | - | 0.001 ± 0.000 | 33.049 ± 0.947 | 38.681 ± 0.624 | 157.505 ± 0.542 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 77.388 ± 1.106 | - | 0.001 ± 0.000 | 61.207 ± 0.105 | 73.608 ± 2.046 | 204.111 ± 11.908 | 74.608 ± 1.339 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 28.058 ± 0.509 | - | 0.001 ± 0.000 | 60.379 ± 1.142 | 40.329 ± 0.652 | 199.387 ± 5.461 | 34.779 ± 1.747 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 41.297 ± 0.209 | - | 41.227 ± 0.381 | 48.785 ± 0.217 | 44.852 ± 0.722 | 67.327 ± 8.320 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 17.199 ± 0.055 | - | 48.594 ± 7.003 | 19.969 ± 0.493 | 25.919 ± 0.583 | 63.998 ± 0.451 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 41.356 ± 0.071 | - | 41.757 ± 0.143 | 44.833 ± 0.309 | 47.217 ± 1.594 | 61.874 ± 5.342 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 17.343 ± 0.161 | - | 56.558 ± 2.681 | 17.070 ± 0.329 | 25.558 ± 0.254 | 64.130 ± 1.654 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
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
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 5.732533 | 358283.31 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 4.434593 | 277162.06 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 4.671119 | 291944.94 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 5.027622 | 314226.38 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 3.450562 | 215660.12 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 1467.998957 | 91749934.81 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 18.301357 | 1143834.81 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 18.304110 | 1144006.88 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 9.425185 | 589074.06 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 323.723547 | 20232721.69 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 8.684993 | 542812.06 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 3.726755 | 232922.19 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 3.373836 | 210864.75 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 3.644756 | 227797.25 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 99.810233 | 6238139.56 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 74.501553 | 4656347.06 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 72.692763 | 4543297.69 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 75.148676 | 4696792.25 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 25.319423 | 1582463.94 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 2.954964 | 184685.25 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 3.234810 | 202175.62 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 335.959919 | 20997494.94 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 143.315765 | 8957235.31 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 41.771312 | 2610707.00 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 6781.825107 | 423864069.19 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 7136.994563 | 446062160.19 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 9.451965 | 590747.81 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 9.468604 | 591787.75 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 209.289510 | 13080594.38 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 208.268004 | 13016750.25 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 15.363555 | 960222.19 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 15.575917 | 973494.81 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 303.928899 | 18995556.19 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 301.844554 | 18865284.62 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 8.751658 | 546978.62 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 8.814335 | 550895.94 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 200.430439 | 12526902.44 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 199.722507 | 12482656.69 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 14.615737 | 913483.56 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 14.794075 | 924629.69 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 291.033543 | 18189596.44 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 294.151616 | 18384476.00 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 1.458546 | 22789.78 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.424956 | 69139.94 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.061413 | 63459.58 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.039337 | 63114.64 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.355763 | 794470.38 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 8.617152 | 1077144.00 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 8.355704 | 1044463.00 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 8.471954 | 1058994.25 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.171513 | 80804.89 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.544707 | 39761.05 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.135764 | 33371.31 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1279.957555 | 19999336.80 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.356987 | 544623.38 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.946376 | 493297.00 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.766410 | 470801.25 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 164.390374 | 20548796.75 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.427055 | 147297.73 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.182975 | 65358.98 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.962201 | 30659.39 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.988921 | 31076.89 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 15.433116 | 1929139.50 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.711538 | 963942.25 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.098667 | 512333.38 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.254873 | 531859.12 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.186689 | 96667.02 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.251989 | 50812.33 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.161844 | 18153.81 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.978866 | 11999669.78 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.580783 | 822597.88 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.505999 | 438249.88 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.047835 | 255979.38 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.993083 | 11999135.38 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.903119 | 107861.23 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.609637 | 134525.58 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.680305 | 104379.77 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.633053 | 103641.45 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.399141 | 1424892.62 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 19.101411 | 2387676.38 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 14.111526 | 1763940.75 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 13.510825 | 1688853.12 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.083700 | 95057.81 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.104984 | 64140.38 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.872862 | 44888.47 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.979955 | 11999686.80 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.162034 | 895254.25 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.714448 | 964306.00 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.736362 | 717045.25 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.991714 | 11998964.25 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.386425 | 115412.89 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.734059 | 105219.67 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.759866 | 43122.91 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.909806 | 61090.72 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.143092 | 1767886.50 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.526176 | 1190772.00 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.516505 | 939563.12 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.834477 | 979309.62 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.996726 | 156198.84 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.135513 | 48992.39 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.428815 | 69200.23 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3328.970471 | 52015163.61 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.452711 | 1181588.88 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.059674 | 507459.25 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.332893 | 541611.62 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 642.552629 | 80319078.62 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.323899 | 192560.92 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.941245 | 77206.95 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.866802 | 60418.78 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.502801 | 70356.27 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 22.951486 | 2868935.75 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.668969 | 1208621.12 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.322716 | 1290339.50 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.532888 | 1316611.00 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.339623 | 177181.61 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.609385 | 40771.64 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.830155 | 75471.17 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4600.970478 | 71890163.72 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 17.943458 | 2242932.25 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.052202 | 506525.25 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.815862 | 726982.75 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 801.395372 | 100174421.50 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.784495 | 231007.73 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 12.572419 | 196444.05 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 13.972345 | 218317.89 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 33.333665 | 4166708.13 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 29.081024 | 3635128.00 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 29.366225 | 3670778.12 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 16.973376 | 265209.00 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 11.289002 | 176390.66 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5876.973854 | 91827716.47 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 18.657910 | 2332238.75 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 12.547663 | 1568457.88 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 966.971783 | 120871472.88 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 1.253041 | 19578.77 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.566200 | 8846.88 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.590386 | 9224.78 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 0.850454 | 13288.34 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 1.826362 | 228295.25 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 1.233259 | 154157.38 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 1.275987 | 159498.38 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 1.342237 | 167779.62 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 1.240371 | 19380.80 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.627351 | 9802.36 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.591506 | 9242.28 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1279.567463 | 19993241.61 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.248829 | 156103.62 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.399436 | 49929.50 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.634399 | 79299.88 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 159.995904 | 19999488.00 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.388170 | 131065.16 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.596944 | 56202.25 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.636929 | 25577.02 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.274956 | 35546.19 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.673410 | 1584176.25 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.167036 | 1145879.50 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.770753 | 471344.12 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.059218 | 507402.25 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.938925 | 108420.70 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.065595 | 63524.92 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.820126 | 28439.47 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1661.969399 | 25968271.86 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.359859 | 1419982.38 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.729056 | 591132.00 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.120688 | 265086.00 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 324.473207 | 40559150.88 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.048216 | 110128.38 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.504664 | 54760.38 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.506719 | 23542.48 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.577606 | 24650.09 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.366838 | 1295854.75 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 8.461782 | 1057722.75 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.993460 | 374182.50 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.187488 | 398436.00 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.804324 | 90692.56 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.236310 | 50567.34 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.871060 | 13610.31 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.979802 | 11999684.41 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.893596 | 736699.50 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.490664 | 436333.00 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.509325 | 188665.62 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.992197 | 11999024.62 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.823809 | 122247.02 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.450063 | 53907.23 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.520966 | 23765.09 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.607096 | 25110.88 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.877234 | 1359654.25 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 8.563180 | 1070397.50 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.176036 | 397004.50 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.397637 | 424704.62 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.844458 | 91319.66 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.291987 | 51437.30 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.951071 | 14860.48 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.980258 | 11999691.53 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.754693 | 844336.62 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.484980 | 435622.50 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.583217 | 197902.12 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.992613 | 11999076.62 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.212186 | 190815.41 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.549913 | 71092.39 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.964126 | 61939.47 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.804488 | 75070.12 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 21.279990 | 2659998.75 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.098028 | 887253.50 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.450337 | 1306292.12 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.268712 | 1283589.00 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.005034 | 187578.66 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.517606 | 39337.59 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.677378 | 88709.03 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4594.997954 | 71796843.03 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 17.669751 | 2208718.88 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.478353 | 309794.12 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.695923 | 711990.38 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 808.965904 | 101120738.00 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.773914 | 230842.41 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.708945 | 120452.27 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.745949 | 74155.45 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.880045 | 76250.70 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 21.917263 | 2739657.88 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 16.986233 | 2123279.12 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.615480 | 1201935.00 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 9.805269 | 1225658.62 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.342322 | 130348.78 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.214487 | 97101.36 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.312362 | 36130.66 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.978199 | 11999659.36 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 12.055034 | 1506879.25 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.575391 | 821923.88 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.235461 | 529432.62 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.994079 | 11999259.88 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.955379 | 124302.80 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 16.578244 | 259035.06 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 39.078054 | 610594.59 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 66.962089 | 1046282.64 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.350209 | 1418776.12 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 46.935022 | 5866877.75 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 87.122970 | 10890371.25 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 156.620997 | 19577624.62 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.697824 | 104653.50 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 16.182036 | 252844.31 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 26.371252 | 412050.81 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5508.420568 | 86069071.38 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.647603 | 830950.38 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 18.797211 | 2349651.38 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 44.187000 | 5523375.00 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 908.908582 | 113613572.75 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 5.749373 | 359335.81 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 4.281025 | 267564.06 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 101.942029 | 6371376.81 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 76.491002 | 4780687.62 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 5.673135 | 354570.94 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 3384.052615 | 211503288.44 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 55.620160 | 3476260.00 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 6797.995458 | 424874716.12 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.013346 | 834.12 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.001552 | 97.00 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.014566 | 910.38 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.001612 | 100.75 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000134 | 8.38 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.012394 | 774.62 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.006325 | 395.31 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000129 | 8.06 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.013761 | 860.06 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.006480 | 405.00 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000463 | 28.94 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.013094 | 818.38 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.004325 | 270.31 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000128 | 8.00 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.012717 | 794.81 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.005245 | 327.81 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000095 | 5.94 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.012344 | 771.50 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003118 | 194.88 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000073 | 4.56 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.013069 | 816.81 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003531 | 220.69 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/norm_fro` (c64, threads=1, shape=`2048x1536`): `julia-base` is 10.1x faster than the slowest successful cell (`tenferro-trace`, 24.327 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=1, shape=`2048x1536`): `tenferro-direct` is 13.6x faster than the slowest successful cell (`tenferro-trace`, 24.327 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `jax-cpu` is 23.5x faster than the slowest successful cell (`tenferro-trace`, 62.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `julia-base` is 11.2x faster than the slowest successful cell (`tenferro-trace`, 62.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `pytorch-cpu` is 10.7x faster than the slowest successful cell (`tenferro-trace`, 62.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 71.2x faster than the slowest successful cell (`tenferro-trace`, 62.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 112.7x faster than the slowest successful cell (`tenferro-trace`, 337.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `julia-base` is 129.0x faster than the slowest successful cell (`tenferro-trace`, 337.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 184.9x faster than the slowest successful cell (`tenferro-trace`, 337.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/chain_log1p_exp_mul` (f64, threads=4, shape=`4194304`): `tenferro-direct` is 14.4x faster than the slowest successful cell (`tenferro-trace`, 100.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/exp` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 12.8x faster than the slowest successful cell (`julia-base`, 50.233 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`julia-base`, 62.576 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log1p` (f64, threads=4, shape=`4194304`): `tenferro-direct` is 13.0x faster than the slowest successful cell (`tenferro-trace`, 35.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/masked_log_softmax_axis1` (f64, threads=4, shape=`1024x64`): `jax-cpu` is 292.0x faster than the slowest successful cell (`tenferro-eager`, 91.750 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/masked_log_softmax_axis1` (f64, threads=4, shape=`1024x64`): `tenferro-direct` is 425.4x faster than the slowest successful cell (`tenferro-eager`, 91.750 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 16.5x faster than the slowest successful cell (`pytorch-cpu`, 25.634 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 18.7x faster than the slowest successful cell (`pytorch-cpu`, 25.634 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 15.0x faster than the slowest successful cell (`pytorch-cpu`, 25.634 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 15.1x faster than the slowest successful cell (`pytorch-cpu`, 25.634 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 21.8x faster than the slowest successful cell (`tenferro-trace`, 20.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 16.0x faster than the slowest successful cell (`tenferro-trace`, 20.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 26.3x faster than the slowest successful cell (`tenferro-trace`, 20.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 19.1x faster than the slowest successful cell (`tenferro-trace`, 21.918 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 23.1x faster than the slowest successful cell (`tenferro-trace`, 21.918 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 36.9x faster than the slowest successful cell (`tenferro-trace`, 21.918 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 16.5x faster than the slowest successful cell (`pytorch-cpu`, 25.009 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 18.3x faster than the slowest successful cell (`pytorch-cpu`, 25.009 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 14.8x faster than the slowest successful cell (`pytorch-cpu`, 25.009 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 25.009 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 25.4x faster than the slowest successful cell (`tenferro-trace`, 21.882 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 17.3x faster than the slowest successful cell (`tenferro-trace`, 21.882 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 29.1x faster than the slowest successful cell (`tenferro-trace`, 21.882 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 28.8x faster than the slowest successful cell (`tenferro-trace`, 20.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 18.5x faster than the slowest successful cell (`tenferro-trace`, 20.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 24.4x faster than the slowest successful cell (`tenferro-trace`, 20.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 35.7x faster than the slowest successful cell (`tenferro-trace`, 20.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 30.9x faster than the slowest successful cell (`tenferro-trace`, 20.870 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 18.2x faster than the slowest successful cell (`tenferro-trace`, 20.870 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 34.9x faster than the slowest successful cell (`tenferro-trace`, 20.870 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 37.3x faster than the slowest successful cell (`tenferro-trace`, 20.870 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 27.7x faster than the slowest successful cell (`tenferro-trace`, 19.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 16.7x faster than the slowest successful cell (`tenferro-trace`, 19.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 21.7x faster than the slowest successful cell (`tenferro-trace`, 19.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 33.2x faster than the slowest successful cell (`tenferro-trace`, 19.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 16.4x faster than the slowest successful cell (`tenferro-trace`, 21.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 18.5x faster than the slowest successful cell (`tenferro-trace`, 21.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 35.2x faster than the slowest successful cell (`tenferro-trace`, 21.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 39.1x faster than the slowest successful cell (`tenferro-trace`, 21.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_squares_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 34.3x faster than the slowest successful cell (`tenferro-eager`, 20.233 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sin` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 11.7x faster than the slowest successful cell (`julia-base`, 56.588 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `jax-cpu` is 10.3x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `pytorch-cpu` is 19.7x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `tenferro-direct` is 20.5x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `jax-cpu` is 18.8x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `pytorch-cpu` is 20.2x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `tenferro-direct` is 20.6x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `jax-cpu` is 175.1x faster than the slowest successful cell (`tenferro-trace`, 40.432 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `julia-base` is 42.0x faster than the slowest successful cell (`tenferro-trace`, 40.432 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 93.1x faster than the slowest successful cell (`tenferro-trace`, 40.432 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `tenferro-direct` is 112.8x faster than the slowest successful cell (`tenferro-trace`, 40.432 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `jax-cpu` is 18.8x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 13.3x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 20.6x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `jax-cpu` is 10.3x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 20.0x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 20.7x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `jax-cpu` is 40.4x faster than the slowest successful cell (`tenferro-trace`, 32.939 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `julia-base` is 24.6x faster than the slowest successful cell (`tenferro-trace`, 32.939 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 49.0x faster than the slowest successful cell (`tenferro-trace`, 32.939 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `tenferro-direct` is 37.3x faster than the slowest successful cell (`tenferro-trace`, 32.939 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 5654.6x faster than the slowest successful cell (`julia-base`, 6.514 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 5481.6x faster than the slowest successful cell (`julia-base`, 6.397 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 16.6x faster than the slowest successful cell (`jax-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `pytorch-cpu` is 11.9x faster than the slowest successful cell (`jax-cpu`, 0.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 32.6x faster than the slowest successful cell (`jax-cpu`, 0.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 16.3x faster than the slowest successful cell (`jax-cpu`, 0.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `pytorch-cpu` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 22.2x faster than the slowest successful cell (`jax-cpu`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-trace` is 13.5x faster than the slowest successful cell (`jax-cpu`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 14.2x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.025 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 13.4x faster than the slowest successful cell (`jax-cpu`, 0.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 14.8x faster than the slowest successful cell (`jax-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 18.2x faster than the slowest successful cell (`jax-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-trace` is 13.3x faster than the slowest successful cell (`jax-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 16.2x faster than the slowest successful cell (`jax-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 17.5x faster than the slowest successful cell (`jax-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 12.2x faster than the slowest successful cell (`jax-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x32x32,rhs=1`): `tenferro-direct` is 11.6x faster than the slowest successful cell (`jax-cpu`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x4x4,rhs=1`): `jax-cpu` is 23.5x faster than the slowest successful cell (`tenferro-trace`, 6.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x4x4,rhs=1`): `pytorch-cpu` is 44.0x faster than the slowest successful cell (`tenferro-trace`, 6.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x4x4,rhs=1`): `tenferro-direct` is 45.1x faster than the slowest successful cell (`tenferro-trace`, 6.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `jax-cpu` is 26.4x faster than the slowest successful cell (`tenferro-trace`, 20.298 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `pytorch-cpu` is 53.6x faster than the slowest successful cell (`tenferro-trace`, 20.298 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `tenferro-direct` is 43.4x faster than the slowest successful cell (`tenferro-trace`, 20.298 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x128x128,rhs=1`): `jax-cpu` is 55.1x faster than the slowest successful cell (`tenferro-trace`, 6.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x128x128,rhs=1`): `pytorch-cpu` is 70.7x faster than the slowest successful cell (`tenferro-trace`, 6.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x128x128,rhs=1`): `tenferro-direct` is 109.5x faster than the slowest successful cell (`tenferro-trace`, 6.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `jax-cpu` is 27.9x faster than the slowest successful cell (`tenferro-trace`, 19.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `pytorch-cpu` is 61.9x faster than the slowest successful cell (`tenferro-trace`, 19.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `tenferro-direct` is 45.8x faster than the slowest successful cell (`tenferro-trace`, 19.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 78.9x faster than the slowest successful cell (`tenferro-trace`, 816.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `julia-base` is 89.4x faster than the slowest successful cell (`tenferro-trace`, 816.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 141.2x faster than the slowest successful cell (`tenferro-trace`, 816.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`256x128`): `jax-cpu` is 13.3x faster than the slowest successful cell (`tenferro-eager`, 20.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`256x128`): `pytorch-cpu` is 113.7x faster than the slowest successful cell (`tenferro-eager`, 20.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`256x128`): `tenferro-direct` is 103.9x faster than the slowest successful cell (`tenferro-eager`, 20.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`768x384`): `jax-cpu` is 49.8x faster than the slowest successful cell (`tenferro-eager`, 446.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 170.9x faster than the slowest successful cell (`tenferro-eager`, 446.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 26.2x faster than the slowest successful cell (`tenferro-trace`, 479.668 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `julia-base` is 37.7x faster than the slowest successful cell (`tenferro-trace`, 479.668 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 57.8x faster than the slowest successful cell (`tenferro-trace`, 479.668 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `jax-cpu` is 11.4x faster than the slowest successful cell (`tenferro-eager`, 460.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `julia-base` is 108.8x faster than the slowest successful cell (`tenferro-eager`, 460.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 157.9x faster than the slowest successful cell (`tenferro-eager`, 460.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 29.1x faster than the slowest successful cell (`tenferro-direct`, 726.232 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `julia-base` is 37.6x faster than the slowest successful cell (`tenferro-direct`, 726.232 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 66.7x faster than the slowest successful cell (`tenferro-direct`, 726.232 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `julia-base` is 13.4x faster than the slowest successful cell (`tenferro-trace`, 20.979 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 18.1x faster than the slowest successful cell (`tenferro-trace`, 20.979 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 36.1x faster than the slowest successful cell (`tenferro-trace`, 20.979 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `jax-cpu` is 15.0x faster than the slowest successful cell (`tenferro-trace`, 241.304 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `julia-base` is 15.2x faster than the slowest successful cell (`tenferro-trace`, 241.304 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `jax-cpu` is 14.3x faster than the slowest successful cell (`tenferro-trace`, 235.704 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `julia-base` is 15.2x faster than the slowest successful cell (`tenferro-trace`, 235.704 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 72.5x faster than the slowest successful cell (`tenferro-trace`, 781.898 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `julia-base` is 85.9x faster than the slowest successful cell (`tenferro-trace`, 781.898 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 134.9x faster than the slowest successful cell (`tenferro-trace`, 781.898 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 13.9x faster than the slowest successful cell (`tenferro-trace`, 650.195 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `julia-base` is 13.6x faster than the slowest successful cell (`tenferro-trace`, 650.195 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/triangular_solve` (f64, threads=4, shape=`4096x4096,rhs=64`): `pytorch-cpu` is 13.8x faster than the slowest successful cell (`jax-cpu`, 100.286 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 247.5x faster than the slowest successful cell (`tenferro-eager`, 19.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 503.0x faster than the slowest successful cell (`tenferro-eager`, 19.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 599.3x faster than the slowest successful cell (`tenferro-eager`, 19.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 37.7x faster than the slowest successful cell (`tenferro-eager`, 20.549 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 41.7x faster than the slowest successful cell (`tenferro-eager`, 20.549 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 43.6x faster than the slowest successful cell (`tenferro-eager`, 20.549 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 124.1x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 236.2x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 661.0x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 14.6x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 27.4x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 46.9x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 126.2x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 187.1x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 267.3x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 13.4x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 12.4x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 16.7x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 333.0x faster than the slowest successful cell (`tenferro-eager`, 52.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 1061.7x faster than the slowest successful cell (`tenferro-eager`, 52.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 751.7x faster than the slowest successful cell (`tenferro-eager`, 52.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 68.0x faster than the slowest successful cell (`tenferro-eager`, 80.319 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 158.3x faster than the slowest successful cell (`tenferro-eager`, 80.319 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 148.3x faster than the slowest successful cell (`tenferro-eager`, 80.319 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 405.7x faster than the slowest successful cell (`tenferro-eager`, 71.890 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 1763.2x faster than the slowest successful cell (`tenferro-eager`, 71.890 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 952.6x faster than the slowest successful cell (`tenferro-eager`, 71.890 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 44.7x faster than the slowest successful cell (`tenferro-eager`, 100.174 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 197.8x faster than the slowest successful cell (`tenferro-eager`, 100.174 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 137.8x faster than the slowest successful cell (`tenferro-eager`, 100.174 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 346.2x faster than the slowest successful cell (`tenferro-eager`, 91.828 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 520.6x faster than the slowest successful cell (`tenferro-eager`, 91.828 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 51.8x faster than the slowest successful cell (`tenferro-eager`, 120.871 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 77.1x faster than the slowest successful cell (`tenferro-eager`, 120.871 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 1031.6x faster than the slowest successful cell (`tenferro-eager`, 19.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 2039.7x faster than the slowest successful cell (`tenferro-eager`, 19.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 2163.3x faster than the slowest successful cell (`tenferro-eager`, 19.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 128.1x faster than the slowest successful cell (`tenferro-eager`, 19.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 400.6x faster than the slowest successful cell (`tenferro-eager`, 19.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 252.2x faster than the slowest successful cell (`tenferro-eager`, 19.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 239.5x faster than the slowest successful cell (`tenferro-eager`, 25.968 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 408.8x faster than the slowest successful cell (`tenferro-eager`, 25.968 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 913.1x faster than the slowest successful cell (`tenferro-eager`, 25.968 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 28.6x faster than the slowest successful cell (`tenferro-eager`, 40.559 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 68.6x faster than the slowest successful cell (`tenferro-eager`, 40.559 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 153.0x faster than the slowest successful cell (`tenferro-eager`, 40.559 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 132.3x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 237.3x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 881.7x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 16.3x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 27.5x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 63.6x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 131.4x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 233.3x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 807.5x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 14.2x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 27.5x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 60.6x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 382.8x faster than the slowest successful cell (`tenferro-eager`, 71.797 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 1825.1x faster than the slowest successful cell (`tenferro-eager`, 71.797 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 809.4x faster than the slowest successful cell (`tenferro-eager`, 71.797 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 45.8x faster than the slowest successful cell (`tenferro-eager`, 101.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 326.4x faster than the slowest successful cell (`tenferro-eager`, 101.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 142.0x faster than the slowest successful cell (`tenferro-eager`, 101.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 92.1x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 123.6x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 332.1x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 14.6x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 22.7x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 13.8x faster than the slowest successful cell (`tenferro-eager`, 19.578 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 822.4x faster than the slowest successful cell (`tenferro-eager`, 86.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 340.4x faster than the slowest successful cell (`tenferro-eager`, 86.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 208.9x faster than the slowest successful cell (`tenferro-eager`, 86.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 136.7x faster than the slowest successful cell (`tenferro-eager`, 113.614 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 48.4x faster than the slowest successful cell (`tenferro-eager`, 113.614 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 20.6x faster than the slowest successful cell (`tenferro-eager`, 113.614 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/output_reuse/solve_read_into` (f64, threads=4, shape=`256x256,rhs=16`): `pytorch-cpu` is 596.5x faster than the slowest successful cell (`tenferro-direct`, 211.503 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/output_reuse/solve_read_into` (f64, threads=4, shape=`768x768,rhs=16`): `pytorch-cpu` is 122.2x faster than the slowest successful cell (`tenferro-direct`, 424.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 79694.9x faster than the slowest successful cell (`julia-base`, 87.824 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 81731.7x faster than the slowest successful cell (`julia-base`, 89.251 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 132412.7x faster than the slowest successful cell (`julia-base`, 152.937 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 136486.5x faster than the slowest successful cell (`julia-base`, 157.505 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 190758.2x faster than the slowest successful cell (`julia-base`, 204.111 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 193017.2x faster than the slowest successful cell (`julia-base`, 199.387 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 96.9x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 107.5x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 28.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 99.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 128.7x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 163.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 14556.4 GB/s at 0.001 ms. The conservative physical lower bound is 0.168 ms, so the cell is 145.6x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 14369.3 GB/s at 0.001 ms. The conservative physical lower bound is 0.042 ms, so the cell is 35.9x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 243589.3 GB/s at 0.001 ms. The conservative physical lower bound is 2.684 ms, so the cell is 2435.9x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 245820.0 GB/s at 0.001 ms. The conservative physical lower bound is 0.671 ms, so the cell is 614.6x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 232411.7 GB/s at 0.001 ms. The conservative physical lower bound is 2.684 ms, so the cell is 2324.1x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 232613.0 GB/s at 0.001 ms. The conservative physical lower bound is 0.671 ms, so the cell is 581.5x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 125437.1 GB/s at 0.001 ms. The conservative physical lower bound is 1.342 ms, so the cell is 1254.4x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 129930.0 GB/s at 0.001 ms. The conservative physical lower bound is 0.336 ms, so the cell is 324.8x faster than the bound.

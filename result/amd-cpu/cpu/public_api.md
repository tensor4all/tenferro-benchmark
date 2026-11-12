# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/amd-cpu/cpu/public_api/20261112_191747/run.yaml`
- Timestamp: `2026-11-12T19:17:47.791904+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/amd-cpu/cpu/public_api/20261112_191747`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

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
- Python platform: `Linux-7.0.0-1019-azure-x86_64-with-glibc2.39`
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

- CSV: `data/results/amd-cpu/cpu/public_api/20261112_191747/cpu_public_api_t1_20261112_191747.csv`
- CSV: `data/results/amd-cpu/cpu/public_api/20261112_191747/cpu_public_api_t4_20261112_191747.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 4.248 ± 0.030 | - | 4.134 ± 0.016 | 4.547 ± 0.039 | 3.908 ± 0.036 | 3.726 ± 0.095 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 4.493 ± 0.037 | - | 4.278 ± 0.160 | 2.254 ± 0.043 | 3.239 ± 0.065 | 3.521 ± 0.339 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 122.833 ± 3.356 | - | 121.821 ± 2.939 | 121.852 ± 1.314 | 125.452 ± 2.294 | 144.712 ± 2.263 | 145.493 ± 2.968 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 40.829 ± 0.550 | - | 142.701 ± 2.044 | 40.546 ± 1.122 | 64.096 ± 0.616 | 143.468 ± 2.655 | 63.016 ± 3.609 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 105.397 ± 1.307 | - | 109.335 ± 1.652 | 92.779 ± 1.073 | 91.083 ± 1.278 | 127.421 ± 1.697 | 179.405 ± 2.372 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 42.746 ± 1.190 | - | 130.445 ± 3.247 | 30.965 ± 0.313 | 44.259 ± 0.213 | 125.760 ± 1.067 | 63.306 ± 2.752 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 43.254 ± 0.123 | - | 43.333 ± 0.190 | 42.209 ± 0.211 | 65.700 ± 0.212 | 46.600 ± 0.255 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 46.540 ± 0.582 | - | 46.366 ± 0.191 | 28.779 ± 0.332 | 29.956 ± 0.635 | 23.202 ± 0.645 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 44.071 ± 0.065 | - | unsupported | 42.729 ± 0.097 | 66.377 ± 0.300 | 44.663 ± 6.168 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 47.542 ± 0.592 | - | unsupported | 29.191 ± 0.875 | 30.235 ± 1.055 | 23.260 ± 3.908 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 10.891 ± 0.021 | - | 10.932 ± 0.140 | 11.606 ± 0.228 | 11.866 ± 0.010 | 11.908 ± 0.082 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 14.219 ± 0.108 | - | 14.259 ± 0.069 | 14.000 ± 0.058 | 17.447 ± 0.078 | 17.465 ± 0.048 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 70.971 ± 0.251 | - | 71.129 ± 0.229 | 106.857 ± 1.235 | 104.808 ± 1.400 | 84.585 ± 7.640 | 88.766 ± 23.511 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 31.063 ± 0.093 | - | 83.999 ± 0.015 | 43.021 ± 0.303 | 52.150 ± 0.812 | 83.004 ± 0.524 | 42.579 ± 10.433 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 115.582 ± 0.098 | - | 115.703 ± 0.170 | 153.222 ± 0.692 | 176.404 ± 1.064 | 225.499 ± 0.969 | 151.388 ± 1.912 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 49.439 ± 1.030 | - | 131.997 ± 0.004 | 63.313 ± 0.469 | 71.699 ± 0.844 | 149.261 ± 1.747 | 62.624 ± 5.378 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 70.301 ± 1.291 | - | 69.506 ± 1.199 | 68.198 ± 0.990 | 70.623 ± 1.079 | 76.628 ± 0.572 | 77.943 ± 0.840 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 24.865 ± 0.310 | - | 92.580 ± 2.406 | 23.010 ± 0.691 | 35.528 ± 0.906 | 76.971 ± 3.092 | 35.050 ± 1.922 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 2.241 ± 0.138 | - | 20.692 ± 0.021 | 24.152 ± 0.101 | 16.832 ± 0.273 | 4.139 ± 0.072 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 0.925 ± 0.053 | - | 59.998 ± 0.006 | 11.280 ± 0.014 | 2.483 ± 0.330 | 4.088 ± 0.016 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 8.626 ± 0.059 | - | 8.605 ± 0.064 | 9.879 ± 0.035 | 7.597 ± 0.037 | 8.997 ± 0.115 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 140.009 ± 6.014 | - | 124.998 ± 14.763 | 5.678 ± 0.048 | 6.833 ± 0.031 | 7.341 ± 0.058 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 5.444 ± 0.120 | - | 5.687 ± 0.351 | 5.735 ± 0.031 | 5.100 ± 0.027 | 4.922 ± 0.087 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 92.009 ± 11.915 | - | 145.999 ± 14.987 | 3.612 ± 0.040 | 4.584 ± 0.481 | 4.272 ± 0.175 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 6.529 ± 0.052 | - | 6.514 ± 0.076 | 7.617 ± 0.041 | 5.483 ± 0.023 | 5.385 ± 0.177 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 11.730 ± 0.131 | - | 11.875 ± 0.217 | 5.781 ± 0.010 | 5.972 ± 0.043 | 6.180 ± 0.606 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 43.286 ± 0.172 | - | 45.880 ± 3.332 | 42.260 ± 0.122 | 65.712 ± 0.354 | 44.016 ± 0.679 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 46.149 ± 0.423 | - | 58.380 ± 14.315 | 28.955 ± 0.648 | 29.706 ± 0.433 | 22.335 ± 0.339 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 48.996 ± 0.136 | - | unsupported | 50.025 ± 0.423 | 61.747 ± 5.953 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 53.689 ± 0.597 | - | unsupported | 25.133 ± 0.199 | 34.867 ± 0.624 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 121.691 ± 4.127 | - | 119.815 ± 0.772 | 119.825 ± 1.590 | 123.728 ± 2.311 | 138.980 ± 3.887 | 155.231 ± 3.719 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 40.774 ± 0.385 | - | 138.401 ± 12.054 | 40.694 ± 0.514 | 64.017 ± 1.110 | 137.318 ± 4.114 | 67.032 ± 0.761 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 130.132 ± 4.088 | - | 130.505 ± 4.427 | 131.962 ± 3.244 | 136.063 ± 1.142 | 143.335 ± 3.625 | 160.924 ± 2.195 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 45.696 ± 1.178 | - | 143.825 ± 9.586 | 45.749 ± 1.590 | 69.850 ± 0.534 | 144.367 ± 3.873 | 71.737 ± 0.272 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 27.073 ± 0.177 | - | 59.543 ± 0.290 | 75.698 ± 0.602 | 50.599 ± 0.190 | 81.033 ± 8.965 | 85.193 ± 10.618 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 12.496 ± 0.196 | - | 119.997 ± 0.009 | 32.388 ± 0.393 | 26.104 ± 0.390 | 87.934 ± 14.677 | 32.612 ± 13.219 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 9.995 ± 0.119 | - | 9.904 ± 0.180 | 36.582 ± 1.041 | 37.515 ± 0.567 | 43.410 ± 9.695 | 40.633 ± 12.655 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 7.183 ± 0.291 | - | 23.998 ± 0.006 | 12.423 ± 0.364 | 19.803 ± 0.606 | 46.257 ± 10.838 | 14.741 ± 20.691 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 21.777 ± 0.821 | - | 21.420 ± 1.003 | 46.613 ± 0.918 | 41.257 ± 1.325 | 20.742 ± 0.635 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 13.393 ± 0.224 | - | 35.997 ± 0.008 | 19.542 ± 0.317 | 24.459 ± 0.860 | 22.474 ± 0.411 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 28.260 ± 0.077 | - | 28.430 ± 0.081 | 56.063 ± 0.767 | 106.945 ± 0.914 | 77.938 ± 15.033 | 71.493 ± 12.662 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 11.964 ± 0.068 | - | 48.000 ± 0.008 | 22.078 ± 0.996 | 54.373 ± 0.365 | 67.679 ± 22.821 | 42.616 ± 12.939 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 133.017 ± 2.402 | - | 132.489 ± 4.329 | 129.766 ± 1.125 | 136.834 ± 0.454 | 149.690 ± 4.389 | 162.614 ± 4.512 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 45.347 ± 0.760 | - | 149.550 ± 12.658 | 45.474 ± 2.175 | 70.278 ± 0.865 | 148.169 ± 4.173 | 71.534 ± 1.229 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 20.442 ± 0.038 | - | 20.784 ± 0.112 | 43.742 ± 0.102 | 40.649 ± 1.012 | 61.961 ± 14.212 | 67.332 ± 12.472 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 8.401 ± 0.020 | - | 35.994 ± 0.020 | 16.671 ± 0.327 | 21.137 ± 0.393 | 71.773 ± 10.089 | 39.327 ± 12.554 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 20.596 ± 0.029 | - | 20.360 ± 0.030 | 31.936 ± 0.578 | 27.487 ± 0.214 | 37.801 ± 9.644 | 40.793 ± 10.299 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 8.096 ± 0.018 | - | 35.999 ± 0.009 | 11.512 ± 0.266 | 14.276 ± 0.691 | 37.666 ± 0.191 | 16.239 ± 12.192 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 25.670 ± 0.030 | - | 25.506 ± 0.081 | 50.115 ± 0.173 | 64.972 ± 0.620 | 84.047 ± 22.514 | 96.565 ± 8.012 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 10.601 ± 0.019 | - | 47.999 ± 0.007 | 18.835 ± 0.349 | 34.277 ± 0.243 | 83.806 ± 14.931 | 38.030 ± 20.738 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 13.188 ± 0.042 | - | 13.208 ± 0.071 | 32.490 ± 0.550 | 36.598 ± 0.260 | 46.037 ± 10.769 | 49.760 ± 8.933 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 5.645 ± 0.028 | - | 35.995 ± 0.006 | 11.670 ± 0.244 | 18.755 ± 0.182 | 46.063 ± 0.301 | 23.597 ± 14.009 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.449 ± 0.009 | 0.487 ± 0.009 | unsupported | unsupported | 0.486 ± 0.005 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.391 ± 0.008 | 51.999 ± 0.375 | unsupported | unsupported | 0.225 ± 0.007 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 130.880 ± 0.827 | - | 135.627 ± 2.162 | 130.462 ± 2.058 | 134.565 ± 0.971 | 149.193 ± 3.500 | 165.246 ± 3.475 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 45.319 ± 0.796 | - | 153.501 ± 2.455 | 44.688 ± 0.776 | 70.143 ± 0.986 | 151.375 ± 0.921 | 71.067 ± 4.388 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 133.124 ± 5.553 | - | 137.579 ± 3.928 | 131.216 ± 2.879 | 135.348 ± 1.332 | 152.314 ± 3.461 | 163.019 ± 4.797 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 44.997 ± 0.374 | - | 153.567 ± 2.933 | 44.737 ± 1.333 | 70.268 ± 0.606 | 151.418 ± 18.687 | 72.490 ± 2.672 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 130.545 ± 3.818 | - | 131.901 ± 2.675 | 130.292 ± 1.381 | 135.913 ± 1.404 | 145.606 ± 1.763 | 163.623 ± 6.050 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 45.825 ± 0.893 | - | 150.185 ± 12.366 | 45.450 ± 0.992 | 70.347 ± 0.381 | 146.532 ± 0.935 | 71.575 ± 4.107 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 119.129 ± 3.724 | - | 121.190 ± 1.094 | 120.870 ± 1.664 | 122.503 ± 0.762 | 139.539 ± 3.120 | 152.991 ± 4.995 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 40.587 ± 0.283 | - | 143.502 ± 3.447 | 40.862 ± 0.940 | 64.369 ± 0.514 | 137.776 ± 5.598 | 65.411 ± 9.939 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 57.803 ± 0.156 | - | 57.751 ± 0.057 | 62.796 ± 1.178 | 73.784 ± 0.193 | 125.315 ± 13.205 | 123.015 ± 11.781 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 23.534 ± 0.429 | - | 71.993 ± 11.999 | 22.041 ± 2.152 | 37.018 ± 0.069 | 125.133 ± 0.214 | 49.808 ± 11.335 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 16.494 ± 0.178 | - | 16.369 ± 0.380 | 12.261 ± 0.128 | 26.089 ± 0.271 | 13.633 ± 0.333 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 6.688 ± 0.213 | - | 36.004 ± 0.016 | 6.299 ± 0.139 | 14.257 ± 0.419 | 15.851 ± 0.602 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 2.012 ± 0.082 | - | 1.988 ± 0.128 | 24.217 ± 0.177 | 2.387 ± 0.023 | 1.660 ± 0.135 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.786 ± 0.007 | - | 23.996 ± 0.010 | 10.881 ± 0.522 | 1.369 ± 0.020 | 1.459 ± 0.062 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 2.456 ± 0.066 | - | 2.428 ± 0.043 | 4.077 ± 0.039 | 2.747 ± 0.450 | 1.683 ± 0.073 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 1.311 ± 0.032 | - | 24.002 ± 0.019 | 2.310 ± 0.017 | 1.398 ± 0.108 | 1.668 ± 0.120 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 16.307 ± 0.211 | - | 16.175 ± 0.131 | 12.192 ± 0.220 | 27.121 ± 1.541 | 14.368 ± 0.611 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 6.793 ± 0.153 | - | 35.998 ± 0.009 | 6.190 ± 0.205 | 13.928 ± 0.176 | 14.663 ± 0.288 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.885 ± 0.064 | - | 1.863 ± 0.063 | 23.428 ± 0.155 | 2.486 ± 0.107 | 1.532 ± 0.082 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.797 ± 0.080 | - | 23.997 ± 0.004 | 10.592 ± 0.649 | 1.430 ± 0.310 | 1.480 ± 0.064 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 10.286 ± 0.085 | - | 10.216 ± 0.078 | 16.291 ± 0.042 | 12.233 ± 0.232 | 6.594 ± 0.151 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 5.367 ± 0.125 | - | 36.000 ± 0.044 | 8.809 ± 0.108 | 6.629 ± 0.215 | 7.054 ± 0.112 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 11.097 ± 0.216 | - | 10.608 ± 0.300 | 10.745 ± 0.144 | 19.919 ± 0.144 | 11.255 ± 0.525 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 6.199 ± 0.090 | - | 23.997 ± 0.009 | 6.339 ± 0.141 | 10.856 ± 0.338 | 11.139 ± 0.397 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 1.190 ± 0.065 | - | 1.080 ± 0.143 | 2.086 ± 0.052 | 1.971 ± 0.007 | 1.139 ± 0.255 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.489 ± 0.016 | - | 24.002 ± 0.012 | 0.927 ± 0.040 | 1.797 ± 0.158 | 1.103 ± 0.096 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.263 ± 0.115 | - | 1.164 ± 0.167 | 0.905 ± 0.056 | 3.140 ± 0.365 | 1.335 ± 0.058 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.716 ± 0.118 | - | 23.999 ± 0.034 | 0.358 ± 0.071 | 1.653 ± 0.166 | 1.235 ± 0.047 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 10.849 ± 0.201 | - | 11.321 ± 0.380 | 11.238 ± 0.493 | 20.281 ± 0.362 | 11.238 ± 0.532 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 6.256 ± 0.173 | - | 23.999 ± 0.024 | 6.093 ± 0.360 | 11.122 ± 0.338 | 11.442 ± 0.629 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 1.126 ± 0.115 | - | 1.046 ± 0.122 | 2.089 ± 0.034 | 1.978 ± 0.015 | 1.229 ± 0.177 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.468 ± 0.065 | - | 23.999 ± 0.010 | 0.942 ± 0.093 | 1.829 ± 0.045 | 0.912 ± 0.156 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.184 ± 0.021 | - | 1.207 ± 0.129 | 0.920 ± 0.045 | 3.115 ± 0.330 | 1.240 ± 0.072 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.650 ± 0.112 | - | 23.997 ± 0.007 | 0.432 ± 0.090 | 1.525 ± 0.296 | 1.281 ± 0.069 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 1.142 ± 0.066 | 1.183 ± 0.046 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.415 ± 0.038 | 23.999 ± 0.062 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 39.369 ± 1.037 | - | 36.602 ± 1.534 | 54.719 ± 0.918 | 70.804 ± 0.471 | 69.529 ± 5.869 | 83.086 ± 8.846 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 17.835 ± 0.046 | - | 59.997 ± 0.003 | 20.443 ± 0.386 | 38.431 ± 2.421 | 70.991 ± 8.835 | 43.413 ± 11.002 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 140.040 ± 3.005 | - | 141.937 ± 4.011 | 144.527 ± 3.938 | 244.399 ± 1.769 | 261.947 ± 3.829 | 262.876 ± 3.656 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 46.710 ± 0.639 | - | 154.781 ± 2.762 | 46.381 ± 1.415 | 125.232 ± 0.752 | 262.853 ± 7.972 | 110.281 ± 9.228 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 135.323 ± 2.319 | - | 136.700 ± 1.731 | 146.097 ± 4.389 | 151.060 ± 1.641 | 150.471 ± 5.588 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 47.658 ± 0.691 | - | 153.356 ± 9.130 | 50.911 ± 0.712 | 79.308 ± 1.460 | 152.774 ± 5.074 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 128.018 ± 5.507 | - | 125.317 ± 3.603 | 125.007 ± 2.146 | 128.520 ± 0.794 | 143.957 ± 3.968 | 147.514 ± 3.376 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 40.721 ± 2.137 | - | 143.199 ± 5.134 | 41.692 ± 1.581 | 65.932 ± 0.381 | 143.006 ± 2.834 | 65.742 ± 2.866 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 24.455 ± 0.026 | - | 24.425 ± 0.034 | 57.461 ± 0.700 | 106.372 ± 0.933 | 74.231 ± 10.394 | 69.161 ± 12.501 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 10.262 ± 0.067 | - | 47.997 ± 0.008 | 22.017 ± 0.265 | 53.913 ± 0.390 | 66.244 ± 13.900 | 29.638 ± 17.532 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 132.069 ± 3.223 | - | 130.311 ± 1.634 | 185.272 ± 2.884 | 135.716 ± 1.999 | 212.389 ± 4.157 | 218.702 ± 2.805 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 43.712 ± 0.859 | - | 150.133 ± 14.209 | 69.381 ± 1.542 | 68.183 ± 1.184 | 213.775 ± 3.438 | 90.029 ± 2.069 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 131.180 ± 4.831 | - | 131.434 ± 4.825 | 131.917 ± 2.209 | 136.265 ± 1.237 | 143.082 ± 4.594 | 161.640 ± 1.766 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 45.673 ± 0.778 | - | 155.238 ± 12.363 | 45.333 ± 0.730 | 70.367 ± 0.730 | 148.192 ± 3.488 | 70.675 ± 4.357 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 51.323 ± 0.055 | - | 51.504 ± 0.038 | 75.768 ± 0.736 | 38.230 ± 1.282 | 78.618 ± 12.126 | 86.441 ± 21.263 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 20.475 ± 0.105 | - | 71.999 ± 0.009 | 28.454 ± 0.262 | 19.859 ± 0.170 | 87.728 ± 10.200 | 34.341 ± 14.255 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 131.470 ± 1.812 | 132.709 ± 3.147 | unsupported | 141.961 ± 3.522 | 150.182 ± 1.654 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 47.324 ± 0.355 | 156.312 ± 3.681 | unsupported | 51.601 ± 1.153 | 77.405 ± 1.184 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.779 ± 0.059 | - | 0.803 ± 0.132 | 0.873 ± 0.075 | 0.940 ± 0.031 | 7.485 ± 3.806 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.324 ± 0.035 | - | 12.004 ± 0.019 | 0.383 ± 0.029 | 1.056 ± 0.239 | 7.338 ± 3.743 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.859 ± 0.285 | - | 0.820 ± 0.097 | 0.872 ± 0.152 | 1.057 ± 0.299 | 7.627 ± 3.964 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.362 ± 0.157 | - | 12.000 ± 0.003 | 0.376 ± 0.030 | 0.724 ± 0.253 | 7.670 ± 3.670 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.192 ± 0.075 | - | unsupported | 1.341 ± 0.166 | 1.380 ± 0.139 | 5.440 ± 5.988 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.613 ± 0.121 | - | unsupported | 0.683 ± 0.094 | 1.085 ± 0.047 | 8.187 ± 3.959 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.460 ± 0.123 | - | 0.485 ± 0.013 | 0.691 ± 0.025 | 0.424 ± 0.028 | 1.239 ± 0.080 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.440 ± 0.031 | - | 48.002 ± 0.005 | 0.349 ± 0.015 | 0.407 ± 0.047 | 1.310 ± 0.107 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.881 ± 0.288 | - | 0.826 ± 0.145 | 1.109 ± 0.075 | 1.018 ± 0.071 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.384 ± 0.088 | - | 12.000 ± 0.009 | 0.541 ± 0.049 | 1.087 ± 0.027 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.805 ± 0.072 | - | 0.828 ± 0.044 | 0.746 ± 0.030 | 0.878 ± 0.126 | 1.627 ± 7.139 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.365 ± 0.054 | - | 11.999 ± 0.005 | 0.403 ± 0.244 | 0.607 ± 0.114 | 6.347 ± 7.147 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.793 ± 0.006 | - | 0.915 ± 0.014 | 0.482 ± 0.032 | 0.541 ± 0.023 | 0.506 ± 0.014 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 1.059 ± 0.020 | - | 36.012 ± 0.197 | 0.631 ± 0.017 | 0.551 ± 0.029 | 1.365 ± 0.100 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.663 ± 0.058 | - | 0.002 ± 0.000 | 1.765 ± 0.047 | 1.597 ± 0.073 | 9.066 ± 4.613 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.684 ± 0.054 | - | 0.002 ± 0.000 | 0.888 ± 0.219 | 0.985 ± 0.144 | 9.596 ± 4.682 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.051 ± 0.000 | - | 0.053 ± 0.000 | 0.075 ± 0.001 | 0.133 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.093 ± 0.001 | - | 0.095 ± 0.002 | 0.124 ± 0.001 | 0.150 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.238 ± 0.001 | - | 0.239 ± 0.010 | 0.277 ± 0.012 | 0.345 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.081 ± 0.002 | - | 0.084 ± 0.004 | 0.148 ± 0.003 | 0.138 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.059 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.002 ± 0.002 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.018 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.018 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.042 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.673 ± 0.011 | - | 0.673 ± 0.010 | 1.154 ± 0.015 | 0.715 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.026 ± 0.000 | - | 0.028 ± 0.000 | 0.038 ± 0.000 | 0.084 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.015 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.055 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.055 ± 0.005 | - | 0.066 ± 0.000 | 0.085 ± 0.017 | 0.085 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.094 ± 0.001 | - | 0.096 ± 0.001 | 0.124 ± 0.002 | 0.160 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.238 ± 0.001 | - | 0.240 ± 0.002 | 0.339 ± 0.013 | 0.352 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.084 ± 0.002 | - | 0.104 ± 0.018 | 0.179 ± 0.013 | 0.120 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.005 ± 0.001 | - | 0.007 ± 0.000 | 0.016 ± 0.001 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.001 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.676 ± 0.028 | - | 0.696 ± 4.463 | 1.375 ± 0.019 | 1.031 ± 0.876 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.028 ± 0.000 | - | 0.033 ± 0.000 | 0.058 ± 0.000 | 0.055 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.012 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.602 ± 0.008 | - | 0.606 ± 0.010 | 0.720 ± 0.003 | 0.929 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 2.122 ± 0.356 | - | 2.095 ± 0.004 | 2.152 ± 0.007 | 2.692 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 6.455 ± 0.010 | - | 6.455 ± 0.008 | 6.491 ± 0.015 | 8.884 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 3.659 ± 0.017 | - | 3.658 ± 0.009 | 3.261 ± 0.106 | 2.602 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.009 | 0.009 ± 0.001 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.109 ± 0.001 | - | 0.112 ± 0.000 | 0.122 ± 0.005 | 0.164 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.017 ± 0.001 | 0.039 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.012 ± 0.001 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.008 | 0.016 ± 0.001 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.020 ± 0.000 | - | 0.023 ± 0.006 | 0.030 ± 0.001 | 0.100 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.001 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.018 ± 0.003 | 0.032 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.026 ± 0.000 | - | 0.029 ± 0.000 | 0.036 ± 0.003 | 0.106 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 29.222 ± 0.059 | - | 29.336 ± 0.174 | 25.560 ± 0.045 | 20.567 ± 0.063 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.008 | 0.014 ± 0.001 | 0.028 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.863 ± 0.010 | - | 0.865 ± 0.002 | 0.878 ± 0.006 | 0.973 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.018 ± 0.000 | - | 0.020 ± 0.003 | 0.026 ± 0.001 | 0.046 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.052 ± 0.000 | - | 0.054 ± 0.003 | 0.061 ± 0.001 | 0.113 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.612 ± 0.029 | - | 0.617 ± 4.792 | 0.724 ± 0.422 | 0.952 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 2.130 ± 0.029 | - | 2.139 ± 0.023 | 2.141 ± 0.018 | 2.163 ± 0.254 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 6.498 ± 0.020 | - | 6.498 ± 0.041 | 9.305 ± 0.017 | 5.760 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 6.672 ± 0.031 | - | 6.765 ± 0.066 | 2.253 ± 0.072 | 3.993 ± 0.627 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.013 ± 0.001 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.110 ± 0.004 | - | 0.119 ± 0.003 | 0.177 ± 0.005 | 0.189 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.001 | 0.032 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.016 ± 0.001 | 0.032 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.017 ± 0.002 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.016 ± 0.001 | 0.042 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.021 ± 0.000 | - | 0.024 ± 0.000 | 0.030 ± 0.004 | 0.068 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.030 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.018 ± 0.001 | 0.034 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.027 ± 0.000 | - | 0.030 ± 0.000 | 0.035 ± 0.001 | 0.099 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 53.350 ± 0.209 | - | 53.554 ± 0.235 | 17.341 ± 0.137 | 27.592 ± 8.616 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.021 ± 0.001 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.867 ± 0.020 | - | 0.887 ± 4.417 | 1.266 ± 0.029 | 1.996 ± 0.063 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.018 ± 0.000 | - | 0.021 ± 0.000 | 0.026 ± 0.001 | 0.045 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.053 ± 0.001 | - | 0.055 ± 0.000 | 0.060 ± 0.001 | 0.133 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.702 ± 0.027 | - | 0.703 ± 0.009 | 0.513 ± 0.010 | 0.452 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 2.244 ± 0.011 | - | 2.254 ± 0.007 | 1.961 ± 0.028 | 1.951 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 5.807 ± 0.003 | - | 5.833 ± 0.005 | 5.717 ± 0.011 | 6.523 ± 1.604 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.313 ± 0.015 | - | 1.319 ± 0.028 | 1.327 ± 0.008 | 1.497 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.008 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.070 ± 0.001 | - | 0.074 ± 0.002 | 0.089 ± 0.003 | 0.164 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.001 | 0.015 ± 0.001 | 0.029 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.005 | 0.028 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.015 ± 0.001 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.027 ± 0.004 | 0.041 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.017 ± 0.000 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.024 ± 0.001 | - | 0.027 ± 0.000 | 0.031 ± 0.001 | 0.048 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 10.398 ± 0.059 | - | 10.359 ± 0.150 | 10.431 ± 0.094 | 11.295 ± 0.087 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.001 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.553 ± 0.016 | - | 0.553 ± 0.013 | 0.625 ± 0.015 | 0.783 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.024 ± 0.006 | 0.045 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.047 ± 0.000 | - | 0.049 ± 0.000 | 0.054 ± 0.002 | 0.091 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.673 ± 0.026 | - | 0.666 ± 4.463 | 0.801 ± 0.129 | 0.436 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 2.253 ± 0.016 | - | 2.250 ± 0.019 | 1.948 ± 0.005 | 1.277 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 5.897 ± 0.027 | - | 5.899 ± 0.021 | 8.570 ± 0.024 | 3.691 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 3.154 ± 0.017 | - | 3.165 ± 0.009 | 1.183 ± 0.038 | 1.795 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.004 ± 0.001 | - | 0.005 ± 0.000 | 0.012 ± 0.002 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.153 ± 0.038 | - | 0.162 ± 0.031 | 0.136 ± 0.004 | 0.152 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.001 | 0.010 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.014 ± 0.001 | 0.033 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.014 ± 0.001 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.014 ± 0.001 | 0.034 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.019 ± 0.000 | - | 0.022 ± 0.000 | 0.026 ± 0.002 | 0.052 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.015 ± 0.001 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.011 ± 0.000 | - | 0.014 ± 0.004 | 0.016 ± 0.001 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.025 ± 0.000 | - | 0.028 ± 0.000 | 0.031 ± 0.001 | 0.058 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 24.960 ± 0.276 | - | 24.867 ± 0.055 | 8.939 ± 0.085 | 13.927 ± 1.372 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.001 | 0.027 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 1.095 ± 0.022 | - | 1.120 ± 0.033 | 0.939 ± 0.003 | 0.450 ± 0.072 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.019 ± 0.000 | - | 0.022 ± 0.000 | 0.023 ± 0.001 | 0.035 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.048 ± 0.002 | - | 0.051 ± 0.000 | 0.053 ± 0.003 | 0.090 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.266 ± 0.008 | - | 0.267 ± 0.012 | 0.236 ± 0.013 | 0.481 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 1.096 ± 0.005 | - | 1.097 ± 0.002 | 0.942 ± 0.001 | 1.939 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 3.092 ± 0.005 | - | 3.105 ± 0.027 | 2.881 ± 0.008 | 6.430 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.509 ± 0.009 | - | 0.517 ± 0.010 | 0.553 ± 0.014 | 1.483 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.010 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.031 ± 0.000 | - | 0.033 ± 0.000 | 0.041 ± 0.005 | 0.159 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.018 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.020 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.020 ± 0.005 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.021 ± 0.001 | 0.041 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 4.090 ± 0.035 | - | 4.082 ± 0.011 | 4.254 ± 0.020 | 11.126 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.001 | 0.021 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.231 ± 0.001 | - | 0.234 ± 0.010 | 0.242 ± 0.004 | 0.773 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.017 ± 0.000 | 0.037 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.025 ± 0.000 | - | 0.027 ± 0.000 | 0.032 ± 0.001 | 0.089 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.265 ± 0.002 | - | 0.270 ± 0.011 | 0.399 ± 0.013 | 0.441 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 1.113 ± 0.009 | - | 1.148 ± 0.039 | 0.943 ± 0.010 | 1.273 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 3.135 ± 0.010 | - | 3.160 ± 0.037 | 3.667 ± 0.019 | 3.649 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 1.993 ± 0.008 | - | 2.004 ± 0.009 | 0.696 ± 0.016 | 1.798 ± 0.096 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.014 ± 0.002 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.032 ± 0.003 | - | 0.035 ± 0.000 | 0.055 ± 0.003 | 0.147 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.001 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.015 ± 0.004 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.019 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.018 ± 0.001 | 0.042 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.001 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.020 ± 0.001 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.021 ± 0.000 | 0.050 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 15.801 ± 0.121 | - | 15.856 ± 0.151 | 5.179 ± 0.015 | 13.880 ± 0.605 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.017 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.233 ± 0.024 | - | 0.238 ± 0.016 | 0.319 ± 0.013 | 0.414 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.024 ± 0.003 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.026 ± 0.000 | - | 0.028 ± 0.000 | 0.032 ± 0.001 | 0.090 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.151 ± 0.003 | - | 0.161 ± 0.012 | 0.188 ± 0.014 | 0.146 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.223 ± 0.002 | - | 0.234 ± 0.013 | 0.254 ± 0.013 | 0.232 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.516 ± 0.009 | - | 0.538 ± 0.013 | 0.528 ± 0.012 | 0.580 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.136 ± 0.001 | - | 0.142 ± 0.003 | 0.173 ± 0.003 | 0.240 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.014 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.025 ± 0.002 | 0.078 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.014 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.014 ± 0.001 | - | 0.017 ± 0.001 | 0.028 ± 0.001 | 0.081 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.013 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.004 | 0.016 ± 0.001 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.018 ± 0.000 | 0.023 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.018 ± 0.001 | 0.065 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.018 ± 0.001 | 0.066 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 1.077 ± 0.017 | - | 1.079 ± 0.006 | 1.292 ± 0.005 | 1.259 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.077 ± 0.000 | - | 0.080 ± 0.000 | 0.087 ± 0.002 | 0.130 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.017 ± 0.001 | 0.060 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.020 ± 0.000 | 0.044 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.164 ± 0.017 | - | 0.183 ± 0.014 | 0.170 ± 0.004 | 0.141 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.226 ± 0.006 | - | 0.239 ± 0.009 | 0.232 ± 0.018 | 0.217 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.521 ± 0.013 | - | 0.549 ± 0.021 | 0.383 ± 0.019 | 0.640 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.140 ± 0.004 | - | 0.195 ± 0.069 | 0.227 ± 0.007 | 0.225 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.003 ± 0.005 | - | 0.005 ± 0.000 | 0.022 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.013 ± 0.000 | - | 0.017 ± 0.000 | 0.039 ± 0.005 | 0.038 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.024 ± 0.005 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.015 ± 0.004 | - | 0.020 ± 0.001 | 0.042 ± 0.001 | 0.058 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.023 ± 0.005 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.025 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.025 ± 0.001 | 0.025 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.027 ± 0.001 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.026 ± 0.000 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.026 ± 0.000 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.028 ± 0.000 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 1.087 ± 0.056 | - | 1.090 ± 0.016 | 0.695 ± 0.016 | 1.122 ± 0.125 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.027 ± 0.001 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.096 ± 0.002 | - | 0.091 ± 0.001 | 0.079 ± 0.001 | 0.184 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.028 ± 0.001 | 0.026 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.031 ± 0.000 | 0.027 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.262 ± 0.011 | - | 0.266 ± 0.022 | 0.220 ± 0.011 | 0.351 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.614 ± 0.009 | - | 0.626 ± 0.009 | 0.530 ± 0.011 | 0.863 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.357 ± 0.009 | - | 1.362 ± 0.013 | 1.362 ± 0.020 | 2.363 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.493 ± 0.011 | - | 0.500 ± 0.014 | 0.565 ± 0.012 | 0.634 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.018 ± 0.000 | - | 0.020 ± 0.000 | 0.027 ± 0.001 | 0.061 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.029 ± 0.001 | 0.067 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.001 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.030 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.024 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.001 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 3.878 ± 0.010 | - | 4.702 ± 0.029 | 4.385 ± 0.029 | 4.719 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.026 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.129 ± 0.001 | - | 0.132 ± 0.003 | 0.176 ± 0.017 | 0.215 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.034 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.016 ± 0.000 | 0.039 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.256 ± 0.001 | - | 0.259 ± 0.011 | 0.320 ± 0.009 | 0.335 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.612 ± 0.026 | - | 0.626 ± 4.467 | 0.760 ± 0.010 | 0.880 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.370 ± 0.047 | - | 1.365 ± 0.025 | 2.029 ± 0.082 | 1.326 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 1.041 ± 0.030 | - | 1.039 ± 0.027 | 0.497 ± 0.011 | 0.979 ± 0.134 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.018 ± 0.005 | - | 0.022 ± 0.000 | 0.044 ± 0.001 | 0.061 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.020 ± 0.008 | - | 0.025 ± 0.001 | 0.047 ± 0.003 | 0.062 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.021 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.027 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.016 ± 0.001 | 0.034 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.001 | 0.026 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.000 | 0.034 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 8.139 ± 0.069 | - | 8.882 ± 0.065 | 3.739 ± 0.029 | 7.451 ± 0.935 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.140 ± 0.004 | - | 0.145 ± 0.011 | 0.289 ± 0.012 | 0.231 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.015 ± 0.001 | 0.028 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.026 ± 0.000 | 0.056 ± 0.074 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.171 ± 0.002 | - | 0.149 ± 0.005 | 0.098 ± 0.001 | 0.280 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.217 ± 0.011 | - | 0.204 ± 0.042 | 0.140 ± 0.001 | 0.380 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.541 ± 0.021 | - | 0.556 ± 0.062 | 0.411 ± 0.011 | 0.826 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.117 ± 0.002 | - | 0.135 ± 0.044 | 0.147 ± 0.006 | 0.180 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.018 ± 0.001 | 0.019 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.011 ± 0.000 | - | 0.017 ± 0.002 | 0.031 ± 0.002 | 0.075 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.016 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.001 | 0.017 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.002 | 0.018 ± 0.001 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.020 ± 0.001 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.002 | 0.018 ± 0.001 | 0.022 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.020 ± 0.002 | 0.028 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.923 ± 0.008 | - | 0.978 ± 0.006 | 1.017 ± 0.010 | 1.028 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.070 ± 0.001 | - | 0.080 ± 0.020 | 0.082 ± 0.002 | 0.130 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.023 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.011 ± 0.002 | 0.021 ± 0.000 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.164 ± 0.016 | - | 0.149 ± 0.007 | 0.131 ± 0.002 | 0.267 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.201 ± 0.002 | - | 0.220 ± 0.043 | 0.174 ± 0.002 | 0.369 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.537 ± 0.030 | - | 5.991 ± 0.031 | 0.382 ± 0.012 | 0.845 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.119 ± 0.007 | - | 0.155 ± 0.040 | 0.202 ± 0.017 | 0.170 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.026 ± 0.003 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.012 ± 0.001 | - | 0.019 ± 0.003 | 0.044 ± 0.004 | 0.043 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.026 ± 0.004 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.026 ± 0.004 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.029 ± 0.004 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.027 ± 0.003 | 0.024 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.006 ± 0.004 | - | 0.010 ± 0.000 | 0.030 ± 0.003 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.026 ± 0.003 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.026 ± 0.003 | 0.019 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.011 ± 0.000 | 0.029 ± 0.004 | 0.027 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.938 ± 0.027 | - | 23.989 ± 0.180 | 0.580 ± 0.041 | 0.743 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.029 ± 0.002 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.073 ± 0.001 | - | 0.086 ± 0.019 | 0.073 ± 0.001 | 0.167 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.010 ± 0.001 | 0.029 ± 0.001 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.007 ± 0.000 | - | 0.013 ± 0.001 | 0.032 ± 0.004 | 0.028 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.807 ± 0.005 | - | 1.814 ± 0.004 | 1.550 ± 0.015 | 2.067 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 6.007 ± 0.009 | - | 6.006 ± 0.012 | 5.161 ± 0.047 | 5.076 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 17.737 ± 0.017 | - | 17.721 ± 0.026 | 15.919 ± 0.155 | 14.642 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 4.950 ± 0.009 | - | 4.966 ± 0.046 | 5.597 ± 0.075 | 4.479 ± 1.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.015 ± 0.001 | 0.033 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.201 ± 0.009 | - | 0.207 ± 0.007 | 0.235 ± 0.015 | 0.239 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.020 ± 0.001 | 0.032 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.179 ± 0.025 | - | 0.185 ± 0.012 | 0.212 ± 0.027 | 0.257 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.020 ± 0.000 | - | 0.023 ± 0.000 | 0.032 ± 0.006 | 0.046 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.017 ± 0.001 | 0.032 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.020 ± 0.002 | - | 0.022 ± 0.000 | 0.029 ± 0.001 | 0.042 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.055 ± 0.006 | - | 0.058 ± 0.005 | 0.065 ± 0.003 | 0.074 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.018 ± 0.002 | 0.033 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.026 ± 0.001 | - | 0.029 ± 0.000 | 0.035 ± 0.001 | 0.049 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.073 ± 0.000 | - | 0.076 ± 0.001 | 0.080 ± 0.004 | 0.090 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 39.374 ± 0.085 | - | 40.901 ± 1.801 | 44.032 ± 0.248 | 33.175 ± 0.160 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.016 ± 0.000 | - | 0.019 ± 0.000 | 0.025 ± 0.001 | 0.043 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.544 ± 0.015 | - | 1.545 ± 0.010 | 1.641 ± 0.016 | 1.530 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.049 ± 0.000 | - | 0.052 ± 0.000 | 0.056 ± 0.001 | 0.071 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.141 ± 0.001 | - | 0.144 ± 0.004 | 0.142 ± 0.006 | 0.161 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.804 ± 0.099 | - | 1.816 ± 0.049 | 1.533 ± 0.012 | 2.065 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 6.063 ± 0.016 | - | 6.042 ± 0.032 | 5.082 ± 0.053 | 2.762 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 17.804 ± 0.178 | - | 17.814 ± 0.072 | 18.535 ± 0.153 | 7.556 ± 0.208 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 9.270 ± 0.059 | - | 9.281 ± 0.112 | 4.630 ± 0.107 | 4.931 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.003 | 0.031 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.308 ± 0.016 | - | 0.324 ± 0.024 | 0.330 ± 0.016 | 0.264 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.009 ± 0.000 | - | 0.013 ± 0.000 | 0.020 ± 0.008 | 0.032 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.223 ± 0.011 | - | 0.248 ± 0.044 | 0.288 ± 0.016 | 0.228 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.021 ± 0.000 | - | 0.025 ± 0.000 | 0.030 ± 0.001 | 0.044 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.024 ± 0.004 | 0.039 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.021 ± 0.000 | - | 0.024 ± 0.000 | 0.030 ± 0.001 | 0.046 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.058 ± 0.001 | - | 0.061 ± 0.000 | 0.063 ± 0.001 | 0.073 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.028 ± 0.005 | 0.033 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.027 ± 0.000 | - | 0.030 ± 0.000 | 0.035 ± 0.004 | 0.060 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.077 ± 0.001 | - | 0.081 ± 0.003 | 0.079 ± 0.002 | 0.089 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 74.885 ± 0.628 | - | 75.560 ± 2.226 | 36.358 ± 0.165 | 91.509 ± 44.458 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.017 ± 0.000 | - | 0.020 ± 0.000 | 0.037 ± 0.003 | 0.051 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 2.278 ± 0.032 | - | 2.291 ± 0.020 | 2.325 ± 0.018 | 0.788 ± 0.101 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.051 ± 0.010 | - | 0.054 ± 0.000 | 0.055 ± 0.001 | 0.070 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.142 ± 0.004 | - | 0.146 ± 0.008 | 0.141 ± 0.004 | 0.163 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 1.361 ± 0.029 | - | 1.354 ± 0.012 | 1.091 ± 0.002 | 1.361 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 5.141 ± 0.025 | - | 5.150 ± 0.025 | 4.343 ± 0.011 | 4.076 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 15.179 ± 0.108 | - | 15.186 ± 0.080 | 13.801 ± 0.118 | 12.320 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 2.297 ± 0.011 | - | 2.314 ± 0.026 | 2.429 ± 0.011 | 2.440 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.001 | 0.026 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.135 ± 0.008 | - | 0.140 ± 0.015 | 0.163 ± 0.007 | 0.222 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.002 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.108 ± 0.009 | - | 0.113 ± 0.011 | 0.131 ± 0.007 | 0.194 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.017 ± 0.000 | - | 0.020 ± 0.008 | 0.027 ± 0.001 | 0.039 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.007 ± 0.003 | - | 0.009 ± 0.000 | 0.016 ± 0.001 | 0.036 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.018 ± 0.000 | - | 0.020 ± 0.000 | 0.028 ± 0.001 | 0.038 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.043 ± 0.000 | - | 0.047 ± 0.005 | 0.053 ± 0.001 | 0.063 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.017 ± 0.001 | 0.032 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.023 ± 0.000 | - | 0.025 ± 0.000 | 0.031 ± 0.008 | 0.043 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.057 ± 0.000 | - | 0.060 ± 0.000 | 0.065 ± 0.002 | 0.073 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 18.394 ± 0.138 | - | 18.218 ± 0.218 | 19.219 ± 0.116 | 18.619 ± 0.077 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.021 ± 0.001 | 0.045 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 1.048 ± 0.002 | - | 1.059 ± 0.004 | 1.158 ± 0.008 | 1.186 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.042 ± 0.002 | - | 0.045 ± 0.000 | 0.048 ± 0.001 | 0.060 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.115 ± 0.000 | - | 0.118 ± 0.004 | 0.117 ± 0.002 | 0.168 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 1.363 ± 0.042 | - | 1.369 ± 0.023 | 1.050 ± 0.015 | 1.310 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 5.262 ± 0.021 | - | 5.271 ± 0.095 | 4.281 ± 0.011 | 2.141 ± 0.397 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 15.244 ± 0.122 | - | 15.309 ± 0.091 | 16.648 ± 0.017 | 6.279 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 3.451 ± 0.109 | - | 3.432 ± 0.050 | 2.749 ± 0.010 | 3.099 ± 0.978 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.229 ± 0.017 | - | 0.242 ± 0.023 | 0.256 ± 0.021 | 0.206 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.018 ± 0.001 | 0.038 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.155 ± 0.027 | - | 0.166 ± 0.015 | 0.197 ± 0.010 | 0.166 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.018 ± 0.011 | - | 0.022 ± 0.000 | 0.027 ± 0.001 | 0.046 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.022 ± 0.004 | 0.031 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.019 ± 0.000 | - | 0.021 ± 0.001 | 0.026 ± 0.000 | 0.046 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.046 ± 0.002 | - | 0.049 ± 0.000 | 0.052 ± 0.004 | 0.062 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.023 ± 0.004 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.024 ± 0.000 | - | 0.027 ± 0.001 | 0.030 ± 0.001 | 0.052 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.061 ± 0.000 | - | 0.063 ± 0.001 | 0.065 ± 0.010 | 0.074 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 27.322 ± 0.244 | - | 27.214 ± 0.213 | 21.469 ± 0.070 | 24.112 ± 8.282 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.030 ± 0.003 | 0.041 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 1.759 ± 0.032 | - | 1.804 ± 0.041 | 1.752 ± 0.009 | 0.655 ± 0.125 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.045 ± 0.001 | - | 0.047 ± 0.000 | 0.047 ± 0.001 | 0.057 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.118 ± 0.011 | - | 0.127 ± 0.011 | 0.118 ± 0.006 | 0.180 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.494 ± 0.010 | - | unsupported | 0.353 ± 0.012 | 0.787 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 2.276 ± 0.007 | - | unsupported | 1.947 ± 0.004 | 2.739 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 5.530 ± 0.011 | - | unsupported | 5.296 ± 0.014 | 6.819 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.974 ± 0.017 | - | unsupported | 1.133 ± 0.013 | 1.110 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.009 ± 0.001 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.047 ± 0.001 | - | unsupported | 0.064 ± 0.004 | 0.082 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | unsupported | 0.012 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.039 ± 0.001 | - | unsupported | 0.054 ± 0.002 | 0.065 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | unsupported | 0.015 ± 0.001 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.011 ± 0.003 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.029 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.017 ± 0.000 | - | unsupported | 0.028 ± 0.005 | 0.046 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | unsupported | 0.017 ± 0.001 | 0.033 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.023 ± 0.000 | - | unsupported | 0.031 ± 0.001 | 0.044 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 7.672 ± 0.095 | - | unsupported | 8.892 ± 0.061 | 8.661 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | unsupported | 0.012 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.375 ± 0.009 | - | unsupported | 0.422 ± 0.012 | 0.478 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.019 ± 0.000 | - | unsupported | 0.024 ± 0.001 | 0.037 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.044 ± 0.000 | - | unsupported | 0.052 ± 0.003 | 0.069 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.484 ± 0.012 | - | unsupported | 0.600 ± 0.029 | 0.778 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 2.304 ± 0.045 | - | unsupported | 1.939 ± 0.856 | 1.413 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 5.594 ± 0.052 | - | unsupported | 7.661 ± 0.009 | 3.516 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.191 ± 0.015 | - | unsupported | 1.445 ± 0.012 | 1.434 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | unsupported | 0.013 ± 0.001 | 0.021 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.048 ± 0.001 | - | unsupported | 0.088 ± 0.016 | 0.093 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | unsupported | 0.034 ± 0.006 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.040 ± 0.000 | - | unsupported | 0.077 ± 0.002 | 0.092 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.008 ± 0.000 | - | unsupported | 0.015 ± 0.001 | 0.026 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | unsupported | 0.016 ± 0.003 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.009 ± 0.000 | - | unsupported | 0.025 ± 0.005 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.018 ± 0.000 | - | unsupported | 0.026 ± 0.004 | 0.036 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | unsupported | 0.016 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.011 ± 0.000 | - | unsupported | 0.026 ± 0.000 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.024 ± 0.000 | - | unsupported | 0.031 ± 0.001 | 0.045 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 9.434 ± 0.263 | - | unsupported | 11.213 ± 0.043 | 11.546 ± 0.139 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | unsupported | 0.018 ± 0.001 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.375 ± 0.009 | - | unsupported | 0.420 ± 0.015 | 0.375 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.019 ± 0.000 | - | unsupported | 0.037 ± 0.005 | 0.039 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.044 ± 0.000 | - | unsupported | 0.051 ± 0.004 | 0.071 ± 0.002 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 1.038 ± 0.001 | - | unsupported | 0.954 ± 0.007 | 3.461 ± 0.023 | 2.568 ± 0.060 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.132 ± 0.003 | - | unsupported | 0.059 ± 0.002 | 0.136 ± 0.006 | 0.132 ± 0.010 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.136 ± 0.004 | - | unsupported | 0.084 ± 0.003 | 0.206 ± 0.004 | 0.220 ± 0.011 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.334 ± 0.009 | - | unsupported | 0.242 ± 0.002 | 0.537 ± 0.040 | 0.729 ± 0.013 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 1.051 ± 0.010 | - | unsupported | 0.487 ± 0.014 | 1.097 ± 0.100 | 2.468 ± 0.880 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.134 ± 0.011 | - | unsupported | 0.061 ± 0.003 | 0.120 ± 0.003 | 0.100 ± 0.005 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.138 ± 0.002 | - | unsupported | 0.079 ± 0.001 | 0.184 ± 0.022 | 0.177 ± 0.011 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.342 ± 0.011 | - | unsupported | 0.159 ± 0.021 | 0.529 ± 0.034 | 0.672 ± 0.053 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.308 ± 0.011 | - | unsupported | 0.376 ± 0.010 | 0.801 ± 0.009 | 0.491 ± 0.014 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.035 ± 0.000 | - | unsupported | 0.046 ± 0.007 | 0.213 ± 0.024 | 0.079 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.059 ± 0.000 | - | unsupported | 0.061 ± 0.001 | 0.246 ± 0.028 | 0.125 ± 0.006 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.200 ± 0.003 | - | unsupported | 0.166 ± 0.011 | 0.377 ± 0.011 | 0.237 ± 0.010 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.321 ± 0.010 | - | unsupported | 0.562 ± 0.015 | 0.919 ± 0.133 | 0.417 ± 0.011 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.035 ± 0.000 | - | unsupported | 0.074 ± 0.005 | 0.207 ± 0.015 | 0.074 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.059 ± 0.004 | - | unsupported | 0.099 ± 0.001 | 0.244 ± 0.020 | 0.117 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.201 ± 0.001 | - | unsupported | 0.227 ± 0.013 | 0.382 ± 0.021 | 0.210 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.194 ± 0.001 | - | 0.197 ± 0.002 | 0.198 ± 0.010 | 0.390 ± 0.019 | 0.349 ± 0.028 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.081 ± 0.000 | - | 0.084 ± 0.000 | 0.073 ± 0.001 | 0.079 ± 0.009 | 0.063 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.092 ± 0.000 | - | 0.095 ± 0.001 | 0.089 ± 0.003 | 0.102 ± 0.004 | 0.097 ± 0.005 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.121 ± 0.000 | - | 0.125 ± 0.000 | 0.123 ± 0.008 | 0.173 ± 0.002 | 0.182 ± 0.015 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.193 ± 0.010 | - | 0.198 ± 0.024 | 0.259 ± 0.014 | 0.392 ± 0.028 | 0.257 ± 0.010 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.082 ± 0.000 | - | 0.097 ± 0.002 | 0.125 ± 0.009 | 0.081 ± 0.003 | 0.060 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.092 ± 0.000 | - | 0.102 ± 0.001 | 0.142 ± 0.005 | 0.105 ± 0.006 | 0.086 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.122 ± 0.000 | - | 0.127 ± 0.001 | 0.178 ± 0.004 | 0.216 ± 0.017 | 0.140 ± 0.001 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 43.682 ± 0.875 | - | 49.687 ± 0.127 | 61.423 ± 0.275 | 69.798 ± 10.032 | 51.711 ± 7.052 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 46.116 ± 0.347 | - | 49.220 ± 0.448 | 25.015 ± 0.227 | 58.815 ± 1.685 | 40.844 ± 6.229 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 24.949 ± 0.144 | - | 24.407 ± 0.209 | 27.075 ± 0.166 | 22.124 ± 0.098 | 20.965 ± 2.763 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 248.936 ± 15.446 | - | 319.984 ± 11.052 | 11.233 ± 0.053 | 14.084 ± 0.122 | 12.506 ± 4.846 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 7.461 ± 0.070 | - | 7.530 ± 0.097 | 7.972 ± 0.040 | 8.350 ± 0.143 | 8.307 ± 0.245 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 16.515 ± 0.070 | - | 16.409 ± 0.062 | 10.395 ± 0.045 | 11.630 ± 0.050 | 12.118 ± 0.025 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 8.695 ± 0.015 | - | 8.792 ± 0.100 | 8.772 ± 0.021 | 9.634 ± 0.091 | 9.173 ± 0.032 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 12.874 ± 0.086 | - | 12.967 ± 0.133 | 11.868 ± 0.150 | 14.197 ± 0.051 | 13.746 ± 0.085 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 21.350 ± 0.701 | - | 21.716 ± 0.242 | 19.427 ± 0.144 | 39.451 ± 0.146 | 14.130 ± 0.684 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 38.640 ± 0.239 | - | 39.283 ± 0.277 | 14.825 ± 0.133 | 32.245 ± 1.134 | 13.458 ± 0.462 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.488 ± 0.002 | 0.509 ± 0.049 | unsupported | 0.605 ± 0.005 | 0.751 ± 0.025 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 7.840 ± 0.141 | 8.700 ± 0.061 | unsupported | 8.786 ± 0.031 | 9.673 ± 0.080 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.791 ± 0.028 | 0.815 ± 0.003 | unsupported | 0.515 ± 0.007 | 1.265 ± 1.519 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 2276.437 ± 3.127 | 2306.347 ± 1.886 | unsupported | 5.734 ± 0.057 | 9.481 ± 0.143 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 44.425 ± 3.821 | - | 41.231 ± 0.346 | 43.822 ± 0.197 | 31.370 ± 0.188 | 27.631 ± 0.540 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 226.423 ± 18.087 | - | 280.121 ± 12.965 | 21.844 ± 9.206 | 19.997 ± 1.156 | 16.549 ± 2.201 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 17.687 ± 0.861 | 17.223 ± 0.188 | 9.382 ± 0.126 | 56.902 ± 0.121 | 9.757 ± 0.079 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 2285.808 ± 11.463 | 2287.692 ± 5.049 | 6.169 ± 0.161 | 54.328 ± 0.973 | 9.303 ± 0.666 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 25.063 ± 0.319 | - | 25.476 ± 0.379 | 32.446 ± 0.369 | 32.523 ± 0.107 | 30.880 ± 9.993 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 199.031 ± 77.461 | - | 236.221 ± 32.441 | 15.941 ± 0.164 | 25.606 ± 1.644 | 27.274 ± 17.506 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 1.118 ± 0.086 | - | 1.164 ± 0.206 | 1.411 ± 0.137 | 4.157 ± 0.317 | 2.738 ± 0.016 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.448 ± 0.048 | - | 23.998 ± 0.014 | 1.285 ± 0.056 | 13.177 ± 1.405 | 2.698 ± 0.026 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 19.872 ± 0.477 | - | 19.556 ± 0.047 | 20.980 ± 0.067 | 20.048 ± 0.074 | 19.673 ± 2.846 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 216.550 ± 3.055 | - | 233.004 ± 24.152 | 16.444 ± 0.061 | 19.068 ± 0.659 | 21.689 ± 5.048 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 19.745 ± 0.329 | - | 19.729 ± 0.135 | 20.988 ± 0.120 | 20.546 ± 0.227 | 19.506 ± 1.721 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 228.112 ± 15.479 | - | 235.112 ± 9.653 | 16.693 ± 0.177 | 20.680 ± 0.703 | 20.016 ± 7.171 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.159 ± 0.009 | 1.169 ± 0.008 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 22.670 ± 0.089 | 22.748 ± 0.158 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.599 ± 0.015 | 1.608 ± 0.012 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 22.522 ± 0.216 | 22.570 ± 0.211 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.080 ± 0.004 | 1.092 ± 0.009 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 21.647 ± 0.062 | 21.749 ± 0.833 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.508 ± 0.014 | 1.533 ± 0.004 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 21.406 ± 0.099 | 21.372 ± 0.085 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 24.902 ± 0.349 | - | 24.362 ± 0.070 | 27.044 ± 0.179 | 21.991 ± 0.159 | 21.043 ± 0.779 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 237.917 ± 17.150 | - | 334.009 ± 25.005 | 11.334 ± 0.174 | 13.992 ± 0.967 | 12.614 ± 2.471 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 60.743 ± 7.224 | 60.103 ± 0.244 | 61.887 ± 0.378 | 66.503 ± 0.196 | 63.819 ± 0.970 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 2334.785 ± 14.941 | 2337.669 ± 11.820 | 38.350 ± 1.290 | 51.973 ± 0.260 | 52.602 ± 5.017 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 33.399 ± 0.123 | - | 33.474 ± 0.216 | 30.224 ± 0.413 | 130.414 ± 2.203 | 29.490 ± 0.402 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 51.574 ± 1.326 | - | 50.344 ± 0.684 | 25.410 ± 1.720 | 123.773 ± 17.725 | 19.698 ± 0.175 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.139 ± 0.001 | 0.139 ± 0.001 | unsupported | 0.102 ± 0.001 | 0.036 ± 0.004 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.208 ± 0.003 | 2.212 ± 0.013 | unsupported | 1.618 ± 0.009 | 1.234 ± 0.024 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.073 ± 0.000 | 5.984 ± 0.031 | unsupported | 0.078 ± 0.004 | 0.125 ± 0.005 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.893 ± 0.004 | 23.999 ± 0.751 | unsupported | 1.044 ± 0.022 | 0.678 ± 0.025 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.078 ± 0.000 | 0.079 ± 0.000 | unsupported | 0.068 ± 0.003 | 0.228 ± 0.006 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.239 ± 0.005 | 1.237 ± 0.013 | unsupported | 1.800 ± 0.032 | 2.642 ± 0.080 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.038 ± 0.034 | 12.000 ± 0.000 | unsupported | 0.092 ± 0.024 | 0.159 ± 0.003 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.558 ± 0.003 | 11.999 ± 0.001 | unsupported | 0.749 ± 0.016 | 1.119 ± 0.032 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.196 ± 0.001 | 0.197 ± 0.000 | unsupported | 0.225 ± 0.001 | 0.207 ± 0.001 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 3.136 ± 0.002 | 3.120 ± 0.013 | unsupported | 4.395 ± 0.019 | 1.826 ± 0.009 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.076 ± 0.000 | 12.000 ± 0.187 | unsupported | 0.084 ± 0.001 | 0.141 ± 0.005 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 1.180 ± 0.025 | 11.999 ± 0.001 | unsupported | 1.536 ± 0.033 | 0.975 ± 0.018 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.061 ± 0.008 | 0.090 ± 0.005 | unsupported | 0.101 ± 0.000 | 0.202 ± 0.002 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.638 ± 0.029 | 0.672 ± 0.047 | unsupported | 0.905 ± 0.011 | 1.836 ± 0.059 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.089 ± 0.008 | 35.937 ± 0.140 | unsupported | 0.042 ± 0.001 | 0.181 ± 0.001 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.422 ± 0.035 | 82.123 ± 46.929 | unsupported | 0.365 ± 0.008 | 1.217 ± 0.185 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.125 ± 0.003 | 0.152 ± 0.001 | unsupported | 0.079 ± 0.001 | 0.219 ± 0.004 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.683 ± 0.018 | 1.777 ± 0.079 | unsupported | 1.122 ± 0.022 | 2.234 ± 0.016 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.150 ± 0.010 | 43.875 ± 0.094 | unsupported | 0.038 ± 0.000 | 0.210 ± 0.002 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.997 ± 0.165 | 120.000 ± 0.248 | unsupported | 0.525 ± 0.024 | 1.803 ± 0.093 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.291 ± 0.003 | 0.332 ± 0.007 | unsupported | unsupported | 0.264 ± 0.001 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 4.306 ± 0.030 | 4.419 ± 0.036 | unsupported | unsupported | 3.180 ± 0.050 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.278 ± 0.006 | 51.765 ± 9.063 | unsupported | unsupported | 0.246 ± 0.015 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 2.040 ± 0.109 | 144.000 ± 0.001 | unsupported | unsupported | 1.869 ± 0.205 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.010 ± 0.001 | 0.018 ± 0.000 | unsupported | 0.013 ± 0.000 | 0.067 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.104 ± 0.002 | 0.113 ± 0.004 | unsupported | 0.092 ± 0.001 | 0.247 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.015 ± 0.000 | 5.968 ± 0.063 | unsupported | 0.015 ± 0.000 | 0.038 ± 0.000 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.125 ± 0.010 | 23.998 ± 0.004 | unsupported | 0.096 ± 0.003 | 0.170 ± 0.014 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.031 ± 0.003 | 0.048 ± 0.003 | unsupported | 0.043 ± 0.001 | 0.187 ± 0.003 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.364 ± 0.028 | 0.373 ± 0.023 | unsupported | 0.568 ± 0.044 | 1.640 ± 0.042 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.045 ± 0.003 | 18.000 ± 0.156 | unsupported | 0.059 ± 0.001 | 0.160 ± 0.002 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.218 ± 0.042 | 48.000 ± 0.125 | unsupported | 0.274 ± 0.036 | 1.062 ± 0.073 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.054 ± 0.000 | 0.055 ± 0.001 | unsupported | 0.062 ± 0.002 | 0.193 ± 0.002 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.854 ± 0.002 | 0.862 ± 0.009 | unsupported | 1.782 ± 0.025 | 1.756 ± 0.023 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.027 ± 0.001 | 12.000 ± 0.032 | unsupported | 0.100 ± 0.004 | 0.136 ± 0.011 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.392 ± 0.006 | 11.999 ± 0.001 | unsupported | 0.704 ± 0.013 | 0.929 ± 0.010 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.056 ± 0.000 | 0.057 ± 0.000 | unsupported | 0.064 ± 0.001 | 0.169 ± 0.003 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.888 ± 0.003 | 0.896 ± 0.008 | unsupported | 1.843 ± 0.033 | 1.754 ± 0.040 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.028 ± 0.000 | 12.000 ± 0.000 | unsupported | 0.102 ± 0.002 | 0.130 ± 0.003 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.406 ± 0.001 | 11.999 ± 0.001 | unsupported | 0.752 ± 0.011 | 0.949 ± 0.177 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.125 ± 0.002 | 0.154 ± 0.007 | unsupported | 0.081 ± 0.000 | 0.215 ± 0.003 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.679 ± 0.028 | 1.767 ± 0.070 | unsupported | 1.152 ± 0.002 | 2.178 ± 0.021 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.143 ± 0.006 | 43.874 ± 0.219 | unsupported | 0.036 ± 0.001 | 0.202 ± 0.004 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 1.011 ± 0.148 | 119.626 ± 1.498 | unsupported | 0.482 ± 0.005 | 1.344 ± 0.264 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.183 ± 0.001 | 0.183 ± 0.000 | unsupported | 0.217 ± 0.001 | 0.313 ± 0.005 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 2.908 ± 0.004 | 2.902 ± 0.015 | unsupported | 4.290 ± 0.023 | 4.013 ± 0.042 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.073 ± 0.001 | 12.000 ± 5.546 | unsupported | 0.202 ± 0.002 | 0.205 ± 0.025 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 1.124 ± 0.020 | 11.999 ± 0.000 | unsupported | 1.513 ± 0.008 | 2.083 ± 0.072 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.748 ± 0.011 | 1.348 ± 0.025 | unsupported | 0.261 ± 0.001 | 0.237 ± 0.003 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 18.200 ± 0.174 | 28.616 ± 0.506 | unsupported | 5.472 ± 0.090 | 2.321 ± 0.036 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.851 ± 0.043 | 93.000 ± 0.688 | unsupported | 0.251 ± 0.014 | 0.166 ± 0.025 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 8.752 ± 0.185 | 154.375 ± 2.625 | unsupported | 2.331 ± 0.060 | 1.241 ± 0.040 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 28.188 ± 0.221 | - | unsupported | 27.852 ± 0.485 | - | 29.445 ± 0.268 | 38.681 ± 10.823 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 21.903 ± 0.739 | - | unsupported | 21.505 ± 0.385 | - | 32.169 ± 0.505 | 21.436 ± 0.715 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 21.391 ± 0.098 | - | unsupported | 26.563 ± 0.269 | - | 22.234 ± 0.719 | 23.920 ± 0.710 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 15.074 ± 0.294 | - | unsupported | 15.487 ± 0.557 | - | 22.422 ± 0.699 | 14.841 ± 0.260 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 13.676 ± 0.177 | - | unsupported | 23.028 ± 0.214 | - | 13.981 ± 0.412 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 15.010 ± 0.421 | - | unsupported | 15.583 ± 3.002 | - | 13.604 ± 0.117 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 27.941 ± 0.246 | - | unsupported | 28.643 ± 0.442 | - | 28.467 ± 0.719 | 34.775 ± 0.416 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 21.325 ± 0.144 | - | unsupported | 21.869 ± 2.946 | - | 29.465 ± 0.383 | 21.781 ± 2.534 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 48.999 ± 0.453 | - | unsupported | 49.871 ± 0.233 | - | 47.852 ± 0.104 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 52.984 ± 0.455 | - | unsupported | 25.140 ± 0.674 | - | 23.522 ± 0.141 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 48.740 ± 0.195 | - | unsupported | 49.765 ± 0.771 | - | 47.626 ± 0.425 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 52.648 ± 0.590 | - | unsupported | 24.953 ± 0.137 | - | 23.311 ± 0.083 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 28.529 ± 0.323 | - | unsupported | 27.698 ± 0.272 | - | 29.165 ± 0.553 | 37.985 ± 0.180 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 21.438 ± 0.676 | - | unsupported | 21.398 ± 0.316 | - | 32.284 ± 0.154 | 21.224 ± 0.331 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 20.916 ± 0.165 | - | unsupported | 23.002 ± 0.229 | - | 22.920 ± 0.344 | 27.295 ± 1.148 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 14.866 ± 0.281 | - | unsupported | 14.938 ± 0.214 | - | 25.911 ± 0.215 | 15.116 ± 0.419 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.748 ± 0.005 | unsupported | unsupported | 0.899 ± 0.002 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 11.125 ± 0.101 | unsupported | unsupported | 12.965 ± 0.022 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 1.475 ± 0.021 | unsupported | unsupported | 0.613 ± 0.004 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 195.442 ± 2.868 | unsupported | unsupported | 6.847 ± 0.074 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 28.326 ± 0.329 | - | unsupported | 27.684 ± 0.270 | - | 29.232 ± 3.707 | 39.285 ± 0.755 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 21.269 ± 1.016 | - | unsupported | 21.463 ± 0.578 | - | 32.440 ± 0.552 | 21.691 ± 0.737 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 106.498 ± 3.730 | - | 0.002 ± 0.000 | 108.002 ± 2.012 | 107.157 ± 0.962 | 130.443 ± 1.239 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 42.689 ± 0.591 | - | 0.002 ± 0.000 | 35.214 ± 1.334 | 54.857 ± 0.287 | 133.536 ± 3.965 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 67.031 ± 2.051 | - | 204.935 ± 6.388 | 67.759 ± 1.336 | 68.837 ± 0.788 | 76.961 ± 1.039 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 22.886 ± 0.323 | - | 253.842 ± 14.645 | 23.130 ± 0.291 | 36.089 ± 0.582 | 76.486 ± 1.754 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 214.128 ± 1.268 | - | 214.991 ± 1.420 | 209.578 ± 2.687 | 214.736 ± 3.639 | 250.308 ± 4.858 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 70.722 ± 0.600 | - | 216.101 ± 9.737 | 69.997 ± 1.193 | 110.715 ± 0.880 | 247.867 ± 6.812 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 66.550 ± 0.910 | - | 67.051 ± 2.182 | 73.061 ± 1.877 | 67.341 ± 1.832 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 42.063 ± 0.324 | - | 69.594 ± 2.422 | 24.461 ± 0.834 | 35.546 ± 0.755 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 133.671 ± 4.688 | - | 0.002 ± 0.000 | 123.612 ± 0.584 | 120.783 ± 3.390 | 153.257 ± 1.776 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 40.394 ± 0.186 | - | 0.002 ± 0.000 | 63.454 ± 27.842 | 44.580 ± 1.985 | 154.123 ± 6.091 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 83.676 ± 1.066 | - | 0.002 ± 0.000 | 77.934 ± 0.743 | 83.105 ± 1.702 | 214.604 ± 2.241 | 128.238 ± 4.267 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 30.236 ± 0.528 | - | 0.002 ± 0.000 | 78.234 ± 0.940 | 43.374 ± 1.072 | 214.455 ± 2.926 | 51.607 ± 1.394 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 58.269 ± 0.753 | - | 59.069 ± 0.337 | 64.290 ± 0.931 | 62.695 ± 1.468 | 80.670 ± 1.128 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 20.891 ± 0.080 | - | 70.536 ± 1.075 | 23.216 ± 0.127 | 33.165 ± 0.672 | 79.766 ± 2.029 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 58.313 ± 0.890 | - | 59.077 ± 0.655 | 62.577 ± 1.310 | 63.609 ± 1.575 | 81.151 ± 1.066 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 20.545 ± 0.280 | - | 70.755 ± 0.899 | 23.078 ± 0.886 | 32.626 ± 0.875 | 79.811 ± 2.209 | - |
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
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 7.771514 | 485719.62 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 7.189747 | 449359.19 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 7.799025 | 487439.06 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 3.600446 | 225027.88 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 6.257292 | 391080.75 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 831.989515 | 51999344.69 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 18.267035 | 1141689.69 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 18.925714 | 1182857.12 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 6.646119 | 415382.44 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 383.991045 | 23999440.31 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 12.017496 | 751093.50 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 9.685027 | 605314.19 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 7.806671 | 487916.94 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 8.137989 | 508624.31 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 154.772372 | 9673273.25 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 140.568327 | 8785520.44 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 125.447977 | 7840498.56 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 139.199643 | 8699977.69 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 20.245872 | 1265367.00 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 8.241818 | 515113.62 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 12.657860 | 791116.25 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 13.032199 | 814512.44 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 151.688960 | 9480560.00 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 91.738521 | 5733657.56 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 36422.990512 | 2276436907.00 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 36901.544750 | 2306346546.88 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 18.544687 | 1159042.94 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 18.708022 | 1169251.38 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 362.720678 | 22670042.38 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 363.973953 | 22748372.06 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 25.588309 | 1599269.31 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 25.723871 | 1607741.94 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 360.358700 | 22522418.75 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 361.116936 | 22569808.50 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 17.275056 | 1079691.00 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 17.479959 | 1092497.44 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 346.359235 | 21647452.19 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 347.986715 | 21749169.69 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 24.122870 | 1507679.38 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 24.528539 | 1533033.69 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 342.493449 | 21405840.56 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 341.959600 | 21372475.00 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.332790 | 36449.84 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.553981 | 102405.95 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.883883 | 138810.67 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.899983 | 139062.23 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 9.871942 | 1233992.75 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 12.943814 | 1617976.75 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 17.663591 | 2207948.88 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 17.699057 | 2212382.12 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.981300 | 124707.81 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.010561 | 78290.02 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.655051 | 72735.17 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 382.990383 | 5984224.73 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.425987 | 678248.38 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.350633 | 1043829.12 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.140954 | 892619.25 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 191.994371 | 23999296.38 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.616690 | 228385.78 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.337551 | 67774.23 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.014646 | 78353.84 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.042258 | 78785.28 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 21.132609 | 2641576.12 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.402268 | 1800283.50 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.911190 | 1238898.75 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 9.896201 | 1237025.12 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.192881 | 159263.77 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.876068 | 91813.56 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.403602 | 37556.28 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.980747 | 11999699.17 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.951242 | 1118905.25 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.989762 | 748720.25 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.467659 | 558457.38 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.994831 | 11999353.88 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.277817 | 207465.89 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 14.421494 | 225335.84 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 12.575838 | 196497.47 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 12.616274 | 197129.28 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.604437 | 1825554.62 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 35.156318 | 4394539.75 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 25.085968 | 3135746.00 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 24.956557 | 3119569.62 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.007097 | 140735.89 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.387316 | 84176.81 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.873949 | 76155.45 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.983821 | 11999747.20 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.800202 | 975025.25 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.291296 | 1536412.00 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.436120 | 1179515.00 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.993579 | 11999197.38 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.940867 | 202201.05 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.476977 | 101202.77 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.881580 | 60649.69 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.749999 | 89843.73 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.687171 | 1835896.38 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.236975 | 904621.88 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.102720 | 637840.00 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.378935 | 672366.88 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.566147 | 180721.05 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.680579 | 41884.05 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.697315 | 89020.55 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2299.968846 | 35937013.22 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.733123 | 1216640.38 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.917250 | 364656.25 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.373165 | 421645.62 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 656.986873 | 82123359.12 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.991231 | 218612.98 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.062265 | 79097.89 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.029798 | 125465.59 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.699967 | 151561.98 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 17.873146 | 2234143.25 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 8.973719 | 1121714.88 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.465878 | 1683234.75 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 14.218623 | 1777327.88 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.457029 | 210266.08 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.463131 | 38486.42 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 9.578377 | 149662.14 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2807.976885 | 43874638.83 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 14.423738 | 1802967.25 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.202622 | 525327.75 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.972075 | 996509.38 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 960.003490 | 120000436.25 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 16.869621 | 263587.83 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 18.654562 | 291477.53 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 21.240191 | 331877.98 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 25.438286 | 3179785.75 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 34.450897 | 4306362.12 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 35.352732 | 4419091.50 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 15.717111 | 245579.86 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 17.819975 | 278437.11 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3312.968981 | 51765140.33 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 14.948265 | 1868533.12 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 16.318759 | 2039844.88 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1152.000677 | 144000084.62 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.281182 | 66893.47 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.856860 | 13388.44 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.637190 | 9956.09 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.157472 | 18085.50 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 1.978648 | 247331.00 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.737497 | 92187.12 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.833367 | 104170.88 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.900261 | 112532.62 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.403801 | 37559.39 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.957569 | 14962.02 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.966697 | 15104.64 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 381.981876 | 5968466.81 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.362926 | 170365.75 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.766222 | 95777.75 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.997163 | 124645.38 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 191.981768 | 23997721.00 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.993968 | 187405.75 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 2.753032 | 43016.12 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.962786 | 30668.53 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.090883 | 48295.05 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.123178 | 1640397.25 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.542814 | 567851.75 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.911367 | 363920.88 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 2.984564 | 373070.50 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.235371 | 159927.67 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.766708 | 58854.81 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.898635 | 45291.17 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1151.968638 | 17999509.97 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.496592 | 1062074.00 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.193479 | 274184.88 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.747336 | 218417.00 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 383.998789 | 47999848.62 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.352719 | 193011.23 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.981075 | 62204.30 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.459371 | 54052.67 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.514985 | 54921.64 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.050771 | 1756346.38 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.259451 | 1782431.38 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.830738 | 853842.25 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.898194 | 862274.25 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.690835 | 135794.30 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.371243 | 99550.67 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.700709 | 26573.58 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.983802 | 11999746.91 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.435831 | 929478.88 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.630080 | 703760.00 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.132813 | 391601.62 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.994443 | 11999305.38 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.845180 | 169455.94 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.124433 | 64444.27 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.585226 | 56019.16 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.624400 | 56631.25 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.032337 | 1754042.12 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.744767 | 1843095.88 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.103737 | 887967.12 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.171264 | 896408.00 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.290748 | 129542.94 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.530992 | 102046.75 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.765960 | 27593.12 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.978187 | 11999659.17 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.593536 | 949192.00 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.012715 | 751589.38 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.249199 | 406149.88 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.992881 | 11999110.12 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.754188 | 214909.19 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.181327 | 80958.23 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 7.985114 | 124767.41 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.848705 | 153886.02 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 17.423296 | 2177912.00 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.219449 | 1152431.12 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.434709 | 1679338.62 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 14.139726 | 1767465.75 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.922555 | 201914.92 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.312430 | 36131.72 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 9.170085 | 143282.58 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2807.966513 | 43874476.77 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.755217 | 1344402.12 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.860002 | 482500.25 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.085086 | 1010635.75 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 957.004428 | 119625553.50 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 20.007796 | 312621.81 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 13.907915 | 217311.17 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 11.685696 | 182589.00 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 11.707025 | 182922.27 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 32.102803 | 4012850.38 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 34.318544 | 4289818.00 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 23.262182 | 2907772.75 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 23.217449 | 2902181.12 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.115330 | 204927.03 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 12.920390 | 201881.09 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.653143 | 72705.36 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.970639 | 11999541.23 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 16.664401 | 2083050.13 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.106080 | 1513260.00 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.992786 | 1124098.25 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.993352 | 11999169.00 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 15.186275 | 237285.55 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 16.695320 | 260864.37 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 47.875398 | 748053.09 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 86.283704 | 1348182.88 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.569227 | 2321153.38 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 43.779934 | 5472491.75 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 145.599248 | 18199906.00 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 228.927154 | 28615894.25 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.602572 | 165665.19 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 16.043025 | 250672.27 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 54.481369 | 851271.39 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5952.002392 | 93000037.38 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.927255 | 1240906.88 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 18.645478 | 2330684.75 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 70.018950 | 8752368.75 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1234.998320 | 154374790.00 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 14.387480 | 899217.50 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 11.962614 | 747663.38 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 207.444776 | 12965298.50 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 178.002358 | 11125147.38 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 9.802564 | 612660.25 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 23.600096 | 1475006.00 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 109.556973 | 6847310.81 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 3127.069798 | 195441862.38 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.023654 | 1478.38 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002465 | 154.06 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.023925 | 1495.31 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002405 | 150.31 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000121 | 7.56 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.022322 | 1395.12 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.007033 | 439.56 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000290 | 18.12 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.022863 | 1428.94 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.006812 | 425.75 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000170 | 10.62 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.023213 | 1450.81 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.005420 | 338.75 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000161 | 10.06 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.023194 | 1449.62 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.004889 | 305.56 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000110 | 6.88 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.020829 | 1301.81 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003006 | 187.88 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000130 | 8.12 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.020929 | 1308.06 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003296 | 206.00 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/norm_fro` (c64, threads=1, shape=`2048x1536`): `tenferro-direct` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 24.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `jax-cpu` is 24.2x faster than the slowest successful cell (`tenferro-trace`, 59.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `julia-base` is 14.7x faster than the slowest successful cell (`tenferro-trace`, 59.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 64.9x faster than the slowest successful cell (`tenferro-trace`, 59.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 20.5x faster than the slowest successful cell (`tenferro-direct`, 140.009 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `julia-base` is 19.1x faster than the slowest successful cell (`tenferro-direct`, 140.009 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 24.7x faster than the slowest successful cell (`tenferro-direct`, 140.009 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 31.8x faster than the slowest successful cell (`tenferro-trace`, 145.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `julia-base` is 34.2x faster than the slowest successful cell (`tenferro-trace`, 145.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 40.4x faster than the slowest successful cell (`tenferro-trace`, 145.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/masked_log_softmax_axis1` (f64, threads=4, shape=`1024x64`): `jax-cpu` is 231.1x faster than the slowest successful cell (`tenferro-eager`, 51.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/masked_log_softmax_axis1` (f64, threads=4, shape=`1024x64`): `tenferro-direct` is 133.0x faster than the slowest successful cell (`tenferro-eager`, 51.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 24.217 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 14.6x faster than the slowest successful cell (`pytorch-cpu`, 24.217 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 24.217 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 12.2x faster than the slowest successful cell (`pytorch-cpu`, 24.217 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 17.5x faster than the slowest successful cell (`tenferro-trace`, 23.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 16.4x faster than the slowest successful cell (`tenferro-trace`, 23.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 30.5x faster than the slowest successful cell (`tenferro-trace`, 23.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 17.2x faster than the slowest successful cell (`tenferro-trace`, 24.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 14.4x faster than the slowest successful cell (`tenferro-trace`, 24.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`tenferro-trace`, 24.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 18.3x faster than the slowest successful cell (`tenferro-trace`, 24.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 15.3x faster than the slowest successful cell (`pytorch-cpu`, 23.428 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 23.428 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 23.428 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 16.8x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 16.2x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 30.1x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 13.4x faster than the slowest successful cell (`tenferro-trace`, 24.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 21.8x faster than the slowest successful cell (`tenferro-trace`, 24.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 25.9x faster than the slowest successful cell (`tenferro-trace`, 24.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 49.1x faster than the slowest successful cell (`tenferro-trace`, 24.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 14.5x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 19.4x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 67.1x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 33.5x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 13.1x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 26.3x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 25.5x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 51.3x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 15.7x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 18.7x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 55.6x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 36.9x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_squares_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 57.8x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `jax-cpu` is 11.4x faster than the slowest successful cell (`tenferro-trace`, 12.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `pytorch-cpu` is 31.3x faster than the slowest successful cell (`tenferro-trace`, 12.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `tenferro-direct` is 37.1x faster than the slowest successful cell (`tenferro-trace`, 12.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `jax-cpu` is 16.6x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `pytorch-cpu` is 31.9x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `tenferro-direct` is 33.1x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 12.0x faster than the slowest successful cell (`julia-base`, 8.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 13.3x faster than the slowest successful cell (`julia-base`, 8.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `jax-cpu` is 118.0x faster than the slowest successful cell (`tenferro-trace`, 48.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `julia-base` is 36.6x faster than the slowest successful cell (`tenferro-trace`, 48.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 137.6x faster than the slowest successful cell (`tenferro-trace`, 48.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `tenferro-direct` is 109.1x faster than the slowest successful cell (`tenferro-trace`, 48.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `jax-cpu` is 11.0x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 22.2x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 31.2x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `jax-cpu` is 19.8x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 29.8x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 32.9x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `jax-cpu` is 65.4x faster than the slowest successful cell (`tenferro-trace`, 36.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `julia-base` is 26.4x faster than the slowest successful cell (`tenferro-trace`, 36.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 57.1x faster than the slowest successful cell (`tenferro-trace`, 36.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `tenferro-direct` is 34.0x faster than the slowest successful cell (`tenferro-trace`, 36.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 5053.7x faster than the slowest successful cell (`julia-base`, 9.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `pytorch-cpu` is 10.8x faster than the slowest successful cell (`julia-base`, 9.596 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-direct` is 14.0x faster than the slowest successful cell (`julia-base`, 9.596 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 5263.6x faster than the slowest successful cell (`julia-base`, 9.596 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 15.1x faster than the slowest successful cell (`jax-cpu`, 0.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 14.2x faster than the slowest successful cell (`jax-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 11.7x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x8x8`): `tenferro-direct` is 12.3x faster than the slowest successful cell (`jax-cpu`, 0.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`3x2x2`): `tenferro-direct` is 11.1x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`3x4x4`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x2x2`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x4x4`): `tenferro-direct` is 13.7x faster than the slowest successful cell (`jax-cpu`, 0.018 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 21.9x faster than the slowest successful cell (`jax-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 11.1x faster than the slowest successful cell (`jax-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x2x2`): `tenferro-direct` is 11.0x faster than the slowest successful cell (`jax-cpu`, 0.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 19.1x faster than the slowest successful cell (`jax-cpu`, 0.055 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-trace` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.055 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=1, shape=`1x2x2`): `tenferro-direct` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.024 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 16.4x faster than the slowest successful cell (`jax-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-trace` is 11.3x faster than the slowest successful cell (`jax-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 15.0x faster than the slowest successful cell (`jax-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 15.0x faster than the slowest successful cell (`jax-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-trace` is 10.0x faster than the slowest successful cell (`jax-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.020 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x2x2,rhs=1`): `tenferro-direct` is 11.0x faster than the slowest successful cell (`jax-cpu`, 0.019 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`3x2x2,rhs=1`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.021 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`3x4x4,rhs=1`): `tenferro-direct` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.023 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `pytorch-cpu` is 15.7x faster than the slowest successful cell (`tenferro-trace`, 5.991 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `tenferro-direct` is 11.2x faster than the slowest successful cell (`tenferro-trace`, 5.991 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x4x4,rhs=1`): `tenferro-direct` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.026 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x8x8,rhs=1`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.026 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`3x2x2,rhs=1`): `tenferro-direct` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.029 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `jax-cpu` is 32.3x faster than the slowest successful cell (`tenferro-trace`, 23.989 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `pytorch-cpu` is 41.3x faster than the slowest successful cell (`tenferro-trace`, 23.989 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `tenferro-direct` is 25.6x faster than the slowest successful cell (`tenferro-trace`, 23.989 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 22.7x faster than the slowest successful cell (`tenferro-trace`, 319.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `julia-base` is 25.6x faster than the slowest successful cell (`tenferro-trace`, 319.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 28.5x faster than the slowest successful cell (`tenferro-trace`, 319.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`768x384`): `jax-cpu` is 243.3x faster than the slowest successful cell (`tenferro-eager`, 2306.347 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 402.2x faster than the slowest successful cell (`tenferro-eager`, 2306.347 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 14.0x faster than the slowest successful cell (`tenferro-trace`, 280.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `julia-base` is 16.9x faster than the slowest successful cell (`tenferro-trace`, 280.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 12.8x faster than the slowest successful cell (`tenferro-trace`, 280.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `jax-cpu` is 42.1x faster than the slowest successful cell (`tenferro-trace`, 2287.692 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `julia-base` is 245.9x faster than the slowest successful cell (`tenferro-trace`, 2287.692 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 370.8x faster than the slowest successful cell (`tenferro-trace`, 2287.692 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 14.8x faster than the slowest successful cell (`tenferro-trace`, 236.221 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 18.7x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 53.6x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `jax-cpu` is 12.2x faster than the slowest successful cell (`tenferro-trace`, 233.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `julia-base` is 10.7x faster than the slowest successful cell (`tenferro-trace`, 233.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 14.2x faster than the slowest successful cell (`tenferro-trace`, 233.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `jax-cpu` is 11.4x faster than the slowest successful cell (`tenferro-trace`, 235.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `julia-base` is 11.7x faster than the slowest successful cell (`tenferro-trace`, 235.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 14.1x faster than the slowest successful cell (`tenferro-trace`, 235.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 23.9x faster than the slowest successful cell (`tenferro-trace`, 334.009 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `julia-base` is 26.5x faster than the slowest successful cell (`tenferro-trace`, 334.009 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 29.5x faster than the slowest successful cell (`tenferro-trace`, 334.009 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 45.0x faster than the slowest successful cell (`tenferro-trace`, 2337.669 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `julia-base` is 44.4x faster than the slowest successful cell (`tenferro-trace`, 2337.669 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 61.0x faster than the slowest successful cell (`tenferro-trace`, 2337.669 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 48.0x faster than the slowest successful cell (`tenferro-eager`, 5.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 76.4x faster than the slowest successful cell (`tenferro-eager`, 5.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 82.3x faster than the slowest successful cell (`tenferro-eager`, 5.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 35.4x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 23.0x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 26.9x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 75.3x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 130.7x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 319.5x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 10.7x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 16.0x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 21.5x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 85.3x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 142.6x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 157.6x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 12.3x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 198.9x faster than the slowest successful cell (`tenferro-eager`, 35.937 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 858.0x faster than the slowest successful cell (`tenferro-eager`, 35.937 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 403.7x faster than the slowest successful cell (`tenferro-eager`, 35.937 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 67.5x faster than the slowest successful cell (`tenferro-eager`, 82.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 225.2x faster than the slowest successful cell (`tenferro-eager`, 82.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 194.8x faster than the slowest successful cell (`tenferro-eager`, 82.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 208.7x faster than the slowest successful cell (`tenferro-eager`, 43.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 1140.0x faster than the slowest successful cell (`tenferro-eager`, 43.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 293.2x faster than the slowest successful cell (`tenferro-eager`, 43.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 66.6x faster than the slowest successful cell (`tenferro-eager`, 120.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 228.4x faster than the slowest successful cell (`tenferro-eager`, 120.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 120.4x faster than the slowest successful cell (`tenferro-eager`, 120.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 210.8x faster than the slowest successful cell (`tenferro-eager`, 51.765 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 185.9x faster than the slowest successful cell (`tenferro-eager`, 51.765 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 77.1x faster than the slowest successful cell (`tenferro-eager`, 144.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 70.6x faster than the slowest successful cell (`tenferro-eager`, 144.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 158.9x faster than the slowest successful cell (`tenferro-eager`, 5.968 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 398.9x faster than the slowest successful cell (`tenferro-eager`, 5.968 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 395.1x faster than the slowest successful cell (`tenferro-eager`, 5.968 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 140.9x faster than the slowest successful cell (`tenferro-eager`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 250.6x faster than the slowest successful cell (`tenferro-eager`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 192.5x faster than the slowest successful cell (`tenferro-eager`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 112.5x faster than the slowest successful cell (`tenferro-eager`, 18.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 305.8x faster than the slowest successful cell (`tenferro-eager`, 18.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 397.4x faster than the slowest successful cell (`tenferro-eager`, 18.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 45.2x faster than the slowest successful cell (`tenferro-eager`, 48.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 175.1x faster than the slowest successful cell (`tenferro-eager`, 48.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 219.8x faster than the slowest successful cell (`tenferro-eager`, 48.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 88.4x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 120.5x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 451.6x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 12.9x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 17.1x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 30.6x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 92.6x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 117.6x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 434.9x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 12.6x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 16.0x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 29.5x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 217.3x faster than the slowest successful cell (`tenferro-eager`, 43.874 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 1214.3x faster than the slowest successful cell (`tenferro-eager`, 43.874 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 306.2x faster than the slowest successful cell (`tenferro-eager`, 43.874 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 89.0x faster than the slowest successful cell (`tenferro-eager`, 119.626 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 247.9x faster than the slowest successful cell (`tenferro-eager`, 119.626 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 118.4x faster than the slowest successful cell (`tenferro-eager`, 119.626 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 58.6x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 59.4x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 165.0x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 12.3x faster than the slowest successful cell (`tenferro-eager`, 28.616 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 561.4x faster than the slowest successful cell (`tenferro-eager`, 93.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 371.0x faster than the slowest successful cell (`tenferro-eager`, 93.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 109.2x faster than the slowest successful cell (`tenferro-eager`, 93.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 124.4x faster than the slowest successful cell (`tenferro-eager`, 154.375 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 66.2x faster than the slowest successful cell (`tenferro-eager`, 154.375 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 17.6x faster than the slowest successful cell (`tenferro-eager`, 154.375 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/output_reuse/solve_read_into` (f64, threads=4, shape=`768x768,rhs=16`): `pytorch-cpu` is 28.5x faster than the slowest successful cell (`tenferro-direct`, 195.442 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 74411.2x faster than the slowest successful cell (`julia-base`, 130.443 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 73614.0x faster than the slowest successful cell (`julia-base`, 133.536 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/cast_f64_f32` (f64->f32, threads=4, shape=`33554432`): `pytorch-cpu` is 11.0x faster than the slowest successful cell (`tenferro-trace`, 253.842 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/cast_f64_f32` (f64->f32, threads=4, shape=`33554432`): `tenferro-direct` is 11.1x faster than the slowest successful cell (`tenferro-trace`, 253.842 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 84022.3x faster than the slowest successful cell (`julia-base`, 153.257 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 85910.3x faster than the slowest successful cell (`julia-base`, 154.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 131417.0x faster than the slowest successful cell (`julia-base`, 214.604 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 130526.3x faster than the slowest successful cell (`julia-base`, 214.455 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 174.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 79.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 131.9x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 145.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 186.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 163.5x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 9347.3 GB/s at 0.002 ms. The conservative physical lower bound is 0.168 ms, so the cell is 93.5x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 9198.6 GB/s at 0.002 ms. The conservative physical lower bound is 0.042 ms, so the cell is 23.0x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 153129.2 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1531.3x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 147979.9 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 369.9x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 147168.6 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1471.7x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 149629.6 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 374.1x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 82190.9 GB/s at 0.002 ms. The conservative physical lower bound is 1.342 ms, so the cell is 821.9x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 81690.6 GB/s at 0.002 ms. The conservative physical lower bound is 0.336 ms, so the cell is 204.2x faster than the bound.

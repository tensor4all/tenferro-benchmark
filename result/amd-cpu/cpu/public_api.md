# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`
- Run metadata: `data/results/amd-cpu/cpu/public_api/20261021_185806/run.yaml`
- Timestamp: `20261021_185806`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Sampling: `quick` profile, 7 measured runs, 3 warmups per row

This file is generated from sequential CPU public API runs under `data/results/amd-cpu/cpu/public_api/20261021_185806`.

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

- Run metadata: `data/results/amd-cpu/cpu/public_api/20261021_185806/run_t1.yaml`
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

- Run metadata: `data/results/amd-cpu/cpu/public_api/20261021_185806/run_t4.yaml`
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

- CSV: `data/results/amd-cpu/cpu/public_api/20261021_185806/cpu_public_api_t1_20261021_185806.csv`
- CSV: `data/results/amd-cpu/cpu/public_api/20261021_185806/cpu_public_api_t4_20261021_185806.csv`
- Source table: `data/results/amd-cpu/cpu/public_api/20261021_185806/cpu_public_api_20261021_185806.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 4.156 ± 0.062 | - | 4.303 ± 0.074 | 4.503 ± 0.115 | 3.833 ± 0.043 | 3.656 ± 0.036 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.418 ± 0.038 | - | 2.613 ± 0.040 | 2.275 ± 0.069 | 3.191 ± 0.505 | 4.071 ± 2.070 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 26.365 ± 0.742 | - | 27.418 ± 1.002 | 25.390 ± 0.604 | 26.658 ± 0.355 | 29.625 ± 0.696 | 32.193 ± 0.742 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 16.474 ± 0.372 | - | 15.958 ± 0.338 | 17.169 ± 0.677 | 15.947 ± 0.183 | 29.269 ± 0.380 | 17.356 ± 0.323 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 57.639 ± 0.126 | - | 57.574 ± 0.174 | 43.257 ± 0.304 | 38.323 ± 0.146 | 70.537 ± 0.322 | 92.208 ± 0.273 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 27.273 ± 1.131 | - | 27.182 ± 0.547 | 17.862 ± 0.377 | 21.554 ± 0.108 | 70.605 ± 0.247 | 38.222 ± 0.346 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 43.305 ± 0.193 | - | 43.560 ± 0.418 | 42.846 ± 0.362 | 66.640 ± 0.590 | 45.181 ± 0.548 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 29.776 ± 0.600 | - | 29.463 ± 0.096 | 29.641 ± 3.082 | 30.039 ± 0.768 | 22.887 ± 0.274 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 44.745 ± 0.502 | - | unsupported | 42.800 ± 0.323 | 67.253 ± 0.524 | 44.862 ± 3.932 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 30.196 ± 0.186 | - | unsupported | 29.536 ± 0.217 | 30.204 ± 0.873 | 22.948 ± 1.734 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 10.904 ± 0.024 | - | 11.021 ± 0.041 | 11.576 ± 0.028 | 11.883 ± 0.036 | 11.897 ± 0.096 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 14.567 ± 0.071 | - | 14.628 ± 0.063 | 14.021 ± 0.130 | 17.920 ± 0.551 | 17.769 ± 0.115 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 71.540 ± 0.174 | - | 71.546 ± 0.551 | 85.689 ± 0.182 | 79.403 ± 0.358 | 56.117 ± 0.357 | 56.248 ± 0.584 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 31.290 ± 0.149 | - | 31.194 ± 0.150 | 36.806 ± 0.190 | 39.978 ± 0.451 | 56.760 ± 0.443 | 25.776 ± 1.137 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 117.027 ± 0.254 | - | 115.996 ± 0.279 | 131.071 ± 0.348 | 152.071 ± 0.931 | 194.987 ± 0.449 | 125.678 ± 0.995 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 49.663 ± 0.142 | - | 49.553 ± 0.121 | 56.626 ± 0.137 | 65.983 ± 1.035 | 199.629 ± 0.390 | 51.232 ± 0.649 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 20.409 ± 0.835 | - | 18.787 ± 0.436 | 17.301 ± 0.379 | 18.005 ± 0.457 | 17.992 ± 0.321 | 19.546 ± 0.253 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 13.277 ± 0.775 | - | 11.981 ± 0.690 | 12.526 ± 0.167 | 11.524 ± 0.409 | 17.992 ± 0.177 | 12.608 ± 0.238 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 2.360 ± 0.089 | - | 20.867 ± 0.132 | 24.213 ± 0.069 | 4.495 ± 0.353 | 4.073 ± 0.056 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 2.707 ± 0.203 | - | 17.056 ± 6.813 | 11.304 ± 0.112 | 2.391 ± 0.342 | 4.070 ± 0.064 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 8.301 ± 0.031 | - | 8.596 ± 0.076 | 9.166 ± 0.099 | 8.083 ± 0.101 | 9.408 ± 0.626 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 5.616 ± 0.058 | - | 5.888 ± 0.020 | 5.785 ± 0.126 | 6.873 ± 0.067 | 7.883 ± 0.183 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 5.360 ± 0.129 | - | 5.457 ± 0.219 | 5.393 ± 0.041 | 5.089 ± 0.066 | 4.821 ± 0.033 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.766 ± 0.155 | - | 3.926 ± 0.117 | 3.642 ± 0.033 | 4.882 ± 1.750 | 4.337 ± 0.100 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 6.526 ± 0.066 | - | 6.504 ± 0.051 | 7.673 ± 0.080 | 5.382 ± 1.243 | 5.265 ± 0.090 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 5.749 ± 0.141 | - | 5.751 ± 0.067 | 5.814 ± 0.052 | 6.042 ± 0.102 | 5.621 ± 0.102 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 43.909 ± 0.500 | - | 45.581 ± 3.915 | 42.697 ± 0.288 | 66.633 ± 0.400 | 44.211 ± 0.313 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 29.681 ± 0.209 | - | 29.090 ± 0.533 | 29.294 ± 0.524 | 29.412 ± 0.617 | 22.400 ± 0.348 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 49.516 ± 0.672 | - | unsupported | 51.606 ± 0.155 | 60.985 ± 3.343 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 25.214 ± 0.191 | - | unsupported | 26.301 ± 0.131 | 34.639 ± 0.339 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 70.349 ± 1.489 | - | 114.632 ± 0.891 | 25.302 ± 0.825 | 27.437 ± 0.604 | 26.666 ± 0.769 | 38.111 ± 1.172 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 16.457 ± 0.542 | - | 16.074 ± 0.563 | 16.658 ± 0.579 | 16.764 ± 0.283 | 26.737 ± 0.708 | 18.286 ± 0.624 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 31.652 ± 0.643 | - | 88.754 ± 0.525 | 32.064 ± 0.568 | 33.449 ± 1.097 | 33.878 ± 0.445 | 43.292 ± 0.422 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 23.182 ± 0.321 | - | 22.674 ± 0.394 | 23.179 ± 0.621 | 22.586 ± 0.926 | 33.922 ± 1.065 | 24.496 ± 0.998 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 26.698 ± 0.343 | - | 55.256 ± 0.309 | 37.448 ± 0.439 | 38.597 ± 0.035 | 69.714 ± 1.303 | 74.797 ± 3.143 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 12.358 ± 0.181 | - | 25.730 ± 0.276 | 20.034 ± 0.165 | 19.876 ± 0.034 | 72.044 ± 0.538 | 30.135 ± 3.046 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 8.220 ± 0.278 | - | 8.343 ± 0.150 | 11.259 ± 0.546 | 9.976 ± 0.234 | 11.502 ± 0.794 | 15.000 ± 1.567 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 6.603 ± 0.264 | - | 6.981 ± 0.464 | 8.174 ± 0.091 | 7.291 ± 1.506 | 11.070 ± 1.193 | 8.961 ± 8.165 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 17.521 ± 0.386 | - | 20.273 ± 0.818 | 25.793 ± 0.377 | 21.608 ± 1.017 | 17.996 ± 0.718 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 13.578 ± 0.259 | - | 13.489 ± 0.314 | 15.523 ± 0.292 | 15.625 ± 0.406 | 17.875 ± 0.653 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 28.671 ± 0.050 | - | 28.772 ± 0.040 | 32.505 ± 0.107 | 83.871 ± 0.183 | 44.817 ± 1.451 | 48.225 ± 1.514 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 12.074 ± 0.113 | - | 12.271 ± 0.109 | 15.684 ± 0.149 | 42.374 ± 0.069 | 45.007 ± 4.722 | 23.884 ± 1.590 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 77.948 ± 0.545 | - | 123.012 ± 0.509 | 31.095 ± 0.500 | 33.399 ± 0.657 | 33.774 ± 0.844 | 43.614 ± 0.862 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 22.990 ± 0.604 | - | 22.907 ± 0.862 | 22.927 ± 0.370 | 22.833 ± 0.570 | 32.184 ± 1.099 | 24.107 ± 1.535 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 20.416 ± 0.031 | - | 20.851 ± 0.039 | 20.264 ± 0.291 | 16.825 ± 0.122 | 39.679 ± 1.389 | 44.569 ± 1.362 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 8.443 ± 0.057 | - | 8.502 ± 0.051 | 10.253 ± 0.309 | 8.924 ± 0.094 | 39.931 ± 1.351 | 21.518 ± 2.613 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 20.413 ± 0.030 | - | 20.638 ± 0.118 | 20.837 ± 0.219 | 15.571 ± 0.141 | 26.923 ± 2.654 | 30.001 ± 3.123 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 8.175 ± 0.023 | - | 8.193 ± 0.111 | 9.264 ± 0.096 | 8.186 ± 0.078 | 26.871 ± 0.992 | 13.181 ± 3.157 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 25.537 ± 1.656 | - | 25.185 ± 0.069 | 26.744 ± 0.113 | 40.317 ± 0.194 | 67.915 ± 10.063 | 76.204 ± 1.142 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 10.512 ± 0.102 | - | 10.708 ± 0.021 | 12.792 ± 0.205 | 20.568 ± 0.051 | 60.974 ± 1.926 | 30.830 ± 1.689 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 13.355 ± 0.036 | - | 13.314 ± 0.043 | 21.331 ± 0.097 | 25.663 ± 0.361 | 35.128 ± 1.272 | 39.012 ± 2.898 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 5.660 ± 0.003 | - | 5.744 ± 0.032 | 9.506 ± 0.069 | 13.617 ± 0.083 | 35.241 ± 0.074 | 17.676 ± 2.723 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.311 ± 0.013 | 0.351 ± 0.012 | unsupported | unsupported | 0.375 ± 0.006 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.412 ± 0.025 | 0.559 ± 0.017 | unsupported | unsupported | 0.301 ± 0.011 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 78.042 ± 1.348 | - | 125.348 ± 0.490 | 32.523 ± 0.542 | 34.438 ± 0.230 | 34.727 ± 0.637 | 46.174 ± 0.250 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 22.937 ± 0.524 | - | 22.903 ± 0.200 | 22.797 ± 1.483 | 23.880 ± 0.934 | 34.008 ± 1.154 | 24.383 ± 1.062 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 76.171 ± 1.325 | - | 127.306 ± 0.960 | 32.920 ± 0.356 | 34.569 ± 0.817 | 33.908 ± 0.457 | 46.075 ± 0.472 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 22.867 ± 0.572 | - | 23.174 ± 0.753 | 22.592 ± 0.578 | 23.988 ± 0.675 | 35.308 ± 1.426 | 24.340 ± 0.953 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 32.562 ± 0.759 | - | 123.258 ± 0.724 | 32.087 ± 0.358 | 34.349 ± 1.322 | 33.011 ± 0.861 | 43.578 ± 0.572 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 23.112 ± 0.939 | - | 22.475 ± 1.576 | 22.989 ± 1.368 | 22.516 ± 0.391 | 33.107 ± 0.939 | 23.562 ± 0.992 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 68.606 ± 0.562 | - | 115.299 ± 0.976 | 25.570 ± 1.036 | 27.444 ± 0.685 | 27.251 ± 3.252 | 37.801 ± 0.950 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 16.565 ± 0.391 | - | 15.883 ± 0.232 | 16.633 ± 0.986 | 16.965 ± 0.196 | 26.873 ± 1.434 | 18.182 ± 0.626 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 58.029 ± 0.114 | - | 58.189 ± 0.205 | 51.603 ± 0.093 | 61.592 ± 0.189 | 115.033 ± 3.696 | 111.303 ± 4.208 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 23.719 ± 0.241 | - | 23.725 ± 0.150 | 19.878 ± 0.093 | 31.082 ± 0.070 | 114.693 ± 0.253 | 47.743 ± 2.842 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 15.555 ± 1.277 | - | 16.676 ± 2.189 | 10.998 ± 0.150 | 30.152 ± 0.366 | 12.654 ± 0.378 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 6.313 ± 0.207 | - | 6.465 ± 0.248 | 5.952 ± 0.130 | 15.230 ± 0.107 | 12.589 ± 0.108 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.815 ± 0.024 | - | 1.774 ± 0.068 | 38.985 ± 0.802 | 2.622 ± 0.059 | 1.274 ± 0.195 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.802 ± 0.029 | - | 0.825 ± 0.039 | 11.823 ± 0.432 | 1.706 ± 0.313 | 1.246 ± 0.344 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 1.733 ± 0.042 | - | 1.800 ± 0.096 | 4.083 ± 0.102 | 2.931 ± 0.472 | 1.468 ± 0.133 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 0.896 ± 0.187 | - | 0.872 ± 0.132 | 2.305 ± 0.027 | 1.199 ± 0.034 | 1.171 ± 0.309 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 14.565 ± 0.212 | - | 14.564 ± 0.240 | 11.063 ± 0.292 | 30.028 ± 0.145 | 13.067 ± 0.387 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 6.198 ± 0.139 | - | 6.186 ± 0.087 | 6.052 ± 0.162 | 15.393 ± 0.264 | 12.952 ± 0.607 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.807 ± 0.078 | - | 1.758 ± 0.066 | 31.745 ± 0.283 | 2.626 ± 0.022 | 1.079 ± 0.269 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.771 ± 0.021 | - | 0.798 ± 0.034 | 11.569 ± 0.340 | 1.747 ± 0.161 | 1.244 ± 0.163 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 7.283 ± 0.074 | - | 7.260 ± 0.300 | 16.057 ± 0.038 | 13.068 ± 0.591 | 6.177 ± 0.270 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 4.001 ± 0.169 | - | 4.032 ± 0.238 | 8.732 ± 0.156 | 8.849 ± 0.646 | 6.177 ± 0.194 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 10.042 ± 1.088 | - | 10.337 ± 0.475 | 9.520 ± 0.357 | 29.262 ± 0.317 | 9.867 ± 0.175 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 5.999 ± 0.132 | - | 6.000 ± 0.135 | 5.932 ± 0.079 | 14.887 ± 0.207 | 10.312 ± 0.173 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.950 ± 0.075 | - | 1.140 ± 0.087 | 2.270 ± 0.387 | 2.209 ± 0.049 | 0.925 ± 0.335 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.393 ± 0.050 | - | 0.380 ± 0.043 | 0.842 ± 0.058 | 1.270 ± 0.070 | 0.883 ± 0.343 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.036 ± 0.114 | - | 1.188 ± 0.255 | 0.759 ± 0.047 | 3.911 ± 0.284 | 0.910 ± 0.094 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.507 ± 0.031 | - | 0.588 ± 0.053 | 0.361 ± 0.033 | 2.060 ± 0.272 | 1.053 ± 0.156 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 9.998 ± 0.390 | - | 10.382 ± 0.529 | 9.192 ± 0.202 | 29.090 ± 0.865 | 9.998 ± 0.327 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 6.012 ± 0.140 | - | 6.245 ± 0.222 | 6.141 ± 0.211 | 14.927 ± 0.549 | 10.034 ± 0.466 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.955 ± 0.101 | - | 0.939 ± 0.187 | 2.180 ± 0.126 | 2.200 ± 0.040 | 1.040 ± 0.039 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.520 ± 0.055 | - | 0.442 ± 0.075 | 0.876 ± 0.110 | 2.374 ± 1.657 | 0.867 ± 0.289 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.069 ± 0.230 | - | 1.044 ± 0.044 | 0.807 ± 0.033 | 3.911 ± 0.420 | 1.123 ± 0.335 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.500 ± 0.048 | - | 0.593 ± 0.046 | 0.450 ± 0.101 | 2.028 ± 0.233 | 1.026 ± 0.128 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 1.091 ± 0.099 | 1.072 ± 0.093 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.384 ± 0.077 | 0.427 ± 0.048 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 36.565 ± 1.051 | - | 37.085 ± 4.292 | 30.998 ± 0.086 | 43.716 ± 0.236 | 47.647 ± 0.721 | 54.919 ± 0.790 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 17.846 ± 0.267 | - | 17.789 ± 0.033 | 14.189 ± 0.128 | 24.650 ± 0.527 | 48.040 ± 1.060 | 26.817 ± 1.158 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 137.636 ± 1.202 | - | 137.280 ± 0.999 | 49.461 ± 0.348 | 154.051 ± 0.353 | 154.761 ± 0.199 | 154.835 ± 0.297 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 21.349 ± 3.642 | - | 22.765 ± 3.310 | 24.432 ± 2.705 | 77.929 ± 0.156 | 154.779 ± 1.118 | 74.443 ± 3.091 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 90.580 ± 1.291 | - | 128.190 ± 0.512 | 40.710 ± 10.253 | 37.208 ± 0.475 | 34.447 ± 0.951 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 25.128 ± 0.412 | - | 26.612 ± 1.568 | 24.175 ± 0.782 | 25.738 ± 1.082 | 35.012 ± 1.031 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 72.811 ± 0.837 | - | 119.833 ± 1.224 | 28.054 ± 0.409 | 29.887 ± 0.588 | 28.555 ± 0.338 | 34.940 ± 1.513 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 18.108 ± 0.806 | - | 16.017 ± 0.272 | 16.199 ± 0.411 | 17.661 ± 0.837 | 28.868 ± 0.894 | 17.656 ± 0.388 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 24.796 ± 0.049 | - | 24.840 ± 0.093 | 34.508 ± 0.107 | 83.219 ± 0.202 | 41.957 ± 0.956 | 45.054 ± 1.694 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 10.364 ± 0.040 | - | 10.561 ± 0.231 | 16.509 ± 0.087 | 42.112 ± 0.190 | 41.459 ± 2.324 | 22.594 ± 1.327 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 126.391 ± 0.670 | - | 126.194 ± 0.746 | 89.446 ± 0.456 | 38.249 ± 0.147 | 104.791 ± 0.413 | 109.629 ± 0.127 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 17.956 ± 1.312 | - | 18.631 ± 0.400 | 40.594 ± 1.048 | 20.573 ± 0.133 | 104.708 ± 0.609 | 52.198 ± 3.520 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 32.574 ± 0.429 | - | 123.246 ± 1.361 | 32.763 ± 0.773 | 34.201 ± 0.371 | 33.443 ± 1.408 | 42.950 ± 0.874 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 23.367 ± 1.048 | - | 23.345 ± 1.845 | 23.371 ± 0.357 | 23.081 ± 0.936 | 32.971 ± 0.824 | 24.517 ± 1.329 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 51.376 ± 0.097 | - | 51.673 ± 0.046 | 53.150 ± 0.196 | 14.921 ± 0.194 | 57.502 ± 4.602 | 71.377 ± 21.613 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 20.522 ± 0.070 | - | 20.621 ± 0.284 | 23.068 ± 0.162 | 7.844 ± 0.065 | 58.066 ± 3.070 | 27.657 ± 1.933 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 123.869 ± 1.747 | 126.387 ± 0.975 | unsupported | 40.658 ± 3.566 | 36.997 ± 0.683 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 25.941 ± 0.308 | 33.593 ± 11.497 | unsupported | 24.425 ± 1.840 | 24.716 ± 0.896 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.732 ± 0.165 | - | 1.680 ± 0.053 | 0.634 ± 0.036 | 0.762 ± 0.310 | 2.126 ± 0.722 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.400 ± 0.186 | - | 0.913 ± 0.162 | 0.324 ± 0.041 | 0.697 ± 0.117 | 2.143 ± 0.717 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.572 ± 0.068 | - | 2.805 ± 0.257 | 0.881 ± 0.161 | 0.807 ± 0.311 | 1.089 ± 0.351 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.325 ± 0.042 | - | 2.013 ± 0.174 | 0.291 ± 0.083 | 0.439 ± 0.341 | 2.373 ± 0.728 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.201 ± 0.101 | - | unsupported | 1.131 ± 0.107 | 1.167 ± 0.060 | 2.543 ± 0.727 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.615 ± 0.152 | - | unsupported | 0.498 ± 0.119 | 0.895 ± 0.107 | 2.638 ± 0.783 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.415 ± 0.041 | - | 0.567 ± 0.055 | 0.687 ± 0.014 | 0.486 ± 0.057 | 1.183 ± 0.087 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.354 ± 0.048 | - | 0.515 ± 0.018 | 0.344 ± 0.034 | 0.382 ± 0.006 | 1.165 ± 0.795 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.607 ± 0.044 | - | 1.599 ± 0.101 | 0.892 ± 0.081 | 0.982 ± 0.071 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.290 ± 0.072 | - | 0.880 ± 0.085 | 0.413 ± 0.027 | 1.063 ± 0.015 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.717 ± 0.164 | - | 1.705 ± 0.087 | 0.639 ± 0.086 | 0.690 ± 0.084 | 1.516 ± 1.522 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.321 ± 0.049 | - | 0.894 ± 0.138 | 0.273 ± 0.034 | 0.636 ± 0.258 | 2.676 ± 1.718 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.811 ± 0.016 | - | 1.084 ± 0.019 | 0.452 ± 0.041 | 0.530 ± 0.036 | 1.289 ± 0.151 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.881 ± 0.078 | - | 1.242 ± 0.050 | 0.613 ± 0.014 | 0.503 ± 0.029 | 1.293 ± 0.033 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.583 ± 0.046 | - | 0.003 ± 0.000 | 1.714 ± 0.144 | 1.318 ± 0.092 | 3.681 ± 0.394 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.665 ± 0.067 | - | 0.003 ± 0.000 | 0.684 ± 0.043 | 0.891 ± 0.061 | 3.423 ± 1.248 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.053 ± 0.005 | - | 0.058 ± 0.001 | 0.076 ± 0.005 | 0.124 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.093 ± 0.001 | - | 0.102 ± 0.000 | 0.127 ± 0.004 | 0.184 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.240 ± 0.020 | - | 0.267 ± 0.004 | 0.276 ± 0.012 | 0.350 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.082 ± 0.002 | - | 0.091 ± 0.002 | 0.149 ± 0.006 | 0.151 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.001 | 0.011 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.001 | 0.068 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.001 | 0.066 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.674 ± 0.014 | - | 0.726 ± 0.011 | 1.157 ± 0.004 | 0.732 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.002 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.026 ± 0.000 | - | 0.033 ± 0.000 | 0.039 ± 0.000 | 0.094 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.001 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.065 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.052 ± 0.000 | - | 0.058 ± 0.000 | 0.086 ± 0.002 | 0.116 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.093 ± 0.003 | - | 0.102 ± 0.000 | 0.124 ± 0.010 | 0.160 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.241 ± 0.032 | - | 0.331 ± 0.042 | 0.340 ± 0.017 | 0.358 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.084 ± 0.003 | - | 0.091 ± 0.006 | 0.181 ± 0.003 | 0.135 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.003 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.008 ± 0.004 | - | 0.009 ± 0.000 | 0.016 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.002 ± 0.001 | - | 0.006 ± 0.008 | 0.010 ± 0.000 | 0.011 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.001 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.001 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.003 ± 0.001 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.004 ± 0.003 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.674 ± 0.021 | - | 0.813 ± 0.041 | 1.396 ± 0.033 | 1.062 ± 0.078 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.003 ± 0.003 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.026 ± 0.000 | - | 0.032 ± 0.000 | 0.059 ± 0.000 | 0.063 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.003 ± 0.001 | - | 0.006 ± 0.002 | 0.010 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.005 ± 0.002 | - | 0.007 ± 0.000 | 0.012 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.627 ± 0.014 | - | 0.640 ± 0.026 | 0.728 ± 0.023 | 0.946 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 2.131 ± 0.017 | - | 2.165 ± 0.027 | 2.158 ± 0.008 | 2.768 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 6.487 ± 0.031 | - | 6.548 ± 0.011 | 6.527 ± 0.030 | 8.941 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 3.717 ± 0.013 | - | 3.715 ± 0.019 | 3.210 ± 0.029 | 2.663 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.110 ± 0.003 | - | 0.115 ± 0.005 | 0.120 ± 0.006 | 0.210 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.001 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.008 ± 0.000 | - | 0.013 ± 0.000 | 0.016 ± 0.002 | 0.032 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.001 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.013 ± 0.003 | 0.015 ± 0.001 | 0.030 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.021 ± 0.001 | - | 0.026 ± 0.000 | 0.029 ± 0.002 | 0.108 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.001 | 0.025 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | 0.015 ± 0.000 | 0.018 ± 0.006 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.027 ± 0.001 | - | 0.032 ± 0.000 | 0.036 ± 0.005 | 0.121 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 29.284 ± 0.127 | - | 29.596 ± 0.140 | 25.591 ± 0.100 | 20.392 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.007 ± 0.000 | - | 0.011 ± 0.002 | 0.014 ± 0.001 | 0.030 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.868 ± 0.008 | - | 0.889 ± 0.010 | 0.878 ± 0.004 | 1.021 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.018 ± 0.000 | - | 0.023 ± 0.003 | 0.026 ± 0.001 | 0.044 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.052 ± 0.000 | - | 0.058 ± 0.000 | 0.061 ± 0.003 | 0.155 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 1.015 ± 0.004 | - | 1.029 ± 0.012 | 0.700 ± 0.019 | 0.950 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 3.421 ± 0.022 | - | 3.449 ± 0.017 | 2.140 ± 0.009 | 2.401 ± 0.058 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 9.996 ± 0.040 | - | 6.649 ± 0.021 | 9.341 ± 0.018 | 6.001 ± 0.137 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 3.599 ± 0.033 | - | 3.623 ± 0.016 | 2.250 ± 0.019 | 3.453 ± 0.363 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.012 ± 0.001 | 0.013 ± 0.001 | 0.025 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.110 ± 0.002 | - | 0.116 ± 0.001 | 0.178 ± 0.005 | 0.196 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | 0.015 ± 0.001 | 0.011 ± 0.001 | 0.026 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.012 ± 0.000 | - | 0.021 ± 0.001 | 0.016 ± 0.001 | 0.033 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.008 ± 0.001 | - | 0.015 ± 0.001 | 0.016 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.013 ± 0.000 | - | 0.022 ± 0.004 | 0.016 ± 0.001 | 0.031 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.032 ± 0.000 | - | 0.042 ± 0.001 | 0.030 ± 0.000 | 0.104 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.008 ± 0.001 | - | 0.015 ± 0.001 | 0.017 ± 0.000 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.016 ± 0.000 | - | 0.025 ± 0.001 | 0.018 ± 0.001 | 0.034 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.041 ± 0.000 | - | 0.051 ± 0.001 | 0.036 ± 0.003 | 0.107 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 28.764 ± 0.212 | - | 29.132 ± 0.795 | 17.789 ± 0.126 | 31.940 ± 5.421 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.011 ± 0.000 | - | 0.019 ± 0.001 | 0.022 ± 0.004 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.881 ± 0.009 | - | 0.895 ± 0.006 | 1.248 ± 0.009 | 1.626 ± 1.379 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.029 ± 0.000 | - | 0.039 ± 0.006 | 0.026 ± 0.001 | 0.044 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.080 ± 0.000 | - | 0.090 ± 0.005 | 0.061 ± 0.005 | 0.140 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.694 ± 0.015 | - | 0.702 ± 0.019 | 0.522 ± 0.014 | 0.449 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 2.233 ± 0.007 | - | 2.247 ± 0.011 | 1.953 ± 0.010 | 1.954 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 5.810 ± 0.010 | - | 5.837 ± 0.065 | 5.697 ± 0.010 | 6.455 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.328 ± 0.019 | - | 1.375 ± 0.036 | 1.349 ± 0.023 | 1.473 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.003 ± 0.001 | - | 0.007 ± 0.000 | 0.008 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.070 ± 0.017 | - | 0.080 ± 0.012 | 0.089 ± 0.003 | 0.134 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.001 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | 0.013 ± 0.004 | 0.015 ± 0.001 | 0.030 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.013 ± 0.000 | 0.015 ± 0.001 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.018 ± 0.000 | - | 0.024 ± 0.000 | 0.027 ± 0.001 | 0.041 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | 0.016 ± 0.002 | 0.017 ± 0.000 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.024 ± 0.000 | - | 0.029 ± 0.000 | 0.032 ± 0.003 | 0.047 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 10.408 ± 0.160 | - | 10.512 ± 0.035 | 10.605 ± 0.095 | 11.236 ± 0.080 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.012 ± 0.001 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.553 ± 0.020 | - | 0.572 ± 0.020 | 0.632 ± 0.022 | 0.753 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.019 ± 0.001 | - | 0.024 ± 0.000 | 0.024 ± 0.000 | 0.036 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.046 ± 0.000 | - | 0.052 ± 0.000 | 0.053 ± 0.003 | 0.095 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 1.117 ± 0.015 | - | 1.134 ± 0.016 | 0.786 ± 0.276 | 0.473 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 3.541 ± 0.011 | - | 3.569 ± 0.021 | 1.958 ± 0.022 | 1.898 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 9.005 ± 0.033 | - | 5.928 ± 0.072 | 8.543 ± 0.019 | 3.696 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.268 ± 0.024 | - | 1.294 ± 0.028 | 1.173 ± 0.048 | 1.770 ± 0.709 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.004 ± 0.001 | - | 0.011 ± 0.001 | 0.012 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.114 ± 0.003 | - | 0.130 ± 0.007 | 0.138 ± 0.004 | 0.160 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | 0.015 ± 0.000 | 0.010 ± 0.000 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.011 ± 0.000 | - | 0.020 ± 0.001 | 0.015 ± 0.001 | 0.029 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.007 ± 0.000 | - | 0.014 ± 0.001 | 0.015 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.015 ± 0.007 | - | 0.022 ± 0.000 | 0.015 ± 0.001 | 0.028 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.029 ± 0.000 | - | 0.038 ± 0.001 | 0.026 ± 0.001 | 0.042 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.009 ± 0.003 | - | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.017 ± 0.004 | - | 0.026 ± 0.001 | 0.017 ± 0.001 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.038 ± 0.000 | - | 0.047 ± 0.001 | 0.032 ± 0.001 | 0.049 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 10.112 ± 0.346 | - | 10.324 ± 0.096 | 9.201 ± 0.242 | 13.907 ± 4.387 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.012 ± 0.002 | - | 0.018 ± 0.000 | 0.018 ± 0.001 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.890 ± 0.015 | - | 0.949 ± 0.056 | 0.946 ± 0.005 | 0.447 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.030 ± 0.000 | - | 0.040 ± 0.001 | 0.024 ± 0.001 | 0.036 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.071 ± 0.001 | - | 0.082 ± 0.004 | 0.053 ± 0.001 | 0.097 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.276 ± 0.010 | - | 0.284 ± 0.007 | 0.236 ± 0.017 | 0.478 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 1.098 ± 0.018 | - | 1.116 ± 0.025 | 0.947 ± 0.002 | 1.953 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 3.108 ± 0.017 | - | 3.123 ± 0.031 | 2.885 ± 0.006 | 6.481 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.533 ± 0.014 | - | 0.534 ± 0.020 | 0.555 ± 0.013 | 1.492 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.002 ± 0.002 | - | 0.006 ± 0.003 | 0.010 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.031 ± 0.000 | - | 0.036 ± 0.014 | 0.041 ± 0.005 | 0.187 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.002 | 0.010 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.013 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.010 ± 0.000 | - | 0.015 ± 0.000 | 0.020 ± 0.001 | 0.037 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.004 | 0.011 ± 0.001 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.014 ± 0.001 | 0.022 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.013 ± 0.000 | - | 0.018 ± 0.000 | 0.022 ± 0.001 | 0.043 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 4.231 ± 0.043 | - | 4.132 ± 0.040 | 4.282 ± 0.027 | 11.043 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.001 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.231 ± 0.012 | - | 0.241 ± 0.016 | 0.245 ± 0.008 | 0.785 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.010 ± 0.000 | - | 0.014 ± 0.000 | 0.017 ± 0.001 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.025 ± 0.000 | - | 0.029 ± 0.000 | 0.033 ± 0.001 | 0.132 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.475 ± 0.024 | - | 0.486 ± 0.029 | 0.397 ± 0.016 | 0.435 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 1.616 ± 0.013 | - | 1.630 ± 0.017 | 0.948 ± 0.004 | 1.282 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 4.192 ± 0.020 | - | 4.300 ± 0.013 | 3.664 ± 0.006 | 3.672 ± 0.486 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.695 ± 0.019 | - | 0.724 ± 0.029 | 0.691 ± 0.020 | 1.788 ± 0.085 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.003 ± 0.002 | - | 0.009 ± 0.001 | 0.014 ± 0.001 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.039 ± 0.000 | - | 0.047 ± 0.001 | 0.054 ± 0.003 | 0.142 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.016 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.007 ± 0.001 | - | 0.014 ± 0.001 | 0.013 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.004 ± 0.001 | - | 0.011 ± 0.002 | 0.016 ± 0.003 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.007 ± 0.003 | - | 0.014 ± 0.001 | 0.018 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.014 ± 0.003 | - | 0.022 ± 0.001 | 0.019 ± 0.001 | 0.045 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.005 ± 0.001 | - | 0.011 ± 0.001 | 0.016 ± 0.001 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.009 ± 0.000 | - | 0.016 ± 0.001 | 0.020 ± 0.006 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.018 ± 0.000 | - | 0.025 ± 0.000 | 0.022 ± 0.001 | 0.041 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 5.439 ± 0.026 | - | 5.660 ± 0.097 | 5.208 ± 0.022 | 13.871 ± 0.654 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.008 ± 0.004 | - | 0.013 ± 0.001 | 0.017 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.293 ± 0.018 | - | 0.306 ± 0.020 | 0.320 ± 0.018 | 0.422 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.014 ± 0.002 | - | 0.023 ± 0.000 | 0.024 ± 0.001 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.034 ± 0.000 | - | 0.043 ± 0.012 | 0.033 ± 0.000 | 0.094 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.151 ± 0.002 | - | 0.164 ± 0.005 | 0.189 ± 0.019 | 0.131 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.217 ± 0.001 | - | 0.233 ± 0.011 | 0.256 ± 0.014 | 0.219 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.498 ± 0.010 | - | 0.535 ± 0.012 | 0.543 ± 0.022 | 0.585 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.134 ± 0.003 | - | 0.147 ± 0.002 | 0.174 ± 0.002 | 0.223 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.001 | 0.013 ± 0.002 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.012 ± 0.001 | - | 0.018 ± 0.000 | 0.026 ± 0.001 | 0.057 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.001 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.014 ± 0.000 | - | 0.022 ± 0.015 | 0.029 ± 0.001 | 0.065 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.005 ± 0.003 | - | 0.007 ± 0.001 | 0.016 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.004 ± 0.000 | - | 0.009 ± 0.002 | 0.018 ± 0.001 | 0.051 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.001 | 0.016 ± 0.000 | 0.024 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.007 | 0.023 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.018 ± 0.000 | 0.050 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 1.067 ± 0.035 | - | 1.100 ± 0.037 | 1.296 ± 0.012 | 1.258 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.002 | 0.017 ± 0.002 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.076 ± 0.000 | - | 0.085 ± 0.000 | 0.088 ± 0.003 | 0.132 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.017 ± 0.000 | 0.054 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.006 ± 0.000 | - | 0.011 ± 0.000 | 0.020 ± 0.001 | 0.058 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.199 ± 0.019 | - | 0.213 ± 0.004 | 0.169 ± 0.005 | 0.150 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.298 ± 0.022 | - | 0.317 ± 0.024 | 0.232 ± 0.021 | 0.234 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.776 ± 0.011 | - | 0.860 ± 0.010 | 0.395 ± 0.039 | 0.663 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.194 ± 0.009 | - | 0.209 ± 0.038 | 0.227 ± 0.022 | 0.224 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.012 ± 0.004 | 0.023 ± 0.003 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.019 ± 0.001 | - | 0.027 ± 0.001 | 0.038 ± 0.001 | 0.041 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.011 ± 0.001 | 0.024 ± 0.005 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.022 ± 0.001 | - | 0.033 ± 0.006 | 0.043 ± 0.001 | 0.063 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.004 ± 0.002 | - | 0.012 ± 0.000 | 0.023 ± 0.001 | 0.026 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.005 ± 0.000 | - | 0.011 ± 0.001 | 0.027 ± 0.002 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.005 ± 0.001 | - | 0.012 ± 0.001 | 0.027 ± 0.004 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.007 ± 0.003 | - | 0.014 ± 0.000 | 0.029 ± 0.006 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.005 ± 0.001 | - | 0.012 ± 0.001 | 0.026 ± 0.000 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | 0.012 ± 0.001 | 0.026 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.007 ± 0.000 | - | 0.014 ± 0.000 | 0.029 ± 0.001 | 0.052 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 1.541 ± 0.058 | - | 1.630 ± 0.038 | 0.681 ± 0.018 | 0.999 ± 0.127 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.006 ± 0.003 | - | 0.013 ± 0.001 | 0.027 ± 0.001 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.122 ± 0.004 | - | 0.135 ± 0.004 | 0.078 ± 0.004 | 0.182 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.006 ± 0.000 | - | 0.014 ± 0.001 | 0.028 ± 0.000 | 0.055 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.010 ± 0.003 | - | 0.018 ± 0.000 | 0.032 ± 0.003 | 0.031 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.260 ± 0.010 | - | 0.266 ± 0.011 | 0.224 ± 0.015 | 0.328 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.607 ± 0.014 | - | 0.614 ± 0.013 | 0.527 ± 0.013 | 0.874 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.380 ± 0.013 | - | 1.375 ± 0.006 | 1.357 ± 0.024 | 2.335 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.494 ± 0.009 | - | 0.515 ± 0.025 | 0.585 ± 0.016 | 0.635 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.018 ± 0.000 | - | 0.023 ± 0.000 | 0.027 ± 0.001 | 0.073 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.006 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.019 ± 0.000 | - | 0.024 ± 0.003 | 0.030 ± 0.001 | 0.065 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.001 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.006 ± 0.000 | - | 0.010 ± 0.002 | 0.010 ± 0.002 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.011 ± 0.001 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 3.925 ± 0.020 | - | 4.720 ± 0.018 | 4.471 ± 0.054 | 4.698 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.129 ± 0.001 | - | 0.137 ± 0.002 | 0.175 ± 0.003 | 0.220 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.006 ± 0.000 | - | 0.011 ± 0.000 | 0.010 ± 0.007 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.012 ± 0.000 | - | 0.017 ± 0.000 | 0.016 ± 0.000 | 0.038 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.264 ± 0.027 | - | 0.272 ± 0.031 | 0.322 ± 0.016 | 0.325 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.615 ± 0.024 | - | 0.626 ± 0.025 | 0.782 ± 0.013 | 0.879 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.354 ± 0.017 | - | 1.435 ± 0.010 | 2.031 ± 0.018 | 1.353 ± 0.157 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.544 ± 0.032 | - | 0.609 ± 0.051 | 0.518 ± 0.021 | 0.891 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.018 ± 0.000 | - | 0.023 ± 0.000 | 0.044 ± 0.001 | 0.076 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.004 ± 0.002 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.031 ± 0.000 | - | 0.041 ± 0.002 | 0.048 ± 0.003 | 0.085 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.006 ± 0.003 | - | 0.008 ± 0.000 | 0.012 ± 0.001 | 0.023 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.005 ± 0.001 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.007 ± 0.003 | - | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.010 ± 0.001 | - | 0.010 ± 0.000 | 0.016 ± 0.001 | 0.028 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.005 ± 0.004 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.004 ± 0.006 | - | 0.009 ± 0.000 | 0.013 ± 0.004 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.013 ± 0.007 | - | 0.012 ± 0.000 | 0.018 ± 0.000 | 0.031 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 4.148 ± 0.021 | - | 5.569 ± 0.074 | 3.796 ± 0.164 | 7.313 ± 1.016 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.007 ± 0.002 | - | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.128 ± 0.002 | - | 0.142 ± 0.009 | 0.290 ± 0.013 | 0.342 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.011 ± 0.003 | - | 0.011 ± 0.000 | 0.016 ± 0.000 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.012 ± 0.007 | - | 0.017 ± 0.000 | 0.026 ± 0.005 | 0.042 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.161 ± 0.013 | - | 0.153 ± 0.007 | 0.099 ± 0.004 | 0.258 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.215 ± 0.015 | - | 0.211 ± 0.040 | 0.140 ± 0.003 | 0.364 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.573 ± 0.027 | - | 0.574 ± 0.023 | 0.405 ± 0.015 | 0.794 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.118 ± 0.002 | - | 0.168 ± 0.040 | 0.148 ± 0.003 | 0.215 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.001 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.011 ± 0.000 | - | 0.019 ± 0.003 | 0.030 ± 0.001 | 0.054 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.000 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.020 ± 0.001 | 0.028 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.020 ± 0.000 | 0.027 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.931 ± 0.004 | - | 0.998 ± 0.013 | 1.024 ± 0.011 | 1.037 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.018 ± 0.001 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.070 ± 0.002 | - | 0.105 ± 0.055 | 0.081 ± 0.003 | 0.173 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.018 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.012 ± 0.001 | 0.022 ± 0.001 | 0.029 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.156 ± 0.002 | - | 0.147 ± 0.005 | 0.131 ± 0.002 | 0.272 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.209 ± 0.030 | - | 0.229 ± 0.042 | 0.173 ± 0.021 | 0.403 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.608 ± 0.027 | - | 0.655 ± 0.022 | 0.382 ± 0.014 | 0.908 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.119 ± 0.003 | - | 0.138 ± 0.041 | 0.204 ± 0.020 | 0.195 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.002 | 0.027 ± 0.003 | 0.016 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.011 ± 0.000 | - | 0.019 ± 0.003 | 0.044 ± 0.003 | 0.045 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.028 ± 0.009 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.004 ± 0.003 | - | 0.008 ± 0.000 | 0.027 ± 0.004 | 0.017 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.002 ± 0.002 | - | 0.008 ± 0.003 | 0.027 ± 0.004 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.027 ± 0.001 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.003 ± 0.002 | - | 0.009 ± 0.000 | 0.031 ± 0.003 | 0.027 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.004 ± 0.001 | - | 0.010 ± 0.007 | 0.026 ± 0.002 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.005 ± 0.001 | - | 0.008 ± 0.000 | 0.027 ± 0.004 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.030 ± 0.003 | 0.023 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.943 ± 0.018 | - | 1.069 ± 0.025 | 0.577 ± 0.011 | 0.791 ± 0.061 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.005 ± 0.002 | - | 0.008 ± 0.000 | 0.032 ± 0.003 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.075 ± 0.003 | - | 0.085 ± 0.020 | 0.074 ± 0.000 | 0.177 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.003 ± 0.002 | - | 0.009 ± 0.000 | 0.030 ± 0.004 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.009 ± 0.004 | - | 0.012 ± 0.001 | 0.033 ± 0.005 | 0.031 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.936 ± 0.680 | - | 1.838 ± 0.791 | 1.578 ± 0.007 | 2.071 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 6.008 ± 0.042 | - | 6.029 ± 0.373 | 5.166 ± 0.048 | 5.155 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 17.733 ± 0.146 | - | 17.781 ± 0.065 | 15.996 ± 0.076 | 14.779 ± 0.150 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 4.960 ± 0.031 | - | 4.978 ± 0.023 | 5.616 ± 0.104 | 4.221 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.005 ± 0.000 | - | 0.010 ± 0.000 | 0.014 ± 0.001 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.201 ± 0.011 | - | 0.212 ± 0.026 | 0.232 ± 0.012 | 0.239 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.009 ± 0.000 | - | 0.015 ± 0.002 | 0.020 ± 0.001 | 0.035 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.178 ± 0.006 | - | 0.195 ± 0.031 | 0.207 ± 0.011 | 0.229 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.021 ± 0.000 | - | 0.027 ± 0.000 | 0.033 ± 0.001 | 0.044 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.008 ± 0.000 | - | 0.014 ± 0.004 | 0.017 ± 0.001 | 0.032 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.020 ± 0.000 | - | 0.026 ± 0.000 | 0.030 ± 0.001 | 0.043 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.055 ± 0.005 | - | 0.061 ± 0.000 | 0.064 ± 0.002 | 0.076 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.010 ± 0.000 | - | 0.015 ± 0.000 | 0.019 ± 0.002 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.026 ± 0.001 | - | 0.032 ± 0.001 | 0.035 ± 0.001 | 0.049 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.073 ± 0.000 | - | 0.079 ± 0.000 | 0.079 ± 0.001 | 0.087 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 39.336 ± 0.092 | - | 40.530 ± 1.313 | 44.188 ± 0.068 | 32.974 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.017 ± 0.000 | - | 0.023 ± 0.003 | 0.026 ± 0.000 | 0.041 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.563 ± 0.016 | - | 1.594 ± 0.016 | 1.651 ± 0.050 | 1.552 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.049 ± 0.001 | - | 0.056 ± 0.000 | 0.056 ± 0.005 | 0.072 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.141 ± 0.001 | - | 0.148 ± 0.005 | 0.140 ± 0.004 | 0.166 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 3.052 ± 0.008 | - | 3.081 ± 0.010 | 1.541 ± 0.018 | 2.063 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 9.779 ± 0.005 | - | 6.001 ± 0.079 | 5.102 ± 0.012 | 2.782 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 17.813 ± 0.236 | - | 17.868 ± 0.116 | 18.572 ± 0.019 | 7.599 ± 1.271 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 4.705 ± 0.024 | - | 4.833 ± 0.057 | 4.663 ± 0.048 | 5.045 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.007 ± 0.000 | - | 0.017 ± 0.004 | 0.020 ± 0.003 | 0.030 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.311 ± 0.021 | - | 0.326 ± 0.028 | 0.336 ± 0.021 | 0.277 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.014 ± 0.000 | - | 0.025 ± 0.001 | 0.021 ± 0.004 | 0.034 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.272 ± 0.025 | - | 0.297 ± 0.028 | 0.290 ± 0.021 | 0.229 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.021 ± 0.000 | - | 0.027 ± 0.003 | 0.032 ± 0.008 | 0.044 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.013 ± 0.002 | - | 0.023 ± 0.001 | 0.026 ± 0.003 | 0.033 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.032 ± 0.000 | - | 0.045 ± 0.001 | 0.030 ± 0.001 | 0.044 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.054 ± 0.000 | - | 0.062 ± 0.001 | 0.063 ± 0.001 | 0.076 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.016 ± 0.004 | - | 0.027 ± 0.001 | 0.027 ± 0.002 | 0.035 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.042 ± 0.000 | - | 0.053 ± 0.011 | 0.035 ± 0.005 | 0.049 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.072 ± 0.000 | - | 0.080 ± 0.009 | 0.080 ± 0.008 | 0.089 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 37.058 ± 0.204 | - | 39.721 ± 0.225 | 36.749 ± 0.283 | 64.307 ± 42.712 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.028 ± 0.001 | - | 0.040 ± 0.001 | 0.037 ± 0.004 | 0.042 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 2.396 ± 0.043 | - | 2.427 ± 0.022 | 2.360 ± 0.013 | 0.785 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.080 ± 0.001 | - | 0.091 ± 0.005 | 0.056 ± 0.001 | 0.068 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.140 ± 0.002 | - | 0.150 ± 0.011 | 0.140 ± 0.006 | 0.180 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 1.345 ± 0.028 | - | 1.372 ± 0.019 | 1.107 ± 0.013 | 1.362 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 5.156 ± 0.057 | - | 5.150 ± 0.038 | 4.362 ± 0.011 | 4.082 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 15.158 ± 0.049 | - | 15.251 ± 0.064 | 13.953 ± 4.700 | 12.244 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 2.337 ± 0.033 | - | 2.340 ± 0.030 | 2.480 ± 0.016 | 2.390 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.013 ± 0.001 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.135 ± 0.021 | - | 0.146 ± 0.003 | 0.167 ± 0.008 | 0.226 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.008 ± 0.000 | - | 0.013 ± 0.001 | 0.018 ± 0.001 | 0.031 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.108 ± 0.006 | - | 0.117 ± 0.003 | 0.134 ± 0.010 | 0.171 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.017 ± 0.000 | - | 0.023 ± 0.006 | 0.028 ± 0.001 | 0.039 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.007 ± 0.001 | - | 0.012 ± 0.000 | 0.016 ± 0.001 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.017 ± 0.000 | - | 0.023 ± 0.000 | 0.027 ± 0.001 | 0.038 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.043 ± 0.000 | - | 0.050 ± 0.008 | 0.052 ± 0.004 | 0.062 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.008 ± 0.000 | - | 0.013 ± 0.000 | 0.017 ± 0.001 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.023 ± 0.001 | - | 0.029 ± 0.000 | 0.038 ± 0.012 | 0.042 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.057 ± 0.000 | - | 0.064 ± 0.000 | 0.067 ± 0.005 | 0.075 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 18.427 ± 0.063 | - | 18.616 ± 0.309 | 19.366 ± 0.108 | 18.436 ± 0.117 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.013 ± 0.000 | - | 0.018 ± 0.000 | 0.021 ± 0.000 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 1.060 ± 0.026 | - | 1.084 ± 0.034 | 1.163 ± 0.004 | 1.140 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.042 ± 0.001 | - | 0.048 ± 0.001 | 0.048 ± 0.003 | 0.057 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.115 ± 0.000 | - | 0.121 ± 0.014 | 0.118 ± 0.004 | 0.174 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 2.241 ± 0.020 | - | 2.264 ± 0.013 | 1.067 ± 0.046 | 1.326 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 8.332 ± 0.019 | - | 8.380 ± 3.277 | 4.320 ± 0.009 | 2.129 ± 0.374 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 15.126 ± 0.064 | - | 15.289 ± 0.370 | 16.715 ± 0.050 | 6.290 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 2.631 ± 0.035 | - | 2.639 ± 0.040 | 2.790 ± 0.014 | 3.008 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.006 ± 0.000 | - | 0.014 ± 0.001 | 0.018 ± 0.002 | 0.032 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.213 ± 0.021 | - | 0.228 ± 0.028 | 0.254 ± 0.013 | 0.218 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.012 ± 0.000 | - | 0.020 ± 0.001 | 0.019 ± 0.008 | 0.030 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.166 ± 0.005 | - | 0.193 ± 0.038 | 0.197 ± 0.005 | 0.198 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.017 ± 0.001 | - | 0.023 ± 0.000 | 0.028 ± 0.001 | 0.041 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.010 ± 0.000 | - | 0.019 ± 0.000 | 0.022 ± 0.001 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.028 ± 0.000 | - | 0.038 ± 0.001 | 0.027 ± 0.001 | 0.038 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.043 ± 0.001 | - | 0.050 ± 0.007 | 0.053 ± 0.010 | 0.063 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.013 ± 0.001 | - | 0.022 ± 0.001 | 0.024 ± 0.002 | 0.032 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.037 ± 0.000 | - | 0.048 ± 0.003 | 0.031 ± 0.000 | 0.042 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.057 ± 0.001 | - | 0.064 ± 0.008 | 0.066 ± 0.004 | 0.076 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 21.009 ± 0.566 | - | 21.296 ± 0.448 | 21.580 ± 0.228 | 31.693 ± 8.391 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.021 ± 0.001 | - | 0.031 ± 0.001 | 0.031 ± 0.002 | 0.034 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 1.640 ± 0.031 | - | 1.676 ± 0.010 | 1.757 ± 0.005 | 0.597 ± 0.180 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.069 ± 0.001 | - | 0.081 ± 0.004 | 0.048 ± 0.001 | 0.058 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.114 ± 0.001 | - | 0.120 ± 0.001 | 0.118 ± 0.009 | 0.146 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.498 ± 0.022 | - | unsupported | 0.353 ± 0.014 | 0.795 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 2.286 ± 0.033 | - | unsupported | 1.968 ± 0.073 | 2.704 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 5.542 ± 0.016 | - | unsupported | 5.319 ± 0.055 | 6.824 ± 0.069 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.989 ± 0.020 | - | unsupported | 1.143 ± 0.012 | 1.120 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.002 ± 0.002 | - | unsupported | 0.009 ± 0.000 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.047 ± 0.000 | - | unsupported | 0.062 ± 0.002 | 0.077 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | unsupported | 0.012 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.039 ± 0.000 | - | unsupported | 0.054 ± 0.002 | 0.069 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | unsupported | 0.016 ± 0.008 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.012 ± 0.004 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.017 ± 0.000 | - | unsupported | 0.026 ± 0.001 | 0.047 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.011 ± 0.001 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | unsupported | 0.018 ± 0.001 | 0.029 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.023 ± 0.000 | - | unsupported | 0.032 ± 0.001 | 0.044 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 7.747 ± 0.097 | - | unsupported | 8.971 ± 0.098 | 8.601 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | unsupported | 0.013 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.374 ± 0.011 | - | unsupported | 0.432 ± 0.015 | 0.474 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.019 ± 0.000 | - | unsupported | 0.025 ± 0.000 | 0.037 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.044 ± 0.000 | - | unsupported | 0.052 ± 0.001 | 0.068 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.813 ± 0.008 | - | unsupported | 0.586 ± 0.017 | 0.776 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 3.588 ± 0.014 | - | unsupported | 1.934 ± 0.003 | 1.434 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 8.374 ± 0.013 | - | unsupported | 7.680 ± 0.012 | 3.565 ± 0.083 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.224 ± 0.017 | - | unsupported | 1.470 ± 0.023 | 1.427 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.004 ± 0.004 | - | unsupported | 0.014 ± 0.001 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.064 ± 0.002 | - | unsupported | 0.089 ± 0.028 | 0.073 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | unsupported | 0.018 ± 0.006 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.056 ± 0.002 | - | unsupported | 0.079 ± 0.007 | 0.089 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.010 ± 0.000 | - | unsupported | 0.016 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.005 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.018 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.013 ± 0.000 | - | unsupported | 0.023 ± 0.004 | 0.027 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.026 ± 0.001 | - | unsupported | 0.027 ± 0.001 | 0.037 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.006 ± 0.000 | - | unsupported | 0.016 ± 0.004 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.016 ± 0.000 | - | unsupported | 0.018 ± 0.001 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.035 ± 0.001 | - | unsupported | 0.032 ± 0.001 | 0.044 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 9.748 ± 0.058 | - | unsupported | 11.234 ± 0.029 | 11.435 ± 0.898 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.009 ± 0.000 | - | unsupported | 0.018 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.508 ± 0.013 | - | unsupported | 0.422 ± 0.017 | 0.317 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.030 ± 0.000 | - | unsupported | 0.025 ± 0.000 | 0.036 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.066 ± 0.000 | - | unsupported | 0.052 ± 0.001 | 0.072 ± 0.008 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 1.052 ± 0.009 | - | unsupported | 0.960 ± 0.009 | 2.648 ± 0.158 | 2.583 ± 0.032 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.133 ± 0.001 | - | unsupported | 0.060 ± 0.006 | 0.145 ± 0.008 | 0.096 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.137 ± 0.003 | - | unsupported | 0.084 ± 0.004 | 0.194 ± 0.018 | 0.179 ± 0.013 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.341 ± 0.015 | - | unsupported | 0.242 ± 0.004 | 0.538 ± 0.044 | 0.672 ± 0.014 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 1.045 ± 0.013 | - | unsupported | 0.480 ± 0.016 | 1.106 ± 0.032 | 2.521 ± 0.735 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.131 ± 0.003 | - | unsupported | 0.060 ± 0.002 | 0.117 ± 0.015 | 0.100 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.141 ± 0.002 | - | unsupported | 0.079 ± 0.001 | 0.184 ± 0.004 | 0.180 ± 0.005 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.337 ± 0.010 | - | unsupported | 0.156 ± 0.020 | 0.558 ± 0.027 | 0.666 ± 0.031 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.306 ± 0.010 | - | unsupported | 0.382 ± 0.013 | 0.811 ± 0.038 | 0.478 ± 0.016 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.034 ± 0.000 | - | unsupported | 0.045 ± 0.005 | 0.196 ± 0.006 | 0.078 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.059 ± 0.000 | - | unsupported | 0.061 ± 0.005 | 0.233 ± 0.010 | 0.126 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.200 ± 0.005 | - | unsupported | 0.167 ± 0.004 | 0.385 ± 0.011 | 0.235 ± 0.007 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.304 ± 0.011 | - | unsupported | 0.563 ± 0.011 | 0.875 ± 0.083 | 0.421 ± 0.013 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.033 ± 0.000 | - | unsupported | 0.075 ± 0.002 | 0.201 ± 0.017 | 0.074 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.059 ± 0.000 | - | unsupported | 0.100 ± 0.005 | 0.250 ± 0.017 | 0.116 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.199 ± 0.000 | - | unsupported | 0.228 ± 0.016 | 0.403 ± 0.016 | 0.210 ± 0.012 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.195 ± 0.017 | - | 0.289 ± 0.021 | 0.196 ± 0.014 | 0.391 ± 0.029 | 0.344 ± 0.023 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.080 ± 0.000 | - | 0.088 ± 0.013 | 0.074 ± 0.001 | 0.081 ± 0.007 | 0.063 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.092 ± 0.001 | - | 0.103 ± 0.003 | 0.088 ± 0.006 | 0.103 ± 0.002 | 0.097 ± 0.006 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.122 ± 0.000 | - | 0.152 ± 0.012 | 0.126 ± 0.007 | 0.180 ± 0.029 | 0.160 ± 0.009 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.194 ± 0.008 | - | 0.342 ± 0.018 | 0.258 ± 0.015 | 0.513 ± 0.144 | 0.269 ± 0.006 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.080 ± 0.000 | - | 0.088 ± 0.005 | 0.125 ± 0.004 | 0.081 ± 0.006 | 0.059 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.092 ± 0.001 | - | 0.108 ± 0.010 | 0.143 ± 0.009 | 0.105 ± 0.003 | 0.085 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.121 ± 0.001 | - | 0.216 ± 0.018 | 0.178 ± 0.003 | 0.230 ± 0.019 | 0.140 ± 0.001 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 44.051 ± 0.173 | - | 46.357 ± 0.151 | 62.350 ± 0.147 | 64.935 ± 0.479 | 42.095 ± 3.942 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 20.757 ± 0.203 | - | 33.494 ± 0.226 | 25.587 ± 0.257 | 57.418 ± 4.958 | 35.259 ± 13.271 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 26.797 ± 0.302 | - | 25.786 ± 0.549 | 26.396 ± 0.201 | 21.802 ± 0.155 | 20.220 ± 2.545 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 13.062 ± 0.251 | - | 12.430 ± 0.158 | 11.565 ± 0.111 | 14.728 ± 0.237 | 12.310 ± 0.202 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 7.466 ± 0.022 | - | 7.505 ± 0.034 | 7.967 ± 0.053 | 8.052 ± 0.053 | 7.967 ± 0.249 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 11.231 ± 0.034 | - | 11.342 ± 0.022 | 10.500 ± 0.058 | 11.634 ± 0.078 | 11.707 ± 0.364 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 8.721 ± 0.059 | - | 8.788 ± 0.033 | 8.906 ± 0.077 | 9.622 ± 0.057 | 9.127 ± 0.026 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 13.227 ± 0.041 | - | 13.422 ± 0.120 | 11.897 ± 0.067 | 14.339 ± 0.220 | 13.909 ± 0.079 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 21.239 ± 0.212 | - | 22.138 ± 0.207 | 19.003 ± 0.175 | 39.844 ± 0.129 | 14.369 ± 0.758 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 15.769 ± 0.853 | - | 17.374 ± 0.127 | 15.148 ± 0.194 | 34.130 ± 0.754 | 12.724 ± 0.507 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.492 ± 0.009 | 0.510 ± 0.004 | unsupported | 0.611 ± 0.005 | 0.737 ± 0.047 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 7.831 ± 0.113 | 7.909 ± 0.024 | unsupported | 8.642 ± 0.049 | 9.473 ± 0.041 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.386 ± 0.002 | 0.412 ± 0.004 | unsupported | 0.507 ± 0.006 | 1.213 ± 0.381 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 6.145 ± 0.076 | 6.358 ± 0.141 | unsupported | 5.647 ± 0.100 | 9.467 ± 0.133 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 41.561 ± 0.203 | - | 42.028 ± 0.968 | 43.033 ± 0.706 | 31.975 ± 0.197 | 27.675 ± 0.154 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 23.564 ± 2.259 | - | 21.047 ± 0.119 | 22.121 ± 0.368 | 19.854 ± 0.730 | 16.680 ± 0.423 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 17.968 ± 0.326 | 18.096 ± 0.117 | 9.134 ± 0.058 | 45.869 ± 0.277 | 9.799 ± 0.537 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 11.397 ± 0.190 | 11.113 ± 0.176 | 6.303 ± 0.139 | 42.865 ± 1.739 | 10.739 ± 0.138 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 26.633 ± 0.204 | - | 27.107 ± 0.158 | 36.542 ± 0.397 | 39.769 ± 0.427 | 28.338 ± 11.744 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 12.977 ± 0.131 | - | 13.578 ± 0.239 | 15.918 ± 0.429 | 31.089 ± 0.896 | 27.684 ± 8.786 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 1.117 ± 0.126 | - | 1.292 ± 0.183 | 1.234 ± 0.127 | 4.008 ± 0.318 | 2.708 ± 0.025 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.793 ± 0.079 | - | 0.711 ± 0.047 | 1.230 ± 0.083 | 4.063 ± 0.535 | 2.707 ± 0.021 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 19.902 ± 0.385 | - | 20.152 ± 0.119 | 20.305 ± 0.754 | 21.229 ± 0.077 | 20.423 ± 2.682 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 15.868 ± 0.219 | - | 16.116 ± 0.162 | 16.307 ± 0.474 | 21.283 ± 1.809 | 19.475 ± 0.413 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 20.233 ± 0.208 | - | 20.486 ± 0.075 | 20.247 ± 0.293 | 21.058 ± 0.063 | 19.570 ± 1.313 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 15.873 ± 0.210 | - | 16.365 ± 0.111 | 16.371 ± 0.265 | 20.835 ± 0.829 | 24.203 ± 1.789 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.159 ± 0.007 | 1.196 ± 0.011 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 21.396 ± 0.071 | 21.378 ± 0.029 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.364 ± 0.024 | 1.378 ± 0.029 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 19.802 ± 0.098 | 19.912 ± 0.052 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.087 ± 0.007 | 1.115 ± 0.006 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 20.361 ± 0.030 | 20.387 ± 0.153 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.253 ± 0.007 | 1.270 ± 0.018 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 18.572 ± 0.113 | 18.609 ± 0.182 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 26.654 ± 0.330 | - | 25.907 ± 0.821 | 26.436 ± 0.226 | 21.814 ± 0.269 | 20.225 ± 0.327 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 13.346 ± 0.379 | - | 12.095 ± 0.228 | 11.612 ± 0.287 | 14.452 ± 1.123 | 12.209 ± 0.390 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 59.780 ± 0.426 | 59.493 ± 0.726 | 60.842 ± 0.451 | 65.136 ± 0.476 | 64.167 ± 5.722 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 38.877 ± 1.448 | 38.158 ± 1.480 | 38.415 ± 0.121 | 53.893 ± 0.428 | 60.606 ± 0.727 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 41.210 ± 0.266 | - | 57.827 ± 0.687 | 30.690 ± 0.546 | 80.466 ± 1.684 | 32.189 ± 0.196 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 30.529 ± 0.244 | - | 44.076 ± 15.922 | 51.604 ± 14.923 | 69.125 ± 7.826 | 21.167 ± 0.677 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.139 ± 0.001 | 0.140 ± 0.000 | unsupported | 0.153 ± 0.007 | 0.036 ± 0.005 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.223 ± 0.007 | 2.217 ± 0.009 | unsupported | 2.073 ± 0.014 | 0.861 ± 0.023 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.115 ± 0.004 | 0.125 ± 0.004 | unsupported | 0.076 ± 0.004 | 0.048 ± 0.006 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.904 ± 0.003 | 0.906 ± 0.041 | unsupported | 0.964 ± 0.012 | 0.255 ± 0.203 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.186 ± 0.001 | 0.222 ± 0.011 | unsupported | 0.134 ± 0.013 | 0.234 ± 0.004 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 2.776 ± 0.020 | 3.094 ± 0.150 | unsupported | 1.365 ± 0.030 | 1.819 ± 0.008 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.236 ± 0.008 | 0.366 ± 0.029 | unsupported | 0.068 ± 0.014 | 0.174 ± 0.002 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 1.262 ± 0.012 | 1.489 ± 0.146 | unsupported | 0.709 ± 0.048 | 0.967 ± 0.035 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.259 ± 0.004 | 0.306 ± 0.009 | unsupported | 0.226 ± 0.000 | 0.189 ± 0.025 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 3.856 ± 0.071 | 4.263 ± 0.133 | unsupported | 4.030 ± 0.027 | 1.001 ± 0.009 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.370 ± 0.019 | 0.533 ± 0.040 | unsupported | 0.105 ± 0.001 | 0.149 ± 0.005 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 1.781 ± 0.023 | 2.110 ± 0.238 | unsupported | 1.459 ± 0.019 | 0.914 ± 0.269 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.059 ± 0.005 | 0.096 ± 0.005 | unsupported | 0.101 ± 0.001 | 0.209 ± 0.003 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.641 ± 0.034 | 0.737 ± 0.030 | unsupported | 1.338 ± 0.023 | 0.970 ± 0.033 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.145 ± 0.012 | 0.276 ± 0.005 | unsupported | 0.066 ± 0.002 | 0.196 ± 0.009 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.460 ± 0.014 | 0.627 ± 0.034 | unsupported | 0.364 ± 0.004 | 0.797 ± 0.038 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.105 ± 0.006 | 0.134 ± 0.003 | unsupported | 0.074 ± 0.001 | 0.220 ± 0.001 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.288 ± 0.016 | 1.566 ± 0.210 | unsupported | 1.653 ± 0.027 | 1.353 ± 0.011 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.213 ± 0.015 | 0.326 ± 0.025 | unsupported | 0.062 ± 0.005 | 0.213 ± 0.002 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.795 ± 0.012 | 0.980 ± 0.067 | unsupported | 0.510 ± 0.020 | 0.996 ± 0.025 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.141 ± 0.002 | 0.186 ± 0.013 | unsupported | unsupported | 0.265 ± 0.024 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 1.885 ± 0.026 | 2.108 ± 0.032 | unsupported | unsupported | 1.810 ± 0.058 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.305 ± 0.009 | 0.436 ± 0.012 | unsupported | unsupported | 0.253 ± 0.010 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 1.278 ± 0.038 | 1.481 ± 0.065 | unsupported | unsupported | 1.376 ± 0.135 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.010 ± 0.000 | 0.020 ± 0.001 | unsupported | 0.013 ± 0.001 | 0.079 ± 0.007 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.123 ± 0.001 | 0.154 ± 0.007 | unsupported | 0.090 ± 0.003 | 0.234 ± 0.003 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.023 ± 0.000 | 0.071 ± 0.006 | unsupported | 0.015 ± 0.000 | 0.045 ± 0.008 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.074 ± 0.002 | 0.127 ± 0.001 | unsupported | 0.097 ± 0.019 | 0.171 ± 0.005 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.034 ± 0.004 | 0.056 ± 0.004 | unsupported | 0.044 ± 0.001 | 0.195 ± 0.002 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.359 ± 0.017 | 0.421 ± 0.017 | unsupported | 1.010 ± 0.015 | 0.781 ± 0.020 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.090 ± 0.012 | 0.151 ± 0.016 | unsupported | 0.090 ± 0.003 | 0.166 ± 0.002 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.242 ± 0.033 | 0.331 ± 0.021 | unsupported | 0.270 ± 0.047 | 0.789 ± 0.054 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.156 ± 0.012 | 0.165 ± 0.006 | unsupported | 0.109 ± 0.001 | 0.172 ± 0.007 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 2.180 ± 0.085 | 2.298 ± 0.109 | unsupported | 1.406 ± 0.031 | 0.946 ± 0.015 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.313 ± 0.031 | 0.355 ± 0.036 | unsupported | 0.080 ± 0.002 | 0.142 ± 0.013 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 1.398 ± 0.133 | 1.442 ± 0.131 | unsupported | 0.666 ± 0.006 | 0.597 ± 0.020 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.154 ± 0.004 | 0.178 ± 0.005 | unsupported | 0.115 ± 0.002 | 0.174 ± 0.004 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 2.332 ± 0.039 | 2.481 ± 0.094 | unsupported | 1.446 ± 0.024 | 0.945 ± 0.013 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.323 ± 0.010 | 0.398 ± 0.037 | unsupported | 0.086 ± 0.008 | 0.156 ± 0.003 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 1.429 ± 0.046 | 1.523 ± 0.168 | unsupported | 0.688 ± 0.043 | 0.624 ± 0.233 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.101 ± 0.002 | 0.137 ± 0.004 | unsupported | 0.077 ± 0.001 | 0.242 ± 0.004 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.293 ± 0.021 | 1.468 ± 0.081 | unsupported | 1.238 ± 0.012 | 1.274 ± 0.033 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.214 ± 0.013 | 0.330 ± 0.010 | unsupported | 0.059 ± 0.003 | 0.217 ± 0.005 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 0.783 ± 0.012 | 0.997 ± 0.085 | unsupported | 0.476 ± 0.013 | 0.934 ± 0.186 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.191 ± 0.003 | 0.203 ± 0.002 | unsupported | 0.266 ± 0.001 | 0.318 ± 0.002 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 2.924 ± 0.035 | 3.091 ± 0.083 | unsupported | 3.900 ± 0.032 | 3.139 ± 0.030 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.306 ± 0.010 | 0.331 ± 0.013 | unsupported | 0.168 ± 0.002 | 0.214 ± 0.012 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 1.596 ± 0.070 | 1.700 ± 0.050 | unsupported | 1.514 ± 0.056 | 1.727 ± 0.004 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.756 ± 0.034 | 1.403 ± 0.017 | unsupported | 0.262 ± 0.002 | 0.244 ± 0.002 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 15.756 ± 0.149 | 20.759 ± 0.201 | unsupported | 4.129 ± 0.129 | 1.475 ± 0.008 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.664 ± 0.004 | 1.364 ± 0.025 | unsupported | 0.187 ± 0.006 | 0.180 ± 0.015 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 8.677 ± 0.220 | 13.679 ± 0.132 | unsupported | 2.086 ± 0.023 | 0.896 ± 0.021 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 24.805 ± 1.083 | - | unsupported | 25.341 ± 0.514 | - | 26.096 ± 0.443 | 29.081 ± 0.267 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 21.620 ± 0.375 | - | unsupported | 21.439 ± 0.390 | - | 25.803 ± 0.718 | 21.696 ± 0.599 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 17.520 ± 0.386 | - | unsupported | 18.266 ± 0.508 | - | 18.213 ± 0.216 | 18.699 ± 1.052 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 15.443 ± 0.682 | - | unsupported | 14.969 ± 0.535 | - | 18.300 ± 0.625 | 14.962 ± 0.581 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 12.511 ± 0.406 | - | unsupported | 18.105 ± 0.770 | - | 12.272 ± 0.223 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 15.227 ± 0.547 | - | unsupported | 15.104 ± 0.362 | - | 12.119 ± 0.161 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 25.130 ± 0.949 | - | unsupported | 25.529 ± 0.703 | - | 25.679 ± 0.893 | 28.784 ± 0.393 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 22.071 ± 0.464 | - | unsupported | 21.114 ± 0.408 | - | 25.675 ± 0.479 | 21.625 ± 0.185 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 48.581 ± 0.445 | - | unsupported | 52.585 ± 12.736 | - | 47.629 ± 0.148 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 24.832 ± 0.221 | - | unsupported | 25.793 ± 0.070 | - | 23.650 ± 0.123 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 48.379 ± 0.142 | - | unsupported | 51.552 ± 0.322 | - | 47.251 ± 0.140 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 24.652 ± 0.276 | - | unsupported | 25.741 ± 0.189 | - | 23.545 ± 0.116 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 24.874 ± 0.607 | - | unsupported | 25.290 ± 0.605 | - | 26.073 ± 0.273 | 28.787 ± 0.645 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 22.008 ± 0.793 | - | unsupported | 21.787 ± 0.747 | - | 26.155 ± 0.413 | 21.715 ± 0.546 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 17.359 ± 0.413 | - | unsupported | 17.408 ± 0.180 | - | 18.046 ± 0.262 | 23.321 ± 0.474 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 15.595 ± 0.585 | - | unsupported | 14.865 ± 0.398 | - | 18.016 ± 1.163 | 14.909 ± 0.108 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.749 ± 0.004 | unsupported | unsupported | 0.908 ± 0.004 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 11.074 ± 0.034 | unsupported | unsupported | 12.930 ± 0.062 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.501 ± 0.004 | unsupported | unsupported | 0.575 ± 0.008 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 6.021 ± 0.045 | unsupported | unsupported | 6.593 ± 0.042 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 24.817 ± 0.277 | - | unsupported | 25.993 ± 1.177 | - | 26.482 ± 0.504 | 28.727 ± 0.785 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 21.600 ± 0.340 | - | unsupported | 21.487 ± 0.414 | - | 25.686 ± 0.734 | 21.121 ± 0.654 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 20.904 ± 0.112 | - | 0.003 ± 0.000 | 19.020 ± 0.649 | 18.204 ± 0.230 | 26.445 ± 0.214 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 16.476 ± 0.138 | - | 0.003 ± 0.000 | 13.650 ± 0.607 | 14.321 ± 0.318 | 26.711 ± 0.563 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 17.716 ± 0.576 | - | 104.900 ± 1.280 | 18.304 ± 0.233 | 17.524 ± 0.449 | 18.732 ± 0.220 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 12.566 ± 0.488 | - | 29.866 ± 0.884 | 12.528 ± 0.698 | 12.078 ± 0.184 | 18.678 ± 0.235 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 133.901 ± 0.761 | - | 133.802 ± 0.823 | 36.214 ± 0.174 | 38.242 ± 1.368 | 37.617 ± 0.239 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 23.961 ± 1.018 | - | 24.539 ± 1.379 | 24.489 ± 0.974 | 22.493 ± 1.383 | 38.741 ± 0.779 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 13.219 ± 0.404 | - | 196.874 ± 1.560 | 22.271 ± 0.569 | 17.463 ± 0.535 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 10.587 ± 0.654 | - | 28.208 ± 1.123 | 12.199 ± 0.128 | 11.976 ± 0.273 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 25.820 ± 0.456 | - | 0.003 ± 0.000 | 25.453 ± 0.691 | 23.396 ± 0.468 | 26.976 ± 0.185 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 17.404 ± 0.752 | - | 0.003 ± 0.000 | 16.880 ± 0.437 | 19.209 ± 0.826 | 27.525 ± 1.097 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 92.003 ± 0.451 | - | 0.003 ± 0.000 | 33.402 ± 0.812 | 45.070 ± 1.814 | 890.123 ± 5.106 | 705.128 ± 0.326 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 32.286 ± 0.481 | - | 0.003 ± 0.000 | 32.771 ± 0.274 | 24.021 ± 0.821 | 891.663 ± 0.744 | 103.480 ± 4.198 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 12.701 ± 0.227 | - | 84.530 ± 0.767 | 15.772 ± 0.167 | 13.488 ± 0.187 | 17.356 ± 0.205 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 7.179 ± 0.121 | - | 16.650 ± 0.493 | 9.968 ± 0.115 | 8.333 ± 0.177 | 20.293 ± 4.675 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 13.130 ± 0.836 | - | 86.118 ± 1.440 | 17.638 ± 0.272 | 13.864 ± 0.244 | 18.116 ± 0.849 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 7.691 ± 0.372 | - | 18.097 ± 0.849 | 9.887 ± 0.096 | 8.199 ± 0.386 | 17.717 ± 0.797 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.001 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 5.997693 | 374855.81 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 4.982350 | 311396.88 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 5.616501 | 351031.31 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 4.818264 | 301141.50 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 6.586667 | 411666.69 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 8.939109 | 558694.31 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 17.455813 | 1090988.31 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 17.151576 | 1071973.50 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 6.140345 | 383771.56 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 6.837083 | 427317.69 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 11.788098 | 736756.12 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 9.782914 | 611432.12 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 7.877149 | 492321.81 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 8.167009 | 510438.06 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 151.573837 | 9473364.81 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 138.266874 | 8641679.62 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 125.296684 | 7831042.75 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 126.545439 | 7909089.94 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 19.413399 | 1213337.44 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 8.117820 | 507363.75 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 6.178438 | 386152.38 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 6.589483 | 411842.69 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 151.466507 | 9466656.69 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 90.344463 | 5646528.94 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 98.320439 | 6145027.44 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 101.733666 | 6358354.12 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 18.545175 | 1159073.44 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 19.131708 | 1195731.75 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 342.337247 | 21396077.94 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 342.046086 | 21377880.38 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 21.819110 | 1363694.38 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 22.044179 | 1377761.19 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 316.828567 | 19801785.44 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 318.593554 | 19912097.12 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 17.393592 | 1087099.50 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 17.832018 | 1114501.12 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 325.771681 | 20360730.06 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 326.193517 | 20387094.81 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 20.040588 | 1252536.75 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 20.327211 | 1270450.69 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 297.154610 | 18572163.12 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 297.744784 | 18609049.00 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.305555 | 36024.30 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 9.822186 | 153471.66 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.917109 | 139329.83 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.959709 | 139995.45 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.890785 | 861348.12 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 16.581949 | 2072743.63 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 17.781621 | 2222702.62 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 17.735155 | 2216894.38 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 3.100716 | 48448.69 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.882834 | 76294.28 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 7.347215 | 114800.23 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.977539 | 124649.05 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.037786 | 254723.25 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.714138 | 964267.25 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.232051 | 904006.38 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.248030 | 906003.75 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.985814 | 234153.34 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.607895 | 134498.36 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 11.910966 | 186108.84 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 14.236378 | 222443.41 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.554550 | 1819318.75 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.923022 | 1365377.75 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 22.205101 | 2775637.62 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 24.752746 | 3094093.25 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.113291 | 173645.17 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.332169 | 67690.14 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 15.114959 | 236171.23 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 23.403638 | 365681.84 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.732473 | 966559.12 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.669630 | 708703.75 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 10.095353 | 1261919.12 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 11.915333 | 1489416.62 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.073849 | 188653.89 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 14.479613 | 226243.95 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 16.594049 | 259282.02 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 19.614344 | 306474.12 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 8.009759 | 1001219.88 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 32.236194 | 4029524.25 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 30.849362 | 3856170.25 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 34.105158 | 4263144.75 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.547996 | 149187.44 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.738621 | 105290.95 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 23.692871 | 370201.11 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 34.088774 | 532637.09 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.309675 | 913709.38 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 11.671760 | 1458970.00 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 14.251926 | 1781490.75 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 16.879249 | 2109906.12 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.376756 | 209011.81 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.439696 | 100620.25 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.749664 | 58588.50 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.163470 | 96304.22 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.758492 | 969811.50 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.704284 | 1338035.50 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.130556 | 641319.50 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.896122 | 737015.25 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.514271 | 195535.48 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.225490 | 66023.28 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 9.274722 | 144917.53 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 17.666783 | 276043.48 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.373321 | 796665.12 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.908818 | 363602.25 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.682409 | 460301.12 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.014971 | 626871.38 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.082741 | 220042.83 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.716806 | 73700.09 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.690913 | 104545.52 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.560666 | 133760.41 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.820775 | 1352596.88 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 13.221442 | 1652680.25 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.306758 | 1288344.75 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 12.528557 | 1566069.62 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.652031 | 213312.98 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.966548 | 61977.31 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 13.613280 | 212707.50 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 20.835641 | 325556.89 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.964646 | 995580.75 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.077896 | 509737.00 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.356239 | 794529.88 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.843649 | 980456.12 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 16.937912 | 264654.88 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 9.020181 | 140940.33 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 11.887432 | 185741.12 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.482035 | 1810254.38 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 15.076593 | 1884574.12 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 16.862169 | 2107771.12 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 16.219883 | 253435.67 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 19.541131 | 305330.17 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 27.918825 | 436231.64 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.008132 | 1376016.50 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 10.221207 | 1277650.88 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 11.849060 | 1481132.50 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.073109 | 79267.33 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.851056 | 13297.75 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.655070 | 10235.47 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.285465 | 20085.39 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 1.872558 | 234069.75 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.722836 | 90354.50 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.986257 | 123282.12 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 1.229079 | 153634.88 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.848506 | 44507.91 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.946082 | 14782.53 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.493092 | 23329.56 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.572096 | 71439.00 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.366716 | 170839.50 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.775434 | 96929.25 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.590650 | 73831.25 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.017174 | 127146.75 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.460299 | 194692.17 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 2.809754 | 43902.41 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.176595 | 34009.30 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.606728 | 56355.12 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.244101 | 780512.62 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 8.079105 | 1009888.12 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.875275 | 359409.38 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.367854 | 420981.75 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.607329 | 165739.52 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.756382 | 89943.47 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.750581 | 89852.83 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 9.666347 | 151036.67 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.315925 | 789490.62 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.162458 | 270307.25 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.935184 | 241898.00 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.651910 | 331488.75 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.995481 | 171804.39 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.970675 | 108916.80 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 9.977705 | 155901.64 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 10.583634 | 165369.28 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.565331 | 945666.38 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.247025 | 1405878.12 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 17.441298 | 2180162.25 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 18.387590 | 2298448.75 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.104025 | 142250.39 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.136527 | 80258.23 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 20.008405 | 312631.33 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 22.721028 | 355016.06 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.772741 | 596592.62 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.330598 | 666324.75 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 11.183942 | 1397992.75 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 11.536648 | 1442081.00 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.141773 | 174090.20 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.390377 | 115474.64 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 9.868612 | 154197.06 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 11.392820 | 178012.81 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.561094 | 945136.75 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.571239 | 1446404.88 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 18.658124 | 2332265.50 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 19.845496 | 2480687.00 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.014515 | 156476.80 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.517767 | 86215.11 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 20.652655 | 322697.73 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 25.440900 | 397514.06 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.988221 | 623527.62 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.500746 | 687593.25 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 11.432945 | 1429118.12 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 12.186870 | 1523358.75 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 15.508487 | 242320.11 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.916067 | 76813.55 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.473989 | 101156.08 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.753054 | 136766.47 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.191523 | 1273940.38 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.904786 | 1238098.25 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.345761 | 1293220.12 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.742232 | 1467779.00 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.897748 | 217152.31 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.787634 | 59181.78 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 13.712911 | 214264.23 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 21.128709 | 330136.08 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.472739 | 934092.38 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.805579 | 475697.38 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.261131 | 782641.38 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.973972 | 996746.50 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 20.379550 | 318430.47 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 16.997243 | 265581.92 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 12.235560 | 191180.62 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 13.008891 | 203263.92 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 25.113969 | 3139246.12 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 31.199051 | 3899881.38 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 23.391181 | 2923897.62 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 24.729483 | 3091185.38 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.698897 | 214045.27 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 10.752457 | 168007.14 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 19.567948 | 305749.19 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 21.171795 | 330809.30 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 13.815253 | 1726906.62 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.111579 | 1513947.38 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 12.767361 | 1595920.12 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 13.598495 | 1699811.88 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 15.589147 | 243580.42 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 16.758909 | 261857.95 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 48.386567 | 756040.11 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 89.822251 | 1403472.67 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.800180 | 1475022.50 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 33.029291 | 4128661.38 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 126.051184 | 15756398.00 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 166.070988 | 20758873.50 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.517244 | 179956.94 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 11.991315 | 187364.30 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 42.468082 | 663563.78 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 87.293928 | 1363967.62 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.170426 | 896303.25 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 16.688424 | 2086053.00 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 69.412663 | 8676582.88 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 109.431566 | 13678945.75 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 14.520869 | 907554.31 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 11.980067 | 748754.19 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 206.874498 | 12929656.12 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 177.185438 | 11074089.88 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 9.197912 | 574869.50 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 8.020119 | 501257.44 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 105.492127 | 6593257.94 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 96.340599 | 6021287.44 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.024455 | 1528.44 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002514 | 157.12 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.039734 | 2483.38 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002524 | 157.75 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000121 | 7.56 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.023634 | 1477.12 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.007324 | 457.75 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000131 | 8.19 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.037120 | 2320.00 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.006462 | 403.88 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000111 | 6.94 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.023414 | 1463.38 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.005381 | 336.31 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000120 | 7.50 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.038011 | 2375.69 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.004529 | 283.06 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000130 | 8.12 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.020970 | 1310.62 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003637 | 227.31 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000100 | 6.25 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.033242 | 2077.62 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003487 | 217.94 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/norm_fro` (c64, threads=1, shape=`2048x1536`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 24.213 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 38.985 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 30.6x faster than the slowest successful cell (`pytorch-cpu`, 38.985 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 21.5x faster than the slowest successful cell (`pytorch-cpu`, 38.985 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 22.0x faster than the slowest successful cell (`pytorch-cpu`, 38.985 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 11.823 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 14.3x faster than the slowest successful cell (`pytorch-cpu`, 11.823 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 12.1x faster than the slowest successful cell (`pytorch-cpu`, 31.745 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 29.4x faster than the slowest successful cell (`pytorch-cpu`, 31.745 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 17.6x faster than the slowest successful cell (`pytorch-cpu`, 31.745 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 18.1x faster than the slowest successful cell (`pytorch-cpu`, 31.745 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 15.0x faster than the slowest successful cell (`pytorch-cpu`, 11.569 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 11.569 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 1130.7x faster than the slowest successful cell (`julia-base`, 3.681 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 1047.9x faster than the slowest successful cell (`julia-base`, 3.423 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 12.6x faster than the slowest successful cell (`jax-cpu`, 0.011 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 16.4x faster than the slowest successful cell (`jax-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 13.5x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x8x8`): `tenferro-direct` is 11.6x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`3x2x2`): `tenferro-direct` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`3x4x4`): `tenferro-direct` is 11.1x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x2x2`): `tenferro-direct` is 12.1x faster than the slowest successful cell (`jax-cpu`, 0.013 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `pytorch-cpu` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 36.0x faster than the slowest successful cell (`jax-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 11.0x faster than the slowest successful cell (`jax-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 24.3x faster than the slowest successful cell (`jax-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 13.3x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 13.3x faster than the slowest successful cell (`jax-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 11.6x faster than the slowest successful cell (`jax-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 13.8x faster than the slowest successful cell (`jax-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.019 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 12.2x faster than the slowest successful cell (`jax-cpu`, 0.025 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x2x2,rhs=1`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.018 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x2x2,rhs=1`): `tenferro-direct` is 15.3x faster than the slowest successful cell (`pytorch-cpu`, 0.027 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`3x2x2,rhs=1`): `tenferro-direct` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 0.027 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`3x4x4,rhs=1`): `tenferro-direct` is 12.2x faster than the slowest successful cell (`pytorch-cpu`, 0.027 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 14.1x faster than the slowest successful cell (`tenferro-eager`, 20.759 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 15.3x faster than the slowest successful cell (`tenferro-eager`, 13.679 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 8277.1x faster than the slowest successful cell (`julia-base`, 26.445 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 7911.9x faster than the slowest successful cell (`julia-base`, 26.711 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/extract_diagonal` (f64, threads=1, shape=`8388608x2x2 -> 8388608x2`): `jax-cpu` is 11.3x faster than the slowest successful cell (`tenferro-trace`, 196.874 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/extract_diagonal` (f64, threads=1, shape=`8388608x2x2 -> 8388608x2`): `tenferro-direct` is 14.9x faster than the slowest successful cell (`tenferro-trace`, 196.874 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 8159.6x faster than the slowest successful cell (`julia-base`, 26.976 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 8250.9x faster than the slowest successful cell (`julia-base`, 27.525 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `jax-cpu` is 19.7x faster than the slowest successful cell (`julia-base`, 890.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `pytorch-cpu` is 26.6x faster than the slowest successful cell (`julia-base`, 890.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 283840.2x faster than the slowest successful cell (`julia-base`, 890.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `jax-cpu` is 37.1x faster than the slowest successful cell (`julia-base`, 891.663 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `pytorch-cpu` is 27.2x faster than the slowest successful cell (`julia-base`, 891.663 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-direct` is 27.6x faster than the slowest successful cell (`julia-base`, 891.663 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 282529.4x faster than the slowest successful cell (`julia-base`, 891.663 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 15.7x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 184.6x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 290.0x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 209.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 297.0x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 163.9x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 346.3x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 5150.2 GB/s at 0.003 ms. The conservative physical lower bound is 0.168 ms, so the cell is 51.5x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 5132.9 GB/s at 0.003 ms. The conservative physical lower bound is 0.042 ms, so the cell is 12.8x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 84017.4 GB/s at 0.003 ms. The conservative physical lower bound is 2.684 ms, so the cell is 840.2x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 79512.9 GB/s at 0.003 ms. The conservative physical lower bound is 0.671 ms, so the cell is 198.8x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 81196.4 GB/s at 0.003 ms. The conservative physical lower bound is 2.684 ms, so the cell is 812.0x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 80466.3 GB/s at 0.003 ms. The conservative physical lower bound is 0.671 ms, so the cell is 201.2x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 42799.0 GB/s at 0.003 ms. The conservative physical lower bound is 1.342 ms, so the cell is 428.0x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 42527.8 GB/s at 0.003 ms. The conservative physical lower bound is 0.336 ms, so the cell is 106.3x faster than the bound.

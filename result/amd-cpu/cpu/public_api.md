# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/amd-cpu/cpu/public_api/20261105_191258/run.yaml`
- Timestamp: `2026-11-05T19:12:58.560776+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/amd-cpu/cpu/public_api/20261105_191258`.

- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

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
- Python platform: `Linux-7.0.0-1017-azure-x86_64-with-glibc2.39`
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

- CSV: `data/results/amd-cpu/cpu/public_api/20261105_191258/cpu_public_api_t1_20261105_191258.csv`
- CSV: `data/results/amd-cpu/cpu/public_api/20261105_191258/cpu_public_api_t4_20261105_191258.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 4.228 ± 0.029 | - | 4.450 ± 0.087 | 4.585 ± 0.056 | 3.959 ± 0.104 | 3.721 ± 1.244 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.400 ± 0.034 | - | 11.002 ± 3.002 | 2.327 ± 0.052 | 3.248 ± 0.064 | 3.335 ± 0.289 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 136.461 ± 7.360 | - | 136.075 ± 8.962 | 133.567 ± 4.314 | 132.088 ± 0.514 | 154.064 ± 4.019 | 160.210 ± 6.338 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 42.852 ± 0.881 | - | 42.640 ± 0.288 | 43.395 ± 0.447 | 67.537 ± 1.081 | 146.568 ± 3.611 | 63.494 ± 0.322 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 112.883 ± 0.856 | - | 116.951 ± 4.339 | 101.782 ± 1.320 | 95.405 ± 0.821 | 131.339 ± 4.429 | 138.351 ± 4.490 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 43.932 ± 0.636 | - | 44.875 ± 0.313 | 31.995 ± 0.336 | 48.562 ± 0.907 | 133.694 ± 1.046 | 65.047 ± 2.210 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 44.554 ± 0.353 | - | 44.521 ± 0.619 | 42.968 ± 0.418 | 66.966 ± 0.472 | 47.236 ± 0.218 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 29.765 ± 0.316 | - | 29.909 ± 0.071 | 29.319 ± 0.817 | 29.862 ± 0.666 | 23.653 ± 0.212 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 46.380 ± 0.731 | - | unsupported | 43.672 ± 0.335 | 68.099 ± 0.558 | 47.481 ± 6.878 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 30.907 ± 0.193 | - | unsupported | 29.556 ± 0.384 | 30.752 ± 0.973 | 24.231 ± 4.822 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 10.977 ± 0.055 | - | 10.979 ± 0.056 | 11.616 ± 0.054 | 12.007 ± 0.068 | 11.928 ± 0.097 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 14.631 ± 0.203 | - | 14.672 ± 0.029 | 14.118 ± 0.430 | 17.669 ± 0.111 | 17.569 ± 0.043 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 72.665 ± 0.077 | - | 71.400 ± 0.069 | 109.572 ± 2.285 | 105.924 ± 2.107 | 85.160 ± 0.666 | 88.484 ± 8.810 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 31.188 ± 0.084 | - | 31.237 ± 0.529 | 43.682 ± 0.642 | 53.042 ± 0.930 | 87.837 ± 1.785 | 39.170 ± 12.968 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 116.114 ± 0.676 | - | 116.185 ± 0.333 | 154.563 ± 2.447 | 178.660 ± 2.797 | 150.868 ± 1.365 | 154.022 ± 1.133 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 49.778 ± 0.317 | - | 49.808 ± 0.149 | 64.228 ± 0.654 | 72.463 ± 0.682 | 152.504 ± 1.292 | 65.180 ± 1.414 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 79.393 ± 1.578 | - | 78.004 ± 4.557 | 73.550 ± 1.798 | 75.187 ± 1.896 | 81.310 ± 0.442 | 81.679 ± 1.111 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 24.968 ± 0.417 | - | 25.252 ± 0.729 | 24.698 ± 0.484 | 40.071 ± 1.065 | 81.235 ± 0.439 | 37.020 ± 1.607 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 2.565 ± 0.124 | - | 21.103 ± 0.167 | 24.252 ± 0.081 | 16.973 ± 0.360 | 4.094 ± 0.030 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 2.990 ± 0.030 | - | 9.567 ± 0.071 | 11.326 ± 0.035 | 2.991 ± 0.299 | 4.192 ± 0.036 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 8.513 ± 0.049 | - | 8.788 ± 0.096 | 9.935 ± 0.100 | 7.662 ± 0.120 | 9.033 ± 0.070 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 5.744 ± 0.057 | - | 13.656 ± 0.242 | 5.767 ± 0.014 | 6.997 ± 0.184 | 6.624 ± 0.201 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 5.614 ± 0.305 | - | 5.799 ± 0.264 | 5.533 ± 0.101 | 5.088 ± 0.021 | 4.863 ± 0.012 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.850 ± 0.135 | - | 19.100 ± 5.948 | 3.654 ± 0.114 | 4.712 ± 0.122 | 4.199 ± 0.016 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 6.552 ± 0.065 | - | 6.576 ± 0.110 | 7.701 ± 0.038 | 5.463 ± 0.046 | 5.290 ± 0.049 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 5.773 ± 0.212 | - | 5.901 ± 0.214 | 5.904 ± 0.086 | 6.202 ± 0.167 | 5.577 ± 0.033 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 45.010 ± 0.841 | - | 43.374 ± 0.161 | 43.032 ± 0.370 | 67.256 ± 0.212 | 44.498 ± 0.298 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 29.771 ± 0.383 | - | 29.060 ± 0.093 | 29.338 ± 0.328 | 29.916 ± 0.506 | 22.201 ± 0.087 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 50.655 ± 0.216 | - | unsupported | 52.024 ± 0.249 | 62.435 ± 1.958 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 25.251 ± 0.187 | - | unsupported | 25.958 ± 0.140 | 35.275 ± 0.332 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 132.581 ± 5.081 | - | 131.983 ± 9.940 | 133.206 ± 2.432 | 138.757 ± 4.262 | 149.335 ± 3.871 | 167.732 ± 10.109 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 42.621 ± 0.397 | - | 42.217 ± 0.615 | 43.019 ± 0.474 | 66.399 ± 1.346 | 145.824 ± 6.046 | 69.476 ± 0.633 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 143.269 ± 8.174 | - | 141.128 ± 8.675 | 149.088 ± 5.995 | 147.804 ± 7.123 | 157.751 ± 8.547 | 170.574 ± 9.752 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 48.456 ± 2.088 | - | 46.615 ± 0.306 | 46.918 ± 0.998 | 74.388 ± 7.345 | 155.959 ± 6.202 | 73.101 ± 2.011 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 27.555 ± 0.241 | - | 61.276 ± 0.213 | 81.022 ± 0.573 | 52.463 ± 0.469 | 81.583 ± 1.003 | 86.455 ± 9.550 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 12.766 ± 0.101 | - | 26.506 ± 0.135 | 34.143 ± 0.802 | 27.046 ± 0.281 | 81.848 ± 0.365 | 32.516 ± 11.531 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 10.681 ± 0.440 | - | 11.737 ± 0.529 | 36.738 ± 1.723 | 39.825 ± 1.046 | 45.884 ± 11.498 | 44.058 ± 13.232 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 7.355 ± 0.204 | - | 7.523 ± 0.166 | 13.364 ± 0.361 | 21.610 ± 1.061 | 47.026 ± 10.574 | 15.659 ± 25.132 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 23.344 ± 0.583 | - | 23.141 ± 0.470 | 55.891 ± 0.765 | 44.876 ± 0.436 | 23.914 ± 0.220 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 14.862 ± 0.912 | - | 14.705 ± 0.260 | 20.836 ± 0.432 | 24.734 ± 0.254 | 21.468 ± 0.498 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 28.549 ± 0.068 | - | 28.873 ± 0.127 | 56.998 ± 2.298 | 110.083 ± 1.217 | 70.196 ± 13.521 | 73.126 ± 13.098 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 11.951 ± 0.079 | - | 12.131 ± 0.026 | 22.543 ± 0.279 | 55.808 ± 0.223 | 71.268 ± 18.903 | 43.628 ± 13.767 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 140.278 ± 2.490 | - | 141.886 ± 8.037 | 144.628 ± 4.525 | 153.654 ± 1.413 | 161.557 ± 7.240 | 175.361 ± 11.502 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 46.367 ± 0.552 | - | 46.198 ± 1.192 | 48.384 ± 2.228 | 79.075 ± 2.088 | 158.468 ± 5.729 | 74.329 ± 0.868 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 20.577 ± 0.032 | - | 20.694 ± 0.059 | 46.352 ± 0.746 | 43.945 ± 0.696 | 65.286 ± 13.488 | 69.842 ± 12.904 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 8.483 ± 0.041 | - | 8.520 ± 0.039 | 17.232 ± 0.436 | 22.318 ± 0.204 | 75.196 ± 10.293 | 40.100 ± 12.438 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 20.428 ± 0.150 | - | 20.679 ± 0.070 | 33.639 ± 0.651 | 30.349 ± 0.099 | 38.749 ± 1.155 | 41.964 ± 10.066 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 8.177 ± 0.016 | - | 8.127 ± 0.084 | 11.567 ± 0.373 | 14.720 ± 0.120 | 38.755 ± 0.285 | 16.822 ± 10.757 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 25.473 ± 0.064 | - | 25.795 ± 0.085 | 53.827 ± 0.713 | 67.645 ± 1.314 | 85.476 ± 23.387 | 98.347 ± 7.510 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 10.816 ± 0.009 | - | 10.683 ± 0.085 | 19.582 ± 0.398 | 34.276 ± 0.206 | 85.675 ± 15.230 | 38.901 ± 17.339 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 13.268 ± 0.062 | - | 13.354 ± 0.038 | 34.156 ± 0.458 | 38.370 ± 0.243 | 47.251 ± 1.136 | 51.283 ± 8.163 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 5.671 ± 0.130 | - | 5.667 ± 0.025 | 12.110 ± 0.327 | 19.601 ± 0.185 | 47.430 ± 0.679 | 20.770 ± 11.440 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.461 ± 0.014 | 0.498 ± 0.018 | unsupported | unsupported | 0.542 ± 0.012 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.358 ± 0.021 | 0.411 ± 0.008 | unsupported | unsupported | 0.433 ± 0.011 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 146.120 ± 4.795 | - | 145.718 ± 8.926 | 143.015 ± 7.220 | 150.179 ± 3.670 | 164.577 ± 5.538 | 180.000 ± 7.535 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 47.306 ± 0.508 | - | 45.839 ± 0.609 | 48.739 ± 1.436 | 73.697 ± 0.779 | 157.369 ± 4.799 | 72.254 ± 1.716 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 142.558 ± 9.890 | - | 147.237 ± 9.894 | 140.099 ± 7.709 | 151.330 ± 1.121 | 167.340 ± 4.776 | 178.092 ± 9.775 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 46.642 ± 1.134 | - | 46.059 ± 0.804 | 47.623 ± 0.853 | 74.211 ± 3.014 | 157.885 ± 1.453 | 73.936 ± 0.734 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 138.790 ± 5.494 | - | 144.129 ± 6.790 | 146.766 ± 5.138 | 147.484 ± 3.605 | 158.398 ± 7.464 | 180.759 ± 8.323 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 47.115 ± 0.750 | - | 47.031 ± 0.871 | 47.762 ± 1.351 | 73.903 ± 0.389 | 157.508 ± 5.968 | 72.564 ± 1.827 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 132.031 ± 5.175 | - | 132.834 ± 9.488 | 135.034 ± 4.161 | 138.299 ± 1.105 | 147.984 ± 5.090 | 168.219 ± 8.209 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 42.246 ± 0.383 | - | 42.357 ± 0.686 | 42.873 ± 0.620 | 67.328 ± 1.012 | 147.552 ± 2.123 | 69.703 ± 0.757 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 58.485 ± 0.250 | - | 58.894 ± 0.116 | 64.440 ± 0.168 | 75.899 ± 0.388 | 126.975 ± 1.075 | 123.927 ± 8.594 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 23.733 ± 0.463 | - | 23.733 ± 1.172 | 22.316 ± 0.320 | 38.193 ± 0.236 | 126.856 ± 1.230 | 49.976 ± 12.680 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 16.880 ± 0.206 | - | 17.027 ± 0.396 | 12.949 ± 0.203 | 27.544 ± 0.375 | 13.278 ± 0.141 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 8.258 ± 0.333 | - | 7.704 ± 0.308 | 7.373 ± 0.123 | 15.019 ± 0.547 | 13.751 ± 0.686 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.936 ± 0.105 | - | 2.068 ± 0.064 | 24.544 ± 0.170 | 2.508 ± 0.021 | 1.150 ± 0.094 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.822 ± 0.026 | - | 0.811 ± 0.027 | 11.308 ± 0.475 | 1.653 ± 0.291 | 1.631 ± 0.076 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 2.556 ± 0.103 | - | 2.493 ± 0.079 | 4.094 ± 0.024 | 2.860 ± 0.204 | 1.485 ± 0.162 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 1.362 ± 0.057 | - | 1.443 ± 0.083 | 2.326 ± 0.033 | 1.860 ± 0.138 | 1.627 ± 0.077 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 16.639 ± 0.259 | - | 16.407 ± 0.130 | 12.971 ± 0.343 | 27.449 ± 0.303 | 13.151 ± 0.285 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 7.658 ± 0.253 | - | 7.585 ± 0.121 | 7.294 ± 0.190 | 14.975 ± 1.115 | 14.528 ± 0.843 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.987 ± 0.112 | - | 1.900 ± 0.124 | 23.680 ± 0.082 | 2.500 ± 0.088 | 1.439 ± 0.044 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.822 ± 0.052 | - | 0.820 ± 0.058 | 10.860 ± 0.193 | 1.379 ± 0.057 | 1.481 ± 0.011 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 10.515 ± 0.149 | - | 11.132 ± 0.689 | 16.280 ± 0.072 | 12.894 ± 0.285 | 6.722 ± 0.202 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 5.657 ± 0.374 | - | 5.642 ± 0.654 | 8.865 ± 0.106 | 7.090 ± 0.673 | 6.981 ± 0.145 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 11.246 ± 0.430 | - | 11.302 ± 0.088 | 11.509 ± 0.134 | 20.985 ± 0.250 | 11.263 ± 0.136 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 7.079 ± 0.134 | - | 7.800 ± 0.416 | 7.181 ± 0.110 | 12.082 ± 0.173 | 11.602 ± 0.275 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 1.179 ± 0.054 | - | 1.073 ± 0.136 | 2.185 ± 0.058 | 2.042 ± 0.052 | 0.976 ± 0.065 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.453 ± 0.077 | - | 0.505 ± 0.029 | 1.067 ± 0.028 | 1.202 ± 0.061 | 1.095 ± 0.113 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.366 ± 0.032 | - | 1.416 ± 0.064 | 0.948 ± 0.024 | 3.356 ± 0.248 | 1.098 ± 0.081 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.724 ± 0.041 | - | 0.688 ± 0.054 | 0.458 ± 0.022 | 1.775 ± 0.157 | 1.244 ± 0.061 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 11.240 ± 0.332 | - | 11.030 ± 0.420 | 11.937 ± 1.180 | 21.134 ± 0.612 | 11.139 ± 0.167 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 7.097 ± 0.238 | - | 7.032 ± 0.252 | 7.063 ± 0.344 | 11.891 ± 0.439 | 11.715 ± 0.269 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 1.059 ± 0.074 | - | 1.103 ± 0.109 | 2.562 ± 0.040 | 2.033 ± 0.016 | 0.950 ± 0.099 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.539 ± 0.060 | - | 0.506 ± 0.019 | 0.984 ± 0.048 | 1.169 ± 0.056 | 1.286 ± 0.100 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.152 ± 0.112 | - | 1.354 ± 0.120 | 1.051 ± 0.043 | 3.233 ± 0.279 | 1.139 ± 0.056 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.646 ± 0.084 | - | 0.836 ± 0.372 | 0.425 ± 0.047 | 1.911 ± 0.135 | 1.296 ± 0.040 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 1.312 ± 0.187 | 1.327 ± 0.083 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.476 ± 0.038 | 0.429 ± 0.047 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 37.683 ± 0.116 | - | 39.120 ± 0.239 | 58.302 ± 0.460 | 72.896 ± 1.401 | 72.737 ± 6.081 | 85.976 ± 7.940 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 18.308 ± 0.310 | - | 18.462 ± 0.374 | 21.026 ± 0.360 | 38.525 ± 1.450 | 73.315 ± 8.167 | 42.904 ± 11.804 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 147.553 ± 6.599 | - | 154.193 ± 9.455 | 153.818 ± 3.544 | 255.151 ± 5.519 | 276.239 ± 3.162 | 275.269 ± 2.497 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 48.109 ± 0.463 | - | 48.401 ± 0.302 | 48.371 ± 0.438 | 130.019 ± 0.844 | 269.806 ± 4.897 | 110.432 ± 2.151 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 139.111 ± 9.540 | - | 148.358 ± 9.779 | 156.178 ± 13.312 | 159.286 ± 2.029 | 165.370 ± 10.614 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 50.052 ± 1.063 | - | 48.778 ± 0.164 | 54.688 ± 0.530 | 82.987 ± 0.918 | 161.367 ± 5.583 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 136.268 ± 3.981 | - | 136.976 ± 9.348 | 139.608 ± 3.004 | 136.748 ± 0.960 | 149.669 ± 4.637 | 162.533 ± 7.079 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 42.310 ± 0.219 | - | 42.351 ± 0.351 | 43.997 ± 0.388 | 67.597 ± 1.023 | 145.792 ± 3.574 | 67.918 ± 3.039 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 24.796 ± 0.127 | - | 25.260 ± 0.067 | 58.800 ± 2.135 | 110.168 ± 0.998 | 75.419 ± 8.557 | 72.545 ± 12.933 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 10.290 ± 0.072 | - | 10.462 ± 0.040 | 22.814 ± 0.425 | 55.460 ± 0.321 | 68.976 ± 14.834 | 30.101 ± 16.549 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 141.746 ± 6.110 | - | 142.040 ± 8.529 | 193.410 ± 2.873 | 139.292 ± 6.491 | 225.777 ± 2.937 | 225.443 ± 2.323 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 45.719 ± 1.185 | - | 45.467 ± 0.844 | 70.191 ± 0.599 | 73.099 ± 0.792 | 221.996 ± 3.839 | 93.526 ± 1.503 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 134.844 ± 8.526 | - | 142.342 ± 8.829 | 147.165 ± 5.132 | 147.724 ± 6.933 | 162.798 ± 7.735 | 181.366 ± 10.708 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 47.189 ± 0.771 | - | 47.118 ± 1.303 | 47.767 ± 1.226 | 76.933 ± 3.429 | 155.917 ± 3.943 | 73.043 ± 0.181 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 51.291 ± 0.042 | - | 51.788 ± 0.123 | 76.750 ± 2.248 | 42.107 ± 1.326 | 81.608 ± 12.178 | 89.276 ± 20.137 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 20.684 ± 0.170 | - | 20.722 ± 0.133 | 29.114 ± 0.628 | 21.321 ± 0.348 | 89.323 ± 14.065 | 34.920 ± 14.074 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 139.344 ± 9.414 | 141.336 ± 3.372 | unsupported | 160.917 ± 10.186 | 156.568 ± 1.540 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 49.369 ± 0.519 | 50.029 ± 1.337 | unsupported | 54.450 ± 1.018 | 82.397 ± 0.983 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.933 ± 0.084 | - | 0.961 ± 0.102 | 0.883 ± 0.055 | 0.972 ± 0.128 | 7.825 ± 4.258 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.421 ± 0.071 | - | 0.443 ± 0.183 | 0.392 ± 0.032 | 1.054 ± 0.147 | 8.228 ± 4.395 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 1.184 ± 0.294 | - | 1.141 ± 0.163 | 0.955 ± 0.171 | 1.338 ± 0.231 | 8.106 ± 4.164 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.556 ± 0.317 | - | 0.432 ± 0.060 | 0.404 ± 0.051 | 0.683 ± 0.112 | 1.591 ± 1.203 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.334 ± 0.083 | - | unsupported | 1.594 ± 0.053 | 1.558 ± 0.248 | 8.480 ± 4.358 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.616 ± 0.184 | - | unsupported | 0.846 ± 0.130 | 1.173 ± 0.163 | 8.808 ± 7.198 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.422 ± 0.037 | - | 0.495 ± 0.026 | 0.696 ± 0.020 | 0.480 ± 0.033 | 1.256 ± 0.132 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.357 ± 0.072 | - | 0.413 ± 0.042 | 0.348 ± 0.017 | 0.390 ± 0.029 | 1.347 ± 0.116 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 1.031 ± 0.214 | - | 0.919 ± 0.048 | 1.352 ± 0.091 | 1.112 ± 0.106 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.479 ± 0.260 | - | 0.546 ± 0.195 | 0.627 ± 0.012 | 1.099 ± 0.043 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.960 ± 0.068 | - | 0.939 ± 0.033 | 0.954 ± 0.112 | 0.930 ± 0.060 | 1.730 ± 7.715 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.407 ± 0.090 | - | 0.518 ± 0.326 | 0.535 ± 0.187 | 0.737 ± 0.232 | 9.205 ± 7.936 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.811 ± 0.020 | - | 0.936 ± 0.022 | 0.531 ± 0.046 | 0.492 ± 0.027 | 0.611 ± 0.842 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 1.193 ± 0.012 | - | 1.343 ± 0.033 | 0.653 ± 0.009 | 0.568 ± 0.036 | 1.392 ± 0.093 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.931 ± 0.044 | - | 0.003 ± 0.000 | 1.861 ± 0.034 | 1.550 ± 0.045 | 9.808 ± 6.793 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.864 ± 0.051 | - | 0.004 ± 0.000 | 0.927 ± 0.101 | 1.134 ± 0.098 | 10.440 ± 1.436 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.050 ± 0.000 | - | 0.054 ± 0.001 | 0.076 ± 0.002 | 0.114 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.093 ± 0.000 | - | 0.101 ± 0.000 | 0.125 ± 0.002 | 0.177 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.233 ± 0.003 | - | 0.264 ± 0.012 | 0.279 ± 0.013 | 0.353 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.081 ± 0.003 | - | 0.091 ± 0.006 | 0.148 ± 0.004 | 0.140 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.057 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.013 ± 0.006 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.054 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.679 ± 0.014 | - | 0.743 ± 0.020 | 1.162 ± 0.011 | 0.727 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.001 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.026 ± 0.000 | - | 0.032 ± 0.000 | 0.039 ± 0.001 | 0.084 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.003 ± 0.004 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.055 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.051 ± 0.000 | - | 0.099 ± 0.003 | 0.085 ± 0.001 | 0.089 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.094 ± 0.000 | - | 0.156 ± 0.005 | 0.125 ± 0.017 | 0.142 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.234 ± 0.012 | - | 0.399 ± 0.019 | 0.338 ± 0.017 | 0.368 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.081 ± 0.001 | - | 0.123 ± 0.005 | 0.181 ± 0.003 | 0.131 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.008 ± 0.004 | 0.010 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.005 ± 0.000 | - | 0.014 ± 0.000 | 0.016 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.008 ± 0.001 | 0.010 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.008 ± 0.001 | 0.009 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.009 ± 0.001 | 0.009 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.004 ± 0.000 | - | 0.009 ± 0.001 | 0.010 ± 0.014 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.008 ± 0.001 | 0.010 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.009 ± 0.001 | 0.010 ± 0.001 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.005 ± 0.000 | - | 0.010 ± 0.002 | 0.011 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.677 ± 0.012 | - | 0.936 ± 0.014 | 1.392 ± 0.028 | 1.996 ± 1.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.009 ± 0.001 | 0.010 ± 0.001 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.027 ± 0.000 | - | 0.052 ± 0.003 | 0.059 ± 0.003 | 0.071 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.003 ± 0.000 | - | 0.009 ± 0.001 | 0.010 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.006 ± 0.000 | - | 0.011 ± 0.001 | 0.012 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.600 ± 0.014 | - | 0.612 ± 0.041 | 0.704 ± 0.016 | 0.863 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 2.104 ± 0.005 | - | 2.120 ± 0.008 | 2.129 ± 0.008 | 2.739 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 6.457 ± 0.050 | - | 6.500 ± 0.028 | 6.526 ± 0.059 | 8.943 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 3.668 ± 0.017 | - | 3.708 ± 0.017 | 3.261 ± 0.019 | 2.615 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.001 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.110 ± 0.002 | - | 0.123 ± 0.031 | 0.121 ± 0.007 | 0.184 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.001 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.008 ± 0.001 | - | 0.012 ± 0.000 | 0.018 ± 0.002 | 0.033 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.007 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.012 ± 0.000 | 0.015 ± 0.001 | 0.037 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.021 ± 0.000 | - | 0.025 ± 0.007 | 0.030 ± 0.001 | 0.100 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.012 ± 0.001 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | 0.014 ± 0.000 | 0.017 ± 0.001 | 0.042 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.027 ± 0.000 | - | 0.031 ± 0.001 | 0.036 ± 0.003 | 0.108 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 29.343 ± 0.166 | - | 29.524 ± 0.161 | 26.032 ± 0.171 | 20.483 ± 0.122 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.014 ± 0.001 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.865 ± 0.007 | - | 0.886 ± 0.032 | 0.885 ± 0.007 | 0.977 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.018 ± 0.000 | - | 0.022 ± 0.000 | 0.026 ± 0.001 | 0.044 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.052 ± 0.000 | - | 0.057 ± 0.003 | 0.060 ± 0.001 | 0.145 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.963 ± 0.007 | - | 0.979 ± 0.004 | 0.719 ± 0.415 | 0.901 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 3.353 ± 0.009 | - | 3.386 ± 0.015 | 2.116 ± 0.009 | 2.227 ± 0.084 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 6.460 ± 0.017 | - | 6.541 ± 0.022 | 9.329 ± 0.023 | 5.941 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 3.668 ± 0.104 | - | 3.685 ± 0.083 | 2.307 ± 0.058 | 3.474 ± 0.546 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.005 ± 0.000 | - | 0.012 ± 0.001 | 0.013 ± 0.002 | 0.026 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.111 ± 0.016 | - | 0.176 ± 0.008 | 0.180 ± 0.025 | 0.198 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.008 ± 0.000 | - | 0.015 ± 0.000 | 0.011 ± 0.001 | 0.025 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.014 ± 0.000 | - | 0.020 ± 0.001 | 0.016 ± 0.001 | 0.032 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.007 ± 0.000 | - | 0.014 ± 0.001 | 0.016 ± 0.002 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.014 ± 0.000 | - | 0.021 ± 0.001 | 0.016 ± 0.001 | 0.033 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.034 ± 0.000 | - | 0.040 ± 0.001 | 0.029 ± 0.001 | 0.063 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.008 ± 0.000 | - | 0.015 ± 0.001 | 0.017 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.017 ± 0.000 | - | 0.024 ± 0.000 | 0.018 ± 0.005 | 0.036 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.043 ± 0.002 | - | 0.050 ± 0.000 | 0.035 ± 0.001 | 0.095 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 28.896 ± 0.317 | - | 39.294 ± 0.294 | 18.220 ± 0.153 | 29.311 ± 7.845 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.012 ± 0.000 | - | 0.018 ± 0.001 | 0.021 ± 0.004 | 0.033 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.869 ± 0.002 | - | 1.321 ± 0.426 | 1.264 ± 0.023 | 1.993 ± 0.397 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.030 ± 0.000 | - | 0.037 ± 0.000 | 0.026 ± 0.001 | 0.045 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.081 ± 0.002 | - | 0.090 ± 0.002 | 0.060 ± 0.000 | 0.106 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.652 ± 0.029 | - | 0.659 ± 0.029 | 0.519 ± 0.013 | 0.456 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 2.175 ± 0.012 | - | 2.190 ± 0.018 | 1.949 ± 0.004 | 1.943 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 5.760 ± 0.049 | - | 5.795 ± 0.058 | 5.674 ± 0.046 | 6.445 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.330 ± 0.018 | - | 1.341 ± 0.017 | 1.341 ± 0.028 | 1.471 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.003 ± 0.001 | - | 0.006 ± 0.000 | 0.008 ± 0.001 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.070 ± 0.003 | - | 0.078 ± 0.007 | 0.091 ± 0.006 | 0.142 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.016 ± 0.001 | 0.028 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.011 | 0.010 ± 0.001 | 0.042 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.012 ± 0.000 | 0.014 ± 0.001 | 0.032 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.018 ± 0.000 | - | 0.023 ± 0.002 | 0.026 ± 0.001 | 0.057 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.003 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | 0.014 ± 0.000 | 0.016 ± 0.000 | 0.031 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.024 ± 0.001 | - | 0.028 ± 0.000 | 0.032 ± 0.002 | 0.063 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 10.524 ± 0.051 | - | 10.682 ± 0.185 | 10.561 ± 0.124 | 11.286 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.007 ± 0.000 | - | 0.011 ± 0.003 | 0.011 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.559 ± 0.033 | - | 0.569 ± 0.014 | 0.632 ± 0.050 | 0.748 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.018 ± 0.000 | - | 0.022 ± 0.000 | 0.023 ± 0.001 | 0.047 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.046 ± 0.001 | - | 0.050 ± 0.000 | 0.053 ± 0.001 | 0.099 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 1.053 ± 0.008 | - | 1.066 ± 0.003 | 0.799 ± 0.091 | 0.449 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 3.448 ± 0.010 | - | 3.482 ± 0.024 | 1.914 ± 0.035 | 1.283 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 5.741 ± 1.939 | - | 5.784 ± 0.021 | 8.430 ± 0.020 | 3.734 ± 0.080 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.273 ± 0.028 | - | 1.299 ± 0.035 | 1.200 ± 0.036 | 1.797 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.005 ± 0.001 | - | 0.011 ± 0.001 | 0.011 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.117 ± 0.004 | - | 0.134 ± 0.016 | 0.136 ± 0.010 | 0.139 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.008 ± 0.000 | - | 0.014 ± 0.000 | 0.010 ± 0.000 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.013 ± 0.000 | - | 0.019 ± 0.001 | 0.014 ± 0.001 | 0.036 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.007 ± 0.000 | - | 0.014 ± 0.000 | 0.014 ± 0.003 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.015 ± 0.001 | - | 0.022 ± 0.001 | 0.014 ± 0.001 | 0.028 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.031 ± 0.001 | - | 0.038 ± 0.004 | 0.027 ± 0.035 | 0.053 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.008 ± 0.000 | - | 0.015 ± 0.000 | 0.015 ± 0.002 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.018 ± 0.001 | - | 0.025 ± 0.001 | 0.017 ± 0.001 | 0.034 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.040 ± 0.003 | - | 0.046 ± 0.003 | 0.031 ± 0.002 | 0.046 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 10.116 ± 0.391 | - | 30.578 ± 0.519 | 9.389 ± 0.050 | 14.987 ± 1.315 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.012 ± 0.000 | - | 0.019 ± 0.000 | 0.017 ± 0.001 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.889 ± 0.008 | - | 0.910 ± 0.003 | 0.953 ± 0.006 | 0.498 ± 0.066 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.032 ± 0.000 | - | 0.038 ± 0.001 | 0.024 ± 0.009 | 0.044 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.074 ± 0.002 | - | 0.082 ± 0.003 | 0.106 ± 0.040 | 0.094 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.267 ± 0.011 | - | 0.276 ± 0.009 | 0.238 ± 0.019 | 0.439 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 1.084 ± 0.004 | - | 1.097 ± 0.015 | 0.949 ± 0.002 | 1.936 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 3.080 ± 0.006 | - | 3.127 ± 0.095 | 2.898 ± 0.034 | 6.415 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.511 ± 0.014 | - | 0.528 ± 0.012 | 0.558 ± 0.012 | 1.454 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.002 ± 0.002 | - | 0.005 ± 0.000 | 0.010 ± 0.004 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.031 ± 0.000 | - | 0.037 ± 0.005 | 0.041 ± 0.001 | 0.129 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.002 | 0.011 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.008 ± 0.001 | 0.013 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.001 | 0.012 ± 0.001 | 0.020 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.010 ± 0.000 | - | 0.014 ± 0.000 | 0.020 ± 0.002 | 0.042 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.002 | 0.011 ± 0.001 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.013 ± 0.001 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.018 ± 0.007 | - | 0.017 ± 0.003 | 0.021 ± 0.000 | 0.050 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 4.125 ± 0.027 | - | 4.135 ± 0.059 | 4.306 ± 0.062 | 11.205 ± 0.097 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.232 ± 0.004 | - | 0.239 ± 0.013 | 0.246 ± 0.017 | 0.742 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.017 ± 0.001 | 0.039 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.025 ± 0.000 | - | 0.029 ± 0.000 | 0.032 ± 0.001 | 0.094 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.463 ± 0.025 | - | 0.490 ± 0.023 | 0.399 ± 0.018 | 0.462 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 1.568 ± 0.020 | - | 1.588 ± 0.017 | 0.953 ± 0.007 | 1.278 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 4.279 ± 0.652 | - | 4.201 ± 0.028 | 3.663 ± 0.018 | 3.701 ± 0.110 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.693 ± 0.009 | - | 0.711 ± 0.032 | 0.703 ± 0.026 | 1.782 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.005 ± 0.002 | - | 0.010 ± 0.000 | 0.014 ± 0.003 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.041 ± 0.004 | - | 0.049 ± 0.006 | 0.055 ± 0.003 | 0.117 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.005 ± 0.001 | - | 0.011 ± 0.001 | 0.016 ± 0.002 | 0.017 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.008 ± 0.001 | - | 0.013 ± 0.003 | 0.012 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.005 ± 0.001 | - | 0.010 ± 0.000 | 0.016 ± 0.007 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.009 ± 0.001 | - | 0.014 ± 0.004 | 0.018 ± 0.001 | 0.027 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.016 ± 0.002 | - | 0.021 ± 0.000 | 0.019 ± 0.001 | 0.033 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.006 ± 0.001 | - | 0.011 ± 0.000 | 0.016 ± 0.002 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.010 ± 0.000 | - | 0.015 ± 0.001 | 0.019 ± 0.002 | 0.025 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.020 ± 0.000 | - | 0.025 ± 0.003 | 0.022 ± 0.001 | 0.053 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 5.463 ± 0.080 | - | 14.774 ± 2.416 | 5.292 ± 0.034 | 13.913 ± 0.111 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.007 ± 0.001 | - | 0.012 ± 0.001 | 0.017 ± 0.002 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.298 ± 0.015 | - | 0.312 ± 0.017 | 0.320 ± 0.020 | 0.449 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.016 ± 0.001 | - | 0.021 ± 0.001 | 0.024 ± 0.003 | 0.036 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.037 ± 0.001 | - | 0.043 ± 0.003 | 0.032 ± 0.000 | 0.077 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.166 ± 0.001 | - | 0.166 ± 0.002 | 0.190 ± 0.021 | 0.129 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.222 ± 0.011 | - | 0.226 ± 0.002 | 0.254 ± 0.018 | 0.215 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.515 ± 0.019 | - | 0.538 ± 0.013 | 0.550 ± 0.016 | 0.596 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.136 ± 0.003 | - | 0.144 ± 0.003 | 0.180 ± 0.029 | 0.207 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.014 ± 0.002 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.012 ± 0.001 | - | 0.016 ± 0.000 | 0.025 ± 0.001 | 0.054 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.000 | 0.022 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.014 ± 0.003 | - | 0.018 ± 0.001 | 0.030 ± 0.021 | 0.057 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.000 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.001 | 0.015 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.016 ± 0.001 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.047 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.002 | 0.016 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.016 ± 0.005 | 0.020 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.006 | 0.037 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 1.102 ± 0.057 | - | 1.132 ± 0.019 | 1.312 ± 0.017 | 1.271 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.001 | 0.016 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.076 ± 0.001 | - | 0.083 ± 0.004 | 0.088 ± 0.003 | 0.126 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.020 ± 0.001 | 0.055 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.193 ± 0.001 | - | 0.213 ± 0.017 | 0.175 ± 0.033 | 0.127 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.294 ± 0.016 | - | 0.317 ± 0.015 | 0.238 ± 0.019 | 0.238 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.773 ± 0.008 | - | 0.828 ± 0.007 | 0.396 ± 0.026 | 0.658 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.194 ± 0.004 | - | 0.205 ± 0.017 | 0.226 ± 0.023 | 0.224 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.024 ± 0.004 | 0.024 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.021 ± 0.000 | - | 0.027 ± 0.004 | 0.037 ± 0.001 | 0.052 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.006 ± 0.001 | - | 0.010 ± 0.000 | 0.023 ± 0.000 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.023 ± 0.001 | - | 0.031 ± 0.004 | 0.041 ± 0.001 | 0.058 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.006 ± 0.001 | - | 0.011 ± 0.000 | 0.023 ± 0.005 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.006 ± 0.001 | - | 0.011 ± 0.000 | 0.025 ± 0.008 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.006 ± 0.001 | - | 0.011 ± 0.001 | 0.026 ± 0.007 | 0.024 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.008 ± 0.000 | - | 0.013 ± 0.001 | 0.028 ± 0.001 | 0.033 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.006 ± 0.001 | - | 0.011 ± 0.000 | 0.025 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.006 ± 0.001 | - | 0.011 ± 0.000 | 0.025 ± 0.001 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.008 ± 0.000 | - | 0.014 ± 0.000 | 0.028 ± 0.016 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 1.522 ± 0.075 | - | 1.564 ± 0.050 | 0.705 ± 0.015 | 0.962 ± 0.171 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.012 ± 0.000 | 0.027 ± 0.004 | 0.019 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.124 ± 0.003 | - | 0.135 ± 0.003 | 0.076 ± 0.001 | 0.174 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.007 ± 0.001 | - | 0.013 ± 0.000 | 0.027 ± 0.000 | 0.028 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.012 ± 0.002 | - | 0.017 ± 0.000 | 0.030 ± 0.000 | 0.050 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.250 ± 0.028 | - | 0.257 ± 0.004 | 0.221 ± 0.004 | 0.337 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.594 ± 0.013 | - | 0.611 ± 0.030 | 0.525 ± 0.011 | 0.867 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.336 ± 0.015 | - | 1.363 ± 0.015 | 1.360 ± 0.016 | 2.332 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.496 ± 0.013 | - | 0.513 ± 0.026 | 0.568 ± 0.012 | 0.632 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.001 | 0.024 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.018 ± 0.000 | - | 0.022 ± 0.000 | 0.027 ± 0.006 | 0.064 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.019 ± 0.000 | - | 0.024 ± 0.007 | 0.029 ± 0.003 | 0.063 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.003 ± 0.001 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.027 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.027 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.006 | 0.007 ± 0.000 | 0.023 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.036 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 3.942 ± 0.095 | - | 4.834 ± 0.009 | 4.496 ± 0.094 | 4.762 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.026 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.130 ± 0.001 | - | 0.135 ± 0.003 | 0.179 ± 0.015 | 0.215 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.032 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.012 ± 0.000 | - | 0.016 ± 0.000 | 0.016 ± 0.000 | 0.039 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.251 ± 0.013 | - | 0.424 ± 0.016 | 0.338 ± 0.016 | 0.342 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.599 ± 0.012 | - | 0.903 ± 0.001 | 0.769 ± 0.006 | 0.895 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.326 ± 0.023 | - | 1.358 ± 0.428 | 2.036 ± 0.014 | 1.363 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.574 ± 0.022 | - | 0.559 ± 0.022 | 0.525 ± 0.030 | 0.996 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.001 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.019 ± 0.000 | - | 0.037 ± 0.003 | 0.044 ± 0.000 | 0.064 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.010 ± 0.001 | 0.011 ± 0.002 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.033 ± 0.001 | - | 0.040 ± 0.002 | 0.048 ± 0.015 | 0.073 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.012 ± 0.000 | 0.012 ± 0.001 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.012 ± 0.000 | 0.012 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.007 ± 0.000 | - | 0.016 ± 0.000 | 0.016 ± 0.000 | 0.033 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.011 ± 0.001 | 0.011 ± 0.001 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | 0.013 ± 0.001 | 0.013 ± 0.000 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.008 ± 0.000 | - | 0.018 ± 0.001 | 0.018 ± 0.000 | 0.035 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 4.598 ± 0.024 | - | 16.705 ± 1.305 | 3.816 ± 0.060 | 7.525 ± 0.477 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.005 ± 0.000 | - | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.130 ± 0.001 | - | 0.227 ± 0.004 | 0.292 ± 0.016 | 0.237 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.007 ± 0.000 | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.032 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.013 ± 0.005 | - | 0.026 ± 0.001 | 0.026 ± 0.001 | 0.045 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.163 ± 0.021 | - | 0.146 ± 0.004 | 0.099 ± 0.002 | 0.260 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.222 ± 0.003 | - | 0.216 ± 0.043 | 0.140 ± 0.004 | 0.366 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.547 ± 0.012 | - | 0.565 ± 0.011 | 0.406 ± 0.015 | 0.809 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.118 ± 0.003 | - | 0.137 ± 0.041 | 0.147 ± 0.005 | 0.183 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.005 | 0.016 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.014 ± 0.000 | - | 0.020 ± 0.013 | 0.030 ± 0.004 | 0.053 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.001 | 0.018 ± 0.004 | 0.018 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.001 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.001 | 0.023 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.019 ± 0.000 | 0.025 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.001 | 0.020 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.019 ± 0.001 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.928 ± 0.009 | - | 1.001 ± 0.011 | 1.029 ± 0.021 | 1.007 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.023 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.074 ± 0.004 | - | 0.079 ± 0.001 | 0.081 ± 0.007 | 0.122 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.018 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.012 ± 0.001 | 0.021 ± 0.001 | 0.025 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.163 ± 0.002 | - | 0.183 ± 0.018 | 0.131 ± 0.001 | 0.268 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.237 ± 0.003 | - | 0.250 ± 0.092 | 0.173 ± 0.002 | 0.385 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.551 ± 0.024 | - | 0.752 ± 0.013 | 0.393 ± 0.020 | 0.858 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.119 ± 0.003 | - | 0.205 ± 0.073 | 0.201 ± 0.004 | 0.177 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.020 ± 0.026 | 0.028 ± 0.005 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.012 ± 0.001 | - | 0.034 ± 0.006 | 0.045 ± 0.006 | 0.054 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.015 ± 0.000 | 0.025 ± 0.001 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.015 ± 0.000 | 0.027 ± 0.003 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.015 ± 0.000 | 0.025 ± 0.002 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.015 ± 0.000 | 0.026 ± 0.004 | 0.020 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.007 ± 0.000 | - | 0.017 ± 0.000 | 0.029 ± 0.003 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.015 ± 0.000 | 0.026 ± 0.004 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.015 ± 0.000 | 0.026 ± 0.002 | 0.018 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.008 ± 0.000 | - | 0.017 ± 0.001 | 0.030 ± 0.004 | 0.030 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.927 ± 0.003 | - | 1.400 ± 0.015 | 0.578 ± 0.032 | 0.775 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.016 ± 0.001 | 0.029 ± 0.003 | 0.023 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.071 ± 0.002 | - | 0.130 ± 0.003 | 0.073 ± 0.007 | 0.160 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.017 ± 0.002 | 0.029 ± 0.003 | 0.028 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.010 ± 0.000 | - | 0.022 ± 0.002 | 0.035 ± 0.003 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.753 ± 0.027 | - | 1.770 ± 0.010 | 1.522 ± 0.004 | 2.069 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 5.941 ± 0.007 | - | 6.001 ± 0.035 | 5.118 ± 0.012 | 5.143 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 17.690 ± 0.165 | - | 17.808 ± 0.139 | 16.069 ± 0.128 | 14.896 ± 0.125 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 4.920 ± 0.023 | - | 5.007 ± 0.090 | 5.611 ± 0.054 | 4.225 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.009 ± 0.004 | 0.016 ± 0.007 | 0.028 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.204 ± 0.017 | - | 0.211 ± 0.007 | 0.232 ± 0.029 | 0.242 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.009 ± 0.000 | - | 0.014 ± 0.003 | 0.019 ± 0.001 | 0.033 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.178 ± 0.005 | - | 0.188 ± 0.005 | 0.206 ± 0.021 | 0.231 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.020 ± 0.000 | - | 0.025 ± 0.001 | 0.032 ± 0.002 | 0.042 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.008 ± 0.000 | - | 0.012 ± 0.000 | 0.017 ± 0.001 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.020 ± 0.000 | - | 0.025 ± 0.000 | 0.030 ± 0.001 | 0.042 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.054 ± 0.003 | - | 0.060 ± 0.000 | 0.063 ± 0.001 | 0.077 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.009 ± 0.000 | - | 0.014 ± 0.000 | 0.019 ± 0.001 | 0.034 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.026 ± 0.000 | - | 0.031 ± 0.000 | 0.034 ± 0.002 | 0.048 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.072 ± 0.000 | - | 0.078 ± 0.001 | 0.079 ± 0.001 | 0.091 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 39.356 ± 0.243 | - | 41.388 ± 0.224 | 44.524 ± 0.096 | 33.312 ± 0.171 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.016 ± 0.000 | - | 0.021 ± 0.000 | 0.025 ± 0.001 | 0.042 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.549 ± 0.014 | - | 1.569 ± 0.012 | 1.648 ± 0.009 | 1.565 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.049 ± 0.000 | - | 0.054 ± 0.010 | 0.058 ± 0.005 | 0.070 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.140 ± 0.001 | - | 0.146 ± 0.004 | 0.141 ± 0.004 | 0.196 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 2.943 ± 0.011 | - | 2.954 ± 0.018 | 1.478 ± 0.012 | 2.044 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 9.679 ± 0.033 | - | 5.954 ± 0.027 | 5.037 ± 0.017 | 2.801 ± 0.472 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 17.809 ± 0.148 | - | 17.951 ± 0.096 | 18.704 ± 0.065 | 7.592 ± 0.103 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 4.707 ± 0.048 | - | 4.679 ± 0.028 | 4.758 ± 0.032 | 5.112 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.009 ± 0.001 | - | 0.016 ± 0.003 | 0.018 ± 0.005 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.314 ± 0.024 | - | 0.342 ± 0.031 | 0.339 ± 0.041 | 0.265 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.016 ± 0.000 | - | 0.025 ± 0.004 | 0.019 ± 0.001 | 0.031 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.275 ± 0.029 | - | 0.326 ± 1.076 | 0.292 ± 0.020 | 0.244 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.022 ± 0.000 | - | 0.047 ± 0.008 | 0.032 ± 0.001 | 0.045 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.015 ± 0.002 | - | 0.022 ± 0.001 | 0.024 ± 0.003 | 0.032 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.034 ± 0.000 | - | 0.045 ± 0.004 | 0.030 ± 0.001 | 0.043 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.055 ± 0.000 | - | 0.100 ± 0.006 | 0.064 ± 0.003 | 0.090 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.018 ± 0.000 | - | 0.024 ± 0.004 | 0.026 ± 0.005 | 0.034 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.044 ± 0.001 | - | 0.053 ± 0.003 | 0.035 ± 0.003 | 0.049 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.073 ± 0.001 | - | 0.129 ± 0.004 | 0.080 ± 0.005 | 0.089 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 37.551 ± 0.456 | - | 46.864 ± 0.653 | 37.356 ± 0.300 | 98.978 ± 43.994 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.029 ± 0.001 | - | 0.037 ± 0.003 | 0.036 ± 0.005 | 0.042 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 2.377 ± 0.021 | - | 2.414 ± 0.022 | 2.363 ± 0.016 | 0.899 ± 0.104 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.083 ± 0.004 | - | 0.092 ± 0.006 | 0.055 ± 0.002 | 0.071 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.141 ± 0.002 | - | 0.235 ± 0.005 | 0.140 ± 0.006 | 0.168 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 1.295 ± 0.008 | - | 1.310 ± 0.020 | 1.063 ± 0.012 | 1.292 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 5.082 ± 0.013 | - | 5.147 ± 0.087 | 4.324 ± 0.029 | 4.014 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 15.123 ± 0.025 | - | 15.142 ± 0.008 | 13.960 ± 0.114 | 12.276 ± 0.159 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 2.288 ± 0.028 | - | 2.321 ± 0.016 | 2.489 ± 0.058 | 2.458 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.001 | 0.031 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.137 ± 0.004 | - | 0.143 ± 0.009 | 0.174 ± 0.023 | 0.226 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.007 ± 0.000 | - | 0.012 ± 0.000 | 0.019 ± 0.008 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.107 ± 0.012 | - | 0.115 ± 0.005 | 0.136 ± 0.008 | 0.198 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.017 ± 0.002 | - | 0.022 ± 0.001 | 0.029 ± 0.001 | 0.038 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.006 ± 0.000 | - | 0.011 ± 0.000 | 0.017 ± 0.001 | 0.029 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.017 ± 0.000 | - | 0.022 ± 0.000 | 0.027 ± 0.002 | 0.038 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.043 ± 0.002 | - | 0.048 ± 0.000 | 0.052 ± 0.001 | 0.062 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.007 ± 0.000 | - | 0.012 ± 0.004 | 0.016 ± 0.001 | 0.029 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.022 ± 0.000 | - | 0.027 ± 0.000 | 0.031 ± 0.001 | 0.042 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.057 ± 0.000 | - | 0.063 ± 0.004 | 0.066 ± 0.004 | 0.073 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 18.497 ± 0.322 | - | 18.581 ± 0.663 | 19.543 ± 0.159 | 18.932 ± 0.072 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.012 ± 0.000 | - | 0.017 ± 0.000 | 0.020 ± 0.001 | 0.036 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 1.059 ± 0.006 | - | 1.076 ± 0.008 | 1.168 ± 0.021 | 1.199 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.042 ± 0.002 | - | 0.047 ± 0.000 | 0.047 ± 0.001 | 0.060 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.114 ± 0.001 | - | 0.119 ± 0.000 | 0.118 ± 0.004 | 0.143 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 2.140 ± 0.008 | - | 2.161 ± 0.014 | 1.026 ± 0.029 | 1.295 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 8.253 ± 0.129 | - | 8.253 ± 3.207 | 4.259 ± 0.022 | 2.103 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 15.126 ± 0.055 | - | 15.228 ± 0.092 | 16.734 ± 0.092 | 6.337 ± 0.060 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 2.655 ± 0.039 | - | 2.678 ± 0.060 | 2.844 ± 0.052 | 3.167 ± 0.070 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.008 ± 0.000 | - | 0.014 ± 0.001 | 0.017 ± 0.004 | 0.031 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.214 ± 0.023 | - | 0.237 ± 0.020 | 0.256 ± 0.023 | 0.211 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.015 ± 0.000 | - | 0.020 ± 0.002 | 0.018 ± 0.001 | 0.036 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.170 ± 0.008 | - | 0.189 ± 0.034 | 0.196 ± 0.008 | 0.183 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.018 ± 0.000 | - | 0.037 ± 0.003 | 0.028 ± 0.001 | 0.037 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.013 ± 0.001 | - | 0.018 ± 0.001 | 0.021 ± 0.003 | 0.030 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.031 ± 0.001 | - | 0.036 ± 0.004 | 0.027 ± 0.002 | 0.043 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.044 ± 0.000 | - | 0.079 ± 0.004 | 0.053 ± 0.002 | 0.062 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.015 ± 0.001 | - | 0.020 ± 0.002 | 0.022 ± 0.003 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.040 ± 0.000 | - | 0.046 ± 0.004 | 0.031 ± 0.002 | 0.042 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.058 ± 0.000 | - | 0.102 ± 0.002 | 0.065 ± 0.001 | 0.078 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 20.971 ± 0.365 | - | 30.938 ± 0.537 | 22.207 ± 0.384 | 25.375 ± 5.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.023 ± 0.000 | - | 0.028 ± 0.002 | 0.031 ± 0.007 | 0.041 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 1.638 ± 0.028 | - | 1.658 ± 0.031 | 1.809 ± 0.242 | 0.587 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.072 ± 0.003 | - | 0.080 ± 0.004 | 0.047 ± 0.001 | 0.058 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.115 ± 0.001 | - | 0.193 ± 0.019 | 0.117 ± 0.005 | 0.160 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.458 ± 0.011 | - | unsupported | 0.350 ± 0.017 | 0.750 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 2.238 ± 0.011 | - | unsupported | 1.914 ± 0.005 | 2.677 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 5.506 ± 0.080 | - | unsupported | 5.363 ± 0.092 | 7.147 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.986 ± 0.010 | - | unsupported | 1.143 ± 0.014 | 1.109 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.002 ± 0.001 | - | unsupported | 0.010 ± 0.001 | 0.022 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.047 ± 0.001 | - | unsupported | 0.066 ± 0.007 | 0.083 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | unsupported | 0.012 ± 0.000 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.039 ± 0.000 | - | unsupported | 0.055 ± 0.004 | 0.083 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | unsupported | 0.016 ± 0.001 | 0.023 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.017 ± 0.000 | - | unsupported | 0.027 ± 0.001 | 0.038 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | unsupported | 0.017 ± 0.001 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.023 ± 0.001 | - | unsupported | 0.033 ± 0.006 | 0.044 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 7.660 ± 0.039 | - | unsupported | 9.012 ± 0.133 | 8.567 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | unsupported | 0.012 ± 0.000 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.371 ± 0.011 | - | unsupported | 0.425 ± 0.015 | 0.475 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.018 ± 0.000 | - | unsupported | 0.025 ± 0.000 | 0.046 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.044 ± 0.000 | - | unsupported | 0.051 ± 0.001 | 0.072 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.769 ± 0.001 | - | unsupported | 0.580 ± 0.017 | 0.781 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 3.515 ± 0.020 | - | unsupported | 1.950 ± 0.014 | 1.417 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 8.284 ± 0.019 | - | unsupported | 7.667 ± 0.024 | 3.521 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.240 ± 0.013 | - | unsupported | 1.477 ± 0.019 | 1.450 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.006 ± 0.002 | - | unsupported | 0.013 ± 0.001 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.066 ± 0.007 | - | unsupported | 0.075 ± 0.031 | 0.076 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.007 ± 0.000 | - | unsupported | 0.017 ± 0.008 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.057 ± 0.006 | - | unsupported | 0.082 ± 0.006 | 0.087 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.012 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.028 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.006 ± 0.001 | - | unsupported | 0.015 ± 0.003 | 0.021 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.014 ± 0.000 | - | unsupported | 0.023 ± 0.002 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.027 ± 0.001 | - | unsupported | 0.027 ± 0.002 | 0.046 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.007 ± 0.001 | - | unsupported | 0.015 ± 0.001 | 0.022 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.017 ± 0.000 | - | unsupported | 0.025 ± 0.004 | 0.028 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.036 ± 0.000 | - | unsupported | 0.032 ± 0.001 | 0.057 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 9.805 ± 0.132 | - | unsupported | 11.415 ± 0.027 | 11.835 ± 1.126 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.011 ± 0.001 | - | unsupported | 0.018 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.512 ± 0.016 | - | unsupported | 0.421 ± 0.016 | 0.327 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.031 ± 0.001 | - | unsupported | 0.025 ± 0.001 | 0.039 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.067 ± 0.001 | - | unsupported | 0.053 ± 0.004 | 0.074 ± 0.005 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 1.060 ± 0.039 | - | unsupported | 0.975 ± 0.059 | 3.683 ± 0.020 | 2.563 ± 0.240 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.131 ± 0.002 | - | unsupported | 0.059 ± 0.002 | 0.136 ± 0.015 | 0.139 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.139 ± 0.005 | - | unsupported | 0.084 ± 0.000 | 0.183 ± 0.007 | 0.228 ± 0.008 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.339 ± 0.030 | - | unsupported | 0.245 ± 0.016 | 0.494 ± 0.014 | 0.758 ± 0.034 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 1.065 ± 0.006 | - | unsupported | 0.499 ± 0.017 | 1.155 ± 0.068 | 2.663 ± 0.243 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.136 ± 0.002 | - | unsupported | 0.063 ± 0.005 | 0.114 ± 0.008 | 0.110 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.140 ± 0.002 | - | unsupported | 0.080 ± 0.004 | 0.187 ± 0.006 | 0.180 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.344 ± 0.012 | - | unsupported | 0.158 ± 0.009 | 0.560 ± 0.029 | 0.709 ± 0.066 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.312 ± 0.013 | - | unsupported | 0.380 ± 0.022 | 0.855 ± 0.020 | 0.426 ± 0.051 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.034 ± 0.000 | - | unsupported | 0.045 ± 0.001 | 0.186 ± 0.006 | 0.074 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.059 ± 0.000 | - | unsupported | 0.061 ± 0.001 | 0.227 ± 0.040 | 0.115 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.200 ± 0.001 | - | unsupported | 0.167 ± 0.005 | 0.375 ± 0.019 | 0.209 ± 0.005 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.324 ± 0.015 | - | unsupported | 0.568 ± 0.013 | 0.895 ± 0.060 | 0.420 ± 0.013 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.035 ± 0.000 | - | unsupported | 0.076 ± 0.008 | 0.195 ± 0.006 | 0.075 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.060 ± 0.000 | - | unsupported | 0.102 ± 0.001 | 0.238 ± 0.011 | 0.115 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.202 ± 0.001 | - | unsupported | 0.238 ± 0.018 | 0.391 ± 0.006 | 0.212 ± 0.010 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.214 ± 0.014 | - | 0.299 ± 0.024 | 0.198 ± 0.020 | 0.394 ± 0.039 | 0.266 ± 0.010 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.080 ± 0.000 | - | 0.085 ± 0.000 | 0.073 ± 0.000 | 0.079 ± 0.003 | 0.059 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.091 ± 0.000 | - | 0.103 ± 0.004 | 0.088 ± 0.001 | 0.106 ± 0.004 | 0.086 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.122 ± 0.000 | - | 0.154 ± 0.003 | 0.126 ± 0.008 | 0.176 ± 0.007 | 0.139 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.223 ± 0.026 | - | 0.390 ± 0.013 | 0.258 ± 0.018 | 0.422 ± 0.053 | 0.261 ± 0.010 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.081 ± 0.001 | - | 0.164 ± 0.021 | 0.126 ± 0.006 | 0.079 ± 0.011 | 0.060 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.094 ± 0.004 | - | 0.195 ± 0.020 | 0.142 ± 0.005 | 0.108 ± 0.013 | 0.086 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.123 ± 0.000 | - | 0.234 ± 0.005 | 0.179 ± 0.018 | 0.228 ± 0.053 | 0.140 ± 0.002 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 46.613 ± 0.356 | - | 49.896 ± 0.197 | 64.184 ± 0.598 | 70.374 ± 0.915 | 51.691 ± 6.998 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 21.656 ± 0.398 | - | 34.526 ± 0.387 | 25.557 ± 0.311 | 62.338 ± 3.995 | 38.709 ± 5.625 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 27.711 ± 0.363 | - | 27.718 ± 0.251 | 28.115 ± 0.250 | 22.390 ± 0.521 | 21.185 ± 0.415 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 13.510 ± 0.289 | - | 26.868 ± 9.579 | 11.533 ± 0.337 | 14.278 ± 0.118 | 12.804 ± 3.355 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 7.652 ± 0.126 | - | 7.698 ± 0.087 | 7.986 ± 0.047 | 8.002 ± 0.023 | 8.267 ± 0.228 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 11.401 ± 0.107 | - | 11.564 ± 0.214 | 10.711 ± 0.085 | 11.742 ± 0.172 | 11.649 ± 0.305 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 8.948 ± 0.072 | - | 9.076 ± 0.236 | 8.988 ± 0.105 | 9.704 ± 0.104 | 9.182 ± 0.027 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 13.296 ± 0.043 | - | 13.301 ± 0.119 | 12.149 ± 0.107 | 14.424 ± 0.191 | 13.854 ± 0.073 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 21.528 ± 0.123 | - | 22.317 ± 0.121 | 19.665 ± 0.113 | 38.843 ± 0.368 | 15.020 ± 0.829 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 15.610 ± 0.181 | - | 26.091 ± 3.172 | 15.859 ± 0.263 | 33.774 ± 1.990 | 14.329 ± 2.346 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.514 ± 0.007 | 0.541 ± 0.005 | unsupported | 0.628 ± 0.004 | 0.751 ± 0.045 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 8.299 ± 0.032 | 8.554 ± 0.022 | unsupported | 8.937 ± 0.018 | 9.886 ± 0.119 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.391 ± 0.003 | 0.416 ± 0.001 | unsupported | 0.522 ± 0.003 | 1.321 ± 0.391 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 6.443 ± 0.075 | 17.231 ± 1.411 | unsupported | 5.647 ± 0.067 | 9.865 ± 0.557 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 45.524 ± 0.329 | - | 46.469 ± 0.188 | 44.742 ± 0.447 | 32.109 ± 0.222 | 28.074 ± 0.144 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 35.113 ± 4.624 | - | 38.182 ± 1.992 | 22.364 ± 0.456 | 20.114 ± 0.266 | 17.027 ± 0.575 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 19.048 ± 0.254 | 18.556 ± 0.066 | 9.565 ± 0.107 | 45.861 ± 0.558 | 9.837 ± 0.094 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 27.958 ± 6.082 | 21.975 ± 1.042 | 6.551 ± 0.168 | 44.983 ± 1.198 | 10.825 ± 0.083 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 27.925 ± 0.476 | - | 28.986 ± 0.426 | 34.665 ± 0.354 | 32.669 ± 0.389 | 31.440 ± 11.088 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 13.285 ± 0.500 | - | 24.933 ± 0.745 | 17.553 ± 0.402 | 25.319 ± 1.372 | 29.401 ± 14.379 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 1.841 ± 0.053 | - | 1.639 ± 0.042 | 1.479 ± 0.054 | 4.767 ± 0.248 | 2.850 ± 0.036 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.509 ± 0.051 | - | 0.486 ± 0.121 | 1.478 ± 0.029 | 4.162 ± 0.834 | 2.730 ± 0.016 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 21.713 ± 0.223 | - | 21.453 ± 0.201 | 21.319 ± 0.148 | 20.423 ± 0.153 | 20.684 ± 1.534 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 25.012 ± 3.120 | - | 28.964 ± 2.011 | 17.007 ± 0.093 | 21.907 ± 2.093 | 33.269 ± 9.055 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 21.646 ± 0.104 | - | 21.413 ± 0.458 | 21.214 ± 0.195 | 20.293 ± 0.198 | 19.712 ± 0.263 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 33.971 ± 1.034 | - | 29.021 ± 1.028 | 17.124 ± 0.071 | 20.188 ± 1.453 | 20.238 ± 0.648 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.208 ± 0.006 | 1.257 ± 0.012 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 24.170 ± 0.399 | 24.588 ± 1.121 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.357 ± 0.009 | 1.391 ± 0.009 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 20.582 ± 0.116 | 24.572 ± 0.655 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.122 ± 0.006 | 1.161 ± 0.014 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 22.338 ± 0.170 | 22.441 ± 0.145 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.256 ± 0.013 | 1.307 ± 0.024 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 19.371 ± 0.136 | 23.395 ± 0.220 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 27.630 ± 0.398 | - | 27.915 ± 0.315 | 28.173 ± 0.269 | 22.339 ± 0.303 | 21.247 ± 0.643 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 13.525 ± 0.413 | - | 35.006 ± 0.997 | 11.723 ± 0.306 | 14.207 ± 0.109 | 12.727 ± 0.218 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 65.064 ± 0.921 | 65.591 ± 0.590 | 64.075 ± 0.565 | 63.470 ± 0.293 | 67.804 ± 1.119 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 47.014 ± 0.696 | 47.804 ± 0.809 | 39.380 ± 0.428 | 53.122 ± 0.404 | 57.168 ± 9.867 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 33.754 ± 0.153 | - | 112.461 ± 4.730 | 31.030 ± 0.173 | 134.970 ± 0.837 | 30.136 ± 0.226 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 26.062 ± 0.158 | - | 71.386 ± 15.102 | 24.819 ± 0.461 | 127.468 ± 19.485 | 20.646 ± 0.454 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.139 ± 0.000 | 0.139 ± 0.001 | unsupported | 0.165 ± 0.033 | 0.149 ± 0.009 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.217 ± 0.007 | 2.210 ± 0.003 | unsupported | 2.524 ± 0.008 | 1.069 ± 0.474 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.075 ± 0.001 | 0.077 ± 0.003 | unsupported | 0.078 ± 0.003 | 0.050 ± 0.012 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.902 ± 0.006 | 0.898 ± 0.026 | unsupported | 1.041 ± 0.027 | 0.834 ± 0.350 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.078 ± 0.000 | 0.079 ± 0.000 | unsupported | 0.130 ± 0.013 | 0.266 ± 0.023 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.239 ± 0.003 | 1.241 ± 0.009 | unsupported | 1.832 ± 0.053 | 2.713 ± 0.053 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.038 ± 0.000 | 0.039 ± 0.001 | unsupported | 0.099 ± 0.031 | 0.164 ± 0.005 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.563 ± 0.002 | 0.566 ± 0.020 | unsupported | 0.750 ± 0.017 | 1.125 ± 0.082 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.197 ± 0.002 | 0.197 ± 0.001 | unsupported | 0.289 ± 0.003 | 0.217 ± 0.012 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 3.125 ± 0.003 | 3.126 ± 0.006 | unsupported | 4.506 ± 0.046 | 1.904 ± 0.039 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.078 ± 0.006 | 0.078 ± 0.001 | unsupported | 0.110 ± 0.009 | 0.153 ± 0.004 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 1.185 ± 0.004 | 1.186 ± 0.009 | unsupported | 1.538 ± 0.043 | 1.050 ± 0.021 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.060 ± 0.007 | 0.100 ± 0.002 | unsupported | 0.165 ± 0.002 | 0.188 ± 0.003 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.645 ± 0.023 | 0.701 ± 0.025 | unsupported | 0.905 ± 0.005 | 1.992 ± 0.036 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.102 ± 0.003 | 0.128 ± 0.004 | unsupported | 0.070 ± 0.002 | 0.194 ± 0.002 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.444 ± 0.007 | 0.513 ± 0.051 | unsupported | 0.376 ± 0.019 | 1.559 ± 0.046 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.131 ± 0.003 | 0.159 ± 0.002 | unsupported | 0.139 ± 0.000 | 0.250 ± 0.018 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.666 ± 0.024 | 1.779 ± 0.059 | unsupported | 1.262 ± 0.020 | 2.321 ± 0.066 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.171 ± 0.004 | 0.214 ± 0.001 | unsupported | 0.075 ± 0.028 | 0.216 ± 0.003 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.955 ± 0.011 | 1.005 ± 0.046 | unsupported | 0.524 ± 0.006 | 1.870 ± 0.098 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.294 ± 0.005 | 0.340 ± 0.005 | unsupported | unsupported | 0.274 ± 0.015 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 4.376 ± 0.047 | 4.465 ± 0.056 | unsupported | unsupported | 3.381 ± 0.080 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.295 ± 0.007 | 0.366 ± 0.015 | unsupported | unsupported | 0.252 ± 0.004 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 2.011 ± 0.025 | 2.104 ± 0.038 | unsupported | unsupported | 2.003 ± 0.046 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.011 ± 0.001 | 0.020 ± 0.001 | unsupported | 0.013 ± 0.001 | 0.067 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.106 ± 0.007 | 0.116 ± 0.017 | unsupported | 0.098 ± 0.011 | 0.226 ± 0.005 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.012 ± 0.001 | 0.026 ± 0.010 | unsupported | 0.015 ± 0.000 | 0.039 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.117 ± 0.043 | 0.436 ± 0.324 | unsupported | 0.092 ± 0.007 | 0.169 ± 0.023 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.036 ± 0.007 | 0.053 ± 0.005 | unsupported | 0.109 ± 0.005 | 0.173 ± 0.003 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.351 ± 0.021 | 0.578 ± 0.504 | unsupported | 0.616 ± 0.018 | 1.781 ± 0.051 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.058 ± 0.006 | 0.090 ± 0.002 | unsupported | 0.099 ± 0.002 | 0.163 ± 0.009 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.241 ± 0.007 | 0.287 ± 0.031 | unsupported | 0.320 ± 0.014 | 1.550 ± 0.034 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.054 ± 0.001 | 0.055 ± 0.001 | unsupported | 0.125 ± 0.002 | 0.206 ± 0.004 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.854 ± 0.007 | 0.864 ± 0.004 | unsupported | 1.892 ± 0.025 | 1.820 ± 0.054 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.028 ± 0.000 | 0.028 ± 0.001 | unsupported | 0.095 ± 0.002 | 0.138 ± 0.012 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.390 ± 0.004 | 0.394 ± 0.004 | unsupported | 0.715 ± 0.009 | 0.975 ± 0.023 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.057 ± 0.000 | 0.057 ± 0.001 | unsupported | 0.127 ± 0.002 | 0.206 ± 0.016 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.882 ± 0.004 | 0.895 ± 0.019 | unsupported | 1.914 ± 0.024 | 1.811 ± 0.023 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.028 ± 0.000 | 0.029 ± 0.000 | unsupported | 0.100 ± 0.003 | 0.141 ± 0.007 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.408 ± 0.004 | 0.407 ± 0.009 | unsupported | 0.734 ± 0.014 | 0.987 ± 0.038 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.126 ± 0.006 | 0.162 ± 0.003 | unsupported | 0.144 ± 0.003 | 0.227 ± 0.006 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.679 ± 0.023 | 1.766 ± 0.068 | unsupported | 1.233 ± 0.009 | 2.209 ± 0.069 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.165 ± 0.009 | 0.171 ± 0.027 | unsupported | 0.063 ± 0.002 | 0.211 ± 0.004 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 0.974 ± 0.653 | 1.022 ± 0.059 | unsupported | 0.481 ± 0.003 | 1.441 ± 0.205 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.182 ± 0.001 | 0.183 ± 0.001 | unsupported | 0.282 ± 0.003 | 0.332 ± 0.027 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 2.908 ± 0.013 | 2.907 ± 0.008 | unsupported | 4.380 ± 0.022 | 4.085 ± 0.057 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.074 ± 0.006 | 0.075 ± 0.002 | unsupported | 0.203 ± 0.004 | 0.222 ± 0.003 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 1.129 ± 0.003 | 1.134 ± 0.008 | unsupported | 1.560 ± 0.080 | 2.135 ± 0.018 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.742 ± 0.017 | 1.417 ± 0.014 | unsupported | 0.283 ± 0.003 | 0.223 ± 0.008 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 19.086 ± 0.510 | 29.466 ± 0.256 | unsupported | 6.059 ± 0.111 | 2.373 ± 0.029 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.968 ± 0.051 | 1.842 ± 0.034 | unsupported | 0.190 ± 0.002 | 0.166 ± 0.003 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 10.027 ± 0.279 | 23.186 ± 0.453 | unsupported | 2.266 ± 0.019 | 1.250 ± 0.052 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 28.048 ± 0.430 | - | unsupported | 30.944 ± 0.271 | - | 31.494 ± 0.990 | 35.697 ± 0.289 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 23.856 ± 0.677 | - | unsupported | 23.736 ± 0.845 | - | 33.435 ± 0.374 | 21.817 ± 1.590 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 27.821 ± 2.141 | - | unsupported | 28.237 ± 0.233 | - | 21.522 ± 0.194 | 22.790 ± 0.501 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 17.069 ± 0.894 | - | unsupported | 17.253 ± 0.468 | - | 24.042 ± 0.338 | 16.586 ± 0.350 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 15.797 ± 0.682 | - | unsupported | 24.542 ± 0.242 | - | 14.868 ± 0.157 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 16.958 ± 0.932 | - | unsupported | 16.979 ± 0.483 | - | 15.224 ± 0.170 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 36.510 ± 1.506 | - | unsupported | 30.517 ± 0.950 | - | 31.231 ± 0.145 | 36.612 ± 0.285 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 24.461 ± 0.963 | - | unsupported | 23.504 ± 1.270 | - | 34.834 ± 0.592 | 23.223 ± 0.564 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 50.717 ± 0.283 | - | unsupported | 51.893 ± 0.253 | - | 48.405 ± 0.263 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 25.190 ± 0.434 | - | unsupported | 25.749 ± 0.123 | - | 23.594 ± 0.154 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 50.159 ± 0.462 | - | unsupported | 51.251 ± 0.404 | - | 47.867 ± 0.236 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 25.132 ± 0.312 | - | unsupported | 25.844 ± 0.280 | - | 23.526 ± 0.091 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 35.822 ± 3.416 | - | unsupported | 30.364 ± 0.548 | - | 31.468 ± 0.133 | 35.797 ± 0.283 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 24.172 ± 0.559 | - | unsupported | 23.239 ± 0.473 | - | 35.593 ± 0.952 | 23.860 ± 0.881 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 23.890 ± 2.155 | - | unsupported | 24.636 ± 0.339 | - | 27.392 ± 0.070 | 28.367 ± 0.302 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 17.003 ± 0.189 | - | unsupported | 16.285 ± 0.253 | - | 28.439 ± 1.078 | 16.578 ± 0.619 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.763 ± 0.006 | unsupported | unsupported | 0.912 ± 0.004 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 12.101 ± 0.069 | unsupported | unsupported | 13.213 ± 0.059 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.511 ± 0.004 | unsupported | unsupported | 0.615 ± 0.005 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 6.089 ± 0.066 | unsupported | unsupported | 6.782 ± 0.068 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 29.252 ± 1.810 | - | unsupported | 31.016 ± 0.848 | - | 31.216 ± 0.247 | 35.767 ± 0.266 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 24.230 ± 1.971 | - | unsupported | 23.227 ± 0.450 | - | 33.145 ± 0.280 | 23.661 ± 1.034 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 112.925 ± 4.270 | - | 0.003 ± 0.000 | 120.117 ± 2.341 | 112.104 ± 1.554 | 145.044 ± 9.575 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 44.806 ± 0.438 | - | 0.004 ± 0.000 | 37.090 ± 0.719 | 58.171 ± 0.609 | 145.036 ± 8.099 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 70.648 ± 4.502 | - | 226.196 ± 12.460 | 74.140 ± 3.319 | 73.710 ± 1.614 | 82.575 ± 0.842 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 24.334 ± 0.354 | - | 89.074 ± 0.715 | 24.225 ± 0.424 | 37.465 ± 1.397 | 82.784 ± 5.293 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 227.337 ± 1.162 | - | 228.123 ± 1.677 | 236.701 ± 4.972 | 231.612 ± 1.098 | 261.457 ± 5.573 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 74.616 ± 1.228 | - | 74.571 ± 0.836 | 74.804 ± 1.741 | 115.495 ± 2.368 | 264.969 ± 4.490 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 73.123 ± 2.196 | - | 73.949 ± 2.459 | 75.007 ± 5.188 | 73.027 ± 4.369 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 37.635 ± 0.302 | - | 37.492 ± 0.158 | 25.743 ± 0.266 | 37.754 ± 0.352 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 146.821 ± 8.012 | - | 0.003 ± 0.000 | 137.129 ± 2.705 | 126.324 ± 1.205 | 165.038 ± 4.325 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 43.076 ± 0.808 | - | 0.004 ± 0.000 | 41.919 ± 0.319 | 46.778 ± 8.383 | 164.902 ± 2.133 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 91.159 ± 2.576 | - | 0.003 ± 0.000 | 84.420 ± 0.526 | 86.291 ± 0.493 | 218.537 ± 11.180 | 133.714 ± 2.630 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 30.103 ± 0.885 | - | 0.004 ± 0.000 | 81.391 ± 1.105 | 45.504 ± 1.235 | 217.114 ± 2.531 | 53.227 ± 0.352 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 64.260 ± 2.094 | - | 65.167 ± 0.646 | 70.932 ± 0.456 | 67.855 ± 1.572 | 84.117 ± 3.237 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 22.064 ± 0.354 | - | 22.277 ± 0.380 | 24.549 ± 0.463 | 34.280 ± 1.021 | 87.843 ± 2.958 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 65.194 ± 1.518 | - | 65.813 ± 4.832 | 69.925 ± 0.701 | 66.862 ± 0.891 | 87.577 ± 3.219 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 21.253 ± 0.241 | - | 21.132 ± 0.267 | 23.625 ± 0.436 | 35.365 ± 0.337 | 87.508 ± 3.993 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.001 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.001 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.001 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 8.667026 | 541689.12 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 7.382792 | 461424.50 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 7.971307 | 498206.69 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 6.928046 | 433002.88 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 5.725149 | 357821.81 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 6.568142 | 410508.88 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 20.986537 | 1311658.56 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 21.231967 | 1326997.94 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 7.608233 | 475514.56 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 6.856041 | 428502.56 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 12.014770 | 750923.12 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 10.042796 | 627674.75 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 8.226928 | 514183.00 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 8.654901 | 540931.31 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 158.170546 | 9885659.12 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 142.999409 | 8937463.06 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 132.784863 | 8299053.94 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 136.871957 | 8554497.31 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 21.142800 | 1321425.00 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 8.357430 | 522339.38 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 6.252221 | 390763.81 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 6.651710 | 415731.88 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 157.845647 | 9865352.94 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 90.358398 | 5647399.88 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 103.087197 | 6442949.81 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 275.691604 | 17230725.25 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 19.323242 | 1207702.62 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 20.114127 | 1257132.94 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 386.719934 | 24169995.88 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 393.414783 | 24588423.94 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 21.710497 | 1356906.06 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 22.255399 | 1390962.44 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 329.306716 | 20581669.75 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 393.157564 | 24572347.75 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 17.956997 | 1122312.31 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 18.577502 | 1161093.88 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 357.413167 | 22338322.94 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 359.055329 | 22440958.06 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 20.096087 | 1256005.44 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 20.916236 | 1307264.75 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 309.939140 | 19371196.25 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 374.316973 | 23394810.81 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.504235 | 148503.67 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 10.555028 | 164922.31 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.900043 | 139063.17 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.927414 | 139490.84 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 8.553800 | 1069225.00 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 20.192783 | 2524097.88 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 17.734633 | 2216829.12 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 17.676524 | 2209565.50 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 3.212290 | 50192.03 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.982658 | 77854.03 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.780239 | 74691.23 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.914972 | 76796.44 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.673739 | 834217.38 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.326001 | 1040750.12 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.213936 | 901742.00 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.183098 | 897887.25 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 17.006721 | 265730.02 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.310714 | 129854.91 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.994703 | 78042.23 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.066468 | 79163.56 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 21.704360 | 2713045.00 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.653804 | 1831725.50 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.915100 | 1239387.50 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 9.925891 | 1240736.38 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.517889 | 164342.02 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.345655 | 99150.86 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.458223 | 38409.73 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.514989 | 39296.70 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.001659 | 1125207.38 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.999476 | 749934.50 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.500835 | 562604.38 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.527856 | 565982.00 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.898778 | 217168.41 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 18.523129 | 289423.89 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 12.603943 | 196936.61 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 12.604865 | 196951.02 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 15.231052 | 1903881.50 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 36.045228 | 4505653.50 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 25.002350 | 3125293.75 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 25.005876 | 3125734.50 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.822370 | 153474.53 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 7.030991 | 109859.23 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.003378 | 78177.78 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.003789 | 78184.20 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.401803 | 1050225.38 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.303493 | 1537936.62 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.482682 | 1185335.25 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 9.488964 | 1186120.50 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.057146 | 188392.91 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 10.548986 | 164827.91 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.838995 | 59984.30 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.413376 | 100209.00 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 15.937950 | 1992243.75 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.238177 | 904772.12 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.159145 | 644893.12 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.607206 | 700900.75 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.386292 | 193535.81 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.472581 | 69884.08 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.528309 | 102004.83 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 8.200467 | 128132.30 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 12.471810 | 1558976.25 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.009476 | 376184.50 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.554389 | 444298.62 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.104541 | 513067.62 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 16.008835 | 250138.05 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.880083 | 138751.30 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.394754 | 131168.03 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 10.191887 | 159248.23 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.565209 | 2320651.12 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.095530 | 1261941.25 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.327973 | 1665996.62 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 14.229347 | 1778668.38 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.828850 | 216075.78 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.806869 | 75107.33 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 10.953453 | 171147.70 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 13.702711 | 214104.86 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 14.961684 | 1870210.50 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.190872 | 523859.00 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.642028 | 955253.50 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 8.042960 | 1005370.00 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 17.520366 | 273755.72 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 18.841460 | 294397.81 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 21.751941 | 339874.08 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 27.045674 | 3380709.25 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 35.006608 | 4375826.00 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 35.719326 | 4464915.75 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 16.153874 | 252404.28 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 18.879194 | 294987.41 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 23.437597 | 366212.45 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 16.020100 | 2002512.50 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 16.090830 | 2011353.75 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 16.834886 | 2104360.75 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.305143 | 67267.86 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.830449 | 12975.77 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.699623 | 10931.61 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.311892 | 20498.31 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 1.809943 | 226242.88 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.786647 | 98330.88 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.849797 | 106224.62 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.929336 | 116167.00 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.491016 | 38922.12 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.931407 | 14553.23 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.755497 | 11804.64 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.678720 | 26230.00 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.349683 | 168710.38 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.737123 | 92140.38 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.932560 | 116570.00 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.484508 | 435563.50 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.077374 | 173083.97 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.988842 | 109200.66 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.275661 | 35557.20 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.385464 | 52897.88 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.251200 | 1781400.00 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.924133 | 615516.62 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.809739 | 351217.38 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.622416 | 577802.00 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.436442 | 163069.41 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.341698 | 99089.03 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.700083 | 57813.80 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.774655 | 90228.98 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 12.403783 | 1550472.88 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.557929 | 319741.12 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.927787 | 240973.38 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.292401 | 286550.12 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.158716 | 205604.94 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.977749 | 124652.33 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.480181 | 54377.83 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.491573 | 54555.83 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.561032 | 1820129.00 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 15.139556 | 1892444.50 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.829673 | 853709.12 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.910284 | 863785.50 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.830206 | 137971.97 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.110844 | 95481.94 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.767076 | 27610.56 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.802352 | 28161.75 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.803579 | 975447.38 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.722416 | 715302.00 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.123670 | 390458.75 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.152085 | 394010.62 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.168304 | 205754.75 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.110317 | 126723.70 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.617278 | 56519.97 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.621427 | 56584.80 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.488957 | 1811119.62 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 15.312139 | 1914017.38 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.059444 | 882430.50 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.161687 | 895210.88 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.003792 | 140684.25 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.395008 | 99922.00 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.817770 | 28402.66 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.872464 | 29257.25 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.892667 | 986583.38 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.871907 | 733988.38 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.261749 | 407718.62 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.257673 | 407209.12 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.534925 | 227108.20 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 9.229008 | 144203.25 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.056098 | 125876.53 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 10.386372 | 162287.06 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 17.672922 | 2209115.25 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.862823 | 1232852.88 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.431938 | 1678992.25 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 14.125392 | 1765674.00 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.479674 | 210619.91 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.027255 | 62925.86 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 10.581835 | 165341.17 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 10.952751 | 171136.73 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.525305 | 1440663.12 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.850363 | 481295.38 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.788864 | 973608.00 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 8.178165 | 1022270.62 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 21.260186 | 332190.41 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 18.058075 | 282157.42 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 11.664079 | 182251.23 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 11.739672 | 183432.38 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 32.677789 | 4084723.62 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 35.043166 | 4380395.75 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 23.260036 | 2907504.50 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 23.253654 | 2906706.75 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 14.207992 | 221999.88 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 12.967167 | 202611.98 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.713895 | 73654.61 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.771463 | 74554.11 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 17.082488 | 2135311.00 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.476387 | 1559548.38 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.035524 | 1129440.50 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 9.071221 | 1133902.62 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.256009 | 222750.14 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 18.080978 | 282515.28 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 47.507928 | 742311.38 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 90.705908 | 1417279.81 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.983486 | 2372935.75 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 48.470962 | 6058870.25 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 152.687279 | 19085909.88 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 235.724700 | 29465587.50 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.624996 | 166015.56 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 12.182927 | 190358.23 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 61.967641 | 968244.39 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 117.906391 | 1842287.36 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.002255 | 1250281.88 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 18.130715 | 2266339.38 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 80.215525 | 10026940.62 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 185.491419 | 23186427.38 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 14.598600 | 912412.50 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 12.207622 | 762976.38 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 211.408081 | 13213005.06 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 193.614155 | 12100884.69 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 9.835253 | 614703.31 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 8.175642 | 510977.62 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 108.511782 | 6781986.38 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 97.430139 | 6089383.69 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.023705 | 1481.56 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002925 | 182.81 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.023794 | 1487.12 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002525 | 157.81 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000130 | 8.12 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.023024 | 1439.00 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.006441 | 402.56 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000220 | 13.75 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.022132 | 1383.25 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.006432 | 402.00 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000170 | 10.62 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.022843 | 1427.69 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.005109 | 319.31 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000170 | 10.62 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.025067 | 1566.69 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.005090 | 318.12 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000120 | 7.50 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.022062 | 1378.88 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003486 | 217.88 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000110 | 6.88 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.020588 | 1286.75 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003417 | 213.56 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 21.3x faster than the slowest successful cell (`pytorch-cpu`, 24.544 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 24.544 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 24.544 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 13.8x faster than the slowest successful cell (`pytorch-cpu`, 11.308 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 13.9x faster than the slowest successful cell (`pytorch-cpu`, 11.308 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 16.5x faster than the slowest successful cell (`pytorch-cpu`, 23.680 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 23.680 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 12.5x faster than the slowest successful cell (`pytorch-cpu`, 23.680 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 13.2x faster than the slowest successful cell (`pytorch-cpu`, 10.860 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 13.2x faster than the slowest successful cell (`pytorch-cpu`, 10.860 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `pytorch-cpu` is 21.0x faster than the slowest successful cell (`julia-base`, 8.228 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `tenferro-direct` is 19.5x faster than the slowest successful cell (`julia-base`, 8.228 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `tenferro-trace` is 18.6x faster than the slowest successful cell (`julia-base`, 8.228 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`julia-base`, 8.808 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 14.3x faster than the slowest successful cell (`julia-base`, 8.808 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `jax-cpu` is 12.5x faster than the slowest successful cell (`julia-base`, 9.205 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 17.2x faster than the slowest successful cell (`julia-base`, 9.205 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 22.6x faster than the slowest successful cell (`julia-base`, 9.205 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-trace` is 17.8x faster than the slowest successful cell (`julia-base`, 9.205 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 3520.5x faster than the slowest successful cell (`julia-base`, 9.808 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `pytorch-cpu` is 11.3x faster than the slowest successful cell (`julia-base`, 10.440 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-direct` is 12.1x faster than the slowest successful cell (`julia-base`, 10.440 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 2604.7x faster than the slowest successful cell (`julia-base`, 10.440 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 12.8x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 13.9x faster than the slowest successful cell (`jax-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 12.9x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x8x8`): `tenferro-direct` is 11.1x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`3x2x2`): `tenferro-direct` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.011 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`3x4x4`): `tenferro-direct` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.011 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x2x2`): `tenferro-direct` is 12.3x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x4x4`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 30.2x faster than the slowest successful cell (`jax-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 20.4x faster than the slowest successful cell (`jax-cpu`, 0.055 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=1, shape=`3x2x2`): `tenferro-direct` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 12.3x faster than the slowest successful cell (`jax-cpu`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 12.8x faster than the slowest successful cell (`jax-cpu`, 0.024 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`3x4x4,rhs=1`): `tenferro-direct` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.023 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svdvals` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.022 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 12.4x faster than the slowest successful cell (`tenferro-eager`, 29.466 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 11.1x faster than the slowest successful cell (`tenferro-eager`, 1.842 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 18.5x faster than the slowest successful cell (`tenferro-eager`, 23.186 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 10.2x faster than the slowest successful cell (`tenferro-eager`, 23.186 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 53819.6x faster than the slowest successful cell (`julia-base`, 145.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 36551.3x faster than the slowest successful cell (`julia-base`, 145.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 60787.5x faster than the slowest successful cell (`julia-base`, 165.038 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 40646.2x faster than the slowest successful cell (`julia-base`, 164.902 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 84540.3x faster than the slowest successful cell (`julia-base`, 218.537 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 57483.1x faster than the slowest successful cell (`julia-base`, 217.114 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 179.9x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 98.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 129.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 142.5x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 172.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 183.9x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 6019.0 GB/s at 0.003 ms. The conservative physical lower bound is 0.168 ms, so the cell is 60.2x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 4183.9 GB/s at 0.004 ms. The conservative physical lower bound is 0.042 ms, so the cell is 10.5x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 99605.0 GB/s at 0.003 ms. The conservative physical lower bound is 2.684 ms, so the cell is 996.0x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 67650.1 GB/s at 0.004 ms. The conservative physical lower bound is 0.671 ms, so the cell is 169.1x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 98871.3 GB/s at 0.003 ms. The conservative physical lower bound is 2.684 ms, so the cell is 988.7x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 66166.0 GB/s at 0.004 ms. The conservative physical lower bound is 0.671 ms, so the cell is 165.4x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 51921.8 GB/s at 0.003 ms. The conservative physical lower bound is 1.342 ms, so the cell is 519.2x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 35535.5 GB/s at 0.004 ms. The conservative physical lower bound is 0.336 ms, so the cell is 88.8x faster than the bound.

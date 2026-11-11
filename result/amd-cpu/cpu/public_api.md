# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/amd-cpu/cpu/public_api/20261111_190311/run.yaml`
- Timestamp: `2026-11-11T19:03:11.926765+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/amd-cpu/cpu/public_api/20261111_190311`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

## CPU Information

- Model: `AMD EPYC 9V74 80-Core Processor`
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

- CSV: `data/results/amd-cpu/cpu/public_api/20261111_190311/cpu_public_api_t1_20261111_190311.csv`
- CSV: `data/results/amd-cpu/cpu/public_api/20261111_190311/cpu_public_api_t4_20261111_190311.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 3.533 ± 0.011 | - | 3.620 ± 0.008 | 3.668 ± 0.015 | 3.521 ± 0.007 | 3.613 ± 0.046 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 3.820 ± 0.005 | - | 3.714 ± 0.010 | 1.880 ± 0.012 | 3.365 ± 5.308 | 2.592 ± 2.182 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 104.493 ± 6.498 | - | 104.152 ± 5.631 | 99.677 ± 1.393 | 104.569 ± 6.346 | 113.841 ± 4.613 | 121.695 ± 5.076 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 35.126 ± 2.216 | - | 118.782 ± 1.376 | 35.097 ± 0.333 | 53.970 ± 3.130 | 113.150 ± 2.935 | 53.280 ± 0.898 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 93.369 ± 9.323 | - | 90.697 ± 3.123 | 83.937 ± 0.349 | 79.673 ± 6.303 | 116.140 ± 0.631 | 121.515 ± 13.928 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 33.140 ± 0.694 | - | 104.584 ± 9.999 | 26.226 ± 0.185 | 41.497 ± 0.281 | 99.516 ± 1.571 | 48.749 ± 1.588 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 40.171 ± 0.111 | - | 40.415 ± 0.196 | 39.619 ± 0.064 | 58.457 ± 0.035 | 42.724 ± 0.092 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 40.675 ± 0.106 | - | 40.639 ± 0.132 | 25.183 ± 1.954 | 25.505 ± 0.338 | 21.175 ± 0.059 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 41.050 ± 0.222 | - | unsupported | 39.926 ± 0.130 | 58.664 ± 0.106 | 41.034 ± 0.261 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 41.447 ± 0.081 | - | unsupported | 25.406 ± 0.077 | 25.964 ± 0.866 | 24.407 ± 12.313 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 9.052 ± 0.037 | - | 9.061 ± 0.060 | 9.456 ± 0.014 | 9.662 ± 0.034 | 9.527 ± 0.075 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 11.804 ± 0.049 | - | 11.764 ± 0.046 | 11.680 ± 0.013 | 14.609 ± 0.023 | 14.715 ± 0.039 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 58.854 ± 0.075 | - | 60.449 ± 1.629 | 92.842 ± 1.741 | 84.243 ± 1.015 | 64.634 ± 1.551 | 67.143 ± 5.714 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 25.501 ± 0.084 | - | 71.996 ± 0.007 | 35.907 ± 0.185 | 43.361 ± 1.001 | 63.270 ± 9.780 | 30.686 ± 11.033 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 87.734 ± 0.300 | - | 87.738 ± 0.229 | 123.022 ± 1.703 | 137.799 ± 0.569 | 117.201 ± 0.859 | 116.195 ± 2.131 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 39.893 ± 0.286 | - | 107.996 ± 0.002 | 51.990 ± 0.475 | 58.829 ± 0.857 | 115.134 ± 3.290 | 59.204 ± 18.434 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 60.735 ± 1.338 | - | 60.677 ± 3.121 | 59.484 ± 1.625 | 63.906 ± 0.461 | 64.645 ± 5.784 | 69.212 ± 0.473 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 20.446 ± 0.163 | - | 82.819 ± 5.041 | 20.597 ± 0.753 | 31.143 ± 0.734 | 65.105 ± 0.755 | 30.865 ± 0.937 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.235 ± 0.046 | - | 15.926 ± 0.102 | 18.211 ± 0.115 | 15.736 ± 0.301 | 3.461 ± 0.012 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 0.803 ± 0.267 | - | 59.998 ± 0.001 | 9.466 ± 0.051 | 2.664 ± 0.112 | 3.460 ± 0.022 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 6.955 ± 0.042 | - | 7.102 ± 0.028 | 7.880 ± 0.584 | 6.535 ± 0.055 | 8.077 ± 0.073 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 126.991 ± 8.975 | - | 94.049 ± 11.953 | 4.796 ± 0.014 | 6.336 ± 0.431 | 5.734 ± 0.093 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 4.572 ± 0.118 | - | 4.707 ± 0.158 | 4.747 ± 0.025 | 4.508 ± 0.011 | 4.223 ± 0.029 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 88.950 ± 11.928 | - | 121.068 ± 24.032 | 2.898 ± 0.030 | 3.335 ± 0.044 | 2.990 ± 0.021 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 5.178 ± 0.057 | - | 5.197 ± 0.021 | 6.277 ± 0.058 | 4.361 ± 0.025 | 4.307 ± 0.051 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 9.554 ± 0.042 | - | 9.575 ± 0.021 | 5.033 ± 0.018 | 5.426 ± 0.578 | 5.490 ± 0.040 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 40.207 ± 0.102 | - | 39.996 ± 0.099 | 39.654 ± 0.047 | 58.521 ± 0.405 | 40.714 ± 0.225 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 40.792 ± 0.148 | - | 40.757 ± 5.725 | 25.209 ± 1.097 | 25.581 ± 0.342 | 20.267 ± 0.105 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 42.135 ± 0.191 | - | unsupported | 43.078 ± 0.072 | 44.369 ± 0.391 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 44.677 ± 0.338 | - | unsupported | 21.749 ± 0.046 | 26.374 ± 0.819 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 104.259 ± 3.988 | - | 103.253 ± 5.438 | 100.688 ± 2.660 | 118.826 ± 11.617 | 114.335 ± 10.773 | 152.868 ± 4.006 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 36.808 ± 2.260 | - | 118.465 ± 3.579 | 34.995 ± 0.279 | 50.400 ± 5.791 | 112.748 ± 2.892 | 66.207 ± 2.160 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 117.185 ± 6.253 | - | 115.462 ± 6.641 | 117.062 ± 3.533 | 127.440 ± 12.348 | 121.963 ± 8.162 | 172.494 ± 6.983 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 39.376 ± 0.360 | - | 129.242 ± 4.525 | 39.264 ± 0.443 | 64.794 ± 0.787 | 120.701 ± 8.477 | 76.421 ± 4.756 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 13.746 ± 1.295 | - | 39.503 ± 0.162 | 53.895 ± 0.982 | 42.212 ± 1.130 | 61.551 ± 18.438 | 67.504 ± 11.112 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 7.639 ± 0.112 | - | 95.999 ± 0.014 | 24.352 ± 0.543 | 21.208 ± 0.153 | 63.003 ± 8.390 | 29.268 ± 13.948 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 7.699 ± 0.208 | - | 7.701 ± 0.056 | 30.801 ± 0.685 | 33.039 ± 0.272 | 40.262 ± 10.196 | 45.435 ± 11.736 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 6.612 ± 0.071 | - | 24.000 ± 0.015 | 11.499 ± 0.382 | 17.577 ± 0.556 | 36.818 ± 5.610 | 15.263 ± 19.587 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 16.573 ± 0.052 | - | 15.134 ± 0.085 | 41.351 ± 0.374 | 39.165 ± 0.279 | 19.535 ± 0.280 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 14.856 ± 0.123 | - | 35.997 ± 0.006 | 16.601 ± 0.213 | 21.163 ± 0.232 | 18.519 ± 0.123 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 13.406 ± 0.014 | - | 13.558 ± 0.092 | 49.680 ± 0.949 | 91.351 ± 0.403 | 51.574 ± 12.857 | 56.923 ± 10.334 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 5.892 ± 0.010 | - | 36.001 ± 0.006 | 19.090 ± 0.319 | 47.145 ± 1.024 | 51.545 ± 18.887 | 32.372 ± 14.229 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 118.520 ± 9.672 | - | 117.202 ± 6.481 | 110.144 ± 3.107 | 125.396 ± 6.676 | 127.767 ± 7.922 | 171.349 ± 5.097 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 39.556 ± 0.234 | - | 131.774 ± 2.194 | 39.041 ± 4.338 | 62.136 ± 0.915 | 130.678 ± 4.415 | 73.237 ± 2.537 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 7.963 ± 0.076 | - | 8.075 ± 0.027 | 38.656 ± 0.790 | 32.631 ± 1.332 | 48.633 ± 12.110 | 52.933 ± 10.267 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 3.840 ± 0.018 | - | 23.998 ± 0.004 | 14.070 ± 0.361 | 16.194 ± 1.306 | 54.295 ± 9.538 | 30.710 ± 11.712 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 10.987 ± 0.036 | - | 10.876 ± 0.032 | 21.164 ± 1.009 | 19.142 ± 0.296 | 29.301 ± 1.985 | 36.187 ± 9.381 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 4.624 ± 0.011 | - | 23.998 ± 0.005 | 7.668 ± 0.277 | 9.919 ± 0.745 | 30.696 ± 0.239 | 12.771 ± 10.794 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 13.828 ± 0.023 | - | 13.897 ± 0.026 | 41.420 ± 0.482 | 51.353 ± 0.589 | 60.929 ± 20.567 | 72.260 ± 10.538 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 6.885 ± 0.033 | - | 36.001 ± 0.016 | 15.611 ± 0.424 | 27.512 ± 0.981 | 60.759 ± 13.159 | 28.723 ± 21.350 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 7.005 ± 0.014 | - | 7.001 ± 0.016 | 18.255 ± 1.086 | 31.437 ± 0.175 | 34.407 ± 1.131 | 36.739 ± 8.582 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 3.403 ± 0.012 | - | 23.999 ± 0.012 | 6.985 ± 0.204 | 15.275 ± 0.269 | 36.466 ± 0.471 | 15.718 ± 12.905 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.292 ± 0.014 | 0.310 ± 0.003 | unsupported | unsupported | 0.389 ± 0.010 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.206 ± 0.009 | 112.624 ± 1.312 | unsupported | unsupported | 0.294 ± 0.015 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 118.398 ± 2.444 | - | 117.461 ± 6.372 | 114.952 ± 1.224 | 112.877 ± 0.595 | 125.583 ± 0.930 | 176.092 ± 5.071 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 38.887 ± 0.181 | - | 129.120 ± 2.564 | 38.585 ± 0.686 | 59.518 ± 2.403 | 129.623 ± 1.375 | 75.037 ± 1.388 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 116.828 ± 5.480 | - | 117.345 ± 6.633 | 115.018 ± 2.151 | 113.889 ± 1.192 | 130.880 ± 7.174 | 171.855 ± 3.826 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 38.867 ± 0.399 | - | 129.243 ± 1.224 | 38.761 ± 3.000 | 58.526 ± 0.199 | 130.250 ± 2.666 | 75.680 ± 3.194 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 115.592 ± 1.091 | - | 115.440 ± 5.629 | 110.557 ± 0.991 | 119.049 ± 10.145 | 127.376 ± 3.025 | 169.886 ± 6.067 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 39.718 ± 1.406 | - | 131.586 ± 2.833 | 40.687 ± 1.562 | 62.692 ± 5.375 | 130.588 ± 1.069 | 72.243 ± 6.043 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 105.393 ± 8.199 | - | 104.672 ± 5.595 | 100.146 ± 2.375 | 118.273 ± 0.557 | 115.470 ± 5.682 | 150.592 ± 5.178 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 35.173 ± 2.030 | - | 119.338 ± 0.449 | 34.620 ± 0.363 | 55.163 ± 5.337 | 113.061 ± 4.389 | 66.911 ± 1.523 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 44.713 ± 1.371 | - | 44.692 ± 3.494 | 33.751 ± 0.828 | 57.866 ± 0.527 | 108.958 ± 24.766 | 98.207 ± 9.476 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 20.374 ± 1.012 | - | 60.001 ± 12.006 | 12.266 ± 0.071 | 28.940 ± 0.216 | 108.308 ± 1.149 | 37.245 ± 9.334 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 11.471 ± 0.032 | - | 11.549 ± 0.419 | 7.559 ± 0.025 | 37.540 ± 0.035 | 7.690 ± 0.101 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 6.714 ± 0.047 | - | 24.000 ± 0.013 | 6.568 ± 0.274 | 18.901 ± 0.026 | 7.650 ± 0.057 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.485 ± 0.010 | - | 1.494 ± 0.019 | 21.677 ± 0.177 | 2.092 ± 0.098 | 0.908 ± 0.046 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.687 ± 0.012 | - | 23.999 ± 0.009 | 9.420 ± 0.297 | 1.348 ± 0.280 | 0.886 ± 0.007 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 1.687 ± 0.132 | - | 1.613 ± 0.131 | 4.637 ± 0.008 | 3.414 ± 0.008 | 0.935 ± 0.092 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 0.969 ± 0.120 | - | 23.999 ± 0.006 | 2.355 ± 0.024 | 2.534 ± 0.013 | 0.872 ± 0.039 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 11.093 ± 0.332 | - | 11.106 ± 0.083 | 7.534 ± 0.024 | 37.569 ± 0.067 | 7.715 ± 0.078 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 6.699 ± 0.049 | - | 23.002 ± 5.965 | 6.527 ± 0.023 | 18.899 ± 0.021 | 8.597 ± 1.490 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.425 ± 0.021 | - | 1.422 ± 0.010 | 22.224 ± 0.194 | 2.015 ± 0.010 | 0.907 ± 0.040 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.683 ± 0.020 | - | 23.998 ± 0.008 | 10.230 ± 0.428 | 1.454 ± 0.168 | 0.884 ± 0.030 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 8.241 ± 0.107 | - | 8.132 ± 0.048 | 13.888 ± 0.031 | 13.450 ± 0.027 | 7.378 ± 0.263 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 4.482 ± 0.451 | - | 35.998 ± 0.012 | 6.997 ± 0.030 | 6.809 ± 0.022 | 7.272 ± 0.078 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 7.575 ± 0.234 | - | 7.283 ± 0.054 | 7.299 ± 0.026 | 27.387 ± 0.067 | 7.588 ± 0.565 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 7.135 ± 0.070 | - | 24.000 ± 0.043 | 7.098 ± 0.027 | 13.803 ± 0.030 | 7.333 ± 0.186 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.616 ± 0.025 | - | 0.604 ± 0.072 | 1.977 ± 0.028 | 1.505 ± 0.027 | 0.601 ± 0.045 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.277 ± 0.083 | - | 24.001 ± 0.023 | 0.904 ± 0.025 | 1.514 ± 0.022 | 0.679 ± 0.088 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.241 ± 0.146 | - | 1.193 ± 0.063 | 0.560 ± 0.014 | 1.022 ± 0.032 | 0.862 ± 0.145 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.693 ± 0.135 | - | 23.999 ± 0.002 | 0.236 ± 0.003 | 0.918 ± 0.022 | 0.858 ± 0.123 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 7.510 ± 0.352 | - | 7.558 ± 0.105 | 7.338 ± 0.090 | 27.386 ± 0.131 | 7.434 ± 0.071 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 7.132 ± 1.358 | - | 23.995 ± 0.007 | 6.685 ± 0.139 | 13.820 ± 0.143 | 7.324 ± 0.038 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.618 ± 0.023 | - | 0.619 ± 0.052 | 2.003 ± 0.038 | 1.506 ± 0.016 | 0.595 ± 0.054 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.272 ± 0.104 | - | 24.000 ± 0.009 | 0.901 ± 0.075 | 0.856 ± 0.016 | 0.579 ± 0.020 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.261 ± 0.159 | - | 1.237 ± 0.042 | 0.574 ± 0.023 | 1.021 ± 0.024 | 0.932 ± 0.060 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.627 ± 0.041 | - | 23.997 ± 0.006 | 0.228 ± 0.030 | 0.904 ± 0.011 | 0.740 ± 0.054 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.635 ± 0.004 | 0.616 ± 0.004 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.240 ± 0.007 | 23.999 ± 0.001 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 30.536 ± 0.258 | - | 33.811 ± 0.155 | 49.639 ± 1.046 | 54.237 ± 0.219 | 55.435 ± 5.178 | 69.897 ± 8.084 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 14.572 ± 0.445 | - | 47.997 ± 0.006 | 18.366 ± 0.406 | 31.565 ± 0.469 | 57.947 ± 8.814 | 33.915 ± 8.773 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 112.958 ± 1.570 | - | 113.365 ± 6.525 | 118.922 ± 3.253 | 173.608 ± 7.670 | 222.574 ± 3.005 | 224.516 ± 2.934 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 36.154 ± 0.122 | - | 131.476 ± 0.777 | 36.741 ± 0.567 | 102.230 ± 4.433 | 222.163 ± 0.981 | 91.864 ± 3.142 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 113.649 ± 5.599 | - | 117.065 ± 6.823 | 132.766 ± 3.675 | 138.270 ± 1.615 | 131.244 ± 5.638 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 43.034 ± 2.140 | - | 131.855 ± 12.645 | 44.818 ± 1.701 | 71.263 ± 6.106 | 132.735 ± 2.033 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 108.037 ± 3.374 | - | 109.682 ± 6.646 | 106.120 ± 2.258 | 111.101 ± 0.882 | 117.814 ± 7.511 | 151.210 ± 2.459 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 34.867 ± 2.249 | - | 139.443 ± 14.317 | 37.384 ± 2.928 | 55.809 ± 2.809 | 116.423 ± 11.550 | 66.118 ± 5.713 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 10.803 ± 0.019 | - | 10.993 ± 0.017 | 51.002 ± 0.710 | 91.444 ± 0.428 | 59.063 ± 8.850 | 55.750 ± 10.104 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 4.997 ± 0.013 | - | 23.996 ± 0.006 | 19.420 ± 0.296 | 47.254 ± 0.395 | 51.500 ± 10.508 | 26.537 ± 18.100 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 104.733 ± 1.507 | - | 105.050 ± 7.121 | 152.359 ± 3.167 | 121.039 ± 9.077 | 176.654 ± 4.520 | 175.277 ± 7.564 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 34.697 ± 0.092 | - | 117.293 ± 3.005 | 54.694 ± 0.589 | 60.649 ± 1.971 | 176.304 ± 3.535 | 73.621 ± 5.135 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 110.937 ± 5.461 | - | 115.916 ± 4.855 | 117.811 ± 1.566 | 118.522 ± 12.078 | 120.980 ± 10.660 | 171.409 ± 2.872 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 39.523 ± 0.216 | - | 129.768 ± 3.089 | 39.245 ± 1.223 | 58.844 ± 4.194 | 129.790 ± 2.001 | 72.238 ± 3.541 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 27.050 ± 0.027 | - | 27.400 ± 0.040 | 68.962 ± 0.263 | 29.707 ± 0.257 | 60.321 ± 8.644 | 66.114 ± 25.528 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 11.833 ± 0.049 | - | 48.001 ± 0.011 | 24.014 ± 0.094 | 15.399 ± 0.400 | 62.119 ± 10.384 | 37.638 ± 22.148 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 110.982 ± 6.999 | 111.758 ± 6.722 | unsupported | 134.815 ± 3.194 | 141.937 ± 2.171 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 43.085 ± 5.909 | 132.168 ± 2.841 | unsupported | 44.726 ± 17.106 | 77.027 ± 2.346 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.700 ± 0.041 | - | 0.706 ± 0.025 | 0.590 ± 0.017 | 0.608 ± 0.033 | 7.018 ± 4.788 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.295 ± 0.072 | - | 11.999 ± 0.007 | 0.234 ± 0.040 | 0.641 ± 0.051 | 1.005 ± 5.921 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.662 ± 0.095 | - | 0.676 ± 0.172 | 0.567 ± 0.032 | 0.603 ± 0.045 | 7.803 ± 4.759 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.330 ± 0.095 | - | 11.999 ± 0.006 | 0.236 ± 0.033 | 0.518 ± 0.134 | 6.725 ± 4.306 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 0.982 ± 0.029 | - | unsupported | 0.941 ± 0.095 | 1.035 ± 0.118 | 5.979 ± 6.821 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.585 ± 0.195 | - | unsupported | 0.542 ± 0.182 | 0.728 ± 0.124 | 6.814 ± 3.890 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.270 ± 0.014 | - | 0.330 ± 0.031 | 0.640 ± 0.016 | 0.265 ± 0.014 | 1.027 ± 0.141 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.282 ± 0.016 | - | 47.999 ± 0.008 | 0.323 ± 0.016 | 0.165 ± 0.007 | 1.201 ± 0.261 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.649 ± 0.087 | - | 0.640 ± 0.113 | 0.796 ± 0.055 | 0.850 ± 0.006 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.269 ± 0.160 | - | 12.000 ± 0.011 | 0.364 ± 0.043 | 0.441 ± 0.043 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.658 ± 0.351 | - | 0.693 ± 0.048 | 0.676 ± 0.064 | 0.682 ± 0.019 | 1.180 ± 6.142 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.255 ± 0.113 | - | 11.999 ± 0.010 | 0.293 ± 0.179 | 0.397 ± 0.020 | 6.313 ± 6.309 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.627 ± 0.012 | - | 0.686 ± 0.011 | 0.309 ± 0.016 | 0.299 ± 0.014 | 1.103 ± 0.147 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.827 ± 0.018 | - | 35.986 ± 0.399 | 0.364 ± 0.012 | 0.306 ± 0.019 | 1.150 ± 0.038 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.243 ± 0.013 | - | 0.002 ± 0.000 | 1.247 ± 0.017 | 1.076 ± 0.250 | 8.377 ± 1.540 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.658 ± 0.059 | - | 0.002 ± 0.000 | 0.689 ± 0.175 | 1.086 ± 0.189 | 7.863 ± 4.422 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.043 ± 0.000 | - | 0.045 ± 0.000 | 0.063 ± 0.001 | 0.075 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.073 ± 0.000 | - | 0.076 ± 0.000 | 0.092 ± 0.001 | 0.121 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.193 ± 0.002 | - | 0.195 ± 0.010 | 0.238 ± 0.001 | 0.241 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.066 ± 0.001 | - | 0.069 ± 0.001 | 0.115 ± 0.002 | 0.090 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.547 ± 0.009 | - | 0.545 ± 0.010 | 0.867 ± 0.007 | 0.565 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.020 ± 0.000 | - | 0.021 ± 0.000 | 0.031 ± 0.000 | 0.043 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.043 ± 0.000 | - | 0.045 ± 0.000 | 0.069 ± 0.000 | 0.057 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.074 ± 0.000 | - | 0.075 ± 0.000 | 0.100 ± 0.005 | 0.101 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.189 ± 0.001 | - | 0.191 ± 0.013 | 0.244 ± 0.011 | 0.278 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.068 ± 0.001 | - | 0.070 ± 0.001 | 0.143 ± 0.001 | 0.090 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.013 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.001 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.544 ± 0.021 | - | 0.555 ± 3.755 | 1.101 ± 0.017 | 0.999 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.020 ± 0.000 | - | 0.022 ± 0.000 | 0.046 ± 0.000 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.408 ± 0.010 | - | 0.409 ± 0.012 | 0.533 ± 0.011 | 0.638 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.686 ± 0.006 | - | 1.686 ± 0.009 | 1.692 ± 0.014 | 2.090 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 5.043 ± 0.020 | - | 5.058 ± 0.022 | 5.251 ± 0.015 | 7.199 ± 0.087 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 2.969 ± 0.008 | - | 2.974 ± 0.008 | 2.661 ± 0.010 | 2.116 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.089 ± 0.001 | - | 0.090 ± 0.001 | 0.095 ± 0.002 | 0.114 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.016 ± 0.000 | - | 0.018 ± 0.000 | 0.021 ± 0.001 | 0.061 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.012 ± 0.001 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.021 ± 0.000 | - | 0.023 ± 0.001 | 0.026 ± 0.000 | 0.067 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 23.788 ± 0.098 | - | 23.712 ± 0.050 | 21.244 ± 0.055 | 18.230 ± 0.058 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.689 ± 0.012 | - | 0.692 ± 0.012 | 0.697 ± 0.008 | 0.751 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.019 ± 0.000 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.041 ± 0.000 | - | 0.043 ± 0.000 | 0.047 ± 0.001 | 0.080 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.406 ± 0.029 | - | 0.414 ± 0.018 | 0.796 ± 0.277 | 0.692 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 1.735 ± 0.021 | - | 1.737 ± 0.025 | 1.692 ± 0.028 | 2.035 ± 0.151 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 5.108 ± 0.027 | - | 5.105 ± 0.019 | 7.387 ± 0.008 | 4.956 ± 0.095 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 5.510 ± 0.028 | - | 5.527 ± 0.069 | 1.955 ± 0.017 | 2.909 ± 0.096 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.089 ± 0.001 | - | 0.092 ± 0.014 | 0.139 ± 0.006 | 0.118 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.017 ± 0.000 | - | 0.019 ± 0.000 | 0.021 ± 0.000 | 0.051 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.022 ± 0.000 | - | 0.024 ± 0.001 | 0.026 ± 0.000 | 0.060 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 44.284 ± 0.100 | - | 44.241 ± 0.138 | 15.364 ± 0.080 | 22.984 ± 6.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.014 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.699 ± 0.021 | - | 0.697 ± 3.753 | 1.047 ± 0.007 | 0.989 ± 0.318 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.015 ± 0.000 | - | 0.017 ± 0.001 | 0.020 ± 0.009 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.042 ± 0.000 | - | 0.044 ± 0.000 | 0.047 ± 0.001 | 0.073 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.474 ± 0.020 | - | 0.475 ± 0.017 | 0.334 ± 0.009 | 0.292 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.707 ± 0.015 | - | 1.712 ± 0.007 | 1.483 ± 0.010 | 1.435 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 4.509 ± 0.003 | - | 4.496 ± 0.013 | 4.367 ± 0.019 | 4.986 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.077 ± 0.020 | - | 1.080 ± 0.025 | 1.083 ± 0.008 | 1.136 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.002 ± 0.001 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.057 ± 0.000 | - | 0.058 ± 0.001 | 0.066 ± 0.002 | 0.094 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.001 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.015 ± 0.000 | - | 0.016 ± 0.000 | 0.019 ± 0.000 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.023 ± 0.001 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 8.629 ± 0.227 | - | 8.587 ± 0.167 | 8.503 ± 0.076 | 8.895 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.444 ± 0.007 | - | 0.439 ± 0.009 | 0.486 ± 0.007 | 0.580 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.017 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.037 ± 0.000 | - | 0.039 ± 0.001 | 0.040 ± 0.000 | 0.064 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.452 ± 0.014 | - | 0.457 ± 0.027 | 0.534 ± 0.009 | 0.280 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.708 ± 0.010 | - | 1.714 ± 0.009 | 1.515 ± 0.016 | 0.961 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 4.528 ± 0.027 | - | 4.527 ± 0.016 | 6.238 ± 0.029 | 2.904 ± 0.457 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 2.653 ± 0.015 | - | 2.657 ± 0.016 | 1.024 ± 0.010 | 1.349 ± 0.083 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.128 ± 0.035 | - | 0.122 ± 0.019 | 0.098 ± 0.002 | 0.103 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.001 | 0.010 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.014 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.015 ± 0.000 | - | 0.017 ± 0.000 | 0.018 ± 0.000 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.016 ± 0.002 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.020 ± 0.001 | - | 0.021 ± 0.000 | 0.023 ± 0.000 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 21.065 ± 0.146 | - | 21.102 ± 0.049 | 8.035 ± 0.107 | 10.806 ± 5.181 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.012 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.902 ± 0.011 | - | 0.910 ± 0.009 | 0.726 ± 0.012 | 0.363 ± 0.108 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.015 ± 0.000 | - | 0.017 ± 0.000 | 0.023 ± 0.002 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.037 ± 0.000 | - | 0.039 ± 0.000 | 0.040 ± 0.000 | 0.057 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.181 ± 0.008 | - | 0.195 ± 0.011 | 0.169 ± 0.003 | 0.299 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 0.834 ± 0.005 | - | 0.831 ± 0.003 | 0.727 ± 0.008 | 1.442 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 2.410 ± 0.016 | - | 2.413 ± 0.009 | 2.279 ± 0.013 | 5.241 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.421 ± 0.010 | - | 0.424 ± 0.010 | 0.442 ± 0.007 | 1.116 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.025 ± 0.000 | - | 0.027 ± 0.000 | 0.031 ± 0.000 | 0.090 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.014 ± 0.001 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.001 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.011 ± 0.000 | - | 0.012 ± 0.000 | 0.015 ± 0.000 | 0.044 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 3.371 ± 0.008 | - | 3.392 ± 0.021 | 3.488 ± 0.023 | 8.734 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.187 ± 0.008 | - | 0.198 ± 0.014 | 0.196 ± 0.001 | 0.573 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.020 ± 0.000 | - | 0.022 ± 0.000 | 0.024 ± 0.000 | 0.075 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.182 ± 0.012 | - | 0.183 ± 0.001 | 0.284 ± 0.010 | 0.292 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 0.837 ± 0.015 | - | 0.863 ± 3.733 | 0.736 ± 0.023 | 0.948 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 2.443 ± 0.016 | - | 2.434 ± 0.025 | 2.765 ± 0.009 | 2.908 ± 0.183 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 1.669 ± 0.011 | - | 1.698 ± 0.017 | 0.599 ± 0.007 | 1.310 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.026 ± 0.000 | - | 0.027 ± 0.000 | 0.040 ± 0.003 | 0.097 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.015 ± 0.001 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 13.454 ± 0.027 | - | 13.351 ± 0.046 | 4.568 ± 0.006 | 11.986 ± 1.690 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.188 ± 0.001 | - | 0.191 ± 0.012 | 0.245 ± 0.002 | 0.319 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.016 ± 0.001 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.020 ± 0.000 | - | 0.022 ± 0.000 | 0.024 ± 0.000 | 0.052 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.150 ± 0.002 | - | 0.142 ± 0.002 | 0.147 ± 0.004 | 0.103 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.181 ± 0.008 | - | 0.180 ± 0.009 | 0.203 ± 0.004 | 0.177 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.410 ± 0.010 | - | 0.414 ± 0.011 | 0.415 ± 0.012 | 0.474 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.106 ± 0.005 | - | 0.108 ± 0.004 | 0.135 ± 0.002 | 0.174 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.015 ± 0.000 | 0.044 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.011 ± 0.001 | - | 0.013 ± 0.001 | 0.017 ± 0.001 | 0.034 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.027 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.001 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.830 ± 0.008 | - | 0.831 ± 0.004 | 0.999 ± 0.003 | 0.966 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.058 ± 0.001 | - | 0.060 ± 0.001 | 0.061 ± 0.002 | 0.095 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.040 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.142 ± 0.002 | - | 0.142 ± 0.004 | 0.125 ± 0.001 | 0.082 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.180 ± 0.015 | - | 0.178 ± 0.002 | 0.168 ± 0.002 | 0.172 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.414 ± 0.018 | - | 0.420 ± 0.014 | 0.282 ± 0.011 | 0.476 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.106 ± 0.002 | - | 0.111 ± 0.003 | 0.167 ± 0.011 | 0.143 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.002 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.024 ± 0.000 | 0.036 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.015 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.012 ± 0.000 | - | 0.014 ± 0.001 | 0.027 ± 0.001 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.015 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.018 ± 0.003 | 0.014 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.017 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.017 ± 0.000 | 0.012 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.019 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.889 ± 0.015 | - | 0.859 ± 0.028 | 0.483 ± 0.018 | 0.734 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.019 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.058 ± 0.001 | - | 0.060 ± 0.000 | 0.053 ± 0.003 | 0.102 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.019 ± 0.001 | 0.013 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.021 ± 0.000 | 0.028 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.206 ± 0.006 | - | 0.210 ± 0.008 | 0.170 ± 0.008 | 0.226 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.441 ± 0.008 | - | 0.447 ± 0.007 | 0.415 ± 0.009 | 0.605 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.033 ± 0.002 | - | 1.049 ± 0.003 | 1.042 ± 0.004 | 1.655 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.399 ± 0.011 | - | 0.413 ± 0.182 | 0.446 ± 0.007 | 0.490 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.020 ± 0.000 | 0.040 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.015 ± 0.000 | - | 0.017 ± 0.001 | 0.022 ± 0.001 | 0.041 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.003 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 3.035 ± 0.015 | - | 3.747 ± 0.026 | 3.540 ± 0.036 | 3.787 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.093 ± 0.001 | - | 0.096 ± 0.001 | 0.129 ± 0.002 | 0.153 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.010 ± 0.000 | - | 0.011 ± 0.001 | 0.012 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.208 ± 0.013 | - | 0.206 ± 0.012 | 0.244 ± 0.001 | 0.216 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.444 ± 0.018 | - | 0.445 ± 0.012 | 0.575 ± 0.011 | 0.598 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.026 ± 0.020 | - | 1.032 ± 0.015 | 1.524 ± 0.019 | 0.986 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.814 ± 0.021 | - | 0.815 ± 0.012 | 0.418 ± 0.011 | 0.997 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.032 ± 0.001 | 0.034 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.016 ± 0.000 | - | 0.018 ± 0.001 | 0.035 ± 0.000 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 6.429 ± 0.036 | - | 7.228 ± 0.025 | 3.215 ± 0.062 | 7.994 ± 1.921 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.095 ± 0.001 | - | 0.096 ± 0.001 | 0.209 ± 0.009 | 0.240 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.019 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.137 ± 0.001 | - | 0.126 ± 0.001 | 0.075 ± 0.001 | 0.183 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.160 ± 0.003 | - | 0.131 ± 0.004 | 0.116 ± 0.001 | 0.265 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.421 ± 0.011 | - | 0.387 ± 0.035 | 0.335 ± 0.010 | 0.582 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.094 ± 0.003 | - | 0.103 ± 0.004 | 0.113 ± 0.001 | 0.133 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.008 ± 0.001 | - | 0.013 ± 0.000 | 0.018 ± 0.000 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.012 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.727 ± 0.013 | - | 0.757 ± 0.024 | 0.812 ± 0.007 | 0.840 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.052 ± 0.000 | - | 0.059 ± 0.003 | 0.057 ± 0.001 | 0.079 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.014 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.138 ± 0.001 | - | 0.136 ± 0.016 | 0.098 ± 0.001 | 0.182 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.165 ± 0.013 | - | 6.001 ± 0.021 | 0.134 ± 0.002 | 0.265 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.422 ± 0.021 | - | 24.000 ± 0.008 | 0.306 ± 0.012 | 0.639 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.092 ± 0.002 | - | 6.003 ± 0.012 | 0.149 ± 0.004 | 0.126 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.009 ± 0.000 | - | 0.014 ± 0.000 | 0.026 ± 0.002 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.018 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.018 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.734 ± 0.016 | - | 23.999 ± 0.028 | 0.443 ± 0.012 | 0.603 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.019 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.053 ± 0.012 | - | 0.061 ± 0.000 | 0.048 ± 0.001 | 0.101 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.020 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.011 ± 0.000 | 0.022 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.406 ± 0.009 | - | 1.387 ± 0.010 | 1.174 ± 0.007 | 1.604 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 4.554 ± 0.011 | - | 4.556 ± 0.005 | 3.951 ± 0.010 | 4.057 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 13.979 ± 0.035 | - | 13.509 ± 0.089 | 11.926 ± 0.012 | 11.826 ± 0.166 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 4.055 ± 0.025 | - | 4.090 ± 0.031 | 4.680 ± 0.044 | 3.476 ± 0.096 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.158 ± 0.005 | - | 0.158 ± 0.006 | 0.179 ± 0.010 | 0.327 ± 0.243 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.141 ± 0.004 | - | 0.144 ± 0.003 | 0.162 ± 0.002 | 0.188 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.016 ± 0.000 | - | 0.018 ± 0.001 | 0.020 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.015 ± 0.000 | - | 0.017 ± 0.000 | 0.018 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.043 ± 0.000 | - | 0.045 ± 0.001 | 0.044 ± 0.001 | 0.048 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.020 ± 0.000 | - | 0.022 ± 0.000 | 0.022 ± 0.000 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.057 ± 0.000 | - | 0.059 ± 0.001 | 0.056 ± 0.000 | 0.060 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 32.287 ± 0.042 | - | 33.511 ± 1.401 | 36.832 ± 0.127 | 27.338 ± 0.149 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.013 ± 0.000 | - | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.238 ± 0.014 | - | 1.227 ± 0.013 | 1.332 ± 0.013 | 1.714 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.038 ± 0.000 | - | 0.040 ± 0.000 | 0.039 ± 0.000 | 0.044 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.113 ± 0.004 | - | 0.111 ± 0.002 | 0.102 ± 0.001 | 0.118 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.422 ± 0.030 | - | 1.433 ± 0.027 | 1.203 ± 0.023 | 1.618 ± 0.337 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 4.644 ± 0.011 | - | 4.627 ± 0.031 | 3.933 ± 0.022 | 1.983 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 13.722 ± 0.115 | - | 13.577 ± 0.068 | 14.408 ± 0.072 | 6.012 ± 0.282 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 7.466 ± 0.050 | - | 7.497 ± 0.049 | 4.070 ± 0.012 | 5.992 ± 1.576 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.235 ± 0.014 | - | 0.245 ± 0.013 | 0.270 ± 0.012 | 0.192 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.178 ± 0.007 | - | 0.181 ± 0.011 | 0.232 ± 0.011 | 0.154 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.017 ± 0.000 | - | 0.019 ± 0.000 | 0.019 ± 0.001 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.014 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.016 ± 0.000 | - | 0.018 ± 0.001 | 0.018 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.044 ± 0.000 | - | 0.045 ± 0.001 | 0.043 ± 0.000 | 0.050 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.015 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.021 ± 0.000 | - | 0.022 ± 0.000 | 0.022 ± 0.000 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.058 ± 0.000 | - | 0.060 ± 0.001 | 0.056 ± 0.001 | 0.061 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 59.660 ± 0.161 | - | 60.655 ± 0.132 | 31.898 ± 0.143 | 55.043 ± 16.946 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.022 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 1.840 ± 0.023 | - | 1.845 ± 0.016 | 2.001 ± 0.009 | 0.676 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.039 ± 0.000 | - | 0.041 ± 0.000 | 0.038 ± 0.000 | 0.045 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.111 ± 0.008 | - | 0.113 ± 0.002 | 0.102 ± 0.001 | 0.126 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 1.030 ± 0.008 | - | 1.030 ± 0.010 | 0.831 ± 0.015 | 0.963 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 3.864 ± 0.011 | - | 3.860 ± 0.007 | 3.384 ± 0.009 | 3.146 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 11.606 ± 0.065 | - | 11.571 ± 0.008 | 10.512 ± 0.011 | 9.923 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.867 ± 0.010 | - | 1.866 ± 0.012 | 2.002 ± 0.034 | 1.827 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.106 ± 0.006 | - | 0.108 ± 0.001 | 0.125 ± 0.003 | 0.134 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.087 ± 0.005 | - | 0.090 ± 0.005 | 0.099 ± 0.003 | 0.136 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.018 ± 0.001 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.017 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.034 ± 0.000 | - | 0.036 ± 0.000 | 0.037 ± 0.001 | 0.039 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.017 ± 0.000 | - | 0.019 ± 0.002 | 0.020 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.045 ± 0.000 | - | 0.048 ± 0.001 | 0.047 ± 0.001 | 0.048 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 14.890 ± 0.036 | - | 14.852 ± 0.027 | 15.803 ± 0.196 | 14.275 ± 0.078 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.013 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 0.822 ± 0.003 | - | 0.827 ± 0.009 | 0.925 ± 0.007 | 0.891 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.033 ± 0.000 | - | 0.035 ± 0.000 | 0.034 ± 0.001 | 0.037 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.089 ± 0.001 | - | 0.091 ± 0.000 | 0.087 ± 0.001 | 0.101 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.996 ± 0.029 | - | 1.003 ± 0.013 | 0.800 ± 0.380 | 0.981 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 3.869 ± 0.027 | - | 3.877 ± 0.024 | 3.325 ± 0.022 | 1.517 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 11.557 ± 0.124 | - | 11.591 ± 0.039 | 12.990 ± 0.036 | 4.895 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 2.801 ± 0.082 | - | 2.816 ± 0.040 | 2.389 ± 0.019 | 2.532 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.182 ± 0.012 | - | 0.184 ± 0.013 | 0.192 ± 0.007 | 0.147 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.007 ± 0.005 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.119 ± 0.008 | - | 0.129 ± 0.006 | 0.146 ± 0.006 | 0.127 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.018 ± 0.001 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.001 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.016 ± 0.001 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.034 ± 0.000 | - | 0.036 ± 0.000 | 0.036 ± 0.000 | 0.039 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.014 ± 0.000 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.018 ± 0.000 | - | 0.020 ± 0.000 | 0.019 ± 0.000 | 0.027 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.046 ± 0.001 | - | 0.048 ± 0.001 | 0.047 ± 0.004 | 0.050 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 22.324 ± 0.100 | - | 22.403 ± 0.104 | 18.541 ± 0.057 | 22.068 ± 6.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.018 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 1.412 ± 0.010 | - | 1.417 ± 0.033 | 1.415 ± 0.017 | 0.458 ± 0.070 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.033 ± 0.000 | - | 0.035 ± 0.000 | 0.033 ± 0.001 | 0.036 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.090 ± 0.001 | - | 0.092 ± 0.002 | 0.086 ± 0.001 | 0.127 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.358 ± 0.009 | - | unsupported | 0.264 ± 0.009 | 0.547 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.661 ± 0.009 | - | unsupported | 1.449 ± 0.006 | 1.972 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 4.123 ± 0.010 | - | unsupported | 3.991 ± 0.006 | 5.206 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.822 ± 0.009 | - | unsupported | 0.924 ± 0.008 | 0.919 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.039 ± 0.004 | - | unsupported | 0.048 ± 0.001 | 0.054 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.031 ± 0.000 | - | unsupported | 0.040 ± 0.001 | 0.047 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | unsupported | 0.010 ± 0.001 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | unsupported | 0.010 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.013 ± 0.000 | - | unsupported | 0.018 ± 0.001 | 0.032 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.007 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.017 ± 0.000 | - | unsupported | 0.022 ± 0.000 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 6.402 ± 0.018 | - | unsupported | 7.315 ± 0.075 | 7.334 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.012 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.297 ± 0.009 | - | unsupported | 0.330 ± 0.011 | 0.378 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.014 ± 0.000 | - | unsupported | 0.017 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.033 ± 0.000 | - | unsupported | 0.038 ± 0.000 | 0.048 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.360 ± 0.010 | - | unsupported | 0.390 ± 0.006 | 0.837 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 1.667 ± 0.041 | - | unsupported | 1.447 ± 0.013 | 1.024 ± 0.100 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 4.195 ± 0.028 | - | unsupported | 5.636 ± 0.025 | 2.659 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 0.987 ± 0.015 | - | unsupported | 1.193 ± 0.013 | 1.385 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.039 ± 0.001 | - | unsupported | 0.064 ± 0.004 | 0.055 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.032 ± 0.000 | - | unsupported | 0.056 ± 0.002 | 0.061 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.006 ± 0.000 | - | unsupported | 0.010 ± 0.001 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.007 ± 0.000 | - | unsupported | 0.014 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.014 ± 0.005 | - | unsupported | 0.018 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.008 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.018 ± 0.000 | - | unsupported | 0.022 ± 0.000 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 7.749 ± 0.033 | - | unsupported | 9.318 ± 0.078 | 11.254 ± 0.077 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.005 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.299 ± 0.008 | - | unsupported | 0.337 ± 0.018 | 0.240 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.014 ± 0.001 | - | unsupported | 0.025 ± 0.001 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.034 ± 0.000 | - | unsupported | 0.038 ± 0.001 | 0.048 ± 0.001 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.820 ± 0.003 | - | unsupported | 0.780 ± 0.011 | 2.709 ± 0.051 | 2.069 ± 1.151 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.099 ± 0.001 | - | unsupported | 0.045 ± 0.001 | 0.088 ± 0.003 | 0.119 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.127 ± 0.001 | - | unsupported | 0.068 ± 0.000 | 0.147 ± 0.009 | 0.149 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.274 ± 0.013 | - | unsupported | 0.201 ± 0.002 | 0.377 ± 0.019 | 0.584 ± 0.212 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.834 ± 0.005 | - | unsupported | 0.389 ± 0.011 | 0.822 ± 0.021 | 2.036 ± 1.038 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.104 ± 0.001 | - | unsupported | 0.047 ± 0.003 | 0.081 ± 0.003 | 0.088 ± 0.005 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.142 ± 0.003 | - | unsupported | 0.060 ± 0.033 | 0.142 ± 0.003 | 0.145 ± 0.005 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.281 ± 0.012 | - | unsupported | 0.118 ± 0.001 | 0.416 ± 0.023 | 0.594 ± 0.078 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.246 ± 0.008 | - | unsupported | 0.300 ± 0.008 | 0.628 ± 0.017 | 0.294 ± 0.008 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.028 ± 0.000 | - | unsupported | 0.033 ± 0.000 | 0.140 ± 0.005 | 0.061 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.050 ± 0.000 | - | unsupported | 0.050 ± 0.001 | 0.173 ± 0.006 | 0.091 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.163 ± 0.004 | - | unsupported | 0.138 ± 0.001 | 0.285 ± 0.005 | 0.152 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.247 ± 0.010 | - | unsupported | 0.430 ± 0.008 | 0.713 ± 0.074 | 0.294 ± 0.008 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.029 ± 0.000 | - | unsupported | 0.056 ± 0.002 | 0.136 ± 0.004 | 0.061 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.051 ± 0.000 | - | unsupported | 0.076 ± 0.002 | 0.159 ± 0.013 | 0.090 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.165 ± 0.001 | - | unsupported | 0.179 ± 0.003 | 0.275 ± 0.013 | 0.151 ± 0.003 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.157 ± 0.002 | - | 0.159 ± 0.002 | 0.161 ± 0.001 | 0.291 ± 0.013 | 0.173 ± 0.010 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.064 ± 0.000 | - | 0.064 ± 0.001 | 0.055 ± 0.001 | 0.056 ± 0.002 | 0.049 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.073 ± 0.000 | - | 0.076 ± 0.000 | 0.067 ± 0.001 | 0.069 ± 0.001 | 0.067 ± 0.007 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.098 ± 0.001 | - | 0.101 ± 0.001 | 0.095 ± 0.002 | 0.123 ± 0.024 | 0.105 ± 0.006 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.155 ± 0.002 | - | 0.160 ± 0.001 | 0.207 ± 0.001 | 0.341 ± 0.056 | 0.168 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.072 ± 0.000 | - | 0.064 ± 0.000 | 0.096 ± 0.002 | 0.058 ± 0.001 | 0.047 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.073 ± 0.000 | - | 0.076 ± 0.000 | 0.108 ± 0.001 | 0.071 ± 0.002 | 0.063 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.099 ± 0.001 | - | 0.102 ± 0.001 | 0.140 ± 0.012 | 0.191 ± 0.051 | 0.102 ± 0.001 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 36.278 ± 0.148 | - | 38.486 ± 0.202 | 52.808 ± 0.441 | 50.849 ± 0.567 | 49.961 ± 6.213 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 38.468 ± 0.133 | - | 37.870 ± 0.185 | 19.644 ± 0.107 | 49.982 ± 5.010 | 42.316 ± 5.661 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 21.600 ± 0.704 | - | 20.139 ± 0.075 | 21.034 ± 0.029 | 19.999 ± 0.084 | 18.237 ± 0.357 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 236.887 ± 15.104 | - | 315.001 ± 34.017 | 9.517 ± 0.242 | 12.098 ± 0.075 | 10.815 ± 3.340 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 6.124 ± 0.011 | - | 6.140 ± 0.031 | 6.475 ± 0.007 | 6.004 ± 0.032 | 5.934 ± 0.062 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 13.759 ± 0.022 | - | 13.796 ± 0.041 | 9.177 ± 0.009 | 8.927 ± 0.148 | 8.719 ± 0.347 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 7.053 ± 0.030 | - | 7.097 ± 0.056 | 7.168 ± 0.108 | 7.212 ± 0.090 | 6.978 ± 0.038 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 10.568 ± 0.018 | - | 10.623 ± 0.091 | 9.742 ± 0.038 | 10.831 ± 0.055 | 10.435 ± 0.030 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 17.216 ± 0.094 | - | 17.238 ± 0.075 | 15.706 ± 0.025 | 31.957 ± 0.085 | 10.812 ± 0.421 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 31.068 ± 0.230 | - | 31.359 ± 0.060 | 12.407 ± 6.145 | 26.692 ± 4.105 | 9.298 ± 0.283 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.405 ± 0.003 | 0.421 ± 0.003 | unsupported | 0.496 ± 0.006 | 0.588 ± 0.017 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 6.428 ± 0.018 | 7.193 ± 0.091 | unsupported | 7.099 ± 0.041 | 7.517 ± 0.088 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.639 ± 0.006 | 28.820 ± 0.866 | unsupported | 0.408 ± 0.000 | 1.385 ± 0.454 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 2265.068 ± 5.042 | 2295.437 ± 2.131 | unsupported | 4.611 ± 0.060 | 8.806 ± 0.277 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 34.636 ± 0.406 | - | 34.719 ± 0.380 | 36.242 ± 0.270 | 34.731 ± 0.202 | 24.152 ± 1.864 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 210.819 ± 12.281 | - | 263.302 ± 10.352 | 18.177 ± 0.291 | 19.987 ± 0.520 | 16.095 ± 1.850 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 14.697 ± 0.022 | 14.493 ± 0.023 | 7.436 ± 0.044 | 45.879 ± 0.812 | 6.214 ± 0.074 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 2277.929 ± 14.930 | 2280.001 ± 6.993 | 5.121 ± 0.057 | 44.307 ± 2.010 | 5.154 ± 0.159 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 20.987 ± 0.247 | - | 21.336 ± 0.285 | 27.128 ± 0.230 | 27.259 ± 0.143 | 28.210 ± 7.367 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 237.166 ± 17.725 | - | 234.357 ± 12.444 | 13.405 ± 0.098 | 19.318 ± 1.006 | 33.078 ± 17.395 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.653 ± 0.040 | - | 0.608 ± 0.026 | 0.937 ± 0.018 | 18.279 ± 0.485 | 2.304 ± 0.009 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.296 ± 0.032 | - | 23.999 ± 0.012 | 0.934 ± 0.019 | 2.946 ± 0.110 | 2.301 ± 0.019 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 16.660 ± 0.111 | - | 16.615 ± 0.040 | 16.680 ± 0.057 | 16.198 ± 0.074 | 17.497 ± 5.117 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 192.315 ± 11.800 | - | 204.583 ± 16.174 | 14.005 ± 0.137 | 17.268 ± 0.713 | 16.924 ± 0.224 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 16.671 ± 0.085 | - | 16.638 ± 0.082 | 16.735 ± 0.088 | 16.253 ± 0.079 | 15.823 ± 0.231 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 203.743 ± 20.053 | - | 193.956 ± 94.700 | 13.979 ± 0.025 | 17.245 ± 2.193 | 17.057 ± 3.684 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 0.974 ± 0.008 | 0.977 ± 0.003 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 17.388 ± 0.170 | 17.373 ± 0.183 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.316 ± 0.012 | 1.326 ± 0.008 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 18.329 ± 0.233 | 18.313 ± 0.238 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 0.908 ± 0.003 | 0.922 ± 0.009 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 16.567 ± 0.038 | 16.566 ± 0.172 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.249 ± 0.006 | 1.255 ± 0.010 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 17.567 ± 0.029 | 17.450 ± 0.047 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 21.326 ± 0.497 | - | 20.359 ± 0.270 | 21.212 ± 0.136 | 19.944 ± 0.042 | 18.265 ± 1.481 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 234.053 ± 3.091 | - | 326.131 ± 6.976 | 9.534 ± 0.229 | 12.261 ± 2.912 | 10.827 ± 0.123 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 48.600 ± 0.192 | 47.749 ± 0.274 | 50.171 ± 0.363 | 49.640 ± 0.311 | 48.787 ± 2.408 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 2313.244 ± 5.735 | 2318.928 ± 9.295 | 32.316 ± 0.146 | 42.738 ± 0.794 | 44.369 ± 12.139 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 29.998 ± 0.166 | - | 30.034 ± 0.104 | 25.470 ± 0.303 | 124.993 ± 6.096 | 25.617 ± 0.272 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 46.527 ± 0.347 | - | 46.515 ± 0.692 | 19.952 ± 8.050 | 112.593 ± 16.083 | 18.242 ± 0.075 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.078 ± 0.001 | 0.079 ± 0.000 | unsupported | 0.086 ± 0.008 | 0.121 ± 0.008 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 1.240 ± 0.004 | 1.246 ± 0.003 | unsupported | 2.018 ± 0.025 | 0.334 ± 0.521 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.039 ± 0.000 | 24.000 ± 0.219 | unsupported | 0.065 ± 0.049 | 0.042 ± 0.005 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.564 ± 0.003 | 23.999 ± 0.001 | unsupported | 0.852 ± 0.009 | 0.569 ± 0.279 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.038 ± 0.000 | 0.038 ± 0.000 | unsupported | 0.036 ± 0.000 | 0.189 ± 0.002 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 0.590 ± 0.002 | 0.615 ± 0.003 | unsupported | 1.297 ± 0.016 | 2.025 ± 0.088 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.019 ± 0.000 | 12.609 ± 1.500 | unsupported | 0.066 ± 0.021 | 0.113 ± 0.003 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.280 ± 0.001 | 11.999 ± 0.001 | unsupported | 0.490 ± 0.003 | 0.856 ± 0.012 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.105 ± 0.000 | 0.119 ± 0.011 | unsupported | 0.111 ± 0.000 | 0.153 ± 0.013 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 1.671 ± 0.004 | 1.686 ± 0.006 | unsupported | 2.507 ± 0.012 | 1.499 ± 0.042 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.050 ± 0.000 | 12.000 ± 0.016 | unsupported | 0.071 ± 0.003 | 0.096 ± 0.001 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 0.761 ± 0.002 | 11.999 ± 0.001 | unsupported | 0.977 ± 0.009 | 1.032 ± 0.163 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.049 ± 0.003 | 0.069 ± 0.009 | unsupported | 0.082 ± 0.000 | 0.155 ± 0.009 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.598 ± 0.011 | 0.622 ± 0.025 | unsupported | 0.684 ± 0.004 | 1.604 ± 0.035 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.073 ± 0.001 | 65.015 ± 0.465 | unsupported | 0.054 ± 0.001 | 0.140 ± 0.002 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.344 ± 0.004 | 82.958 ± 87.664 | unsupported | 0.293 ± 0.002 | 0.970 ± 0.043 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.070 ± 0.019 | 0.080 ± 0.001 | unsupported | 0.042 ± 0.000 | 0.201 ± 0.005 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 0.861 ± 0.014 | 0.909 ± 0.068 | unsupported | 0.752 ± 0.006 | 2.479 ± 0.082 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.090 ± 0.008 | 88.844 ± 0.434 | unsupported | 0.043 ± 0.002 | 0.185 ± 0.004 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.475 ± 0.021 | 119.999 ± 0.253 | unsupported | 0.303 ± 0.005 | 1.430 ± 0.149 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.219 ± 0.005 | 0.242 ± 0.009 | unsupported | unsupported | 0.219 ± 0.007 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 3.222 ± 0.015 | 3.293 ± 0.014 | unsupported | unsupported | 3.204 ± 0.061 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.168 ± 0.008 | 112.763 ± 0.442 | unsupported | unsupported | 0.219 ± 0.005 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 1.347 ± 0.018 | 144.000 ± 0.885 | unsupported | unsupported | 1.441 ± 0.695 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.012 ± 0.001 | 0.021 ± 0.001 | unsupported | 0.010 ± 0.000 | 0.036 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.087 ± 0.009 | 0.105 ± 0.004 | unsupported | 0.060 ± 0.002 | 0.254 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.010 ± 0.000 | 23.985 ± 0.187 | unsupported | 0.011 ± 0.000 | 0.028 ± 0.006 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.048 ± 0.003 | 23.999 ± 0.001 | unsupported | 0.038 ± 0.001 | 0.158 ± 0.001 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.026 ± 0.003 | 0.038 ± 0.006 | unsupported | 0.029 ± 0.001 | 0.147 ± 0.016 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.318 ± 0.012 | 0.329 ± 0.015 | unsupported | 0.504 ± 0.020 | 1.466 ± 0.023 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.037 ± 0.002 | 32.636 ± 0.529 | unsupported | 0.086 ± 0.001 | 0.120 ± 0.012 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.180 ± 0.019 | 47.999 ± 0.001 | unsupported | 0.198 ± 0.007 | 0.869 ± 0.037 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.026 ± 0.000 | 0.026 ± 0.000 | unsupported | 0.028 ± 0.001 | 0.154 ± 0.009 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.401 ± 0.002 | 0.449 ± 0.005 | unsupported | 1.184 ± 0.013 | 1.353 ± 0.014 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.014 ± 0.000 | 12.000 ± 0.000 | unsupported | 0.060 ± 0.001 | 0.102 ± 0.005 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.348 ± 0.037 | 12.000 ± 0.001 | unsupported | 0.448 ± 0.024 | 0.712 ± 0.012 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.027 ± 0.000 | 0.027 ± 0.000 | unsupported | 0.028 ± 0.001 | 0.150 ± 0.009 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.416 ± 0.001 | 0.445 ± 0.003 | unsupported | 1.202 ± 0.180 | 1.398 ± 0.107 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.014 ± 0.001 | 12.000 ± 0.000 | unsupported | 0.061 ± 0.002 | 0.103 ± 0.002 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.200 ± 0.003 | 11.999 ± 0.001 | unsupported | 0.457 ± 0.015 | 0.749 ± 0.020 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.087 ± 0.001 | 0.112 ± 0.002 | unsupported | 0.049 ± 0.000 | 0.195 ± 0.001 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 0.860 ± 0.017 | 0.863 ± 0.004 | unsupported | 0.785 ± 0.009 | 2.379 ± 0.225 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.086 ± 0.001 | 88.924 ± 10.717 | unsupported | 0.043 ± 0.001 | 0.179 ± 0.002 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 0.499 ± 0.015 | 119.999 ± 0.126 | unsupported | 0.282 ± 0.005 | 1.353 ± 0.016 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.092 ± 0.000 | 0.092 ± 0.000 | unsupported | 0.100 ± 0.001 | 0.256 ± 0.006 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 1.455 ± 0.002 | 1.461 ± 0.002 | unsupported | 2.308 ± 0.015 | 3.387 ± 0.032 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.040 ± 0.001 | 12.000 ± 0.328 | unsupported | 0.114 ± 0.003 | 0.153 ± 0.009 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 0.608 ± 0.002 | 11.999 ± 0.001 | unsupported | 0.885 ± 0.011 | 1.732 ± 0.022 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.520 ± 0.002 | 1.083 ± 0.003 | unsupported | 0.178 ± 0.001 | 0.155 ± 0.001 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 12.684 ± 0.060 | 19.274 ± 0.799 | unsupported | 4.686 ± 0.130 | 1.367 ± 0.040 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.408 ± 0.005 | 104.140 ± 10.727 | unsupported | 0.237 ± 0.006 | 0.105 ± 0.007 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 6.547 ± 0.102 | 134.124 ± 3.500 | unsupported | 1.838 ± 0.015 | 0.753 ± 0.028 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 23.063 ± 0.230 | - | unsupported | 20.678 ± 0.144 | - | 22.498 ± 0.323 | 70.229 ± 0.608 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 20.133 ± 0.152 | - | unsupported | 19.768 ± 0.073 | - | 22.394 ± 0.320 | 26.710 ± 0.111 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 16.796 ± 0.230 | - | unsupported | 14.264 ± 0.072 | - | 16.166 ± 0.577 | 15.644 ± 0.637 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 13.680 ± 0.100 | - | unsupported | 13.516 ± 0.062 | - | 15.779 ± 0.167 | 13.522 ± 0.074 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 17.901 ± 0.164 | - | unsupported | 14.503 ± 0.159 | - | 18.338 ± 0.175 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 13.638 ± 0.093 | - | unsupported | 13.567 ± 0.111 | - | 18.359 ± 0.099 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 21.599 ± 0.339 | - | unsupported | 21.743 ± 0.488 | - | 24.504 ± 0.474 | 73.203 ± 0.689 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 20.202 ± 0.144 | - | unsupported | 19.933 ± 0.195 | - | 21.343 ± 0.149 | 26.597 ± 0.050 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 41.376 ± 0.293 | - | unsupported | 43.002 ± 0.101 | - | 39.525 ± 0.151 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 46.736 ± 0.486 | - | unsupported | 21.665 ± 0.403 | - | 19.268 ± 8.364 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 41.220 ± 0.063 | - | unsupported | 42.926 ± 0.283 | - | 39.602 ± 0.329 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 45.339 ± 0.624 | - | unsupported | 21.683 ± 0.282 | - | 19.185 ± 0.185 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 23.147 ± 0.220 | - | unsupported | 20.818 ± 0.257 | - | 22.968 ± 0.410 | 71.345 ± 0.801 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 20.144 ± 0.338 | - | unsupported | 19.742 ± 0.054 | - | 22.565 ± 0.329 | 26.924 ± 0.556 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 16.790 ± 0.293 | - | unsupported | 14.383 ± 0.194 | - | 15.248 ± 2.507 | 55.836 ± 0.278 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 13.829 ± 0.139 | - | unsupported | 16.047 ± 8.207 | - | 14.541 ± 0.116 | 20.700 ± 5.861 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.594 ± 0.003 | unsupported | unsupported | 0.768 ± 0.004 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 9.190 ± 0.024 | unsupported | unsupported | 10.877 ± 0.118 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 1.262 ± 0.002 | unsupported | unsupported | 0.517 ± 0.005 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 182.504 ± 5.749 | unsupported | unsupported | 5.367 ± 0.040 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 22.972 ± 0.289 | - | unsupported | 20.634 ± 0.339 | - | 22.776 ± 0.358 | 70.572 ± 0.980 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 20.072 ± 0.439 | - | unsupported | 19.723 ± 0.099 | - | 22.659 ± 0.328 | 27.507 ± 3.914 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 89.457 ± 4.983 | - | 0.001 ± 0.000 | 86.376 ± 2.567 | 84.783 ± 3.603 | 110.299 ± 5.511 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 37.354 ± 0.400 | - | 0.001 ± 0.000 | 28.926 ± 1.030 | 49.107 ± 3.443 | 109.978 ± 4.982 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 56.266 ± 1.653 | - | 177.346 ± 13.925 | 61.302 ± 6.076 | 65.803 ± 4.403 | 68.863 ± 1.410 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 20.675 ± 1.292 | - | 219.410 ± 7.742 | 20.560 ± 0.994 | 32.908 ± 1.778 | 69.184 ± 1.167 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 175.160 ± 2.595 | - | 176.332 ± 5.954 | 184.301 ± 5.736 | 173.461 ± 0.886 | 195.586 ± 8.814 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 58.273 ± 3.285 | - | 179.399 ± 1.287 | 57.653 ± 0.755 | 89.755 ± 11.184 | 202.076 ± 5.185 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 56.933 ± 3.545 | - | 57.599 ± 8.796 | 63.592 ± 2.062 | 60.850 ± 1.502 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 36.695 ± 0.924 | - | 58.815 ± 4.169 | 21.153 ± 0.324 | 32.331 ± 0.293 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 113.508 ± 5.155 | - | 0.002 ± 0.000 | 97.737 ± 4.592 | 98.363 ± 3.700 | 128.879 ± 6.128 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 34.959 ± 1.252 | - | 0.002 ± 0.000 | 34.752 ± 0.181 | 38.139 ± 6.731 | 128.495 ± 5.252 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 77.879 ± 1.522 | - | 0.001 ± 0.000 | 70.661 ± 0.341 | 66.412 ± 0.331 | 176.947 ± 9.847 | 115.592 ± 6.264 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 26.082 ± 0.458 | - | 0.001 ± 0.000 | 70.335 ± 0.546 | 33.261 ± 0.689 | 177.278 ± 3.558 | 44.254 ± 1.574 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 47.573 ± 2.227 | - | 49.362 ± 0.568 | 58.372 ± 0.414 | 51.809 ± 3.798 | 67.373 ± 4.428 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 19.879 ± 0.093 | - | 61.008 ± 9.082 | 19.912 ± 0.287 | 28.922 ± 0.319 | 65.446 ± 3.094 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 48.983 ± 2.173 | - | 48.806 ± 0.494 | 52.914 ± 2.624 | 49.623 ± 2.502 | 67.662 ± 3.244 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 18.886 ± 0.142 | - | 69.967 ± 12.000 | 18.144 ± 0.411 | 28.854 ± 0.562 | 67.231 ± 1.147 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.001 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 6.230701 | 389418.81 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 4.677854 | 292365.88 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 4.952741 | 309546.31 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 4.697770 | 293610.62 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 3.297470 | 206091.88 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 1801.990564 | 112624410.25 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 10.166267 | 635391.69 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 9.855227 | 615951.69 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 3.844454 | 240278.38 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 383.984484 | 23999030.25 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 9.400411 | 587525.69 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 7.933822 | 495863.88 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 6.479551 | 404971.94 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 6.739747 | 421234.19 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 120.276646 | 7517290.38 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 113.580752 | 7098797.00 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 102.850931 | 6428183.19 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 115.081425 | 7192589.06 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 22.156324 | 1384770.25 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 6.523463 | 407716.44 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 10.224605 | 639037.81 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 461.123464 | 28820216.50 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 140.892867 | 8805804.19 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 73.781264 | 4611329.00 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 36241.090192 | 2265068137.00 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 36726.990021 | 2295436876.31 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 15.586512 | 974157.00 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 15.637847 | 977365.44 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 278.206326 | 17387895.38 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 277.968314 | 17373019.62 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 21.048608 | 1315538.00 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 21.218250 | 1326140.62 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 293.264971 | 18329060.69 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 293.007782 | 18312986.38 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 14.527003 | 907937.69 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 14.744325 | 921520.31 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 265.075521 | 16567220.06 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 265.060629 | 16566289.31 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 19.976009 | 1248500.56 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 20.074736 | 1254671.00 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 281.069487 | 17566842.94 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 279.205689 | 17450355.56 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.739655 | 120932.11 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.478969 | 85608.89 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.003336 | 78177.12 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.035964 | 78686.94 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 2.674949 | 334368.62 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 16.145946 | 2018243.25 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.917008 | 1239626.00 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 9.968254 | 1246031.75 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.698975 | 42171.48 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.180332 | 65317.69 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.498715 | 39042.42 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1535.971050 | 23999547.66 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.552660 | 569082.50 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.812061 | 851507.62 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.514781 | 564347.62 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 191.992339 | 23999042.38 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.076494 | 188695.22 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 2.325341 | 36333.45 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.400543 | 37508.48 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.438850 | 38107.03 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 16.196833 | 2024604.13 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.375245 | 1296905.62 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.719805 | 589975.62 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.923818 | 615477.25 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.207850 | 112622.66 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.215534 | 65867.72 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.247957 | 19499.33 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 806.973134 | 12608955.22 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.846195 | 855774.38 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.923551 | 490443.88 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.242692 | 280336.50 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.993867 | 11999233.38 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.790885 | 152982.58 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.096533 | 110883.33 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.722219 | 105034.67 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 7.632807 | 119262.61 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.989335 | 1498666.88 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 20.057676 | 2507209.50 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.368365 | 1671045.62 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 13.485389 | 1685673.62 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.163705 | 96307.89 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.517903 | 70592.23 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.173886 | 49591.97 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.971640 | 11999556.88 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.255712 | 1031964.00 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.819052 | 977381.50 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.087146 | 760893.25 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.995008 | 11999376.00 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.905180 | 154768.44 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.271522 | 82367.53 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.104004 | 48500.06 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.434894 | 69295.22 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.828166 | 1603520.75 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 5.471097 | 683887.12 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.786875 | 598359.38 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.979850 | 622481.25 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.928938 | 139514.66 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.446325 | 53848.83 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.653348 | 72708.56 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4160.941057 | 65014704.02 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.758698 | 969837.25 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.347217 | 293402.12 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.752967 | 344120.88 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 663.666796 | 82958349.50 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.856979 | 200890.30 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 2.707598 | 42306.22 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.476215 | 69940.86 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.100008 | 79687.62 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 19.829229 | 2478653.62 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 6.016083 | 752010.38 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.885000 | 860625.00 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.268618 | 908577.25 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.826257 | 184785.27 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.773418 | 43334.66 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.777224 | 90269.12 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5686.006549 | 88843852.33 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.437443 | 1429680.38 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.426975 | 303371.88 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.796596 | 474574.50 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 959.990472 | 119998809.00 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.024658 | 219135.28 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 14.019008 | 219047.00 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 15.491027 | 242047.30 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 25.629894 | 3203736.75 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 25.779504 | 3222438.00 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 26.346834 | 3293354.25 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 14.002614 | 218790.84 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 10.782452 | 168475.81 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7216.812369 | 112762693.27 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.526885 | 1440860.62 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 10.772371 | 1346546.38 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1151.996551 | 143999568.88 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.301076 | 35954.31 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.620789 | 9699.83 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.776088 | 12126.38 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.333593 | 20837.39 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 2.030575 | 253821.87 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.481843 | 60230.38 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.698724 | 87340.50 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.837810 | 104726.25 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 1.803905 | 28186.02 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.691234 | 10800.53 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.658435 | 10288.05 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1535.013242 | 23984581.91 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.260947 | 157618.38 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.306173 | 38271.62 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.383639 | 47954.88 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 191.995426 | 23999428.25 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.387684 | 146682.56 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 1.854615 | 28978.36 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.636462 | 25569.72 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.432921 | 38014.39 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.724393 | 1465549.12 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.032288 | 504036.00 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.542563 | 317820.38 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 2.628821 | 328602.62 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.697235 | 120269.30 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.507769 | 86058.89 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.343464 | 36616.62 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2088.679225 | 32635612.89 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.950516 | 868814.50 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.587741 | 198467.62 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.441155 | 180144.38 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 383.995600 | 47999450.00 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.842337 | 153786.52 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 1.783970 | 27874.53 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.647779 | 25746.55 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.676411 | 26193.92 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.826454 | 1353306.75 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.474733 | 1184341.62 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.206887 | 400860.88 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.592878 | 449109.75 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.529226 | 102019.16 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.852296 | 60192.12 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.889526 | 13898.84 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.979202 | 11999675.03 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.695362 | 711920.25 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.586293 | 448286.62 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.786279 | 348284.88 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.996390 | 11999548.75 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.631676 | 150494.94 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 1.822177 | 28471.52 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.701138 | 26580.28 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.722839 | 26919.36 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.187619 | 1398452.38 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.618816 | 1202352.00 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.326573 | 415821.62 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.557765 | 444720.62 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.577578 | 102774.66 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.930602 | 61415.66 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.925420 | 14459.69 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.979093 | 11999673.33 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.995096 | 749387.00 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.652049 | 456506.12 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.599019 | 199877.38 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.995979 | 11999497.38 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.452710 | 194573.59 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.142791 | 49106.11 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.593409 | 87397.02 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 7.142632 | 111603.62 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 19.028243 | 2378530.38 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 6.276610 | 784576.25 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.881955 | 860244.38 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.901655 | 862706.88 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.463951 | 179124.23 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.768370 | 43255.78 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.532397 | 86443.70 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5691.111027 | 88923609.80 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.827199 | 1353399.88 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.258255 | 282281.88 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.989679 | 498709.88 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 959.993275 | 119999159.38 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 16.380894 | 255951.47 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.368366 | 99505.72 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.861185 | 91581.02 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.887434 | 91991.16 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 27.096164 | 3387020.50 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 18.463306 | 2307913.25 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 11.639627 | 1454953.38 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.684573 | 1460571.62 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.809421 | 153272.20 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 7.313434 | 114272.41 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.566137 | 40095.89 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.999766 | 11999996.34 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 13.853899 | 1731737.38 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.077183 | 884647.88 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.860643 | 607580.38 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.994557 | 11999319.62 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.906843 | 154794.42 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 11.415445 | 178366.33 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 33.260496 | 519695.25 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 69.338534 | 1083414.59 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.938249 | 1367281.12 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 37.485920 | 4685740.00 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 101.471976 | 12683997.00 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 154.195549 | 19274443.62 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.690184 | 104534.12 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 15.184042 | 237250.66 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 26.115463 | 408054.11 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6664.969907 | 104140154.80 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.025737 | 753217.12 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 14.704591 | 1838073.88 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 52.374993 | 6546874.12 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1072.993511 | 134124188.87 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 12.294476 | 768404.75 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 9.505900 | 594118.75 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 174.025191 | 10876574.44 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 147.044220 | 9190263.75 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 8.276940 | 517308.75 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 20.197616 | 1262351.00 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 85.869249 | 5366828.06 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 2920.070067 | 182504379.19 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.017716 | 1107.25 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002073 | 129.56 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.018067 | 1129.19 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002133 | 133.31 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000091 | 5.69 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.016474 | 1029.62 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.007851 | 490.69 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000101 | 6.31 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.017196 | 1074.75 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.008012 | 500.75 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000110 | 6.88 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.018047 | 1127.94 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.005799 | 362.44 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000430 | 26.88 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.017356 | 1084.75 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.005899 | 368.69 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000090 | 5.62 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.015783 | 986.44 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003255 | 203.44 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000081 | 5.06 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.015753 | 984.56 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003094 | 193.38 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/norm_fro` (c64, threads=1, shape=`2048x1536`): `tenferro-direct` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 18.211 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `jax-cpu` is 22.5x faster than the slowest successful cell (`tenferro-trace`, 59.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `julia-base` is 17.3x faster than the slowest successful cell (`tenferro-trace`, 59.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 74.7x faster than the slowest successful cell (`tenferro-trace`, 59.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 20.0x faster than the slowest successful cell (`tenferro-direct`, 126.991 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `julia-base` is 22.1x faster than the slowest successful cell (`tenferro-direct`, 126.991 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 26.5x faster than the slowest successful cell (`tenferro-direct`, 126.991 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 36.3x faster than the slowest successful cell (`tenferro-trace`, 121.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `julia-base` is 40.5x faster than the slowest successful cell (`tenferro-trace`, 121.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 41.8x faster than the slowest successful cell (`tenferro-trace`, 121.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/chain_log1p_exp_mul` (f64, threads=4, shape=`4194304`): `tenferro-direct` is 12.6x faster than the slowest successful cell (`tenferro-trace`, 95.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/exp` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 14.1x faster than the slowest successful cell (`julia-base`, 54.295 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log1p` (f64, threads=4, shape=`4194304`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`julia-base`, 36.466 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/masked_log_softmax_axis1` (f64, threads=4, shape=`1024x64`): `jax-cpu` is 383.6x faster than the slowest successful cell (`tenferro-eager`, 112.624 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/masked_log_softmax_axis1` (f64, threads=4, shape=`1024x64`): `tenferro-direct` is 546.5x faster than the slowest successful cell (`tenferro-eager`, 112.624 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 10.4x faster than the slowest successful cell (`pytorch-cpu`, 21.677 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 23.9x faster than the slowest successful cell (`pytorch-cpu`, 21.677 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 14.6x faster than the slowest successful cell (`pytorch-cpu`, 21.677 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 21.677 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 17.8x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 27.1x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 35.0x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 27.5x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 10.2x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 24.8x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 22.224 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 24.5x faster than the slowest successful cell (`pytorch-cpu`, 22.224 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 15.6x faster than the slowest successful cell (`pytorch-cpu`, 22.224 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`pytorch-cpu`, 22.224 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 16.5x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 27.1x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 35.1x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 15.9x faster than the slowest successful cell (`tenferro-trace`, 24.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 35.3x faster than the slowest successful cell (`tenferro-trace`, 24.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 26.5x faster than the slowest successful cell (`tenferro-trace`, 24.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 86.7x faster than the slowest successful cell (`tenferro-trace`, 24.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 26.2x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 28.0x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 101.7x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 34.6x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 28.0x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 41.4x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 26.6x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 88.1x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 26.6x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 32.4x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 105.1x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 38.3x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_squares_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 99.9x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sin` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`julia-base`, 51.500 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=1, shape=`1048576+1048576`): `jax-cpu` is 11.5x faster than the slowest successful cell (`julia-base`, 7.018 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=1, shape=`1048576+1048576`): `pytorch-cpu` is 11.9x faster than the slowest successful cell (`julia-base`, 7.018 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=1, shape=`1048576+1048576`): `tenferro-direct` is 10.0x faster than the slowest successful cell (`julia-base`, 7.018 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `jax-cpu` is 18.7x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `julia-base` is 11.9x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `pytorch-cpu` is 51.3x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `tenferro-direct` is 40.6x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=1, shape=`4194304 -> 2097152`): `jax-cpu` is 12.9x faster than the slowest successful cell (`julia-base`, 7.803 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=1, shape=`4194304 -> 2097152`): `pytorch-cpu` is 13.8x faster than the slowest successful cell (`julia-base`, 7.803 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=1, shape=`4194304 -> 2097152`): `tenferro-direct` is 11.8x faster than the slowest successful cell (`julia-base`, 7.803 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=1, shape=`4194304 -> 2097152`): `tenferro-trace` is 11.5x faster than the slowest successful cell (`julia-base`, 7.803 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `jax-cpu` is 23.2x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `pytorch-cpu` is 50.9x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `tenferro-direct` is 36.4x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 12.6x faster than the slowest successful cell (`julia-base`, 6.814 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 11.6x faster than the slowest successful cell (`julia-base`, 6.814 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `jax-cpu` is 290.8x faster than the slowest successful cell (`tenferro-trace`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `julia-base` is 40.0x faster than the slowest successful cell (`tenferro-trace`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 148.6x faster than the slowest successful cell (`tenferro-trace`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `tenferro-direct` is 169.9x faster than the slowest successful cell (`tenferro-trace`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `jax-cpu` is 27.2x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 33.0x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 44.6x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `jax-cpu` is 30.2x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 40.9x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 47.1x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `jax-cpu` is 117.4x faster than the slowest successful cell (`tenferro-trace`, 35.986 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `julia-base` is 31.3x faster than the slowest successful cell (`tenferro-trace`, 35.986 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 98.9x faster than the slowest successful cell (`tenferro-trace`, 35.986 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `tenferro-direct` is 43.5x faster than the slowest successful cell (`tenferro-trace`, 35.986 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 5010.0x faster than the slowest successful cell (`julia-base`, 8.377 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `pytorch-cpu` is 11.4x faster than the slowest successful cell (`julia-base`, 7.863 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-direct` is 12.0x faster than the slowest successful cell (`julia-base`, 7.863 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 4877.9x faster than the slowest successful cell (`julia-base`, 7.863 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 14.6x faster than the slowest successful cell (`jax-cpu`, 0.023 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.024 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x4x4,rhs=1`): `jax-cpu` is 22.6x faster than the slowest successful cell (`tenferro-trace`, 6.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x4x4,rhs=1`): `pytorch-cpu` is 44.6x faster than the slowest successful cell (`tenferro-trace`, 6.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x4x4,rhs=1`): `tenferro-direct` is 36.4x faster than the slowest successful cell (`tenferro-trace`, 6.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `jax-cpu` is 37.6x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `pytorch-cpu` is 78.5x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `tenferro-direct` is 56.9x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x128x128,rhs=1`): `jax-cpu` is 47.6x faster than the slowest successful cell (`tenferro-trace`, 6.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x128x128,rhs=1`): `pytorch-cpu` is 40.4x faster than the slowest successful cell (`tenferro-trace`, 6.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x128x128,rhs=1`): `tenferro-direct` is 65.2x faster than the slowest successful cell (`tenferro-trace`, 6.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `jax-cpu` is 39.8x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `pytorch-cpu` is 54.2x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `tenferro-direct` is 32.7x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 26.0x faster than the slowest successful cell (`tenferro-trace`, 315.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `julia-base` is 29.1x faster than the slowest successful cell (`tenferro-trace`, 315.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 33.1x faster than the slowest successful cell (`tenferro-trace`, 315.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`256x128`): `jax-cpu` is 20.8x faster than the slowest successful cell (`tenferro-eager`, 28.820 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`256x128`): `pytorch-cpu` is 70.7x faster than the slowest successful cell (`tenferro-eager`, 28.820 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`256x128`): `tenferro-direct` is 45.1x faster than the slowest successful cell (`tenferro-eager`, 28.820 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`768x384`): `jax-cpu` is 260.7x faster than the slowest successful cell (`tenferro-eager`, 2295.437 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 497.8x faster than the slowest successful cell (`tenferro-eager`, 2295.437 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 13.2x faster than the slowest successful cell (`tenferro-trace`, 263.302 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `julia-base` is 16.4x faster than the slowest successful cell (`tenferro-trace`, 263.302 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 14.5x faster than the slowest successful cell (`tenferro-trace`, 263.302 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `jax-cpu` is 51.5x faster than the slowest successful cell (`tenferro-trace`, 2280.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `julia-base` is 442.4x faster than the slowest successful cell (`tenferro-trace`, 2280.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 445.2x faster than the slowest successful cell (`tenferro-trace`, 2280.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 12.3x faster than the slowest successful cell (`tenferro-direct`, 237.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 17.7x faster than the slowest successful cell (`tenferro-direct`, 237.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `pytorch-cpu` is 19.5x faster than the slowest successful cell (`jax-cpu`, 18.279 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 28.0x faster than the slowest successful cell (`jax-cpu`, 18.279 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 30.1x faster than the slowest successful cell (`jax-cpu`, 18.279 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `julia-base` is 10.4x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 25.7x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 81.2x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `jax-cpu` is 11.8x faster than the slowest successful cell (`tenferro-trace`, 204.583 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `julia-base` is 12.1x faster than the slowest successful cell (`tenferro-trace`, 204.583 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 14.6x faster than the slowest successful cell (`tenferro-trace`, 204.583 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `jax-cpu` is 11.8x faster than the slowest successful cell (`tenferro-direct`, 203.743 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `julia-base` is 11.9x faster than the slowest successful cell (`tenferro-direct`, 203.743 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 14.6x faster than the slowest successful cell (`tenferro-direct`, 203.743 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 26.6x faster than the slowest successful cell (`tenferro-trace`, 326.131 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `julia-base` is 30.1x faster than the slowest successful cell (`tenferro-trace`, 326.131 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 34.2x faster than the slowest successful cell (`tenferro-trace`, 326.131 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 54.3x faster than the slowest successful cell (`tenferro-trace`, 2318.928 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `julia-base` is 52.3x faster than the slowest successful cell (`tenferro-trace`, 2318.928 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 71.8x faster than the slowest successful cell (`tenferro-trace`, 2318.928 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 569.1x faster than the slowest successful cell (`tenferro-eager`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 367.4x faster than the slowest successful cell (`tenferro-eager`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 614.7x faster than the slowest successful cell (`tenferro-eager`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 42.2x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 28.2x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 42.5x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 112.0x faster than the slowest successful cell (`tenferro-eager`, 12.609 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 191.4x faster than the slowest successful cell (`tenferro-eager`, 12.609 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 646.6x faster than the slowest successful cell (`tenferro-eager`, 12.609 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 14.0x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 24.5x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 42.8x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 124.6x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 170.0x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 242.0x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 11.6x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 12.3x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 15.8x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 466.0x faster than the slowest successful cell (`tenferro-eager`, 65.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 1207.4x faster than the slowest successful cell (`tenferro-eager`, 65.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 894.2x faster than the slowest successful cell (`tenferro-eager`, 65.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 85.5x faster than the slowest successful cell (`tenferro-eager`, 82.958 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 282.7x faster than the slowest successful cell (`tenferro-eager`, 82.958 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 241.1x faster than the slowest successful cell (`tenferro-eager`, 82.958 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 480.8x faster than the slowest successful cell (`tenferro-eager`, 88.844 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 2050.2x faster than the slowest successful cell (`tenferro-eager`, 88.844 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 984.2x faster than the slowest successful cell (`tenferro-eager`, 88.844 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 83.9x faster than the slowest successful cell (`tenferro-eager`, 119.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 395.6x faster than the slowest successful cell (`tenferro-eager`, 119.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 252.9x faster than the slowest successful cell (`tenferro-eager`, 119.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 515.4x faster than the slowest successful cell (`tenferro-eager`, 112.763 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 669.3x faster than the slowest successful cell (`tenferro-eager`, 112.763 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 99.9x faster than the slowest successful cell (`tenferro-eager`, 144.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 106.9x faster than the slowest successful cell (`tenferro-eager`, 144.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 850.9x faster than the slowest successful cell (`tenferro-eager`, 23.985 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 2220.6x faster than the slowest successful cell (`tenferro-eager`, 23.985 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 2331.3x faster than the slowest successful cell (`tenferro-eager`, 23.985 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 152.3x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 627.1x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 500.5x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 271.4x faster than the slowest successful cell (`tenferro-eager`, 32.636 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 379.2x faster than the slowest successful cell (`tenferro-eager`, 32.636 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 891.3x faster than the slowest successful cell (`tenferro-eager`, 32.636 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 55.2x faster than the slowest successful cell (`tenferro-eager`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 241.8x faster than the slowest successful cell (`tenferro-eager`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 266.5x faster than the slowest successful cell (`tenferro-eager`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 117.6x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 199.4x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 863.3x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 16.9x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 26.8x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 34.5x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 116.8x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 195.4x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 829.9x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 16.0x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 26.3x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 60.0x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 496.4x faster than the slowest successful cell (`tenferro-eager`, 88.924 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 2055.8x faster than the slowest successful cell (`tenferro-eager`, 88.924 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 1028.7x faster than the slowest successful cell (`tenferro-eager`, 88.924 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 88.7x faster than the slowest successful cell (`tenferro-eager`, 119.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 425.1x faster than the slowest successful cell (`tenferro-eager`, 119.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 240.6x faster than the slowest successful cell (`tenferro-eager`, 119.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 78.3x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 105.0x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 299.3x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 13.6x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 19.7x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 14.1x faster than the slowest successful cell (`tenferro-eager`, 19.274 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 996.2x faster than the slowest successful cell (`tenferro-eager`, 104.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 438.9x faster than the slowest successful cell (`tenferro-eager`, 104.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 255.2x faster than the slowest successful cell (`tenferro-eager`, 104.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 178.1x faster than the slowest successful cell (`tenferro-eager`, 134.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 73.0x faster than the slowest successful cell (`tenferro-eager`, 134.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 20.5x faster than the slowest successful cell (`tenferro-eager`, 134.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/output_reuse/solve_read_into` (f64, threads=4, shape=`768x768,rhs=16`): `pytorch-cpu` is 34.0x faster than the slowest successful cell (`tenferro-direct`, 182.504 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 74425.9x faster than the slowest successful cell (`julia-base`, 110.299 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 76267.4x faster than the slowest successful cell (`julia-base`, 109.978 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/cast_f64_f32` (f64->f32, threads=4, shape=`33554432`): `pytorch-cpu` is 10.7x faster than the slowest successful cell (`tenferro-trace`, 219.410 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/cast_f64_f32` (f64->f32, threads=4, shape=`33554432`): `tenferro-direct` is 10.6x faster than the slowest successful cell (`tenferro-trace`, 219.410 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 81465.6x faster than the slowest successful cell (`julia-base`, 128.879 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 79711.7x faster than the slowest successful cell (`julia-base`, 128.495 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 122624.1x faster than the slowest successful cell (`julia-base`, 176.947 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 122938.7x faster than the slowest successful cell (`julia-base`, 177.278 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 171.7x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 179.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 161.1x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 40.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 164.3x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 197.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 10029.3 GB/s at 0.002 ms. The conservative physical lower bound is 0.168 ms, so the cell is 100.3x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 10402.6 GB/s at 0.002 ms. The conservative physical lower bound is 0.042 ms, so the cell is 26.0x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 181130.5 GB/s at 0.001 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1811.3x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 186155.0 GB/s at 0.001 ms. The conservative physical lower bound is 0.671 ms, so the cell is 465.4x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 169681.1 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1696.8x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 166523.2 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 416.3x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 93013.0 GB/s at 0.001 ms. The conservative physical lower bound is 1.342 ms, so the cell is 930.1x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 93077.5 GB/s at 0.001 ms. The conservative physical lower bound is 0.336 ms, so the cell is 232.7x faster than the bound.

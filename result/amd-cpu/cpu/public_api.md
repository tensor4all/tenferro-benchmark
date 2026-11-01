# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/amd-cpu/cpu/public_api/20261101_023353/run.yaml`
- Timestamp: `2026-11-01T02:33:53.900896+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/amd-cpu/cpu/public_api/20261101_023353`.

- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`

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

- CSV: `data/results/amd-cpu/cpu/public_api/20261101_023353/cpu_public_api_t1_20261101_023353.csv`
- CSV: `data/results/amd-cpu/cpu/public_api/20261101_023353/cpu_public_api_t4_20261101_023353.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 4.135 ± 0.025 | - | 4.240 ± 0.056 | 4.520 ± 0.041 | 3.929 ± 0.024 | 4.841 ± 2.323 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.359 ± 0.024 | - | 6.993 ± 1.026 | 2.375 ± 0.056 | 3.276 ± 0.764 | 3.492 ± 3.628 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 132.850 ± 4.428 | - | 129.342 ± 3.419 | 125.494 ± 2.830 | 132.565 ± 1.350 | 151.336 ± 1.815 | 154.657 ± 3.400 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 42.862 ± 1.003 | - | 43.443 ± 0.871 | 44.262 ± 0.437 | 69.361 ± 1.856 | 149.803 ± 1.906 | 63.940 ± 1.839 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 109.378 ± 2.676 | - | 115.068 ± 1.897 | 95.474 ± 1.244 | 94.674 ± 1.199 | 132.858 ± 2.637 | 180.180 ± 1.743 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 44.694 ± 0.316 | - | 44.646 ± 0.461 | 32.166 ± 0.429 | 49.679 ± 1.278 | 128.013 ± 1.551 | 62.568 ± 0.671 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 43.610 ± 0.180 | - | 43.503 ± 0.217 | 42.449 ± 0.162 | 67.364 ± 0.269 | 47.150 ± 0.667 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 29.536 ± 0.227 | - | 29.473 ± 0.100 | 29.516 ± 0.533 | 29.903 ± 0.376 | 23.680 ± 0.256 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 44.461 ± 0.363 | - | unsupported | 43.146 ± 0.337 | 67.860 ± 0.632 | 44.846 ± 2.377 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 30.492 ± 0.220 | - | unsupported | 29.459 ± 0.839 | 30.714 ± 0.858 | 22.950 ± 4.854 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 10.953 ± 1.641 | - | 10.941 ± 0.160 | 11.563 ± 0.014 | 11.957 ± 0.063 | 11.874 ± 0.223 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 14.654 ± 0.349 | - | 14.656 ± 0.033 | 14.109 ± 0.056 | 17.982 ± 0.344 | 17.728 ± 0.050 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 70.276 ± 0.096 | - | 70.239 ± 0.080 | 109.403 ± 0.828 | 106.200 ± 2.241 | 85.092 ± 0.669 | 84.671 ± 10.391 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 31.252 ± 0.600 | - | 31.054 ± 0.049 | 43.686 ± 0.319 | 53.991 ± 1.139 | 84.736 ± 1.481 | 41.686 ± 9.063 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 116.671 ± 0.032 | - | 116.592 ± 0.189 | 155.706 ± 0.191 | 177.616 ± 1.300 | 260.157 ± 0.915 | 158.747 ± 1.240 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 49.642 ± 0.850 | - | 49.642 ± 0.038 | 64.127 ± 0.454 | 73.372 ± 0.929 | 150.715 ± 0.819 | 63.417 ± 0.863 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 75.529 ± 1.435 | - | 75.743 ± 2.357 | 70.219 ± 0.534 | 73.708 ± 0.873 | 81.948 ± 1.487 | 82.736 ± 1.776 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 25.591 ± 1.001 | - | 25.435 ± 1.024 | 24.385 ± 0.495 | 39.927 ± 1.893 | 78.763 ± 0.225 | 34.178 ± 0.612 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 2.544 ± 0.091 | - | 21.011 ± 0.179 | 24.161 ± 0.013 | 16.808 ± 0.384 | 4.127 ± 0.059 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 5.999 ± 0.002 | - | 9.447 ± 0.052 | 11.268 ± 0.013 | 2.758 ± 0.301 | 4.121 ± 0.030 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 8.619 ± 0.061 | - | 8.719 ± 0.101 | 9.847 ± 0.042 | 7.617 ± 0.029 | 9.164 ± 0.218 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 5.401 ± 0.042 | - | 13.559 ± 1.010 | 5.688 ± 0.059 | 7.988 ± 0.986 | 7.436 ± 0.294 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 5.556 ± 0.311 | - | 5.593 ± 0.119 | 5.566 ± 0.130 | 5.174 ± 0.037 | 5.695 ± 0.884 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.779 ± 0.078 | - | 13.583 ± 6.609 | 3.637 ± 0.049 | 4.699 ± 0.103 | 5.573 ± 1.117 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 6.436 ± 0.044 | - | 6.426 ± 0.096 | 7.650 ± 0.029 | 5.476 ± 0.043 | 5.293 ± 0.184 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 5.805 ± 0.075 | - | 5.753 ± 0.146 | 5.913 ± 0.048 | 6.123 ± 0.039 | 5.606 ± 0.034 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 44.027 ± 1.101 | - | 43.113 ± 0.264 | 42.382 ± 0.079 | 67.258 ± 0.325 | 44.337 ± 0.529 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 29.424 ± 0.120 | - | 30.136 ± 0.938 | 29.191 ± 0.930 | 29.553 ± 0.370 | 22.252 ± 0.208 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 48.894 ± 0.188 | - | unsupported | 49.588 ± 0.622 | 61.364 ± 3.446 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 24.730 ± 0.209 | - | unsupported | 25.673 ± 0.404 | 34.696 ± 0.415 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 127.919 ± 0.768 | - | 127.146 ± 1.743 | 128.955 ± 1.917 | 133.869 ± 2.767 | 144.007 ± 4.448 | 159.070 ± 2.019 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 42.788 ± 1.078 | - | 42.274 ± 0.232 | 42.195 ± 0.942 | 67.663 ± 0.631 | 147.788 ± 2.113 | 69.762 ± 0.358 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 141.483 ± 4.273 | - | 140.870 ± 1.028 | 142.919 ± 2.585 | 139.418 ± 2.978 | 150.009 ± 4.457 | 174.666 ± 13.603 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 47.163 ± 0.765 | - | 45.914 ± 0.477 | 47.035 ± 0.559 | 76.862 ± 1.784 | 153.929 ± 2.924 | 74.710 ± 0.973 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 27.269 ± 0.151 | - | 59.811 ± 0.044 | 76.807 ± 0.835 | 52.189 ± 0.242 | 81.492 ± 9.174 | 86.532 ± 0.340 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 12.400 ± 0.963 | - | 26.135 ± 0.477 | 34.303 ± 0.744 | 26.805 ± 0.169 | 82.029 ± 0.530 | 33.326 ± 13.224 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 10.622 ± 0.851 | - | 10.929 ± 0.166 | 37.346 ± 1.890 | 40.131 ± 1.710 | 45.427 ± 9.658 | 42.049 ± 12.943 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 7.153 ± 0.173 | - | 6.893 ± 0.351 | 12.925 ± 0.375 | 23.052 ± 0.431 | 47.047 ± 10.212 | 14.924 ± 26.308 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 21.475 ± 0.170 | - | 22.552 ± 0.521 | 54.577 ± 0.895 | 43.528 ± 1.108 | 23.081 ± 0.428 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 13.829 ± 0.223 | - | 14.064 ± 0.419 | 20.053 ± 0.242 | 25.036 ± 0.863 | 24.964 ± 0.338 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 28.387 ± 0.120 | - | 28.623 ± 0.058 | 56.753 ± 1.299 | 108.838 ± 0.337 | 67.725 ± 15.057 | 72.348 ± 12.267 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 11.896 ± 0.020 | - | 11.922 ± 0.048 | 22.612 ± 0.581 | 55.120 ± 0.219 | 70.119 ± 18.798 | 30.779 ± 16.585 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 141.423 ± 3.073 | - | 141.758 ± 1.473 | 139.017 ± 2.133 | 139.151 ± 0.727 | 155.772 ± 3.076 | 170.097 ± 4.178 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 46.275 ± 1.004 | - | 46.194 ± 0.253 | 46.855 ± 0.991 | 75.821 ± 1.404 | 156.427 ± 2.139 | 71.831 ± 0.796 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 20.474 ± 0.058 | - | 20.564 ± 0.084 | 45.313 ± 0.483 | 42.935 ± 0.121 | 63.952 ± 13.068 | 70.495 ± 13.289 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 8.509 ± 0.259 | - | 8.654 ± 0.034 | 17.060 ± 0.556 | 21.829 ± 0.120 | 74.253 ± 10.726 | 37.993 ± 11.402 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 20.641 ± 0.027 | - | 20.377 ± 0.040 | 32.484 ± 0.481 | 28.314 ± 0.167 | 38.370 ± 9.133 | 42.189 ± 0.334 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 8.110 ± 0.089 | - | 8.111 ± 0.015 | 11.637 ± 0.572 | 14.917 ± 0.142 | 38.176 ± 0.391 | 23.567 ± 6.415 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 25.402 ± 0.088 | - | 25.624 ± 0.071 | 52.007 ± 0.462 | 66.821 ± 1.197 | 85.625 ± 22.832 | 94.388 ± 8.015 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 10.606 ± 0.082 | - | 10.610 ± 0.022 | 19.552 ± 0.267 | 33.512 ± 0.575 | 85.278 ± 15.473 | 37.322 ± 17.595 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 13.209 ± 0.057 | - | 13.237 ± 0.046 | 32.858 ± 0.355 | 37.811 ± 0.108 | 46.829 ± 8.498 | 50.665 ± 0.247 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 5.648 ± 0.034 | - | 5.666 ± 0.027 | 11.999 ± 0.182 | 20.027 ± 0.247 | 47.073 ± 0.254 | 20.659 ± 12.667 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.452 ± 0.012 | 0.495 ± 0.007 | unsupported | unsupported | 0.500 ± 0.004 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.351 ± 0.006 | 0.421 ± 0.008 | unsupported | unsupported | 0.406 ± 0.039 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 142.264 ± 2.628 | - | 142.017 ± 5.544 | 138.182 ± 3.653 | 142.875 ± 1.632 | 158.670 ± 3.484 | 173.137 ± 3.831 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 46.382 ± 0.902 | - | 45.638 ± 0.222 | 46.980 ± 0.813 | 74.598 ± 1.570 | 166.069 ± 2.058 | 73.970 ± 1.227 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 143.010 ± 2.630 | - | 140.437 ± 4.245 | 138.376 ± 2.133 | 141.621 ± 0.327 | 159.594 ± 4.289 | 172.967 ± 4.343 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 46.819 ± 1.006 | - | 46.053 ± 0.381 | 46.561 ± 0.582 | 74.084 ± 6.454 | 166.060 ± 4.150 | 74.679 ± 0.949 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 139.060 ± 1.806 | - | 139.253 ± 2.303 | 141.273 ± 2.085 | 142.861 ± 1.867 | 154.607 ± 1.363 | 174.593 ± 5.378 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 46.237 ± 1.205 | - | 46.363 ± 0.599 | 47.443 ± 0.942 | 74.699 ± 0.872 | 156.660 ± 3.398 | 73.152 ± 1.620 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 129.499 ± 2.665 | - | 129.391 ± 2.880 | 130.412 ± 1.191 | 129.873 ± 0.949 | 144.943 ± 4.156 | 157.097 ± 3.026 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 42.721 ± 0.352 | - | 42.377 ± 0.328 | 42.287 ± 0.633 | 67.935 ± 0.986 | 145.499 ± 4.446 | 68.965 ± 1.978 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 57.937 ± 0.069 | - | 57.992 ± 0.184 | 63.804 ± 0.591 | 75.350 ± 0.419 | 126.361 ± 10.945 | 123.242 ± 0.390 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 23.573 ± 0.119 | - | 23.687 ± 0.103 | 22.478 ± 0.142 | 37.880 ± 0.213 | 126.747 ± 0.691 | 53.054 ± 10.803 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 17.784 ± 0.809 | - | 17.134 ± 0.375 | 12.598 ± 0.387 | 26.284 ± 0.941 | 13.674 ± 0.208 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 6.930 ± 0.221 | - | 6.991 ± 0.212 | 6.279 ± 0.116 | 14.234 ± 0.223 | 13.818 ± 1.030 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.918 ± 0.183 | - | 2.023 ± 0.118 | 24.125 ± 0.303 | 2.449 ± 0.063 | 1.326 ± 0.254 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.798 ± 0.026 | - | 0.801 ± 0.014 | 11.561 ± 0.274 | 1.485 ± 0.255 | 1.283 ± 0.075 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 2.593 ± 0.056 | - | 2.528 ± 0.073 | 4.062 ± 0.030 | 2.407 ± 0.306 | 1.264 ± 0.118 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 1.279 ± 0.102 | - | 1.243 ± 0.014 | 2.347 ± 0.034 | 1.344 ± 0.069 | 1.431 ± 0.176 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 17.830 ± 0.355 | - | 17.760 ± 0.978 | 12.686 ± 0.152 | 26.069 ± 0.220 | 13.653 ± 0.216 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 6.597 ± 0.102 | - | 6.910 ± 0.110 | 6.461 ± 0.245 | 14.499 ± 0.270 | 13.762 ± 0.207 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.934 ± 0.088 | - | 2.055 ± 0.165 | 23.099 ± 0.150 | 2.462 ± 0.055 | 1.216 ± 0.195 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.794 ± 0.018 | - | 0.807 ± 0.025 | 10.832 ± 0.325 | 1.354 ± 0.044 | 1.173 ± 0.463 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 11.006 ± 0.445 | - | 10.886 ± 0.157 | 16.467 ± 0.162 | 12.515 ± 0.154 | 6.460 ± 0.302 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 5.325 ± 0.234 | - | 5.714 ± 0.119 | 8.827 ± 0.023 | 7.222 ± 0.652 | 7.225 ± 0.068 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 11.681 ± 0.683 | - | 11.445 ± 0.233 | 11.002 ± 0.097 | 19.560 ± 0.172 | 10.857 ± 0.194 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 6.461 ± 0.242 | - | 6.409 ± 0.108 | 6.322 ± 0.186 | 11.029 ± 0.207 | 11.757 ± 0.318 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 1.057 ± 0.070 | - | 1.092 ± 0.141 | 2.107 ± 0.060 | 1.996 ± 0.005 | 1.099 ± 0.158 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.488 ± 0.067 | - | 0.429 ± 0.049 | 0.938 ± 0.043 | 1.787 ± 0.032 | 0.997 ± 0.109 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.155 ± 0.219 | - | 1.360 ± 0.176 | 0.915 ± 0.041 | 3.149 ± 0.295 | 1.098 ± 0.079 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.615 ± 0.038 | - | 0.621 ± 0.046 | 0.377 ± 0.050 | 1.334 ± 0.114 | 1.200 ± 0.051 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 12.549 ± 0.801 | - | 12.096 ± 0.578 | 10.954 ± 0.214 | 20.309 ± 0.162 | 11.028 ± 0.398 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 6.235 ± 0.169 | - | 6.431 ± 0.124 | 6.525 ± 0.288 | 11.065 ± 0.035 | 11.523 ± 0.752 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 1.048 ± 0.119 | - | 1.119 ± 0.256 | 2.077 ± 0.106 | 2.027 ± 0.043 | 0.916 ± 0.110 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.503 ± 0.036 | - | 0.479 ± 0.046 | 0.927 ± 0.021 | 1.158 ± 0.056 | 1.005 ± 0.218 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.303 ± 0.212 | - | 1.292 ± 0.136 | 0.881 ± 0.025 | 3.190 ± 0.355 | 1.016 ± 0.196 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.632 ± 0.095 | - | 0.608 ± 0.035 | 0.456 ± 0.058 | 1.332 ± 0.262 | 1.118 ± 0.128 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 1.236 ± 0.152 | 1.166 ± 0.058 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.391 ± 0.023 | 0.378 ± 0.021 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 38.155 ± 0.584 | - | 40.159 ± 0.167 | 56.518 ± 0.432 | 72.050 ± 0.973 | 72.029 ± 6.713 | 87.201 ± 9.518 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 17.993 ± 0.293 | - | 18.036 ± 0.188 | 21.070 ± 0.327 | 42.275 ± 5.245 | 77.211 ± 7.963 | 42.855 ± 10.887 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 148.836 ± 4.472 | - | 150.966 ± 4.427 | 149.461 ± 0.554 | 256.248 ± 3.284 | 270.693 ± 3.777 | 270.874 ± 3.631 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 47.239 ± 0.275 | - | 47.186 ± 0.071 | 48.595 ± 0.608 | 129.471 ± 0.829 | 270.558 ± 1.770 | 109.644 ± 2.465 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 143.102 ± 3.025 | - | 145.783 ± 2.151 | 151.216 ± 6.247 | 158.451 ± 3.933 | 158.976 ± 5.790 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 49.291 ± 1.577 | - | 48.174 ± 1.935 | 54.832 ± 2.175 | 84.667 ± 2.456 | 162.342 ± 3.069 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 133.431 ± 1.268 | - | 133.812 ± 5.168 | 134.090 ± 1.069 | 141.069 ± 3.355 | 147.173 ± 4.866 | 154.115 ± 4.534 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 42.444 ± 1.083 | - | 41.895 ± 0.125 | 43.531 ± 1.302 | 70.660 ± 0.541 | 151.591 ± 2.831 | 66.755 ± 2.201 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 24.784 ± 0.074 | - | 24.917 ± 0.047 | 58.026 ± 1.119 | 108.455 ± 0.362 | 67.608 ± 9.248 | 71.439 ± 13.984 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 10.221 ± 0.038 | - | 10.265 ± 0.066 | 22.727 ± 0.452 | 54.932 ± 0.183 | 67.957 ± 15.126 | 29.808 ± 17.897 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 138.197 ± 3.230 | - | 138.312 ± 3.945 | 191.138 ± 0.556 | 141.859 ± 2.073 | 220.714 ± 2.838 | 225.931 ± 4.246 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 45.672 ± 0.366 | - | 45.506 ± 0.329 | 71.046 ± 0.731 | 72.414 ± 1.590 | 222.043 ± 4.107 | 94.585 ± 0.654 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 142.750 ± 3.287 | - | 141.428 ± 4.056 | 141.874 ± 4.272 | 142.009 ± 2.021 | 149.612 ± 4.909 | 172.271 ± 1.758 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 46.465 ± 0.457 | - | 45.624 ± 0.193 | 47.559 ± 0.698 | 74.549 ± 1.844 | 158.231 ± 1.400 | 73.191 ± 0.941 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 51.350 ± 0.061 | - | 51.802 ± 0.036 | 76.610 ± 0.645 | 40.428 ± 0.418 | 80.091 ± 12.583 | 89.945 ± 20.353 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 20.481 ± 0.064 | - | 20.538 ± 0.684 | 28.998 ± 0.532 | 20.931 ± 0.499 | 81.763 ± 10.688 | 35.430 ± 13.802 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 143.802 ± 3.983 | 145.428 ± 6.869 | unsupported | 157.417 ± 5.075 | 162.285 ± 8.564 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 48.409 ± 1.458 | 47.591 ± 0.868 | unsupported | 53.943 ± 0.891 | 85.270 ± 1.475 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.906 ± 0.057 | - | 1.018 ± 0.047 | 0.851 ± 0.037 | 0.853 ± 0.134 | 7.744 ± 4.014 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.347 ± 0.039 | - | 0.366 ± 0.045 | 0.328 ± 0.031 | 0.937 ± 0.204 | 1.597 ± 6.578 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.844 ± 0.167 | - | 0.898 ± 0.152 | 0.947 ± 0.063 | 0.933 ± 0.212 | 1.676 ± 1.302 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.418 ± 0.214 | - | 0.390 ± 0.051 | 0.361 ± 0.030 | 0.736 ± 0.086 | 1.567 ± 1.243 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.262 ± 0.094 | - | unsupported | 1.430 ± 0.122 | 1.342 ± 0.167 | 7.491 ± 6.919 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.532 ± 0.107 | - | unsupported | 0.694 ± 0.041 | 1.215 ± 0.118 | 8.642 ± 4.095 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.394 ± 0.022 | - | 0.463 ± 0.018 | 0.692 ± 0.016 | 0.429 ± 0.031 | 1.231 ± 0.044 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.340 ± 0.028 | - | 0.412 ± 0.023 | 0.359 ± 0.018 | 0.374 ± 0.020 | 1.345 ± 0.097 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.849 ± 0.362 | - | 0.865 ± 0.255 | 1.377 ± 0.280 | 1.106 ± 0.059 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.364 ± 0.197 | - | 0.343 ± 0.058 | 0.531 ± 0.033 | 1.068 ± 0.024 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.910 ± 0.134 | - | 0.980 ± 0.166 | 0.800 ± 0.170 | 0.793 ± 0.129 | 1.949 ± 7.376 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.316 ± 0.077 | - | 0.353 ± 0.070 | 0.339 ± 0.082 | 0.730 ± 0.042 | 8.836 ± 7.942 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.794 ± 0.014 | - | 0.907 ± 0.036 | 0.518 ± 0.110 | 0.495 ± 0.034 | 0.565 ± 0.927 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 1.067 ± 0.014 | - | 1.218 ± 0.032 | 0.633 ± 0.007 | 0.507 ± 0.036 | 1.377 ± 0.123 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.679 ± 0.027 | - | 0.003 ± 0.000 | 1.894 ± 0.049 | 1.432 ± 0.016 | 9.740 ± 6.494 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.753 ± 0.051 | - | 0.004 ± 0.000 | 0.883 ± 0.368 | 1.043 ± 0.027 | 9.845 ± 1.604 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.050 ± 0.000 | - | 0.055 ± 0.006 | 0.076 ± 0.002 | 0.093 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.093 ± 0.001 | - | 0.102 ± 0.002 | 0.127 ± 0.002 | 0.183 ± 0.087 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.238 ± 0.012 | - | 0.265 ± 0.011 | 0.276 ± 0.003 | 0.323 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.081 ± 0.002 | - | 0.091 ± 0.001 | 0.148 ± 0.004 | 0.118 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.004 ± 0.000 | - | 0.008 ± 0.002 | 0.010 ± 0.001 | 0.043 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.001 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.002 | 0.005 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.002 | 0.036 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.674 ± 0.012 | - | 0.725 ± 0.012 | 1.164 ± 0.006 | 0.716 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.025 ± 0.000 | - | 0.032 ± 0.000 | 0.038 ± 0.001 | 0.070 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.001 | 0.031 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.051 ± 0.000 | - | 0.101 ± 0.001 | 0.086 ± 0.000 | 0.102 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.094 ± 0.000 | - | 0.161 ± 0.002 | 0.124 ± 0.002 | 0.162 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.240 ± 0.013 | - | 0.412 ± 0.016 | 0.342 ± 0.017 | 0.352 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.084 ± 0.004 | - | 0.124 ± 0.001 | 0.178 ± 0.003 | 0.127 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.002 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.005 ± 0.000 | - | 0.014 ± 0.000 | 0.017 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.009 ± 0.001 | 0.010 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.009 ± 0.002 | 0.011 ± 0.003 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.009 ± 0.001 | 0.010 ± 0.000 | 0.012 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.009 ± 0.001 | 0.010 ± 0.000 | 0.012 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.003 ± 0.000 | - | 0.010 ± 0.001 | 0.011 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.009 ± 0.001 | 0.010 ± 0.001 | 0.012 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.009 ± 0.001 | 0.010 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.003 ± 0.000 | - | 0.010 ± 0.001 | 0.011 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.673 ± 0.009 | - | 0.931 ± 0.015 | 1.399 ± 0.016 | 1.991 ± 0.913 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.027 ± 0.000 | - | 0.053 ± 0.003 | 0.059 ± 0.003 | 0.067 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.003 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.002 | 0.017 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.004 ± 0.000 | - | 0.012 ± 0.001 | 0.012 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.632 ± 0.008 | - | 0.649 ± 0.010 | 0.746 ± 0.039 | 0.940 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 2.125 ± 0.005 | - | 2.153 ± 0.013 | 2.171 ± 0.005 | 2.711 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 6.473 ± 0.017 | - | 6.520 ± 0.028 | 6.548 ± 0.014 | 8.954 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 3.675 ± 0.008 | - | 3.680 ± 0.020 | 3.221 ± 0.008 | 2.688 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.007 ± 0.001 | 0.009 ± 0.001 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.109 ± 0.001 | - | 0.115 ± 0.005 | 0.122 ± 0.006 | 0.206 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.001 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.008 ± 0.000 | - | 0.013 ± 0.000 | 0.017 ± 0.002 | 0.031 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.012 ± 0.001 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.013 ± 0.000 | 0.016 ± 0.002 | 0.036 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.021 ± 0.000 | - | 0.025 ± 0.000 | 0.029 ± 0.001 | 0.076 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.032 ± 0.005 | 0.029 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | 0.015 ± 0.000 | 0.018 ± 0.001 | 0.040 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.027 ± 0.000 | - | 0.032 ± 0.003 | 0.036 ± 0.001 | 0.089 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 29.187 ± 0.178 | - | 29.469 ± 0.171 | 25.700 ± 0.279 | 20.467 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.015 ± 0.001 | 0.034 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.865 ± 0.002 | - | 0.879 ± 0.002 | 0.883 ± 0.006 | 1.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.018 ± 0.000 | - | 0.023 ± 0.001 | 0.027 ± 0.001 | 0.043 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.052 ± 0.000 | - | 0.057 ± 0.002 | 0.061 ± 0.001 | 0.141 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 1.059 ± 0.012 | - | 1.069 ± 0.006 | 0.738 ± 0.432 | 0.929 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 3.451 ± 0.013 | - | 3.516 ± 0.057 | 2.145 ± 0.008 | 2.221 ± 0.071 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 6.520 ± 0.167 | - | 6.549 ± 0.047 | 9.364 ± 0.050 | 5.883 ± 0.099 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 3.593 ± 0.020 | - | 3.652 ± 0.025 | 2.253 ± 0.025 | 3.460 ± 0.062 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.005 ± 0.000 | - | 0.013 ± 0.008 | 0.013 ± 0.001 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.111 ± 0.003 | - | 0.175 ± 0.021 | 0.179 ± 0.024 | 0.202 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.008 ± 0.001 | - | 0.016 ± 0.000 | 0.011 ± 0.001 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.014 ± 0.000 | - | 0.022 ± 0.001 | 0.017 ± 0.001 | 0.035 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.008 ± 0.001 | - | 0.014 ± 0.000 | 0.017 ± 0.002 | 0.031 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.014 ± 0.000 | - | 0.023 ± 0.001 | 0.016 ± 0.002 | 0.031 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.033 ± 0.000 | - | 0.042 ± 0.001 | 0.030 ± 0.001 | 0.088 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.008 ± 0.001 | - | 0.015 ± 0.000 | 0.017 ± 0.002 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.018 ± 0.000 | - | 0.026 ± 0.000 | 0.018 ± 0.000 | 0.036 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.043 ± 0.001 | - | 0.051 ± 0.000 | 0.036 ± 0.001 | 0.074 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 28.867 ± 0.199 | - | 39.283 ± 1.436 | 17.750 ± 0.397 | 30.989 ± 7.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.013 ± 0.000 | - | 0.020 ± 0.000 | 0.022 ± 0.000 | 0.036 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.869 ± 0.002 | - | 1.307 ± 0.407 | 1.255 ± 0.017 | 1.744 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.031 ± 0.000 | - | 0.040 ± 0.001 | 0.026 ± 0.000 | 0.046 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.082 ± 0.002 | - | 0.090 ± 0.001 | 0.060 ± 0.004 | 0.118 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.690 ± 0.014 | - | 0.705 ± 0.033 | 0.529 ± 0.008 | 0.481 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 2.235 ± 0.009 | - | 2.247 ± 0.008 | 1.958 ± 0.023 | 2.009 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 5.816 ± 0.009 | - | 5.845 ± 0.008 | 5.773 ± 0.013 | 6.529 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.318 ± 0.020 | - | 1.332 ± 0.018 | 1.335 ± 0.015 | 1.457 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.003 ± 0.001 | - | 0.006 ± 0.000 | 0.008 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.070 ± 0.002 | - | 0.078 ± 0.017 | 0.090 ± 0.007 | 0.163 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.001 | 0.042 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | 0.012 ± 0.000 | 0.017 ± 0.002 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.013 ± 0.000 | 0.015 ± 0.001 | 0.031 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.019 ± 0.001 | - | 0.023 ± 0.000 | 0.026 ± 0.001 | 0.041 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.026 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.011 ± 0.000 | - | 0.015 ± 0.000 | 0.017 ± 0.001 | 0.030 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.024 ± 0.000 | - | 0.029 ± 0.001 | 0.032 ± 0.001 | 0.046 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 10.438 ± 0.059 | - | 10.504 ± 0.156 | 10.567 ± 0.054 | 11.290 ± 0.117 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.013 ± 0.001 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.556 ± 0.013 | - | 0.570 ± 0.016 | 0.629 ± 0.016 | 0.753 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.019 ± 0.000 | - | 0.024 ± 0.002 | 0.024 ± 0.001 | 0.045 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.046 ± 0.000 | - | 0.052 ± 0.001 | 0.054 ± 0.001 | 0.118 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 1.125 ± 0.013 | - | 1.150 ± 1.195 | 0.802 ± 0.114 | 0.449 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 3.593 ± 0.019 | - | 3.613 ± 0.008 | 1.929 ± 0.004 | 1.308 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 5.818 ± 0.033 | - | 5.844 ± 0.029 | 8.533 ± 0.014 | 3.730 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.251 ± 0.023 | - | 1.274 ± 0.029 | 1.172 ± 0.018 | 1.955 ± 0.171 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.006 ± 0.001 | - | 0.012 ± 0.000 | 0.012 ± 0.002 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.115 ± 0.005 | - | 0.142 ± 0.021 | 0.139 ± 0.005 | 0.154 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.008 ± 0.000 | - | 0.016 ± 0.001 | 0.011 ± 0.000 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.013 ± 0.001 | - | 0.020 ± 0.001 | 0.015 ± 0.001 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.007 ± 0.000 | - | 0.014 ± 0.000 | 0.015 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.015 ± 0.000 | - | 0.022 ± 0.000 | 0.015 ± 0.000 | 0.034 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.031 ± 0.001 | - | 0.038 ± 0.003 | 0.027 ± 0.003 | 0.040 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.009 ± 0.000 | - | 0.016 ± 0.000 | 0.016 ± 0.008 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.018 ± 0.000 | - | 0.026 ± 0.001 | 0.017 ± 0.001 | 0.029 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.040 ± 0.001 | - | 0.047 ± 0.004 | 0.032 ± 0.001 | 0.060 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 9.957 ± 0.150 | - | 19.837 ± 0.225 | 9.122 ± 0.128 | 15.069 ± 3.184 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.013 ± 0.000 | - | 0.020 ± 0.000 | 0.018 ± 0.001 | 0.030 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.885 ± 0.006 | - | 0.902 ± 0.004 | 0.953 ± 0.015 | 0.479 ± 0.068 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.031 ± 0.000 | - | 0.040 ± 0.001 | 0.024 ± 0.000 | 0.045 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.074 ± 0.002 | - | 0.083 ± 0.003 | 0.053 ± 0.001 | 0.093 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.277 ± 0.022 | - | 0.282 ± 0.010 | 0.238 ± 0.017 | 0.438 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 1.098 ± 0.004 | - | 1.108 ± 0.005 | 0.948 ± 0.003 | 1.942 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 3.088 ± 0.005 | - | 3.122 ± 0.013 | 2.897 ± 0.004 | 6.490 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.511 ± 0.008 | - | 0.529 ± 0.012 | 0.563 ± 0.017 | 1.476 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.002 ± 0.002 | - | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.031 ± 0.001 | - | 0.035 ± 0.001 | 0.041 ± 0.000 | 0.166 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.001 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.013 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.010 ± 0.000 | - | 0.015 ± 0.000 | 0.019 ± 0.001 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.016 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.014 ± 0.001 | 0.024 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.013 ± 0.000 | - | 0.018 ± 0.000 | 0.022 ± 0.000 | 0.039 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 4.080 ± 0.057 | - | 4.168 ± 0.053 | 4.365 ± 0.087 | 11.080 ± 0.089 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.001 | 0.017 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.231 ± 0.009 | - | 0.238 ± 0.003 | 0.246 ± 0.019 | 0.786 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.010 ± 0.000 | - | 0.014 ± 0.000 | 0.017 ± 0.001 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.025 ± 0.000 | - | 0.030 ± 0.002 | 0.032 ± 0.001 | 0.114 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.481 ± 0.014 | - | 0.496 ± 0.019 | 0.396 ± 0.018 | 0.484 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 1.626 ± 0.033 | - | 1.636 ± 0.017 | 0.952 ± 0.017 | 1.291 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 4.214 ± 0.017 | - | 4.255 ± 0.654 | 3.679 ± 0.010 | 3.704 ± 0.059 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.694 ± 0.016 | - | 0.710 ± 0.015 | 0.711 ± 0.022 | 1.778 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.004 ± 0.002 | - | 0.010 ± 0.000 | 0.015 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.032 ± 0.000 | - | 0.049 ± 0.004 | 0.056 ± 0.003 | 0.147 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.012 ± 0.000 | 0.017 ± 0.001 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.008 ± 0.000 | - | 0.014 ± 0.000 | 0.013 ± 0.001 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.005 ± 0.001 | - | 0.011 ± 0.002 | 0.016 ± 0.001 | 0.021 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.008 ± 0.000 | - | 0.015 ± 0.000 | 0.019 ± 0.002 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.016 ± 0.000 | - | 0.023 ± 0.000 | 0.020 ± 0.001 | 0.035 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.012 ± 0.000 | 0.016 ± 0.002 | 0.017 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.010 ± 0.000 | - | 0.017 ± 0.000 | 0.020 ± 0.001 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.020 ± 0.000 | - | 0.027 ± 0.000 | 0.023 ± 0.000 | 0.042 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 5.446 ± 0.039 | - | 14.635 ± 2.026 | 5.293 ± 0.014 | 13.937 ± 0.719 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.014 ± 0.000 | 0.018 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.233 ± 0.003 | - | 0.311 ± 0.018 | 0.320 ± 0.021 | 0.442 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.016 ± 0.000 | - | 0.023 ± 0.001 | 0.024 ± 0.001 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.036 ± 0.000 | - | 0.043 ± 0.000 | 0.033 ± 0.002 | 0.090 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.148 ± 0.004 | - | 0.157 ± 0.005 | 0.189 ± 0.019 | 0.121 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.218 ± 0.011 | - | 0.228 ± 0.003 | 0.253 ± 0.018 | 0.222 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.509 ± 0.013 | - | 0.540 ± 0.011 | 0.529 ± 0.017 | 0.585 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.137 ± 0.003 | - | 0.150 ± 0.006 | 0.181 ± 0.045 | 0.225 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.014 ± 0.005 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.012 ± 0.001 | - | 0.017 ± 0.007 | 0.026 ± 0.002 | 0.060 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.014 ± 0.001 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.014 ± 0.001 | - | 0.020 ± 0.005 | 0.028 ± 0.001 | 0.053 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.014 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.016 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.000 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.016 ± 0.002 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.019 ± 0.005 | 0.037 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 1.062 ± 0.014 | - | 1.103 ± 0.009 | 1.292 ± 0.018 | 1.255 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.077 ± 0.001 | - | 0.084 ± 0.001 | 0.087 ± 0.002 | 0.122 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.000 | 0.043 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.020 ± 0.000 | 0.053 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.199 ± 0.016 | - | 0.209 ± 0.017 | 0.176 ± 0.003 | 0.125 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.318 ± 0.022 | - | 0.330 ± 0.019 | 0.240 ± 0.022 | 0.216 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.812 ± 0.013 | - | 0.847 ± 2.999 | 0.386 ± 0.023 | 0.642 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.201 ± 0.006 | - | 0.209 ± 0.022 | 0.229 ± 0.023 | 0.244 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.006 ± 0.000 | - | 0.011 ± 0.000 | 0.025 ± 0.006 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.021 ± 0.000 | - | 0.028 ± 0.003 | 0.038 ± 0.001 | 0.042 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | 0.011 ± 0.000 | 0.023 ± 0.003 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.025 ± 0.001 | - | 0.032 ± 0.003 | 0.043 ± 0.001 | 0.063 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.006 ± 0.001 | - | 0.012 ± 0.000 | 0.023 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.006 ± 0.001 | - | 0.012 ± 0.003 | 0.025 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.006 ± 0.000 | - | 0.012 ± 0.000 | 0.026 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.008 ± 0.001 | - | 0.013 ± 0.000 | 0.028 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.006 ± 0.000 | - | 0.011 ± 0.000 | 0.026 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.006 ± 0.000 | - | 0.012 ± 0.001 | 0.026 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.008 ± 0.000 | - | 0.014 ± 0.000 | 0.028 ± 0.000 | 0.020 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 1.528 ± 0.025 | - | 1.560 ± 0.055 | 0.704 ± 0.014 | 0.857 ± 0.087 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.007 ± 0.001 | - | 0.013 ± 0.004 | 0.027 ± 0.000 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.127 ± 0.001 | - | 0.138 ± 0.002 | 0.079 ± 0.008 | 0.183 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.007 ± 0.001 | - | 0.013 ± 0.000 | 0.028 ± 0.001 | 0.022 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.012 ± 0.000 | - | 0.017 ± 0.001 | 0.031 ± 0.000 | 0.022 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.260 ± 0.016 | - | 0.270 ± 0.011 | 0.224 ± 0.013 | 0.325 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.602 ± 0.010 | - | 0.621 ± 0.037 | 0.537 ± 0.014 | 0.905 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.380 ± 0.009 | - | 1.385 ± 0.009 | 1.356 ± 0.010 | 2.366 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.495 ± 0.016 | - | 0.511 ± 0.022 | 0.590 ± 0.023 | 0.675 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.001 | 0.022 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.018 ± 0.000 | - | 0.022 ± 0.000 | 0.027 ± 0.001 | 0.110 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.001 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.019 ± 0.000 | - | 0.024 ± 0.004 | 0.030 ± 0.001 | 0.087 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.001 | 0.024 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.002 | 0.006 ± 0.000 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.006 ± 0.000 | - | 0.010 ± 0.002 | 0.010 ± 0.003 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.002 | 0.007 ± 0.001 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.036 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 3.881 ± 0.017 | - | 4.717 ± 0.037 | 4.413 ± 0.013 | 4.753 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.129 ± 0.001 | - | 0.137 ± 0.002 | 0.176 ± 0.004 | 0.256 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.006 ± 0.000 | - | 0.011 ± 0.000 | 0.009 ± 0.000 | 0.027 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.012 ± 0.000 | - | 0.017 ± 0.001 | 0.016 ± 0.000 | 0.046 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.263 ± 0.005 | - | 0.420 ± 0.015 | 0.322 ± 0.019 | 0.346 ± 0.059 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.607 ± 0.010 | - | 0.904 ± 0.003 | 0.765 ± 0.016 | 0.865 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.350 ± 0.019 | - | 1.394 ± 0.400 | 2.027 ± 0.011 | 1.365 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.565 ± 0.020 | - | 0.602 ± 0.024 | 0.519 ± 0.026 | 1.016 ± 0.140 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.001 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.019 ± 0.000 | - | 0.038 ± 0.003 | 0.044 ± 0.001 | 0.056 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.033 ± 0.001 | - | 0.041 ± 0.001 | 0.047 ± 0.001 | 0.051 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.007 ± 0.000 | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.011 ± 0.000 | 0.011 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.008 ± 0.002 | - | 0.013 ± 0.001 | 0.012 ± 0.001 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.012 ± 0.000 | - | 0.016 ± 0.000 | 0.017 ± 0.003 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.012 ± 0.000 | 0.011 ± 0.001 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.008 ± 0.000 | - | 0.013 ± 0.000 | 0.013 ± 0.001 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.014 ± 0.006 | - | 0.019 ± 0.001 | 0.018 ± 0.000 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 4.427 ± 0.072 | - | 15.772 ± 0.692 | 3.795 ± 0.014 | 7.995 ± 1.561 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.005 ± 0.000 | - | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.131 ± 0.002 | - | 0.227 ± 0.017 | 0.289 ± 0.017 | 0.264 ± 0.059 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.012 ± 0.000 | - | 0.017 ± 0.000 | 0.016 ± 0.000 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.014 ± 0.005 | - | 0.026 ± 0.001 | 0.026 ± 0.000 | 0.040 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.159 ± 0.002 | - | 0.147 ± 0.005 | 0.099 ± 0.002 | 0.286 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.213 ± 0.010 | - | 0.191 ± 0.041 | 0.140 ± 0.003 | 0.385 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.546 ± 0.015 | - | 0.559 ± 0.012 | 0.406 ± 0.015 | 0.803 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.119 ± 0.004 | - | 0.133 ± 0.041 | 0.148 ± 0.010 | 0.199 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.001 | 0.020 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.011 ± 0.000 | - | 0.018 ± 0.003 | 0.030 ± 0.001 | 0.075 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.001 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.003 | 0.023 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.019 ± 0.002 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.021 ± 0.002 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.018 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.020 ± 0.001 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.926 ± 0.004 | - | 0.983 ± 0.005 | 1.029 ± 0.013 | 1.016 ± 0.059 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.070 ± 0.003 | - | 0.080 ± 0.005 | 0.081 ± 0.008 | 0.123 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.018 ± 0.000 | 0.021 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.006 ± 0.001 | - | 0.012 ± 0.001 | 0.021 ± 0.001 | 0.025 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.170 ± 0.003 | - | 0.187 ± 0.003 | 0.132 ± 0.014 | 0.280 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.215 ± 0.027 | - | 0.259 ± 0.072 | 0.174 ± 0.003 | 0.415 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.545 ± 0.010 | - | 0.755 ± 0.018 | 0.383 ± 0.016 | 0.848 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.123 ± 0.015 | - | 0.212 ± 0.072 | 0.207 ± 0.010 | 0.180 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.015 ± 0.004 | 0.027 ± 0.004 | 0.017 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.012 ± 0.000 | - | 0.031 ± 0.007 | 0.047 ± 0.004 | 0.071 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.005 ± 0.000 | - | 0.015 ± 0.001 | 0.027 ± 0.003 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.016 ± 0.000 | 0.027 ± 0.004 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.016 ± 0.001 | 0.027 ± 0.002 | 0.018 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.005 ± 0.000 | - | 0.016 ± 0.000 | 0.027 ± 0.005 | 0.019 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.017 ± 0.000 | 0.030 ± 0.003 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.016 ± 0.001 | 0.026 ± 0.002 | 0.022 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.006 ± 0.000 | - | 0.016 ± 0.000 | 0.027 ± 0.002 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.005 ± 0.003 | - | 0.018 ± 0.000 | 0.032 ± 0.005 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.926 ± 0.003 | - | 1.417 ± 0.039 | 0.590 ± 0.025 | 0.786 ± 0.053 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.016 ± 0.001 | 0.029 ± 0.001 | 0.021 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.072 ± 0.003 | - | 0.132 ± 0.002 | 0.075 ± 0.005 | 0.161 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.007 ± 0.000 | - | 0.017 ± 0.001 | 0.029 ± 0.004 | 0.024 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.011 ± 0.002 | - | 0.023 ± 0.004 | 0.034 ± 0.001 | 0.027 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.818 ± 0.004 | - | 1.854 ± 0.042 | 1.566 ± 0.019 | 2.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 6.021 ± 0.032 | - | 6.055 ± 0.068 | 5.162 ± 0.077 | 5.102 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 17.752 ± 0.067 | - | 17.791 ± 0.061 | 15.959 ± 0.016 | 14.705 ± 0.063 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 4.942 ± 0.044 | - | 4.979 ± 0.021 | 5.658 ± 0.191 | 4.177 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.005 ± 0.001 | - | 0.009 ± 0.001 | 0.015 ± 0.001 | 0.027 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.199 ± 0.018 | - | 0.210 ± 0.009 | 0.231 ± 0.026 | 0.237 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.009 ± 0.000 | - | 0.014 ± 0.000 | 0.021 ± 0.008 | 0.032 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.179 ± 0.007 | - | 0.188 ± 0.008 | 0.209 ± 0.019 | 0.222 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.021 ± 0.000 | - | 0.026 ± 0.001 | 0.032 ± 0.000 | 0.042 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.008 ± 0.000 | - | 0.013 ± 0.000 | 0.018 ± 0.001 | 0.034 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.020 ± 0.000 | - | 0.026 ± 0.001 | 0.030 ± 0.001 | 0.043 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.054 ± 0.000 | - | 0.060 ± 0.001 | 0.064 ± 0.002 | 0.074 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.010 ± 0.000 | - | 0.015 ± 0.000 | 0.019 ± 0.001 | 0.034 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.026 ± 0.000 | - | 0.032 ± 0.007 | 0.036 ± 0.005 | 0.050 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.073 ± 0.001 | - | 0.079 ± 0.014 | 0.080 ± 0.001 | 0.090 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 39.128 ± 0.147 | - | 41.218 ± 0.122 | 44.261 ± 0.319 | 33.439 ± 0.290 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.017 ± 0.000 | - | 0.022 ± 0.003 | 0.026 ± 0.000 | 0.041 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.549 ± 0.021 | - | 1.574 ± 0.020 | 1.646 ± 0.010 | 1.537 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.049 ± 0.000 | - | 0.055 ± 0.000 | 0.057 ± 0.001 | 0.085 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.140 ± 0.001 | - | 0.147 ± 0.003 | 0.143 ± 0.004 | 0.161 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 3.095 ± 0.011 | - | 3.114 ± 0.007 | 1.531 ± 0.013 | 2.031 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 9.854 ± 0.030 | - | 6.050 ± 0.009 | 5.076 ± 0.024 | 2.785 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 17.834 ± 0.183 | - | 17.880 ± 0.050 | 18.582 ± 0.098 | 8.758 ± 0.122 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 4.710 ± 0.048 | - | 4.775 ± 0.044 | 4.669 ± 0.153 | 5.123 ± 0.998 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.009 ± 0.001 | - | 0.016 ± 0.001 | 0.020 ± 0.009 | 0.032 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.323 ± 0.055 | - | 0.327 ± 0.027 | 0.338 ± 0.021 | 0.261 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.017 ± 0.000 | - | 0.024 ± 0.004 | 0.020 ± 0.001 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.280 ± 0.020 | - | 0.309 ± 0.036 | 0.291 ± 0.025 | 0.242 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.022 ± 0.000 | - | 0.045 ± 0.004 | 0.032 ± 0.002 | 0.047 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.016 ± 0.000 | - | 0.022 ± 0.001 | 0.025 ± 0.001 | 0.034 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.035 ± 0.001 | - | 0.043 ± 0.003 | 0.030 ± 0.001 | 0.043 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.057 ± 0.004 | - | 0.099 ± 0.002 | 0.064 ± 0.001 | 0.076 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.019 ± 0.000 | - | 0.025 ± 0.001 | 0.031 ± 0.006 | 0.036 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.045 ± 0.000 | - | 0.055 ± 0.007 | 0.035 ± 0.002 | 0.050 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.075 ± 0.002 | - | 0.128 ± 0.003 | 0.079 ± 0.001 | 0.094 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 37.383 ± 0.212 | - | 48.375 ± 0.748 | 36.651 ± 0.556 | 67.065 ± 19.339 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.031 ± 0.001 | - | 0.037 ± 0.003 | 0.037 ± 0.003 | 0.043 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 2.389 ± 0.024 | - | 2.404 ± 0.023 | 2.364 ± 0.023 | 0.902 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.084 ± 0.002 | - | 0.093 ± 0.004 | 0.058 ± 0.007 | 0.072 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.144 ± 0.002 | - | 0.236 ± 0.022 | 0.140 ± 0.003 | 0.182 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 1.364 ± 0.009 | - | 1.375 ± 0.009 | 1.100 ± 0.002 | 1.369 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 5.130 ± 0.008 | - | 5.155 ± 0.040 | 4.398 ± 0.036 | 4.088 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 15.145 ± 0.204 | - | 15.194 ± 0.023 | 13.884 ± 0.029 | 12.253 ± 0.110 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 2.301 ± 0.026 | - | 2.319 ± 0.019 | 2.461 ± 0.016 | 2.452 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.001 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.136 ± 0.006 | - | 0.146 ± 0.002 | 0.175 ± 0.016 | 0.221 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.008 ± 0.000 | - | 0.012 ± 0.000 | 0.018 ± 0.001 | 0.036 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.107 ± 0.012 | - | 0.115 ± 0.004 | 0.133 ± 0.004 | 0.162 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.017 ± 0.000 | - | 0.022 ± 0.000 | 0.028 ± 0.002 | 0.038 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.006 ± 0.001 | - | 0.011 ± 0.000 | 0.016 ± 0.001 | 0.028 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.018 ± 0.000 | - | 0.023 ± 0.000 | 0.029 ± 0.001 | 0.044 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.043 ± 0.000 | - | 0.049 ± 0.005 | 0.053 ± 0.001 | 0.064 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.008 ± 0.000 | - | 0.012 ± 0.005 | 0.018 ± 0.001 | 0.035 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.023 ± 0.000 | - | 0.028 ± 0.000 | 0.032 ± 0.001 | 0.041 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.057 ± 0.000 | - | 0.063 ± 0.000 | 0.066 ± 0.001 | 0.076 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 18.171 ± 0.110 | - | 18.480 ± 0.131 | 19.418 ± 0.067 | 18.697 ± 0.079 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.013 ± 0.001 | - | 0.018 ± 0.000 | 0.021 ± 0.001 | 0.035 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 1.060 ± 0.014 | - | 1.078 ± 0.009 | 1.165 ± 0.011 | 1.193 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.042 ± 0.000 | - | 0.048 ± 0.000 | 0.048 ± 0.001 | 0.057 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.114 ± 0.000 | - | 0.121 ± 0.003 | 0.119 ± 0.003 | 0.166 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 2.307 ± 0.014 | - | 2.323 ± 0.014 | 1.048 ± 0.026 | 1.352 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 8.411 ± 0.010 | - | 5.193 ± 3.282 | 4.305 ± 0.009 | 2.465 ± 0.359 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 15.188 ± 0.059 | - | 15.249 ± 0.040 | 16.703 ± 0.033 | 6.290 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 2.654 ± 0.029 | - | 2.669 ± 0.026 | 2.795 ± 0.018 | 3.147 ± 0.069 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.008 ± 0.000 | - | 0.014 ± 0.000 | 0.019 ± 0.008 | 0.043 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.220 ± 0.027 | - | 0.242 ± 0.020 | 0.249 ± 0.025 | 0.176 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.014 ± 0.000 | - | 0.021 ± 0.000 | 0.018 ± 0.007 | 0.040 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.172 ± 0.014 | - | 0.186 ± 0.038 | 0.197 ± 0.012 | 0.154 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.018 ± 0.000 | - | 0.040 ± 0.004 | 0.027 ± 0.001 | 0.041 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.013 ± 0.001 | - | 0.019 ± 0.001 | 0.022 ± 0.001 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.030 ± 0.000 | - | 0.037 ± 0.004 | 0.027 ± 0.001 | 0.039 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.044 ± 0.000 | - | 0.080 ± 0.002 | 0.053 ± 0.005 | 0.062 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.015 ± 0.001 | - | 0.021 ± 0.001 | 0.025 ± 0.004 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.040 ± 0.001 | - | 0.046 ± 0.003 | 0.032 ± 0.009 | 0.044 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.059 ± 0.000 | - | 0.103 ± 0.002 | 0.066 ± 0.001 | 0.075 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 21.092 ± 0.128 | - | 31.856 ± 2.969 | 21.855 ± 0.070 | 26.506 ± 7.065 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.024 ± 0.001 | - | 0.030 ± 0.001 | 0.030 ± 0.004 | 0.035 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 1.686 ± 0.024 | - | 1.697 ± 0.032 | 1.754 ± 0.011 | 0.659 ± 0.129 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.075 ± 0.007 | - | 0.106 ± 0.129 | 0.047 ± 0.001 | 0.059 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.116 ± 0.001 | - | 0.197 ± 0.020 | 0.118 ± 0.005 | 0.136 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.482 ± 0.024 | - | unsupported | 0.357 ± 0.017 | 0.752 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 2.287 ± 0.011 | - | unsupported | 1.940 ± 0.007 | 2.713 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 5.535 ± 0.017 | - | unsupported | 5.314 ± 0.062 | 6.840 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.982 ± 0.010 | - | unsupported | 1.151 ± 0.007 | 1.105 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.002 ± 0.002 | - | unsupported | 0.009 ± 0.001 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.047 ± 0.001 | - | unsupported | 0.062 ± 0.002 | 0.078 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | unsupported | 0.012 ± 0.000 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.039 ± 0.001 | - | unsupported | 0.053 ± 0.002 | 0.066 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | unsupported | 0.015 ± 0.001 | 0.027 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.017 ± 0.000 | - | unsupported | 0.026 ± 0.001 | 0.036 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.011 ± 0.001 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | unsupported | 0.018 ± 0.004 | 0.027 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.023 ± 0.000 | - | unsupported | 0.031 ± 0.001 | 0.045 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 7.742 ± 0.144 | - | unsupported | 8.912 ± 0.017 | 8.583 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | unsupported | 0.012 ± 0.001 | 0.022 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.371 ± 0.015 | - | unsupported | 0.429 ± 0.014 | 0.477 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.019 ± 0.000 | - | unsupported | 0.025 ± 0.001 | 0.037 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.044 ± 0.000 | - | unsupported | 0.052 ± 0.001 | 0.086 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.813 ± 0.001 | - | unsupported | 0.585 ± 0.017 | 0.756 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 3.589 ± 0.021 | - | unsupported | 1.934 ± 0.004 | 1.420 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 8.406 ± 0.014 | - | unsupported | 7.698 ± 0.015 | 3.525 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.240 ± 0.022 | - | unsupported | 1.456 ± 0.020 | 1.460 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.005 ± 0.002 | - | unsupported | 0.014 ± 0.002 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.066 ± 0.002 | - | unsupported | 0.091 ± 0.027 | 0.078 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.008 ± 0.000 | - | unsupported | 0.017 ± 0.002 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.057 ± 0.001 | - | unsupported | 0.080 ± 0.007 | 0.066 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.012 ± 0.000 | - | unsupported | 0.016 ± 0.001 | 0.028 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.007 ± 0.000 | - | unsupported | 0.016 ± 0.001 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.014 ± 0.000 | - | unsupported | 0.023 ± 0.001 | 0.034 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.028 ± 0.001 | - | unsupported | 0.027 ± 0.005 | 0.045 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.007 ± 0.001 | - | unsupported | 0.016 ± 0.002 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.018 ± 0.000 | - | unsupported | 0.026 ± 0.001 | 0.029 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.036 ± 0.000 | - | unsupported | 0.032 ± 0.000 | 0.047 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 9.704 ± 0.010 | - | unsupported | 11.360 ± 0.021 | 11.463 ± 1.305 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.011 ± 0.001 | - | unsupported | 0.019 ± 0.004 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.514 ± 0.017 | - | unsupported | 0.434 ± 0.024 | 0.263 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.031 ± 0.000 | - | unsupported | 0.025 ± 0.000 | 0.037 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.068 ± 0.002 | - | unsupported | 0.051 ± 0.002 | 0.079 ± 0.021 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 1.058 ± 0.010 | - | unsupported | 0.948 ± 0.004 | 3.672 ± 0.086 | 2.501 ± 0.911 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.139 ± 0.002 | - | unsupported | 0.061 ± 0.003 | 0.107 ± 0.009 | 0.134 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.143 ± 0.002 | - | unsupported | 0.084 ± 0.012 | 0.181 ± 0.017 | 0.228 ± 0.126 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.345 ± 0.010 | - | unsupported | 0.242 ± 0.016 | 0.492 ± 0.011 | 0.680 ± 0.029 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 1.054 ± 0.011 | - | unsupported | 0.485 ± 0.016 | 1.099 ± 0.123 | 2.522 ± 0.933 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.141 ± 0.002 | - | unsupported | 0.062 ± 0.002 | 0.108 ± 0.015 | 0.102 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.139 ± 0.009 | - | unsupported | 0.082 ± 0.003 | 0.174 ± 0.023 | 0.180 ± 0.023 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.343 ± 0.012 | - | unsupported | 0.161 ± 0.021 | 0.546 ± 0.032 | 0.704 ± 0.025 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.306 ± 0.011 | - | unsupported | 0.369 ± 0.010 | 0.810 ± 0.032 | 0.419 ± 0.016 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.034 ± 0.000 | - | unsupported | 0.045 ± 0.001 | 0.202 ± 0.046 | 0.074 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.059 ± 0.000 | - | unsupported | 0.061 ± 0.001 | 0.224 ± 0.011 | 0.116 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.199 ± 0.001 | - | unsupported | 0.166 ± 0.004 | 0.378 ± 0.026 | 0.211 ± 0.010 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.309 ± 0.011 | - | unsupported | 0.569 ± 0.018 | 0.898 ± 0.045 | 0.419 ± 0.010 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.035 ± 0.000 | - | unsupported | 0.074 ± 0.003 | 0.191 ± 0.010 | 0.074 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.060 ± 0.000 | - | unsupported | 0.099 ± 0.001 | 0.235 ± 0.013 | 0.116 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.202 ± 0.002 | - | unsupported | 0.228 ± 0.017 | 0.385 ± 0.031 | 0.213 ± 0.010 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.196 ± 0.001 | - | 0.294 ± 0.011 | 0.195 ± 0.015 | 0.403 ± 0.024 | 0.261 ± 0.011 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.081 ± 0.000 | - | 0.091 ± 0.021 | 0.073 ± 0.002 | 0.077 ± 0.003 | 0.058 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.092 ± 0.001 | - | 0.102 ± 0.001 | 0.087 ± 0.001 | 0.101 ± 0.006 | 0.085 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.123 ± 0.004 | - | 0.151 ± 0.002 | 0.122 ± 0.012 | 0.178 ± 0.030 | 0.139 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.220 ± 0.012 | - | 0.383 ± 0.014 | 0.259 ± 0.017 | 0.442 ± 0.043 | 0.267 ± 0.010 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.082 ± 0.001 | - | 0.160 ± 0.003 | 0.125 ± 0.002 | 0.081 ± 0.010 | 0.061 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.094 ± 0.000 | - | 0.196 ± 0.006 | 0.145 ± 0.004 | 0.111 ± 0.028 | 0.087 ± 0.003 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.124 ± 0.001 | - | 0.235 ± 0.019 | 0.180 ± 0.018 | 0.190 ± 0.004 | 0.142 ± 0.002 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 42.997 ± 0.357 | - | 47.006 ± 0.488 | 62.067 ± 0.682 | 60.300 ± 0.473 | 52.838 ± 6.088 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 21.384 ± 0.618 | - | 32.943 ± 0.397 | 25.387 ± 0.241 | 64.029 ± 15.761 | 39.439 ± 12.995 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 24.907 ± 0.225 | - | 24.339 ± 0.399 | 26.230 ± 0.225 | 22.965 ± 0.355 | 21.384 ± 2.431 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 13.173 ± 0.473 | - | 29.100 ± 11.198 | 11.517 ± 0.191 | 14.156 ± 0.240 | 12.733 ± 0.244 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 7.551 ± 0.328 | - | 7.536 ± 0.061 | 8.010 ± 0.067 | 7.967 ± 0.037 | 8.001 ± 0.140 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 11.381 ± 0.248 | - | 11.479 ± 0.104 | 10.658 ± 0.114 | 12.993 ± 0.005 | 11.715 ± 0.054 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 8.808 ± 0.095 | - | 8.773 ± 0.036 | 8.818 ± 0.069 | 9.563 ± 0.053 | 9.341 ± 0.223 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 13.316 ± 0.192 | - | 13.313 ± 0.035 | 12.008 ± 0.032 | 14.268 ± 0.045 | 13.902 ± 0.034 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 21.135 ± 0.085 | - | 21.896 ± 0.205 | 19.249 ± 0.193 | 39.339 ± 0.115 | 14.184 ± 0.670 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 15.339 ± 0.242 | - | 28.534 ± 0.782 | 15.201 ± 0.121 | 32.969 ± 1.480 | 13.204 ± 0.337 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.489 ± 0.002 | 0.515 ± 0.013 | unsupported | 0.619 ± 0.008 | 0.767 ± 0.029 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 7.792 ± 0.047 | 7.905 ± 0.108 | unsupported | 8.785 ± 0.052 | 9.682 ± 0.089 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.397 ± 0.005 | 0.420 ± 0.007 | unsupported | 0.525 ± 0.003 | 1.447 ± 0.617 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 6.428 ± 0.008 | 16.193 ± 1.377 | unsupported | 5.747 ± 0.067 | 9.738 ± 0.301 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 41.438 ± 0.271 | - | 41.901 ± 0.584 | 42.798 ± 0.469 | 32.323 ± 0.359 | 27.850 ± 0.394 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 31.123 ± 4.157 | - | 33.852 ± 4.771 | 21.758 ± 0.379 | 20.146 ± 0.497 | 17.047 ± 2.523 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 17.606 ± 0.133 | 17.464 ± 0.155 | 9.118 ± 0.042 | 46.442 ± 0.413 | 9.674 ± 0.097 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 26.960 ± 0.793 | 23.752 ± 0.552 | 6.126 ± 0.230 | 44.078 ± 0.575 | 9.561 ± 0.125 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 25.386 ± 0.523 | - | 25.868 ± 0.467 | 32.625 ± 0.302 | 32.968 ± 0.214 | 31.078 ± 9.121 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 13.177 ± 0.072 | - | 24.725 ± 1.879 | 16.132 ± 0.247 | 26.390 ± 5.353 | 29.275 ± 9.993 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 1.366 ± 0.101 | - | 1.207 ± 0.134 | 1.272 ± 0.050 | 20.663 ± 0.618 | 2.754 ± 0.038 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.495 ± 0.062 | - | 0.459 ± 0.061 | 1.394 ± 0.153 | 2.584 ± 0.315 | 2.794 ± 0.083 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 19.895 ± 0.252 | - | 19.471 ± 0.118 | 19.876 ± 0.080 | 20.158 ± 0.181 | 20.586 ± 2.079 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 25.999 ± 2.639 | - | 27.082 ± 2.102 | 16.445 ± 0.070 | 21.467 ± 0.586 | 20.231 ± 1.063 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 19.653 ± 0.087 | - | 19.490 ± 0.190 | 19.858 ± 0.247 | 20.086 ± 0.137 | 19.402 ± 1.398 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 30.209 ± 8.401 | - | 29.018 ± 7.021 | 16.564 ± 0.233 | 20.252 ± 0.632 | 20.314 ± 0.209 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.175 ± 0.008 | 1.201 ± 0.005 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 21.323 ± 0.053 | 21.497 ± 0.049 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.360 ± 0.020 | 1.387 ± 0.005 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 20.387 ± 0.434 | 24.288 ± 0.333 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.090 ± 0.012 | 1.123 ± 0.024 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 20.302 ± 0.061 | 20.497 ± 0.082 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.247 ± 0.006 | 1.294 ± 0.024 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 19.076 ± 0.080 | 23.041 ± 0.240 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 24.894 ± 0.356 | - | 24.226 ± 0.288 | 26.181 ± 0.188 | 22.624 ± 0.189 | 21.358 ± 0.504 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 13.467 ± 0.266 | - | 35.009 ± 11.466 | 11.490 ± 0.281 | 14.121 ± 0.139 | 12.638 ± 0.279 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 59.003 ± 0.627 | 58.830 ± 0.608 | 60.520 ± 0.263 | 66.141 ± 0.481 | 63.498 ± 6.485 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 46.911 ± 0.292 | 47.225 ± 0.504 | 39.326 ± 0.821 | 56.074 ± 0.479 | 59.082 ± 7.994 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 33.884 ± 0.473 | - | 111.040 ± 1.662 | 30.577 ± 0.903 | 132.996 ± 1.768 | 30.303 ± 0.592 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 24.739 ± 0.280 | - | 68.828 ± 1.075 | 25.594 ± 2.961 | 115.155 ± 6.886 | 20.158 ± 0.442 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.140 ± 0.000 | 0.140 ± 0.001 | unsupported | 0.160 ± 0.035 | 0.036 ± 0.005 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.230 ± 0.004 | 2.220 ± 0.013 | unsupported | 2.472 ± 0.032 | 1.295 ± 0.130 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.074 ± 0.001 | 0.076 ± 0.000 | unsupported | 0.079 ± 0.008 | 0.145 ± 0.016 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.907 ± 0.006 | 0.903 ± 0.005 | unsupported | 1.057 ± 0.034 | 0.250 ± 0.198 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.081 ± 0.001 | 0.079 ± 0.001 | unsupported | 0.129 ± 0.012 | 0.233 ± 0.015 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.236 ± 0.004 | 1.238 ± 0.007 | unsupported | 1.765 ± 0.015 | 2.714 ± 0.039 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.038 ± 0.000 | 0.039 ± 0.000 | unsupported | 0.094 ± 0.025 | 0.163 ± 0.004 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.563 ± 0.002 | 0.567 ± 0.073 | unsupported | 0.764 ± 0.012 | 1.121 ± 0.098 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.198 ± 0.001 | 0.197 ± 0.000 | unsupported | 0.286 ± 0.002 | 0.187 ± 0.002 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 3.128 ± 0.036 | 3.122 ± 0.006 | unsupported | 4.449 ± 0.012 | 1.874 ± 0.065 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.077 ± 0.000 | 0.078 ± 0.000 | unsupported | 0.108 ± 0.001 | 0.142 ± 0.009 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 1.185 ± 0.072 | 1.184 ± 0.005 | unsupported | 1.542 ± 0.102 | 1.014 ± 0.039 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.060 ± 0.008 | 0.092 ± 0.006 | unsupported | 0.160 ± 0.001 | 0.181 ± 0.002 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.626 ± 0.029 | 0.736 ± 0.017 | unsupported | 0.905 ± 0.016 | 1.902 ± 0.021 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.080 ± 0.006 | 0.122 ± 0.003 | unsupported | 0.070 ± 0.001 | 0.190 ± 0.002 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.411 ± 0.009 | 0.477 ± 0.039 | unsupported | 0.377 ± 0.493 | 1.242 ± 0.137 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.127 ± 0.008 | 0.156 ± 0.003 | unsupported | 0.138 ± 0.003 | 0.224 ± 0.029 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.642 ± 0.005 | 1.734 ± 0.074 | unsupported | 1.242 ± 0.017 | 2.361 ± 0.050 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.138 ± 0.004 | 0.205 ± 0.001 | unsupported | 0.064 ± 0.001 | 0.217 ± 0.004 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.954 ± 0.089 | 1.036 ± 0.039 | unsupported | 0.499 ± 0.008 | 1.422 ± 0.035 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.291 ± 0.004 | 0.334 ± 0.002 | unsupported | unsupported | 0.265 ± 0.016 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 4.291 ± 0.022 | 4.414 ± 0.143 | unsupported | unsupported | 3.249 ± 0.022 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.273 ± 0.007 | 0.307 ± 0.003 | unsupported | unsupported | 0.252 ± 0.014 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 1.981 ± 0.012 | 2.132 ± 0.097 | unsupported | unsupported | 1.985 ± 0.031 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.011 ± 0.000 | 0.021 ± 0.000 | unsupported | 0.013 ± 0.000 | 0.050 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.101 ± 0.003 | 0.115 ± 0.009 | unsupported | 0.103 ± 0.006 | 0.250 ± 0.005 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.012 ± 0.000 | 0.026 ± 0.009 | unsupported | 0.015 ± 0.000 | 0.039 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.127 ± 0.003 | 0.141 ± 0.012 | unsupported | 0.097 ± 0.002 | 0.219 ± 0.007 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.028 ± 0.007 | 0.052 ± 0.005 | unsupported | 0.103 ± 0.002 | 0.167 ± 0.005 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.383 ± 0.030 | 0.401 ± 0.038 | unsupported | 0.556 ± 0.037 | 1.709 ± 0.066 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.048 ± 0.003 | 0.087 ± 0.004 | unsupported | 0.104 ± 0.003 | 0.160 ± 0.002 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.215 ± 0.008 | 0.259 ± 0.039 | unsupported | 0.288 ± 0.081 | 1.102 ± 0.063 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.054 ± 0.001 | 0.055 ± 0.001 | unsupported | 0.118 ± 0.006 | 0.176 ± 0.034 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.852 ± 0.006 | 0.866 ± 0.010 | unsupported | 1.816 ± 0.010 | 1.789 ± 0.045 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.028 ± 0.001 | 0.028 ± 0.000 | unsupported | 0.099 ± 0.002 | 0.138 ± 0.008 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.393 ± 0.002 | 0.392 ± 0.004 | unsupported | 0.705 ± 0.019 | 1.001 ± 0.032 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.057 ± 0.000 | 0.057 ± 0.001 | unsupported | 0.123 ± 0.002 | 0.202 ± 0.006 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.891 ± 0.004 | 0.904 ± 0.004 | unsupported | 1.863 ± 0.009 | 1.773 ± 0.042 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.028 ± 0.001 | 0.029 ± 0.000 | unsupported | 0.103 ± 0.001 | 0.140 ± 0.003 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.406 ± 0.004 | 0.415 ± 0.041 | unsupported | 0.741 ± 0.025 | 1.007 ± 0.030 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.124 ± 0.004 | 0.158 ± 0.004 | unsupported | 0.141 ± 0.001 | 0.244 ± 0.002 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.648 ± 0.022 | 1.752 ± 0.070 | unsupported | 1.183 ± 0.002 | 2.235 ± 0.038 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.138 ± 0.006 | 0.213 ± 0.014 | unsupported | 0.062 ± 0.002 | 0.217 ± 0.006 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 1.027 ± 0.097 | 1.106 ± 0.013 | unsupported | 0.482 ± 0.002 | 1.413 ± 0.117 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.185 ± 0.001 | 0.184 ± 0.001 | unsupported | 0.276 ± 0.002 | 0.349 ± 0.004 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 2.909 ± 0.006 | 2.908 ± 0.008 | unsupported | 4.331 ± 0.017 | 4.033 ± 0.027 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.074 ± 0.001 | 0.075 ± 0.003 | unsupported | 0.204 ± 0.003 | 0.213 ± 0.022 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 1.133 ± 0.009 | 1.135 ± 0.032 | unsupported | 1.559 ± 0.058 | 2.102 ± 0.021 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.736 ± 0.003 | 1.415 ± 0.010 | unsupported | 0.267 ± 0.002 | 0.217 ± 0.004 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 18.587 ± 0.121 | 29.718 ± 0.188 | unsupported | 5.780 ± 0.089 | 2.348 ± 0.049 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.853 ± 0.041 | 1.760 ± 0.018 | unsupported | 0.268 ± 0.003 | 0.179 ± 0.013 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 9.393 ± 0.108 | 22.561 ± 0.543 | unsupported | 2.463 ± 0.239 | 1.250 ± 0.013 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 28.962 ± 1.054 | - | unsupported | 29.652 ± 0.624 | - | 35.184 ± 2.636 | 41.381 ± 0.565 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 21.609 ± 0.809 | - | unsupported | 22.411 ± 0.408 | - | 32.166 ± 0.810 | 22.597 ± 1.003 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 21.884 ± 0.552 | - | unsupported | 27.593 ± 0.469 | - | 24.028 ± 0.902 | 25.430 ± 0.734 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 14.824 ± 0.249 | - | unsupported | 15.902 ± 0.463 | - | 21.965 ± 0.265 | 15.865 ± 0.537 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 14.525 ± 0.933 | - | unsupported | 24.458 ± 0.687 | - | 14.254 ± 0.657 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 14.908 ± 0.264 | - | unsupported | 15.758 ± 0.838 | - | 14.320 ± 0.274 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 28.730 ± 0.280 | - | unsupported | 29.229 ± 0.934 | - | 33.002 ± 0.701 | 39.997 ± 2.043 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 21.622 ± 0.851 | - | unsupported | 22.338 ± 0.833 | - | 33.712 ± 0.891 | 21.986 ± 1.982 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 48.239 ± 0.266 | - | unsupported | 49.841 ± 0.253 | - | 47.664 ± 0.129 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 24.790 ± 0.036 | - | unsupported | 25.284 ± 0.125 | - | 23.606 ± 0.170 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 47.902 ± 0.260 | - | unsupported | 49.601 ± 0.446 | - | 47.366 ± 0.064 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 24.626 ± 0.098 | - | unsupported | 25.257 ± 0.120 | - | 23.208 ± 0.027 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 29.325 ± 0.583 | - | unsupported | 29.494 ± 0.581 | - | 33.957 ± 0.529 | 41.782 ± 3.936 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 21.434 ± 0.337 | - | unsupported | 22.311 ± 0.134 | - | 32.368 ± 2.015 | 23.116 ± 0.815 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 22.803 ± 0.566 | - | unsupported | 24.717 ± 1.126 | - | 26.535 ± 0.241 | 29.406 ± 0.255 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 15.107 ± 0.217 | - | unsupported | 16.065 ± 0.832 | - | 27.205 ± 0.588 | 15.830 ± 1.191 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.742 ± 0.005 | unsupported | unsupported | 0.909 ± 0.004 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 10.974 ± 0.027 | unsupported | unsupported | 12.482 ± 0.069 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.512 ± 0.006 | unsupported | unsupported | 0.621 ± 0.001 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 6.080 ± 0.069 | unsupported | unsupported | 6.938 ± 0.063 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 29.906 ± 0.347 | - | unsupported | 29.419 ± 0.224 | - | 33.100 ± 0.455 | 42.231 ± 1.076 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 21.361 ± 0.572 | - | unsupported | 22.752 ± 0.689 | - | 33.756 ± 0.864 | 22.621 ± 0.607 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 111.923 ± 3.588 | - | 0.003 ± 0.000 | 114.118 ± 1.181 | 111.341 ± 3.213 | 140.052 ± 3.298 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 44.952 ± 0.408 | - | 0.004 ± 0.000 | 37.271 ± 0.546 | 58.159 ± 1.073 | 138.437 ± 4.176 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 69.332 ± 2.101 | - | 216.949 ± 5.367 | 71.200 ± 0.869 | 72.432 ± 0.546 | 81.852 ± 3.486 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 23.503 ± 0.379 | - | 90.118 ± 1.724 | 24.343 ± 0.157 | 37.759 ± 0.479 | 80.899 ± 3.004 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 226.711 ± 1.528 | - | 226.887 ± 1.992 | 228.201 ± 5.067 | 231.066 ± 5.416 | 261.497 ± 3.259 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 76.177 ± 1.553 | - | 75.788 ± 2.468 | 73.873 ± 1.940 | 116.320 ± 1.189 | 263.592 ± 4.728 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 70.112 ± 1.605 | - | 70.679 ± 0.762 | 76.951 ± 2.691 | 70.442 ± 2.141 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 35.947 ± 0.835 | - | 35.918 ± 0.935 | 26.003 ± 0.115 | 37.607 ± 0.510 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 140.765 ± 3.245 | - | 0.003 ± 0.000 | 130.991 ± 1.965 | 124.177 ± 1.894 | 162.865 ± 5.012 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 43.437 ± 0.346 | - | 0.004 ± 0.000 | 43.289 ± 1.282 | 46.806 ± 6.219 | 161.387 ± 1.175 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 89.331 ± 1.020 | - | 0.003 ± 0.000 | 83.296 ± 0.301 | 85.650 ± 0.611 | 215.938 ± 9.712 | 131.378 ± 1.327 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 29.791 ± 0.466 | - | 0.004 ± 0.000 | 82.438 ± 1.408 | 45.244 ± 1.445 | 218.198 ± 3.022 | 52.229 ± 0.605 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 63.170 ± 1.158 | - | 61.965 ± 0.334 | 67.558 ± 1.300 | 66.348 ± 1.094 | 85.034 ± 3.509 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 21.849 ± 0.137 | - | 22.034 ± 0.703 | 25.026 ± 0.665 | 34.957 ± 0.813 | 85.124 ± 1.565 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 62.603 ± 0.547 | - | 61.259 ± 0.380 | 66.169 ± 0.958 | 66.162 ± 0.524 | 84.773 ± 2.108 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 21.246 ± 0.147 | - | 21.451 ± 0.532 | 24.014 ± 0.105 | 34.452 ± 0.644 | 85.243 ± 2.731 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | - | - |
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
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 7.994054 | 499628.38 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 7.231701 | 451981.31 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 7.917579 | 494848.69 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 6.488289 | 405518.06 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 5.615713 | 350982.06 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 6.731712 | 420732.00 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 19.771940 | 1235746.25 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 18.654288 | 1165893.00 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 6.251578 | 390723.62 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 6.040235 | 377514.69 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 12.267437 | 766714.81 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 9.907849 | 619240.56 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 7.827693 | 489230.81 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 8.235613 | 514725.81 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 154.904671 | 9681541.94 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 140.553954 | 8784622.12 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 124.668436 | 7791777.25 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 126.487454 | 7905465.88 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 23.153254 | 1447078.38 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 8.395229 | 524701.81 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 6.357556 | 397347.25 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 6.725000 | 420312.50 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 155.806842 | 9737927.62 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 91.953180 | 5747073.75 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 102.843318 | 6427707.38 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 259.081763 | 16192610.19 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 18.807378 | 1175461.12 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 19.222521 | 1201407.56 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 341.172210 | 21323263.12 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 343.952654 | 21497040.88 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 21.758410 | 1359900.62 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 22.198741 | 1387421.31 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 326.191086 | 20386942.88 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 388.605897 | 24287868.56 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 17.436444 | 1089777.75 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 17.972373 | 1123273.31 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 324.837191 | 20302324.44 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 327.949249 | 20496828.06 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 19.948048 | 1246753.00 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 20.704236 | 1294014.75 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 305.209590 | 19075599.38 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 368.657579 | 23041098.69 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.309032 | 36078.62 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 10.244866 | 160076.03 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.957968 | 139968.25 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.971364 | 140177.56 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.362578 | 1295322.25 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 19.773518 | 2471689.75 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 17.842982 | 2230372.75 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 17.759356 | 2219919.50 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.279098 | 144985.91 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.029208 | 78581.38 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.743727 | 74120.73 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.844575 | 75696.48 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.002257 | 250282.13 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.454019 | 1056752.38 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.258225 | 907278.12 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.223421 | 902927.62 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.922994 | 233171.78 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.236865 | 128701.02 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.156266 | 80566.66 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.070747 | 79230.42 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 21.709821 | 2713727.62 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.116038 | 1764504.75 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.885286 | 1235660.75 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 9.904091 | 1238011.38 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.445391 | 163209.23 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.997222 | 93706.59 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.449553 | 38274.27 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.496390 | 39006.09 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.966941 | 1120867.62 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.109581 | 763697.62 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.503482 | 562935.25 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.536273 | 567034.12 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.980953 | 187202.39 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 18.311054 | 286110.22 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 12.652942 | 197702.22 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 12.604432 | 196944.25 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.990671 | 1873833.88 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 35.590586 | 4448823.25 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 25.027343 | 3128417.88 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 24.977542 | 3122192.75 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.074837 | 141794.33 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.927525 | 108242.58 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.945434 | 77272.41 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.981120 | 77830.00 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.114245 | 1014280.62 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.333345 | 1541668.12 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.476620 | 1184577.50 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 9.469908 | 1183738.50 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.575487 | 180866.98 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 10.260806 | 160325.09 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.820377 | 59693.39 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.870767 | 91730.73 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 15.216111 | 1902013.88 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.239527 | 904940.88 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.007862 | 625982.75 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.887480 | 735935.00 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.174482 | 190226.28 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.460048 | 69688.25 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.134486 | 80226.34 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.827524 | 122305.06 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.934883 | 1241860.38 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.013826 | 376728.25 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.288548 | 411068.50 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.814038 | 476754.75 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.357981 | 224343.45 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.832054 | 138000.84 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.129365 | 127021.33 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 10.009478 | 156398.09 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.890993 | 2361374.12 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.939598 | 1242449.75 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.138762 | 1642345.25 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 13.869544 | 1733693.00 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.899571 | 217180.80 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.114695 | 64292.11 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.841438 | 138147.47 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 13.088728 | 204511.38 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.379303 | 1422412.88 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.991038 | 498879.75 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.630880 | 953860.00 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 8.289928 | 1036241.00 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 16.986829 | 265419.20 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 18.641539 | 291274.05 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 21.358721 | 333730.02 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 25.988332 | 3248541.50 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 34.328926 | 4291115.75 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 35.310074 | 4413759.25 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 16.115566 | 251805.72 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 17.464647 | 272885.11 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 19.666508 | 307289.19 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 15.880120 | 1985015.00 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 15.850628 | 1981328.50 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 17.052878 | 2131609.75 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 3.207416 | 50115.88 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.826649 | 12916.39 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.731563 | 11430.67 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.334626 | 20853.53 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 2.002531 | 250316.37 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.824055 | 103006.88 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.811832 | 101479.00 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.917379 | 114672.38 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.481073 | 38766.77 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.932958 | 14577.47 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.748968 | 11702.62 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.670639 | 26103.73 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.751420 | 218927.50 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.772830 | 96603.75 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.017075 | 127134.38 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.125448 | 140681.00 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.656656 | 166510.25 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.561565 | 102524.45 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.789213 | 27956.45 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.328421 | 52006.58 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.669790 | 1708723.75 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.447406 | 555925.75 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.066033 | 383254.12 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.209942 | 401242.75 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.259986 | 160312.28 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.645950 | 103842.97 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.041739 | 47527.17 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.547748 | 86683.56 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.816972 | 1102121.50 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.305936 | 288242.00 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.722480 | 215310.00 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.069427 | 258678.38 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.256933 | 175889.58 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.551298 | 117989.03 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.464485 | 54132.58 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.490813 | 54543.95 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.310183 | 1788772.88 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.529128 | 1816141.00 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.814616 | 851827.00 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.926324 | 865790.50 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.817287 | 137770.11 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.328258 | 98879.03 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.765677 | 27588.70 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.799169 | 28112.02 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.005231 | 1000653.88 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.642891 | 705361.38 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.143097 | 392887.12 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.138418 | 392302.25 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.944658 | 202260.28 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.880371 | 123130.80 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.646794 | 56981.16 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.672712 | 57386.12 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.185080 | 1773135.00 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.905770 | 1863221.25 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.131436 | 891429.50 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.229458 | 903682.25 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.980491 | 140320.17 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.578384 | 102787.25 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.820349 | 28442.95 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.865152 | 29143.00 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.057789 | 1007223.62 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.931749 | 741468.62 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.250227 | 406278.38 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.322011 | 415251.38 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 15.603592 | 243806.12 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 9.022018 | 140969.03 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 7.907472 | 123554.25 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 10.131525 | 158305.08 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 17.878772 | 2234846.50 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.465755 | 1183219.38 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.186801 | 1648350.12 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 14.013121 | 1751640.12 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.865092 | 216642.06 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.985775 | 62277.73 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.840997 | 138140.58 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 13.624618 | 212884.66 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.300946 | 1412618.25 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.857065 | 482133.12 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.218194 | 1027274.25 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 8.849902 | 1106237.75 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 22.343099 | 349110.92 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.644842 | 275700.66 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 11.859023 | 185297.23 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 11.784294 | 184129.59 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 32.266099 | 4033262.37 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 34.644784 | 4330598.00 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 23.271826 | 2908978.25 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 23.263740 | 2907967.50 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.630556 | 212977.44 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 13.064567 | 204133.86 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.712238 | 73628.72 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.776398 | 74631.22 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 16.819059 | 2102382.38 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.474348 | 1559293.50 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.064321 | 1133040.12 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 9.079319 | 1134914.88 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.895300 | 217114.06 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.086392 | 266974.88 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 47.134401 | 736475.02 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 90.580343 | 1415317.86 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.782055 | 2347756.88 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 46.241028 | 5780128.50 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 148.697652 | 18587206.50 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 237.741218 | 29717652.25 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.484687 | 179448.23 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 17.135450 | 267741.41 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 54.574133 | 852720.83 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 112.614817 | 1759606.52 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.999995 | 1249999.38 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 19.705524 | 2463190.50 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 75.141336 | 9392667.00 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 180.489954 | 22561244.25 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 14.542694 | 908918.38 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 11.879343 | 742458.94 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 199.709974 | 12481873.38 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 175.586749 | 10974171.81 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 9.937712 | 621107.00 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 8.193955 | 512122.19 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 111.008206 | 6938012.88 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 97.285147 | 6080321.69 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.023755 | 1484.69 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002504 | 156.50 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.024015 | 1500.94 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002404 | 150.25 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000120 | 7.50 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.023073 | 1442.06 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.007404 | 462.75 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000230 | 14.38 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.022683 | 1417.69 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.007054 | 440.88 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000130 | 8.12 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.023543 | 1471.44 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.005400 | 337.50 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000552 | 34.50 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.023363 | 1460.19 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.004569 | 285.56 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000130 | 8.12 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.021170 | 1323.12 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003397 | 212.31 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000110 | 6.88 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.021209 | 1325.56 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003246 | 202.88 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 18.2x faster than the slowest successful cell (`pytorch-cpu`, 24.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 24.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 24.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 11.561 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 14.4x faster than the slowest successful cell (`pytorch-cpu`, 11.561 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 19.0x faster than the slowest successful cell (`pytorch-cpu`, 23.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 23.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 23.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 13.6x faster than the slowest successful cell (`pytorch-cpu`, 10.832 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 13.4x faster than the slowest successful cell (`pytorch-cpu`, 10.832 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 12.5x faster than the slowest successful cell (`julia-base`, 8.642 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 16.2x faster than the slowest successful cell (`julia-base`, 8.642 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `jax-cpu` is 12.1x faster than the slowest successful cell (`julia-base`, 8.836 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 26.1x faster than the slowest successful cell (`julia-base`, 8.836 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 28.0x faster than the slowest successful cell (`julia-base`, 8.836 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-trace` is 25.0x faster than the slowest successful cell (`julia-base`, 8.836 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 3434.3x faster than the slowest successful cell (`julia-base`, 9.740 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `pytorch-cpu` is 11.1x faster than the slowest successful cell (`julia-base`, 9.845 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-direct` is 13.1x faster than the slowest successful cell (`julia-base`, 9.845 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 2373.4x faster than the slowest successful cell (`julia-base`, 9.845 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 13.3x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.043 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 12.1x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x8x8`): `tenferro-direct` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`3x2x2`): `tenferro-direct` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.011 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`3x4x4`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x2x2`): `tenferro-direct` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x4x4`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 19.5x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.031 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 11.1x faster than the slowest successful cell (`jax-cpu`, 0.043 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.022 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 11.7x faster than the slowest successful cell (`jax-cpu`, 0.026 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`3x2x2`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.024 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x2x2,rhs=1`): `tenferro-direct` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.020 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x4x4,rhs=1`): `tenferro-direct` is 10.0x faster than the slowest successful cell (`pytorch-cpu`, 0.018 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`3x2x2,rhs=1`): `tenferro-direct` is 11.3x faster than the slowest successful cell (`jax-cpu`, 0.023 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `pytorch-cpu` is 16.2x faster than the slowest successful cell (`jax-cpu`, 20.663 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 15.1x faster than the slowest successful cell (`jax-cpu`, 20.663 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 17.1x faster than the slowest successful cell (`jax-cpu`, 20.663 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 12.7x faster than the slowest successful cell (`tenferro-eager`, 29.718 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 18.0x faster than the slowest successful cell (`tenferro-eager`, 22.561 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 52161.0x faster than the slowest successful cell (`julia-base`, 140.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 33864.3x faster than the slowest successful cell (`julia-base`, 138.437 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 60657.2x faster than the slowest successful cell (`julia-base`, 162.865 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 39001.3x faster than the slowest successful cell (`julia-base`, 161.387 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 84515.9x faster than the slowest successful cell (`julia-base`, 215.938 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 57299.9x faster than the slowest successful cell (`julia-base`, 218.198 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 10.0x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 180.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 101.3x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 183.9x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 42.9x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 165.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 189.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 5912.9 GB/s at 0.003 ms. The conservative physical lower bound is 0.168 ms, so the cell is 59.1x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 4042.7 GB/s at 0.004 ms. The conservative physical lower bound is 0.042 ms, so the cell is 10.1x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 99976.0 GB/s at 0.003 ms. The conservative physical lower bound is 2.684 ms, so the cell is 999.8x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 65664.3 GB/s at 0.004 ms. The conservative physical lower bound is 0.671 ms, so the cell is 164.2x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 99976.0 GB/s at 0.003 ms. The conservative physical lower bound is 2.684 ms, so the cell is 999.8x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 64870.8 GB/s at 0.004 ms. The conservative physical lower bound is 0.671 ms, so the cell is 162.2x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 52531.4 GB/s at 0.003 ms. The conservative physical lower bound is 1.342 ms, so the cell is 525.3x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 35246.3 GB/s at 0.004 ms. The conservative physical lower bound is 0.336 ms, so the cell is 88.1x faster than the bound.

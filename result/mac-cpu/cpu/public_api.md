# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20261127_182201/run.yaml`
- Timestamp: `2026-11-27T18:22:01.699676+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20261127_182201`.

- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

## CPU Information

- Model: `Apple M1 (Virtual)`
- Vendor: `Apple`
- Logical CPUs: `3`
- Physical CPUs: `3`
- Sockets: `1`
- Cores per socket: `3`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Standard: 3 physical / 3 logical (L1i 192 KiB, L1d 128 KiB, L2 12 MiB, 3 CPUs/L2)`
- Python platform: `macOS-26.7.2-arm64-arm-64bit-Mach-O`
- Container: `none`

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

- CSV: `data/results/mac-cpu/cpu/public_api/20261127_182201/cpu_public_api_t1_20261127_182201.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20261127_182201/cpu_public_api_t4_20261127_182201.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 5.030 ± 0.880 | - | 4.887 ± 0.366 | 3.894 ± 0.034 | 1.540 ± 0.029 | 4.340 ± 3.211 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 3.650 ± 0.155 | - | 3.820 ± 0.482 | 4.069 ± 0.138 | 1.491 ± 0.069 | 71.264 ± 18.690 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 12.453 ± 1.543 | - | 12.092 ± 3.015 | 12.118 ± 0.138 | 12.357 ± 0.821 | 12.403 ± 2.054 | 11.872 ± 21.439 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 11.472 ± 0.499 | - | 11.869 ± 1.387 | 11.476 ± 1.363 | 10.306 ± 1.661 | 11.367 ± 2.391 | 13.290 ± 3.013 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 46.572 ± 1.922 | - | 48.334 ± 4.612 | 19.758 ± 0.485 | 35.593 ± 1.011 | 46.522 ± 0.670 | 52.196 ± 7.326 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 16.554 ± 1.440 | - | 20.566 ± 5.830 | 10.356 ± 1.401 | 17.931 ± 2.175 | 54.120 ± 10.087 | 23.458 ± 3.912 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 84.239 ± 8.348 | - | 84.920 ± 5.200 | 10.545 ± 0.419 | 47.353 ± 0.713 | 63.262 ± 7.192 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 27.057 ± 1.224 | - | 27.189 ± 2.110 | 10.011 ± 0.256 | 27.140 ± 2.381 | 39.884 ± 12.471 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 70.362 ± 2.674 | - | unsupported | 10.925 ± 0.265 | 54.626 ± 3.094 | 61.894 ± 14.473 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 25.804 ± 0.911 | - | unsupported | 11.203 ± 0.275 | 22.054 ± 3.059 | 28.320 ± 19.401 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 7.176 ± 1.272 | - | 7.204 ± 0.202 | 11.089 ± 0.071 | 19.819 ± 0.486 | 8.619 ± 0.241 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 9.674 ± 0.844 | - | 9.874 ± 0.501 | 10.878 ± 0.776 | 22.145 ± 1.698 | 441.473 ± 14.983 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 35.410 ± 1.510 | - | 34.524 ± 0.872 | 36.066 ± 1.161 | 42.475 ± 0.436 | 47.302 ± 19.467 | 46.664 ± 0.980 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 16.974 ± 2.011 | - | 17.180 ± 1.281 | 15.644 ± 1.465 | 21.505 ± 0.562 | 47.568 ± 8.209 | 21.909 ± 7.022 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 49.914 ± 7.777 | - | 54.339 ± 2.003 | 55.788 ± 0.587 | 89.419 ± 0.914 | 70.331 ± 1.344 | 87.635 ± 6.590 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 21.442 ± 2.423 | - | 21.489 ± 4.252 | 19.962 ± 2.245 | 36.023 ± 4.967 | 79.972 ± 11.637 | 48.468 ± 21.559 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 9.220 ± 0.450 | - | 9.253 ± 0.363 | 9.405 ± 0.200 | 8.884 ± 0.419 | 8.925 ± 0.491 | 9.394 ± 0.493 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 9.302 ± 1.261 | - | 9.781 ± 1.427 | 8.949 ± 0.413 | 8.051 ± 0.171 | 12.241 ± 2.103 | 10.430 ± 0.576 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.268 ± 0.064 | - | 7.425 ± 0.924 | 6.084 ± 0.378 | 4.544 ± 0.153 | 8.683 ± 0.299 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 1.011 ± 0.017 | - | 3.713 ± 0.049 | 3.847 ± 0.970 | 3.276 ± 0.152 | 11.513 ± 1.726 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 9.400 ± 0.043 | - | 10.168 ± 0.321 | 7.380 ± 0.341 | 6.614 ± 0.045 | 10.512 ± 1.380 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 6.478 ± 1.438 | - | 6.648 ± 0.233 | 7.574 ± 0.498 | 7.808 ± 0.373 | 151.582 ± 28.461 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 6.581 ± 0.173 | - | 6.481 ± 0.199 | 5.229 ± 0.255 | 2.625 ± 0.371 | 6.565 ± 0.246 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 5.393 ± 0.191 | - | 5.444 ± 0.314 | 6.054 ± 3.090 | 2.520 ± 0.091 | 286.310 ± 35.952 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 9.174 ± 1.252 | - | 8.358 ± 0.199 | 7.286 ± 0.095 | 6.377 ± 0.043 | 7.754 ± 0.115 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 7.929 ± 0.176 | - | 8.667 ± 0.429 | 7.457 ± 0.204 | 6.440 ± 0.074 | 642.758 ± 619.838 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 69.581 ± 1.294 | - | 67.974 ± 1.748 | 10.559 ± 0.306 | 54.529 ± 2.112 | 58.892 ± 1.958 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 24.448 ± 2.313 | - | 25.501 ± 2.854 | 10.873 ± 0.222 | 19.881 ± 1.156 | 24.844 ± 1.657 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 50.449 ± 2.167 | - | unsupported | 10.965 ± 0.198 | 53.747 ± 3.323 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 19.811 ± 0.716 | - | unsupported | 11.929 ± 0.889 | 23.918 ± 5.511 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 12.167 ± 2.458 | - | 13.079 ± 5.655 | 11.235 ± 0.542 | 13.707 ± 0.931 | 12.003 ± 1.572 | 17.808 ± 12.582 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 11.812 ± 1.198 | - | 11.594 ± 0.097 | 11.788 ± 1.073 | 10.101 ± 0.084 | 16.715 ± 54.045 | 13.104 ± 2.211 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 31.374 ± 9.731 | - | 17.433 ± 2.852 | 18.952 ± 1.875 | 25.002 ± 1.544 | 19.808 ± 12.544 | 20.833 ± 12.267 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 17.173 ± 1.598 | - | 17.962 ± 2.007 | 18.007 ± 0.994 | 21.860 ± 2.206 | 18.415 ± 10.798 | 22.222 ± 0.615 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 27.883 ± 4.127 | - | 35.267 ± 1.864 | 38.818 ± 1.034 | 33.911 ± 4.098 | 48.693 ± 3.329 | 53.986 ± 6.752 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 14.241 ± 0.275 | - | 20.814 ± 3.754 | 19.309 ± 0.306 | 14.492 ± 0.173 | 48.484 ± 0.488 | 24.163 ± 4.552 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 5.907 ± 0.103 | - | 6.319 ± 0.373 | 6.007 ± 0.201 | 8.280 ± 0.094 | 6.649 ± 1.144 | 6.770 ± 2.549 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 7.043 ± 0.986 | - | 6.623 ± 0.675 | 6.441 ± 0.286 | 7.157 ± 0.387 | 6.710 ± 1.601 | 12.236 ± 12.107 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 12.329 ± 0.449 | - | 14.007 ± 1.146 | 15.531 ± 0.467 | 14.983 ± 0.203 | 18.923 ± 5.405 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 11.687 ± 0.136 | - | 12.723 ± 2.634 | 12.517 ± 0.276 | 16.034 ± 1.002 | 18.163 ± 7.277 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 31.034 ± 2.196 | - | 31.086 ± 0.936 | 33.400 ± 3.582 | 29.003 ± 0.311 | 36.462 ± 4.108 | 37.209 ± 1.590 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 18.373 ± 1.634 | - | 18.453 ± 4.922 | 18.781 ± 8.328 | 14.166 ± 0.563 | 36.153 ± 3.343 | 18.296 ± 5.011 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 19.719 ± 8.974 | - | 21.656 ± 2.570 | 20.911 ± 1.757 | 23.906 ± 0.355 | 18.155 ± 2.343 | 19.587 ± 12.957 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 17.070 ± 0.722 | - | 17.198 ± 1.130 | 17.968 ± 1.791 | 22.551 ± 0.288 | 26.773 ± 19.160 | 21.268 ± 3.704 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 19.434 ± 4.124 | - | 17.212 ± 1.454 | 26.721 ± 2.482 | 17.800 ± 1.560 | 34.781 ± 51.708 | 34.306 ± 1.364 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 11.784 ± 6.760 | - | 11.607 ± 5.095 | 13.408 ± 2.061 | 8.581 ± 1.044 | 33.149 ± 1.946 | 24.456 ± 11.310 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 28.371 ± 4.149 | - | 27.909 ± 1.557 | 28.263 ± 0.976 | 18.374 ± 5.643 | 25.112 ± 1.464 | 25.485 ± 1.014 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 13.105 ± 1.040 | - | 11.939 ± 0.287 | 15.309 ± 6.126 | 8.310 ± 0.043 | 24.871 ± 2.940 | 13.181 ± 2.087 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 31.913 ± 2.600 | - | 31.306 ± 0.455 | 41.888 ± 1.243 | 27.747 ± 0.559 | 42.378 ± 0.678 | 43.262 ± 1.437 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 22.444 ± 9.507 | - | 21.954 ± 6.341 | 25.798 ± 6.618 | 13.045 ± 7.461 | 43.213 ± 2.940 | 20.806 ± 3.057 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 16.265 ± 1.051 | - | 19.407 ± 6.658 | 23.122 ± 1.166 | 23.973 ± 7.541 | 22.101 ± 0.793 | 22.530 ± 0.477 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 7.803 ± 2.459 | - | 7.667 ± 0.079 | 11.337 ± 1.118 | 8.896 ± 0.098 | 21.890 ± 0.149 | 12.251 ± 2.059 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.254 ± 0.011 | 0.267 ± 0.004 | unsupported | unsupported | 0.264 ± 0.007 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.270 ± 0.032 | 0.272 ± 0.020 | unsupported | unsupported | 0.222 ± 0.264 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 17.762 ± 1.414 | - | 17.201 ± 3.178 | 17.386 ± 0.841 | 24.504 ± 1.672 | 27.640 ± 21.381 | 19.523 ± 8.655 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 17.643 ± 0.668 | - | 16.930 ± 1.215 | 17.554 ± 1.189 | 22.048 ± 0.633 | 85.205 ± 200.817 | 22.354 ± 34.273 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 18.497 ± 0.956 | - | 17.772 ± 0.411 | 17.186 ± 0.783 | 23.959 ± 0.347 | 18.627 ± 2.306 | 19.000 ± 3.351 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 16.821 ± 0.270 | - | 16.881 ± 0.681 | 17.613 ± 0.864 | 22.029 ± 0.331 | 19.960 ± 6.782 | 23.551 ± 0.833 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 17.310 ± 0.741 | - | 20.949 ± 9.048 | 20.291 ± 4.428 | 23.783 ± 0.460 | 17.601 ± 2.195 | 19.751 ± 9.413 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 17.114 ± 1.243 | - | 17.490 ± 0.405 | 18.111 ± 1.586 | 22.038 ± 0.360 | 17.482 ± 0.846 | 21.267 ± 6.627 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 13.115 ± 1.604 | - | 11.869 ± 0.464 | 11.166 ± 0.418 | 14.589 ± 1.153 | 10.968 ± 1.240 | 14.508 ± 17.487 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 11.334 ± 0.354 | - | 11.567 ± 0.331 | 10.817 ± 1.855 | 10.302 ± 3.715 | 19.933 ± 7.799 | 14.040 ± 1.748 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 36.254 ± 0.841 | - | 35.360 ± 1.283 | 68.356 ± 1.490 | 44.719 ± 0.220 | 94.577 ± 19.108 | 74.501 ± 0.902 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 16.375 ± 2.560 | - | 16.154 ± 1.487 | 30.727 ± 1.335 | 18.683 ± 4.412 | 86.563 ± 25.277 | 35.735 ± 8.828 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 7.003 ± 0.042 | - | 6.913 ± 0.127 | 11.803 ± 0.124 | 22.242 ± 0.525 | 6.041 ± 0.068 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 5.697 ± 0.322 | - | 5.690 ± 0.280 | 7.165 ± 0.401 | 11.553 ± 0.655 | 7.209 ± 1.327 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 0.860 ± 0.015 | - | 0.861 ± 0.011 | 14.577 ± 0.544 | 2.247 ± 0.140 | 0.748 ± 0.021 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.782 ± 0.012 | - | 0.777 ± 0.022 | 10.798 ± 0.447 | 2.143 ± 0.093 | 0.716 ± 0.054 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 0.838 ± 0.067 | - | 0.791 ± 0.034 | 1.575 ± 0.029 | 0.977 ± 0.033 | 0.816 ± 0.047 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 1.935 ± 0.433 | - | 2.074 ± 0.110 | 1.109 ± 0.051 | 1.040 ± 0.025 | 0.849 ± 0.107 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 6.894 ± 0.110 | - | 6.988 ± 0.321 | 12.213 ± 1.406 | 22.494 ± 0.575 | 5.701 ± 0.447 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 5.696 ± 0.265 | - | 6.961 ± 2.954 | 6.917 ± 1.167 | 11.774 ± 1.823 | 6.513 ± 1.149 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 0.861 ± 0.013 | - | 0.858 ± 0.034 | 16.525 ± 2.755 | 2.201 ± 0.079 | 0.709 ± 0.017 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.805 ± 0.107 | - | 0.867 ± 0.144 | 10.748 ± 0.415 | 2.083 ± 0.079 | 0.738 ± 0.177 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 3.868 ± 0.146 | - | 3.219 ± 0.158 | 6.155 ± 0.073 | 3.392 ± 0.064 | 3.344 ± 0.217 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 5.826 ± 0.932 | - | 6.175 ± 0.716 | 3.536 ± 0.075 | 3.429 ± 0.083 | 3.252 ± 0.068 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 6.362 ± 0.155 | - | 6.625 ± 0.459 | 6.318 ± 0.205 | 40.828 ± 1.215 | 6.017 ± 0.525 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 5.369 ± 0.232 | - | 6.300 ± 0.198 | 5.741 ± 0.099 | 20.577 ± 1.704 | 9.276 ± 3.102 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.759 ± 0.033 | - | 0.742 ± 0.020 | 4.901 ± 3.023 | 2.262 ± 0.022 | 0.742 ± 0.042 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.791 ± 0.344 | - | 0.789 ± 0.096 | 2.438 ± 0.092 | 1.525 ± 0.047 | 0.779 ± 0.052 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 0.758 ± 0.021 | - | 0.853 ± 0.128 | 0.866 ± 0.151 | 1.215 ± 0.055 | 0.874 ± 0.207 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 1.899 ± 0.218 | - | 1.899 ± 0.101 | 0.783 ± 0.065 | 0.865 ± 0.065 | 0.754 ± 0.028 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 5.883 ± 0.193 | - | 5.722 ± 0.094 | 5.612 ± 0.152 | 30.916 ± 0.392 | 6.122 ± 0.064 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 7.261 ± 2.202 | - | 5.787 ± 0.425 | 5.542 ± 0.083 | 15.871 ± 0.414 | 7.327 ± 0.469 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.633 ± 0.024 | - | 0.656 ± 0.038 | 4.540 ± 0.557 | 2.222 ± 0.039 | 0.717 ± 0.039 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.938 ± 0.228 | - | 0.830 ± 0.132 | 2.562 ± 0.218 | 2.051 ± 0.123 | 0.686 ± 0.024 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 0.739 ± 0.013 | - | 0.773 ± 0.056 | 0.795 ± 0.102 | 1.025 ± 0.077 | 0.781 ± 0.040 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 1.897 ± 0.141 | - | 2.725 ± 1.589 | 0.768 ± 0.045 | 1.087 ± 0.159 | 0.776 ± 0.039 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.794 ± 0.061 | 0.743 ± 0.037 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.693 ± 0.035 | 0.736 ± 0.063 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 28.957 ± 3.136 | - | 27.457 ± 1.429 | 34.431 ± 3.835 | 39.743 ± 2.966 | 78.565 ± 14.917 | 33.946 ± 3.340 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 19.367 ± 4.132 | - | 22.945 ± 2.914 | 14.992 ± 1.320 | 28.266 ± 9.975 | 52.733 ± 49.690 | 14.679 ± 2.097 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 19.048 ± 0.998 | - | 18.396 ± 0.884 | 18.907 ± 1.245 | 36.043 ± 0.700 | 44.925 ± 18.878 | 44.227 ± 5.184 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 12.975 ± 1.810 | - | 13.491 ± 3.781 | 12.792 ± 0.585 | 18.250 ± 3.358 | 41.674 ± 1.278 | 21.146 ± 7.002 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 25.603 ± 1.470 | - | 25.154 ± 0.718 | 19.084 ± 0.236 | 24.916 ± 2.439 | 25.898 ± 2.725 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 23.597 ± 0.968 | - | 23.069 ± 0.611 | 19.972 ± 2.398 | 28.456 ± 2.876 | 23.658 ± 0.490 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 12.223 ± 2.694 | - | 12.539 ± 1.499 | 11.365 ± 0.538 | 15.229 ± 0.357 | 16.238 ± 1.421 | 20.974 ± 32.155 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 12.597 ± 0.540 | - | 11.542 ± 0.224 | 11.665 ± 0.477 | 10.196 ± 0.370 | 19.182 ± 16.278 | 15.558 ± 3.785 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 24.648 ± 0.167 | - | 24.433 ± 2.455 | 31.421 ± 2.981 | 26.330 ± 0.415 | 31.030 ± 3.094 | 32.104 ± 1.802 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 14.407 ± 3.939 | - | 16.696 ± 2.659 | 14.912 ± 6.037 | 12.958 ± 2.669 | 30.966 ± 1.318 | 14.545 ± 3.603 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 12.023 ± 0.237 | - | 12.310 ± 0.707 | 12.682 ± 0.363 | 20.485 ± 1.092 | 45.051 ± 54.993 | 32.572 ± 2.573 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 11.778 ± 0.122 | - | 12.840 ± 1.050 | 11.843 ± 1.579 | 12.961 ± 2.830 | 29.441 ± 9.414 | 12.990 ± 1.501 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 17.546 ± 4.944 | - | 17.129 ± 2.183 | 17.603 ± 0.461 | 24.072 ± 0.277 | 19.208 ± 1.486 | 25.111 ± 3.143 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 18.980 ± 2.478 | - | 17.267 ± 0.437 | 17.780 ± 1.335 | 22.250 ± 0.596 | 20.315 ± 21.634 | 20.269 ± 1.474 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 66.475 ± 2.223 | - | 66.107 ± 1.415 | 68.397 ± 2.256 | 13.836 ± 0.252 | 44.076 ± 1.722 | 47.665 ± 2.521 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 40.003 ± 15.160 | - | 37.169 ± 10.064 | 29.146 ± 7.004 | 6.790 ± 0.139 | 43.144 ± 0.589 | 24.410 ± 6.507 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 26.408 ± 2.678 | 24.784 ± 0.697 | unsupported | 19.759 ± 1.333 | 24.675 ± 0.805 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 23.578 ± 0.771 | 23.377 ± 0.470 | unsupported | 19.347 ± 1.617 | 30.861 ± 4.741 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.620 ± 0.009 | - | 0.650 ± 0.018 | 0.654 ± 0.031 | 0.627 ± 0.019 | 0.795 ± 0.017 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.635 ± 0.045 | - | 0.642 ± 0.043 | 0.761 ± 0.165 | 0.699 ± 0.026 | 0.671 ± 0.011 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.627 ± 0.019 | - | 0.621 ± 0.055 | 0.719 ± 0.019 | 0.731 ± 0.020 | 0.925 ± 0.183 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.642 ± 0.045 | - | 0.674 ± 0.066 | 0.894 ± 0.494 | 0.850 ± 0.023 | 0.740 ± 0.042 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 0.916 ± 0.019 | - | unsupported | 1.099 ± 0.773 | 1.133 ± 0.060 | 1.183 ± 4.774 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 1.016 ± 0.014 | - | unsupported | 1.427 ± 0.601 | 1.993 ± 0.379 | 0.955 ± 2.426 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.335 ± 0.004 | - | 0.397 ± 0.049 | 0.353 ± 0.045 | 0.334 ± 0.021 | 1.357 ± 2.599 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.278 ± 0.052 | - | 0.398 ± 0.039 | 0.298 ± 0.013 | 0.238 ± 0.021 | 0.385 ± 1.552 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.601 ± 0.015 | - | 0.666 ± 0.066 | 1.065 ± 0.066 | 0.750 ± 0.008 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.612 ± 0.027 | - | 0.647 ± 0.034 | 1.269 ± 0.113 | 1.152 ± 0.473 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.637 ± 0.045 | - | 0.663 ± 0.047 | 0.849 ± 0.104 | 0.693 ± 0.045 | 1.208 ± 9.241 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.639 ± 0.012 | - | 0.602 ± 0.110 | 0.718 ± 0.057 | 1.372 ± 0.515 | 1.280 ± 2.269 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.555 ± 0.020 | - | 0.636 ± 0.030 | 0.443 ± 0.080 | 0.393 ± 0.015 | 0.430 ± 2.824 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.742 ± 0.019 | - | 0.915 ± 0.071 | 0.550 ± 0.026 | 0.461 ± 0.011 | 1.356 ± 4.104 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.138 ± 0.069 | - | 0.005 ± 0.001 | 1.289 ± 0.709 | 1.302 ± 0.237 | 1.823 ± 0.375 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 1.188 ± 0.240 | - | 0.001 ± 0.000 | 1.582 ± 0.409 | 1.473 ± 0.341 | 1.452 ± 0.061 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.050 ± 0.011 | - | 0.049 ± 0.000 | 0.141 ± 0.001 | 0.095 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.098 ± 0.001 | - | 0.101 ± 0.000 | 0.189 ± 0.001 | 0.132 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.217 ± 0.002 | - | 0.219 ± 0.002 | 0.297 ± 0.007 | 0.233 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.049 ± 0.000 | - | 0.050 ± 0.002 | 0.040 ± 0.004 | 0.087 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.070 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.088 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.381 ± 0.003 | - | 0.382 ± 0.006 | 0.245 ± 0.001 | 0.282 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.018 ± 0.000 | - | 0.020 ± 0.000 | 0.017 ± 0.002 | 0.052 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.054 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.048 ± 0.004 | - | 0.056 ± 0.011 | 0.174 ± 0.066 | 0.111 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.131 ± 0.048 | - | 0.123 ± 0.026 | 0.186 ± 0.032 | 0.145 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.222 ± 0.009 | - | 0.222 ± 0.018 | 0.336 ± 0.048 | 0.249 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.050 ± 0.000 | - | 0.051 ± 0.000 | 0.100 ± 0.023 | 0.063 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.079 ± 0.014 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.099 ± 0.079 | 0.057 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.054 ± 0.093 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.001 ± 0.000 | - | 0.004 ± 0.003 | 0.049 ± 0.032 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.001 ± 0.000 | - | 0.006 ± 0.000 | 0.082 ± 0.028 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.073 ± 0.024 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.002 ± 0.003 | - | 0.003 ± 0.000 | 0.057 ± 0.031 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.002 ± 0.000 | 0.047 ± 0.037 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.040 ± 0.054 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.003 | 0.061 ± 0.026 | 0.049 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.305 ± 0.074 | - | 0.247 ± 0.011 | 0.278 ± 0.074 | 0.265 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.003 ± 0.000 | 0.040 ± 0.057 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.018 ± 0.000 | - | 0.020 ± 0.005 | 0.092 ± 0.050 | 0.068 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.063 ± 0.038 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.006 ± 0.003 | - | 0.004 ± 0.002 | 0.054 ± 0.032 | 0.058 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.304 ± 0.003 | - | 0.307 ± 0.001 | 0.433 ± 0.010 | 1.962 ± 1.108 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.465 ± 0.025 | - | 1.472 ± 0.059 | 1.710 ± 0.011 | 2.354 ± 0.528 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 4.997 ± 0.202 | - | 4.836 ± 0.056 | 6.485 ± 0.012 | 9.672 ± 2.142 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 2.378 ± 0.068 | - | 2.379 ± 0.030 | 3.300 ± 0.026 | 4.012 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.076 ± 0.001 | - | 0.078 ± 0.000 | 0.097 ± 0.001 | 0.153 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.001 | 0.006 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.002 | 0.031 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.002 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.015 ± 0.000 | - | 0.016 ± 0.000 | 0.023 ± 0.000 | 0.069 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.020 ± 0.000 | - | 0.021 ± 0.000 | 0.029 ± 0.006 | 0.173 ± 0.099 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 19.002 ± 0.206 | - | 19.218 ± 0.927 | 26.087 ± 0.921 | 31.913 ± 1.565 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.608 ± 0.026 | - | 0.616 ± 0.027 | 0.740 ± 0.014 | 0.885 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.013 ± 0.000 | - | 0.014 ± 0.000 | 0.018 ± 0.000 | 0.028 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.038 ± 0.000 | - | 0.040 ± 0.000 | 0.054 ± 0.000 | 0.282 ± 0.106 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.342 ± 0.031 | - | 0.372 ± 0.040 | 0.580 ± 0.368 | 1.112 ± 0.351 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 0.850 ± 0.041 | - | 0.870 ± 0.511 | 1.763 ± 0.119 | 1.144 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 2.647 ± 0.077 | - | 2.611 ± 0.250 | 6.843 ± 0.277 | 3.707 ± 0.095 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 2.363 ± 0.192 | - | 2.706 ± 0.189 | 3.473 ± 0.600 | 3.840 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.076 ± 0.001 | - | 0.079 ± 0.001 | 0.093 ± 0.010 | 0.166 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.007 ± 0.001 | - | 0.008 ± 0.002 | 0.009 ± 0.000 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.019 ± 0.002 | - | 0.019 ± 0.004 | 0.023 ± 0.000 | 0.085 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.001 | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.007 ± 0.002 | - | 0.011 ± 0.002 | 0.011 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.021 ± 0.005 | - | 0.026 ± 0.002 | 0.030 ± 0.004 | 0.095 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 10.469 ± 1.485 | - | 10.440 ± 1.697 | 26.656 ± 1.582 | 15.964 ± 0.077 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.004 ± 0.001 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.348 ± 0.006 | - | 0.352 ± 0.003 | 0.728 ± 0.050 | 0.493 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.016 ± 0.003 | - | 0.014 ± 0.000 | 0.018 ± 0.000 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.048 ± 0.006 | - | 0.044 ± 0.008 | 0.056 ± 0.005 | 0.118 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.288 ± 0.003 | - | 0.293 ± 0.003 | 0.238 ± 0.000 | 0.542 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.246 ± 0.012 | - | 1.273 ± 0.031 | 1.246 ± 0.023 | 1.525 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 4.197 ± 0.027 | - | 4.226 ± 0.040 | 4.942 ± 0.291 | 5.446 ± 0.343 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.375 ± 0.021 | - | 1.383 ± 0.050 | 1.004 ± 0.033 | 1.121 ± 0.074 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.061 ± 0.001 | - | 0.062 ± 0.000 | 0.073 ± 0.002 | 0.109 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.019 ± 0.004 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.017 ± 0.000 | - | 0.018 ± 0.000 | 0.023 ± 0.000 | 0.031 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 11.274 ± 0.596 | - | 11.017 ± 0.045 | 7.929 ± 0.079 | 8.339 ± 0.167 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.484 ± 0.003 | - | 0.488 ± 0.021 | 0.551 ± 0.007 | 0.607 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.013 ± 0.002 | - | 0.012 ± 0.000 | 0.014 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.034 ± 0.000 | - | 0.035 ± 0.000 | 0.042 ± 0.001 | 0.090 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.287 ± 0.006 | - | 0.293 ± 0.007 | 0.385 ± 0.109 | 0.597 ± 0.062 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.254 ± 0.035 | - | 1.388 ± 0.095 | 1.315 ± 0.142 | 0.974 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 2.294 ± 0.120 | - | 2.433 ± 0.190 | 5.311 ± 0.721 | 3.559 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.518 ± 0.079 | - | 1.645 ± 0.118 | 0.949 ± 0.011 | 1.163 ± 0.146 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.010 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.062 ± 0.010 | - | 0.065 ± 0.014 | 0.066 ± 0.001 | 0.163 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.013 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.002 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.001 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.029 ± 0.000 | - | 0.015 ± 0.000 | 0.022 ± 0.004 | 0.074 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.018 ± 0.004 | - | 0.019 ± 0.000 | 0.024 ± 0.005 | 0.031 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 5.545 ± 0.352 | - | 5.670 ± 1.354 | 7.675 ± 0.107 | 5.166 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.479 ± 0.225 | - | 0.507 ± 0.204 | 0.511 ± 0.013 | 0.456 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.011 ± 0.003 | - | 0.015 ± 0.003 | 0.014 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.034 ± 0.000 | - | 0.040 ± 0.006 | 0.043 ± 0.001 | 0.140 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.219 ± 0.013 | - | 0.217 ± 0.003 | 0.126 ± 0.005 | 0.761 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 1.151 ± 0.027 | - | 1.193 ± 0.726 | 0.742 ± 0.030 | 1.519 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 3.778 ± 0.064 | - | 3.912 ± 0.282 | 2.405 ± 0.048 | 5.434 ± 0.188 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.799 ± 0.045 | - | 0.784 ± 0.025 | 0.477 ± 0.029 | 1.073 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.052 ± 0.006 | - | 0.053 ± 0.007 | 0.030 ± 0.001 | 0.111 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.012 ± 0.000 | - | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.023 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.015 ± 0.000 | - | 0.017 ± 0.000 | 0.014 ± 0.000 | 0.029 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 6.188 ± 0.267 | - | 6.371 ± 0.212 | 3.647 ± 0.128 | 8.407 ± 0.251 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.394 ± 0.031 | - | 0.406 ± 0.016 | 0.208 ± 0.027 | 0.600 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.030 ± 0.000 | - | 0.031 ± 0.002 | 0.023 ± 0.000 | 0.089 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.219 ± 0.004 | - | 0.219 ± 0.012 | 0.232 ± 0.016 | 0.549 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 1.160 ± 0.025 | - | 1.178 ± 0.149 | 0.793 ± 0.097 | 1.006 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 1.951 ± 0.054 | - | 1.960 ± 0.255 | 2.444 ± 0.077 | 2.795 ± 0.105 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.818 ± 0.058 | - | 0.799 ± 0.022 | 0.463 ± 0.025 | 1.054 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.013 ± 0.010 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.047 ± 0.004 | - | 0.048 ± 0.003 | 0.029 ± 0.001 | 0.128 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.022 ± 0.007 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.006 ± 0.001 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.022 ± 0.007 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.006 | 0.014 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.012 ± 0.000 | - | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.006 | 0.010 ± 0.012 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.005 ± 0.001 | - | 0.008 ± 0.002 | 0.008 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.015 ± 0.000 | - | 0.017 ± 0.000 | 0.014 ± 0.000 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 3.166 ± 0.104 | - | 3.256 ± 0.069 | 3.665 ± 0.095 | 4.324 ± 0.067 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.008 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.348 ± 0.141 | - | 0.290 ± 0.125 | 0.194 ± 0.002 | 0.367 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.012 ± 0.002 | - | 0.013 ± 0.001 | 0.022 ± 0.008 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.030 ± 0.000 | - | 0.031 ± 0.003 | 0.023 ± 0.000 | 0.100 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.071 ± 0.001 | - | 0.072 ± 0.001 | 0.180 ± 0.011 | 0.145 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.163 ± 0.007 | - | 0.170 ± 0.014 | 0.267 ± 0.013 | 0.219 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.443 ± 0.020 | - | 0.434 ± 0.005 | 0.503 ± 0.013 | 0.424 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.130 ± 0.001 | - | 0.132 ± 0.011 | 0.069 ± 0.001 | 0.136 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.001 | 0.059 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.011 ± 0.000 | - | 0.012 ± 0.000 | 0.013 ± 0.000 | 0.053 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.002 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.073 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.064 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 1.036 ± 0.026 | - | 1.031 ± 0.032 | 0.471 ± 0.009 | 0.603 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.054 ± 0.002 | - | 0.055 ± 0.000 | 0.039 ± 0.003 | 0.095 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.062 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.068 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.080 ± 0.015 | - | 0.081 ± 0.013 | 0.247 ± 0.003 | 0.133 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.188 ± 0.015 | - | 0.181 ± 0.016 | 0.278 ± 0.005 | 0.218 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.488 ± 0.029 | - | 0.459 ± 0.035 | 0.560 ± 0.054 | 0.636 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.189 ± 0.105 | - | 0.188 ± 0.026 | 0.131 ± 0.007 | 0.177 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.061 ± 0.016 | 0.012 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.010 ± 0.002 | - | 0.009 ± 0.000 | 0.068 ± 0.014 | 0.094 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.050 ± 0.029 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.011 ± 0.003 | - | 0.012 ± 0.000 | 0.070 ± 0.031 | 0.077 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.047 ± 0.011 | 0.022 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.074 ± 0.013 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.003 ± 0.001 | 0.060 ± 0.026 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.004 ± 0.000 | - | 0.004 ± 0.000 | 0.051 ± 0.011 | 0.065 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.060 ± 0.008 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.060 ± 0.021 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.004 ± 0.001 | - | 0.005 ± 0.001 | 0.071 ± 0.019 | 0.074 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.717 ± 0.007 | - | 0.718 ± 0.023 | 0.573 ± 0.043 | 0.823 ± 0.059 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.001 | 0.095 ± 0.016 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.055 ± 0.004 | - | 0.056 ± 0.000 | 0.128 ± 0.012 | 0.167 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.004 ± 0.000 | - | 0.004 ± 0.000 | 0.077 ± 0.034 | 0.014 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.085 ± 0.039 | 0.096 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.149 ± 0.001 | - | 0.150 ± 0.000 | 0.312 ± 0.010 | 0.348 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.355 ± 0.007 | - | 0.359 ± 0.014 | 0.655 ± 0.011 | 0.658 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.006 ± 0.022 | - | 1.249 ± 0.563 | 1.553 ± 0.029 | 1.369 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.399 ± 0.002 | - | 0.398 ± 0.001 | 0.341 ± 0.012 | 0.369 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.020 ± 0.001 | - | 0.022 ± 0.004 | 0.019 ± 0.000 | 0.056 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.031 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.023 ± 0.000 | - | 0.024 ± 0.000 | 0.017 ± 0.000 | 0.067 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 3.222 ± 0.025 | - | 3.257 ± 0.245 | 2.789 ± 0.023 | 2.766 ± 0.187 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.154 ± 0.013 | - | 0.153 ± 0.014 | 0.137 ± 0.014 | 0.152 ± 0.327 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.009 ± 0.000 | - | 0.010 ± 0.000 | 0.015 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.148 ± 0.001 | - | 0.150 ± 0.000 | 0.330 ± 0.092 | 0.423 ± 0.075 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.360 ± 0.064 | - | 0.357 ± 0.013 | 0.864 ± 0.139 | 0.769 ± 0.109 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 0.544 ± 0.024 | - | 0.553 ± 0.015 | 1.800 ± 0.114 | 1.441 ± 0.222 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.402 ± 0.013 | - | 0.403 ± 0.019 | 0.419 ± 0.021 | 0.420 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.032 ± 0.013 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.020 ± 0.000 | - | 0.021 ± 0.006 | 0.073 ± 0.029 | 0.046 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.053 ± 0.046 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.023 ± 0.000 | - | 0.025 ± 0.000 | 0.101 ± 0.044 | 0.080 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.048 ± 0.019 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.047 ± 0.052 | 0.024 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.025 ± 0.023 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.004 ± 0.001 | - | 0.005 ± 0.000 | 0.045 ± 0.026 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.033 ± 0.038 | 0.011 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.032 ± 0.013 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.048 ± 0.029 | 0.015 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 1.726 ± 0.016 | - | 1.716 ± 0.013 | 3.123 ± 0.398 | 2.685 ± 0.245 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.036 ± 0.017 | 0.012 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.151 ± 0.001 | - | 0.152 ± 0.001 | 0.195 ± 0.021 | 0.237 ± 0.275 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.004 ± 0.001 | - | 0.005 ± 0.000 | 0.014 ± 0.004 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.009 ± 0.000 | - | 0.010 ± 0.000 | 0.045 ± 0.016 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.099 ± 0.001 | - | 0.083 ± 0.005 | 0.106 ± 0.000 | 0.173 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.149 ± 0.003 | - | 0.137 ± 0.003 | 0.167 ± 0.013 | 0.244 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.345 ± 0.002 | - | 0.344 ± 0.003 | 0.347 ± 0.014 | 0.452 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.089 ± 0.009 | - | 0.088 ± 0.001 | 0.066 ± 0.001 | 0.166 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.014 ± 0.010 | 0.011 ± 0.001 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.014 ± 0.000 | 0.046 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.003 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.013 ± 0.003 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.664 ± 0.025 | - | 0.722 ± 0.032 | 0.410 ± 0.015 | 0.538 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.053 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.035 ± 0.001 | - | 0.040 ± 0.001 | 0.038 ± 0.000 | 0.089 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.014 ± 0.003 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.025 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.100 ± 0.007 | - | 0.080 ± 0.011 | 0.298 ± 0.148 | 0.245 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.150 ± 0.003 | - | 0.135 ± 0.003 | 0.208 ± 0.016 | 0.267 ± 0.078 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.301 ± 0.050 | - | 0.398 ± 0.127 | 0.389 ± 0.019 | 0.474 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.159 ± 0.025 | - | 0.175 ± 0.062 | 0.066 ± 0.005 | 0.135 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.011 ± 0.010 | 0.013 ± 0.017 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.006 ± 0.002 | - | 0.009 ± 0.000 | 0.016 ± 0.004 | 0.087 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.014 ± 0.052 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.002 ± 0.001 | - | 0.005 ± 0.000 | 0.010 ± 0.002 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.003 ± 0.001 | - | 0.006 ± 0.000 | 0.012 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.005 ± 0.005 | - | 0.005 ± 0.000 | 0.010 ± 0.003 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.026 ± 0.014 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.003 ± 0.001 | - | 0.006 ± 0.000 | 0.013 ± 0.003 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.552 ± 0.010 | - | 0.643 ± 0.015 | 0.447 ± 0.086 | 0.547 ± 0.072 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.041 ± 0.026 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.034 ± 0.003 | - | 0.040 ± 0.000 | 0.111 ± 0.020 | 0.127 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.074 ± 0.040 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.065 ± 0.025 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 0.743 ± 0.004 | - | 0.685 ± 0.020 | 1.176 ± 0.024 | 4.368 ± 1.383 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 2.768 ± 0.077 | - | 2.626 ± 0.064 | 4.256 ± 0.021 | 6.375 ± 0.179 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 9.115 ± 0.050 | - | 10.812 ± 3.512 | 14.770 ± 0.350 | 17.594 ± 1.245 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 4.780 ± 0.317 | - | 4.764 ± 0.144 | 4.566 ± 0.095 | 5.589 ± 2.676 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.154 ± 0.016 | - | 0.152 ± 0.013 | 0.201 ± 0.021 | 0.260 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.126 ± 0.014 | - | 0.127 ± 0.001 | 0.174 ± 0.007 | 0.257 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.010 ± 0.000 | - | 0.011 ± 0.000 | 0.022 ± 0.006 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.004 ± 0.001 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.062 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.009 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.000 | 0.035 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.027 ± 0.000 | - | 0.027 ± 0.008 | 0.050 ± 0.000 | 0.063 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.025 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.012 ± 0.000 | - | 0.013 ± 0.000 | 0.023 ± 0.002 | 0.041 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.036 ± 0.003 | - | 0.035 ± 0.000 | 0.066 ± 0.000 | 0.080 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 36.455 ± 0.231 | - | 36.410 ± 0.314 | 36.258 ± 0.248 | 42.012 ± 0.228 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.015 ± 0.000 | 0.093 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.185 ± 0.014 | - | 1.183 ± 0.069 | 1.438 ± 0.018 | 1.751 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.023 ± 0.001 | - | 0.023 ± 0.000 | 0.039 ± 0.003 | 0.061 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.070 ± 0.000 | - | 0.068 ± 0.000 | 0.121 ± 0.006 | 0.458 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 0.824 ± 0.039 | - | 0.745 ± 0.017 | 1.299 ± 0.022 | 2.878 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 1.465 ± 0.152 | - | 1.480 ± 0.071 | 4.641 ± 0.103 | 3.136 ± 0.379 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 4.689 ± 0.286 | - | 5.117 ± 2.511 | 14.826 ± 0.225 | 8.672 ± 0.389 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 5.352 ± 0.815 | - | 5.264 ± 0.790 | 4.666 ± 0.269 | 5.552 ± 0.631 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.143 ± 0.002 | - | 0.147 ± 0.011 | 0.184 ± 0.016 | 0.269 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.126 ± 0.003 | - | 0.131 ± 0.018 | 0.186 ± 0.005 | 0.253 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.013 ± 0.002 | - | 0.014 ± 0.002 | 0.025 ± 0.007 | 0.030 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.002 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.012 ± 0.002 | - | 0.013 ± 0.000 | 0.018 ± 0.006 | 0.030 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.028 ± 0.006 | - | 0.034 ± 0.007 | 0.050 ± 0.005 | 0.063 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.005 ± 0.001 | - | 0.007 ± 0.001 | 0.013 ± 0.003 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.012 ± 0.003 | - | 0.013 ± 0.000 | 0.022 ± 0.004 | 0.036 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.036 ± 0.003 | - | 0.044 ± 0.002 | 0.067 ± 0.011 | 0.080 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 16.621 ± 1.490 | - | 18.431 ± 2.940 | 37.117 ± 0.700 | 21.291 ± 0.318 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.009 ± 0.002 | - | 0.011 ± 0.001 | 0.015 ± 0.003 | 0.035 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 0.642 ± 0.141 | - | 0.704 ± 0.131 | 1.535 ± 0.057 | 0.962 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.023 ± 0.006 | - | 0.023 ± 0.000 | 0.039 ± 0.008 | 0.059 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.077 ± 0.012 | - | 0.075 ± 0.010 | 0.128 ± 0.008 | 0.199 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.520 ± 0.016 | - | 0.530 ± 0.023 | 0.582 ± 0.018 | 2.000 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 2.200 ± 0.031 | - | 2.199 ± 0.051 | 2.701 ± 0.019 | 4.020 ± 0.067 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 7.696 ± 0.773 | - | 7.311 ± 0.069 | 10.518 ± 2.553 | 12.142 ± 0.383 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.910 ± 0.039 | - | 1.930 ± 0.011 | 1.489 ± 0.024 | 1.587 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.001 | 0.019 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.104 ± 0.001 | - | 0.105 ± 0.003 | 0.127 ± 0.001 | 0.167 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.001 | 0.009 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.076 ± 0.000 | - | 0.077 ± 0.000 | 0.102 ± 0.001 | 0.382 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.016 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.014 ± 0.000 | 0.033 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.021 ± 0.000 | - | 0.023 ± 0.000 | 0.033 ± 0.000 | 0.045 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.003 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.010 ± 0.002 | - | 0.011 ± 0.002 | 0.017 ± 0.000 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.034 ± 0.003 | - | 0.029 ± 0.000 | 0.044 ± 0.004 | 0.056 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 15.287 ± 0.251 | - | 15.213 ± 0.051 | 11.806 ± 0.073 | 13.536 ± 1.234 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.006 ± 0.002 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.028 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 0.818 ± 0.029 | - | 0.821 ± 0.012 | 0.968 ± 0.008 | 1.083 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.018 ± 0.000 | - | 0.019 ± 0.000 | 0.027 ± 0.002 | 0.048 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.057 ± 0.008 | - | 0.058 ± 0.007 | 0.083 ± 0.001 | 0.142 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.510 ± 0.038 | - | 0.524 ± 0.034 | 0.572 ± 0.009 | 1.929 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 2.176 ± 0.050 | - | 2.207 ± 0.065 | 2.712 ± 0.102 | 2.442 ± 0.256 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 3.672 ± 0.060 | - | 3.705 ± 0.146 | 11.508 ± 0.381 | 6.148 ± 0.079 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 2.024 ± 0.131 | - | 1.960 ± 0.106 | 1.570 ± 0.045 | 1.601 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.001 | 0.007 ± 0.002 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.110 ± 0.003 | - | 0.115 ± 0.011 | 0.127 ± 0.004 | 0.203 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.001 | 0.008 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.110 ± 0.012 | - | 0.086 ± 0.008 | 0.101 ± 0.011 | 0.171 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.010 ± 0.002 | - | 0.010 ± 0.003 | 0.016 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.010 ± 0.002 | - | 0.012 ± 0.002 | 0.013 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.022 ± 0.005 | - | 0.024 ± 0.006 | 0.033 ± 0.001 | 0.044 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.013 ± 0.002 | - | 0.014 ± 0.001 | 0.016 ± 0.000 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.028 ± 0.006 | - | 0.029 ± 0.000 | 0.043 ± 0.000 | 0.056 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 8.382 ± 0.170 | - | 8.339 ± 0.329 | 12.300 ± 0.326 | 6.611 ± 0.068 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.027 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 0.729 ± 0.254 | - | 0.470 ± 0.014 | 0.992 ± 0.027 | 0.730 ± 0.072 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.019 ± 0.004 | - | 0.019 ± 0.000 | 0.027 ± 0.000 | 0.043 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.056 ± 0.000 | - | 0.057 ± 0.006 | 0.082 ± 0.002 | 0.162 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.313 ± 0.014 | - | unsupported | 0.218 ± 0.013 | 1.319 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.489 ± 0.375 | - | unsupported | 1.371 ± 0.019 | 3.084 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 4.237 ± 0.016 | - | unsupported | 3.860 ± 0.028 | 5.267 ± 0.145 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 1.188 ± 0.048 | - | unsupported | 0.783 ± 0.017 | 0.790 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.046 ± 0.000 | - | unsupported | 0.041 ± 0.001 | 0.047 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.041 ± 0.008 | - | unsupported | 0.035 ± 0.004 | 0.041 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.014 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.013 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | unsupported | 0.010 ± 0.002 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.017 ± 0.000 | - | unsupported | 0.020 ± 0.000 | 0.032 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 9.282 ± 0.028 | - | unsupported | 6.251 ± 0.305 | 6.552 ± 0.072 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.383 ± 0.036 | - | unsupported | 0.292 ± 0.013 | 0.314 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.012 ± 0.000 | - | unsupported | 0.016 ± 0.004 | 0.032 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.033 ± 0.000 | - | unsupported | 0.034 ± 0.000 | 0.120 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.330 ± 0.017 | - | unsupported | 0.250 ± 0.012 | 1.375 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 1.597 ± 0.041 | - | unsupported | 1.471 ± 0.027 | 1.447 ± 0.117 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 2.342 ± 0.065 | - | unsupported | 4.350 ± 0.153 | 2.785 ± 0.071 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.277 ± 0.039 | - | unsupported | 0.802 ± 0.049 | 0.799 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | unsupported | 0.006 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.050 ± 0.009 | - | unsupported | 0.042 ± 0.002 | 0.047 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.045 ± 0.008 | - | unsupported | 0.035 ± 0.001 | 0.043 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | unsupported | 0.011 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | unsupported | 0.011 ± 0.002 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.013 ± 0.000 | - | unsupported | 0.018 ± 0.004 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.008 ± 0.002 | - | unsupported | 0.013 ± 0.002 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.018 ± 0.004 | - | unsupported | 0.020 ± 0.002 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 5.287 ± 0.146 | - | unsupported | 6.758 ± 0.343 | 3.510 ± 1.254 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.229 ± 0.029 | - | unsupported | 0.296 ± 0.003 | 0.225 ± 0.053 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.012 ± 0.000 | - | unsupported | 0.019 ± 0.003 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.040 ± 0.007 | - | unsupported | 0.034 ± 0.009 | 0.048 ± 0.002 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.735 ± 0.040 | - | unsupported | 0.826 ± 0.053 | 1.006 ± 0.024 | 4.028 ± 3.076 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.059 ± 0.005 | - | unsupported | 0.073 ± 0.023 | 0.143 ± 0.002 | 0.105 ± 0.066 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.099 ± 0.008 | - | unsupported | 0.125 ± 0.016 | 0.190 ± 0.002 | 0.148 ± 0.011 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.206 ± 0.001 | - | unsupported | 0.278 ± 0.020 | 0.363 ± 0.014 | 1.407 ± 1.188 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.597 ± 0.015 | - | unsupported | 0.694 ± 0.298 | 0.621 ± 0.037 | 2.513 ± 3.499 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.094 ± 0.092 | - | unsupported | 0.126 ± 0.038 | 0.259 ± 0.051 | 0.086 ± 0.013 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.253 ± 0.088 | - | unsupported | 0.168 ± 0.031 | 0.206 ± 0.018 | 0.169 ± 0.109 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.223 ± 0.026 | - | unsupported | 0.397 ± 0.030 | 0.314 ± 0.024 | 0.815 ± 0.084 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.418 ± 0.003 | - | unsupported | 0.188 ± 0.001 | 0.464 ± 0.024 | 0.552 ± 0.134 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.032 ± 0.000 | - | unsupported | 0.043 ± 0.001 | 0.085 ± 0.004 | 0.085 ± 0.066 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.042 ± 0.001 | - | unsupported | 0.051 ± 0.002 | 0.104 ± 0.005 | 0.097 ± 0.015 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.129 ± 0.002 | - | unsupported | 0.081 ± 0.000 | 0.173 ± 0.013 | 0.228 ± 0.032 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.368 ± 0.021 | - | unsupported | 0.202 ± 0.020 | 0.571 ± 0.099 | 0.473 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.032 ± 0.000 | - | unsupported | 0.049 ± 0.005 | 0.100 ± 0.005 | 0.055 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.042 ± 0.000 | - | unsupported | 0.053 ± 0.008 | 0.121 ± 0.008 | 0.079 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.128 ± 0.001 | - | unsupported | 0.083 ± 0.010 | 0.188 ± 0.015 | 0.203 ± 0.009 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.197 ± 0.003 | - | 0.198 ± 0.003 | 0.111 ± 0.014 | 0.177 ± 0.014 | 0.434 ± 0.396 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.021 ± 0.000 | - | 0.031 ± 0.008 | 0.029 ± 0.020 | 0.026 ± 0.006 | 0.045 ± 0.017 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.026 ± 0.000 | - | 0.028 ± 0.000 | 0.043 ± 0.019 | 0.036 ± 0.007 | 0.045 ± 0.007 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.063 ± 0.003 | - | 0.064 ± 0.000 | 0.052 ± 0.015 | 0.057 ± 0.001 | 0.136 ± 0.008 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.194 ± 0.019 | - | 0.198 ± 0.022 | 0.094 ± 0.014 | 0.243 ± 0.012 | 0.223 ± 0.003 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.021 ± 0.000 | - | 0.026 ± 0.009 | 0.053 ± 0.027 | 0.025 ± 0.002 | 0.033 ± 0.003 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.095 ± 0.005 | - | 0.095 ± 0.001 | 0.042 ± 0.033 | 0.036 ± 0.010 | 0.041 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.222 ± 0.168 | - | 0.080 ± 0.010 | 0.039 ± 0.020 | 0.082 ± 0.004 | 0.093 ± 0.018 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 36.170 ± 1.422 | - | 36.237 ± 1.490 | 23.789 ± 0.117 | 16.081 ± 0.218 | 37.517 ± 18.440 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 22.083 ± 0.663 | - | 22.092 ± 1.940 | 16.613 ± 2.025 | 13.880 ± 2.811 | 369.864 ± 202.980 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 24.580 ± 1.180 | - | 21.045 ± 0.869 | 7.144 ± 0.155 | 8.278 ± 0.289 | 25.306 ± 8.307 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 19.213 ± 0.272 | - | 16.198 ± 0.331 | 7.531 ± 0.341 | 10.261 ± 3.456 | 236.589 ± 20.050 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 6.398 ± 0.049 | - | 6.403 ± 0.056 | 7.184 ± 0.531 | 8.573 ± 0.328 | 7.377 ± 0.429 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 7.991 ± 0.269 | - | 9.279 ± 1.595 | 6.609 ± 0.188 | 8.127 ± 0.695 | 301.404 ± 9.143 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 7.554 ± 0.216 | - | 7.576 ± 0.060 | 7.851 ± 0.196 | 7.395 ± 0.054 | 7.897 ± 0.428 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 10.543 ± 0.278 | - | 10.726 ± 0.238 | 8.041 ± 0.570 | 7.142 ± 0.082 | 271.117 ± 28.079 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 19.427 ± 1.161 | - | 19.236 ± 0.189 | 21.592 ± 0.179 | 30.079 ± 8.365 | 15.033 ± 0.898 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 19.382 ± 0.204 | - | 19.341 ± 0.316 | 20.236 ± 1.173 | 25.306 ± 0.146 | 652.094 ± 187.771 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.352 ± 0.007 | 0.342 ± 0.038 | unsupported | 0.316 ± 0.002 | 0.362 ± 0.013 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 6.222 ± 0.265 | 6.280 ± 0.165 | unsupported | 3.875 ± 0.310 | 4.458 ± 0.177 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.701 ± 0.105 | 0.728 ± 0.126 | unsupported | 0.349 ± 0.077 | 0.375 ± 0.019 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 6.669 ± 0.718 | 7.073 ± 0.486 | unsupported | 5.845 ± 1.280 | 4.681 ± 0.080 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 33.903 ± 1.725 | - | 34.299 ± 0.512 | 9.222 ± 0.638 | 9.434 ± 0.088 | 28.533 ± 2.077 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 23.683 ± 0.555 | - | 23.306 ± 1.036 | 9.438 ± 0.237 | 14.950 ± 10.030 | 410.931 ± 321.420 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 19.230 ± 1.637 | 20.320 ± 3.013 | 5.147 ± 0.118 | 31.218 ± 0.043 | 11.121 ± 4.407 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 18.498 ± 2.314 | 17.608 ± 3.211 | 5.306 ± 0.068 | 30.627 ± 2.024 | 132.350 ± 44.754 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 27.897 ± 5.570 | - | 25.974 ± 5.243 | 11.536 ± 0.587 | 12.535 ± 0.139 | 29.660 ± 16.264 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 27.165 ± 7.803 | - | 24.913 ± 1.663 | 10.067 ± 0.445 | 9.962 ± 0.145 | 256.093 ± 32.428 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.799 ± 1.072 | - | 1.212 ± 0.487 | 2.236 ± 0.016 | 7.169 ± 1.663 | 5.734 ± 0.043 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.830 ± 0.049 | - | 0.965 ± 0.052 | 2.309 ± 0.033 | 3.423 ± 0.042 | 7.939 ± 2.409 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 16.999 ± 0.453 | - | 16.833 ± 0.060 | 10.661 ± 0.043 | 14.344 ± 1.830 | 20.655 ± 4.371 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 17.035 ± 1.645 | - | 16.440 ± 0.526 | 11.563 ± 0.455 | 11.645 ± 0.882 | 959.926 ± 76.170 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 17.228 ± 2.315 | - | 21.680 ± 4.003 | 10.633 ± 0.074 | 13.133 ± 0.151 | 21.094 ± 6.422 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 19.855 ± 2.259 | - | 20.678 ± 5.038 | 11.514 ± 0.230 | 11.947 ± 2.069 | 965.992 ± 35.108 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.076 ± 0.013 | 1.083 ± 0.021 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 26.871 ± 1.393 | 26.540 ± 2.038 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.091 ± 0.018 | 1.093 ± 0.048 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 23.924 ± 0.914 | 22.979 ± 1.444 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.071 ± 0.010 | 1.100 ± 0.052 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 26.693 ± 4.739 | 26.491 ± 1.513 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.085 ± 0.023 | 1.084 ± 0.012 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 23.902 ± 7.049 | 23.238 ± 0.902 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 24.764 ± 0.356 | - | 20.979 ± 0.642 | 7.176 ± 1.155 | 8.568 ± 0.453 | 19.877 ± 2.209 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 19.317 ± 1.318 | - | 16.196 ± 0.708 | 7.527 ± 0.198 | 9.783 ± 5.424 | 235.878 ± 39.864 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 58.350 ± 10.252 | 62.724 ± 0.681 | 34.512 ± 0.166 | 39.935 ± 5.072 | 70.927 ± 7.809 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 62.939 ± 14.186 | 62.447 ± 3.968 | 40.380 ± 1.109 | 35.364 ± 0.410 | 1936.674 ± 527.099 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 36.136 ± 0.094 | - | 36.907 ± 0.620 | 12.708 ± 1.039 | 67.672 ± 2.025 | 31.868 ± 11.617 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 24.924 ± 1.547 | - | 25.112 ± 0.332 | 13.131 ± 0.119 | 44.325 ± 8.203 | 18.083 ± 7.438 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.185 ± 0.030 | 0.184 ± 0.004 | unsupported | 0.058 ± 0.022 | 0.035 ± 0.002 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.817 ± 0.127 | 2.838 ± 0.047 | unsupported | 0.887 ± 0.158 | 0.478 ± 0.017 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.180 ± 0.074 | 0.181 ± 0.038 | unsupported | 0.142 ± 0.041 | 0.063 ± 0.033 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 1.598 ± 0.640 | 1.472 ± 0.237 | unsupported | 0.574 ± 0.172 | 0.309 ± 0.049 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.083 ± 0.004 | 0.084 ± 0.045 | unsupported | 0.091 ± 0.004 | 0.344 ± 0.068 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.260 ± 0.010 | 1.259 ± 0.023 | unsupported | 1.994 ± 1.429 | 2.395 ± 0.039 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.115 ± 0.003 | 0.112 ± 0.017 | unsupported | 0.139 ± 0.033 | 0.190 ± 0.009 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.669 ± 0.032 | 0.708 ± 0.056 | unsupported | 0.852 ± 0.235 | 1.288 ± 0.241 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.358 ± 0.104 | 0.286 ± 0.063 | unsupported | 0.319 ± 0.026 | 0.180 ± 0.087 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 4.225 ± 0.101 | 4.425 ± 0.112 | unsupported | 6.507 ± 1.888 | 0.927 ± 0.013 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.223 ± 0.005 | 0.212 ± 0.032 | unsupported | 0.311 ± 0.039 | 0.071 ± 0.013 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 2.182 ± 0.749 | 3.118 ± 0.848 | unsupported | 2.791 ± 0.205 | 0.539 ± 0.013 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.052 ± 0.004 | 0.064 ± 0.002 | unsupported | 0.049 ± 0.002 | 0.092 ± 0.004 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.922 ± 0.230 | 0.875 ± 0.310 | unsupported | 0.657 ± 0.126 | 0.656 ± 0.009 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.048 ± 0.001 | 0.061 ± 0.002 | unsupported | 0.083 ± 0.004 | 0.094 ± 0.001 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 1.500 ± 0.722 | 1.698 ± 0.420 | unsupported | 0.420 ± 0.028 | 0.653 ± 0.385 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.134 ± 0.087 | 0.119 ± 0.123 | unsupported | 0.096 ± 0.021 | 0.122 ± 0.002 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.228 ± 0.027 | 1.415 ± 0.205 | unsupported | 1.558 ± 0.496 | 1.197 ± 0.026 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.095 ± 0.006 | 0.109 ± 0.013 | unsupported | 0.133 ± 0.042 | 0.153 ± 0.039 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 2.202 ± 0.789 | 3.077 ± 0.729 | unsupported | 1.074 ± 0.228 | 0.833 ± 0.025 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.212 ± 0.049 | 0.265 ± 0.117 | unsupported | unsupported | 0.142 ± 0.015 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 2.139 ± 0.165 | 2.434 ± 0.721 | unsupported | unsupported | 1.700 ± 0.099 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.131 ± 0.010 | 0.144 ± 0.007 | unsupported | unsupported | 0.129 ± 0.019 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 3.369 ± 1.048 | 3.407 ± 1.097 | unsupported | unsupported | 1.121 ± 0.041 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.019 ± 0.017 | 0.023 ± 0.025 | unsupported | 0.008 ± 0.001 | 0.098 ± 0.027 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.108 ± 0.004 | 0.115 ± 0.008 | unsupported | 0.075 ± 0.077 | 0.240 ± 0.010 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.013 ± 0.000 | 0.017 ± 0.001 | unsupported | 0.032 ± 0.008 | 0.064 ± 0.002 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.131 ± 0.029 | 0.228 ± 0.233 | unsupported | 0.112 ± 0.013 | 0.175 ± 0.034 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.029 ± 0.041 | 0.038 ± 0.014 | unsupported | 0.029 ± 0.006 | 0.082 ± 0.008 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.387 ± 0.090 | 0.482 ± 0.088 | unsupported | 0.362 ± 0.029 | 0.589 ± 0.039 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.026 ± 0.002 | 0.034 ± 0.002 | unsupported | 0.149 ± 0.038 | 0.089 ± 0.003 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.712 ± 0.100 | 0.565 ± 0.249 | unsupported | 0.456 ± 0.004 | 0.454 ± 0.020 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.051 ± 0.001 | 0.052 ± 0.001 | unsupported | 0.084 ± 0.006 | 0.140 ± 0.040 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.778 ± 0.024 | 0.771 ± 0.024 | unsupported | 1.296 ± 0.040 | 0.811 ± 0.012 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.065 ± 0.040 | 0.079 ± 0.017 | unsupported | 0.174 ± 0.044 | 0.073 ± 0.041 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.432 ± 0.022 | 0.430 ± 0.016 | unsupported | 0.816 ± 0.220 | 0.806 ± 0.104 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.052 ± 0.001 | 0.053 ± 0.001 | unsupported | 0.084 ± 0.005 | 0.169 ± 0.059 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.817 ± 0.031 | 0.859 ± 0.047 | unsupported | 1.495 ± 0.506 | 0.880 ± 0.013 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.064 ± 0.018 | 0.057 ± 0.041 | unsupported | 0.173 ± 0.029 | 0.075 ± 0.015 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.448 ± 0.005 | 0.448 ± 0.005 | unsupported | 0.963 ± 0.239 | 0.546 ± 0.042 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.144 ± 0.087 | 0.235 ± 0.117 | unsupported | 0.104 ± 0.004 | 0.125 ± 0.104 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.282 ± 0.027 | 1.260 ± 0.053 | unsupported | 2.181 ± 0.970 | 1.176 ± 0.023 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.099 ± 0.003 | 0.106 ± 0.001 | unsupported | 0.114 ± 0.030 | 0.180 ± 0.021 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 2.371 ± 0.473 | 2.490 ± 0.452 | unsupported | 0.988 ± 0.036 | 0.747 ± 0.040 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.233 ± 0.047 | 0.230 ± 0.096 | unsupported | 0.292 ± 0.003 | 0.607 ± 0.182 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 3.691 ± 0.035 | 3.532 ± 0.065 | unsupported | 6.752 ± 1.212 | 4.277 ± 0.276 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.212 ± 0.008 | 0.214 ± 0.007 | unsupported | 0.337 ± 0.120 | 0.206 ± 0.009 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 1.855 ± 0.084 | 1.917 ± 0.545 | unsupported | 2.554 ± 0.273 | 2.166 ± 0.045 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.806 ± 0.242 | 0.672 ± 0.224 | unsupported | 0.130 ± 0.009 | 0.111 ± 0.004 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 13.307 ± 3.654 | 14.718 ± 1.291 | unsupported | 2.370 ± 1.116 | 1.778 ± 1.133 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.774 ± 0.034 | 0.798 ± 0.133 | unsupported | 0.131 ± 0.055 | 0.146 ± 0.100 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 8.868 ± 0.674 | 9.282 ± 2.056 | unsupported | 1.422 ± 0.056 | 0.623 ± 0.031 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 22.749 ± 2.990 | - | unsupported | 17.492 ± 4.491 | - | 17.812 ± 0.942 | 17.468 ± 0.297 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 18.327 ± 5.274 | - | unsupported | 21.966 ± 2.241 | - | 14.928 ± 1.265 | 15.581 ± 0.293 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 10.308 ± 0.203 | - | unsupported | 11.224 ± 0.035 | - | 10.575 ± 0.142 | 10.994 ± 0.776 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 11.779 ± 0.734 | - | unsupported | 12.735 ± 1.236 | - | 11.062 ± 0.648 | 12.271 ± 0.233 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 10.384 ± 0.336 | - | unsupported | 11.284 ± 0.305 | - | 11.130 ± 1.242 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 15.126 ± 0.406 | - | unsupported | 14.174 ± 3.823 | - | 10.039 ± 1.895 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 17.804 ± 0.684 | - | unsupported | 19.410 ± 6.110 | - | 18.018 ± 0.773 | 17.510 ± 0.239 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 18.467 ± 2.152 | - | unsupported | 20.235 ± 2.836 | - | 14.552 ± 0.622 | 17.207 ± 2.145 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 49.137 ± 1.338 | - | unsupported | 10.964 ± 0.253 | - | 47.860 ± 0.718 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 28.015 ± 7.141 | - | unsupported | 13.092 ± 1.874 | - | 21.398 ± 1.951 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 49.526 ± 0.680 | - | unsupported | 12.102 ± 0.285 | - | 48.119 ± 1.787 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 31.379 ± 3.792 | - | unsupported | 12.365 ± 2.262 | - | 21.967 ± 1.586 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 18.706 ± 9.143 | - | unsupported | 18.708 ± 7.744 | - | 18.488 ± 0.777 | 19.490 ± 1.646 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 18.334 ± 1.078 | - | unsupported | 20.482 ± 1.536 | - | 14.220 ± 0.184 | 18.807 ± 3.043 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 11.376 ± 0.148 | - | unsupported | 12.488 ± 1.770 | - | 10.718 ± 0.973 | 11.869 ± 0.871 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 11.923 ± 0.608 | - | unsupported | 13.577 ± 2.254 | - | 10.906 ± 0.275 | 10.919 ± 1.897 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.527 ± 0.022 | unsupported | unsupported | 0.295 ± 0.013 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 9.777 ± 0.558 | unsupported | unsupported | 3.711 ± 0.172 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 1.023 ± 0.047 | unsupported | unsupported | 0.530 ± 0.144 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 8.587 ± 3.259 | unsupported | unsupported | 5.031 ± 1.148 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 19.387 ± 1.367 | - | unsupported | 20.736 ± 13.336 | - | 18.947 ± 0.771 | 19.176 ± 1.211 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 17.916 ± 1.335 | - | unsupported | 21.009 ± 4.314 | - | 17.267 ± 0.648 | 16.574 ± 0.875 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 8.136 ± 0.206 | - | 0.002 ± 0.000 | 5.868 ± 0.700 | 5.054 ± 0.218 | 13.511 ± 0.539 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 8.587 ± 0.875 | - | 0.002 ± 0.000 | 5.377 ± 0.162 | 4.584 ± 0.121 | 13.910 ± 0.267 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 9.202 ± 0.217 | - | 9.023 ± 0.154 | 9.607 ± 2.544 | 9.116 ± 0.192 | 9.280 ± 0.528 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 9.870 ± 1.109 | - | 9.223 ± 0.229 | 9.305 ± 0.598 | 8.147 ± 1.376 | 8.942 ± 0.323 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 10.791 ± 0.216 | - | 10.721 ± 1.042 | 10.854 ± 0.353 | 17.058 ± 0.362 | 10.810 ± 0.234 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 12.913 ± 2.282 | - | 12.233 ± 1.736 | 13.045 ± 1.531 | 9.354 ± 4.999 | 11.414 ± 1.538 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 5.527 ± 0.122 | - | 5.692 ± 0.629 | 9.571 ± 0.252 | 8.646 ± 0.403 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 4.911 ± 0.269 | - | 5.885 ± 0.232 | 11.551 ± 1.141 | 7.951 ± 0.090 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 10.593 ± 0.140 | - | 0.003 ± 0.003 | 12.882 ± 1.934 | 11.295 ± 0.200 | 13.872 ± 3.906 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 11.608 ± 0.342 | - | 0.001 ± 0.000 | 12.053 ± 0.792 | 9.893 ± 0.171 | 11.125 ± 1.223 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 30.690 ± 1.263 | - | 0.002 ± 0.006 | 51.303 ± 4.369 | 49.590 ± 1.875 | 189.191 ± 22.951 | 30.801 ± 3.020 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 17.416 ± 2.212 | - | 0.001 ± 0.000 | 43.279 ± 2.001 | 25.902 ± 0.196 | 207.510 ± 30.293 | 15.714 ± 1.517 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 7.561 ± 0.319 | - | 8.312 ± 0.153 | 7.899 ± 0.386 | 5.779 ± 0.130 | 10.989 ± 2.657 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 5.289 ± 0.401 | - | 5.476 ± 0.660 | 7.667 ± 1.136 | 5.900 ± 1.275 | 9.617 ± 0.426 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 8.268 ± 0.131 | - | 8.692 ± 0.807 | 8.053 ± 0.713 | 5.858 ± 1.316 | 10.347 ± 13.220 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 5.547 ± 0.360 | - | 5.667 ± 0.969 | 7.715 ± 1.829 | 5.248 ± 0.461 | 9.302 ± 0.117 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.003 ± 0.002 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.003 ± 0.004 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 4.217625 | 263601.56 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 4.069750 | 254359.38 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 4.265208 | 266575.50 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 3.553041 | 222065.06 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 4.321500 | 270093.75 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 4.359792 | 272487.00 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 12.704167 | 794010.44 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 11.881250 | 742578.12 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 11.087708 | 692981.75 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 11.773917 | 735869.81 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 5.797125 | 362320.31 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 5.059084 | 316192.75 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 5.633042 | 352065.12 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 5.474416 | 342151.00 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 71.327166 | 4457947.88 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 62.005750 | 3875359.38 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 99.550625 | 6221914.06 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 100.486833 | 6280427.06 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 6.007417 | 375463.56 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 5.585541 | 349096.31 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 11.218291 | 701143.19 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 11.647917 | 727994.81 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 74.893084 | 4680817.75 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 93.515125 | 5844695.31 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 106.698125 | 6668632.81 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 113.163166 | 7072697.88 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 17.214875 | 1075929.69 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 17.326250 | 1082890.62 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 429.936750 | 26871046.88 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 424.639959 | 26539997.44 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 17.459083 | 1091192.69 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 17.490875 | 1093179.69 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 382.787166 | 23924197.88 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 367.669625 | 22979351.56 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 17.135708 | 1070981.75 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 17.607041 | 1100440.06 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 427.092708 | 26693294.25 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 423.863292 | 26491455.75 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 17.362000 | 1085125.00 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 17.336208 | 1083513.00 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 382.436750 | 23902296.88 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 371.802916 | 23237682.25 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.267833 | 35434.89 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.726208 | 58222.00 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 11.818042 | 184656.91 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 11.780708 | 184073.56 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 3.821958 | 477744.75 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.092458 | 886557.25 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 22.532166 | 2816520.75 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 22.702791 | 2837848.88 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.037208 | 63081.37 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 9.102333 | 142223.95 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 11.526666 | 180104.16 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 11.606333 | 181348.95 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.470250 | 308781.25 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.593750 | 574218.75 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 12.785083 | 1598135.38 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 11.777458 | 1472182.25 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 21.991875 | 343623.05 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.849000 | 91390.62 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.282041 | 82531.89 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.400292 | 84379.56 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 19.162042 | 2395255.25 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 15.955750 | 1994468.75 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.082042 | 1260255.25 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.069792 | 1258724.00 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.145750 | 189777.34 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 8.913834 | 139278.66 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 7.382500 | 115351.56 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.180750 | 112199.22 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.302958 | 1287869.75 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.815000 | 851875.00 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.351542 | 668942.75 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.667500 | 708437.50 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.520709 | 180011.08 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 20.390083 | 318595.05 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 22.902375 | 357849.61 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 18.314166 | 286158.84 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.415875 | 926984.38 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 52.059709 | 6507463.62 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 33.802042 | 4225255.25 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 35.402708 | 4425338.50 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.564375 | 71318.36 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 19.935583 | 311493.48 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 14.247292 | 222613.94 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 13.571791 | 212059.23 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.315625 | 539453.12 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 22.328417 | 2791052.12 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 17.453083 | 2181635.38 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 24.940500 | 3117562.50 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.865792 | 91653.00 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.152041 | 49250.64 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.340458 | 52194.66 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.102375 | 64099.61 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.245250 | 655656.25 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 5.257958 | 657244.75 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.375833 | 921979.12 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.002792 | 875349.00 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.984083 | 93501.30 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.286167 | 82596.36 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.056750 | 47761.72 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3.890917 | 60795.58 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.221833 | 652729.12 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.362958 | 420369.75 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 11.996875 | 1499609.38 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 13.585875 | 1698234.38 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.825959 | 122280.61 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.114166 | 95533.84 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.589625 | 134212.89 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 7.589875 | 118591.80 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 9.574625 | 1196828.12 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 12.464875 | 1558109.38 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.823375 | 1227921.88 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.319875 | 1414984.38 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.764458 | 152569.66 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 8.495667 | 132744.80 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.049042 | 94516.28 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6.973833 | 108966.14 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.660959 | 832619.88 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.592916 | 1074114.50 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 17.618708 | 2202338.50 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 24.617000 | 3077125.00 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.084666 | 141947.91 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 13.587500 | 212304.69 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 16.972375 | 265193.36 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.597750 | 1699718.75 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 17.112625 | 2139078.12 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 19.472667 | 2434083.38 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.234209 | 128659.52 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.383542 | 130992.84 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 9.202792 | 143793.62 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.966125 | 1120765.62 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 26.949458 | 3368682.25 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 27.255042 | 3406880.25 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.287625 | 98244.14 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.522833 | 8169.27 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.236417 | 19319.02 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.495917 | 23373.70 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 1.923833 | 240479.12 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.600250 | 75031.25 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.860375 | 107546.88 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.922541 | 115317.62 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.078542 | 63727.22 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.042250 | 31910.16 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.831000 | 12984.38 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.056292 | 16504.56 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.399875 | 174984.38 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.895542 | 111942.75 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.045208 | 130651.00 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.826333 | 228291.62 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.262834 | 82231.78 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 1.879458 | 29366.53 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.845375 | 28833.98 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.402792 | 37543.62 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 4.710375 | 588796.88 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 2.897459 | 362182.38 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.097667 | 387208.38 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.859459 | 482432.38 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.710000 | 89218.75 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 9.559584 | 149368.50 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.637667 | 25588.55 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.193500 | 34273.44 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.635000 | 454375.00 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.644000 | 455500.00 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.693250 | 711656.25 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.518584 | 564823.00 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.958291 | 139973.30 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.345083 | 83516.92 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.246333 | 50723.95 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.300792 | 51574.88 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.489083 | 811135.38 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.364417 | 1295552.12 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.221875 | 777734.38 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.169542 | 771192.75 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.669125 | 72955.08 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 11.147250 | 174175.78 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.179125 | 65298.83 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.027958 | 78561.84 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.451750 | 806468.75 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.524209 | 815526.12 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.454166 | 431770.75 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.439375 | 429921.88 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.829208 | 169206.38 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.369916 | 83904.94 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.357541 | 52461.58 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.409000 | 53265.62 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.039208 | 879901.00 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.962791 | 1495348.88 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.538333 | 817291.62 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.870750 | 858843.75 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.804791 | 75074.86 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 11.057666 | 172776.03 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.082625 | 63791.02 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3.643417 | 56928.39 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.364666 | 545583.25 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.705542 | 963192.75 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.587083 | 448385.38 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.582958 | 447869.75 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.010584 | 125165.38 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.651584 | 103931.00 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 9.238541 | 144352.20 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 15.018416 | 234662.75 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 9.407583 | 1175947.88 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 17.444375 | 2180546.88 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.252916 | 1281614.50 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.079708 | 1259963.50 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.511292 | 179863.94 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 7.291916 | 113936.19 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.358292 | 99348.31 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6.780000 | 105937.50 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.977084 | 747135.50 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.900500 | 987562.50 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 18.968458 | 2371057.25 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 19.923250 | 2490406.25 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 38.853334 | 607083.34 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 18.673250 | 291769.53 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 14.934375 | 233349.61 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 14.738291 | 230285.80 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 34.213916 | 4276739.50 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 54.015542 | 6751942.75 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 29.528750 | 3691093.75 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 28.253000 | 3531625.00 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.195292 | 206176.44 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 21.553166 | 336768.22 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 13.576541 | 212133.45 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 13.683500 | 213804.69 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 17.329666 | 2166208.25 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 20.433834 | 2554229.25 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 14.843833 | 1855479.12 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 15.336875 | 1917109.38 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.117167 | 111205.73 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.344333 | 130380.20 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 51.584791 | 806012.36 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 43.025916 | 672279.94 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.227500 | 1778437.50 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 18.956500 | 2369562.50 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 106.458333 | 13307291.62 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 117.747458 | 14718432.25 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.364542 | 146320.97 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 8.384709 | 131011.08 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 49.562625 | 774416.02 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 51.069500 | 797960.94 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.984958 | 623119.75 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 11.379209 | 1422401.12 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 70.947250 | 8868406.25 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 74.259625 | 9282453.12 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 4.712334 | 294520.88 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 8.427250 | 526703.12 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 59.375625 | 3710976.56 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 156.427250 | 9776703.12 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 8.478084 | 529880.25 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 16.361667 | 1022604.19 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 80.496042 | 5031002.62 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 137.394750 | 8587171.88 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.013875 | 867.19 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.001417 | 88.56 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.013125 | 820.31 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.001333 | 83.31 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.012834 | 802.12 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.041000 | 2562.50 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000125 | 7.81 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.012250 | 765.62 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.050625 | 3164.06 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.012542 | 783.88 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.002708 | 169.25 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000084 | 5.25 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.010000 | 625.00 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.002917 | 182.31 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.011375 | 710.94 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002792 | 174.50 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.011458 | 716.12 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003041 | 190.06 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `jax-cpu` is 47.8x faster than the slowest successful cell (`julia-base`, 71.264 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `pytorch-cpu` is 17.5x faster than the slowest successful cell (`julia-base`, 71.264 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-direct` is 19.5x faster than the slowest successful cell (`julia-base`, 71.264 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-trace` is 18.7x faster than the slowest successful cell (`julia-base`, 71.264 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `jax-cpu` is 19.9x faster than the slowest successful cell (`julia-base`, 441.473 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `pytorch-cpu` is 40.6x faster than the slowest successful cell (`julia-base`, 441.473 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-direct` is 45.6x faster than the slowest successful cell (`julia-base`, 441.473 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-trace` is 44.7x faster than the slowest successful cell (`julia-base`, 441.473 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 11.4x faster than the slowest successful cell (`julia-base`, 11.513 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 19.4x faster than the slowest successful cell (`julia-base`, 151.582 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 20.0x faster than the slowest successful cell (`julia-base`, 151.582 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-direct` is 23.4x faster than the slowest successful cell (`julia-base`, 151.582 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-trace` is 22.8x faster than the slowest successful cell (`julia-base`, 151.582 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 113.6x faster than the slowest successful cell (`julia-base`, 286.310 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 47.3x faster than the slowest successful cell (`julia-base`, 286.310 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-direct` is 53.1x faster than the slowest successful cell (`julia-base`, 286.310 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-trace` is 52.6x faster than the slowest successful cell (`julia-base`, 286.310 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `jax-cpu` is 99.8x faster than the slowest successful cell (`julia-base`, 642.758 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `pytorch-cpu` is 86.2x faster than the slowest successful cell (`julia-base`, 642.758 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-direct` is 81.1x faster than the slowest successful cell (`julia-base`, 642.758 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-trace` is 74.2x faster than the slowest successful cell (`julia-base`, 642.758 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 19.5x faster than the slowest successful cell (`pytorch-cpu`, 14.577 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 16.9x faster than the slowest successful cell (`pytorch-cpu`, 14.577 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 16.9x faster than the slowest successful cell (`pytorch-cpu`, 14.577 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 15.1x faster than the slowest successful cell (`pytorch-cpu`, 10.798 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 13.8x faster than the slowest successful cell (`pytorch-cpu`, 10.798 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 13.9x faster than the slowest successful cell (`pytorch-cpu`, 10.798 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 23.3x faster than the slowest successful cell (`pytorch-cpu`, 16.525 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 19.2x faster than the slowest successful cell (`pytorch-cpu`, 16.525 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 19.3x faster than the slowest successful cell (`pytorch-cpu`, 16.525 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 14.6x faster than the slowest successful cell (`pytorch-cpu`, 10.748 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 13.3x faster than the slowest successful cell (`pytorch-cpu`, 10.748 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 10.748 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 331.5x faster than the slowest successful cell (`julia-base`, 1.823 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 1117.2x faster than the slowest successful cell (`pytorch-cpu`, 1.582 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `pytorch-cpu` is 12.3x faster than the slowest successful cell (`jax-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 23.0x faster than the slowest successful cell (`jax-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-trace` is 17.3x faster than the slowest successful cell (`jax-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `pytorch-cpu` is 17.7x faster than the slowest successful cell (`jax-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 54.1x faster than the slowest successful cell (`jax-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 30.6x faster than the slowest successful cell (`jax-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 22.1x faster than the slowest successful cell (`jax-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-trace` is 14.8x faster than the slowest successful cell (`jax-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `jax-cpu` is 10.4x faster than the slowest successful cell (`pytorch-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 32.1x faster than the slowest successful cell (`pytorch-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 29.1x faster than the slowest successful cell (`pytorch-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 32.5x faster than the slowest successful cell (`pytorch-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-trace` is 23.7x faster than the slowest successful cell (`pytorch-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 41.8x faster than the slowest successful cell (`pytorch-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 23.6x faster than the slowest successful cell (`pytorch-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 41.8x faster than the slowest successful cell (`pytorch-cpu`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 12.3x faster than the slowest successful cell (`pytorch-cpu`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `jax-cpu` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 79.0x faster than the slowest successful cell (`pytorch-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 13.3x faster than the slowest successful cell (`pytorch-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 58.7x faster than the slowest successful cell (`pytorch-cpu`, 0.073 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 22.9x faster than the slowest successful cell (`pytorch-cpu`, 0.073 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 34.4x faster than the slowest successful cell (`pytorch-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 17.0x faster than the slowest successful cell (`pytorch-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 16.5x faster than the slowest successful cell (`pytorch-cpu`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 19.3x faster than the slowest successful cell (`pytorch-cpu`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 24.6x faster than the slowest successful cell (`pytorch-cpu`, 0.040 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 15.8x faster than the slowest successful cell (`pytorch-cpu`, 0.040 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 27.7x faster than the slowest successful cell (`pytorch-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 15.0x faster than the slowest successful cell (`pytorch-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 0.040 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 0.040 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 29.1x faster than the slowest successful cell (`pytorch-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 21.0x faster than the slowest successful cell (`pytorch-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 14.6x faster than the slowest successful cell (`jax-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.013 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.022 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.022 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 23.6x faster than the slowest successful cell (`jax-cpu`, 0.073 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-trace` is 17.5x faster than the slowest successful cell (`jax-cpu`, 0.073 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 18.4x faster than the slowest successful cell (`jax-cpu`, 0.064 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 14.0x faster than the slowest successful cell (`jax-cpu`, 0.064 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 20.4x faster than the slowest successful cell (`jax-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-trace` is 14.6x faster than the slowest successful cell (`jax-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 13.1x faster than the slowest successful cell (`jax-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x8x8`): `tenferro-trace` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 30.5x faster than the slowest successful cell (`pytorch-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 16.2x faster than the slowest successful cell (`pytorch-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 23.8x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 15.7x faster than the slowest successful cell (`pytorch-cpu`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 34.6x faster than the slowest successful cell (`pytorch-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 18.2x faster than the slowest successful cell (`pytorch-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 19.8x faster than the slowest successful cell (`pytorch-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 17.6x faster than the slowest successful cell (`pytorch-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 16.5x faster than the slowest successful cell (`jax-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 15.5x faster than the slowest successful cell (`jax-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 25.9x faster than the slowest successful cell (`pytorch-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 18.4x faster than the slowest successful cell (`pytorch-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 23.1x faster than the slowest successful cell (`pytorch-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 16.6x faster than the slowest successful cell (`pytorch-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 17.5x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 15.8x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 36.7x faster than the slowest successful cell (`pytorch-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 21.7x faster than the slowest successful cell (`pytorch-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 18.8x faster than the slowest successful cell (`pytorch-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 18.2x faster than the slowest successful cell (`pytorch-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 18.1x faster than the slowest successful cell (`jax-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`jax-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 19.3x faster than the slowest successful cell (`jax-cpu`, 0.031 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`1x4x4`): `tenferro-trace` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.031 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 20.5x faster than the slowest successful cell (`pytorch-cpu`, 0.032 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 0.032 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 30.0x faster than the slowest successful cell (`pytorch-cpu`, 0.053 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 18.3x faster than the slowest successful cell (`pytorch-cpu`, 0.053 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 20.1x faster than the slowest successful cell (`pytorch-cpu`, 0.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 13.8x faster than the slowest successful cell (`pytorch-cpu`, 0.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 21.2x faster than the slowest successful cell (`pytorch-cpu`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 16.5x faster than the slowest successful cell (`pytorch-cpu`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 10.4x faster than the slowest successful cell (`pytorch-cpu`, 0.045 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 16.7x faster than the slowest successful cell (`pytorch-cpu`, 0.033 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.032 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x32x32,rhs=1`): `tenferro-direct` is 15.0x faster than the slowest successful cell (`jax-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x4x4,rhs=1`): `tenferro-direct` is 19.3x faster than the slowest successful cell (`jax-cpu`, 0.031 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`4x4x4,rhs=1`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.026 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x2x2,rhs=1`): `tenferro-direct` is 17.8x faster than the slowest successful cell (`pytorch-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x4x4,rhs=1`): `tenferro-direct` is 27.9x faster than the slowest successful cell (`pytorch-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x4x4,rhs=1`): `tenferro-trace` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x8x8,rhs=1`): `tenferro-direct` is 15.5x faster than the slowest successful cell (`pytorch-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (c64, threads=1, shape=`3x2x2`): `tenferro-direct` is 13.9x faster than the slowest successful cell (`jax-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (c64, threads=1, shape=`3x2x2`): `tenferro-trace` is 12.7x faster than the slowest successful cell (`jax-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (c64, threads=1, shape=`8x2x2`): `tenferro-direct` is 12.6x faster than the slowest successful cell (`jax-cpu`, 0.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (c64, threads=1, shape=`8x2x2`): `tenferro-trace` is 11.3x faster than the slowest successful cell (`jax-cpu`, 0.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `jax-cpu` is 26.6x faster than the slowest successful cell (`julia-base`, 369.864 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `pytorch-cpu` is 22.3x faster than the slowest successful cell (`julia-base`, 369.864 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-direct` is 16.7x faster than the slowest successful cell (`julia-base`, 369.864 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-trace` is 16.7x faster than the slowest successful cell (`julia-base`, 369.864 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 23.1x faster than the slowest successful cell (`julia-base`, 236.589 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 31.4x faster than the slowest successful cell (`julia-base`, 236.589 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 12.3x faster than the slowest successful cell (`julia-base`, 236.589 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 14.6x faster than the slowest successful cell (`julia-base`, 236.589 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `jax-cpu` is 37.1x faster than the slowest successful cell (`julia-base`, 301.404 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `pytorch-cpu` is 45.6x faster than the slowest successful cell (`julia-base`, 301.404 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-direct` is 37.7x faster than the slowest successful cell (`julia-base`, 301.404 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-trace` is 32.5x faster than the slowest successful cell (`julia-base`, 301.404 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `jax-cpu` is 38.0x faster than the slowest successful cell (`julia-base`, 271.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `pytorch-cpu` is 33.7x faster than the slowest successful cell (`julia-base`, 271.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-direct` is 25.7x faster than the slowest successful cell (`julia-base`, 271.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-trace` is 25.3x faster than the slowest successful cell (`julia-base`, 271.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `jax-cpu` is 25.8x faster than the slowest successful cell (`julia-base`, 652.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 32.2x faster than the slowest successful cell (`julia-base`, 652.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-direct` is 33.6x faster than the slowest successful cell (`julia-base`, 652.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-trace` is 33.7x faster than the slowest successful cell (`julia-base`, 652.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 27.5x faster than the slowest successful cell (`julia-base`, 410.931 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 43.5x faster than the slowest successful cell (`julia-base`, 410.931 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-direct` is 17.4x faster than the slowest successful cell (`julia-base`, 410.931 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-trace` is 17.6x faster than the slowest successful cell (`julia-base`, 410.931 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 24.9x faster than the slowest successful cell (`julia-base`, 132.350 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 25.7x faster than the slowest successful cell (`julia-base`, 256.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 25.4x faster than the slowest successful cell (`julia-base`, 256.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 10.3x faster than the slowest successful cell (`julia-base`, 256.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `jax-cpu` is 82.4x faster than the slowest successful cell (`julia-base`, 959.926 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 83.0x faster than the slowest successful cell (`julia-base`, 959.926 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 56.4x faster than the slowest successful cell (`julia-base`, 959.926 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 58.4x faster than the slowest successful cell (`julia-base`, 959.926 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `jax-cpu` is 80.9x faster than the slowest successful cell (`julia-base`, 965.992 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 83.9x faster than the slowest successful cell (`julia-base`, 965.992 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 48.7x faster than the slowest successful cell (`julia-base`, 965.992 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 46.7x faster than the slowest successful cell (`julia-base`, 965.992 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 24.1x faster than the slowest successful cell (`julia-base`, 235.878 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 31.3x faster than the slowest successful cell (`julia-base`, 235.878 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 12.2x faster than the slowest successful cell (`julia-base`, 235.878 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 14.6x faster than the slowest successful cell (`julia-base`, 235.878 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 54.8x faster than the slowest successful cell (`julia-base`, 1936.674 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 48.0x faster than the slowest successful cell (`julia-base`, 1936.674 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-eager` is 30.8x faster than the slowest successful cell (`julia-base`, 1936.674 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-trace` is 31.0x faster than the slowest successful cell (`julia-base`, 1936.674 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=1, shape=`1024x64`): `pytorch-cpu` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 14.9x faster than the slowest successful cell (`tenferro-eager`, 9.282 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 8110.1x faster than the slowest successful cell (`julia-base`, 13.511 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 8786.8x faster than the slowest successful cell (`julia-base`, 13.910 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 4268.4x faster than the slowest successful cell (`julia-base`, 13.872 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 9041.7x faster than the slowest successful cell (`pytorch-cpu`, 12.053 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 113491.8x faster than the slowest successful cell (`julia-base`, 189.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `strided-jl` is 13.2x faster than the slowest successful cell (`julia-base`, 207.510 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-direct` is 11.9x faster than the slowest successful cell (`julia-base`, 207.510 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 166007.8x faster than the slowest successful cell (`julia-base`, 207.510 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 512.6x faster than the slowest successful cell (`tenferro-direct`, 0.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 395.5x faster than the slowest successful cell (`tenferro-direct`, 0.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 156.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 125.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 142.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 143.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 3048.9 GB/s at 0.005 ms. The conservative physical lower bound is 0.168 ms, so the cell is 30.5x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 11842.5 GB/s at 0.001 ms. The conservative physical lower bound is 0.042 ms, so the cell is 29.6x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 161125.7 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1611.3x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 169573.9 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 423.9x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 82595.5 GB/s at 0.003 ms. The conservative physical lower bound is 2.684 ms, so the cell is 826.0x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 201376.9 GB/s at 0.001 ms. The conservative physical lower bound is 0.671 ms, so the cell is 503.4x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 80514.5 GB/s at 0.002 ms. The conservative physical lower bound is 1.342 ms, so the cell is 805.1x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 107374.2 GB/s at 0.001 ms. The conservative physical lower bound is 0.336 ms, so the cell is 268.4x faster than the bound.

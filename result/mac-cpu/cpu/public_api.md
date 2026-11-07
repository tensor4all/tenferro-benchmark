# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20261107_181900/run.yaml`
- Timestamp: `2026-11-07T18:19:00.910430+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20261107_181900`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

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

- CSV: `data/results/mac-cpu/cpu/public_api/20261107_181900/cpu_public_api_t1_20261107_181900.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20261107_181900/cpu_public_api_t4_20261107_181900.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 3.949 ± 0.665 | - | 3.992 ± 0.295 | 4.689 ± 0.485 | 1.815 ± 0.115 | 6.683 ± 0.802 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 4.382 ± 2.338 | - | 5.652 ± 1.894 | 3.997 ± 0.728 | 1.567 ± 0.369 | 74.062 ± 18.037 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 12.325 ± 2.485 | - | 13.761 ± 3.154 | 13.424 ± 0.873 | 16.000 ± 5.665 | 15.938 ± 2.028 | 23.211 ± 17.287 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 14.825 ± 1.503 | - | 16.505 ± 1.465 | 11.246 ± 5.750 | 17.926 ± 3.788 | 18.232 ± 5.200 | 14.593 ± 1.506 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 53.389 ± 3.800 | - | 49.044 ± 3.970 | 24.484 ± 0.464 | 49.905 ± 10.222 | 53.513 ± 5.380 | 53.330 ± 7.812 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 17.207 ± 3.371 | - | 22.619 ± 11.612 | 11.491 ± 0.751 | 17.877 ± 0.326 | 55.070 ± 5.218 | 32.575 ± 8.311 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 10.516 ± 1.466 | - | 10.528 ± 0.718 | 12.352 ± 2.707 | 49.878 ± 2.556 | 58.942 ± 2.056 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 11.367 ± 1.031 | - | 11.698 ± 1.554 | 11.978 ± 0.638 | 27.582 ± 4.166 | 41.734 ± 17.828 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 11.964 ± 0.548 | - | unsupported | 11.473 ± 0.585 | 56.688 ± 6.916 | 58.687 ± 1.879 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 11.988 ± 1.026 | - | unsupported | 12.020 ± 2.480 | 26.327 ± 7.571 | 43.206 ± 6.247 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 11.458 ± 0.920 | - | 13.651 ± 0.795 | 11.048 ± 1.407 | 24.353 ± 6.538 | 8.826 ± 0.642 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 13.471 ± 4.533 | - | 11.048 ± 1.181 | 11.970 ± 3.823 | 21.282 ± 1.340 | 464.507 ± 27.676 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 47.064 ± 5.847 | - | 75.820 ± 30.327 | 49.803 ± 17.293 | 79.577 ± 15.839 | 47.301 ± 1.147 | 46.870 ± 10.160 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 18.213 ± 4.666 | - | 23.198 ± 2.758 | 18.101 ± 3.635 | 22.389 ± 18.518 | 54.508 ± 4.353 | 27.050 ± 5.603 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 48.991 ± 8.120 | - | 46.577 ± 1.843 | 70.420 ± 13.044 | 96.579 ± 13.437 | 70.429 ± 5.298 | 69.613 ± 0.762 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 24.272 ± 6.418 | - | 24.127 ± 2.958 | 20.052 ± 3.421 | 39.383 ± 18.374 | 95.743 ± 12.491 | 65.075 ± 19.693 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 10.309 ± 1.027 | - | 10.692 ± 3.091 | 10.509 ± 0.338 | 14.426 ± 6.415 | 8.803 ± 1.906 | 12.153 ± 0.833 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 10.885 ± 0.578 | - | 10.528 ± 0.436 | 8.538 ± 0.634 | 11.028 ± 1.781 | 9.942 ± 0.690 | 10.959 ± 0.775 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.269 ± 0.103 | - | 7.713 ± 0.279 | 7.539 ± 0.817 | 4.527 ± 0.118 | 8.549 ± 0.093 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 2.509 ± 1.516 | - | 6.527 ± 1.976 | 4.310 ± 0.708 | 5.247 ± 1.070 | 9.146 ± 4.160 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 8.113 ± 0.177 | - | 9.841 ± 5.105 | 8.830 ± 2.284 | 9.857 ± 4.299 | 9.717 ± 0.773 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 8.448 ± 0.156 | - | 7.191 ± 0.764 | 7.376 ± 0.771 | 8.581 ± 0.951 | 156.138 ± 25.091 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 7.276 ± 2.177 | - | 6.492 ± 1.413 | 6.127 ± 0.439 | 2.380 ± 0.690 | 8.097 ± 3.043 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 6.920 ± 2.363 | - | 7.828 ± 2.788 | 6.039 ± 0.418 | 2.331 ± 0.093 | 307.311 ± 78.219 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 6.455 ± 0.206 | - | 7.753 ± 0.747 | 7.890 ± 0.928 | 6.879 ± 0.212 | 7.395 ± 1.693 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 6.824 ± 0.397 | - | 8.381 ± 0.462 | 7.462 ± 0.226 | 7.174 ± 1.425 | 629.773 ± 639.101 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 10.997 ± 1.090 | - | 11.760 ± 2.729 | 15.258 ± 3.641 | 63.937 ± 12.524 | 59.635 ± 1.372 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 12.213 ± 1.204 | - | 11.977 ± 1.211 | 11.532 ± 1.285 | 42.152 ± 5.108 | 43.030 ± 11.154 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 21.107 ± 7.517 | - | unsupported | 9.987 ± 0.504 | 54.863 ± 12.947 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 9.953 ± 0.493 | - | unsupported | 11.957 ± 0.549 | 29.719 ± 3.685 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 13.261 ± 3.107 | - | 18.742 ± 6.472 | 14.821 ± 1.990 | 17.637 ± 3.573 | 12.274 ± 1.587 | 13.548 ± 1.202 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 11.137 ± 0.626 | - | 10.635 ± 0.327 | 11.153 ± 2.866 | 15.713 ± 3.260 | 12.245 ± 1.160 | 12.943 ± 2.908 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 26.278 ± 5.800 | - | 22.842 ± 9.770 | 20.840 ± 5.214 | 27.160 ± 3.934 | 29.532 ± 25.760 | 21.375 ± 6.769 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 15.647 ± 4.413 | - | 15.018 ± 1.687 | 18.842 ± 4.488 | 22.766 ± 2.004 | 20.457 ± 5.331 | 25.531 ± 7.828 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 28.121 ± 1.220 | - | 35.010 ± 1.380 | 49.892 ± 15.721 | 34.575 ± 3.861 | 50.043 ± 10.927 | 65.544 ± 7.492 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 16.303 ± 3.509 | - | 20.213 ± 7.173 | 19.668 ± 0.652 | 14.420 ± 0.165 | 53.343 ± 4.215 | 25.602 ± 8.975 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 6.867 ± 2.003 | - | 6.681 ± 0.799 | 7.288 ± 0.594 | 9.569 ± 1.826 | 8.925 ± 1.703 | 9.103 ± 23.250 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 6.324 ± 0.571 | - | 6.098 ± 0.338 | 9.456 ± 1.156 | 8.115 ± 0.841 | 6.438 ± 1.503 | 7.241 ± 1.400 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 21.502 ± 4.010 | - | 21.621 ± 3.463 | 20.326 ± 4.874 | 16.416 ± 1.667 | 13.819 ± 4.891 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 12.586 ± 1.421 | - | 12.478 ± 0.383 | 15.800 ± 2.014 | 15.293 ± 1.173 | 15.811 ± 2.591 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 34.664 ± 2.613 | - | 42.610 ± 8.273 | 33.874 ± 1.978 | 30.820 ± 0.471 | 38.525 ± 4.765 | 41.901 ± 6.094 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 15.579 ± 1.181 | - | 12.874 ± 5.108 | 15.825 ± 2.732 | 17.089 ± 1.484 | 40.136 ± 3.559 | 22.222 ± 7.688 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 28.339 ± 7.439 | - | 26.778 ± 5.005 | 17.295 ± 1.307 | 29.459 ± 4.181 | 21.422 ± 4.654 | 20.700 ± 1.711 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 14.894 ± 0.488 | - | 15.289 ± 0.491 | 21.865 ± 5.206 | 27.481 ± 4.281 | 23.247 ± 35.062 | 23.296 ± 4.159 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 23.267 ± 9.356 | - | 23.058 ± 2.482 | 30.071 ± 2.343 | 23.107 ± 3.821 | 47.203 ± 16.603 | 43.038 ± 4.751 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 8.620 ± 0.377 | - | 8.595 ± 0.709 | 16.427 ± 4.621 | 9.044 ± 0.493 | 35.573 ± 7.933 | 18.990 ± 2.632 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 30.402 ± 8.560 | - | 33.995 ± 3.720 | 29.090 ± 3.346 | 19.233 ± 3.311 | 30.870 ± 5.611 | 31.285 ± 1.179 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 11.097 ± 1.463 | - | 14.097 ± 4.321 | 15.766 ± 6.523 | 8.558 ± 0.334 | 27.386 ± 13.552 | 12.920 ± 4.594 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 36.124 ± 1.649 | - | 34.702 ± 5.315 | 51.333 ± 12.006 | 41.558 ± 26.756 | 49.859 ± 27.136 | 45.574 ± 0.309 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 13.874 ± 1.940 | - | 17.834 ± 5.754 | 19.633 ± 6.933 | 13.494 ± 0.444 | 54.940 ± 27.208 | 22.220 ± 4.057 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 17.350 ± 1.001 | - | 18.271 ± 4.950 | 22.136 ± 0.361 | 17.648 ± 0.246 | 24.387 ± 8.180 | 25.245 ± 2.086 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 10.222 ± 1.657 | - | 10.199 ± 4.633 | 9.479 ± 0.758 | 9.110 ± 0.176 | 25.975 ± 3.317 | 13.614 ± 4.836 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.292 ± 0.243 | 0.383 ± 0.134 | unsupported | unsupported | 0.299 ± 0.051 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.252 ± 0.008 | 0.273 ± 0.013 | unsupported | unsupported | 0.192 ± 0.022 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 26.647 ± 10.009 | - | 37.406 ± 16.694 | 21.942 ± 5.263 | 28.281 ± 2.551 | 26.952 ± 46.664 | 22.073 ± 0.837 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 16.863 ± 0.577 | - | 16.596 ± 0.724 | 21.817 ± 2.421 | 26.865 ± 1.297 | 19.528 ± 1.107 | 23.537 ± 15.285 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 34.729 ± 18.261 | - | 26.182 ± 7.244 | 25.076 ± 4.069 | 26.583 ± 0.870 | 18.576 ± 2.014 | 20.932 ± 4.849 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 16.907 ± 0.580 | - | 17.646 ± 1.213 | 20.885 ± 1.838 | 29.283 ± 5.210 | 20.632 ± 6.630 | 18.572 ± 3.793 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 29.640 ± 11.676 | - | 23.694 ± 8.182 | 18.267 ± 2.862 | 30.022 ± 0.963 | 20.025 ± 0.959 | 22.256 ± 1.605 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 16.156 ± 1.324 | - | 15.463 ± 1.293 | 19.797 ± 2.128 | 21.818 ± 3.166 | 18.236 ± 2.062 | 21.083 ± 3.894 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 20.247 ± 4.055 | - | 14.064 ± 3.765 | 21.055 ± 4.675 | 18.046 ± 1.052 | 13.530 ± 1.309 | 13.942 ± 3.105 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 11.426 ± 1.179 | - | 10.796 ± 0.944 | 16.906 ± 3.461 | 14.439 ± 3.850 | 12.418 ± 0.600 | 15.009 ± 2.728 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 52.583 ± 6.177 | - | 45.375 ± 4.739 | 72.245 ± 4.592 | 47.474 ± 1.745 | 91.073 ± 43.671 | 81.470 ± 9.592 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 18.571 ± 3.517 | - | 17.108 ± 6.489 | 37.992 ± 7.744 | 18.592 ± 0.321 | 87.098 ± 14.735 | 33.772 ± 3.947 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 7.619 ± 1.211 | - | 6.686 ± 0.162 | 14.238 ± 1.789 | 21.215 ± 0.051 | 6.677 ± 0.383 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 4.965 ± 0.404 | - | 4.996 ± 0.086 | 6.862 ± 0.248 | 11.207 ± 0.059 | 8.671 ± 8.154 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 0.906 ± 0.033 | - | 0.965 ± 0.078 | 19.390 ± 1.062 | 2.032 ± 0.209 | 0.787 ± 0.048 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 1.032 ± 0.474 | - | 0.752 ± 0.047 | 11.579 ± 0.906 | 1.685 ± 0.191 | 0.741 ± 0.020 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 1.003 ± 0.260 | - | 0.837 ± 0.020 | 2.285 ± 0.344 | 1.012 ± 0.132 | 0.765 ± 0.043 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 1.333 ± 0.044 | - | 1.352 ± 0.031 | 1.079 ± 0.093 | 0.956 ± 0.053 | 0.772 ± 0.024 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 8.225 ± 0.999 | - | 8.773 ± 1.429 | 13.191 ± 1.148 | 21.179 ± 0.142 | 7.665 ± 1.144 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 5.005 ± 0.048 | - | 5.574 ± 0.433 | 6.973 ± 0.151 | 11.525 ± 0.604 | 9.194 ± 4.617 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 0.952 ± 0.205 | - | 0.995 ± 0.102 | 15.731 ± 1.088 | 2.432 ± 0.357 | 0.685 ± 0.030 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.720 ± 0.024 | - | 0.730 ± 0.010 | 11.144 ± 0.615 | 1.605 ± 0.125 | 0.925 ± 0.044 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 3.525 ± 0.265 | - | 4.976 ± 0.567 | 7.194 ± 0.840 | 3.808 ± 0.934 | 2.991 ± 0.062 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 5.412 ± 0.183 | - | 5.091 ± 0.041 | 3.962 ± 0.493 | 3.192 ± 0.237 | 3.283 ± 0.220 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 6.826 ± 1.089 | - | 6.833 ± 0.721 | 8.335 ± 1.761 | 38.511 ± 0.674 | 6.315 ± 0.982 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 6.003 ± 0.461 | - | 4.951 ± 0.319 | 6.043 ± 0.172 | 20.765 ± 0.901 | 8.582 ± 4.154 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.903 ± 0.026 | - | 0.972 ± 0.058 | 4.214 ± 0.190 | 2.489 ± 1.142 | 0.858 ± 0.091 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.706 ± 0.039 | - | 0.795 ± 0.042 | 2.576 ± 0.258 | 1.744 ± 0.167 | 0.988 ± 0.143 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 0.891 ± 0.099 | - | 1.224 ± 0.430 | 0.740 ± 0.035 | 1.174 ± 0.124 | 1.017 ± 0.286 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 1.788 ± 0.071 | - | 1.422 ± 0.388 | 0.765 ± 0.068 | 0.924 ± 0.176 | 0.913 ± 0.161 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 5.992 ± 0.705 | - | 5.735 ± 0.825 | 8.940 ± 2.454 | 33.730 ± 2.214 | 6.890 ± 1.130 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 6.710 ± 2.208 | - | 6.101 ± 0.421 | 5.844 ± 0.930 | 16.610 ± 0.764 | 7.594 ± 0.594 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.794 ± 0.081 | - | 0.960 ± 0.175 | 4.348 ± 0.167 | 2.245 ± 0.071 | 0.663 ± 0.015 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.682 ± 0.068 | - | 0.714 ± 0.015 | 2.422 ± 0.164 | 1.588 ± 0.142 | 0.721 ± 0.027 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 0.891 ± 0.172 | - | 0.973 ± 0.289 | 0.782 ± 0.128 | 0.986 ± 0.027 | 0.770 ± 0.042 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 1.397 ± 0.179 | - | 1.447 ± 0.030 | 0.686 ± 0.035 | 0.910 ± 0.027 | 0.797 ± 0.055 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.881 ± 0.058 | 0.948 ± 0.087 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.678 ± 0.024 | 0.685 ± 0.008 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 27.190 ± 0.318 | - | 31.705 ± 1.228 | 32.060 ± 2.160 | 51.703 ± 3.459 | 34.998 ± 43.420 | 33.346 ± 11.357 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 11.976 ± 1.002 | - | 11.715 ± 1.412 | 12.341 ± 3.030 | 20.163 ± 0.495 | 28.153 ± 3.007 | 15.796 ± 2.601 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 23.086 ± 8.308 | - | 18.179 ± 0.679 | 19.904 ± 2.719 | 35.824 ± 1.303 | 59.950 ± 19.405 | 62.869 ± 79.532 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 10.117 ± 0.438 | - | 10.544 ± 0.418 | 15.248 ± 1.726 | 17.735 ± 0.089 | 48.256 ± 1.182 | 19.442 ± 1.769 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 38.211 ± 1.944 | - | 35.354 ± 15.005 | 34.426 ± 8.231 | 27.502 ± 4.332 | 35.187 ± 49.279 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 25.759 ± 0.982 | - | 24.058 ± 0.751 | 21.274 ± 2.363 | 29.002 ± 3.505 | 28.429 ± 2.588 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 13.575 ± 4.367 | - | 21.392 ± 7.324 | 14.694 ± 1.839 | 17.682 ± 1.101 | 89.963 ± 105.161 | 20.428 ± 4.497 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 10.619 ± 0.241 | - | 11.045 ± 1.232 | 13.780 ± 2.109 | 16.944 ± 2.326 | 13.614 ± 2.421 | 12.285 ± 1.882 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 27.132 ± 6.651 | - | 24.346 ± 3.406 | 34.749 ± 7.845 | 31.302 ± 5.513 | 35.157 ± 4.903 | 33.203 ± 3.078 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 12.902 ± 2.025 | - | 12.313 ± 1.278 | 16.887 ± 3.984 | 16.649 ± 4.020 | 32.514 ± 1.461 | 17.204 ± 7.153 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 12.418 ± 2.674 | - | 13.101 ± 1.898 | 14.416 ± 0.984 | 23.633 ± 1.441 | 30.759 ± 5.864 | 28.407 ± 1.237 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 9.699 ± 0.065 | - | 10.113 ± 0.625 | 13.298 ± 4.513 | 10.799 ± 1.566 | 30.462 ± 0.816 | 14.438 ± 1.639 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 17.585 ± 1.762 | - | 19.131 ± 1.839 | 20.968 ± 4.772 | 26.286 ± 6.545 | 21.925 ± 31.044 | 21.334 ± 1.376 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 18.438 ± 6.773 | - | 15.633 ± 3.775 | 23.239 ± 4.287 | 19.983 ± 2.931 | 17.670 ± 3.318 | 17.059 ± 2.303 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 81.129 ± 13.150 | - | 85.525 ± 21.766 | 71.963 ± 11.285 | 14.457 ± 0.373 | 48.703 ± 23.590 | 47.717 ± 12.147 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 26.576 ± 7.603 | - | 33.439 ± 2.691 | 29.930 ± 9.553 | 8.299 ± 1.462 | 51.975 ± 3.020 | 25.620 ± 7.454 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 31.408 ± 3.708 | 23.512 ± 1.365 | unsupported | 25.525 ± 5.359 | 27.495 ± 2.677 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 23.952 ± 0.281 | 24.880 ± 2.815 | unsupported | 22.429 ± 2.315 | 23.065 ± 1.004 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.835 ± 0.074 | - | 0.828 ± 0.032 | 0.668 ± 0.045 | 0.960 ± 0.147 | 0.909 ± 0.148 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.630 ± 0.073 | - | 0.718 ± 0.029 | 0.857 ± 0.148 | 0.827 ± 0.129 | 0.818 ± 0.067 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.763 ± 0.058 | - | 0.800 ± 0.070 | 0.689 ± 0.043 | 1.194 ± 0.182 | 0.904 ± 0.152 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.679 ± 0.144 | - | 0.660 ± 0.107 | 0.641 ± 0.018 | 0.882 ± 0.191 | 0.898 ± 0.114 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.239 ± 0.072 | - | unsupported | 1.169 ± 0.135 | 1.514 ± 0.240 | 1.221 ± 0.349 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.912 ± 0.057 | - | unsupported | 1.117 ± 0.070 | 1.298 ± 0.160 | 1.170 ± 0.282 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.346 ± 0.043 | - | 0.518 ± 0.057 | 0.300 ± 0.015 | 0.391 ± 0.090 | 2.523 ± 1.882 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.297 ± 0.035 | - | 0.350 ± 0.010 | 0.306 ± 0.010 | 0.273 ± 0.063 | 1.137 ± 0.913 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.832 ± 0.025 | - | 0.815 ± 0.078 | 1.153 ± 0.116 | 1.001 ± 0.240 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.736 ± 0.111 | - | 0.600 ± 0.057 | 1.082 ± 0.113 | 1.167 ± 0.324 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.860 ± 0.069 | - | 0.797 ± 0.064 | 0.877 ± 0.181 | 0.973 ± 0.032 | 2.877 ± 1.299 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.681 ± 0.045 | - | 0.641 ± 0.106 | 0.832 ± 0.031 | 1.107 ± 0.082 | 1.531 ± 0.429 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.686 ± 0.023 | - | 0.775 ± 0.056 | 0.402 ± 0.041 | 0.566 ± 0.217 | 0.360 ± 0.029 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.564 ± 0.010 | - | 0.661 ± 0.124 | 0.470 ± 0.036 | 0.498 ± 0.029 | 1.181 ± 1.634 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.923 ± 0.499 | - | 0.001 ± 0.000 | 1.154 ± 0.078 | 1.464 ± 0.145 | 1.531 ± 0.382 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 1.020 ± 0.108 | - | 0.002 ± 0.000 | 1.121 ± 0.089 | 1.525 ± 0.163 | 1.465 ± 0.358 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.064 ± 0.284 | - | 0.082 ± 0.023 | 0.141 ± 0.001 | 0.104 ± 0.088 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.095 ± 0.003 | - | 0.120 ± 0.002 | 0.194 ± 0.002 | 0.134 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.201 ± 0.001 | - | 0.225 ± 0.062 | 0.293 ± 0.008 | 0.254 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.026 ± 0.001 | - | 0.028 ± 0.002 | 0.040 ± 0.003 | 0.128 ± 0.067 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.003 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.089 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.006 ± 0.001 | - | 0.012 ± 0.003 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.006 ± 0.001 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.041 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.174 ± 0.006 | - | 0.334 ± 0.116 | 0.246 ± 0.001 | 0.363 ± 0.335 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.014 ± 0.000 | - | 0.017 ± 0.001 | 0.017 ± 0.002 | 0.105 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.108 ± 0.121 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.055 ± 0.000 | - | 0.075 ± 0.000 | 0.160 ± 0.057 | 0.134 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.088 ± 0.000 | - | 0.113 ± 0.000 | 0.219 ± 0.119 | 0.183 ± 0.119 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.188 ± 0.007 | - | 0.220 ± 0.008 | 0.286 ± 0.031 | 0.386 ± 0.133 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.025 ± 0.000 | - | 0.029 ± 0.002 | 0.129 ± 0.051 | 0.127 ± 0.089 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.048 ± 0.026 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.067 ± 0.014 | 0.009 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.066 ± 0.022 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.075 ± 0.033 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.050 ± 0.016 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.056 ± 0.008 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.062 ± 0.032 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.053 ± 0.017 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.059 ± 0.038 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.051 ± 0.029 | 0.066 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.164 ± 0.003 | - | 0.170 ± 0.000 | 0.291 ± 0.113 | 0.548 ± 0.141 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.060 ± 0.021 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.011 ± 0.001 | - | 0.014 ± 0.000 | 0.077 ± 0.062 | 0.136 ± 0.072 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.038 ± 0.037 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.110 ± 0.053 | 0.081 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.260 ± 0.025 | - | 0.727 ± 0.042 | 0.419 ± 0.011 | 1.087 ± 0.138 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 2.299 ± 1.199 | - | 1.877 ± 0.399 | 1.746 ± 0.074 | 2.472 ± 0.123 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 7.661 ± 5.011 | - | 7.939 ± 2.080 | 6.674 ± 0.142 | 7.982 ± 2.087 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 5.357 ± 0.330 | - | 7.611 ± 2.718 | 3.306 ± 0.109 | 5.211 ± 1.494 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.012 ± 0.003 | - | 0.027 ± 0.008 | 0.005 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.134 ± 0.068 | - | 0.100 ± 0.014 | 0.098 ± 0.011 | 0.344 ± 0.278 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.008 ± 0.000 | - | 0.013 ± 0.001 | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.009 ± 0.004 | - | 0.013 ± 0.024 | 0.010 ± 0.000 | 0.018 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.010 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.016 ± 0.000 | - | 0.021 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.058 ± 0.030 | - | 0.030 ± 0.005 | 0.023 ± 0.001 | 0.071 ± 0.062 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.006 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.008 ± 0.012 | - | 0.008 ± 0.003 | 0.011 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.074 ± 0.005 | - | 0.093 ± 0.029 | 0.029 ± 0.001 | 0.075 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 48.029 ± 4.355 | - | 58.459 ± 5.106 | 26.498 ± 0.390 | 37.587 ± 2.350 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.004 ± 0.001 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.752 ± 0.119 | - | 0.845 ± 0.147 | 0.714 ± 0.018 | 1.146 ± 0.410 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.015 ± 0.003 | - | 0.016 ± 0.003 | 0.018 ± 0.000 | 0.026 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.062 ± 0.096 | - | 0.064 ± 0.008 | 0.054 ± 0.003 | 0.110 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.284 ± 0.023 | - | 0.294 ± 0.021 | 0.426 ± 0.020 | 0.818 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 1.646 ± 0.115 | - | 1.576 ± 0.086 | 1.754 ± 0.144 | 2.559 ± 0.968 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 6.076 ± 0.098 | - | 6.144 ± 0.191 | 7.325 ± 0.906 | 3.942 ± 0.096 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 4.416 ± 0.025 | - | 5.344 ± 1.450 | 3.264 ± 0.417 | 5.241 ± 2.215 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.092 ± 0.001 | - | 0.093 ± 0.002 | 0.093 ± 0.002 | 0.304 ± 0.274 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.001 | 0.006 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.009 ± 0.001 | - | 0.009 ± 0.001 | 0.014 ± 0.007 | 0.056 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.001 | 0.006 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.007 ± 0.000 | - | 0.008 ± 0.002 | 0.009 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.024 ± 0.003 | - | 0.021 ± 0.001 | 0.029 ± 0.006 | 0.109 ± 0.062 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.006 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.010 ± 0.001 | - | 0.011 ± 0.002 | 0.011 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.025 ± 0.005 | - | 0.027 ± 0.004 | 0.036 ± 0.008 | 0.329 ± 0.905 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 35.629 ± 0.479 | - | 45.323 ± 5.700 | 27.594 ± 0.787 | 24.189 ± 1.998 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.715 ± 0.015 | - | 0.717 ± 0.024 | 0.829 ± 0.476 | 0.809 ± 0.259 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.016 ± 0.001 | - | 0.015 ± 0.004 | 0.017 ± 0.000 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.052 ± 0.005 | - | 0.049 ± 0.002 | 0.054 ± 0.000 | 0.129 ± 0.059 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.299 ± 0.208 | - | 0.291 ± 0.192 | 0.239 ± 0.010 | 0.715 ± 0.289 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 2.151 ± 1.973 | - | 1.740 ± 2.218 | 1.255 ± 0.032 | 2.154 ± 1.047 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 7.940 ± 2.950 | - | 6.064 ± 0.318 | 5.158 ± 0.178 | 6.276 ± 1.073 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.149 ± 0.048 | - | 1.092 ± 0.056 | 1.001 ± 0.042 | 1.192 ± 0.269 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.072 ± 0.071 | - | 0.068 ± 0.046 | 0.071 ± 0.007 | 0.110 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.006 | 0.006 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.007 ± 0.001 | - | 0.007 ± 0.001 | 0.008 ± 0.000 | 0.022 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.007 ± 0.001 | - | 0.008 ± 0.001 | 0.008 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.016 ± 0.000 | - | 0.017 ± 0.031 | 0.019 ± 0.001 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.010 ± 0.002 | 0.005 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.007 ± 0.000 | - | 0.010 ± 0.002 | 0.009 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.024 ± 0.008 | - | 0.028 ± 0.004 | 0.023 ± 0.001 | 0.032 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 10.063 ± 1.158 | - | 8.854 ± 0.287 | 7.994 ± 0.295 | 8.911 ± 0.871 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.008 ± 0.000 | - | 0.005 ± 0.001 | 0.006 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.534 ± 0.079 | - | 0.566 ± 0.045 | 0.520 ± 0.007 | 0.723 ± 0.266 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.012 ± 0.003 | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.039 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.044 ± 0.007 | - | 0.051 ± 0.010 | 0.048 ± 0.010 | 0.092 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.256 ± 0.011 | - | 0.258 ± 0.019 | 0.267 ± 0.115 | 1.248 ± 0.442 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.284 ± 0.061 | - | 1.295 ± 0.026 | 1.491 ± 0.557 | 1.714 ± 0.350 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 5.021 ± 0.073 | - | 5.295 ± 0.175 | 5.471 ± 0.825 | 3.749 ± 0.422 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 0.994 ± 0.043 | - | 1.000 ± 0.019 | 1.146 ± 0.357 | 1.283 ± 0.123 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.002 ± 0.001 | - | 0.004 ± 0.001 | 0.006 ± 0.001 | 0.010 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.069 ± 0.001 | - | 0.068 ± 0.006 | 0.071 ± 0.007 | 0.163 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.002 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.006 ± 0.002 | - | 0.008 ± 0.001 | 0.012 ± 0.006 | 0.018 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.006 ± 0.001 | - | 0.007 ± 0.002 | 0.008 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.016 ± 0.004 | - | 0.017 ± 0.000 | 0.025 ± 0.002 | 0.031 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.008 ± 0.002 | - | 0.009 ± 0.001 | 0.010 ± 0.002 | 0.016 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.021 ± 0.001 | - | 0.027 ± 0.005 | 0.025 ± 0.017 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 8.108 ± 0.092 | - | 8.200 ± 0.083 | 8.501 ± 2.167 | 5.098 ± 0.182 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.001 | 0.006 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.528 ± 0.034 | - | 0.529 ± 0.017 | 0.530 ± 0.047 | 0.511 ± 0.074 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.012 ± 0.003 | - | 0.013 ± 0.000 | 0.014 ± 0.003 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.039 ± 0.001 | - | 0.041 ± 0.000 | 0.051 ± 0.009 | 0.131 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.153 ± 0.004 | - | 0.160 ± 0.021 | 0.126 ± 0.004 | 0.928 ± 0.549 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 0.779 ± 0.058 | - | 0.788 ± 0.493 | 0.771 ± 0.031 | 2.011 ± 1.612 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 2.787 ± 1.312 | - | 3.132 ± 0.753 | 2.422 ± 0.100 | 12.137 ± 2.299 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.587 ± 0.047 | - | 0.575 ± 0.048 | 0.452 ± 0.003 | 1.198 ± 0.254 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.002 ± 0.001 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.031 ± 0.004 | - | 0.034 ± 0.006 | 0.030 ± 0.001 | 0.134 ± 0.151 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.010 ± 0.136 | 0.005 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.012 ± 0.002 | 0.068 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.004 ± 0.001 | 0.005 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.032 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 4.867 ± 0.481 | - | 5.063 ± 0.636 | 3.707 ± 0.108 | 12.862 ± 4.619 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.206 ± 0.016 | - | 0.220 ± 0.083 | 0.203 ± 0.033 | 0.623 ± 1.134 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.057 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.019 ± 0.004 | - | 0.022 ± 0.000 | 0.023 ± 0.000 | 0.184 ± 0.158 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.115 ± 0.003 | - | 0.123 ± 0.006 | 0.148 ± 0.017 | 0.753 ± 0.706 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 0.731 ± 0.025 | - | 0.728 ± 0.004 | 0.794 ± 0.050 | 1.257 ± 0.681 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 2.352 ± 0.053 | - | 2.347 ± 0.027 | 2.760 ± 0.245 | 2.848 ± 0.162 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.447 ± 0.015 | - | 0.447 ± 0.013 | 0.514 ± 0.065 | 1.092 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.026 ± 0.001 | - | 0.027 ± 0.001 | 0.032 ± 0.008 | 0.133 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.004 ± 0.001 | - | 0.005 ± 0.000 | 0.008 ± 0.002 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.011 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.004 ± 0.001 | - | 0.005 ± 0.000 | 0.009 ± 0.001 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.010 ± 0.002 | - | 0.010 ± 0.000 | 0.014 ± 0.003 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.002 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.002 | 0.010 ± 0.002 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.013 ± 0.003 | - | 0.012 ± 0.000 | 0.014 ± 0.004 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 3.569 ± 0.032 | - | 3.589 ± 0.017 | 4.252 ± 0.439 | 4.627 ± 0.414 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.002 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.190 ± 0.000 | - | 0.192 ± 0.001 | 0.205 ± 0.010 | 0.336 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.008 ± 0.002 | - | 0.009 ± 0.002 | 0.013 ± 0.002 | 0.024 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.020 ± 0.000 | - | 0.021 ± 0.000 | 0.024 ± 0.006 | 0.078 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.107 ± 0.022 | - | 0.111 ± 0.039 | 0.180 ± 0.001 | 0.158 ± 0.149 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.215 ± 0.139 | - | 0.206 ± 0.029 | 0.268 ± 0.011 | 0.300 ± 0.167 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.458 ± 0.542 | - | 1.081 ± 0.132 | 0.511 ± 0.012 | 0.493 ± 0.903 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.157 ± 0.111 | - | 0.062 ± 0.001 | 0.064 ± 0.006 | 0.136 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.006 ± 0.005 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.010 ± 0.002 | - | 0.010 ± 0.003 | 0.011 ± 0.000 | 0.108 ± 0.064 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.010 ± 0.002 | - | 0.015 ± 0.089 | 0.013 ± 0.000 | 0.113 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.004 ± 0.001 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.010 ± 0.001 | 0.008 ± 0.000 | 0.051 ± 0.073 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.001 | 0.008 ± 0.000 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.001 | 0.066 ± 0.081 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.003 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.003 | 0.009 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.104 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 1.068 ± 0.916 | - | 0.940 ± 0.481 | 0.443 ± 0.014 | 2.006 ± 1.276 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.001 | 0.009 ± 0.000 | 0.037 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.050 ± 0.009 | - | 0.051 ± 0.000 | 0.039 ± 0.000 | 0.090 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.072 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.005 ± 0.000 | - | 0.008 ± 0.001 | 0.011 ± 0.000 | 0.048 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.109 ± 0.008 | - | 0.088 ± 0.001 | 0.249 ± 0.052 | 0.171 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.188 ± 0.001 | - | 0.189 ± 0.001 | 0.285 ± 0.022 | 0.244 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.421 ± 0.016 | - | 0.423 ± 0.002 | 0.531 ± 0.170 | 0.349 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.057 ± 0.002 | - | 0.060 ± 0.004 | 0.130 ± 0.015 | 0.166 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.004 | 0.121 ± 0.071 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.006 ± 0.001 | - | 0.007 ± 0.002 | 0.070 ± 0.034 | 0.027 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.001 | 0.071 ± 0.041 | 0.013 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.011 ± 0.003 | - | 0.010 ± 0.000 | 0.082 ± 0.029 | 0.078 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.070 ± 0.028 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.004 ± 0.001 | - | 0.005 ± 0.001 | 0.117 ± 0.041 | 0.010 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.072 ± 0.006 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.002 | 0.077 ± 0.018 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.097 ± 0.056 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.067 ± 0.029 | 0.011 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.001 | 0.063 ± 0.021 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.438 ± 0.013 | - | 0.452 ± 0.011 | 0.468 ± 0.053 | 0.889 ± 0.309 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.004 ± 0.001 | - | 0.005 ± 0.001 | 0.131 ± 0.107 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.032 ± 0.000 | - | 0.033 ± 0.001 | 0.133 ± 0.050 | 0.141 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.001 | 0.078 ± 0.058 | 0.017 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.006 ± 0.001 | - | 0.007 ± 0.002 | 0.122 ± 0.140 | 0.025 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.589 ± 0.081 | - | 0.223 ± 0.072 | 0.312 ± 0.010 | 0.630 ± 0.137 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 1.360 ± 0.062 | - | 1.006 ± 0.738 | 0.666 ± 0.047 | 0.861 ± 0.263 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.580 ± 1.140 | - | 1.635 ± 0.457 | 1.572 ± 0.028 | 1.800 ± 0.094 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.372 ± 0.111 | - | 0.327 ± 0.061 | 0.351 ± 0.002 | 0.442 ± 0.069 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.011 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.021 ± 0.006 | - | 0.031 ± 0.007 | 0.020 ± 0.000 | 0.058 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.015 ± 0.010 | - | 0.016 ± 0.004 | 0.018 ± 0.004 | 0.067 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | 0.006 ± 0.002 | 0.005 ± 0.000 | 0.014 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.016 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.012 ± 0.038 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.009 ± 0.002 | 0.056 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.009 ± 0.000 | - | 0.013 ± 0.000 | 0.007 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.021 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 3.058 ± 0.269 | - | 3.057 ± 0.865 | 2.890 ± 0.194 | 3.506 ± 0.580 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.034 ± 0.111 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.092 ± 0.007 | - | 0.164 ± 0.192 | 0.148 ± 0.010 | 0.180 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.014 ± 0.000 | - | 0.018 ± 0.000 | 0.008 ± 0.000 | 0.021 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.014 ± 0.000 | - | 0.019 ± 0.004 | 0.015 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.189 ± 0.000 | - | 0.192 ± 0.007 | 0.376 ± 0.073 | 0.637 ± 0.162 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.472 ± 0.039 | - | 0.475 ± 0.005 | 0.725 ± 0.218 | 1.636 ± 0.293 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.162 ± 0.015 | - | 1.163 ± 0.027 | 1.908 ± 0.533 | 1.553 ± 0.086 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.293 ± 0.003 | - | 0.295 ± 0.003 | 0.434 ± 0.086 | 0.442 ± 0.111 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.001 | 0.032 ± 0.068 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.012 ± 0.000 | - | 0.013 ± 0.000 | 0.053 ± 0.144 | 0.212 ± 0.391 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.024 ± 0.016 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.016 ± 0.003 | - | 0.016 ± 0.001 | 0.042 ± 0.021 | 0.104 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.003 ± 0.001 | - | 0.005 ± 0.002 | 0.075 ± 0.010 | 0.045 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.002 ± 0.001 | - | 0.005 ± 0.001 | 0.053 ± 0.044 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.003 ± 0.001 | - | 0.006 ± 0.000 | 0.032 ± 0.020 | 0.015 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.002 | 0.062 ± 0.037 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.036 ± 0.017 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.026 ± 0.004 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.008 ± 0.002 | - | 0.008 ± 0.002 | 0.041 ± 0.087 | 0.050 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 2.377 ± 0.204 | - | 2.451 ± 0.085 | 2.965 ± 0.096 | 2.970 ± 1.183 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.041 ± 0.010 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.085 ± 0.000 | - | 0.086 ± 0.000 | 0.171 ± 0.073 | 0.210 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.006 ± 0.001 | - | 0.007 ± 0.002 | 0.039 ± 0.016 | 0.048 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.013 ± 0.003 | - | 0.012 ± 0.000 | 0.055 ± 0.019 | 0.032 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.156 ± 0.007 | - | 0.228 ± 0.081 | 0.106 ± 0.001 | 0.192 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.213 ± 0.042 | - | 0.196 ± 0.066 | 0.194 ± 0.016 | 0.296 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.414 ± 0.010 | - | 0.442 ± 0.036 | 0.458 ± 0.166 | 0.535 ± 0.879 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.052 ± 0.002 | - | 0.059 ± 0.003 | 0.065 ± 0.002 | 0.121 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.013 ± 0.010 | 0.010 ± 0.000 | 0.009 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.006 ± 0.000 | - | 0.015 ± 0.002 | 0.048 ± 0.019 | 0.046 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.009 ± 0.002 | 0.010 ± 0.002 | 0.014 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.010 ± 0.002 | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.004 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.010 ± 0.000 | 0.011 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.010 ± 0.000 | 0.013 ± 0.002 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.021 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.395 ± 0.007 | - | 0.439 ± 0.022 | 0.484 ± 0.151 | 0.562 ± 0.085 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.038 ± 0.004 | - | 0.048 ± 0.007 | 0.064 ± 0.034 | 0.094 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.036 | 0.011 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.010 ± 0.000 | 0.012 ± 0.009 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.135 ± 0.002 | - | 0.120 ± 0.001 | 0.134 ± 0.011 | 0.359 ± 0.291 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.195 ± 0.001 | - | 0.208 ± 0.012 | 0.176 ± 0.032 | 0.362 ± 0.152 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.345 ± 0.000 | - | 0.326 ± 0.002 | 0.315 ± 0.012 | 0.769 ± 0.231 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.057 ± 0.003 | - | 0.064 ± 0.006 | 0.065 ± 0.002 | 0.146 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.012 | 0.013 ± 0.001 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.006 ± 0.001 | - | 0.026 ± 0.001 | 0.013 ± 0.001 | 0.079 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.010 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.001 | 0.012 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.002 ± 0.001 | - | 0.006 ± 0.001 | 0.010 ± 0.000 | 0.015 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.003 ± 0.001 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.002 ± 0.001 | - | 0.007 ± 0.000 | 0.010 ± 0.002 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.003 ± 0.001 | - | 0.007 ± 0.001 | 0.010 ± 0.000 | 0.015 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.400 ± 0.022 | - | 0.427 ± 0.022 | 0.342 ± 0.128 | 0.820 ± 0.158 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.041 ± 0.011 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.041 ± 0.001 | - | 0.055 ± 0.009 | 0.073 ± 0.004 | 0.210 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.007 ± 0.001 | 0.053 ± 0.026 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.005 ± 0.001 | - | 0.009 ± 0.000 | 0.055 ± 0.019 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.546 ± 0.041 | - | 1.188 ± 0.068 | 1.581 ± 0.378 | 3.675 ± 0.897 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 4.482 ± 0.633 | - | 5.249 ± 1.709 | 4.386 ± 0.327 | 8.608 ± 2.488 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 17.234 ± 2.224 | - | 16.193 ± 1.558 | 15.059 ± 0.109 | 23.646 ± 10.296 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 6.565 ± 0.233 | - | 5.992 ± 0.086 | 4.591 ± 0.098 | 6.104 ± 0.988 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.002 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.179 ± 0.011 | - | 0.238 ± 0.083 | 0.193 ± 0.015 | 0.540 ± 0.249 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.202 ± 0.162 | - | 0.229 ± 0.111 | 0.170 ± 0.011 | 0.247 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.017 ± 0.000 | - | 0.018 ± 0.000 | 0.022 ± 0.001 | 0.031 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.003 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.022 ± 0.008 | - | 0.019 ± 0.001 | 0.025 ± 0.011 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.046 ± 0.001 | - | 0.046 ± 0.001 | 0.050 ± 0.000 | 0.079 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.002 | 0.032 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.022 ± 0.004 | - | 0.020 ± 0.003 | 0.031 ± 0.003 | 0.067 ± 0.072 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.061 ± 0.001 | - | 0.062 ± 0.001 | 0.067 ± 0.008 | 0.125 ± 0.059 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 57.161 ± 12.302 | - | 63.365 ± 5.496 | 36.361 ± 0.893 | 49.163 ± 6.249 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.011 ± 0.003 | - | 0.012 ± 0.000 | 0.019 ± 0.001 | 0.105 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.711 ± 0.129 | - | 1.455 ± 0.524 | 1.419 ± 0.045 | 3.390 ± 2.244 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.035 ± 0.001 | - | 0.036 ± 0.000 | 0.102 ± 0.054 | 0.171 ± 0.111 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.118 ± 0.002 | - | 0.119 ± 0.001 | 0.135 ± 0.019 | 0.209 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.137 ± 0.017 | - | 1.164 ± 0.067 | 1.270 ± 0.078 | 3.197 ± 0.062 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 4.519 ± 0.189 | - | 4.230 ± 0.330 | 4.664 ± 0.898 | 2.610 ± 0.836 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 15.079 ± 1.364 | - | 14.717 ± 0.416 | 14.080 ± 0.638 | 8.680 ± 0.359 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 5.729 ± 0.143 | - | 6.131 ± 0.552 | 5.806 ± 2.066 | 5.622 ± 0.134 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.173 ± 0.001 | - | 0.182 ± 0.013 | 0.216 ± 0.072 | 0.351 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.007 ± 0.003 | - | 0.009 ± 0.003 | 0.014 ± 0.003 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.167 ± 0.009 | - | 0.168 ± 0.007 | 0.186 ± 0.018 | 0.291 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.018 ± 0.000 | - | 0.019 ± 0.000 | 0.023 ± 0.003 | 0.034 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.007 ± 0.001 | - | 0.007 ± 0.002 | 0.010 ± 0.001 | 0.022 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.018 ± 0.004 | - | 0.017 ± 0.000 | 0.023 ± 0.004 | 0.036 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.045 ± 0.001 | - | 0.047 ± 0.001 | 0.060 ± 0.023 | 0.062 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.008 ± 0.002 | - | 0.009 ± 0.000 | 0.023 ± 0.012 | 0.030 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.023 ± 0.004 | - | 0.025 ± 0.001 | 0.022 ± 0.007 | 0.041 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.062 ± 0.000 | - | 0.064 ± 0.001 | 0.070 ± 0.015 | 0.083 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 45.519 ± 0.650 | - | 45.450 ± 0.926 | 37.548 ± 9.395 | 23.107 ± 1.585 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.012 ± 0.003 | - | 0.013 ± 0.000 | 0.015 ± 0.004 | 0.037 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 1.369 ± 0.046 | - | 1.384 ± 0.048 | 1.351 ± 0.058 | 1.094 ± 0.095 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.040 ± 0.008 | - | 0.042 ± 0.009 | 0.042 ± 0.006 | 0.061 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.117 ± 0.003 | - | 0.118 ± 0.001 | 0.123 ± 0.001 | 0.159 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.767 ± 0.068 | - | 0.783 ± 0.061 | 0.567 ± 0.019 | 3.439 ± 2.491 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 3.449 ± 0.980 | - | 4.311 ± 1.449 | 2.771 ± 0.064 | 6.586 ± 1.740 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 13.819 ± 2.197 | - | 14.617 ± 2.892 | 10.826 ± 0.139 | 16.818 ± 3.152 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.886 ± 0.267 | - | 1.984 ± 1.181 | 1.563 ± 0.103 | 1.780 ± 0.801 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.001 | 0.018 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.136 ± 0.031 | - | 0.128 ± 0.024 | 0.128 ± 0.008 | 0.177 ± 0.068 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.001 | 0.009 ± 0.000 | 0.019 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.100 ± 0.034 | - | 0.106 ± 0.022 | 0.143 ± 0.042 | 0.158 ± 0.093 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.014 ± 0.001 | - | 0.016 ± 0.000 | 0.017 ± 0.001 | 0.068 ± 0.101 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.012 ± 0.003 | - | 0.013 ± 0.003 | 0.014 ± 0.001 | 0.074 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.036 ± 0.006 | - | 0.035 ± 0.007 | 0.034 ± 0.001 | 0.045 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.006 ± 0.001 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.022 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.017 ± 0.003 | - | 0.018 ± 0.004 | 0.016 ± 0.000 | 0.039 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.047 ± 0.008 | - | 0.046 ± 0.010 | 0.044 ± 0.006 | 0.067 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 14.752 ± 1.252 | - | 15.396 ± 3.338 | 12.172 ± 0.838 | 15.969 ± 6.294 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.042 ± 0.141 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 1.191 ± 0.169 | - | 1.166 ± 0.143 | 0.958 ± 0.038 | 1.433 ± 0.230 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.031 ± 0.004 | - | 0.034 ± 0.003 | 0.027 ± 0.000 | 0.054 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.092 ± 0.010 | - | 0.093 ± 0.010 | 0.083 ± 0.002 | 0.256 ± 0.249 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.560 ± 0.015 | - | 0.570 ± 0.015 | 0.622 ± 0.061 | 2.674 ± 0.745 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 2.680 ± 0.042 | - | 2.657 ± 0.036 | 2.971 ± 0.267 | 1.821 ± 0.084 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 10.152 ± 0.233 | - | 9.941 ± 0.638 | 11.800 ± 1.218 | 7.101 ± 1.687 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 1.402 ± 0.016 | - | 1.451 ± 0.113 | 1.473 ± 0.087 | 1.740 ± 0.127 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.001 | 0.009 ± 0.001 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.130 ± 0.005 | - | 0.116 ± 0.004 | 0.127 ± 0.018 | 0.168 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.006 ± 0.001 | - | 0.007 ± 0.001 | 0.009 ± 0.000 | 0.020 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.092 ± 0.002 | - | 0.093 ± 0.002 | 0.096 ± 0.001 | 0.170 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.012 ± 0.000 | - | 0.014 ± 0.001 | 0.019 ± 0.003 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.005 ± 0.001 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.012 ± 0.003 | - | 0.012 ± 0.001 | 0.013 ± 0.000 | 0.031 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.030 ± 0.001 | - | 0.031 ± 0.001 | 0.034 ± 0.002 | 0.044 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.005 ± 0.001 | - | 0.006 ± 0.001 | 0.008 ± 0.000 | 0.026 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.014 ± 0.003 | - | 0.015 ± 0.001 | 0.017 ± 0.000 | 0.031 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.040 ± 0.001 | - | 0.042 ± 0.002 | 0.044 ± 0.001 | 0.067 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 12.324 ± 1.321 | - | 11.841 ± 0.442 | 12.238 ± 0.393 | 6.944 ± 0.263 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.009 ± 0.001 | - | 0.008 ± 0.000 | 0.013 ± 0.001 | 0.033 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 0.889 ± 0.007 | - | 0.900 ± 0.012 | 1.001 ± 0.055 | 0.818 ± 0.131 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.024 ± 0.000 | - | 0.026 ± 0.000 | 0.027 ± 0.006 | 0.080 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.078 ± 0.001 | - | 0.080 ± 0.002 | 0.092 ± 0.011 | 0.176 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.245 ± 0.059 | - | unsupported | 0.219 ± 0.000 | 1.571 ± 0.137 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.738 ± 0.159 | - | unsupported | 1.382 ± 0.069 | 3.417 ± 0.388 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 7.110 ± 3.022 | - | unsupported | 3.882 ± 0.093 | 8.102 ± 2.985 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 2.120 ± 0.658 | - | unsupported | 0.765 ± 0.011 | 1.620 ± 1.217 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.059 ± 0.047 | - | unsupported | 0.042 ± 0.001 | 0.047 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.030 ± 0.003 | - | unsupported | 0.035 ± 0.003 | 0.045 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.009 ± 0.007 | - | unsupported | 0.009 ± 0.000 | 0.018 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.018 | - | unsupported | 0.007 ± 0.001 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.005 ± 0.001 | - | unsupported | 0.012 ± 0.000 | 0.017 ± 0.062 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.022 ± 0.012 | - | unsupported | 0.017 ± 0.001 | 0.026 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.008 ± 0.001 | - | unsupported | 0.010 ± 0.000 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.041 ± 0.006 | - | unsupported | 0.020 ± 0.002 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 8.891 ± 2.974 | - | unsupported | 6.180 ± 0.070 | 8.955 ± 1.656 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.022 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.297 ± 0.089 | - | unsupported | 0.293 ± 0.001 | 0.378 ± 0.303 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.014 ± 0.002 | - | unsupported | 0.016 ± 0.000 | 0.030 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.099 ± 0.059 | - | unsupported | 0.037 ± 0.025 | 0.078 ± 0.139 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.211 ± 0.001 | - | unsupported | 0.658 ± 0.301 | 1.483 ± 0.073 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 1.372 ± 0.034 | - | unsupported | 1.421 ± 0.215 | 1.299 ± 0.109 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 3.829 ± 0.029 | - | unsupported | 5.580 ± 1.534 | 2.844 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 0.766 ± 0.020 | - | unsupported | 0.865 ± 0.043 | 0.859 ± 0.084 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.038 ± 0.002 | - | unsupported | 0.050 ± 0.007 | 0.048 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.015 ± 0.017 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.031 ± 0.001 | - | unsupported | 0.038 ± 0.006 | 0.053 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | unsupported | 0.042 ± 0.113 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.012 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.017 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.007 ± 0.000 | - | unsupported | 0.072 ± 0.165 | 0.020 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.016 ± 0.001 | - | unsupported | 0.020 ± 0.000 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 6.137 ± 0.067 | - | unsupported | 8.629 ± 3.424 | 4.466 ± 1.029 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.002 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.290 ± 0.008 | - | unsupported | 0.363 ± 0.034 | 0.308 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.012 ± 0.000 | - | unsupported | 0.028 ± 0.056 | 0.033 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.030 ± 0.000 | - | unsupported | 0.045 ± 0.017 | 0.054 ± 0.042 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.918 ± 0.114 | - | unsupported | 0.843 ± 0.076 | 0.927 ± 0.006 | 3.309 ± 0.841 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.056 ± 0.006 | - | unsupported | 0.065 ± 0.002 | 0.142 ± 0.007 | 0.071 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.304 ± 0.003 | - | unsupported | 0.119 ± 0.016 | 0.206 ± 0.016 | 0.136 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.304 ± 0.093 | - | unsupported | 0.268 ± 0.001 | 0.372 ± 0.026 | 0.651 ± 0.011 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.948 ± 0.101 | - | unsupported | 0.827 ± 0.503 | 1.014 ± 0.107 | 3.677 ± 2.314 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.054 ± 0.001 | - | unsupported | 0.129 ± 0.151 | 0.252 ± 0.082 | 0.121 ± 0.008 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.108 ± 0.000 | - | unsupported | 0.135 ± 0.120 | 0.239 ± 0.005 | 0.257 ± 0.033 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.250 ± 0.001 | - | unsupported | 0.267 ± 0.201 | 0.596 ± 0.398 | 0.843 ± 0.013 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.247 ± 0.299 | - | unsupported | 0.190 ± 0.008 | 0.520 ± 0.345 | 0.529 ± 0.016 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.126 ± 0.005 | - | unsupported | 0.043 ± 0.001 | 0.085 ± 0.006 | 0.068 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.165 ± 0.003 | - | unsupported | 0.050 ± 0.003 | 0.102 ± 0.040 | 0.094 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.109 ± 0.020 | - | unsupported | 0.082 ± 0.001 | 0.189 ± 0.023 | 0.227 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.215 ± 0.000 | - | unsupported | 0.199 ± 0.015 | 1.266 ± 0.810 | 0.524 ± 0.052 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.058 ± 0.000 | - | unsupported | 0.044 ± 0.007 | 0.184 ± 0.094 | 0.068 ± 0.013 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.070 ± 0.001 | - | unsupported | 0.052 ± 0.006 | 0.299 ± 0.156 | 0.086 ± 0.010 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.096 ± 0.000 | - | unsupported | 0.092 ± 0.017 | 0.323 ± 0.353 | 0.257 ± 0.027 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.108 ± 0.001 | - | 0.123 ± 0.133 | 0.084 ± 0.018 | 0.273 ± 0.063 | 0.262 ± 0.016 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.043 ± 0.007 | - | 0.050 ± 0.037 | 0.026 ± 0.015 | 0.031 ± 0.003 | 0.047 ± 0.007 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.176 ± 0.002 | - | 0.047 ± 0.002 | 0.026 ± 0.000 | 0.034 ± 0.001 | 0.052 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.061 ± 0.001 | - | 0.063 ± 0.001 | 0.037 ± 0.017 | 0.067 ± 0.008 | 0.105 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.103 ± 0.001 | - | 0.105 ± 0.001 | 0.094 ± 0.024 | 0.445 ± 0.144 | 0.350 ± 0.026 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.044 ± 0.000 | - | 0.049 ± 0.007 | 0.030 ± 0.017 | 0.038 ± 0.005 | 0.044 ± 0.004 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.047 ± 0.004 | - | 0.054 ± 0.012 | 0.046 ± 0.017 | 0.038 ± 0.005 | 0.053 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.055 ± 0.002 | - | 0.058 ± 0.002 | 0.054 ± 0.027 | 0.107 ± 0.015 | 0.147 ± 0.022 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 14.471 ± 5.640 | - | 25.173 ± 5.361 | 22.872 ± 0.937 | 16.530 ± 2.484 | 51.445 ± 12.625 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 7.830 ± 0.304 | - | 7.798 ± 0.225 | 17.778 ± 4.259 | 17.217 ± 2.159 | 378.315 ± 31.049 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 12.491 ± 6.917 | - | 11.932 ± 4.826 | 7.489 ± 0.341 | 9.771 ± 1.902 | 20.569 ± 6.489 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 7.624 ± 0.138 | - | 7.525 ± 0.183 | 9.407 ± 5.820 | 12.812 ± 4.514 | 232.637 ± 4.746 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 11.901 ± 3.917 | - | 9.334 ± 4.350 | 6.733 ± 0.400 | 7.530 ± 0.128 | 7.196 ± 0.455 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 6.269 ± 0.061 | - | 6.265 ± 0.022 | 8.208 ± 5.309 | 8.993 ± 0.747 | 315.796 ± 57.880 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 11.088 ± 4.769 | - | 9.303 ± 1.177 | 7.712 ± 0.294 | 7.368 ± 0.188 | 7.703 ± 0.543 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 7.068 ± 0.065 | - | 7.105 ± 0.088 | 8.592 ± 4.843 | 8.159 ± 3.353 | 316.022 ± 298.930 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 30.423 ± 4.413 | - | 31.203 ± 5.367 | 21.514 ± 2.028 | 28.883 ± 5.642 | 13.856 ± 0.299 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 18.809 ± 1.411 | - | 19.871 ± 0.941 | 26.162 ± 5.397 | 37.023 ± 4.471 | 655.382 ± 35.511 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.329 ± 0.144 | 0.566 ± 0.137 | unsupported | 0.289 ± 0.043 | 0.378 ± 0.100 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 6.139 ± 1.142 | 6.233 ± 3.321 | unsupported | 3.663 ± 0.724 | 4.543 ± 0.091 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.286 ± 0.002 | 0.298 ± 0.002 | unsupported | 0.326 ± 0.026 | 0.537 ± 0.058 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 3.676 ± 0.238 | 3.796 ± 0.459 | unsupported | 4.564 ± 0.476 | 5.834 ± 0.436 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 11.185 ± 0.889 | - | 11.737 ± 0.775 | 10.240 ± 0.688 | 10.145 ± 0.400 | 35.671 ± 2.160 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 8.234 ± 0.136 | - | 8.815 ± 0.035 | 18.150 ± 10.510 | 11.474 ± 1.360 | 399.758 ± 55.703 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 9.618 ± 0.299 | 9.842 ± 3.557 | 5.156 ± 0.296 | 25.195 ± 0.891 | 8.434 ± 0.365 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 8.521 ± 0.132 | 8.995 ± 0.665 | 5.496 ± 0.462 | 34.108 ± 1.561 | 153.624 ± 25.361 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 12.222 ± 8.313 | - | 12.134 ± 2.967 | 11.539 ± 0.811 | 14.507 ± 1.698 | 30.804 ± 9.292 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 7.353 ± 0.352 | - | 7.348 ± 0.340 | 14.397 ± 8.051 | 15.278 ± 3.893 | 231.355 ± 49.281 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.940 ± 0.830 | - | 1.036 ± 0.200 | 2.664 ± 0.297 | 6.216 ± 3.721 | 5.774 ± 0.192 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.668 ± 0.030 | - | 0.697 ± 0.013 | 2.246 ± 0.288 | 5.493 ± 0.843 | 9.439 ± 1.270 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 15.144 ± 2.002 | - | 14.108 ± 5.915 | 11.195 ± 0.488 | 13.907 ± 1.511 | 24.712 ± 1.618 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 10.703 ± 0.483 | - | 10.267 ± 0.212 | 13.017 ± 2.670 | 16.114 ± 3.712 | 1031.977 ± 77.285 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 14.405 ± 3.928 | - | 12.073 ± 2.729 | 10.527 ± 0.290 | 14.698 ± 1.287 | 23.164 ± 4.049 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 10.461 ± 0.673 | - | 10.088 ± 0.372 | 12.159 ± 2.171 | 17.411 ± 5.038 | 1344.165 ± 606.400 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 2.063 ± 0.298 | 2.229 ± 0.250 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 19.690 ± 7.895 | 23.565 ± 4.776 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.394 ± 0.006 | 1.396 ± 0.012 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 15.573 ± 2.068 | 14.566 ± 1.728 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.927 ± 0.389 | 1.894 ± 0.387 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 17.337 ± 2.763 | 14.304 ± 0.730 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.406 ± 0.010 | 1.400 ± 0.022 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 12.661 ± 1.803 | 11.953 ± 0.074 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 14.008 ± 5.322 | - | 9.979 ± 1.322 | 7.877 ± 0.809 | 9.592 ± 2.078 | 19.440 ± 0.752 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 7.646 ± 0.196 | - | 7.451 ± 0.135 | 8.998 ± 5.444 | 10.717 ± 1.350 | 267.421 ± 186.511 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 43.540 ± 12.473 | 45.969 ± 14.487 | 39.247 ± 1.799 | 38.887 ± 3.957 | 69.452 ± 2.953 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 35.825 ± 1.420 | 34.562 ± 3.451 | 41.621 ± 7.347 | 53.375 ± 8.120 | 1778.656 ± 77.814 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 23.289 ± 4.602 | - | 28.354 ± 13.782 | 13.280 ± 0.257 | 77.434 ± 14.906 | 31.944 ± 3.570 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 15.433 ± 2.005 | - | 14.343 ± 1.059 | 16.101 ± 7.422 | 64.350 ± 10.139 | 16.066 ± 1.614 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.287 ± 0.085 | 0.296 ± 0.113 | unsupported | 0.052 ± 0.007 | 0.082 ± 0.067 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 3.379 ± 0.584 | 2.897 ± 0.146 | unsupported | 0.802 ± 0.023 | 1.006 ± 0.420 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.170 ± 0.016 | 0.174 ± 0.006 | unsupported | 0.080 ± 0.034 | 0.027 ± 0.006 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 1.822 ± 0.369 | 1.937 ± 0.133 | unsupported | 0.952 ± 0.308 | 0.304 ± 0.089 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.092 ± 0.019 | 0.091 ± 0.003 | unsupported | 0.087 ± 0.003 | 0.291 ± 0.058 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.318 ± 0.071 | 1.339 ± 0.059 | unsupported | 1.572 ± 0.186 | 3.120 ± 0.142 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.053 ± 0.009 | 0.055 ± 0.005 | unsupported | 0.146 ± 0.042 | 0.134 ± 0.018 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.718 ± 0.027 | 0.717 ± 0.032 | unsupported | 1.206 ± 0.193 | 1.371 ± 0.019 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.292 ± 0.043 | 0.303 ± 0.012 | unsupported | 0.301 ± 0.013 | 0.116 ± 0.012 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 4.840 ± 0.758 | 4.738 ± 0.503 | unsupported | 6.990 ± 2.585 | 1.238 ± 0.251 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.162 ± 0.049 | 0.213 ± 0.030 | unsupported | 0.507 ± 0.180 | 0.073 ± 0.013 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 2.412 ± 0.163 | 2.179 ± 0.328 | unsupported | 3.149 ± 0.898 | 0.616 ± 0.172 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.049 ± 0.010 | 0.071 ± 0.008 | unsupported | 0.045 ± 0.000 | 0.130 ± 0.055 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.656 ± 0.046 | 0.652 ± 0.301 | unsupported | 0.609 ± 0.089 | 0.904 ± 0.137 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.051 ± 0.002 | 0.071 ± 0.007 | unsupported | 0.116 ± 0.018 | 0.108 ± 0.026 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 1.176 ± 0.133 | 2.054 ± 0.893 | unsupported | 0.463 ± 0.119 | 0.536 ± 0.167 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.102 ± 0.012 | 0.127 ± 0.006 | unsupported | 0.090 ± 0.003 | 0.156 ± 0.032 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.256 ± 0.165 | 1.411 ± 0.134 | unsupported | 2.209 ± 1.013 | 1.574 ± 0.071 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.091 ± 0.004 | 0.109 ± 0.004 | unsupported | 0.162 ± 0.032 | 0.186 ± 0.018 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 1.198 ± 0.207 | 1.312 ± 0.023 | unsupported | 1.214 ± 0.427 | 0.733 ± 0.043 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.141 ± 0.012 | 0.169 ± 0.036 | unsupported | unsupported | 0.168 ± 0.059 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 2.189 ± 0.338 | 2.150 ± 0.068 | unsupported | unsupported | 2.206 ± 0.291 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.129 ± 0.002 | 0.149 ± 0.005 | unsupported | unsupported | 0.136 ± 0.110 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 2.087 ± 2.952 | 2.074 ± 0.223 | unsupported | unsupported | 1.187 ± 0.298 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.014 ± 0.002 | 0.022 ± 0.011 | unsupported | 0.008 ± 0.001 | 0.090 ± 0.044 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.104 ± 0.015 | 0.128 ± 0.004 | unsupported | 0.080 ± 0.019 | 0.286 ± 0.016 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.017 ± 0.019 | 0.018 ± 0.000 | unsupported | 0.028 ± 0.004 | 0.093 ± 0.008 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.111 ± 0.015 | 0.108 ± 0.018 | unsupported | 0.154 ± 0.016 | 0.158 ± 0.029 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.026 ± 0.001 | 0.043 ± 0.008 | unsupported | 0.026 ± 0.001 | 0.100 ± 0.024 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.367 ± 0.048 | 0.410 ± 0.386 | unsupported | 0.548 ± 0.186 | 0.742 ± 0.090 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.027 ± 0.002 | 0.041 ± 0.002 | unsupported | 0.151 ± 0.039 | 0.097 ± 0.013 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.737 ± 0.119 | 0.941 ± 0.553 | unsupported | 0.514 ± 0.048 | 0.385 ± 0.035 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.061 ± 0.004 | 0.063 ± 0.012 | unsupported | 0.077 ± 0.001 | 0.152 ± 0.040 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 1.251 ± 0.616 | 1.090 ± 0.333 | unsupported | 1.178 ± 0.019 | 1.691 ± 0.796 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.030 ± 0.000 | 0.031 ± 0.008 | unsupported | 0.107 ± 0.039 | 0.068 ± 0.002 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.466 ± 0.028 | 0.458 ± 0.154 | unsupported | 1.156 ± 0.390 | 0.670 ± 0.413 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.079 ± 0.017 | 0.098 ± 0.051 | unsupported | 0.080 ± 0.002 | 0.164 ± 0.087 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.906 ± 0.889 | 1.003 ± 0.437 | unsupported | 1.197 ± 0.019 | 1.232 ± 0.429 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.031 ± 0.000 | 0.033 ± 0.003 | unsupported | 0.092 ± 0.026 | 0.068 ± 0.015 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.495 ± 0.207 | 0.623 ± 0.161 | unsupported | 0.967 ± 0.600 | 0.956 ± 0.442 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.121 ± 0.041 | 0.128 ± 0.033 | unsupported | 0.099 ± 0.003 | 0.188 ± 0.061 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.396 ± 0.216 | 1.452 ± 0.440 | unsupported | 2.611 ± 0.849 | 1.686 ± 0.354 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.093 ± 0.001 | 0.108 ± 0.001 | unsupported | 0.106 ± 0.033 | 0.191 ± 0.081 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 1.069 ± 0.191 | 1.132 ± 0.156 | unsupported | 1.319 ± 0.175 | 0.718 ± 0.022 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.272 ± 0.024 | 0.262 ± 0.041 | unsupported | 0.280 ± 0.004 | 0.412 ± 0.056 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 4.096 ± 0.453 | 4.041 ± 0.306 | unsupported | 4.701 ± 0.384 | 5.826 ± 1.186 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.156 ± 0.029 | 0.146 ± 0.030 | unsupported | 0.336 ± 0.225 | 0.171 ± 0.008 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 2.085 ± 0.241 | 1.947 ± 0.121 | unsupported | 3.392 ± 0.408 | 2.402 ± 0.081 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.666 ± 0.147 | 1.013 ± 0.448 | unsupported | 0.115 ± 0.009 | 0.205 ± 0.109 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 10.410 ± 1.117 | 12.062 ± 0.683 | unsupported | 2.736 ± 0.577 | 1.575 ± 0.179 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.879 ± 0.285 | 0.889 ± 0.244 | unsupported | 0.178 ± 0.124 | 0.090 ± 0.010 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 11.901 ± 1.843 | 13.432 ± 1.399 | unsupported | 1.643 ± 0.259 | 0.571 ± 0.074 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 31.924 ± 4.854 | - | unsupported | 18.108 ± 3.064 | - | 18.723 ± 1.278 | 28.455 ± 10.248 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 16.557 ± 1.461 | - | unsupported | 22.073 ± 2.591 | - | 18.169 ± 5.610 | 19.568 ± 3.043 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 18.024 ± 3.900 | - | unsupported | 10.335 ± 0.237 | - | 9.834 ± 0.437 | 12.797 ± 1.015 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 11.532 ± 1.852 | - | unsupported | 15.284 ± 1.758 | - | 14.144 ± 1.551 | 12.459 ± 2.492 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 13.845 ± 4.794 | - | unsupported | 10.651 ± 1.750 | - | 11.861 ± 0.929 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 10.754 ± 3.179 | - | unsupported | 14.630 ± 2.695 | - | 12.649 ± 1.110 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 34.563 ± 5.789 | - | unsupported | 19.104 ± 1.402 | - | 23.666 ± 3.414 | 18.160 ± 0.422 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 18.417 ± 4.652 | - | unsupported | 23.030 ± 1.346 | - | 20.742 ± 2.142 | 22.781 ± 1.022 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 20.695 ± 6.409 | - | unsupported | 10.274 ± 0.615 | - | 56.614 ± 3.086 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 10.781 ± 0.218 | - | unsupported | 22.759 ± 7.699 | - | 35.892 ± 13.799 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 23.669 ± 10.359 | - | unsupported | 11.330 ± 1.407 | - | 50.726 ± 10.778 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 10.378 ± 0.985 | - | unsupported | 14.361 ± 1.153 | - | 33.915 ± 5.265 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 30.339 ± 7.090 | - | unsupported | 20.303 ± 3.053 | - | 25.092 ± 2.644 | 30.060 ± 4.034 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 18.241 ± 2.035 | - | unsupported | 24.907 ± 5.090 | - | 19.408 ± 2.719 | 20.957 ± 2.110 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 16.185 ± 2.037 | - | unsupported | 12.180 ± 0.772 | - | 10.571 ± 1.296 | 11.699 ± 1.236 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 10.564 ± 0.475 | - | unsupported | 15.284 ± 3.083 | - | 12.789 ± 0.649 | 12.917 ± 0.714 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.392 ± 0.282 | unsupported | unsupported | 0.275 ± 0.077 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 4.654 ± 1.670 | unsupported | unsupported | 3.223 ± 0.373 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.204 ± 0.001 | unsupported | unsupported | 0.315 ± 0.032 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 2.580 ± 0.074 | unsupported | unsupported | 3.822 ± 1.077 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 33.135 ± 9.774 | - | unsupported | 17.431 ± 1.536 | - | 25.811 ± 5.525 | 27.060 ± 4.786 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 17.510 ± 0.907 | - | unsupported | 22.393 ± 3.445 | - | 17.073 ± 1.519 | 21.467 ± 0.740 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 8.493 ± 0.779 | - | 0.001 ± 0.001 | 4.799 ± 0.767 | 6.033 ± 0.244 | 15.944 ± 0.765 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 7.885 ± 0.692 | - | 0.003 ± 0.001 | 4.516 ± 0.222 | 6.375 ± 0.790 | 14.284 ± 0.734 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 9.605 ± 1.622 | - | 25.114 ± 4.442 | 7.897 ± 0.170 | 10.766 ± 3.077 | 8.601 ± 0.575 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 7.388 ± 0.205 | - | 17.124 ± 4.009 | 7.653 ± 0.326 | 10.540 ± 3.174 | 9.258 ± 0.730 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 9.816 ± 0.846 | - | 9.344 ± 1.148 | 21.882 ± 11.712 | 19.619 ± 0.881 | 12.493 ± 0.858 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 8.946 ± 0.345 | - | 12.704 ± 3.624 | 10.580 ± 0.814 | 13.228 ± 1.054 | 12.276 ± 0.577 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 5.234 ± 0.762 | - | 4.865 ± 0.243 | 9.706 ± 0.540 | 11.123 ± 1.528 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 4.791 ± 0.177 | - | 4.771 ± 0.691 | 8.091 ± 0.578 | 10.766 ± 2.546 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 13.480 ± 4.067 | - | 0.002 ± 0.000 | 11.361 ± 1.147 | 12.332 ± 0.411 | 15.791 ± 3.516 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 9.513 ± 0.462 | - | 0.001 ± 0.002 | 11.728 ± 1.095 | 10.826 ± 0.330 | 12.148 ± 3.950 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 43.734 ± 2.294 | - | 0.002 ± 0.006 | 42.682 ± 2.902 | 59.621 ± 11.381 | 233.327 ± 115.911 | 40.677 ± 1.626 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 18.524 ± 1.729 | - | 0.001 ± 0.001 | 44.722 ± 4.242 | 35.207 ± 2.881 | 200.029 ± 9.486 | 20.589 ± 4.210 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 6.968 ± 0.387 | - | 7.086 ± 0.975 | 8.373 ± 0.809 | 7.517 ± 0.981 | 10.679 ± 1.118 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 5.261 ± 0.239 | - | 5.182 ± 0.244 | 7.482 ± 0.714 | 5.996 ± 0.860 | 11.056 ± 2.844 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 6.920 ± 0.035 | - | 7.033 ± 0.289 | 8.167 ± 0.871 | 8.292 ± 0.922 | 10.229 ± 0.424 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 4.959 ± 0.052 | - | 5.017 ± 0.077 | 6.585 ± 0.448 | 6.368 ± 1.043 | 9.935 ± 2.222 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.001 ± 0.001 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.001 ± 0.001 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.002 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.001 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 4.781750 | 298859.38 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 4.675000 | 292187.50 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 6.133333 | 383333.31 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 3.064625 | 191539.06 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 4.036750 | 252296.87 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 4.361917 | 272619.81 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 14.091000 | 880687.50 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 15.167125 | 947945.31 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 10.845625 | 677851.56 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 10.955416 | 684713.50 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 6.050416 | 378151.00 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 4.625375 | 289085.94 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 5.271750 | 329484.38 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 9.058167 | 566135.44 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 72.684708 | 4542794.25 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 58.606791 | 3662924.44 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 98.223791 | 6138986.94 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 99.722750 | 6232671.88 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 8.595500 | 537218.75 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 5.208417 | 325526.06 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 4.573000 | 285812.50 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 4.774166 | 298385.38 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 93.348000 | 5834250.00 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 73.026625 | 4564164.06 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 58.820750 | 3676296.88 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 60.728625 | 3795539.06 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 33.001500 | 2062593.75 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 35.663042 | 2228940.12 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 315.046000 | 19690375.00 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 377.040875 | 23565054.69 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 22.298625 | 1393664.06 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 22.340792 | 1396299.50 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 249.160083 | 15572505.19 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 233.052833 | 14565802.06 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 30.839875 | 1927492.19 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 30.306709 | 1894169.31 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 277.388625 | 17336789.06 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 228.871333 | 14304458.31 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 22.500458 | 1406278.62 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 22.393375 | 1399585.94 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 202.582833 | 12661427.06 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 191.250000 | 11953125.00 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.220125 | 81564.45 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.353333 | 52395.83 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 18.396459 | 287444.67 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 18.949500 | 296085.94 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 8.051083 | 1006385.38 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 6.419042 | 802380.25 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 27.032083 | 3379010.38 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 23.178750 | 2897343.75 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 1.746708 | 27292.31 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.117833 | 79966.14 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 10.877583 | 169962.23 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 11.134458 | 173975.91 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.434000 | 304250.00 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.619417 | 952427.12 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 14.575083 | 1821885.38 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 15.498417 | 1937302.12 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 18.647375 | 291365.23 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.554000 | 86781.25 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.889708 | 92026.69 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.799250 | 90613.28 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 24.958334 | 3119791.75 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 12.577709 | 1572213.62 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.546125 | 1318265.62 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.715125 | 1339390.62 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.563625 | 133806.64 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 9.343875 | 145998.05 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.389166 | 52955.72 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3.496750 | 54636.72 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.965916 | 1370739.50 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 9.648166 | 1206020.75 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.742083 | 717760.38 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.734916 | 716864.50 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.452625 | 116447.27 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 19.248833 | 300763.02 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 18.663917 | 291623.70 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 19.417333 | 303395.83 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 9.904709 | 1238088.62 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 55.923292 | 6990411.50 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 38.720292 | 4840036.50 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 37.900417 | 4737552.12 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.686625 | 73228.52 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 32.472834 | 507388.03 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 10.375083 | 162110.67 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 13.620458 | 212819.66 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.931750 | 616468.75 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 25.188708 | 3148588.50 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 19.297708 | 2412213.50 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 17.432125 | 2179015.62 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.323667 | 130057.30 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 2.871875 | 44873.05 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.150458 | 49225.91 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.559084 | 71235.69 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.235125 | 904390.62 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.871292 | 608911.50 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.250750 | 656343.75 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.215333 | 651916.62 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.901625 | 107837.89 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 7.455750 | 116496.09 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.242042 | 50656.91 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.571333 | 71427.08 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.284791 | 535598.88 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.706167 | 463270.88 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.406417 | 1175802.12 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 16.433000 | 2054125.00 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.994292 | 156160.81 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.739208 | 89675.12 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.534959 | 102108.73 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.146292 | 127285.81 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.592917 | 1574114.62 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 17.674667 | 2209333.38 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.047459 | 1255932.38 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.285708 | 1410713.50 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.920875 | 186263.67 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 10.346167 | 161658.86 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.799958 | 90624.34 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6.951542 | 108617.84 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.867625 | 733453.12 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 9.709708 | 1213713.50 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.582458 | 1197807.25 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 10.496584 | 1312073.00 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.724500 | 167570.31 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 9.026042 | 141031.91 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 10.796083 | 168688.80 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 17.646667 | 2205833.38 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 17.509125 | 2188640.62 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 17.203542 | 2150442.75 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.683208 | 135675.12 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.251917 | 128936.20 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 9.524125 | 148814.45 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.499542 | 1187442.75 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 16.698041 | 2087255.12 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 16.592583 | 2074072.88 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.766583 | 90102.86 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.512791 | 8012.36 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.897959 | 14030.61 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.380083 | 21563.80 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 2.291542 | 286442.75 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.642500 | 80312.50 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.831042 | 103880.25 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 1.026541 | 128317.62 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.951625 | 92994.14 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 1.774833 | 27731.77 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.107041 | 17297.52 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.122458 | 17538.41 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.260708 | 157588.50 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.232000 | 154000.00 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.887250 | 110906.25 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 0.861959 | 107744.88 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.369958 | 99530.59 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 1.663541 | 25992.83 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.659584 | 25931.00 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.767500 | 43242.19 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.937000 | 742125.00 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.384375 | 548046.88 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.938459 | 367307.38 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.279875 | 409984.38 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.186625 | 96666.02 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 9.679500 | 151242.19 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.745292 | 27270.19 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.618417 | 40912.77 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.081250 | 385156.25 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.108750 | 513593.75 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.898000 | 737250.00 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.528667 | 941083.38 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.720084 | 151876.31 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.910375 | 76724.61 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.919875 | 61248.05 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.022291 | 62848.30 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.530416 | 1691302.00 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.424250 | 1178031.25 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.010750 | 1251343.75 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 8.723167 | 1090395.88 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.347375 | 67927.73 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.845709 | 106964.20 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.918750 | 29980.47 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.986500 | 31039.06 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.361417 | 670177.12 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 9.250833 | 1156354.12 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.731333 | 466416.62 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.662208 | 457776.00 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.492959 | 163952.48 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.121583 | 80024.73 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.080041 | 79375.64 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.298708 | 98417.31 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 9.857458 | 1232182.25 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.572417 | 1196552.12 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.250833 | 906354.12 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 8.027292 | 1003411.50 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.369125 | 68267.58 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.870125 | 91720.70 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.013750 | 31464.84 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.117166 | 33080.72 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.651458 | 956432.25 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.733625 | 966703.12 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.962250 | 495281.25 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.981417 | 622677.12 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.033917 | 188029.95 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.355209 | 99300.14 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 7.720500 | 120632.81 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.165083 | 127579.42 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.487750 | 1685968.75 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 20.888916 | 2611114.50 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 11.171750 | 1396468.75 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.614125 | 1451765.62 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.232167 | 191127.61 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.779666 | 105932.28 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.951000 | 92984.38 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6.917416 | 108084.62 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.747292 | 718411.50 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 10.550708 | 1318838.50 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.548417 | 1068552.12 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 9.053375 | 1131671.88 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 26.374584 | 412102.88 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.943750 | 280371.09 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 17.395083 | 271798.17 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 16.764542 | 261945.97 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 46.611625 | 5826453.12 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 37.610208 | 4701276.00 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 32.771208 | 4096401.00 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 32.329083 | 4041135.37 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.955000 | 171171.88 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 21.485125 | 335705.08 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 10.013750 | 156464.84 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 9.352292 | 146129.56 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 19.215875 | 2401984.38 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 27.138292 | 3392286.50 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 16.681084 | 2085135.50 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 15.576125 | 1947015.62 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.129000 | 205140.62 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.364292 | 115067.06 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 42.650125 | 666408.20 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 64.800917 | 1012514.33 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.601042 | 1575130.25 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 21.888292 | 2736036.50 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 83.278958 | 10409869.75 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 96.498750 | 12062343.75 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.745000 | 89765.62 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 11.398667 | 178104.17 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 56.239792 | 878746.75 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 56.870209 | 888597.02 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.568875 | 571109.38 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 13.144833 | 1643104.12 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 95.206334 | 11900791.75 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 107.458291 | 13432286.38 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 4.406750 | 275421.88 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 6.276625 | 392289.06 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 51.563417 | 3222713.56 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 74.465708 | 4654106.75 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 5.046917 | 315432.31 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 3.269417 | 204338.56 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 61.149083 | 3821817.69 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 41.279042 | 2579940.12 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.013333 | 833.31 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.004250 | 265.62 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.016083 | 1005.19 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.001291 | 80.69 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.012500 | 781.25 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.018833 | 1177.06 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000084 | 5.25 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.015417 | 963.56 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.019541 | 1221.31 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000125 | 7.81 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.011292 | 705.75 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.003542 | 221.38 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000167 | 10.44 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.023125 | 1445.31 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.001958 | 122.38 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000167 | 10.44 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.010959 | 684.94 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.006042 | 377.62 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000084 | 5.25 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.015000 | 937.50 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002417 | 151.06 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `jax-cpu` is 47.3x faster than the slowest successful cell (`julia-base`, 74.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `pytorch-cpu` is 18.5x faster than the slowest successful cell (`julia-base`, 74.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-direct` is 16.9x faster than the slowest successful cell (`julia-base`, 74.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-trace` is 13.1x faster than the slowest successful cell (`julia-base`, 74.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `jax-cpu` is 21.8x faster than the slowest successful cell (`julia-base`, 464.507 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `pytorch-cpu` is 38.8x faster than the slowest successful cell (`julia-base`, 464.507 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-direct` is 34.5x faster than the slowest successful cell (`julia-base`, 464.507 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-trace` is 42.0x faster than the slowest successful cell (`julia-base`, 464.507 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 18.2x faster than the slowest successful cell (`julia-base`, 156.138 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 21.2x faster than the slowest successful cell (`julia-base`, 156.138 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-direct` is 18.5x faster than the slowest successful cell (`julia-base`, 156.138 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-trace` is 21.7x faster than the slowest successful cell (`julia-base`, 156.138 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 131.8x faster than the slowest successful cell (`julia-base`, 307.311 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 50.9x faster than the slowest successful cell (`julia-base`, 307.311 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-direct` is 44.4x faster than the slowest successful cell (`julia-base`, 307.311 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-trace` is 39.3x faster than the slowest successful cell (`julia-base`, 307.311 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `jax-cpu` is 87.8x faster than the slowest successful cell (`julia-base`, 629.773 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `pytorch-cpu` is 84.4x faster than the slowest successful cell (`julia-base`, 629.773 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-direct` is 92.3x faster than the slowest successful cell (`julia-base`, 629.773 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-trace` is 75.1x faster than the slowest successful cell (`julia-base`, 629.773 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 24.6x faster than the slowest successful cell (`pytorch-cpu`, 19.390 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 21.4x faster than the slowest successful cell (`pytorch-cpu`, 19.390 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 20.1x faster than the slowest successful cell (`pytorch-cpu`, 19.390 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 15.6x faster than the slowest successful cell (`pytorch-cpu`, 11.579 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 11.579 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 11.579 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 23.0x faster than the slowest successful cell (`pytorch-cpu`, 15.731 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 16.5x faster than the slowest successful cell (`pytorch-cpu`, 15.731 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 15.8x faster than the slowest successful cell (`pytorch-cpu`, 15.731 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 12.1x faster than the slowest successful cell (`pytorch-cpu`, 11.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 15.5x faster than the slowest successful cell (`pytorch-cpu`, 11.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 15.3x faster than the slowest successful cell (`pytorch-cpu`, 11.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 1357.0x faster than the slowest successful cell (`tenferro-direct`, 1.923 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 938.7x faster than the slowest successful cell (`jax-cpu`, 1.525 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `pytorch-cpu` is 16.5x faster than the slowest successful cell (`jax-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 32.0x faster than the slowest successful cell (`jax-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-trace` is 15.8x faster than the slowest successful cell (`jax-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 26.8x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `pytorch-cpu` is 17.5x faster than the slowest successful cell (`jax-cpu`, 0.108 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 46.3x faster than the slowest successful cell (`jax-cpu`, 0.108 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-trace` is 25.6x faster than the slowest successful cell (`jax-cpu`, 0.108 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 36.3x faster than the slowest successful cell (`pytorch-cpu`, 0.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 0.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 25.2x faster than the slowest successful cell (`pytorch-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 48.3x faster than the slowest successful cell (`pytorch-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 19.2x faster than the slowest successful cell (`pytorch-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 51.6x faster than the slowest successful cell (`pytorch-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 22.0x faster than the slowest successful cell (`pytorch-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 30.0x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 14.3x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 32.0x faster than the slowest successful cell (`pytorch-cpu`, 0.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`pytorch-cpu`, 0.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 29.8x faster than the slowest successful cell (`pytorch-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 34.1x faster than the slowest successful cell (`pytorch-cpu`, 0.053 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 0.053 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 33.6x faster than the slowest successful cell (`pytorch-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 15.7x faster than the slowest successful cell (`pytorch-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 30.7x faster than the slowest successful cell (`jax-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 16.1x faster than the slowest successful cell (`jax-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 31.8x faster than the slowest successful cell (`pytorch-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x32x32`): `tenferro-direct` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x32x32`): `tenferro-trace` is 10.0x faster than the slowest successful cell (`jax-cpu`, 0.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 17.1x faster than the slowest successful cell (`pytorch-cpu`, 0.038 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 36.2x faster than the slowest successful cell (`pytorch-cpu`, 0.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 22.4x faster than the slowest successful cell (`pytorch-cpu`, 0.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=4, shape=`4x8x8`): `tenferro-direct` is 13.0x faster than the slowest successful cell (`jax-cpu`, 0.329 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=4, shape=`4x8x8`): `tenferro-trace` is 12.4x faster than the slowest successful cell (`jax-cpu`, 0.329 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 14.2x faster than the slowest successful cell (`jax-cpu`, 0.030 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.108 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`1x32x32`): `tenferro-trace` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.108 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`1x64x24`): `tenferro-direct` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x2x2`): `tenferro-direct` is 17.2x faster than the slowest successful cell (`jax-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x4x4`): `tenferro-direct` is 14.0x faster than the slowest successful cell (`jax-cpu`, 0.035 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 21.3x faster than the slowest successful cell (`jax-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-trace` is 12.8x faster than the slowest successful cell (`jax-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `pytorch-cpu` is 11.0x faster than the slowest successful cell (`jax-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 30.2x faster than the slowest successful cell (`jax-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 23.4x faster than the slowest successful cell (`jax-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x2x2`): `tenferro-direct` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.037 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 21.2x faster than the slowest successful cell (`jax-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-trace` is 12.9x faster than the slowest successful cell (`jax-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `jax-cpu` is 12.5x faster than the slowest successful cell (`pytorch-cpu`, 0.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 44.7x faster than the slowest successful cell (`pytorch-cpu`, 0.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 19.1x faster than the slowest successful cell (`pytorch-cpu`, 0.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 24.7x faster than the slowest successful cell (`pytorch-cpu`, 0.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 19.8x faster than the slowest successful cell (`pytorch-cpu`, 0.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 24.5x faster than the slowest successful cell (`pytorch-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 17.4x faster than the slowest successful cell (`pytorch-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `jax-cpu` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 31.2x faster than the slowest successful cell (`pytorch-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 23.4x faster than the slowest successful cell (`pytorch-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 23.7x faster than the slowest successful cell (`pytorch-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 14.8x faster than the slowest successful cell (`pytorch-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 21.9x faster than the slowest successful cell (`pytorch-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 30.7x faster than the slowest successful cell (`pytorch-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 19.6x faster than the slowest successful cell (`pytorch-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 21.3x faster than the slowest successful cell (`pytorch-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 14.3x faster than the slowest successful cell (`pytorch-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 15.8x faster than the slowest successful cell (`pytorch-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.131 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 32.5x faster than the slowest successful cell (`pytorch-cpu`, 0.131 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 28.7x faster than the slowest successful cell (`pytorch-cpu`, 0.131 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 20.2x faster than the slowest successful cell (`pytorch-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 12.3x faster than the slowest successful cell (`pytorch-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 22.0x faster than the slowest successful cell (`pytorch-cpu`, 0.122 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 18.1x faster than the slowest successful cell (`pytorch-cpu`, 0.122 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`8x2x2`): `tenferro-direct` is 12.1x faster than the slowest successful cell (`jax-cpu`, 0.034 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 17.7x faster than the slowest successful cell (`pytorch-cpu`, 0.032 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 17.8x faster than the slowest successful cell (`jax-cpu`, 0.212 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x32x32`): `tenferro-trace` is 15.9x faster than the slowest successful cell (`jax-cpu`, 0.212 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 10.4x faster than the slowest successful cell (`pytorch-cpu`, 0.024 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 23.5x faster than the slowest successful cell (`pytorch-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 16.4x faster than the slowest successful cell (`pytorch-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 23.1x faster than the slowest successful cell (`pytorch-cpu`, 0.053 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.053 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 13.6x faster than the slowest successful cell (`pytorch-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 12.2x faster than the slowest successful cell (`pytorch-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x2x2,rhs=1`): `tenferro-direct` is 15.8x faster than the slowest successful cell (`jax-cpu`, 0.029 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x32x32,rhs=1`): `tenferro-direct` is 13.2x faster than the slowest successful cell (`jax-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x2x2,rhs=1`): `tenferro-direct` is 13.9x faster than the slowest successful cell (`pytorch-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x4x4,rhs=1`): `tenferro-direct` is 13.2x faster than the slowest successful cell (`pytorch-cpu`, 0.053 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x8x8,rhs=1`): `tenferro-direct` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 0.055 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svdvals` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `jax-cpu` is 22.0x faster than the slowest successful cell (`julia-base`, 378.315 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `pytorch-cpu` is 21.3x faster than the slowest successful cell (`julia-base`, 378.315 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-direct` is 48.3x faster than the slowest successful cell (`julia-base`, 378.315 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-trace` is 48.5x faster than the slowest successful cell (`julia-base`, 378.315 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 18.2x faster than the slowest successful cell (`julia-base`, 232.637 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 24.7x faster than the slowest successful cell (`julia-base`, 232.637 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 30.5x faster than the slowest successful cell (`julia-base`, 232.637 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 30.9x faster than the slowest successful cell (`julia-base`, 232.637 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `jax-cpu` is 35.1x faster than the slowest successful cell (`julia-base`, 315.796 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `pytorch-cpu` is 38.5x faster than the slowest successful cell (`julia-base`, 315.796 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-direct` is 50.4x faster than the slowest successful cell (`julia-base`, 315.796 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-trace` is 50.4x faster than the slowest successful cell (`julia-base`, 315.796 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `jax-cpu` is 38.7x faster than the slowest successful cell (`julia-base`, 316.022 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `pytorch-cpu` is 36.8x faster than the slowest successful cell (`julia-base`, 316.022 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-direct` is 44.7x faster than the slowest successful cell (`julia-base`, 316.022 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-trace` is 44.5x faster than the slowest successful cell (`julia-base`, 316.022 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `jax-cpu` is 17.7x faster than the slowest successful cell (`julia-base`, 655.382 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 25.1x faster than the slowest successful cell (`julia-base`, 655.382 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-direct` is 34.8x faster than the slowest successful cell (`julia-base`, 655.382 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-trace` is 33.0x faster than the slowest successful cell (`julia-base`, 655.382 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 34.8x faster than the slowest successful cell (`julia-base`, 399.758 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 22.0x faster than the slowest successful cell (`julia-base`, 399.758 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-direct` is 48.6x faster than the slowest successful cell (`julia-base`, 399.758 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-trace` is 45.4x faster than the slowest successful cell (`julia-base`, 399.758 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 28.0x faster than the slowest successful cell (`julia-base`, 153.624 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-eager` is 18.0x faster than the slowest successful cell (`julia-base`, 153.624 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-trace` is 17.1x faster than the slowest successful cell (`julia-base`, 153.624 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 15.1x faster than the slowest successful cell (`julia-base`, 231.355 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 16.1x faster than the slowest successful cell (`julia-base`, 231.355 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 31.5x faster than the slowest successful cell (`julia-base`, 231.355 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 31.5x faster than the slowest successful cell (`julia-base`, 231.355 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 14.1x faster than the slowest successful cell (`julia-base`, 9.439 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 13.5x faster than the slowest successful cell (`julia-base`, 9.439 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `jax-cpu` is 64.0x faster than the slowest successful cell (`julia-base`, 1031.977 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 79.3x faster than the slowest successful cell (`julia-base`, 1031.977 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 96.4x faster than the slowest successful cell (`julia-base`, 1031.977 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 100.5x faster than the slowest successful cell (`julia-base`, 1031.977 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `jax-cpu` is 77.2x faster than the slowest successful cell (`julia-base`, 1344.165 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 110.6x faster than the slowest successful cell (`julia-base`, 1344.165 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 128.5x faster than the slowest successful cell (`julia-base`, 1344.165 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 133.2x faster than the slowest successful cell (`julia-base`, 1344.165 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 25.0x faster than the slowest successful cell (`julia-base`, 267.421 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 29.7x faster than the slowest successful cell (`julia-base`, 267.421 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 35.0x faster than the slowest successful cell (`julia-base`, 267.421 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 35.9x faster than the slowest successful cell (`julia-base`, 267.421 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 33.3x faster than the slowest successful cell (`julia-base`, 1778.656 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 42.7x faster than the slowest successful cell (`julia-base`, 1778.656 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-eager` is 49.6x faster than the slowest successful cell (`julia-base`, 1778.656 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-trace` is 51.5x faster than the slowest successful cell (`julia-base`, 1778.656 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=1, shape=`1024x64`): `pytorch-cpu` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 23.5x faster than the slowest successful cell (`tenferro-eager`, 13.432 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 11252.2x faster than the slowest successful cell (`julia-base`, 15.944 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 5441.5x faster than the slowest successful cell (`julia-base`, 14.284 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 8614.8x faster than the slowest successful cell (`julia-base`, 15.791 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 8579.0x faster than the slowest successful cell (`julia-base`, 12.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 147395.5x faster than the slowest successful cell (`julia-base`, 233.327 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-direct` is 10.8x faster than the slowest successful cell (`julia-base`, 200.029 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 137194.4x faster than the slowest successful cell (`julia-base`, 200.029 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 235.4x faster than the slowest successful cell (`tenferro-direct`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 244.2x faster than the slowest successful cell (`tenferro-direct`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 88.3x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 144.5x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-direct` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 68.5x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 187.6x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 11834.2 GB/s at 0.001 ms. The conservative physical lower bound is 0.168 ms, so the cell is 118.3x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 10319.4 GB/s at 0.002 ms. The conservative physical lower bound is 0.042 ms, so the cell is 25.8x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 189439.3 GB/s at 0.001 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1894.4x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 102261.1 GB/s at 0.003 ms. The conservative physical lower bound is 0.671 ms, so the cell is 255.7x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 146446.0 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1464.5x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 189573.1 GB/s at 0.001 ms. The conservative physical lower bound is 0.671 ms, so the cell is 473.9x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 84786.9 GB/s at 0.002 ms. The conservative physical lower bound is 1.342 ms, so the cell is 847.9x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 92056.1 GB/s at 0.001 ms. The conservative physical lower bound is 0.336 ms, so the cell is 230.1x faster than the bound.

# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20261108_182210/run.yaml`
- Timestamp: `2026-11-08T18:22:10.663541+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20261108_182210`.

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

- CSV: `data/results/mac-cpu/cpu/public_api/20261108_182210/cpu_public_api_t1_20261108_182210.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20261108_182210/cpu_public_api_t4_20261108_182210.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 3.810 ± 0.151 | - | 3.543 ± 0.062 | 5.122 ± 1.885 | 2.025 ± 0.126 | 15.967 ± 9.681 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 3.927 ± 0.144 | - | 3.811 ± 0.159 | 5.035 ± 1.898 | 2.086 ± 0.281 | 73.895 ± 11.083 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 11.585 ± 0.563 | - | 11.787 ± 0.959 | 12.896 ± 3.085 | 12.104 ± 0.593 | 12.323 ± 4.218 | 10.532 ± 0.219 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 21.909 ± 4.024 | - | 14.202 ± 1.855 | 13.907 ± 2.351 | 14.430 ± 1.045 | 12.576 ± 1.534 | 10.931 ± 0.549 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 45.174 ± 1.575 | - | 53.351 ± 4.269 | 19.287 ± 1.345 | 39.705 ± 0.675 | 47.725 ± 1.760 | 52.884 ± 1.836 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 27.812 ± 5.204 | - | 25.878 ± 5.485 | 9.291 ± 0.902 | 25.852 ± 6.088 | 46.451 ± 4.037 | 21.052 ± 4.462 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 10.654 ± 1.069 | - | 10.746 ± 0.274 | 12.263 ± 1.692 | 51.416 ± 4.313 | 65.310 ± 12.843 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 11.434 ± 1.379 | - | 11.244 ± 0.663 | 11.128 ± 0.473 | 17.347 ± 5.219 | 30.004 ± 5.868 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 11.856 ± 0.542 | - | unsupported | 15.283 ± 4.745 | 67.553 ± 18.916 | 94.507 ± 31.548 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 12.422 ± 1.449 | - | unsupported | 11.172 ± 0.432 | 17.472 ± 0.748 | 28.251 ± 3.837 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 11.296 ± 0.476 | - | 9.785 ± 0.514 | 19.437 ± 9.868 | 23.215 ± 1.548 | 11.333 ± 2.057 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 12.501 ± 0.695 | - | 11.832 ± 4.547 | 14.621 ± 1.294 | 20.299 ± 0.147 | 453.315 ± 432.296 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 47.460 ± 11.526 | - | 35.067 ± 1.223 | 34.908 ± 0.644 | 50.215 ± 12.068 | 46.612 ± 0.851 | 50.018 ± 0.777 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 27.656 ± 6.264 | - | 23.713 ± 10.447 | 15.200 ± 5.140 | 24.989 ± 4.404 | 46.973 ± 3.740 | 23.374 ± 2.705 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 45.828 ± 0.408 | - | 46.122 ± 0.638 | 57.643 ± 1.552 | 96.883 ± 7.451 | 76.983 ± 0.518 | 75.399 ± 1.748 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 22.906 ± 3.307 | - | 16.136 ± 6.118 | 25.441 ± 4.896 | 34.411 ± 17.247 | 71.197 ± 0.688 | 33.164 ± 9.728 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 9.085 ± 0.881 | - | 8.845 ± 0.119 | 9.458 ± 0.198 | 11.019 ± 1.631 | 10.843 ± 1.054 | 9.553 ± 1.166 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 13.936 ± 3.044 | - | 11.249 ± 1.726 | 9.954 ± 1.260 | 11.533 ± 1.949 | 10.919 ± 0.387 | 11.433 ± 0.729 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.019 ± 0.018 | - | 7.104 ± 0.458 | 9.570 ± 8.983 | 5.679 ± 0.188 | 9.280 ± 1.354 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 1.898 ± 1.913 | - | 5.580 ± 0.410 | 3.836 ± 0.444 | 3.038 ± 0.151 | 11.574 ± 2.062 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 8.165 ± 0.156 | - | 6.664 ± 0.077 | 10.796 ± 3.010 | 7.552 ± 1.145 | 14.019 ± 4.010 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 7.991 ± 0.285 | - | 6.588 ± 0.251 | 8.888 ± 3.963 | 6.860 ± 0.080 | 154.008 ± 11.221 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 5.001 ± 0.371 | - | 5.169 ± 0.232 | 10.036 ± 6.069 | 2.980 ± 0.169 | 9.273 ± 3.149 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 6.367 ± 0.791 | - | 6.098 ± 0.859 | 7.038 ± 0.529 | 3.298 ± 0.694 | 155.101 ± 22.232 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 6.573 ± 0.173 | - | 7.849 ± 0.190 | 9.296 ± 3.601 | 8.622 ± 3.060 | 8.594 ± 0.822 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 6.769 ± 0.623 | - | 8.222 ± 0.114 | 7.762 ± 0.319 | 6.812 ± 0.258 | 625.930 ± 670.085 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 10.624 ± 0.246 | - | 10.507 ± 0.188 | 11.951 ± 1.465 | 62.875 ± 13.937 | 78.993 ± 12.076 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 11.386 ± 0.741 | - | 10.827 ± 0.433 | 11.348 ± 1.768 | 17.291 ± 2.527 | 28.516 ± 6.024 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 14.419 ± 7.159 | - | unsupported | 10.753 ± 0.709 | 60.628 ± 1.219 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 11.024 ± 1.997 | - | unsupported | 12.577 ± 1.091 | 21.662 ± 5.879 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 10.317 ± 0.126 | - | 10.845 ± 0.970 | 11.579 ± 0.806 | 15.602 ± 2.208 | 14.571 ± 5.191 | 15.672 ± 10.549 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 14.554 ± 2.730 | - | 14.233 ± 3.324 | 12.394 ± 2.132 | 15.748 ± 2.492 | 12.630 ± 2.272 | 15.679 ± 19.634 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 16.876 ± 2.940 | - | 17.794 ± 0.863 | 16.603 ± 0.795 | 25.346 ± 2.539 | 28.355 ± 2.329 | 21.267 ± 6.599 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 23.473 ± 2.477 | - | 24.238 ± 3.455 | 19.093 ± 2.667 | 25.960 ± 0.387 | 18.636 ± 10.887 | 20.060 ± 2.634 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 25.460 ± 0.690 | - | 33.499 ± 1.146 | 41.726 ± 4.567 | 31.880 ± 3.256 | 66.864 ± 8.370 | 55.360 ± 5.710 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 28.064 ± 11.539 | - | 26.863 ± 7.285 | 26.268 ± 10.907 | 20.558 ± 8.124 | 57.193 ± 5.788 | 22.610 ± 2.287 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 5.761 ± 0.241 | - | 5.640 ± 0.062 | 6.276 ± 1.033 | 10.384 ± 1.367 | 8.677 ± 8.135 | 9.455 ± 63.127 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 8.184 ± 0.937 | - | 8.237 ± 1.664 | 6.928 ± 0.431 | 9.535 ± 1.104 | 8.856 ± 2.480 | 7.069 ± 17.437 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 9.965 ± 0.048 | - | 10.398 ± 0.334 | 15.840 ± 0.646 | 17.149 ± 0.689 | 25.406 ± 7.941 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 13.941 ± 1.548 | - | 13.654 ± 0.699 | 17.189 ± 8.161 | 15.314 ± 2.187 | 20.439 ± 7.054 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 32.938 ± 15.894 | - | 35.291 ± 4.655 | 33.995 ± 0.725 | 33.319 ± 6.906 | 41.342 ± 5.850 | 53.958 ± 15.151 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 23.748 ± 5.616 | - | 24.400 ± 7.644 | 18.102 ± 2.916 | 14.567 ± 0.638 | 37.254 ± 0.600 | 18.016 ± 3.333 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 18.255 ± 0.256 | - | 17.929 ± 2.055 | 21.580 ± 1.934 | 30.027 ± 10.040 | 24.256 ± 13.705 | 22.654 ± 3.785 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 21.471 ± 2.241 | - | 27.335 ± 3.479 | 24.758 ± 3.691 | 27.952 ± 3.605 | 21.257 ± 2.270 | 21.333 ± 3.170 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 16.643 ± 0.047 | - | 17.030 ± 2.445 | 25.583 ± 0.699 | 17.468 ± 2.274 | 39.115 ± 3.365 | 39.317 ± 3.410 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 13.015 ± 9.120 | - | 12.280 ± 5.103 | 11.970 ± 1.853 | 8.913 ± 0.775 | 36.409 ± 3.854 | 25.935 ± 8.259 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 24.717 ± 0.208 | - | 24.706 ± 0.996 | 28.748 ± 0.299 | 18.821 ± 0.572 | 34.525 ± 15.729 | 34.795 ± 7.904 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 18.163 ± 2.666 | - | 20.681 ± 12.402 | 14.594 ± 2.848 | 12.836 ± 4.085 | 29.034 ± 2.311 | 16.223 ± 2.341 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 31.477 ± 1.413 | - | 33.248 ± 0.690 | 44.479 ± 7.163 | 27.472 ± 4.209 | 47.816 ± 23.843 | 50.719 ± 3.343 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 19.973 ± 5.367 | - | 21.677 ± 4.375 | 18.173 ± 2.005 | 19.702 ± 4.583 | 55.946 ± 13.448 | 21.724 ± 3.109 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 15.171 ± 0.284 | - | 15.103 ± 0.151 | 22.826 ± 0.975 | 20.778 ± 0.666 | 41.613 ± 14.252 | 47.731 ± 13.661 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 11.223 ± 4.918 | - | 12.692 ± 3.836 | 16.494 ± 6.668 | 14.582 ± 4.406 | 28.367 ± 5.100 | 13.612 ± 2.689 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.242 ± 0.004 | 0.259 ± 0.027 | unsupported | unsupported | 0.312 ± 0.071 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.436 ± 0.370 | 0.594 ± 0.425 | unsupported | unsupported | 0.271 ± 0.148 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 16.858 ± 1.273 | - | 16.622 ± 0.797 | 23.267 ± 4.733 | 27.802 ± 1.082 | 21.848 ± 2.922 | 23.202 ± 3.236 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 23.030 ± 3.287 | - | 22.564 ± 3.066 | 19.206 ± 1.187 | 30.603 ± 1.723 | 26.985 ± 9.807 | 21.771 ± 10.861 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 14.854 ± 0.935 | - | 14.232 ± 0.180 | 20.355 ± 4.812 | 27.430 ± 2.426 | 21.064 ± 2.158 | 21.599 ± 0.591 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 23.563 ± 4.345 | - | 24.933 ± 6.638 | 20.668 ± 2.509 | 28.787 ± 1.028 | 31.526 ± 5.033 | 25.685 ± 4.558 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 18.889 ± 1.787 | - | 18.033 ± 0.661 | 16.120 ± 0.715 | 28.852 ± 4.072 | 20.487 ± 4.860 | 20.270 ± 1.215 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 26.788 ± 2.755 | - | 20.978 ± 2.919 | 23.472 ± 8.145 | 26.364 ± 2.170 | 21.087 ± 6.729 | 18.951 ± 1.479 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 10.736 ± 1.026 | - | 10.725 ± 0.365 | 11.505 ± 0.416 | 15.270 ± 1.013 | 14.993 ± 5.355 | 17.010 ± 3.884 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 14.144 ± 3.236 | - | 15.998 ± 3.393 | 12.310 ± 2.449 | 13.577 ± 3.640 | 13.660 ± 2.873 | 16.116 ± 4.944 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 35.247 ± 1.447 | - | 33.842 ± 0.098 | 69.929 ± 0.342 | 48.951 ± 3.616 | 86.953 ± 37.338 | 101.141 ± 18.977 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 27.520 ± 6.356 | - | 28.690 ± 8.075 | 25.023 ± 2.967 | 30.735 ± 8.250 | 74.673 ± 36.802 | 65.692 ± 22.498 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 6.421 ± 0.100 | - | 6.227 ± 0.031 | 12.430 ± 0.814 | 21.698 ± 0.151 | 7.820 ± 0.655 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 6.983 ± 0.946 | - | 8.061 ± 2.053 | 7.285 ± 2.110 | 17.540 ± 2.284 | 6.335 ± 1.346 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 0.805 ± 0.003 | - | 0.823 ± 0.004 | 17.324 ± 1.957 | 2.472 ± 0.105 | 1.060 ± 0.209 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.876 ± 0.066 | - | 1.027 ± 0.314 | 12.937 ± 2.158 | 2.192 ± 0.646 | 0.933 ± 0.078 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 0.909 ± 0.257 | - | 0.747 ± 0.020 | 1.830 ± 0.092 | 1.232 ± 0.282 | 1.004 ± 0.128 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 2.458 ± 0.535 | - | 2.332 ± 0.719 | 0.974 ± 0.070 | 1.615 ± 0.663 | 1.144 ± 0.385 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 6.260 ± 0.032 | - | 6.303 ± 0.121 | 12.810 ± 0.830 | 22.389 ± 1.114 | 7.292 ± 0.566 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 7.238 ± 1.943 | - | 7.430 ± 2.468 | 7.354 ± 0.758 | 17.432 ± 4.079 | 6.659 ± 2.987 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 0.822 ± 0.005 | - | 0.819 ± 0.029 | 15.714 ± 1.350 | 3.498 ± 0.934 | 0.983 ± 0.132 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 1.067 ± 0.314 | - | 1.014 ± 0.123 | 13.057 ± 1.571 | 2.179 ± 0.081 | 1.098 ± 0.178 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 3.282 ± 0.124 | - | 3.004 ± 0.457 | 6.944 ± 0.176 | 4.379 ± 1.182 | 4.228 ± 0.399 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 8.874 ± 0.318 | - | 9.437 ± 1.394 | 3.975 ± 0.237 | 6.259 ± 0.388 | 4.387 ± 0.979 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 6.011 ± 0.146 | - | 5.872 ± 0.063 | 6.268 ± 0.364 | 40.853 ± 1.820 | 7.297 ± 0.117 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 6.455 ± 0.288 | - | 7.096 ± 1.399 | 9.623 ± 3.317 | 22.132 ± 3.565 | 6.014 ± 1.369 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.707 ± 0.025 | - | 0.729 ± 0.050 | 5.663 ± 0.589 | 3.528 ± 0.832 | 0.959 ± 0.289 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.792 ± 0.144 | - | 1.181 ± 0.484 | 3.296 ± 0.538 | 2.690 ± 1.103 | 1.080 ± 0.309 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 0.736 ± 0.021 | - | 0.744 ± 0.009 | 1.287 ± 0.372 | 1.451 ± 0.264 | 0.987 ± 0.129 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 3.612 ± 1.952 | - | 3.701 ± 1.816 | 1.221 ± 0.181 | 2.234 ± 0.375 | 0.994 ± 0.117 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 5.972 ± 0.693 | - | 5.632 ± 0.711 | 5.999 ± 0.203 | 34.067 ± 1.155 | 8.381 ± 0.104 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 7.960 ± 2.533 | - | 6.917 ± 1.973 | 7.567 ± 1.842 | 19.898 ± 3.073 | 4.983 ± 0.087 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.560 ± 0.011 | - | 0.563 ± 0.033 | 4.466 ± 0.102 | 2.609 ± 0.259 | 0.805 ± 0.141 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.971 ± 0.086 | - | 1.544 ± 0.493 | 2.946 ± 0.157 | 1.527 ± 0.334 | 0.875 ± 0.221 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 0.762 ± 0.091 | - | 0.718 ± 0.026 | 0.854 ± 0.033 | 1.234 ± 0.255 | 1.034 ± 0.312 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 2.331 ± 0.323 | - | 2.544 ± 0.885 | 0.857 ± 0.076 | 1.141 ± 0.309 | 1.001 ± 0.153 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.678 ± 0.059 | 0.715 ± 0.041 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 1.084 ± 0.280 | 0.978 ± 0.149 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 28.772 ± 0.412 | - | 28.379 ± 0.245 | 32.635 ± 3.063 | 39.487 ± 6.317 | 30.776 ± 8.223 | 32.810 ± 2.267 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 17.821 ± 3.344 | - | 18.633 ± 5.955 | 37.959 ± 13.654 | 25.567 ± 5.092 | 26.633 ± 1.520 | 18.447 ± 3.353 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 20.151 ± 0.948 | - | 19.567 ± 0.542 | 18.562 ± 2.545 | 39.305 ± 2.610 | 43.907 ± 1.606 | 45.169 ± 10.272 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 19.691 ± 7.174 | - | 20.264 ± 7.654 | 18.237 ± 2.766 | 21.974 ± 3.060 | 46.936 ± 1.610 | 21.458 ± 3.484 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 21.138 ± 1.215 | - | 23.723 ± 1.634 | 20.780 ± 2.222 | 29.169 ± 1.073 | 30.415 ± 2.196 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 27.205 ± 1.848 | - | 29.566 ± 2.503 | 24.827 ± 5.220 | 34.753 ± 2.587 | 28.220 ± 6.669 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 10.805 ± 1.698 | - | 10.507 ± 0.856 | 11.527 ± 0.469 | 19.626 ± 7.714 | 13.368 ± 1.033 | 20.715 ± 7.102 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 15.070 ± 2.513 | - | 24.999 ± 8.035 | 11.908 ± 2.534 | 15.572 ± 2.686 | 11.849 ± 0.423 | 16.375 ± 5.065 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 25.494 ± 0.963 | - | 24.586 ± 0.163 | 30.233 ± 2.285 | 31.802 ± 11.500 | 34.807 ± 0.767 | 35.497 ± 5.433 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 18.541 ± 6.929 | - | 18.722 ± 6.844 | 13.404 ± 1.855 | 21.594 ± 4.705 | 30.410 ± 0.601 | 15.379 ± 2.536 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 12.819 ± 3.436 | - | 14.498 ± 1.245 | 12.800 ± 1.014 | 23.252 ± 2.380 | 32.485 ± 4.553 | 28.095 ± 7.576 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 14.763 ± 1.910 | - | 17.648 ± 3.304 | 16.907 ± 4.095 | 16.192 ± 2.160 | 31.214 ± 3.798 | 17.368 ± 2.615 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 17.994 ± 1.818 | - | 18.284 ± 2.830 | 15.921 ± 0.286 | 28.241 ± 4.746 | 18.799 ± 2.869 | 20.009 ± 2.740 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 22.967 ± 2.303 | - | 22.610 ± 1.890 | 24.421 ± 4.325 | 28.229 ± 2.517 | 26.414 ± 3.952 | 21.221 ± 1.267 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 67.552 ± 0.329 | - | 66.765 ± 0.598 | 72.433 ± 5.151 | 14.252 ± 3.252 | 62.447 ± 11.783 | 74.098 ± 21.679 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 46.944 ± 9.124 | - | 43.784 ± 35.720 | 37.648 ± 2.665 | 10.767 ± 5.289 | 44.159 ± 1.323 | 20.392 ± 9.860 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 23.703 ± 1.707 | 25.939 ± 4.812 | unsupported | 20.051 ± 1.249 | 28.756 ± 1.031 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 29.279 ± 2.648 | 36.078 ± 6.334 | unsupported | 21.029 ± 2.241 | 33.836 ± 4.964 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.618 ± 0.024 | - | 0.627 ± 0.023 | 0.830 ± 0.210 | 0.825 ± 0.117 | 0.962 ± 0.239 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 1.706 ± 1.338 | - | 0.831 ± 0.033 | 1.166 ± 0.562 | 0.642 ± 0.050 | 0.734 ± 0.098 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.748 ± 0.135 | - | 0.642 ± 0.079 | 0.823 ± 0.107 | 1.187 ± 0.521 | 0.918 ± 0.334 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.797 ± 0.047 | - | 0.776 ± 0.040 | 0.838 ± 0.126 | 1.006 ± 0.194 | 0.879 ± 0.098 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.006 ± 0.082 | - | unsupported | 1.249 ± 0.531 | 1.352 ± 0.132 | 1.434 ± 0.289 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 1.493 ± 0.463 | - | unsupported | 1.834 ± 0.129 | 1.055 ± 0.241 | 1.168 ± 0.135 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.330 ± 0.018 | - | 0.406 ± 0.091 | 0.419 ± 0.088 | 0.411 ± 0.066 | 1.634 ± 2.217 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.492 ± 0.152 | - | 0.785 ± 0.096 | 0.306 ± 0.044 | 0.551 ± 0.238 | 1.192 ± 1.139 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.653 ± 0.055 | - | 0.664 ± 0.195 | 1.248 ± 0.098 | 1.356 ± 0.452 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 1.056 ± 0.362 | - | 1.420 ± 0.534 | 1.988 ± 0.718 | 0.783 ± 0.054 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.619 ± 0.015 | - | 0.620 ± 0.032 | 0.874 ± 0.185 | 0.777 ± 0.030 | 1.187 ± 0.364 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.861 ± 0.325 | - | 0.875 ± 0.137 | 1.022 ± 0.509 | 0.832 ± 0.053 | 1.346 ± 0.389 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.540 ± 0.011 | - | 0.674 ± 0.064 | 0.540 ± 0.128 | 0.465 ± 0.101 | 0.377 ± 0.053 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.913 ± 0.451 | - | 0.991 ± 0.353 | 0.555 ± 0.149 | 1.052 ± 0.523 | 1.286 ± 2.460 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.090 ± 0.133 | - | 0.002 ± 0.000 | 1.371 ± 0.427 | 1.497 ± 0.181 | 1.780 ± 1.266 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 1.578 ± 0.702 | - | 0.004 ± 0.000 | 1.547 ± 0.146 | 1.325 ± 0.071 | 1.531 ± 0.118 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.055 ± 0.004 | - | 0.079 ± 0.002 | 0.145 ± 0.022 | 0.103 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.095 ± 0.007 | - | 0.120 ± 0.001 | 0.209 ± 0.028 | 0.158 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.200 ± 0.001 | - | 0.219 ± 0.003 | 0.302 ± 0.012 | 0.485 ± 0.298 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.026 ± 0.001 | - | 0.026 ± 0.002 | 0.039 ± 0.003 | 0.067 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.104 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.034 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.118 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.168 ± 0.010 | - | 0.173 ± 0.006 | 0.265 ± 0.010 | 0.261 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.010 ± 0.000 | - | 0.012 ± 0.001 | 0.017 ± 0.001 | 0.084 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.074 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.062 ± 0.002 | - | 0.088 ± 0.036 | 0.334 ± 0.105 | 0.142 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.100 ± 0.049 | - | 0.123 ± 0.009 | 0.311 ± 0.052 | 0.145 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.219 ± 0.058 | - | 0.227 ± 0.004 | 0.445 ± 0.085 | 0.389 ± 0.180 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.027 ± 0.001 | - | 0.049 ± 0.004 | 0.187 ± 0.094 | 0.151 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.002 ± 0.006 | - | 0.004 ± 0.000 | 0.050 ± 0.074 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.061 ± 0.060 | 0.077 ± 0.092 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.009 ± 0.010 | 0.089 ± 0.055 | 0.036 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.009 ± 0.000 | 0.047 ± 0.014 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.124 ± 0.023 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.010 ± 0.011 | 0.119 ± 0.039 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.005 ± 0.000 | - | 0.010 ± 0.002 | 0.110 ± 0.016 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.059 ± 0.077 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.089 ± 0.061 | 0.009 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.003 ± 0.000 | - | 0.010 ± 0.000 | 0.072 ± 0.043 | 0.059 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.197 ± 0.123 | - | 0.195 ± 0.081 | 0.467 ± 0.145 | 0.550 ± 0.219 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.080 ± 0.028 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.012 ± 0.001 | - | 0.014 ± 0.001 | 0.117 ± 0.093 | 0.071 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.091 ± 0.029 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.001 | 0.107 ± 0.028 | 0.073 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.247 ± 0.001 | - | 0.249 ± 0.001 | 0.419 ± 0.004 | 0.897 ± 0.109 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.467 ± 0.027 | - | 1.472 ± 0.035 | 1.680 ± 0.045 | 2.155 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 6.076 ± 0.176 | - | 6.067 ± 0.043 | 6.894 ± 1.045 | 7.461 ± 0.092 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 4.361 ± 0.055 | - | 6.155 ± 0.048 | 3.060 ± 0.055 | 4.770 ± 1.138 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.090 ± 0.001 | - | 0.091 ± 0.000 | 0.090 ± 0.006 | 0.164 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.019 ± 0.001 | - | 0.020 ± 0.001 | 0.023 ± 0.000 | 0.096 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.024 ± 0.005 | - | 0.026 ± 0.000 | 0.029 ± 0.001 | 0.072 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 43.491 ± 9.364 | - | 58.826 ± 16.014 | 26.114 ± 0.824 | 38.351 ± 8.256 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.698 ± 0.036 | - | 0.697 ± 0.017 | 0.692 ± 0.032 | 1.082 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.013 ± 0.000 | - | 0.014 ± 0.000 | 0.018 ± 0.000 | 0.026 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.048 ± 0.001 | - | 0.049 ± 0.001 | 0.055 ± 0.006 | 0.103 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.658 ± 0.592 | - | 0.850 ± 0.231 | 0.422 ± 0.005 | 0.992 ± 0.145 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 2.150 ± 0.554 | - | 1.759 ± 0.508 | 1.759 ± 0.185 | 1.236 ± 0.093 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 8.011 ± 1.167 | - | 8.228 ± 0.857 | 6.918 ± 2.335 | 3.727 ± 0.120 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 6.718 ± 2.248 | - | 7.482 ± 0.826 | 3.342 ± 0.165 | 4.012 ± 0.068 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.001 | 0.005 ± 0.001 | 0.011 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.097 ± 0.010 | - | 0.097 ± 0.025 | 0.093 ± 0.005 | 0.443 ± 0.224 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.011 ± 0.001 | - | 0.017 ± 0.002 | 0.007 ± 0.002 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.023 ± 0.000 | - | 0.029 ± 0.001 | 0.011 ± 0.000 | 0.018 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.001 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.007 ± 0.000 | - | 0.010 ± 0.002 | 0.010 ± 0.003 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.055 ± 0.001 | - | 0.061 ± 0.001 | 0.027 ± 0.005 | 0.101 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.010 ± 0.001 | - | 0.010 ± 0.001 | 0.011 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.071 ± 0.003 | - | 0.079 ± 0.060 | 0.032 ± 0.007 | 0.098 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 50.348 ± 11.410 | - | 50.252 ± 3.082 | 28.959 ± 1.456 | 16.962 ± 0.735 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.783 ± 0.037 | - | 1.152 ± 1.362 | 0.765 ± 0.078 | 0.923 ± 0.094 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.015 ± 0.003 | - | 0.015 ± 0.001 | 0.018 ± 0.004 | 0.027 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.060 ± 0.007 | - | 0.058 ± 0.001 | 0.064 ± 0.004 | 0.247 ± 0.114 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.239 ± 0.001 | - | 0.247 ± 0.015 | 0.258 ± 0.039 | 0.656 ± 0.123 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.261 ± 0.134 | - | 1.230 ± 0.016 | 1.348 ± 0.141 | 1.844 ± 0.106 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 4.830 ± 0.163 | - | 4.836 ± 0.066 | 4.979 ± 0.553 | 6.126 ± 0.266 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.017 ± 0.097 | - | 0.930 ± 0.010 | 0.953 ± 0.102 | 1.280 ± 0.122 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.064 ± 0.003 | - | 0.068 ± 0.001 | 0.072 ± 0.007 | 0.117 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.002 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.002 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.015 ± 0.001 | - | 0.016 ± 0.000 | 0.019 ± 0.001 | 0.026 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.002 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.020 ± 0.000 | - | 0.021 ± 0.000 | 0.024 ± 0.001 | 0.034 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 7.432 ± 0.054 | - | 7.423 ± 0.014 | 7.818 ± 0.245 | 9.251 ± 0.334 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.523 ± 0.006 | - | 0.492 ± 0.005 | 0.514 ± 0.013 | 0.648 ± 0.065 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.014 ± 0.004 | 0.025 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.039 ± 0.000 | - | 0.040 ± 0.000 | 0.042 ± 0.001 | 0.099 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.279 ± 0.043 | - | 0.282 ± 0.053 | 0.295 ± 0.075 | 0.609 ± 0.114 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.446 ± 0.187 | - | 1.531 ± 0.662 | 1.426 ± 0.148 | 1.024 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 5.543 ± 0.939 | - | 6.104 ± 0.850 | 5.495 ± 1.147 | 3.524 ± 0.286 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.126 ± 0.064 | - | 1.275 ± 0.125 | 1.056 ± 0.070 | 1.143 ± 0.084 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.002 ± 0.006 | - | 0.012 ± 0.002 | 0.006 ± 0.002 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.073 ± 0.088 | - | 0.070 ± 0.002 | 0.073 ± 0.006 | 0.188 ± 0.076 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.012 ± 0.002 | - | 0.017 ± 0.002 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.007 ± 0.002 | - | 0.025 ± 0.001 | 0.011 ± 0.002 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.008 ± 0.001 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.017 ± 0.003 | - | 0.013 ± 0.012 | 0.009 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.019 ± 0.004 | - | 0.052 ± 0.001 | 0.019 ± 0.005 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.022 ± 0.005 | - | 0.027 ± 0.002 | 0.010 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.058 ± 0.001 | - | 0.065 ± 0.002 | 0.027 ± 0.005 | 0.031 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 11.397 ± 4.689 | - | 9.776 ± 3.349 | 9.402 ± 2.004 | 4.694 ± 0.314 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.011 ± 0.001 | - | 0.017 ± 0.011 | 0.006 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.549 ± 0.017 | - | 0.652 ± 0.098 | 0.541 ± 0.048 | 0.397 ± 0.104 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.029 ± 0.013 | - | 0.013 ± 0.003 | 0.014 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.111 ± 0.037 | - | 0.116 ± 0.086 | 0.044 ± 0.002 | 0.149 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.155 ± 0.017 | - | 0.162 ± 0.009 | 0.128 ± 0.018 | 0.920 ± 0.386 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 0.816 ± 0.100 | - | 0.765 ± 0.022 | 0.749 ± 0.015 | 2.406 ± 0.551 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 2.504 ± 0.100 | - | 2.465 ± 0.105 | 2.474 ± 0.243 | 8.641 ± 2.242 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.469 ± 0.049 | - | 0.488 ± 0.033 | 0.465 ± 0.034 | 1.696 ± 0.571 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.002 ± 0.001 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.025 ± 0.001 | - | 0.027 ± 0.003 | 0.031 ± 0.003 | 0.271 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.002 | 0.040 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.002 | 0.039 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.015 ± 0.003 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.006 ± 0.001 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.006 ± 0.002 | - | 0.007 ± 0.001 | 0.008 ± 0.000 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.010 ± 0.000 | - | 0.011 ± 0.000 | 0.015 ± 0.003 | 0.057 ± 0.063 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 3.937 ± 0.245 | - | 3.847 ± 0.151 | 3.679 ± 0.041 | 11.719 ± 3.420 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.189 ± 0.001 | - | 0.202 ± 0.012 | 0.194 ± 0.000 | 0.901 ± 0.383 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.008 ± 0.005 | - | 0.010 ± 0.002 | 0.011 ± 0.001 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.025 ± 0.008 | - | 0.025 ± 0.002 | 0.024 ± 0.002 | 0.232 ± 0.121 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.118 ± 0.001 | - | 0.137 ± 0.018 | 0.146 ± 0.032 | 0.574 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 0.699 ± 0.001 | - | 0.700 ± 0.011 | 0.888 ± 0.211 | 1.010 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 2.245 ± 0.015 | - | 2.250 ± 0.020 | 2.718 ± 0.361 | 2.868 ± 0.088 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.429 ± 0.003 | - | 0.444 ± 0.015 | 0.518 ± 0.058 | 1.131 ± 0.098 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.001 | 0.005 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.025 ± 0.000 | - | 0.026 ± 0.001 | 0.032 ± 0.004 | 0.139 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.003 ± 0.002 | - | 0.005 ± 0.001 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.007 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.002 ± 0.001 | - | 0.004 ± 0.001 | 0.006 ± 0.007 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.002 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.010 ± 0.002 | - | 0.009 ± 0.003 | 0.013 ± 0.007 | 0.027 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.006 ± 0.001 | - | 0.007 ± 0.000 | 0.009 ± 0.007 | 0.016 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.012 ± 0.002 | - | 0.011 ± 0.003 | 0.015 ± 0.000 | 0.029 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 3.561 ± 0.096 | - | 3.925 ± 0.110 | 6.244 ± 2.660 | 4.431 ± 0.135 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.001 | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.184 ± 0.003 | - | 0.185 ± 0.003 | 0.405 ± 0.018 | 0.331 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.009 ± 0.000 | - | 0.011 ± 0.002 | 0.011 ± 0.003 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.019 ± 0.006 | - | 0.020 ± 0.000 | 0.029 ± 0.012 | 0.108 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.089 ± 0.001 | - | 0.091 ± 0.000 | 0.201 ± 0.026 | 0.158 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.191 ± 0.002 | - | 0.192 ± 0.001 | 0.277 ± 0.014 | 0.245 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.425 ± 0.005 | - | 0.426 ± 0.002 | 0.556 ± 0.028 | 0.458 ± 0.077 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.057 ± 0.001 | - | 0.057 ± 0.001 | 0.069 ± 0.003 | 0.397 ± 0.115 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.008 ± 0.001 | 0.011 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.051 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.014 ± 0.000 | 0.055 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.008 ± 0.001 | 0.014 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.003 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.002 | 0.107 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.002 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.012 ± 0.000 | 0.077 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.434 ± 0.007 | - | 0.454 ± 0.030 | 0.487 ± 0.038 | 0.705 ± 0.080 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.032 ± 0.004 | - | 0.032 ± 0.000 | 0.039 ± 0.003 | 0.222 ± 0.188 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.052 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.001 | 0.087 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.102 ± 0.008 | - | 0.129 ± 0.015 | 0.373 ± 0.043 | 0.224 ± 0.150 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.267 ± 0.046 | - | 0.271 ± 0.268 | 0.481 ± 0.212 | 0.257 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.688 ± 0.177 | - | 0.525 ± 0.085 | 0.742 ± 0.266 | 0.351 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.065 ± 0.003 | - | 0.067 ± 0.047 | 0.304 ± 0.148 | 0.230 ± 0.062 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.142 ± 0.034 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.008 ± 0.002 | - | 0.008 ± 0.000 | 0.076 ± 0.023 | 0.062 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.005 ± 0.001 | - | 0.012 ± 0.000 | 0.112 ± 0.126 | 0.010 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.009 ± 0.001 | - | 0.011 ± 0.000 | 0.147 ± 0.043 | 0.058 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | 0.005 ± 0.000 | 0.138 ± 0.019 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.121 ± 0.061 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.008 ± 0.000 | - | 0.005 ± 0.000 | 0.129 ± 0.057 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.054 ± 0.077 | 0.132 ± 0.061 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.005 ± 0.001 | - | 0.006 ± 0.000 | 0.117 ± 0.031 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.005 ± 0.001 | - | 0.004 ± 0.000 | 0.095 ± 0.029 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.006 ± 0.001 | - | 0.005 ± 0.000 | 0.108 ± 0.038 | 0.086 ± 0.102 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.487 ± 0.056 | - | 0.534 ± 0.054 | 1.021 ± 0.370 | 0.880 ± 0.280 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.006 ± 0.001 | - | 0.006 ± 0.000 | 0.156 ± 0.079 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.039 ± 0.003 | - | 0.092 ± 0.001 | 0.267 ± 0.033 | 0.162 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.159 ± 0.070 | 0.067 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.153 ± 0.089 | 0.052 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.200 ± 0.003 | - | 0.197 ± 0.002 | 0.321 ± 0.045 | 0.561 ± 0.064 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.496 ± 0.014 | - | 0.492 ± 0.004 | 0.657 ± 0.186 | 0.931 ± 0.283 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.211 ± 0.039 | - | 1.210 ± 0.020 | 1.335 ± 0.113 | 2.599 ± 0.404 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.327 ± 0.042 | - | 0.298 ± 0.001 | 0.361 ± 0.054 | 0.388 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.012 ± 0.000 | - | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.096 ± 0.101 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.014 ± 0.000 | - | 0.016 ± 0.003 | 0.021 ± 0.001 | 0.062 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.015 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.004 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 2.369 ± 0.024 | - | 2.377 ± 0.093 | 2.895 ± 0.083 | 3.079 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.001 | 0.005 ± 0.000 | 0.015 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.087 ± 0.000 | - | 0.088 ± 0.000 | 0.093 ± 0.001 | 0.255 ± 0.070 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.011 ± 0.000 | - | 0.012 ± 0.000 | 0.013 ± 0.000 | 0.025 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.220 ± 0.057 | - | 0.242 ± 0.056 | 0.541 ± 0.281 | 0.369 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.549 ± 0.133 | - | 0.617 ± 0.108 | 1.280 ± 0.641 | 0.773 ± 0.294 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.798 ± 0.124 | - | 1.846 ± 0.109 | 1.892 ± 0.778 | 1.112 ± 0.250 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.376 ± 0.061 | - | 0.338 ± 0.054 | 0.399 ± 0.078 | 0.493 ± 0.094 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.032 ± 0.014 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.020 ± 0.005 | - | 0.026 ± 0.002 | 0.053 ± 0.036 | 0.092 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.077 ± 0.077 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.016 ± 0.003 | - | 0.017 ± 0.002 | 0.062 ± 0.051 | 0.084 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.005 ± 0.002 | - | 0.008 ± 0.001 | 0.036 ± 0.065 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.040 ± 0.039 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.005 ± 0.001 | - | 0.006 ± 0.000 | 0.039 ± 0.048 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.033 ± 0.020 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.031 ± 0.003 | 0.012 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.011 ± 0.002 | - | 0.008 ± 0.002 | 0.049 ± 0.038 | 0.014 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.009 ± 0.000 | - | 0.008 ± 0.000 | 0.087 ± 0.011 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 2.810 ± 0.114 | - | 3.177 ± 0.286 | 4.100 ± 1.197 | 1.571 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.005 ± 0.000 | - | 0.005 ± 0.000 | 0.036 ± 0.028 | 0.012 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.151 ± 0.032 | - | 0.151 ± 0.040 | 0.166 ± 0.034 | 0.159 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.009 ± 0.002 | - | 0.012 ± 0.000 | 0.032 ± 0.028 | 0.016 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.015 ± 0.001 | - | 0.017 ± 0.004 | 0.094 ± 0.027 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.134 ± 0.003 | - | 0.123 ± 0.002 | 0.105 ± 0.001 | 0.181 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.214 ± 0.010 | - | 0.194 ± 0.003 | 0.181 ± 0.031 | 0.247 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.379 ± 0.028 | - | 0.392 ± 0.026 | 0.353 ± 0.023 | 0.597 ± 0.207 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.047 ± 0.001 | - | 0.053 ± 0.001 | 0.064 ± 0.005 | 0.115 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.013 ± 0.018 | 0.010 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.014 ± 0.002 | 0.054 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.003 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.001 | 0.010 ± 0.000 | 0.011 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.003 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.364 ± 0.013 | - | 0.416 ± 0.016 | 0.415 ± 0.014 | 0.964 ± 1.187 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.013 ± 0.002 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.027 ± 0.000 | - | 0.033 ± 0.000 | 0.038 ± 0.003 | 0.135 ± 0.089 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.003 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.153 ± 0.039 | - | 0.106 ± 0.033 | 0.205 ± 0.151 | 0.182 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.214 ± 0.052 | - | 0.180 ± 0.026 | 0.370 ± 0.172 | 0.283 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.420 ± 0.055 | - | 0.448 ± 0.042 | 0.484 ± 0.272 | 0.662 ± 0.168 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.055 ± 0.003 | - | 0.063 ± 0.004 | 0.069 ± 0.006 | 0.207 ± 0.204 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.002 ± 0.001 | - | 0.017 ± 0.010 | 0.010 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.005 ± 0.001 | - | 0.012 ± 0.002 | 0.014 ± 0.004 | 0.074 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.001 | 0.010 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.003 ± 0.001 | - | 0.021 ± 0.001 | 0.030 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.026 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.004 ± 0.001 | - | 0.010 ± 0.002 | 0.031 ± 0.020 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.002 | 0.011 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.001 | 0.011 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.455 ± 0.035 | - | 0.615 ± 0.118 | 0.738 ± 0.312 | 0.568 ± 0.166 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.008 ± 0.000 | - | 0.009 ± 0.001 | 0.032 ± 0.011 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.034 ± 0.005 | - | 0.034 ± 0.007 | 0.139 ± 0.027 | 0.119 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.010 ± 0.002 | 0.024 ± 0.016 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.010 ± 0.003 | 0.027 ± 0.048 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.233 ± 0.128 | - | 1.237 ± 0.111 | 1.168 ± 0.031 | 2.956 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 4.526 ± 0.067 | - | 4.484 ± 0.113 | 4.561 ± 0.129 | 6.073 ± 0.141 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 15.544 ± 0.461 | - | 15.097 ± 0.324 | 14.564 ± 0.924 | 19.662 ± 4.909 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 5.903 ± 0.119 | - | 5.907 ± 0.066 | 4.650 ± 0.217 | 5.392 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.181 ± 0.011 | - | 0.197 ± 0.025 | 0.187 ± 0.007 | 0.314 ± 0.208 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.006 ± 0.001 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.187 ± 0.007 | - | 0.191 ± 0.010 | 0.205 ± 0.035 | 0.226 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.017 ± 0.005 | - | 0.018 ± 0.000 | 0.022 ± 0.000 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.014 ± 0.000 | - | 0.015 ± 0.000 | 0.019 ± 0.000 | 0.030 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.045 ± 0.003 | - | 0.046 ± 0.003 | 0.053 ± 0.002 | 0.063 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.018 ± 0.003 | - | 0.024 ± 0.002 | 0.023 ± 0.000 | 0.036 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.068 ± 0.008 | - | 0.066 ± 0.013 | 0.076 ± 0.013 | 0.080 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 46.364 ± 1.759 | - | 44.874 ± 0.781 | 37.639 ± 1.287 | 42.154 ± 7.727 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.011 ± 0.000 | - | 0.012 ± 0.000 | 0.015 ± 0.003 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.406 ± 0.053 | - | 1.452 ± 0.087 | 1.468 ± 0.030 | 1.748 ± 0.068 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.044 ± 0.005 | - | 0.037 ± 0.006 | 0.039 ± 0.000 | 0.197 ± 0.080 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.118 ± 0.009 | - | 0.128 ± 0.008 | 0.127 ± 0.003 | 0.238 ± 0.119 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.121 ± 0.037 | - | 1.217 ± 0.272 | 1.563 ± 0.985 | 4.070 ± 3.302 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 4.722 ± 0.290 | - | 4.623 ± 0.311 | 4.799 ± 0.755 | 4.226 ± 0.995 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 15.962 ± 0.613 | - | 18.813 ± 4.163 | 21.657 ± 1.852 | 8.738 ± 0.471 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 5.681 ± 0.165 | - | 6.000 ± 0.145 | 4.940 ± 1.226 | 5.472 ± 0.096 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.001 | 0.007 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.235 ± 0.145 | - | 0.202 ± 0.156 | 0.197 ± 0.011 | 0.258 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.008 ± 0.001 | - | 0.009 ± 0.002 | 0.011 ± 0.000 | 0.025 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.166 ± 0.005 | - | 0.168 ± 0.001 | 0.309 ± 0.259 | 0.269 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.018 ± 0.000 | - | 0.021 ± 0.024 | 0.023 ± 0.003 | 0.033 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.002 | 0.055 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.018 ± 0.002 | - | 0.020 ± 0.002 | 0.019 ± 0.002 | 0.033 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.048 ± 0.001 | - | 0.050 ± 0.001 | 0.051 ± 0.004 | 0.065 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.009 ± 0.001 | - | 0.009 ± 0.000 | 0.014 ± 0.003 | 0.026 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.023 ± 0.001 | - | 0.024 ± 0.005 | 0.022 ± 0.006 | 0.038 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.067 ± 0.002 | - | 0.075 ± 0.009 | 0.066 ± 0.001 | 0.085 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 54.084 ± 1.032 | - | 53.195 ± 6.716 | 45.153 ± 6.150 | 21.860 ± 2.656 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.013 ± 0.003 | - | 0.013 ± 0.000 | 0.016 ± 0.001 | 0.073 ± 0.197 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 1.735 ± 0.124 | - | 1.475 ± 0.306 | 1.789 ± 1.052 | 0.974 ± 0.086 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.037 ± 0.007 | - | 0.045 ± 0.005 | 0.042 ± 0.009 | 0.065 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.126 ± 0.010 | - | 0.128 ± 0.003 | 0.122 ± 0.011 | 0.196 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.575 ± 0.051 | - | 0.592 ± 0.043 | 0.580 ± 0.034 | 2.548 ± 1.305 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 2.874 ± 0.056 | - | 2.963 ± 0.153 | 2.753 ± 0.094 | 4.426 ± 2.451 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 11.674 ± 0.716 | - | 13.753 ± 0.929 | 10.576 ± 0.162 | 15.907 ± 4.945 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.630 ± 0.110 | - | 1.594 ± 0.170 | 1.468 ± 0.024 | 2.009 ± 2.192 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.007 ± 0.001 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.125 ± 0.016 | - | 0.139 ± 0.042 | 0.121 ± 0.001 | 0.483 ± 0.166 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.001 | 0.010 ± 0.002 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.097 ± 0.003 | - | 0.100 ± 0.047 | 0.096 ± 0.001 | 0.138 ± 0.211 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.015 ± 0.001 | - | 0.013 ± 0.001 | 0.016 ± 0.000 | 0.025 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.001 | 0.008 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.010 ± 0.000 | - | 0.012 ± 0.003 | 0.014 ± 0.000 | 0.028 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.032 ± 0.006 | - | 0.031 ± 0.004 | 0.034 ± 0.000 | 0.052 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.058 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.014 ± 0.003 | - | 0.014 ± 0.000 | 0.017 ± 0.004 | 0.090 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.040 ± 0.003 | - | 0.043 ± 0.007 | 0.044 ± 0.001 | 0.066 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 12.484 ± 0.198 | - | 12.385 ± 0.448 | 11.753 ± 0.122 | 12.962 ± 0.715 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.002 | 0.011 ± 0.000 | 0.058 ± 0.126 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 0.990 ± 0.122 | - | 0.977 ± 0.133 | 0.930 ± 0.017 | 2.726 ± 1.173 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.024 ± 0.007 | - | 0.025 ± 0.001 | 0.027 ± 0.003 | 0.044 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.080 ± 0.002 | - | 0.093 ± 0.013 | 0.083 ± 0.001 | 0.150 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.554 ± 0.023 | - | 0.544 ± 0.014 | 0.776 ± 0.175 | 2.218 ± 0.270 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 2.610 ± 0.283 | - | 2.782 ± 0.120 | 4.660 ± 1.723 | 3.885 ± 1.705 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 11.852 ± 0.859 | - | 11.783 ± 0.353 | 14.357 ± 2.857 | 8.223 ± 3.104 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 1.397 ± 0.021 | - | 1.394 ± 0.033 | 1.914 ± 0.597 | 1.889 ± 0.269 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.022 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.122 ± 0.020 | - | 0.117 ± 0.002 | 0.137 ± 0.010 | 0.292 ± 0.074 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.005 ± 0.001 | - | 0.008 ± 0.002 | 0.009 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.268 ± 0.037 | - | 0.105 ± 0.028 | 0.111 ± 0.013 | 0.200 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.013 ± 0.001 | - | 0.014 ± 0.000 | 0.017 ± 0.001 | 0.035 ± 0.073 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.025 ± 0.002 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.012 ± 0.002 | - | 0.011 ± 0.000 | 0.015 ± 0.004 | 0.027 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.030 ± 0.001 | - | 0.031 ± 0.001 | 0.034 ± 0.001 | 0.098 ± 0.064 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.005 ± 0.001 | - | 0.006 ± 0.001 | 0.026 ± 0.002 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.013 ± 0.004 | - | 0.014 ± 0.000 | 0.019 ± 0.006 | 0.035 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.040 ± 0.000 | - | 0.043 ± 0.001 | 0.044 ± 0.001 | 0.068 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 11.113 ± 0.123 | - | 12.164 ± 0.511 | 16.020 ± 5.366 | 9.827 ± 2.940 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.010 ± 0.002 | 0.012 ± 0.001 | 0.030 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 0.887 ± 0.063 | - | 0.889 ± 0.008 | 1.686 ± 0.856 | 1.313 ± 0.357 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.023 ± 0.000 | - | 0.024 ± 0.000 | 0.036 ± 0.009 | 0.047 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.080 ± 0.001 | - | 0.089 ± 0.009 | 0.086 ± 0.012 | 0.244 ± 0.084 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.249 ± 0.004 | - | unsupported | 0.239 ± 0.051 | 1.921 ± 0.273 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.426 ± 0.424 | - | unsupported | 1.445 ± 0.070 | 3.072 ± 0.266 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 3.797 ± 0.018 | - | unsupported | 4.109 ± 0.179 | 6.230 ± 0.235 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.767 ± 0.016 | - | unsupported | 0.784 ± 0.076 | 0.929 ± 0.150 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | unsupported | 0.006 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.037 ± 0.000 | - | unsupported | 0.042 ± 0.005 | 0.059 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.004 ± 0.001 | - | unsupported | 0.006 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.030 ± 0.000 | - | unsupported | 0.035 ± 0.004 | 0.041 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.008 ± 0.001 | - | unsupported | 0.009 ± 0.000 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.012 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.025 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.009 ± 0.002 | - | unsupported | 0.011 ± 0.002 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.016 ± 0.000 | - | unsupported | 0.025 ± 0.004 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 6.047 ± 0.017 | - | unsupported | 6.634 ± 0.212 | 7.498 ± 0.460 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.001 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.286 ± 0.001 | - | unsupported | 0.315 ± 0.027 | 0.372 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.011 ± 0.001 | - | unsupported | 0.016 ± 0.004 | 0.030 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.030 ± 0.002 | - | unsupported | 0.035 ± 0.000 | 0.050 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.209 ± 0.002 | - | unsupported | 0.243 ± 0.052 | 1.351 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 1.421 ± 0.154 | - | unsupported | 1.561 ± 0.328 | 1.956 ± 0.266 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 5.033 ± 0.793 | - | unsupported | 4.876 ± 1.314 | 3.038 ± 0.105 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 0.957 ± 0.098 | - | unsupported | 0.845 ± 0.063 | 0.784 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.017 ± 0.019 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.040 ± 0.007 | - | unsupported | 0.053 ± 0.047 | 0.053 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.036 ± 0.008 | - | unsupported | 0.084 ± 0.072 | 0.041 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.007 ± 0.001 | - | unsupported | 0.010 ± 0.003 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | unsupported | 0.010 ± 0.002 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.013 ± 0.004 | - | unsupported | 0.016 ± 0.003 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.003 ± 0.001 | - | unsupported | 0.006 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.007 ± 0.000 | - | unsupported | 0.010 ± 0.003 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.017 ± 0.004 | - | unsupported | 0.021 ± 0.005 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 7.876 ± 0.662 | - | unsupported | 11.795 ± 6.878 | 4.235 ± 0.473 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.002 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.329 ± 0.098 | - | unsupported | 0.332 ± 0.016 | 0.282 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.012 ± 0.000 | - | unsupported | 0.016 ± 0.004 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.038 ± 0.022 | - | unsupported | 0.035 ± 0.001 | 0.049 ± 0.003 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.816 ± 0.064 | - | unsupported | 0.918 ± 0.135 | 1.157 ± 0.067 | 4.262 ± 3.435 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.056 ± 0.000 | - | unsupported | 0.071 ± 0.024 | 0.136 ± 0.004 | 0.130 ± 0.351 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.112 ± 0.000 | - | unsupported | 0.125 ± 0.015 | 0.209 ± 0.077 | 0.161 ± 0.052 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.259 ± 0.000 | - | unsupported | 0.310 ± 0.021 | 0.467 ± 0.185 | 0.783 ± 0.061 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.816 ± 0.025 | - | unsupported | 0.528 ± 0.074 | 0.702 ± 0.301 | 3.171 ± 0.729 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.056 ± 0.000 | - | unsupported | 0.145 ± 0.022 | 0.366 ± 0.076 | 0.108 ± 0.027 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.112 ± 0.010 | - | unsupported | 0.200 ± 0.089 | 0.237 ± 0.045 | 0.213 ± 0.023 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.260 ± 0.010 | - | unsupported | 0.240 ± 0.012 | 0.274 ± 0.117 | 0.769 ± 0.049 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.219 ± 0.006 | - | unsupported | 0.193 ± 0.007 | 0.465 ± 0.015 | 0.533 ± 0.045 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.060 ± 0.001 | - | unsupported | 0.047 ± 0.010 | 0.081 ± 0.002 | 0.074 ± 0.011 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.077 ± 0.008 | - | unsupported | 0.057 ± 0.006 | 0.099 ± 0.003 | 0.106 ± 0.010 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.098 ± 0.000 | - | unsupported | 0.084 ± 0.011 | 0.168 ± 0.011 | 0.226 ± 0.020 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.272 ± 0.115 | - | unsupported | 0.186 ± 0.001 | 0.487 ± 0.060 | 0.508 ± 0.420 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.062 ± 0.002 | - | unsupported | 0.043 ± 0.001 | 0.101 ± 0.019 | 0.068 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.076 ± 0.008 | - | unsupported | 0.051 ± 0.005 | 0.107 ± 0.013 | 0.080 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.100 ± 0.001 | - | unsupported | 0.085 ± 0.006 | 0.183 ± 0.024 | 0.234 ± 0.030 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.106 ± 0.007 | - | 0.109 ± 0.026 | 0.092 ± 0.018 | 0.182 ± 0.008 | 0.261 ± 0.323 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.046 ± 0.000 | - | 0.043 ± 0.008 | 0.025 ± 0.007 | 0.025 ± 0.001 | 0.042 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.044 ± 0.004 | - | 0.046 ± 0.000 | 0.041 ± 0.006 | 0.031 ± 0.002 | 0.063 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.056 ± 0.000 | - | 0.058 ± 0.000 | 0.046 ± 0.005 | 0.058 ± 0.002 | 0.109 ± 0.010 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.107 ± 0.011 | - | 0.112 ± 0.002 | 0.100 ± 0.026 | 0.325 ± 0.075 | 0.306 ± 0.034 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.172 ± 0.141 | - | 0.051 ± 0.018 | 0.039 ± 0.029 | 0.038 ± 0.026 | 0.044 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.057 ± 0.097 | - | 0.047 ± 0.007 | 0.026 ± 0.001 | 0.033 ± 0.002 | 0.051 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.057 ± 0.001 | - | 0.059 ± 0.003 | 0.037 ± 0.000 | 0.104 ± 0.089 | 0.133 ± 0.002 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 10.608 ± 0.651 | - | 11.383 ± 0.957 | 25.015 ± 1.605 | 16.848 ± 3.351 | 47.544 ± 8.020 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 9.598 ± 0.254 | - | 9.541 ± 0.865 | 20.617 ± 1.656 | 14.245 ± 2.720 | 349.813 ± 22.805 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 9.897 ± 0.700 | - | 8.288 ± 0.856 | 7.551 ± 0.750 | 10.429 ± 0.966 | 21.363 ± 6.591 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 8.752 ± 0.933 | - | 8.449 ± 0.704 | 9.277 ± 3.246 | 9.117 ± 1.753 | 242.098 ± 68.471 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 9.543 ± 5.422 | - | 6.834 ± 0.189 | 6.976 ± 0.147 | 8.345 ± 0.257 | 8.335 ± 0.611 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 7.074 ± 0.869 | - | 6.890 ± 0.104 | 6.858 ± 0.735 | 11.609 ± 5.181 | 306.589 ± 15.893 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 8.203 ± 0.856 | - | 8.587 ± 0.358 | 8.080 ± 0.460 | 7.899 ± 0.541 | 8.857 ± 0.853 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 8.092 ± 0.473 | - | 7.861 ± 0.402 | 10.065 ± 2.932 | 7.795 ± 0.313 | 438.195 ± 257.425 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 20.978 ± 1.363 | - | 21.628 ± 2.145 | 21.390 ± 0.506 | 30.925 ± 2.657 | 14.836 ± 1.117 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 21.464 ± 0.478 | - | 21.138 ± 0.724 | 27.064 ± 5.035 | 28.040 ± 1.058 | 656.858 ± 41.883 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.300 ± 0.011 | 0.390 ± 0.185 | unsupported | 0.326 ± 0.014 | 0.392 ± 0.078 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 4.005 ± 0.275 | 4.208 ± 0.191 | unsupported | 5.189 ± 1.561 | 4.729 ± 0.176 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.349 ± 0.034 | 0.342 ± 0.047 | unsupported | 0.353 ± 0.061 | 0.598 ± 0.105 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 3.865 ± 0.205 | 3.854 ± 0.123 | unsupported | 3.819 ± 0.318 | 4.640 ± 1.390 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 9.324 ± 1.598 | - | 9.296 ± 0.936 | 10.345 ± 0.915 | 9.877 ± 0.121 | 33.167 ± 5.087 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 12.276 ± 1.762 | - | 9.507 ± 0.296 | 11.857 ± 5.292 | 11.015 ± 4.335 | 394.610 ± 34.734 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 9.442 ± 0.389 | 8.918 ± 0.975 | 5.861 ± 0.279 | 26.003 ± 0.410 | 8.551 ± 0.539 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 9.626 ± 0.362 | 9.634 ± 0.597 | 5.312 ± 0.323 | 30.113 ± 3.481 | 139.448 ± 26.573 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 10.156 ± 1.557 | - | 8.963 ± 0.579 | 14.150 ± 2.230 | 13.491 ± 0.213 | 31.131 ± 8.220 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 8.144 ± 0.062 | - | 8.535 ± 0.664 | 12.993 ± 3.730 | 10.444 ± 0.573 | 246.517 ± 42.847 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 2.033 ± 0.705 | - | 1.062 ± 0.098 | 2.590 ± 1.169 | 6.473 ± 0.429 | 6.451 ± 0.699 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 1.004 ± 0.080 | - | 1.278 ± 0.309 | 2.360 ± 0.235 | 3.465 ± 0.230 | 6.692 ± 2.078 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 11.588 ± 0.758 | - | 10.891 ± 0.185 | 10.734 ± 0.120 | 12.897 ± 0.049 | 22.291 ± 4.119 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 12.015 ± 0.857 | - | 10.991 ± 0.911 | 12.087 ± 5.717 | 15.775 ± 2.661 | 969.884 ± 339.935 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 10.979 ± 0.157 | - | 11.249 ± 0.814 | 10.783 ± 0.320 | 12.937 ± 0.473 | 22.354 ± 1.592 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 10.998 ± 0.497 | - | 10.692 ± 1.109 | 11.689 ± 3.386 | 12.422 ± 0.630 | 970.832 ± 55.185 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.518 ± 0.301 | 1.818 ± 0.150 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 14.886 ± 0.448 | 13.604 ± 0.446 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.767 ± 0.225 | 1.747 ± 0.218 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 14.425 ± 0.662 | 14.645 ± 0.682 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.802 ± 0.261 | 1.979 ± 0.285 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 13.372 ± 1.923 | 12.211 ± 0.937 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.588 ± 0.042 | 1.657 ± 0.103 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 13.532 ± 0.751 | 15.398 ± 1.672 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 8.509 ± 0.912 | - | 8.526 ± 0.738 | 8.197 ± 2.006 | 10.557 ± 1.404 | 23.327 ± 4.076 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 8.903 ± 2.463 | - | 8.839 ± 0.320 | 10.933 ± 10.575 | 7.736 ± 0.293 | 248.036 ± 51.882 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 43.584 ± 9.761 | 42.422 ± 8.435 | 37.087 ± 1.130 | 35.769 ± 3.026 | 71.698 ± 7.870 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 33.819 ± 3.010 | 38.047 ± 3.067 | 40.078 ± 12.869 | 34.600 ± 0.754 | 1696.930 ± 137.083 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 15.997 ± 1.131 | - | 16.787 ± 4.081 | 13.217 ± 0.312 | 91.471 ± 11.526 | 37.432 ± 3.895 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 16.254 ± 1.141 | - | 16.055 ± 0.490 | 19.011 ± 3.959 | 54.742 ± 6.572 | 31.151 ± 11.867 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.174 ± 0.006 | 0.180 ± 0.008 | unsupported | 0.053 ± 0.001 | 0.035 ± 0.001 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.840 ± 0.071 | 2.802 ± 0.084 | unsupported | 0.856 ± 0.032 | 0.699 ± 0.552 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.220 ± 0.025 | 0.183 ± 0.014 | unsupported | 0.055 ± 0.027 | 0.085 ± 0.004 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 2.385 ± 1.084 | 2.248 ± 0.738 | unsupported | 0.969 ± 0.196 | 0.539 ± 0.084 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.080 ± 0.003 | 0.080 ± 0.003 | unsupported | 0.087 ± 0.002 | 0.241 ± 0.037 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.262 ± 0.051 | 1.279 ± 0.024 | unsupported | 1.820 ± 0.069 | 3.926 ± 0.847 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.127 ± 0.024 | 0.128 ± 0.040 | unsupported | 0.076 ± 0.032 | 0.295 ± 0.040 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 1.097 ± 0.434 | 1.414 ± 1.135 | unsupported | 1.624 ± 0.109 | 2.052 ± 0.377 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.269 ± 0.002 | 0.270 ± 0.003 | unsupported | 0.298 ± 0.004 | 0.110 ± 0.008 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 4.342 ± 0.056 | 4.363 ± 0.072 | unsupported | 5.643 ± 0.437 | 1.365 ± 0.744 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.191 ± 0.032 | 0.200 ± 0.023 | unsupported | 0.204 ± 0.032 | 0.096 ± 0.049 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 3.449 ± 0.425 | 3.206 ± 0.312 | unsupported | 3.297 ± 0.613 | 0.906 ± 0.258 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.048 ± 0.005 | 0.067 ± 0.010 | unsupported | 0.047 ± 0.002 | 0.100 ± 0.005 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.599 ± 0.027 | 0.624 ± 0.009 | unsupported | 0.721 ± 0.058 | 0.791 ± 0.099 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.071 ± 0.011 | 0.090 ± 0.005 | unsupported | 0.071 ± 0.009 | 0.156 ± 0.013 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.920 ± 0.101 | 0.916 ± 0.099 | unsupported | 0.394 ± 0.028 | 0.732 ± 0.378 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.096 ± 0.004 | 0.114 ± 0.011 | unsupported | 0.093 ± 0.002 | 0.138 ± 0.023 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.224 ± 0.025 | 1.267 ± 0.039 | unsupported | 1.884 ± 0.126 | 1.458 ± 0.062 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.121 ± 0.031 | 0.269 ± 0.148 | unsupported | 0.076 ± 0.012 | 0.136 ± 0.004 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 2.106 ± 0.144 | 1.201 ± 0.046 | unsupported | 0.753 ± 0.064 | 1.434 ± 0.102 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.128 ± 0.006 | 0.146 ± 0.010 | unsupported | unsupported | 0.156 ± 0.013 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 1.852 ± 0.023 | 1.901 ± 0.090 | unsupported | unsupported | 2.072 ± 0.164 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.177 ± 0.050 | 0.232 ± 0.068 | unsupported | unsupported | 0.262 ± 0.008 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 2.512 ± 0.637 | 2.814 ± 1.289 | unsupported | unsupported | 1.104 ± 0.271 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.014 ± 0.004 | 0.017 ± 0.003 | unsupported | 0.008 ± 0.000 | 0.069 ± 0.008 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.107 ± 0.007 | 0.112 ± 0.004 | unsupported | 0.068 ± 0.008 | 0.303 ± 0.038 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.013 ± 0.001 | 0.024 ± 0.006 | unsupported | 0.039 ± 0.004 | 0.063 ± 0.011 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.235 ± 0.029 | 0.252 ± 0.047 | unsupported | 0.106 ± 0.012 | 0.382 ± 0.110 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.028 ± 0.002 | 0.040 ± 0.002 | unsupported | 0.027 ± 0.001 | 0.112 ± 0.018 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.330 ± 0.022 | 0.351 ± 0.015 | unsupported | 0.467 ± 0.063 | 0.804 ± 0.107 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.034 ± 0.003 | 0.053 ± 0.001 | unsupported | 0.184 ± 0.034 | 0.164 ± 0.012 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.389 ± 0.070 | 0.488 ± 0.021 | unsupported | 0.370 ± 0.102 | 1.286 ± 0.213 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.050 ± 0.001 | 0.054 ± 0.003 | unsupported | 0.076 ± 0.000 | 0.097 ± 0.013 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.761 ± 0.005 | 0.761 ± 0.005 | unsupported | 1.261 ± 0.023 | 1.165 ± 0.346 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.081 ± 0.020 | 0.089 ± 0.007 | unsupported | 0.074 ± 0.013 | 0.188 ± 0.012 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.740 ± 0.193 | 0.626 ± 0.192 | unsupported | 1.275 ± 0.255 | 0.582 ± 0.568 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.052 ± 0.003 | 0.052 ± 0.003 | unsupported | 0.078 ± 0.003 | 0.098 ± 0.001 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.798 ± 0.015 | 0.796 ± 0.012 | unsupported | 1.289 ± 0.023 | 1.323 ± 0.353 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.078 ± 0.038 | 0.032 ± 0.003 | unsupported | 0.079 ± 0.026 | 0.200 ± 0.069 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.855 ± 0.267 | 1.000 ± 0.094 | unsupported | 1.241 ± 0.152 | 0.636 ± 0.106 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.092 ± 0.004 | 0.107 ± 0.015 | unsupported | 0.098 ± 0.006 | 0.137 ± 0.016 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.259 ± 0.076 | 1.282 ± 0.040 | unsupported | 2.038 ± 0.405 | 1.520 ± 0.088 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.125 ± 0.007 | 0.145 ± 0.008 | unsupported | 0.081 ± 0.009 | 0.135 ± 0.004 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 2.109 ± 0.166 | 2.308 ± 0.077 | unsupported | 1.245 ± 0.408 | 1.650 ± 0.417 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.226 ± 0.008 | 0.225 ± 0.013 | unsupported | 0.277 ± 0.005 | 0.313 ± 0.027 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 3.492 ± 0.026 | 3.524 ± 0.062 | unsupported | 4.779 ± 0.411 | 6.063 ± 0.916 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.138 ± 0.003 | 0.267 ± 0.075 | unsupported | 0.180 ± 0.129 | 0.366 ± 0.077 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 3.127 ± 0.337 | 2.909 ± 0.284 | unsupported | 4.088 ± 0.875 | 3.061 ± 0.857 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.558 ± 0.011 | 0.710 ± 0.063 | unsupported | 0.124 ± 0.005 | 0.171 ± 0.035 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 9.388 ± 0.206 | 10.439 ± 0.232 | unsupported | 2.599 ± 0.639 | 1.661 ± 0.121 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 1.234 ± 0.218 | 1.142 ± 0.188 | unsupported | 0.175 ± 0.110 | 0.188 ± 0.028 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 7.549 ± 0.594 | 10.687 ± 0.945 | unsupported | 1.394 ± 0.095 | 1.574 ± 0.560 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 30.359 ± 15.333 | - | unsupported | 18.186 ± 1.345 | - | 28.750 ± 12.935 | 22.958 ± 5.740 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 16.089 ± 2.222 | - | unsupported | 19.946 ± 3.903 | - | 19.938 ± 4.955 | 19.596 ± 1.656 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 11.495 ± 0.646 | - | unsupported | 11.802 ± 0.967 | - | 12.855 ± 2.580 | 24.173 ± 4.420 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 14.395 ± 1.508 | - | unsupported | 13.710 ± 0.479 | - | 15.805 ± 7.140 | 17.324 ± 2.089 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 10.949 ± 3.984 | - | unsupported | 11.703 ± 0.772 | - | 21.841 ± 1.687 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 13.279 ± 0.793 | - | unsupported | 13.604 ± 3.040 | - | 15.241 ± 5.424 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 23.725 ± 3.027 | - | unsupported | 17.911 ± 1.228 | - | 22.847 ± 7.012 | 21.593 ± 0.734 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 20.061 ± 1.431 | - | unsupported | 25.289 ± 1.757 | - | 25.692 ± 3.604 | 19.565 ± 1.233 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 10.891 ± 0.648 | - | unsupported | 10.505 ± 0.503 | - | 126.899 ± 3.942 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 10.542 ± 0.230 | - | unsupported | 13.179 ± 3.512 | - | 28.026 ± 24.058 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 11.865 ± 0.665 | - | unsupported | 12.087 ± 0.458 | - | 58.401 ± 9.315 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 11.945 ± 0.884 | - | unsupported | 15.259 ± 1.454 | - | 27.184 ± 4.761 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 24.201 ± 10.513 | - | unsupported | 18.045 ± 1.212 | - | 19.406 ± 2.267 | 25.064 ± 7.894 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 16.906 ± 1.003 | - | unsupported | 21.733 ± 4.233 | - | 20.849 ± 2.952 | 20.877 ± 1.342 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 12.562 ± 1.510 | - | unsupported | 11.315 ± 0.556 | - | 13.787 ± 1.711 | 16.112 ± 3.708 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 13.689 ± 1.227 | - | unsupported | 13.784 ± 1.289 | - | 15.395 ± 1.581 | 13.907 ± 2.935 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.221 ± 0.007 | unsupported | unsupported | 0.303 ± 0.063 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 2.966 ± 0.332 | unsupported | unsupported | 3.879 ± 0.692 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.211 ± 0.003 | unsupported | unsupported | 0.413 ± 0.090 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 3.007 ± 0.929 | unsupported | unsupported | 3.549 ± 0.896 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 28.464 ± 6.786 | - | unsupported | 19.376 ± 1.287 | - | 18.904 ± 2.090 | 20.921 ± 1.608 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 16.458 ± 1.162 | - | unsupported | 21.559 ± 2.514 | - | 16.955 ± 2.959 | 18.290 ± 3.083 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 7.712 ± 0.145 | - | 0.001 ± 0.000 | 6.090 ± 0.788 | 6.089 ± 0.881 | 20.455 ± 6.382 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 18.695 ± 4.607 | - | 0.004 ± 0.001 | 6.982 ± 0.999 | 5.979 ± 0.411 | 15.472 ± 1.195 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 8.595 ± 0.496 | - | 18.975 ± 2.521 | 11.430 ± 1.791 | 11.241 ± 0.690 | 12.438 ± 7.577 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 13.224 ± 1.979 | - | 29.551 ± 12.365 | 11.294 ± 0.955 | 8.038 ± 1.152 | 10.122 ± 0.776 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 8.484 ± 0.081 | - | 9.234 ± 1.180 | 10.436 ± 0.808 | 17.871 ± 0.879 | 13.698 ± 2.023 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 14.078 ± 1.619 | - | 13.501 ± 0.903 | 13.029 ± 3.256 | 11.209 ± 2.198 | 12.912 ± 2.971 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 4.677 ± 0.112 | - | 4.929 ± 1.984 | 9.379 ± 0.912 | 10.709 ± 1.163 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 8.002 ± 0.725 | - | 8.893 ± 0.708 | 11.917 ± 2.842 | 8.476 ± 1.494 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 9.936 ± 0.576 | - | 0.002 ± 0.000 | 12.686 ± 1.249 | 11.303 ± 1.712 | 16.159 ± 2.798 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 21.254 ± 4.507 | - | 0.001 ± 0.000 | 15.315 ± 0.755 | 10.796 ± 0.479 | 13.855 ± 4.864 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 38.928 ± 1.386 | - | 0.002 ± 0.001 | 53.250 ± 4.059 | 58.012 ± 4.563 | 217.862 ± 24.906 | 38.248 ± 3.368 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 33.724 ± 3.656 | - | 0.002 ± 0.000 | 53.162 ± 10.630 | 36.001 ± 8.506 | 210.972 ± 36.761 | 15.410 ± 1.086 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 6.841 ± 0.060 | - | 6.891 ± 1.099 | 7.692 ± 0.094 | 6.596 ± 0.437 | 10.953 ± 1.809 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 7.053 ± 2.931 | - | 7.375 ± 0.978 | 8.186 ± 2.994 | 6.398 ± 0.666 | 10.424 ± 1.197 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 6.846 ± 0.355 | - | 6.849 ± 0.060 | 7.407 ± 0.501 | 6.692 ± 1.559 | 11.007 ± 1.400 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 6.815 ± 0.721 | - | 7.355 ± 0.521 | 7.501 ± 0.581 | 6.039 ± 0.410 | 10.579 ± 0.991 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.002 ± 0.008 | - | unsupported | 0.003 ± 0.004 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.002 ± 0.002 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 4.984833 | 311552.06 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 3.864417 | 241526.06 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 4.145250 | 259078.12 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 4.343667 | 271479.19 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 6.977542 | 436096.38 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 9.499541 | 593721.31 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 10.841542 | 677596.38 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 11.437750 | 714859.38 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 17.351541 | 1084471.31 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 15.649084 | 978067.75 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 6.276292 | 392268.25 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 5.217958 | 326122.38 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 4.807000 | 300437.50 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 6.245708 | 390356.75 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 75.667875 | 4729242.19 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 83.025666 | 5189104.12 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 64.073792 | 4004612.00 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 67.329084 | 4208067.75 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 9.562834 | 597677.12 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 5.643583 | 352723.94 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 5.586875 | 349179.69 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 5.476333 | 342270.81 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 74.239416 | 4639963.50 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 61.103042 | 3818940.12 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 61.837667 | 3864854.19 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 61.658083 | 3853630.19 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 24.284625 | 1517789.06 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 29.080667 | 1817541.69 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 238.181333 | 14886333.31 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 217.661292 | 13603830.75 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 28.270417 | 1766901.06 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 27.946584 | 1746661.50 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 230.792958 | 14424559.88 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 234.315209 | 14644700.56 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 28.825208 | 1801575.50 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 31.666833 | 1979177.06 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 213.948333 | 13371770.81 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 195.368166 | 12210510.38 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 25.410542 | 1588158.88 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 26.506458 | 1656653.62 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 216.506542 | 13531658.88 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 246.370541 | 15398158.81 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.247708 | 35120.44 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.370625 | 52666.02 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 11.128333 | 173880.20 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 11.515375 | 179927.73 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.591375 | 698921.88 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 6.847083 | 855885.38 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 22.719417 | 2839927.12 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 22.413625 | 2801703.12 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.426666 | 84791.66 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.523833 | 55059.89 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 14.077083 | 219954.42 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 11.687167 | 182611.98 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.308583 | 538572.88 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.755417 | 969427.12 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 19.083167 | 2385395.88 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 17.987167 | 2248395.88 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 15.395083 | 240548.17 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.538084 | 86532.56 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.139083 | 80298.17 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.132834 | 80200.53 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 31.406875 | 3925859.38 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.557250 | 1819656.25 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.099084 | 1262385.50 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.232417 | 1279052.12 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 18.911209 | 295487.64 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.845625 | 75712.89 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.152958 | 127389.97 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 8.202708 | 128167.31 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 16.415791 | 2051973.87 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.994792 | 1624349.00 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.772333 | 1096541.62 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 11.312833 | 1414104.12 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.022750 | 109730.47 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 19.080292 | 298129.56 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 17.196959 | 268702.48 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 17.281541 | 270024.08 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.917791 | 1364723.88 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 45.146709 | 5643338.62 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 34.739959 | 4342494.88 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 34.905791 | 4363223.88 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.146291 | 96035.80 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 13.029750 | 203589.84 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 12.243959 | 191311.86 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 12.817125 | 200267.58 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.245125 | 905640.62 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 26.376417 | 3297052.12 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 27.588958 | 3448619.75 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 25.646416 | 3205802.00 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.371209 | 99550.14 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 2.983166 | 46611.97 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.084500 | 48195.31 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.280125 | 66876.95 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.324084 | 790510.50 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 5.767708 | 720963.50 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.793750 | 599218.75 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.991959 | 623994.88 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.991542 | 156117.84 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.532750 | 70824.22 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.562250 | 71285.16 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.767125 | 90111.33 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.857334 | 732166.75 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.150625 | 393828.12 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.360792 | 920099.00 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.327375 | 915921.88 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.804666 | 137572.91 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.928042 | 92625.66 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.142666 | 95979.16 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 7.297167 | 114018.23 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.662375 | 1457796.88 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 15.073958 | 1884244.75 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.792708 | 1224088.50 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.139375 | 1267421.88 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.695583 | 135868.48 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.842667 | 75666.67 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 7.755875 | 121185.55 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 17.232708 | 269261.06 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.469125 | 1433640.62 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.021584 | 752698.00 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 16.851625 | 2106453.12 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 9.608833 | 1201104.12 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.970667 | 155791.67 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.189042 | 127953.78 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.368500 | 146382.81 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 16.575416 | 2071927.00 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 14.819292 | 1852411.50 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 15.206917 | 1900864.62 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 16.798750 | 262480.47 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 11.318291 | 176848.30 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 14.843375 | 231927.73 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.835625 | 1104453.12 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 20.094083 | 2511760.38 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 22.512042 | 2814005.25 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.392541 | 68633.45 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.497333 | 7770.83 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.894334 | 13973.97 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.056625 | 16509.77 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 2.424833 | 303104.12 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.543250 | 67906.25 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.858541 | 107317.62 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.893458 | 111682.25 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.045708 | 63214.19 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.498458 | 39038.41 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.853125 | 13330.08 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.540333 | 24067.70 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.054333 | 381791.62 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.844667 | 105583.38 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.882375 | 235296.88 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.012334 | 251541.75 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.185291 | 112270.17 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 1.737042 | 27141.28 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.784042 | 27875.66 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.588667 | 40447.92 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.433584 | 804198.00 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 3.733208 | 466651.00 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.642458 | 330307.25 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 2.811333 | 351416.62 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.497833 | 164028.64 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 11.763875 | 183810.55 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.193541 | 34274.08 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3.371000 | 52671.88 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.288000 | 1286000.00 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.962209 | 370276.12 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.115375 | 389421.88 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.902708 | 487838.50 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.235000 | 97421.88 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.868083 | 76063.80 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.170000 | 49531.25 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.469500 | 54210.94 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 9.321125 | 1165140.62 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.090750 | 1261343.75 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.085291 | 760661.38 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.089917 | 761239.62 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.002208 | 187534.50 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.764875 | 74451.17 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.174833 | 80856.77 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.699541 | 89055.33 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.659459 | 582432.38 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 10.198875 | 1274859.38 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.918667 | 739833.38 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.011875 | 626484.38 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.273042 | 98016.28 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.022292 | 78473.31 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.311417 | 51740.89 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.329125 | 52017.58 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.580042 | 1322505.25 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.310500 | 1288812.50 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.380084 | 797510.50 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.368958 | 796119.75 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.802500 | 200039.06 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.063041 | 79110.02 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.018792 | 78418.62 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.045458 | 31960.28 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.086416 | 635802.00 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 9.928041 | 1241005.12 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.838459 | 854807.38 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 8.003625 | 1000453.12 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.746458 | 136663.41 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.286208 | 98222.00 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.872458 | 91757.16 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.876500 | 107445.31 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.160292 | 1520036.50 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 16.303291 | 2037911.38 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.071291 | 1258911.38 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.256666 | 1282083.25 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.663583 | 135368.48 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.173292 | 80832.69 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.023375 | 125365.23 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 9.298458 | 145288.41 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 13.198917 | 1649864.62 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 9.958500 | 1244812.50 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 16.870375 | 2108796.88 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 18.464000 | 2308000.00 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 20.009042 | 312641.28 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.735458 | 277116.53 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 14.476625 | 226197.27 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 14.388417 | 224819.02 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 48.507625 | 6063453.12 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 38.232458 | 4779057.25 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 27.938542 | 3492317.75 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 28.193917 | 3524239.62 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 23.392791 | 365512.36 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 11.489000 | 179515.62 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.854959 | 138358.73 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 17.111166 | 267361.97 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 24.490625 | 3061328.12 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 32.705709 | 4088213.62 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 25.019125 | 3127390.62 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 23.274292 | 2909286.50 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.929083 | 170766.92 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.941250 | 124082.03 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 35.721083 | 558141.92 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 45.431750 | 709871.09 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.290208 | 1661276.00 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 20.791875 | 2598984.38 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 75.101750 | 9387718.75 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 83.513459 | 10439182.38 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.014375 | 187724.61 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 11.179125 | 174673.83 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 78.977250 | 1234019.53 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 73.107500 | 1142304.69 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 12.588458 | 1573557.25 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 11.151959 | 1393994.88 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 60.393334 | 7549166.75 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 85.492959 | 10686619.88 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 4.848750 | 303046.88 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 3.531375 | 220710.94 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 62.069292 | 3879330.75 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 47.455458 | 2965966.12 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 6.613000 | 413312.50 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 3.382875 | 211429.69 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 56.788458 | 3549278.62 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 48.109458 | 3006841.12 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.015792 | 987.00 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.001625 | 101.56 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.013459 | 841.19 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.001292 | 80.75 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000084 | 5.25 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.049250 | 3078.12 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.039875 | 2492.19 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.013417 | 838.56 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.038042 | 2377.62 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000125 | 7.81 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.014834 | 927.12 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.003250 | 203.12 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000084 | 5.25 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.011334 | 708.38 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.002584 | 161.50 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000084 | 5.25 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.015208 | 950.50 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002708 | 169.25 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000084 | 5.25 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.011750 | 734.38 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002916 | 182.25 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `jax-cpu` is 35.4x faster than the slowest successful cell (`julia-base`, 73.895 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `pytorch-cpu` is 14.7x faster than the slowest successful cell (`julia-base`, 73.895 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-direct` is 18.8x faster than the slowest successful cell (`julia-base`, 73.895 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-trace` is 19.4x faster than the slowest successful cell (`julia-base`, 73.895 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `jax-cpu` is 22.3x faster than the slowest successful cell (`julia-base`, 453.315 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `pytorch-cpu` is 31.0x faster than the slowest successful cell (`julia-base`, 453.315 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-direct` is 36.3x faster than the slowest successful cell (`julia-base`, 453.315 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-trace` is 38.3x faster than the slowest successful cell (`julia-base`, 453.315 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 22.4x faster than the slowest successful cell (`julia-base`, 154.008 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 17.3x faster than the slowest successful cell (`julia-base`, 154.008 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-direct` is 19.3x faster than the slowest successful cell (`julia-base`, 154.008 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-trace` is 23.4x faster than the slowest successful cell (`julia-base`, 154.008 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 47.0x faster than the slowest successful cell (`julia-base`, 155.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 22.0x faster than the slowest successful cell (`julia-base`, 155.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-direct` is 24.4x faster than the slowest successful cell (`julia-base`, 155.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-trace` is 25.4x faster than the slowest successful cell (`julia-base`, 155.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `jax-cpu` is 91.9x faster than the slowest successful cell (`julia-base`, 625.930 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `pytorch-cpu` is 80.6x faster than the slowest successful cell (`julia-base`, 625.930 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-direct` is 92.5x faster than the slowest successful cell (`julia-base`, 625.930 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-trace` is 76.1x faster than the slowest successful cell (`julia-base`, 625.930 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 16.3x faster than the slowest successful cell (`pytorch-cpu`, 17.324 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 21.5x faster than the slowest successful cell (`pytorch-cpu`, 17.324 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 21.0x faster than the slowest successful cell (`pytorch-cpu`, 17.324 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 13.9x faster than the slowest successful cell (`pytorch-cpu`, 12.937 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 14.8x faster than the slowest successful cell (`pytorch-cpu`, 12.937 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 12.937 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 16.0x faster than the slowest successful cell (`pytorch-cpu`, 15.714 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 19.1x faster than the slowest successful cell (`pytorch-cpu`, 15.714 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 19.2x faster than the slowest successful cell (`pytorch-cpu`, 15.714 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 13.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 12.2x faster than the slowest successful cell (`pytorch-cpu`, 13.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 13.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 970.6x faster than the slowest successful cell (`julia-base`, 1.780 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 430.4x faster than the slowest successful cell (`tenferro-direct`, 1.578 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `pytorch-cpu` is 18.0x faster than the slowest successful cell (`jax-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 55.3x faster than the slowest successful cell (`jax-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-trace` is 30.4x faster than the slowest successful cell (`jax-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x8x8`): `tenferro-direct` is 24.6x faster than the slowest successful cell (`jax-cpu`, 0.024 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x4x4`): `tenferro-direct` is 31.2x faster than the slowest successful cell (`jax-cpu`, 0.034 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x4x4`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`jax-cpu`, 0.034 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `pytorch-cpu` is 22.5x faster than the slowest successful cell (`jax-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 74.5x faster than the slowest successful cell (`jax-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 35.8x faster than the slowest successful cell (`jax-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `pytorch-cpu` is 11.3x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 31.7x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-trace` is 17.4x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 27.3x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 13.6x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 23.0x faster than the slowest successful cell (`jax-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-trace` is 17.3x faster than the slowest successful cell (`jax-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 23.7x faster than the slowest successful cell (`pytorch-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 21.5x faster than the slowest successful cell (`pytorch-cpu`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `jax-cpu` is 14.8x faster than the slowest successful cell (`pytorch-cpu`, 0.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 67.7x faster than the slowest successful cell (`pytorch-cpu`, 0.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 33.5x faster than the slowest successful cell (`pytorch-cpu`, 0.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `jax-cpu` is 14.0x faster than the slowest successful cell (`pytorch-cpu`, 0.119 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 58.5x faster than the slowest successful cell (`pytorch-cpu`, 0.119 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 12.5x faster than the slowest successful cell (`pytorch-cpu`, 0.119 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `jax-cpu` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 0.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 23.5x faster than the slowest successful cell (`pytorch-cpu`, 0.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 27.0x faster than the slowest successful cell (`pytorch-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`pytorch-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 42.7x faster than the slowest successful cell (`pytorch-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 22.0x faster than the slowest successful cell (`pytorch-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 26.0x faster than the slowest successful cell (`pytorch-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 33.7x faster than the slowest successful cell (`pytorch-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 19.4x faster than the slowest successful cell (`pytorch-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `jax-cpu` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 39.5x faster than the slowest successful cell (`pytorch-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 20.3x faster than the slowest successful cell (`pytorch-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 29.6x faster than the slowest successful cell (`pytorch-cpu`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 19.4x faster than the slowest successful cell (`pytorch-cpu`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.271 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`1x32x32`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.271 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`1x8x8`): `tenferro-direct` is 12.7x faster than the slowest successful cell (`jax-cpu`, 0.040 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`3x4x4`): `tenferro-direct` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `pytorch-cpu` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 35.8x faster than the slowest successful cell (`jax-cpu`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-trace` is 26.3x faster than the slowest successful cell (`jax-cpu`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x2x2`): `tenferro-direct` is 13.9x faster than the slowest successful cell (`jax-cpu`, 0.034 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 23.2x faster than the slowest successful cell (`jax-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 17.2x faster than the slowest successful cell (`jax-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 15.9x faster than the slowest successful cell (`jax-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`jax-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 17.2x faster than the slowest successful cell (`jax-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x8x8`): `tenferro-trace` is 14.3x faster than the slowest successful cell (`jax-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `jax-cpu` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 0.142 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 42.1x faster than the slowest successful cell (`pytorch-cpu`, 0.142 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 37.5x faster than the slowest successful cell (`pytorch-cpu`, 0.142 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 10.0x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x32x32`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 21.5x faster than the slowest successful cell (`pytorch-cpu`, 0.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x64x24`): `tenferro-direct` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x64x24`): `tenferro-trace` is 13.8x faster than the slowest successful cell (`pytorch-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `jax-cpu` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 0.138 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 26.3x faster than the slowest successful cell (`pytorch-cpu`, 0.138 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 27.3x faster than the slowest successful cell (`pytorch-cpu`, 0.138 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `jax-cpu` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 26.9x faster than the slowest successful cell (`pytorch-cpu`, 0.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 21.5x faster than the slowest successful cell (`pytorch-cpu`, 0.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `jax-cpu` is 12.3x faster than the slowest successful cell (`pytorch-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 15.6x faster than the slowest successful cell (`pytorch-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 24.6x faster than the slowest successful cell (`pytorch-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 33.0x faster than the slowest successful cell (`jax-cpu`, 0.132 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 22.3x faster than the slowest successful cell (`jax-cpu`, 0.132 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `jax-cpu` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 22.9x faster than the slowest successful cell (`pytorch-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 20.5x faster than the slowest successful cell (`pytorch-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 17.7x faster than the slowest successful cell (`pytorch-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 21.2x faster than the slowest successful cell (`pytorch-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 19.5x faster than the slowest successful cell (`pytorch-cpu`, 0.108 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 19.8x faster than the slowest successful cell (`pytorch-cpu`, 0.108 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `jax-cpu` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 0.156 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 26.5x faster than the slowest successful cell (`pytorch-cpu`, 0.156 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 24.8x faster than the slowest successful cell (`pytorch-cpu`, 0.156 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 35.2x faster than the slowest successful cell (`pytorch-cpu`, 0.159 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 31.4x faster than the slowest successful cell (`pytorch-cpu`, 0.159 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 27.0x faster than the slowest successful cell (`pytorch-cpu`, 0.153 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 0.153 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.032 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 24.3x faster than the slowest successful cell (`pytorch-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 20.1x faster than the slowest successful cell (`pytorch-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.040 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x32x32,rhs=1`): `tenferro-direct` is 12.2x faster than the slowest successful cell (`jax-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x32x32,rhs=1`): `tenferro-direct` is 13.4x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x8x8,rhs=1`): `tenferro-direct` is 10.4x faster than the slowest successful cell (`pytorch-cpu`, 0.030 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (f64, threads=1, shape=`4x2x2`): `tenferro-direct` is 13.6x faster than the slowest successful cell (`jax-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `jax-cpu` is 24.6x faster than the slowest successful cell (`julia-base`, 349.813 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `pytorch-cpu` is 17.0x faster than the slowest successful cell (`julia-base`, 349.813 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-direct` is 36.4x faster than the slowest successful cell (`julia-base`, 349.813 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-trace` is 36.7x faster than the slowest successful cell (`julia-base`, 349.813 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 26.6x faster than the slowest successful cell (`julia-base`, 242.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 26.1x faster than the slowest successful cell (`julia-base`, 242.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 27.7x faster than the slowest successful cell (`julia-base`, 242.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 28.7x faster than the slowest successful cell (`julia-base`, 242.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `jax-cpu` is 26.4x faster than the slowest successful cell (`julia-base`, 306.589 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `pytorch-cpu` is 44.7x faster than the slowest successful cell (`julia-base`, 306.589 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-direct` is 43.3x faster than the slowest successful cell (`julia-base`, 306.589 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-trace` is 44.5x faster than the slowest successful cell (`julia-base`, 306.589 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `jax-cpu` is 56.2x faster than the slowest successful cell (`julia-base`, 438.195 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `pytorch-cpu` is 43.5x faster than the slowest successful cell (`julia-base`, 438.195 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-direct` is 54.2x faster than the slowest successful cell (`julia-base`, 438.195 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-trace` is 55.7x faster than the slowest successful cell (`julia-base`, 438.195 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `jax-cpu` is 23.4x faster than the slowest successful cell (`julia-base`, 656.858 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 24.3x faster than the slowest successful cell (`julia-base`, 656.858 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-direct` is 30.6x faster than the slowest successful cell (`julia-base`, 656.858 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-trace` is 31.1x faster than the slowest successful cell (`julia-base`, 656.858 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 35.8x faster than the slowest successful cell (`julia-base`, 394.610 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 33.3x faster than the slowest successful cell (`julia-base`, 394.610 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-direct` is 32.1x faster than the slowest successful cell (`julia-base`, 394.610 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-trace` is 41.5x faster than the slowest successful cell (`julia-base`, 394.610 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 26.3x faster than the slowest successful cell (`julia-base`, 139.448 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-eager` is 14.5x faster than the slowest successful cell (`julia-base`, 139.448 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-trace` is 14.5x faster than the slowest successful cell (`julia-base`, 139.448 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 23.6x faster than the slowest successful cell (`julia-base`, 246.517 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 19.0x faster than the slowest successful cell (`julia-base`, 246.517 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 30.3x faster than the slowest successful cell (`julia-base`, 246.517 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 28.9x faster than the slowest successful cell (`julia-base`, 246.517 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `jax-cpu` is 61.5x faster than the slowest successful cell (`julia-base`, 969.884 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 80.2x faster than the slowest successful cell (`julia-base`, 969.884 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 80.7x faster than the slowest successful cell (`julia-base`, 969.884 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 88.2x faster than the slowest successful cell (`julia-base`, 969.884 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `jax-cpu` is 78.2x faster than the slowest successful cell (`julia-base`, 970.832 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 83.1x faster than the slowest successful cell (`julia-base`, 970.832 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 88.3x faster than the slowest successful cell (`julia-base`, 970.832 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 90.8x faster than the slowest successful cell (`julia-base`, 970.832 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 32.1x faster than the slowest successful cell (`julia-base`, 248.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 22.7x faster than the slowest successful cell (`julia-base`, 248.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 27.9x faster than the slowest successful cell (`julia-base`, 248.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 28.1x faster than the slowest successful cell (`julia-base`, 248.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 49.0x faster than the slowest successful cell (`julia-base`, 1696.930 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 42.3x faster than the slowest successful cell (`julia-base`, 1696.930 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-eager` is 50.2x faster than the slowest successful cell (`julia-base`, 1696.930 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-trace` is 44.6x faster than the slowest successful cell (`julia-base`, 1696.930 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/output_reuse/dot_general_read_into` (f64, threads=1, shape=`1024x1024`): `pytorch-cpu` is 12.1x faster than the slowest successful cell (`julia-base`, 126.899 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/output_reuse/dot_general_read_into` (f64, threads=1, shape=`1024x1024`): `tenferro-direct` is 11.7x faster than the slowest successful cell (`julia-base`, 126.899 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 14029.7x faster than the slowest successful cell (`julia-base`, 20.455 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 4985.3x faster than the slowest successful cell (`tenferro-direct`, 18.695 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 10772.4x faster than the slowest successful cell (`julia-base`, 16.159 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 14999.2x faster than the slowest successful cell (`tenferro-direct`, 21.254 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 124492.5x faster than the slowest successful cell (`julia-base`, 217.862 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `strided-jl` is 13.7x faster than the slowest successful cell (`julia-base`, 210.972 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 129828.7x faster than the slowest successful cell (`julia-base`, 210.972 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 10.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 615.6x faster than the slowest successful cell (`pytorch-cpu`, 0.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 475.6x faster than the slowest successful cell (`tenferro-direct`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 115.9x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 141.6x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 190.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 146.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 9143.4 GB/s at 0.002 ms. The conservative physical lower bound is 0.168 ms, so the cell is 91.4x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 4573.0 GB/s at 0.004 ms. The conservative physical lower bound is 0.042 ms, so the cell is 11.4x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 184112.1 GB/s at 0.001 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1841.1x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 71582.8 GB/s at 0.004 ms. The conservative physical lower bound is 0.671 ms, so the cell is 179.0x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 178957.0 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1789.6x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 189439.3 GB/s at 0.001 ms. The conservative physical lower bound is 0.671 ms, so the cell is 473.6x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 76695.8 GB/s at 0.002 ms. The conservative physical lower bound is 1.342 ms, so the cell is 767.0x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 82595.5 GB/s at 0.002 ms. The conservative physical lower bound is 0.336 ms, so the cell is 206.5x faster than the bound.

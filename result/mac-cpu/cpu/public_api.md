# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20261114_182120/run.yaml`
- Timestamp: `2026-11-14T18:21:20.631460+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20261114_182120`.

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

- CSV: `data/results/mac-cpu/cpu/public_api/20261114_182120/cpu_public_api_t1_20261114_182120.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20261114_182120/cpu_public_api_t4_20261114_182120.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 4.150 ± 1.684 | - | 4.384 ± 2.347 | 4.262 ± 0.134 | 2.869 ± 2.571 | 8.514 ± 6.532 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 4.241 ± 0.362 | - | 4.135 ± 0.142 | 4.181 ± 0.172 | 1.571 ± 0.410 | 78.125 ± 26.254 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 9.618 ± 1.204 | - | 11.239 ± 0.334 | 12.198 ± 0.642 | 12.324 ± 1.679 | 17.818 ± 9.007 | 16.252 ± 10.544 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 13.005 ± 1.505 | - | 13.107 ± 1.001 | 13.760 ± 1.104 | 12.000 ± 1.102 | 18.741 ± 7.490 | 14.970 ± 1.840 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 51.677 ± 5.094 | - | 56.722 ± 9.151 | 22.823 ± 1.698 | 40.327 ± 3.647 | 55.466 ± 14.411 | 57.401 ± 2.299 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 22.132 ± 2.208 | - | 32.731 ± 2.398 | 13.272 ± 0.948 | 35.753 ± 5.927 | 55.271 ± 3.606 | 26.653 ± 10.043 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 11.049 ± 0.548 | - | 13.271 ± 3.110 | 15.034 ± 2.761 | 49.683 ± 6.768 | 66.130 ± 9.723 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 13.677 ± 2.412 | - | 15.598 ± 2.841 | 12.200 ± 2.041 | 19.554 ± 5.621 | 47.503 ± 3.785 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 14.048 ± 3.693 | - | unsupported | 12.710 ± 1.372 | 48.238 ± 1.225 | 62.497 ± 8.236 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 14.612 ± 2.300 | - | unsupported | 11.361 ± 0.856 | 20.186 ± 4.105 | 49.035 ± 52.191 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 11.274 ± 0.421 | - | 10.983 ± 1.424 | 12.947 ± 0.384 | 23.339 ± 1.709 | 10.198 ± 3.322 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 12.471 ± 0.396 | - | 11.522 ± 0.598 | 12.169 ± 1.506 | 30.541 ± 7.786 | 480.483 ± 280.645 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 37.671 ± 3.657 | - | 41.663 ± 4.551 | 40.476 ± 7.068 | 48.084 ± 1.657 | 57.630 ± 13.376 | 54.341 ± 3.416 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 25.807 ± 2.792 | - | 27.573 ± 7.132 | 20.293 ± 2.608 | 40.507 ± 29.128 | 48.528 ± 19.975 | 19.389 ± 2.519 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 49.049 ± 5.712 | - | 52.913 ± 3.957 | 64.953 ± 2.102 | 100.531 ± 4.339 | 113.611 ± 21.078 | 118.019 ± 44.098 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 28.093 ± 6.841 | - | 33.702 ± 4.256 | 26.638 ± 2.178 | 65.046 ± 14.414 | 79.911 ± 3.039 | 43.967 ± 14.097 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 10.133 ± 3.547 | - | 10.381 ± 2.132 | 12.451 ± 1.278 | 14.507 ± 1.395 | 10.468 ± 0.349 | 10.607 ± 0.863 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 11.435 ± 0.215 | - | 10.768 ± 1.359 | 11.882 ± 1.964 | 12.656 ± 1.995 | 12.274 ± 2.911 | 9.361 ± 3.492 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.504 ± 0.912 | - | 8.403 ± 1.197 | 6.672 ± 0.243 | 5.696 ± 0.510 | 9.603 ± 0.124 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 1.096 ± 0.084 | - | 4.949 ± 0.777 | 4.281 ± 0.937 | 3.188 ± 0.164 | 11.265 ± 3.788 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 8.889 ± 1.400 | - | 8.504 ± 0.481 | 8.145 ± 0.119 | 8.411 ± 2.166 | 11.655 ± 2.844 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 10.196 ± 3.331 | - | 8.996 ± 0.495 | 7.729 ± 0.416 | 13.052 ± 7.075 | 152.814 ± 20.784 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 5.726 ± 2.284 | - | 5.669 ± 0.700 | 5.505 ± 0.546 | 2.911 ± 0.190 | 6.779 ± 0.469 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 6.624 ± 0.555 | - | 6.308 ± 0.347 | 5.989 ± 0.704 | 2.676 ± 0.597 | 140.340 ± 32.684 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 7.731 ± 0.629 | - | 8.803 ± 4.158 | 8.267 ± 0.497 | 8.466 ± 1.697 | 7.874 ± 0.604 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 8.158 ± 0.360 | - | 8.151 ± 0.777 | 7.926 ± 0.506 | 8.196 ± 0.514 | 639.758 ± 78.958 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 12.090 ± 1.293 | - | 11.993 ± 1.380 | 13.594 ± 1.032 | 62.544 ± 11.987 | 59.132 ± 2.244 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 13.378 ± 0.883 | - | 12.149 ± 0.488 | 10.751 ± 0.773 | 19.423 ± 2.473 | 81.865 ± 29.476 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 11.920 ± 2.748 | - | unsupported | 11.301 ± 1.051 | 62.208 ± 2.676 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 12.125 ± 0.534 | - | unsupported | 13.446 ± 4.791 | 40.717 ± 4.512 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 10.240 ± 0.136 | - | 10.105 ± 0.120 | 17.830 ± 4.740 | 19.098 ± 6.984 | 23.235 ± 7.221 | 15.335 ± 1.583 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 14.249 ± 1.627 | - | 14.522 ± 2.167 | 12.456 ± 1.520 | 13.589 ± 1.834 | 16.364 ± 2.973 | 13.955 ± 8.969 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 22.077 ± 1.439 | - | 21.826 ± 4.579 | 18.726 ± 6.114 | 35.502 ± 6.111 | 25.751 ± 32.702 | 24.283 ± 6.772 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 17.221 ± 1.103 | - | 18.925 ± 1.542 | 19.384 ± 1.424 | 26.083 ± 2.459 | 96.000 ± 237.273 | 26.712 ± 7.882 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 29.515 ± 4.188 | - | 40.421 ± 3.825 | 45.867 ± 9.713 | 31.699 ± 6.613 | 59.991 ± 10.625 | 62.029 ± 7.045 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 20.196 ± 3.706 | - | 23.016 ± 5.183 | 25.496 ± 4.047 | 22.356 ± 4.174 | 57.914 ± 9.320 | 28.519 ± 8.650 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 6.762 ± 0.246 | - | 9.419 ± 2.622 | 6.669 ± 0.232 | 10.999 ± 1.097 | 9.431 ± 5.273 | 11.978 ± 7.590 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 7.620 ± 1.082 | - | 7.443 ± 0.997 | 7.486 ± 1.670 | 11.064 ± 2.325 | 9.822 ± 6.593 | 10.544 ± 5.222 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 13.198 ± 0.773 | - | 13.278 ± 0.904 | 23.912 ± 5.081 | 23.643 ± 3.974 | 20.707 ± 6.277 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 17.343 ± 2.364 | - | 16.667 ± 2.494 | 20.054 ± 17.430 | 17.009 ± 3.013 | 17.727 ± 3.055 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 33.159 ± 4.970 | - | 39.454 ± 6.993 | 33.158 ± 1.061 | 39.931 ± 10.042 | 42.458 ± 1.580 | 41.328 ± 3.171 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 19.108 ± 2.920 | - | 19.257 ± 2.281 | 21.299 ± 7.661 | 22.021 ± 2.337 | 43.286 ± 6.581 | 28.918 ± 22.708 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 19.664 ± 3.582 | - | 30.580 ± 18.338 | 22.340 ± 5.216 | 30.341 ± 8.406 | 21.614 ± 4.453 | 31.673 ± 17.340 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 20.135 ± 2.045 | - | 21.732 ± 2.372 | 22.212 ± 6.276 | 26.204 ± 2.687 | 23.641 ± 5.458 | 28.605 ± 23.559 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 19.475 ± 4.355 | - | 19.171 ± 2.396 | 32.351 ± 9.060 | 16.841 ± 1.782 | 34.421 ± 12.698 | 35.536 ± 11.390 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 11.573 ± 5.556 | - | 10.401 ± 2.687 | 20.844 ± 6.758 | 14.044 ± 6.653 | 63.925 ± 32.148 | 25.110 ± 8.216 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 26.783 ± 1.392 | - | 26.273 ± 0.421 | 30.846 ± 0.767 | 19.364 ± 2.871 | 25.498 ± 4.913 | 27.525 ± 3.046 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 13.538 ± 1.468 | - | 14.304 ± 2.607 | 17.228 ± 4.381 | 14.248 ± 0.108 | 27.464 ± 1.475 | 10.801 ± 2.149 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 37.309 ± 3.663 | - | 36.632 ± 5.455 | 41.407 ± 1.227 | 29.107 ± 2.284 | 42.375 ± 6.179 | 45.789 ± 8.479 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 18.774 ± 3.465 | - | 16.764 ± 2.232 | 29.713 ± 2.433 | 20.872 ± 5.289 | 58.579 ± 15.911 | 37.561 ± 22.345 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 15.623 ± 3.390 | - | 19.419 ± 1.067 | 24.490 ± 1.389 | 22.759 ± 7.477 | 26.141 ± 1.780 | 28.942 ± 6.571 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 9.626 ± 1.694 | - | 11.823 ± 2.132 | 13.641 ± 3.918 | 15.544 ± 1.029 | 22.677 ± 1.172 | 12.773 ± 3.004 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.261 ± 0.027 | 0.271 ± 0.062 | unsupported | unsupported | 0.247 ± 0.032 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.283 ± 0.024 | 0.331 ± 0.059 | unsupported | unsupported | 0.452 ± 0.014 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 16.492 ± 1.676 | - | 17.917 ± 1.620 | 27.375 ± 5.331 | 34.468 ± 5.372 | 22.513 ± 4.130 | 31.729 ± 19.164 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 22.158 ± 2.303 | - | 21.085 ± 2.874 | 23.579 ± 4.424 | 30.754 ± 5.620 | 17.808 ± 1.099 | 23.113 ± 3.128 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 26.072 ± 7.391 | - | 18.153 ± 0.568 | 26.937 ± 4.937 | 34.388 ± 8.829 | 18.577 ± 1.197 | 34.577 ± 63.799 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 21.842 ± 2.464 | - | 22.799 ± 2.519 | 27.984 ± 4.889 | 30.525 ± 8.602 | 18.433 ± 2.882 | 22.058 ± 7.061 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 17.974 ± 3.287 | - | 20.252 ± 7.420 | 22.367 ± 5.557 | 33.172 ± 4.173 | 22.905 ± 6.283 | 25.622 ± 17.036 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 20.461 ± 1.534 | - | 21.619 ± 1.825 | 21.534 ± 3.569 | 24.868 ± 3.162 | 26.846 ± 7.104 | 26.313 ± 5.227 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 11.035 ± 2.167 | - | 10.348 ± 1.066 | 17.187 ± 2.642 | 16.310 ± 7.725 | 15.654 ± 4.946 | 38.397 ± 75.905 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 13.120 ± 1.124 | - | 13.915 ± 0.937 | 14.103 ± 2.884 | 13.473 ± 2.961 | 17.353 ± 4.203 | 19.346 ± 7.673 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 48.174 ± 6.812 | - | 37.735 ± 4.039 | 76.117 ± 6.188 | 62.286 ± 5.619 | 132.458 ± 17.525 | 81.357 ± 10.907 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 22.335 ± 6.012 | - | 20.740 ± 3.342 | 37.322 ± 4.676 | 26.442 ± 4.428 | 105.277 ± 32.908 | 39.677 ± 9.700 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 7.610 ± 0.298 | - | 7.090 ± 0.536 | 12.920 ± 0.551 | 25.408 ± 0.349 | 7.881 ± 2.316 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 7.044 ± 0.582 | - | 7.505 ± 1.212 | 6.708 ± 1.681 | 18.158 ± 3.730 | 6.521 ± 2.117 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 0.842 ± 0.009 | - | 0.954 ± 0.140 | 20.254 ± 2.984 | 3.147 ± 1.819 | 0.895 ± 0.348 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.888 ± 0.668 | - | 0.920 ± 0.067 | 13.748 ± 1.268 | 1.858 ± 0.728 | 0.833 ± 0.081 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 0.863 ± 0.086 | - | 0.963 ± 0.097 | 1.742 ± 0.031 | 0.914 ± 0.012 | 0.983 ± 0.106 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 2.229 ± 0.233 | - | 2.151 ± 0.275 | 1.318 ± 0.257 | 1.329 ± 0.562 | 1.085 ± 0.130 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 6.966 ± 0.350 | - | 6.685 ± 0.507 | 13.066 ± 0.939 | 24.357 ± 1.141 | 9.373 ± 2.157 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 8.776 ± 3.164 | - | 7.700 ± 1.750 | 8.234 ± 1.329 | 16.180 ± 2.729 | 8.390 ± 3.662 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.014 ± 0.155 | - | 0.940 ± 0.018 | 17.271 ± 1.842 | 2.092 ± 0.076 | 0.898 ± 0.040 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.921 ± 0.209 | - | 1.672 ± 0.878 | 12.996 ± 0.304 | 2.280 ± 0.626 | 1.124 ± 0.301 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 4.025 ± 0.300 | - | 3.529 ± 0.133 | 6.840 ± 0.124 | 9.790 ± 6.158 | 3.611 ± 0.336 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 6.708 ± 0.720 | - | 9.044 ± 2.469 | 3.652 ± 0.099 | 4.213 ± 0.877 | 3.405 ± 0.161 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 6.484 ± 0.480 | - | 6.117 ± 0.838 | 8.288 ± 1.919 | 46.398 ± 1.943 | 8.686 ± 0.915 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 7.536 ± 0.534 | - | 6.805 ± 0.286 | 6.578 ± 1.130 | 26.719 ± 2.326 | 7.594 ± 1.005 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.854 ± 0.115 | - | 0.842 ± 0.044 | 4.321 ± 0.353 | 2.468 ± 0.101 | 0.952 ± 0.104 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 1.226 ± 0.581 | - | 1.038 ± 0.380 | 3.377 ± 1.350 | 2.258 ± 1.287 | 0.954 ± 0.031 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 0.926 ± 0.145 | - | 0.880 ± 0.128 | 0.919 ± 0.087 | 1.112 ± 0.049 | 0.950 ± 0.070 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 2.434 ± 0.360 | - | 2.230 ± 0.815 | 0.958 ± 0.080 | 1.240 ± 0.086 | 0.885 ± 0.047 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 6.222 ± 0.421 | - | 6.955 ± 0.742 | 7.986 ± 1.403 | 35.632 ± 0.518 | 9.702 ± 1.044 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 8.799 ± 0.858 | - | 9.559 ± 0.380 | 6.419 ± 1.370 | 19.736 ± 1.272 | 8.249 ± 1.466 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.702 ± 0.045 | - | 0.813 ± 0.127 | 4.346 ± 0.193 | 2.281 ± 0.148 | 0.839 ± 0.056 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 1.068 ± 0.368 | - | 0.871 ± 0.421 | 2.860 ± 0.396 | 2.057 ± 0.555 | 0.829 ± 0.046 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 0.954 ± 0.157 | - | 0.890 ± 0.090 | 0.762 ± 0.096 | 0.988 ± 0.102 | 0.885 ± 0.071 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 3.045 ± 1.925 | - | 3.899 ± 0.866 | 0.887 ± 0.064 | 1.357 ± 0.300 | 0.871 ± 0.102 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.916 ± 0.105 | 0.748 ± 0.062 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 1.169 ± 0.235 | 1.148 ± 0.522 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 31.482 ± 0.855 | - | 29.914 ± 1.155 | 52.214 ± 10.018 | 38.215 ± 4.052 | 28.991 ± 7.745 | 48.859 ± 16.141 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 17.098 ± 9.139 | - | 15.837 ± 4.391 | 24.143 ± 5.440 | 29.298 ± 2.522 | 95.703 ± 12.931 | 20.285 ± 4.414 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 21.841 ± 1.324 | - | 22.566 ± 4.666 | 22.091 ± 7.646 | 37.689 ± 3.824 | 49.663 ± 26.234 | 50.827 ± 25.628 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 13.938 ± 1.478 | - | 13.344 ± 1.434 | 15.069 ± 1.824 | 21.782 ± 2.572 | 46.179 ± 6.391 | 19.585 ± 4.013 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 26.933 ± 4.882 | - | 25.559 ± 1.826 | 29.119 ± 7.353 | 33.786 ± 4.856 | 26.668 ± 3.131 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 29.380 ± 2.529 | - | 34.864 ± 2.399 | 22.595 ± 1.542 | 30.216 ± 5.802 | 39.032 ± 11.483 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 12.518 ± 3.057 | - | 10.468 ± 0.573 | 15.283 ± 5.207 | 23.441 ± 7.686 | 14.613 ± 4.132 | 72.944 ± 67.739 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 14.335 ± 2.330 | - | 13.692 ± 1.259 | 15.422 ± 2.819 | 14.651 ± 2.476 | 17.188 ± 9.534 | 18.591 ± 18.711 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 28.050 ± 2.830 | - | 29.755 ± 4.824 | 28.984 ± 0.774 | 28.582 ± 2.353 | 40.573 ± 10.143 | 35.506 ± 2.748 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 14.521 ± 5.584 | - | 13.838 ± 2.921 | 21.918 ± 8.476 | 18.688 ± 4.292 | 41.935 ± 13.912 | 17.519 ± 8.540 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 16.367 ± 3.577 | - | 14.382 ± 0.342 | 13.708 ± 4.022 | 20.528 ± 3.490 | 201.405 ± 263.391 | 39.659 ± 12.697 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 14.198 ± 5.268 | - | 14.109 ± 1.458 | 14.492 ± 1.268 | 14.096 ± 1.736 | 44.618 ± 32.235 | 14.882 ± 3.609 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 27.136 ± 4.122 | - | 21.631 ± 3.746 | 21.164 ± 4.767 | 32.028 ± 6.691 | 24.565 ± 2.302 | 25.778 ± 74.066 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 20.754 ± 3.540 | - | 19.786 ± 2.491 | 20.912 ± 1.986 | 27.720 ± 6.152 | 22.780 ± 5.151 | 37.859 ± 16.822 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 97.852 ± 25.052 | - | 85.340 ± 19.153 | 67.097 ± 4.384 | 13.612 ± 1.092 | 52.988 ± 11.637 | 53.094 ± 2.574 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 36.950 ± 1.642 | - | 33.716 ± 4.424 | 41.356 ± 8.104 | 12.047 ± 0.314 | 49.621 ± 13.715 | 24.710 ± 5.394 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 25.463 ± 1.400 | 25.088 ± 1.322 | unsupported | 25.613 ± 3.553 | 32.749 ± 6.125 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 31.620 ± 3.509 | 28.824 ± 1.965 | unsupported | 24.194 ± 3.055 | 28.441 ± 4.950 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.809 ± 0.184 | - | 0.951 ± 0.196 | 0.739 ± 0.055 | 0.932 ± 0.474 | 0.975 ± 0.252 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.842 ± 0.182 | - | 0.945 ± 0.224 | 1.495 ± 2.003 | 1.491 ± 0.878 | 0.910 ± 0.168 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.659 ± 0.062 | - | 0.651 ± 0.128 | 0.763 ± 0.020 | 0.972 ± 0.095 | 0.816 ± 0.066 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.668 ± 0.169 | - | 1.058 ± 2.092 | 0.761 ± 0.063 | 1.091 ± 0.041 | 0.870 ± 0.204 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.078 ± 0.050 | - | unsupported | 1.250 ± 0.045 | 1.133 ± 0.063 | 1.341 ± 3.689 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 1.528 ± 0.536 | - | unsupported | 1.428 ± 0.630 | 1.441 ± 0.233 | 1.088 ± 2.728 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.340 ± 0.128 | - | 0.484 ± 0.048 | 0.333 ± 0.032 | 0.335 ± 0.038 | 1.200 ± 0.903 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.564 ± 0.277 | - | 0.727 ± 0.466 | 0.276 ± 0.010 | 0.442 ± 0.013 | 1.362 ± 4.726 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.618 ± 0.059 | - | 0.831 ± 0.065 | 1.161 ± 0.035 | 0.912 ± 0.096 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 1.213 ± 0.616 | - | 1.081 ± 0.608 | 1.526 ± 0.177 | 1.336 ± 1.381 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.993 ± 0.197 | - | 0.986 ± 0.238 | 1.046 ± 0.268 | 0.825 ± 0.030 | 1.182 ± 10.783 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.979 ± 0.385 | - | 1.827 ± 0.170 | 1.283 ± 0.207 | 1.294 ± 0.656 | 1.533 ± 5.344 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.703 ± 0.262 | - | 0.910 ± 0.204 | 0.455 ± 0.076 | 0.420 ± 0.012 | 0.355 ± 4.687 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.912 ± 0.130 | - | 1.063 ± 1.174 | 0.712 ± 0.117 | 0.566 ± 0.041 | 1.227 ± 0.589 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.250 ± 0.126 | - | 0.001 ± 0.000 | 1.298 ± 0.103 | 1.125 ± 0.216 | 1.614 ± 0.315 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 2.142 ± 0.578 | - | 0.001 ± 0.000 | 1.377 ± 0.130 | 1.438 ± 0.079 | 1.544 ± 0.183 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.054 ± 0.000 | - | 0.056 ± 0.000 | 0.133 ± 0.078 | 0.124 ± 0.077 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.091 ± 0.006 | - | 0.093 ± 0.000 | 0.345 ± 0.013 | 0.118 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.198 ± 0.000 | - | 0.201 ± 0.001 | 0.304 ± 0.057 | 0.216 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.026 ± 0.001 | - | 0.028 ± 0.002 | 0.038 ± 0.004 | 0.108 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.026 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.002 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.011 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.011 ± 0.001 | 0.013 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.012 ± 0.001 | 0.038 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.173 ± 0.001 | - | 0.222 ± 0.058 | 0.245 ± 0.001 | 0.260 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.011 ± 0.046 | - | 0.013 ± 0.000 | 0.017 ± 0.002 | 0.176 ± 0.171 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.014 ± 0.001 | 0.065 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.055 ± 0.000 | - | 0.056 ± 0.000 | 0.193 ± 0.013 | 0.138 ± 0.096 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.092 ± 0.000 | - | 0.094 ± 0.001 | 0.254 ± 0.121 | 0.181 ± 0.091 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.224 ± 0.010 | - | 0.230 ± 0.020 | 0.331 ± 0.051 | 0.335 ± 0.165 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.026 ± 0.001 | - | 0.029 ± 0.003 | 0.152 ± 0.026 | 0.091 ± 0.061 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.081 ± 0.035 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.003 ± 0.000 | - | 0.011 ± 0.002 | 0.075 ± 0.042 | 0.039 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.073 ± 0.058 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.054 ± 0.034 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.004 ± 0.000 | 0.071 ± 0.040 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.034 ± 0.049 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.080 ± 0.051 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.006 | 0.082 ± 0.083 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.091 ± 0.031 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.055 ± 0.059 | 0.017 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.180 ± 0.011 | - | 0.183 ± 0.003 | 0.367 ± 0.014 | 0.404 ± 0.360 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.087 ± 0.059 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.016 ± 0.002 | - | 0.028 ± 0.008 | 0.102 ± 0.024 | 0.091 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.110 ± 0.085 | 0.010 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.078 ± 0.032 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.384 ± 0.124 | - | 0.312 ± 0.009 | 1.242 ± 0.086 | 1.031 ± 0.082 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.640 ± 0.025 | - | 1.647 ± 0.013 | 1.745 ± 0.183 | 3.771 ± 1.999 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 6.901 ± 0.489 | - | 6.074 ± 0.026 | 11.365 ± 2.925 | 7.149 ± 0.058 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 4.383 ± 0.037 | - | 4.193 ± 0.216 | 4.008 ± 0.634 | 4.534 ± 0.246 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.038 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.100 ± 0.015 | - | 0.093 ± 0.009 | 0.118 ± 0.158 | 0.175 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.019 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.031 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.029 ± 0.000 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.021 ± 0.001 | - | 0.023 ± 0.000 | 0.064 ± 0.000 | 0.113 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.034 ± 0.000 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.027 ± 0.000 | - | 0.029 ± 0.000 | 0.080 ± 0.000 | 0.072 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 42.056 ± 1.616 | - | 37.852 ± 3.608 | 28.948 ± 1.613 | 34.909 ± 4.781 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.706 ± 0.021 | - | 0.723 ± 0.099 | 0.885 ± 0.302 | 0.955 ± 0.103 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.015 ± 0.000 | - | 0.016 ± 0.000 | 0.051 ± 0.001 | 0.033 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.053 ± 0.000 | - | 0.055 ± 0.001 | 0.145 ± 0.001 | 0.101 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.351 ± 0.043 | - | 0.341 ± 0.033 | 0.449 ± 0.031 | 0.858 ± 0.110 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 1.653 ± 0.068 | - | 1.668 ± 0.046 | 1.906 ± 0.519 | 1.588 ± 0.119 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 7.719 ± 0.809 | - | 6.946 ± 0.990 | 6.606 ± 0.324 | 6.491 ± 1.382 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 5.015 ± 0.253 | - | 4.782 ± 0.463 | 3.263 ± 0.034 | 5.716 ± 1.602 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.002 ± 0.001 | - | 0.005 ± 0.000 | 0.005 ± 0.002 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.091 ± 0.010 | - | 0.094 ± 0.007 | 0.095 ± 0.005 | 0.243 ± 0.162 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.005 ± 0.001 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.042 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.009 ± 0.001 | - | 0.009 ± 0.002 | 0.012 ± 0.002 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.015 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.005 | 0.012 ± 0.003 | 0.020 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.021 ± 0.005 | - | 0.023 ± 0.005 | 0.023 ± 0.000 | 0.111 ± 0.087 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.014 ± 0.000 | 0.008 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.008 ± 0.015 | - | 0.009 ± 0.000 | 0.012 ± 0.003 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.029 ± 0.005 | - | 0.026 ± 0.001 | 0.029 ± 0.004 | 0.133 ± 0.082 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 40.866 ± 0.849 | - | 36.785 ± 1.334 | 26.733 ± 1.017 | 18.393 ± 2.655 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.002 | 0.021 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.805 ± 0.068 | - | 1.150 ± 0.501 | 0.809 ± 0.055 | 1.019 ± 0.191 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.013 ± 0.004 | - | 0.045 ± 0.000 | 0.017 ± 0.004 | 0.027 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.052 ± 0.007 | - | 0.053 ± 0.011 | 0.053 ± 0.001 | 0.220 ± 0.071 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.236 ± 0.010 | - | 0.238 ± 0.000 | 0.239 ± 0.001 | 0.650 ± 0.104 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.264 ± 0.116 | - | 1.250 ± 0.038 | 1.284 ± 0.126 | 1.879 ± 0.107 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 4.908 ± 0.268 | - | 4.860 ± 0.091 | 6.167 ± 0.546 | 6.212 ± 0.508 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 0.973 ± 0.017 | - | 0.971 ± 0.027 | 1.592 ± 0.751 | 1.143 ± 0.089 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.067 ± 0.001 | - | 0.068 ± 0.001 | 0.083 ± 0.026 | 0.117 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.002 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.001 | 0.008 ± 0.000 | 0.016 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.015 ± 0.003 | - | 0.017 ± 0.000 | 0.019 ± 0.004 | 0.027 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.002 | 0.010 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.020 ± 0.000 | - | 0.021 ± 0.005 | 0.023 ± 0.001 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 7.762 ± 0.056 | - | 7.860 ± 1.358 | 11.092 ± 5.891 | 11.419 ± 4.546 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.524 ± 0.001 | - | 0.524 ± 0.008 | 1.039 ± 0.662 | 0.728 ± 0.156 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.011 ± 0.002 | - | 0.012 ± 0.000 | 0.014 ± 0.000 | 0.024 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.039 ± 0.002 | - | 0.039 ± 0.001 | 0.042 ± 0.005 | 0.088 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.248 ± 0.025 | - | 0.252 ± 0.052 | 0.248 ± 0.010 | 1.074 ± 0.401 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.418 ± 0.092 | - | 1.421 ± 0.055 | 1.393 ± 1.330 | 1.410 ± 0.519 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 5.442 ± 0.545 | - | 5.087 ± 1.896 | 5.472 ± 0.332 | 3.888 ± 0.856 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.027 ± 0.390 | - | 1.062 ± 0.379 | 1.032 ± 0.061 | 1.313 ± 0.305 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.003 ± 0.001 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.068 ± 0.001 | - | 0.069 ± 0.001 | 0.070 ± 0.002 | 0.220 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.006 ± 0.000 | - | 0.010 ± 0.002 | 0.011 ± 0.016 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.002 | 0.005 ± 0.000 | 0.011 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.016 ± 0.001 | - | 0.018 ± 0.004 | 0.019 ± 0.006 | 0.031 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.002 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.025 ± 0.003 | - | 0.023 ± 0.003 | 0.026 ± 0.005 | 0.031 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 8.641 ± 0.140 | - | 8.643 ± 0.884 | 8.474 ± 0.503 | 6.215 ± 1.540 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.004 ± 0.001 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.523 ± 0.003 | - | 0.562 ± 0.026 | 0.560 ± 0.035 | 0.532 ± 0.112 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.033 ± 0.001 | - | 0.013 ± 0.026 | 0.013 ± 0.000 | 0.071 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.044 ± 0.010 | - | 0.041 ± 0.002 | 0.042 ± 0.004 | 0.104 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.120 ± 0.034 | - | 0.121 ± 0.030 | 0.137 ± 0.031 | 0.823 ± 0.260 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 0.825 ± 0.081 | - | 0.850 ± 0.050 | 0.842 ± 0.086 | 1.783 ± 0.092 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 2.656 ± 0.219 | - | 2.630 ± 0.036 | 2.708 ± 0.127 | 5.412 ± 2.695 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.508 ± 0.011 | - | 0.506 ± 0.047 | 0.499 ± 0.029 | 1.316 ± 0.193 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.005 ± 0.002 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.031 ± 0.005 | - | 0.026 ± 0.004 | 0.030 ± 0.008 | 0.102 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.002 | 0.013 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.010 ± 0.001 | - | 0.010 ± 0.035 | 0.012 ± 0.000 | 0.023 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.002 ± 0.000 | 0.006 ± 0.005 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.001 | 0.008 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.010 ± 0.000 | - | 0.011 ± 0.002 | 0.014 ± 0.004 | 0.084 ± 0.100 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 4.121 ± 0.402 | - | 4.091 ± 0.166 | 4.201 ± 0.222 | 12.421 ± 3.628 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.001 | 0.007 ± 0.001 | 0.015 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.206 ± 0.027 | - | 0.202 ± 0.034 | 0.226 ± 0.022 | 0.688 ± 0.349 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.019 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.019 ± 0.005 | - | 0.020 ± 0.005 | 0.023 ± 0.000 | 0.096 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.113 ± 0.021 | - | 0.120 ± 0.032 | 0.132 ± 0.005 | 0.936 ± 0.188 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 0.838 ± 0.031 | - | 0.826 ± 0.128 | 0.766 ± 0.078 | 1.681 ± 0.413 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 2.676 ± 0.459 | - | 3.350 ± 1.245 | 2.541 ± 0.065 | 3.890 ± 1.399 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.811 ± 0.393 | - | 0.454 ± 0.039 | 0.460 ± 0.018 | 1.126 ± 0.110 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.027 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.026 ± 0.004 | - | 0.060 ± 0.007 | 0.029 ± 0.001 | 0.247 ± 0.131 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.009 ± 0.013 | - | 0.005 ± 0.001 | 0.007 ± 0.000 | 0.041 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.041 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.011 ± 0.002 | - | 0.010 ± 0.001 | 0.012 ± 0.002 | 0.067 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.010 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.021 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.011 ± 0.000 | - | 0.026 ± 0.001 | 0.014 ± 0.000 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 4.195 ± 0.877 | - | 4.594 ± 0.729 | 3.763 ± 0.068 | 6.094 ± 2.787 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.004 ± 0.001 | 0.006 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.197 ± 0.018 | - | 0.217 ± 0.031 | 0.196 ± 0.010 | 1.198 ± 1.492 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.002 | 0.010 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.020 ± 0.000 | - | 0.043 ± 0.021 | 0.023 ± 0.004 | 0.143 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.098 ± 0.009 | - | 0.101 ± 0.005 | 0.199 ± 0.020 | 0.179 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.211 ± 0.011 | - | 0.210 ± 0.019 | 0.305 ± 0.023 | 0.289 ± 0.077 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.484 ± 0.058 | - | 0.483 ± 0.043 | 0.516 ± 0.026 | 0.492 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.062 ± 0.001 | - | 0.061 ± 0.001 | 0.068 ± 0.001 | 0.137 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.009 ± 0.001 | 0.010 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.007 ± 0.001 | - | 0.008 ± 0.002 | 0.011 ± 0.000 | 0.059 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.008 ± 0.000 | - | 0.026 ± 0.000 | 0.013 ± 0.000 | 0.105 ± 0.066 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.011 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.003 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.068 ± 0.087 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.010 ± 0.002 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.011 ± 0.002 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.001 | 0.009 ± 0.000 | 0.087 ± 0.082 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.504 ± 0.084 | - | 0.505 ± 0.041 | 0.465 ± 0.021 | 0.643 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.002 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.032 ± 0.008 | - | 0.035 ± 0.005 | 0.039 ± 0.000 | 0.107 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.050 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.003 | 0.054 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.092 ± 0.000 | - | 0.093 ± 0.003 | 0.260 ± 0.194 | 0.378 ± 0.172 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.193 ± 0.010 | - | 0.195 ± 0.010 | 0.365 ± 0.068 | 0.249 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.542 ± 0.022 | - | 0.577 ± 0.122 | 1.522 ± 1.986 | 0.725 ± 0.313 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.208 ± 0.128 | - | 0.215 ± 0.155 | 0.096 ± 0.044 | 0.232 ± 0.156 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.063 ± 0.025 | 0.018 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.012 ± 0.003 | - | 0.013 ± 0.003 | 0.023 ± 0.005 | 0.136 ± 0.067 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.004 ± 0.000 | 0.034 ± 0.036 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.025 ± 0.001 | - | 0.010 ± 0.000 | 0.038 ± 0.078 | 0.068 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.060 ± 0.040 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.084 ± 0.148 | 0.035 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.009 ± 0.006 | - | 0.005 ± 0.000 | 0.118 ± 0.068 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.036 ± 0.035 | 0.127 ± 0.073 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.004 ± 0.001 | - | 0.005 ± 0.000 | 0.096 ± 0.038 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.009 ± 0.000 | - | 0.005 ± 0.000 | 0.104 ± 0.042 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.005 ± 0.000 | - | 0.005 ± 0.000 | 0.031 ± 0.036 | 0.076 ± 0.067 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.516 ± 0.101 | - | 0.508 ± 0.055 | 0.910 ± 1.118 | 0.854 ± 0.203 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.004 ± 0.001 | - | 0.004 ± 0.000 | 0.078 ± 0.027 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.061 ± 0.006 | - | 0.059 ± 0.011 | 0.070 ± 0.199 | 0.164 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.010 ± 0.001 | - | 0.006 ± 0.001 | 0.073 ± 0.025 | 0.048 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.079 ± 0.036 | 0.041 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.260 ± 0.010 | - | 0.278 ± 0.010 | 0.323 ± 0.298 | 0.341 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.607 ± 0.045 | - | 0.612 ± 0.061 | 0.812 ± 0.162 | 0.708 ± 1.198 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.522 ± 0.020 | - | 1.237 ± 0.049 | 1.512 ± 1.789 | 1.541 ± 0.062 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.309 ± 0.010 | - | 0.308 ± 0.019 | 0.368 ± 0.112 | 0.395 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.013 ± 0.003 | - | 0.013 ± 0.000 | 0.015 ± 0.002 | 0.056 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.011 ± 0.000 | 0.038 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.017 ± 0.000 | 0.065 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.004 ± 0.001 | - | 0.005 ± 0.001 | 0.005 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.003 | 0.008 ± 0.000 | 0.014 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.039 ± 0.067 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.001 | 0.015 ± 0.003 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 2.464 ± 0.041 | - | 2.497 ± 0.077 | 2.841 ± 0.284 | 2.863 ± 0.093 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.087 ± 0.000 | - | 0.090 ± 0.005 | 0.092 ± 0.000 | 0.168 ± 0.059 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.007 ± 0.001 | - | 0.009 ± 0.000 | 0.021 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.014 ± 0.000 | - | 0.016 ± 0.003 | 0.015 ± 0.001 | 0.022 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.194 ± 0.022 | - | 0.207 ± 0.019 | 0.248 ± 0.009 | 0.681 ± 0.468 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.554 ± 0.152 | - | 0.723 ± 0.057 | 0.810 ± 0.279 | 1.010 ± 0.109 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.317 ± 0.421 | - | 1.328 ± 0.153 | 1.644 ± 0.984 | 1.385 ± 0.377 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.309 ± 0.011 | - | 0.323 ± 0.291 | 0.603 ± 0.223 | 0.497 ± 0.137 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.029 ± 0.018 | 0.034 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.013 ± 0.001 | - | 0.014 ± 0.002 | 0.040 ± 0.007 | 0.054 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.037 ± 0.010 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.016 ± 0.004 | - | 0.018 ± 0.003 | 0.044 ± 0.021 | 0.166 ± 0.093 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | 0.005 ± 0.000 | 0.036 ± 0.081 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.044 ± 0.031 | 0.015 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.001 | 0.032 ± 0.006 | 0.019 ± 0.053 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.006 ± 0.001 | - | 0.011 ± 0.002 | 0.035 ± 0.012 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.058 ± 0.042 | 0.038 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.027 ± 0.088 | 0.015 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.011 ± 0.002 | - | 0.008 ± 0.000 | 0.038 ± 0.009 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 2.666 ± 0.439 | - | 2.778 ± 0.192 | 3.195 ± 0.285 | 1.697 ± 0.173 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.004 ± 0.001 | - | 0.005 ± 0.000 | 0.028 ± 0.008 | 0.016 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.088 ± 0.000 | - | 0.090 ± 0.000 | 0.131 ± 0.011 | 0.342 ± 0.210 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.031 ± 0.021 | 0.018 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.013 ± 0.004 | - | 0.013 ± 0.000 | 0.074 ± 0.077 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.132 ± 0.001 | - | 0.113 ± 0.002 | 0.118 ± 0.004 | 0.182 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.208 ± 0.018 | - | 0.191 ± 0.006 | 0.178 ± 0.036 | 0.256 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.413 ± 0.071 | - | 0.396 ± 0.117 | 0.519 ± 0.642 | 0.587 ± 0.239 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.051 ± 0.000 | - | 0.058 ± 0.004 | 0.066 ± 0.011 | 0.113 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.013 ± 0.013 | 0.010 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.004 ± 0.001 | - | 0.008 ± 0.000 | 0.039 ± 0.001 | 0.104 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.010 | - | 0.005 ± 0.000 | 0.010 ± 0.002 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.003 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.014 ± 0.011 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.001 | 0.010 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.013 ± 0.002 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.391 ± 0.015 | - | 0.419 ± 0.026 | 0.541 ± 0.267 | 0.665 ± 0.301 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.003 | 0.034 ± 0.063 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.027 ± 0.000 | - | 0.032 ± 0.014 | 0.044 ± 0.010 | 0.126 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.001 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.105 ± 0.004 | - | 0.092 ± 0.004 | 0.289 ± 0.117 | 0.195 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.169 ± 0.008 | - | 0.149 ± 0.007 | 0.304 ± 0.100 | 0.333 ± 0.078 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.344 ± 0.020 | - | 0.333 ± 0.009 | 0.437 ± 0.191 | 0.779 ± 0.165 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.053 ± 0.003 | - | 0.062 ± 0.000 | 0.065 ± 0.004 | 0.126 ± 0.075 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.005 ± 0.001 | - | 0.013 ± 0.048 | 0.010 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.005 ± 0.000 | - | 0.010 ± 0.000 | 0.014 ± 0.003 | 0.141 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.001 | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.007 ± 0.004 | - | 0.009 ± 0.005 | 0.009 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.002 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.004 ± 0.001 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.038 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.401 ± 0.005 | - | 0.498 ± 0.008 | 0.485 ± 0.076 | 0.893 ± 0.477 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.046 ± 0.088 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.027 ± 0.000 | - | 0.034 ± 0.000 | 0.136 ± 0.057 | 0.447 ± 0.109 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.038 ± 0.033 | 0.014 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.009 ± 0.001 | 0.041 ± 0.016 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.302 ± 0.052 | - | 1.385 ± 0.098 | 1.408 ± 0.342 | 3.047 ± 0.068 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 4.803 ± 0.499 | - | 4.539 ± 0.305 | 5.819 ± 1.075 | 6.181 ± 0.201 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 17.503 ± 0.798 | - | 17.558 ± 0.426 | 16.897 ± 1.634 | 17.603 ± 2.163 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 6.772 ± 0.173 | - | 6.712 ± 0.364 | 5.143 ± 0.077 | 5.456 ± 0.139 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.018 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.201 ± 0.049 | - | 0.201 ± 0.027 | 0.195 ± 0.029 | 0.243 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.018 ± 0.029 | - | 0.007 ± 0.000 | 0.012 ± 0.002 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.177 ± 0.019 | - | 0.180 ± 0.030 | 0.170 ± 0.002 | 0.227 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.018 ± 0.001 | - | 0.019 ± 0.002 | 0.024 ± 0.000 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.001 | 0.010 ± 0.003 | 0.063 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.014 ± 0.030 | - | 0.015 ± 0.000 | 0.022 ± 0.005 | 0.032 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.046 ± 0.001 | - | 0.063 ± 0.106 | 0.065 ± 0.008 | 0.065 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.008 ± 0.001 | - | 0.008 ± 0.000 | 0.011 ± 0.001 | 0.024 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.018 ± 0.000 | - | 0.019 ± 0.001 | 0.026 ± 0.003 | 0.037 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.062 ± 0.009 | - | 0.072 ± 0.014 | 0.083 ± 0.007 | 0.081 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 50.361 ± 1.409 | - | 50.780 ± 1.277 | 38.380 ± 2.785 | 42.725 ± 1.424 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.012 ± 0.002 | - | 0.012 ± 0.002 | 0.016 ± 0.003 | 0.035 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.743 ± 0.377 | - | 1.744 ± 0.328 | 1.756 ± 0.527 | 1.640 ± 0.076 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.036 ± 0.008 | - | 0.048 ± 0.036 | 0.051 ± 0.006 | 0.061 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.120 ± 0.016 | - | 0.132 ± 0.028 | 0.154 ± 0.046 | 0.178 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.638 ± 0.876 | - | 1.566 ± 0.475 | 1.189 ± 0.022 | 4.136 ± 1.026 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 5.693 ± 0.827 | - | 5.620 ± 1.187 | 4.380 ± 0.078 | 6.225 ± 1.296 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 18.008 ± 5.245 | - | 19.658 ± 0.659 | 14.979 ± 0.540 | 10.921 ± 0.771 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 8.212 ± 2.267 | - | 6.492 ± 1.001 | 4.589 ± 0.408 | 6.617 ± 2.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.080 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.197 ± 0.048 | - | 0.213 ± 0.043 | 0.184 ± 0.024 | 0.263 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.008 ± 0.001 | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.168 ± 0.002 | - | 0.178 ± 0.017 | 0.172 ± 0.013 | 0.567 ± 0.082 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.018 ± 0.002 | - | 0.060 ± 0.002 | 0.021 ± 0.000 | 0.035 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.002 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.015 ± 0.003 | - | 0.018 ± 0.004 | 0.018 ± 0.004 | 0.030 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.048 ± 0.004 | - | 0.049 ± 0.005 | 0.050 ± 0.005 | 0.064 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.007 ± 0.002 | - | 0.009 ± 0.000 | 0.011 ± 0.002 | 0.035 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.019 ± 0.000 | - | 0.026 ± 0.010 | 0.022 ± 0.001 | 0.036 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.157 ± 0.099 | - | 0.078 ± 0.090 | 0.065 ± 0.000 | 0.081 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 51.056 ± 2.839 | - | 47.067 ± 0.707 | 37.082 ± 1.463 | 23.254 ± 7.283 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.015 ± 0.004 | 0.043 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 1.956 ± 1.244 | - | 1.805 ± 1.200 | 1.439 ± 0.024 | 1.271 ± 0.402 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.036 ± 0.001 | - | 0.038 ± 0.001 | 0.039 ± 0.000 | 0.183 ± 0.105 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.128 ± 0.012 | - | 0.281 ± 0.137 | 0.120 ± 0.001 | 0.180 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.640 ± 0.086 | - | 0.570 ± 0.089 | 0.602 ± 0.071 | 3.348 ± 1.137 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 3.027 ± 0.057 | - | 3.033 ± 0.080 | 3.239 ± 0.337 | 4.296 ± 2.944 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 12.143 ± 0.834 | - | 12.008 ± 0.294 | 12.095 ± 0.403 | 18.808 ± 3.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.823 ± 0.147 | - | 1.629 ± 0.433 | 1.786 ± 0.215 | 2.165 ± 0.303 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.003 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.135 ± 0.020 | - | 0.121 ± 0.001 | 0.127 ± 0.016 | 0.241 ± 0.109 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.001 | 0.017 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.095 ± 0.003 | - | 0.096 ± 0.001 | 0.116 ± 0.057 | 0.193 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.012 ± 0.005 | - | 0.013 ± 0.003 | 0.017 ± 0.001 | 0.046 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.002 | 0.025 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.009 ± 0.002 | - | 0.013 ± 0.001 | 0.014 ± 0.000 | 0.075 ± 0.086 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.031 ± 0.005 | - | 0.033 ± 0.006 | 0.037 ± 0.005 | 0.048 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.005 ± 0.001 | - | 0.006 ± 0.001 | 0.008 ± 0.000 | 0.025 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.012 ± 0.003 | - | 0.014 ± 0.003 | 0.017 ± 0.001 | 0.101 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.040 ± 0.006 | - | 0.045 ± 0.007 | 0.048 ± 0.006 | 0.197 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 12.805 ± 0.362 | - | 12.950 ± 0.350 | 13.150 ± 1.119 | 15.376 ± 3.583 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.003 | 0.029 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 1.042 ± 0.046 | - | 1.016 ± 0.121 | 1.127 ± 0.253 | 1.385 ± 0.110 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.023 ± 0.005 | - | 0.024 ± 0.000 | 0.027 ± 0.005 | 0.046 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.085 ± 0.035 | - | 0.086 ± 0.039 | 0.093 ± 0.041 | 0.187 ± 0.113 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.655 ± 0.107 | - | 0.718 ± 0.295 | 0.578 ± 0.193 | 2.818 ± 0.354 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 3.506 ± 0.470 | - | 3.973 ± 0.832 | 2.733 ± 0.076 | 3.084 ± 0.276 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 12.707 ± 4.036 | - | 15.247 ± 3.059 | 10.539 ± 0.310 | 7.522 ± 1.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 2.186 ± 1.901 | - | 2.924 ± 1.403 | 1.467 ± 0.023 | 1.990 ± 2.227 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.011 ± 0.001 | - | 0.006 ± 0.001 | 0.006 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.124 ± 0.011 | - | 0.149 ± 0.052 | 0.125 ± 0.001 | 0.365 ± 0.211 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.101 ± 0.004 | - | 0.101 ± 0.004 | 0.100 ± 0.001 | 0.128 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.047 ± 0.010 | - | 0.024 ± 0.037 | 0.016 ± 0.004 | 0.027 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.012 ± 0.002 | - | 0.012 ± 0.001 | 0.013 ± 0.000 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.038 ± 0.003 | - | 0.036 ± 0.006 | 0.033 ± 0.001 | 0.054 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.006 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.014 ± 0.013 | - | 0.021 ± 0.073 | 0.016 ± 0.000 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.043 ± 0.003 | - | 0.044 ± 0.005 | 0.043 ± 0.000 | 0.061 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 20.282 ± 8.006 | - | 14.172 ± 2.245 | 11.986 ± 1.330 | 9.379 ± 2.163 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.007 ± 0.001 | - | 0.009 ± 0.002 | 0.010 ± 0.003 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 1.128 ± 0.179 | - | 1.439 ± 0.775 | 0.951 ± 0.015 | 1.395 ± 0.490 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.025 ± 0.000 | - | 0.030 ± 0.072 | 0.027 ± 0.000 | 0.057 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.082 ± 0.001 | - | 0.082 ± 0.001 | 0.082 ± 0.001 | 0.322 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.228 ± 0.043 | - | unsupported | 0.219 ± 0.019 | 2.350 ± 1.409 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.649 ± 0.172 | - | unsupported | 1.358 ± 0.021 | 2.743 ± 0.085 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 4.335 ± 0.166 | - | unsupported | 3.828 ± 1.811 | 5.131 ± 0.090 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.858 ± 0.024 | - | unsupported | 0.773 ± 0.008 | 0.910 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.041 ± 0.025 | - | unsupported | 0.041 ± 0.000 | 0.049 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.031 ± 0.003 | - | unsupported | 0.038 ± 0.006 | 0.043 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.039 ± 0.080 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.012 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | unsupported | 0.010 ± 0.002 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.016 ± 0.004 | - | unsupported | 0.020 ± 0.003 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 6.797 ± 0.210 | - | unsupported | 6.264 ± 0.092 | 7.423 ± 1.678 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.336 ± 0.044 | - | unsupported | 0.292 ± 0.002 | 0.366 ± 0.072 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.011 ± 0.000 | - | unsupported | 0.015 ± 0.000 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.030 ± 0.007 | - | unsupported | 0.034 ± 0.000 | 0.049 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.208 ± 0.002 | - | unsupported | 0.218 ± 0.001 | 3.298 ± 0.950 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 1.359 ± 0.025 | - | unsupported | 1.401 ± 0.056 | 2.321 ± 0.506 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 4.280 ± 0.452 | - | unsupported | 3.951 ± 0.076 | 3.439 ± 0.106 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 0.780 ± 0.052 | - | unsupported | 0.788 ± 0.028 | 0.818 ± 0.676 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.038 ± 0.001 | - | unsupported | 0.041 ± 0.001 | 0.114 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.015 ± 0.105 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.031 ± 0.000 | - | unsupported | 0.048 ± 0.019 | 0.047 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.016 ± 0.108 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.006 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.064 ± 0.062 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.012 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.024 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.007 ± 0.000 | - | unsupported | 0.012 ± 0.001 | 0.020 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.017 ± 0.000 | - | unsupported | 0.020 ± 0.000 | 0.049 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 6.480 ± 2.547 | - | unsupported | 6.433 ± 1.350 | 3.985 ± 0.469 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.055 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.288 ± 0.020 | - | unsupported | 0.293 ± 0.009 | 0.290 ± 0.118 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.012 ± 0.000 | - | unsupported | 0.015 ± 0.000 | 0.031 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.033 ± 0.007 | - | unsupported | 0.034 ± 0.000 | 0.051 ± 0.013 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.819 ± 0.021 | - | unsupported | 0.836 ± 0.024 | 1.238 ± 0.051 | 2.923 ± 3.083 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.057 ± 0.000 | - | unsupported | 0.086 ± 0.030 | 0.135 ± 0.004 | 0.081 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.115 ± 0.007 | - | unsupported | 0.143 ± 0.019 | 0.236 ± 0.044 | 0.185 ± 0.014 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.271 ± 0.019 | - | unsupported | 0.288 ± 0.018 | 0.383 ± 0.076 | 0.842 ± 0.088 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.839 ± 0.057 | - | unsupported | 1.248 ± 0.832 | 1.758 ± 0.076 | 2.666 ± 0.888 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.148 ± 0.004 | - | unsupported | 0.174 ± 0.140 | 0.198 ± 0.061 | 0.083 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.306 ± 0.028 | - | unsupported | 0.216 ± 0.096 | 0.205 ± 0.030 | 0.195 ± 0.014 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.607 ± 0.413 | - | unsupported | 0.435 ± 0.084 | 0.524 ± 0.095 | 1.390 ± 0.470 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.244 ± 0.074 | - | unsupported | 0.188 ± 0.009 | 0.552 ± 0.085 | 0.571 ± 0.041 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.065 ± 0.001 | - | unsupported | 0.053 ± 0.025 | 0.091 ± 0.011 | 0.069 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.101 ± 0.003 | - | unsupported | 0.068 ± 0.013 | 0.114 ± 0.022 | 0.089 ± 0.005 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.101 ± 0.010 | - | unsupported | 0.320 ± 0.025 | 0.184 ± 0.192 | 0.256 ± 0.038 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.211 ± 0.010 | - | unsupported | 0.187 ± 0.002 | 0.815 ± 0.199 | 0.552 ± 0.046 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.049 ± 0.007 | - | unsupported | 0.043 ± 0.001 | 0.121 ± 0.041 | 0.059 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.055 ± 0.011 | - | unsupported | 0.050 ± 0.004 | 0.190 ± 0.113 | 0.089 ± 0.018 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.081 ± 0.006 | - | unsupported | 0.081 ± 0.010 | 0.191 ± 0.038 | 0.248 ± 0.029 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.112 ± 0.011 | - | 0.109 ± 0.001 | 0.136 ± 0.139 | 0.188 ± 0.004 | 0.347 ± 0.079 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.045 ± 0.000 | - | 0.051 ± 0.014 | 0.079 ± 0.004 | 0.028 ± 0.005 | 0.034 ± 0.013 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.048 ± 0.007 | - | 0.045 ± 0.000 | 0.094 ± 0.001 | 0.032 ± 0.002 | 0.046 ± 0.010 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.056 ± 0.001 | - | 0.058 ± 0.000 | 0.042 ± 0.012 | 0.059 ± 0.001 | 0.130 ± 0.017 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.109 ± 0.009 | - | 0.111 ± 0.000 | 0.082 ± 0.039 | 0.252 ± 0.140 | 0.248 ± 0.010 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.118 ± 0.002 | - | 0.042 ± 0.009 | 0.041 ± 0.021 | 0.080 ± 0.022 | 0.035 ± 0.003 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.085 ± 0.093 | - | 0.047 ± 0.006 | 0.045 ± 0.023 | 0.033 ± 0.003 | 0.044 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.055 ± 0.000 | - | 0.064 ± 0.005 | 0.057 ± 0.031 | 0.121 ± 0.136 | 0.102 ± 0.009 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 12.776 ± 6.487 | - | 11.205 ± 1.522 | 24.880 ± 0.766 | 20.189 ± 3.221 | 49.532 ± 13.260 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 10.925 ± 0.666 | - | 10.741 ± 6.545 | 22.913 ± 3.987 | 17.870 ± 5.504 | 362.448 ± 75.875 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 7.889 ± 0.440 | - | 8.572 ± 1.711 | 7.549 ± 0.122 | 13.423 ± 2.169 | 22.047 ± 3.246 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 7.782 ± 0.117 | - | 9.768 ± 2.706 | 8.666 ± 3.173 | 10.106 ± 1.308 | 266.426 ± 32.312 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 12.944 ± 4.439 | - | 6.928 ± 6.392 | 6.824 ± 0.068 | 8.733 ± 0.501 | 9.163 ± 3.164 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 8.318 ± 2.956 | - | 6.869 ± 1.483 | 7.133 ± 2.316 | 11.647 ± 1.849 | 317.879 ± 23.068 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 7.973 ± 5.562 | - | 9.591 ± 3.620 | 7.623 ± 0.148 | 7.258 ± 0.255 | 7.328 ± 0.090 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 7.818 ± 1.450 | - | 7.521 ± 0.104 | 7.940 ± 0.683 | 11.513 ± 6.309 | 265.994 ± 29.119 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 21.593 ± 4.297 | - | 21.645 ± 2.404 | 21.398 ± 0.351 | 27.974 ± 0.902 | 13.838 ± 0.627 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 20.750 ± 0.033 | - | 20.867 ± 0.283 | 23.410 ± 3.536 | 40.411 ± 4.213 | 682.908 ± 45.086 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.342 ± 0.067 | 0.324 ± 0.019 | unsupported | 0.352 ± 0.028 | 0.409 ± 0.011 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 4.265 ± 0.130 | 4.332 ± 0.357 | unsupported | 4.519 ± 0.509 | 5.414 ± 0.857 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.333 ± 0.019 | 0.357 ± 0.050 | unsupported | 0.346 ± 0.156 | 0.581 ± 0.052 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 4.988 ± 1.642 | 5.936 ± 1.147 | unsupported | 4.465 ± 0.512 | 5.107 ± 0.665 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 8.583 ± 2.205 | - | 9.682 ± 0.628 | 9.124 ± 0.301 | 9.539 ± 0.455 | 33.441 ± 1.658 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 9.023 ± 0.233 | - | 9.154 ± 0.231 | 10.979 ± 2.632 | 9.832 ± 0.397 | 365.899 ± 64.942 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 10.008 ± 5.757 | 10.207 ± 0.806 | 5.111 ± 0.066 | 31.094 ± 5.827 | 10.346 ± 2.944 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 12.598 ± 2.094 | 11.066 ± 0.400 | 6.194 ± 3.417 | 37.542 ± 3.528 | 392.130 ± 100.765 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 11.039 ± 1.984 | - | 11.728 ± 2.047 | 10.472 ± 1.287 | 16.627 ± 2.012 | 28.268 ± 16.468 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 10.105 ± 1.193 | - | 8.900 ± 2.084 | 10.451 ± 0.557 | 11.078 ± 1.210 | 251.830 ± 21.801 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.640 ± 0.011 | - | 1.046 ± 0.288 | 2.176 ± 0.022 | 7.656 ± 3.213 | 6.144 ± 0.404 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.987 ± 0.203 | - | 0.962 ± 0.630 | 2.867 ± 0.434 | 4.744 ± 0.861 | 6.312 ± 2.812 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 12.156 ± 4.168 | - | 10.586 ± 0.265 | 10.597 ± 0.149 | 13.300 ± 0.940 | 24.100 ± 3.160 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 12.824 ± 0.810 | - | 12.408 ± 0.611 | 14.732 ± 4.466 | 14.525 ± 1.644 | 1151.584 ± 652.739 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 10.851 ± 0.206 | - | 10.737 ± 0.715 | 10.696 ± 0.262 | 14.799 ± 1.466 | 21.985 ± 3.071 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 12.262 ± 0.326 | - | 11.629 ± 1.649 | 12.164 ± 1.333 | 14.275 ± 2.277 | 1008.961 ± 767.906 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.598 ± 0.325 | 1.653 ± 0.084 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 14.815 ± 2.292 | 15.603 ± 0.752 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.590 ± 0.053 | 1.611 ± 0.120 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 17.000 ± 5.165 | 17.322 ± 0.590 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.625 ± 0.092 | 1.584 ± 0.154 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 17.172 ± 2.256 | 12.334 ± 0.563 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.616 ± 0.156 | 1.582 ± 0.250 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 14.127 ± 1.609 | 16.880 ± 2.991 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 8.691 ± 1.183 | - | 7.970 ± 0.451 | 7.468 ± 0.420 | 13.871 ± 1.830 | 24.506 ± 4.531 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 7.940 ± 0.171 | - | 8.938 ± 1.255 | 9.531 ± 4.868 | 9.771 ± 2.532 | 401.582 ± 281.660 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 39.313 ± 1.232 | 39.032 ± 3.654 | 34.580 ± 0.899 | 35.281 ± 3.257 | 78.668 ± 4.099 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 46.262 ± 5.004 | 46.674 ± 4.403 | 44.415 ± 2.105 | 45.872 ± 2.987 | 1954.307 ± 490.443 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 14.860 ± 1.563 | - | 15.944 ± 1.862 | 12.352 ± 1.035 | 96.203 ± 31.669 | 28.887 ± 0.795 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 15.060 ± 0.370 | - | 15.925 ± 0.819 | 16.927 ± 6.282 | 55.267 ± 14.794 | 32.182 ± 7.810 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.203 ± 0.038 | 0.191 ± 0.093 | unsupported | 0.054 ± 0.051 | 0.035 ± 0.001 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.777 ± 0.144 | 2.847 ± 1.188 | unsupported | 0.851 ± 0.046 | 0.730 ± 0.827 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.175 ± 0.051 | 0.177 ± 0.003 | unsupported | 0.123 ± 0.008 | 0.036 ± 0.012 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 2.020 ± 0.322 | 2.022 ± 0.208 | unsupported | 0.782 ± 0.242 | 0.502 ± 0.191 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.079 ± 0.002 | 0.080 ± 0.005 | unsupported | 0.086 ± 0.002 | 0.215 ± 0.020 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.402 ± 0.073 | 1.423 ± 0.072 | unsupported | 1.372 ± 0.025 | 3.792 ± 0.512 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.126 ± 0.004 | 0.136 ± 0.016 | unsupported | 0.161 ± 0.011 | 0.191 ± 0.081 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 1.078 ± 0.537 | 1.417 ± 0.450 | unsupported | 1.161 ± 0.299 | 2.328 ± 0.425 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.267 ± 0.009 | 0.376 ± 0.113 | unsupported | 0.301 ± 0.032 | 0.103 ± 0.013 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 5.000 ± 1.223 | 6.616 ± 1.313 | unsupported | 4.792 ± 0.219 | 1.642 ± 0.505 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.326 ± 0.044 | 0.263 ± 0.031 | unsupported | 0.329 ± 0.031 | 0.090 ± 0.009 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 2.936 ± 0.436 | 2.908 ± 0.330 | unsupported | 3.644 ± 0.511 | 1.241 ± 0.546 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.045 ± 0.001 | 0.065 ± 0.001 | unsupported | 0.046 ± 0.003 | 0.129 ± 0.016 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 1.008 ± 0.310 | 1.166 ± 0.764 | unsupported | 0.564 ± 0.008 | 0.823 ± 0.219 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.052 ± 0.001 | 0.070 ± 0.001 | unsupported | 0.108 ± 0.008 | 0.117 ± 0.023 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 1.157 ± 0.127 | 1.312 ± 0.379 | unsupported | 0.480 ± 0.048 | 0.878 ± 0.760 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.091 ± 0.002 | 0.105 ± 0.002 | unsupported | 0.090 ± 0.001 | 0.142 ± 0.016 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.943 ± 0.875 | 1.931 ± 0.679 | unsupported | 1.431 ± 0.056 | 1.548 ± 0.261 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.097 ± 0.004 | 0.111 ± 0.004 | unsupported | 0.152 ± 0.017 | 0.158 ± 0.010 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 1.880 ± 0.127 | 2.720 ± 0.861 | unsupported | 1.099 ± 0.177 | 0.868 ± 0.522 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.124 ± 0.001 | 0.147 ± 0.044 | unsupported | unsupported | 0.179 ± 0.029 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 3.669 ± 1.354 | 2.600 ± 1.021 | unsupported | unsupported | 2.081 ± 0.228 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.130 ± 0.003 | 0.149 ± 0.001 | unsupported | unsupported | 0.151 ± 0.030 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 2.124 ± 0.182 | 2.211 ± 0.645 | unsupported | unsupported | 1.988 ± 1.527 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.012 ± 0.000 | 0.016 ± 0.001 | unsupported | 0.008 ± 0.000 | 0.062 ± 0.006 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.143 ± 0.075 | 0.173 ± 0.074 | unsupported | 0.060 ± 0.004 | 0.244 ± 0.084 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.032 ± 0.021 | 0.025 ± 0.031 | unsupported | 0.060 ± 0.013 | 0.051 ± 0.006 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.226 ± 0.062 | 0.259 ± 0.082 | unsupported | 0.145 ± 0.028 | 0.359 ± 0.219 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.026 ± 0.000 | 0.038 ± 0.002 | unsupported | 0.028 ± 0.001 | 0.103 ± 0.007 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.326 ± 0.008 | 0.332 ± 0.029 | unsupported | 0.345 ± 0.045 | 0.684 ± 0.016 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.028 ± 0.002 | 0.041 ± 0.002 | unsupported | 0.189 ± 0.150 | 0.099 ± 0.015 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.630 ± 0.083 | 0.664 ± 0.041 | unsupported | 0.566 ± 0.076 | 0.486 ± 0.085 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.049 ± 0.005 | 0.049 ± 0.002 | unsupported | 0.077 ± 0.001 | 0.096 ± 0.003 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.769 ± 0.002 | 0.768 ± 0.008 | unsupported | 1.253 ± 0.046 | 1.453 ± 0.405 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.078 ± 0.025 | 0.071 ± 0.061 | unsupported | 0.133 ± 0.015 | 0.099 ± 0.030 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.662 ± 0.062 | 0.761 ± 0.176 | unsupported | 0.971 ± 0.168 | 0.982 ± 0.316 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.050 ± 0.001 | 0.051 ± 0.001 | unsupported | 0.078 ± 0.001 | 0.187 ± 0.149 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.796 ± 0.004 | 0.811 ± 0.017 | unsupported | 1.260 ± 0.006 | 1.430 ± 0.607 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.197 ± 0.165 | 0.103 ± 0.026 | unsupported | 0.142 ± 0.019 | 0.164 ± 0.042 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.694 ± 0.083 | 0.756 ± 0.363 | unsupported | 1.164 ± 0.188 | 0.675 ± 0.247 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.088 ± 0.003 | 0.103 ± 0.001 | unsupported | 0.099 ± 0.000 | 0.153 ± 0.041 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.838 ± 0.712 | 2.080 ± 1.496 | unsupported | 1.690 ± 0.028 | 1.410 ± 0.084 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.100 ± 0.012 | 0.142 ± 0.034 | unsupported | 0.153 ± 0.006 | 0.147 ± 0.084 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 2.012 ± 0.162 | 1.905 ± 0.071 | unsupported | 1.251 ± 0.080 | 1.018 ± 0.589 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.219 ± 0.001 | 0.220 ± 0.001 | unsupported | 0.278 ± 0.002 | 0.305 ± 0.016 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 3.976 ± 0.549 | 3.960 ± 0.132 | unsupported | 4.458 ± 0.054 | 6.578 ± 0.908 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.302 ± 0.128 | 0.248 ± 0.052 | unsupported | 0.291 ± 0.018 | 0.395 ± 0.307 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 2.537 ± 0.592 | 2.573 ± 0.275 | unsupported | 3.546 ± 0.465 | 3.583 ± 0.456 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.529 ± 0.017 | 0.648 ± 0.046 | unsupported | 0.118 ± 0.001 | 0.171 ± 0.010 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 11.080 ± 0.772 | 12.793 ± 0.735 | unsupported | 2.355 ± 0.199 | 1.686 ± 0.325 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.808 ± 0.136 | 1.020 ± 0.334 | unsupported | 0.255 ± 0.023 | 0.126 ± 0.048 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 8.749 ± 0.893 | 11.541 ± 1.520 | unsupported | 1.635 ± 0.204 | 0.782 ± 0.409 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 16.388 ± 0.184 | - | unsupported | 31.549 ± 13.320 | - | 27.869 ± 10.638 | 20.663 ± 2.072 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 17.990 ± 0.797 | - | unsupported | 21.561 ± 1.817 | - | 22.319 ± 2.399 | 23.133 ± 4.035 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 12.105 ± 0.638 | - | unsupported | 14.887 ± 2.157 | - | 12.477 ± 2.885 | 12.173 ± 1.003 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 12.777 ± 0.547 | - | unsupported | 13.674 ± 0.918 | - | 17.763 ± 3.869 | 16.201 ± 4.226 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 9.738 ± 1.948 | - | unsupported | 13.571 ± 1.207 | - | 10.287 ± 1.002 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 13.649 ± 1.796 | - | unsupported | 13.658 ± 0.822 | - | 18.574 ± 6.740 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 21.122 ± 5.775 | - | unsupported | 29.921 ± 13.598 | - | 20.045 ± 6.557 | 19.915 ± 1.316 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 19.732 ± 0.779 | - | unsupported | 19.747 ± 1.420 | - | 24.575 ± 1.568 | 23.298 ± 2.132 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 10.437 ± 1.687 | - | unsupported | 17.801 ± 6.544 | - | 56.737 ± 8.580 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 12.788 ± 1.363 | - | unsupported | 12.064 ± 0.813 | - | 41.113 ± 7.884 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 13.275 ± 10.326 | - | unsupported | 14.018 ± 0.677 | - | 62.447 ± 6.418 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 13.334 ± 0.229 | - | unsupported | 14.133 ± 0.868 | - | 35.866 ± 7.799 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 17.184 ± 2.246 | - | unsupported | 31.113 ± 8.098 | - | 19.012 ± 0.684 | 18.723 ± 0.811 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 19.823 ± 1.909 | - | unsupported | 21.301 ± 2.791 | - | 24.027 ± 0.657 | 21.888 ± 5.726 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 14.321 ± 2.442 | - | unsupported | 22.669 ± 5.014 | - | 11.118 ± 0.254 | 12.153 ± 0.503 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 12.843 ± 0.967 | - | unsupported | 12.156 ± 0.716 | - | 14.111 ± 2.584 | 9.923 ± 1.451 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.250 ± 0.043 | unsupported | unsupported | 0.297 ± 0.042 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 4.660 ± 1.404 | unsupported | unsupported | 4.317 ± 0.360 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.227 ± 0.018 | unsupported | unsupported | 0.347 ± 0.033 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 3.018 ± 0.423 | unsupported | unsupported | 4.112 ± 0.193 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 16.861 ± 2.492 | - | unsupported | 26.112 ± 3.565 | - | 21.916 ± 1.476 | 21.731 ± 5.414 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 19.781 ± 1.531 | - | unsupported | 20.326 ± 2.627 | - | 21.959 ± 3.898 | 21.384 ± 2.042 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 8.251 ± 0.687 | - | 0.002 ± 0.000 | 5.919 ± 0.718 | 7.157 ± 1.045 | 13.502 ± 9.192 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 19.817 ± 4.847 | - | 0.004 ± 0.000 | 6.032 ± 0.367 | 5.829 ± 0.464 | 15.766 ± 0.687 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 9.832 ± 0.577 | - | 20.411 ± 4.493 | 10.357 ± 0.691 | 14.467 ± 7.059 | 9.725 ± 0.170 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 15.518 ± 6.284 | - | 26.236 ± 2.504 | 10.770 ± 0.573 | 10.725 ± 1.352 | 10.628 ± 1.282 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 10.382 ± 3.833 | - | 11.095 ± 2.116 | 15.565 ± 4.907 | 18.046 ± 3.128 | 13.703 ± 1.754 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 14.651 ± 4.684 | - | 15.409 ± 3.443 | 13.688 ± 1.873 | 15.392 ± 3.769 | 15.804 ± 3.045 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 6.059 ± 0.568 | - | 5.217 ± 0.563 | 8.839 ± 0.279 | 12.133 ± 2.822 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 6.366 ± 1.343 | - | 6.335 ± 0.143 | 10.792 ± 0.557 | 10.109 ± 0.912 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 11.306 ± 1.844 | - | 0.002 ± 0.000 | 11.968 ± 1.207 | 13.001 ± 1.787 | 15.024 ± 6.957 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 15.973 ± 0.940 | - | 0.002 ± 0.000 | 12.680 ± 1.122 | 11.822 ± 1.997 | 16.583 ± 8.827 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 46.605 ± 2.353 | - | 0.002 ± 0.007 | 50.628 ± 8.145 | 56.881 ± 4.861 | 230.642 ± 32.417 | 40.550 ± 8.728 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 67.848 ± 26.474 | - | 0.002 ± 0.015 | 55.233 ± 4.440 | 37.475 ± 9.063 | 245.319 ± 44.022 | 20.984 ± 3.036 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 8.010 ± 1.861 | - | 7.191 ± 0.440 | 9.152 ± 0.898 | 5.934 ± 2.223 | 9.719 ± 0.581 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 6.132 ± 0.166 | - | 5.847 ± 0.245 | 7.908 ± 0.552 | 6.329 ± 1.083 | 12.646 ± 3.383 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 7.340 ± 0.542 | - | 8.035 ± 0.544 | 9.074 ± 0.825 | 5.590 ± 0.403 | 10.804 ± 2.095 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 6.134 ± 0.435 | - | 5.853 ± 0.544 | 7.797 ± 0.437 | 7.444 ± 1.344 | 10.891 ± 3.465 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.003 ± 0.002 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.003 ± 0.001 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 3.951208 | 246950.50 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 4.169291 | 260580.69 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 4.338250 | 271140.62 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 7.224958 | 451559.88 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 4.529542 | 283096.38 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 5.301583 | 331348.94 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 14.656000 | 916000.00 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 11.968667 | 748041.69 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 18.711375 | 1169460.94 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 18.374250 | 1148390.62 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 6.544000 | 409000.00 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 5.624041 | 351502.56 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 5.470083 | 341880.19 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 5.191792 | 324487.00 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 86.619750 | 5413734.38 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 72.298167 | 4518635.44 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 68.240875 | 4265054.69 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 69.314666 | 4332166.62 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 9.300083 | 581255.19 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 5.535667 | 345979.19 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 5.328583 | 333036.44 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 5.718875 | 357429.69 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 81.717291 | 5107330.69 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 71.437584 | 4464849.00 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 79.803792 | 4987737.00 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 94.983458 | 5936466.12 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 25.569833 | 1598114.56 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 26.452500 | 1653281.25 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 237.034500 | 14814656.25 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 249.646916 | 15602932.25 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 25.434958 | 1589684.88 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 25.773958 | 1610872.38 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 271.994833 | 16999677.06 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 277.150541 | 17321908.81 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 26.002958 | 1625184.88 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 25.340791 | 1583799.44 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 274.759834 | 17172489.62 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 197.341958 | 12333872.38 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 25.848083 | 1615505.19 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 25.306208 | 1581638.00 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 226.036875 | 14127304.69 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 270.087917 | 16880494.81 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.233125 | 34892.58 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.429750 | 53589.84 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 12.980625 | 202822.27 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 12.221875 | 190966.80 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.841209 | 730151.12 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 6.811833 | 851479.12 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 22.219542 | 2777442.75 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 22.777625 | 2847203.12 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.293542 | 35836.59 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 7.878333 | 123098.95 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 11.218292 | 175285.81 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 11.312542 | 176758.47 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.013166 | 501645.75 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.254875 | 781859.38 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 16.156500 | 2019562.50 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 16.175750 | 2021968.75 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.765375 | 215083.98 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.498083 | 85907.55 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.072625 | 79259.77 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.091458 | 79554.03 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 30.335667 | 3791958.38 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.978625 | 1372328.12 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 11.216833 | 1402104.12 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.386459 | 1423307.38 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.226125 | 191033.20 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 10.292083 | 160813.80 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.050875 | 125794.92 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 8.714958 | 136171.22 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 18.623833 | 2327979.12 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 9.285959 | 1160744.88 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.624459 | 1078057.38 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 11.334416 | 1416802.00 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.620292 | 103442.06 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 19.280792 | 301262.38 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 17.059750 | 266558.59 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 24.081333 | 376270.83 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.133959 | 1641744.88 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 38.338584 | 4792323.00 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 39.999458 | 4999932.25 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 52.926667 | 6615833.38 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.787292 | 90426.44 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 21.087167 | 329486.98 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 20.865417 | 326022.14 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 16.826083 | 262907.55 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.928167 | 1241020.88 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 29.153000 | 3644125.00 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 23.485625 | 2935703.12 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 23.266333 | 2908291.62 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.259708 | 129057.94 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 2.944500 | 46007.81 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.897792 | 45278.00 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.138875 | 64669.92 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.586917 | 823364.62 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.513583 | 564197.88 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 8.061292 | 1007661.50 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 9.325917 | 1165739.62 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.486542 | 116977.22 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.891667 | 107682.30 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.319417 | 51865.89 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.510792 | 70481.12 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.025250 | 878156.25 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.836375 | 479546.88 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.259375 | 1157421.88 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 10.492458 | 1311557.25 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.074708 | 141792.31 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.772625 | 90197.27 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.851500 | 91429.69 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.740417 | 105319.02 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.384667 | 1548083.38 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.447041 | 1430880.12 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 15.542625 | 1942828.12 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 15.447208 | 1930901.00 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.116666 | 158072.91 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 9.740750 | 152199.22 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.178125 | 96533.20 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.134375 | 111474.61 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.940291 | 867536.38 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.794125 | 1099265.62 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 15.040834 | 1880104.25 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 21.763416 | 2720427.00 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.457292 | 179020.19 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 7.965750 | 124464.84 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.415916 | 147123.69 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 16.644209 | 2080526.12 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 29.352000 | 3669000.00 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 20.796666 | 2599583.25 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.650375 | 150787.11 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.306000 | 129781.25 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 9.509625 | 148587.89 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 15.902541 | 1987817.62 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 16.995167 | 2124395.88 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 17.689250 | 2211156.25 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 3.946291 | 61660.80 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.491416 | 7678.38 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.741958 | 11593.09 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.031459 | 16116.55 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 1.949583 | 243697.88 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.483083 | 60385.38 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 1.143708 | 142963.50 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 1.380125 | 172515.62 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 3.284833 | 51325.52 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.871750 | 60496.09 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.067208 | 32300.12 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.618084 | 25282.56 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.873792 | 359224.00 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.159125 | 144890.62 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.807375 | 225921.88 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.068667 | 258583.38 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.580209 | 102815.77 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 1.766250 | 27597.66 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.640667 | 25635.42 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.418291 | 37785.80 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.473708 | 684213.50 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 2.761834 | 345229.25 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.609083 | 326135.38 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 2.654542 | 331817.75 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.358917 | 99358.08 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 12.126333 | 189473.95 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.792625 | 28009.77 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.641000 | 41265.62 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.888916 | 486114.50 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.529500 | 566187.50 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.041792 | 630224.00 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.313125 | 664140.62 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.144000 | 96000.00 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.903084 | 76610.69 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.124000 | 48812.50 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.118958 | 48733.72 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.627000 | 1453375.00 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.020042 | 1252505.25 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.154583 | 769322.88 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.146375 | 768296.88 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.351750 | 99246.09 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 8.542833 | 133481.77 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.990541 | 77977.20 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.549709 | 71089.20 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.854834 | 981854.25 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.771666 | 971458.25 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.298167 | 662270.88 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 6.088833 | 761104.12 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.962583 | 186915.36 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.001584 | 78149.75 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.223083 | 50360.67 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.257667 | 50901.05 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.441416 | 1430177.00 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.082958 | 1260369.75 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.365000 | 795625.00 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.487209 | 810901.12 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.485458 | 163835.28 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 9.116833 | 142450.52 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 12.630250 | 197347.66 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6.620333 | 103442.70 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.397250 | 674656.25 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 9.314291 | 1164286.38 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.553208 | 694151.00 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 6.044667 | 755583.38 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.798416 | 153100.25 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.347834 | 99184.91 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.657167 | 88393.23 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.580875 | 102826.17 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.280166 | 1410020.75 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 13.520917 | 1690114.62 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 14.702458 | 1837807.25 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 16.642667 | 2080333.38 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.408375 | 147005.86 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 9.812500 | 153320.31 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.408416 | 100131.50 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 9.095333 | 142114.58 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.142250 | 1017781.25 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 10.005375 | 1250671.88 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 16.092417 | 2011552.13 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 15.239958 | 1904994.75 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 19.492833 | 304575.52 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.760375 | 277505.86 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 14.046875 | 219482.42 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 14.054709 | 219604.83 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 52.620958 | 6577619.75 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 35.667791 | 4458473.88 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 31.810042 | 3976255.25 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 31.679500 | 3959937.50 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 25.279375 | 394990.23 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 18.636833 | 291200.52 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 19.326000 | 301968.75 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 15.864167 | 247877.61 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 28.667625 | 3583453.12 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 28.371583 | 3546447.88 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 20.294333 | 2536791.62 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 20.583916 | 2572989.50 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.930292 | 170785.81 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.581292 | 118457.69 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 33.835542 | 528680.34 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 41.441125 | 647517.58 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.485125 | 1685640.62 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 18.837750 | 2354718.75 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 88.640417 | 11080052.12 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 102.341750 | 12792718.75 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.095208 | 126487.62 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 16.304625 | 254759.77 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 51.699292 | 807801.44 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 65.289834 | 1020153.66 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.252750 | 781593.75 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 13.081834 | 1635229.25 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 69.990917 | 8748864.62 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 92.328167 | 11541020.88 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 4.748042 | 296752.62 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 4.007625 | 250476.56 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 69.074667 | 4317166.69 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 74.557708 | 4659856.75 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 5.552750 | 347046.88 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 3.639833 | 227489.56 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 65.794625 | 4112164.06 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 48.288500 | 3018031.25 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.013292 | 830.75 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.001333 | 83.31 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.012208 | 763.00 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.001375 | 85.94 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000084 | 5.25 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.012375 | 773.44 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.045209 | 2825.56 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000125 | 7.81 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.011666 | 729.12 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.050458 | 3153.62 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000125 | 7.81 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.011708 | 731.75 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.001833 | 114.56 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.009833 | 614.56 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.005250 | 328.12 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.011708 | 731.75 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002417 | 151.06 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000084 | 5.25 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.010417 | 651.06 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003916 | 244.75 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `jax-cpu` is 49.7x faster than the slowest successful cell (`julia-base`, 78.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `pytorch-cpu` is 18.7x faster than the slowest successful cell (`julia-base`, 78.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-direct` is 18.4x faster than the slowest successful cell (`julia-base`, 78.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-trace` is 18.9x faster than the slowest successful cell (`julia-base`, 78.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `jax-cpu` is 15.7x faster than the slowest successful cell (`julia-base`, 480.483 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `pytorch-cpu` is 39.5x faster than the slowest successful cell (`julia-base`, 480.483 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-direct` is 38.5x faster than the slowest successful cell (`julia-base`, 480.483 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-trace` is 41.7x faster than the slowest successful cell (`julia-base`, 480.483 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`julia-base`, 11.265 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 11.7x faster than the slowest successful cell (`julia-base`, 152.814 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 19.8x faster than the slowest successful cell (`julia-base`, 152.814 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-direct` is 15.0x faster than the slowest successful cell (`julia-base`, 152.814 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-trace` is 17.0x faster than the slowest successful cell (`julia-base`, 152.814 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 52.4x faster than the slowest successful cell (`julia-base`, 140.340 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 23.4x faster than the slowest successful cell (`julia-base`, 140.340 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-direct` is 21.2x faster than the slowest successful cell (`julia-base`, 140.340 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-trace` is 22.2x faster than the slowest successful cell (`julia-base`, 140.340 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `jax-cpu` is 78.1x faster than the slowest successful cell (`julia-base`, 639.758 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `pytorch-cpu` is 80.7x faster than the slowest successful cell (`julia-base`, 639.758 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-direct` is 78.4x faster than the slowest successful cell (`julia-base`, 639.758 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-trace` is 78.5x faster than the slowest successful cell (`julia-base`, 639.758 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 22.6x faster than the slowest successful cell (`pytorch-cpu`, 20.254 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 24.0x faster than the slowest successful cell (`pytorch-cpu`, 20.254 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 21.2x faster than the slowest successful cell (`pytorch-cpu`, 20.254 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 16.5x faster than the slowest successful cell (`pytorch-cpu`, 13.748 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 15.5x faster than the slowest successful cell (`pytorch-cpu`, 13.748 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 15.0x faster than the slowest successful cell (`pytorch-cpu`, 13.748 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 19.2x faster than the slowest successful cell (`pytorch-cpu`, 17.271 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 17.0x faster than the slowest successful cell (`pytorch-cpu`, 17.271 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 18.4x faster than the slowest successful cell (`pytorch-cpu`, 17.271 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 12.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 14.1x faster than the slowest successful cell (`pytorch-cpu`, 12.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sqrt` (f64, threads=1, shape=`33554432`): `pytorch-cpu` is 14.7x faster than the slowest successful cell (`julia-base`, 201.405 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sqrt` (f64, threads=1, shape=`33554432`): `tenferro-direct` is 12.3x faster than the slowest successful cell (`julia-base`, 201.405 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sqrt` (f64, threads=1, shape=`33554432`): `tenferro-trace` is 14.0x faster than the slowest successful cell (`julia-base`, 201.405 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 1106.4x faster than the slowest successful cell (`julia-base`, 1.614 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 1469.2x faster than the slowest successful cell (`tenferro-direct`, 2.142 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 31.8x faster than the slowest successful cell (`jax-cpu`, 0.026 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x2x2`): `tenferro-trace` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.026 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 14.0x faster than the slowest successful cell (`jax-cpu`, 0.027 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.008 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x8x8`): `tenferro-direct` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.011 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 23.4x faster than the slowest successful cell (`jax-cpu`, 0.038 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 11.7x faster than the slowest successful cell (`jax-cpu`, 0.038 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x32x32`): `pytorch-cpu` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x32x32`): `tenferro-direct` is 15.4x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x32x32`): `tenferro-trace` is 13.9x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 27.3x faster than the slowest successful cell (`jax-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-trace` is 16.0x faster than the slowest successful cell (`jax-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 51.1x faster than the slowest successful cell (`pytorch-cpu`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 23.1x faster than the slowest successful cell (`pytorch-cpu`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 26.0x faster than the slowest successful cell (`pytorch-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 35.8x faster than the slowest successful cell (`pytorch-cpu`, 0.073 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 20.9x faster than the slowest successful cell (`pytorch-cpu`, 0.073 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 23.4x faster than the slowest successful cell (`pytorch-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 18.1x faster than the slowest successful cell (`pytorch-cpu`, 0.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 19.7x faster than the slowest successful cell (`pytorch-cpu`, 0.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 17.4x faster than the slowest successful cell (`pytorch-cpu`, 0.034 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 30.9x faster than the slowest successful cell (`pytorch-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 19.7x faster than the slowest successful cell (`pytorch-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 35.7x faster than the slowest successful cell (`pytorch-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `jax-cpu` is 10.4x faster than the slowest successful cell (`pytorch-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 42.7x faster than the slowest successful cell (`pytorch-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 22.9x faster than the slowest successful cell (`pytorch-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 19.8x faster than the slowest successful cell (`pytorch-cpu`, 0.055 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.055 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `jax-cpu` is 10.0x faster than the slowest successful cell (`pytorch-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 38.1x faster than the slowest successful cell (`pytorch-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 19.6x faster than the slowest successful cell (`pytorch-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 44.7x faster than the slowest successful cell (`pytorch-cpu`, 0.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 23.5x faster than the slowest successful cell (`pytorch-cpu`, 0.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 21.4x faster than the slowest successful cell (`pytorch-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 14.8x faster than the slowest successful cell (`pytorch-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=1, shape=`1x2x2`): `tenferro-direct` is 17.4x faster than the slowest successful cell (`jax-cpu`, 0.038 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 13.4x faster than the slowest successful cell (`jax-cpu`, 0.027 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`1x64x24`): `tenferro-direct` is 12.5x faster than the slowest successful cell (`jax-cpu`, 0.105 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 22.2x faster than the slowest successful cell (`jax-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-trace` is 13.4x faster than the slowest successful cell (`jax-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 25.2x faster than the slowest successful cell (`jax-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 15.7x faster than the slowest successful cell (`jax-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 15.1x faster than the slowest successful cell (`jax-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 22.3x faster than the slowest successful cell (`pytorch-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 17.0x faster than the slowest successful cell (`pytorch-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 11.0x faster than the slowest successful cell (`jax-cpu`, 0.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x32x32`): `tenferro-trace` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 19.1x faster than the slowest successful cell (`pytorch-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 15.1x faster than the slowest successful cell (`pytorch-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 25.7x faster than the slowest successful cell (`pytorch-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 21.8x faster than the slowest successful cell (`pytorch-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `jax-cpu` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 25.4x faster than the slowest successful cell (`pytorch-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 29.7x faster than the slowest successful cell (`jax-cpu`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 26.6x faster than the slowest successful cell (`jax-cpu`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 24.4x faster than the slowest successful cell (`pytorch-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 19.4x faster than the slowest successful cell (`pytorch-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 21.5x faster than the slowest successful cell (`pytorch-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 15.6x faster than the slowest successful cell (`jax-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 14.8x faster than the slowest successful cell (`jax-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 19.9x faster than the slowest successful cell (`pytorch-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 18.2x faster than the slowest successful cell (`pytorch-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 12.3x faster than the slowest successful cell (`pytorch-cpu`, 0.073 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 12.8x faster than the slowest successful cell (`pytorch-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 14.9x faster than the slowest successful cell (`jax-cpu`, 0.038 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`4x2x2`): `tenferro-direct` is 13.7x faster than the slowest successful cell (`jax-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 15.8x faster than the slowest successful cell (`jax-cpu`, 0.034 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 0.037 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x64x24`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 0.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 18.2x faster than the slowest successful cell (`pytorch-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 13.9x faster than the slowest successful cell (`pytorch-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x32x32,rhs=1`): `tenferro-direct` is 23.9x faster than the slowest successful cell (`jax-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x32x32,rhs=1`): `tenferro-trace` is 13.2x faster than the slowest successful cell (`jax-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`8x2x2,rhs=1`): `tenferro-direct` is 15.1x faster than the slowest successful cell (`jax-cpu`, 0.034 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x32x32,rhs=1`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x32x32,rhs=1`): `tenferro-direct` is 28.7x faster than the slowest successful cell (`jax-cpu`, 0.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x32x32,rhs=1`): `tenferro-trace` is 14.7x faster than the slowest successful cell (`jax-cpu`, 0.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x2x2,rhs=1`): `tenferro-direct` is 16.9x faster than the slowest successful cell (`pytorch-cpu`, 0.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x32x32,rhs=1`): `tenferro-direct` is 16.3x faster than the slowest successful cell (`jax-cpu`, 0.447 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x32x32,rhs=1`): `tenferro-trace` is 13.2x faster than the slowest successful cell (`jax-cpu`, 0.447 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x4x4,rhs=1`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.038 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (c64, threads=1, shape=`3x2x2`): `tenferro-direct` is 12.1x faster than the slowest successful cell (`jax-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (c64, threads=4, shape=`1x2x2`): `pytorch-cpu` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (c64, threads=4, shape=`1x2x2`): `tenferro-direct` is 19.9x faster than the slowest successful cell (`jax-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (c64, threads=4, shape=`1x2x2`): `tenferro-trace` is 14.1x faster than the slowest successful cell (`jax-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svdvals` (f64, threads=1, shape=`1024x2x2`): `pytorch-cpu` is 10.7x faster than the slowest successful cell (`jax-cpu`, 2.350 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svdvals` (f64, threads=1, shape=`1024x2x2`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`jax-cpu`, 2.350 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svdvals` (f64, threads=1, shape=`3x2x2`): `tenferro-direct` is 22.5x faster than the slowest successful cell (`jax-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svdvals` (f64, threads=4, shape=`1024x2x2`): `pytorch-cpu` is 15.1x faster than the slowest successful cell (`jax-cpu`, 3.298 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svdvals` (f64, threads=4, shape=`1024x2x2`): `tenferro-direct` is 15.9x faster than the slowest successful cell (`jax-cpu`, 3.298 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svdvals` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.064 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svdvals` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 16.1x faster than the slowest successful cell (`jax-cpu`, 0.055 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `jax-cpu` is 20.3x faster than the slowest successful cell (`julia-base`, 362.448 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `pytorch-cpu` is 15.8x faster than the slowest successful cell (`julia-base`, 362.448 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-direct` is 33.2x faster than the slowest successful cell (`julia-base`, 362.448 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-trace` is 33.7x faster than the slowest successful cell (`julia-base`, 362.448 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 26.4x faster than the slowest successful cell (`julia-base`, 266.426 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 30.7x faster than the slowest successful cell (`julia-base`, 266.426 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 34.2x faster than the slowest successful cell (`julia-base`, 266.426 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 27.3x faster than the slowest successful cell (`julia-base`, 266.426 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `jax-cpu` is 27.3x faster than the slowest successful cell (`julia-base`, 317.879 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `pytorch-cpu` is 44.6x faster than the slowest successful cell (`julia-base`, 317.879 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-direct` is 38.2x faster than the slowest successful cell (`julia-base`, 317.879 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-trace` is 46.3x faster than the slowest successful cell (`julia-base`, 317.879 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `jax-cpu` is 23.1x faster than the slowest successful cell (`julia-base`, 265.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `pytorch-cpu` is 33.5x faster than the slowest successful cell (`julia-base`, 265.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-direct` is 34.0x faster than the slowest successful cell (`julia-base`, 265.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-trace` is 35.4x faster than the slowest successful cell (`julia-base`, 265.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `jax-cpu` is 16.9x faster than the slowest successful cell (`julia-base`, 682.908 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 29.2x faster than the slowest successful cell (`julia-base`, 682.908 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-direct` is 32.9x faster than the slowest successful cell (`julia-base`, 682.908 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-trace` is 32.7x faster than the slowest successful cell (`julia-base`, 682.908 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 37.2x faster than the slowest successful cell (`julia-base`, 365.899 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 33.3x faster than the slowest successful cell (`julia-base`, 365.899 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-direct` is 40.6x faster than the slowest successful cell (`julia-base`, 365.899 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-trace` is 40.0x faster than the slowest successful cell (`julia-base`, 365.899 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `jax-cpu` is 10.4x faster than the slowest successful cell (`julia-base`, 392.130 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 63.3x faster than the slowest successful cell (`julia-base`, 392.130 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-eager` is 31.1x faster than the slowest successful cell (`julia-base`, 392.130 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-trace` is 35.4x faster than the slowest successful cell (`julia-base`, 392.130 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 22.7x faster than the slowest successful cell (`julia-base`, 251.830 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 24.1x faster than the slowest successful cell (`julia-base`, 251.830 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 24.9x faster than the slowest successful cell (`julia-base`, 251.830 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 28.3x faster than the slowest successful cell (`julia-base`, 251.830 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 12.0x faster than the slowest successful cell (`jax-cpu`, 7.656 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `jax-cpu` is 79.3x faster than the slowest successful cell (`julia-base`, 1151.584 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 78.2x faster than the slowest successful cell (`julia-base`, 1151.584 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 89.8x faster than the slowest successful cell (`julia-base`, 1151.584 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 92.8x faster than the slowest successful cell (`julia-base`, 1151.584 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `jax-cpu` is 70.7x faster than the slowest successful cell (`julia-base`, 1008.961 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 82.9x faster than the slowest successful cell (`julia-base`, 1008.961 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 82.3x faster than the slowest successful cell (`julia-base`, 1008.961 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 86.8x faster than the slowest successful cell (`julia-base`, 1008.961 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 41.1x faster than the slowest successful cell (`julia-base`, 401.582 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 42.1x faster than the slowest successful cell (`julia-base`, 401.582 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 50.6x faster than the slowest successful cell (`julia-base`, 401.582 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 44.9x faster than the slowest successful cell (`julia-base`, 401.582 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 42.6x faster than the slowest successful cell (`julia-base`, 1954.307 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 44.0x faster than the slowest successful cell (`julia-base`, 1954.307 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-eager` is 42.2x faster than the slowest successful cell (`julia-base`, 1954.307 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-trace` is 41.9x faster than the slowest successful cell (`julia-base`, 1954.307 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 14.8x faster than the slowest successful cell (`tenferro-eager`, 11.541 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 7201.2x faster than the slowest successful cell (`julia-base`, 13.502 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 5529.2x faster than the slowest successful cell (`tenferro-direct`, 19.817 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 10015.8x faster than the slowest successful cell (`julia-base`, 15.024 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 9259.3x faster than the slowest successful cell (`julia-base`, 16.583 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 149670.6x faster than the slowest successful cell (`julia-base`, 230.642 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `strided-jl` is 11.7x faster than the slowest successful cell (`julia-base`, 245.319 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 154970.9x faster than the slowest successful cell (`julia-base`, 245.319 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 10.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 565.2x faster than the slowest successful cell (`tenferro-direct`, 0.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 394.3x faster than the slowest successful cell (`tenferro-direct`, 0.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 91.5x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 123.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 146.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 130.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 11493.5 GB/s at 0.001 ms. The conservative physical lower bound is 0.168 ms, so the cell is 114.9x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 11501.4 GB/s at 0.001 ms. The conservative physical lower bound is 0.042 ms, so the cell is 28.8x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 143165.6 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1431.7x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 74898.3 GB/s at 0.004 ms. The conservative physical lower bound is 0.671 ms, so the cell is 187.2x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 178957.0 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1789.6x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 149880.2 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 374.7x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 87097.8 GB/s at 0.002 ms. The conservative physical lower bound is 1.342 ms, so the cell is 871.0x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 84786.9 GB/s at 0.002 ms. The conservative physical lower bound is 0.336 ms, so the cell is 212.0x faster than the bound.

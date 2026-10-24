# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20261024_181407/run.yaml`
- Timestamp: `20261024_181407`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Sampling: `quick` profile, 7 measured runs, 3 warmups per row

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20261024_181407`.

- tenferro-rs commit: `fa39e8d78d9c58e36575084b67448a8112ec1d6c`

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
- Python platform: `macOS-26.6.2-arm64-arm-64bit-Mach-O`

## Thread Environments

### Threads: 1

- Run metadata: `data/results/mac-cpu/cpu/public_api/20261024_181407/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/public_api/20261024_181407/run_t4.yaml`
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

- CSV: `data/results/mac-cpu/cpu/public_api/20261024_181407/cpu_public_api_t1_20261024_181407.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20261024_181407/cpu_public_api_t4_20261024_181407.csv`
- Source table: `data/results/mac-cpu/cpu/public_api/20261024_181407/cpu_public_api_20261024_181407.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 4.011 ± 0.158 | - | 4.196 ± 0.197 | 4.312 ± 0.263 | 1.325 ± 0.025 | 6.727 ± 2.901 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 3.244 ± 0.134 | - | 3.458 ± 0.859 | 3.348 ± 0.033 | 1.184 ± 0.027 | 75.337 ± 11.235 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 14.081 ± 1.301 | - | 11.783 ± 2.126 | 11.737 ± 0.823 | 11.929 ± 0.687 | 13.824 ± 1.197 | 10.856 ± 2.733 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 10.641 ± 1.232 | - | 9.388 ± 0.448 | 9.239 ± 0.138 | 9.118 ± 0.059 | 17.407 ± 6.588 | 13.761 ± 1.288 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 46.018 ± 2.056 | - | 44.923 ± 3.130 | 19.771 ± 0.283 | 33.078 ± 1.051 | 42.735 ± 0.222 | 46.678 ± 0.283 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 19.857 ± 0.067 | - | 19.869 ± 0.058 | 8.671 ± 0.683 | 16.785 ± 0.109 | 54.071 ± 7.013 | 23.777 ± 4.888 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 10.263 ± 0.947 | - | 11.094 ± 1.389 | 15.030 ± 5.195 | 50.557 ± 5.680 | 62.916 ± 4.752 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 10.077 ± 0.205 | - | 10.139 ± 0.562 | 8.605 ± 0.241 | 17.249 ± 0.472 | 48.302 ± 12.034 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 11.416 ± 0.938 | - | unsupported | 13.012 ± 4.107 | 52.343 ± 8.245 | 65.274 ± 12.941 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 9.516 ± 0.319 | - | unsupported | 8.985 ± 0.148 | 17.039 ± 1.314 | 31.994 ± 7.993 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 11.543 ± 0.353 | - | 12.510 ± 0.577 | 16.433 ± 6.022 | 20.957 ± 0.797 | 8.069 ± 1.177 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 10.088 ± 0.222 | - | 10.100 ± 0.192 | 10.138 ± 0.048 | 20.560 ± 0.359 | 501.867 ± 377.060 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 31.408 ± 0.738 | - | 31.320 ± 0.117 | 37.055 ± 2.779 | 43.971 ± 3.684 | 46.669 ± 2.190 | 43.709 ± 2.535 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 16.317 ± 4.588 | - | 16.341 ± 0.415 | 13.311 ± 2.168 | 20.518 ± 0.170 | 51.384 ± 6.576 | 18.727 ± 4.433 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 46.253 ± 3.723 | - | 46.203 ± 3.153 | 51.144 ± 11.381 | 78.511 ± 2.521 | 66.260 ± 3.698 | 67.397 ± 0.896 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 20.445 ± 0.138 | - | 22.772 ± 2.078 | 20.883 ± 1.979 | 32.690 ± 10.606 | 78.493 ± 2.461 | 33.741 ± 18.732 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 9.718 ± 2.337 | - | 9.481 ± 0.620 | 8.779 ± 0.765 | 8.483 ± 0.663 | 9.327 ± 3.677 | 8.000 ± 0.651 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 9.351 ± 0.641 | - | 7.935 ± 0.764 | 7.171 ± 0.064 | 7.171 ± 0.020 | 9.235 ± 0.286 | 15.813 ± 7.505 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.219 ± 0.363 | - | 7.687 ± 0.373 | 6.549 ± 0.583 | 5.427 ± 0.377 | 8.919 ± 0.898 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 0.899 ± 0.032 | - | 3.499 ± 0.057 | 2.910 ± 0.023 | 2.962 ± 0.413 | 12.346 ± 3.207 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 8.238 ± 0.184 | - | 8.336 ± 0.253 | 9.209 ± 3.351 | 7.466 ± 0.335 | 8.643 ± 4.010 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 7.422 ± 1.185 | - | 7.234 ± 0.187 | 7.207 ± 0.041 | 5.872 ± 0.021 | 154.765 ± 52.075 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 5.512 ± 0.479 | - | 5.657 ± 0.323 | 5.277 ± 1.365 | 2.049 ± 0.039 | 5.495 ± 1.774 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 4.619 ± 0.145 | - | 4.708 ± 0.270 | 4.672 ± 0.052 | 2.038 ± 0.028 | 117.880 ± 14.264 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 7.545 ± 0.267 | - | 7.525 ± 0.546 | 8.913 ± 1.365 | 6.385 ± 0.492 | 6.953 ± 0.446 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 6.807 ± 0.395 | - | 6.760 ± 0.046 | 6.770 ± 0.015 | 5.802 ± 0.082 | 618.355 ± 28.864 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 10.920 ± 0.457 | - | 10.252 ± 0.904 | 11.638 ± 2.959 | 57.112 ± 18.635 | 58.423 ± 5.998 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 9.041 ± 0.996 | - | 8.562 ± 0.345 | 8.698 ± 0.478 | 16.268 ± 0.251 | 36.410 ± 10.777 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 8.211 ± 0.401 | - | unsupported | 9.716 ± 0.327 | 88.353 ± 11.377 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 8.236 ± 0.370 | - | unsupported | 9.892 ± 0.268 | 29.583 ± 4.281 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 10.420 ± 0.173 | - | 10.844 ± 0.476 | 9.700 ± 0.469 | 18.234 ± 3.583 | 12.792 ± 1.229 | 14.954 ± 0.579 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 10.241 ± 0.177 | - | 10.307 ± 0.277 | 10.309 ± 0.939 | 9.221 ± 0.100 | 9.325 ± 0.026 | 9.955 ± 7.042 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 15.679 ± 1.414 | - | 17.092 ± 2.985 | 16.181 ± 1.219 | 27.351 ± 1.843 | 20.615 ± 39.926 | 27.316 ± 12.520 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 14.792 ± 3.304 | - | 14.474 ± 0.242 | 14.786 ± 0.183 | 18.977 ± 1.844 | 14.345 ± 11.627 | 17.613 ± 2.812 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 23.381 ± 0.293 | - | 31.260 ± 0.294 | 35.265 ± 1.703 | 42.087 ± 23.180 | 68.760 ± 18.877 | 72.076 ± 18.053 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 13.164 ± 0.162 | - | 18.513 ± 4.418 | 18.576 ± 2.854 | 13.889 ± 0.606 | 45.094 ± 2.278 | 23.703 ± 2.774 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 4.886 ± 0.384 | - | 4.873 ± 0.079 | 4.812 ± 0.208 | 8.485 ± 0.515 | 5.053 ± 0.185 | 8.335 ± 1.527 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 4.966 ± 0.353 | - | 5.544 ± 0.439 | 5.184 ± 0.108 | 7.028 ± 0.128 | 5.014 ± 0.950 | 6.358 ± 4.867 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 13.502 ± 4.464 | - | 10.661 ± 0.826 | 13.681 ± 0.111 | 16.829 ± 4.864 | 16.680 ± 2.586 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 12.428 ± 0.626 | - | 15.213 ± 1.202 | 11.524 ± 0.569 | 10.771 ± 0.704 | 12.032 ± 0.797 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 28.256 ± 0.076 | - | 29.236 ± 2.016 | 34.265 ± 0.443 | 31.745 ± 2.576 | 42.413 ± 11.116 | 36.712 ± 11.857 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 15.089 ± 3.707 | - | 15.989 ± 2.117 | 13.235 ± 4.920 | 15.473 ± 1.881 | 33.352 ± 4.428 | 13.481 ± 0.682 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 14.132 ± 0.912 | - | 17.396 ± 0.339 | 14.112 ± 0.288 | 29.124 ± 3.651 | 19.804 ± 1.182 | 17.747 ± 5.604 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 15.915 ± 1.972 | - | 16.634 ± 0.689 | 14.997 ± 0.544 | 18.969 ± 0.310 | 14.635 ± 1.347 | 16.333 ± 4.484 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 15.378 ± 0.057 | - | 15.836 ± 1.669 | 24.145 ± 0.115 | 18.939 ± 1.578 | 35.919 ± 1.916 | 36.466 ± 7.991 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 8.289 ± 0.187 | - | 8.470 ± 0.432 | 10.482 ± 2.520 | 8.888 ± 0.240 | 29.990 ± 0.469 | 14.747 ± 4.639 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 24.215 ± 0.881 | - | 25.053 ± 2.426 | 28.107 ± 0.428 | 24.928 ± 7.835 | 27.355 ± 0.468 | 26.961 ± 3.150 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 13.157 ± 0.413 | - | 12.523 ± 0.090 | 11.871 ± 0.773 | 9.839 ± 2.340 | 23.085 ± 1.872 | 10.432 ± 1.010 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 27.508 ± 0.552 | - | 27.487 ± 0.821 | 41.072 ± 3.047 | 29.382 ± 2.788 | 46.195 ± 18.221 | 47.122 ± 1.052 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 14.592 ± 0.674 | - | 15.342 ± 1.699 | 18.167 ± 3.913 | 16.336 ± 4.164 | 38.778 ± 1.191 | 19.408 ± 20.746 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 13.911 ± 0.036 | - | 13.915 ± 0.103 | 21.508 ± 2.564 | 29.179 ± 8.266 | 23.234 ± 4.380 | 23.304 ± 2.728 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 7.350 ± 0.169 | - | 7.344 ± 0.081 | 9.866 ± 0.419 | 8.738 ± 0.426 | 20.475 ± 2.084 | 9.708 ± 2.320 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.270 ± 0.006 | 0.302 ± 0.008 | unsupported | unsupported | 0.290 ± 0.143 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.236 ± 0.009 | 0.268 ± 0.022 | unsupported | unsupported | 0.187 ± 0.004 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 17.708 ± 2.347 | - | 17.685 ± 2.404 | 14.519 ± 0.979 | 26.685 ± 2.437 | 19.678 ± 1.443 | 20.981 ± 0.825 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 17.264 ± 1.513 | - | 16.760 ± 0.555 | 15.201 ± 0.580 | 19.068 ± 1.673 | 14.801 ± 1.006 | 14.836 ± 0.745 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 17.209 ± 2.278 | - | 19.171 ± 1.360 | 14.065 ± 0.721 | 33.787 ± 3.861 | 19.576 ± 4.170 | 22.057 ± 0.748 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 17.761 ± 1.031 | - | 17.609 ± 1.608 | 15.169 ± 0.122 | 19.570 ± 2.466 | 14.604 ± 2.056 | 16.890 ± 0.988 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 14.446 ± 0.584 | - | 14.047 ± 0.071 | 15.544 ± 2.084 | 27.057 ± 3.846 | 18.390 ± 4.928 | 20.863 ± 1.744 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 16.367 ± 1.528 | - | 15.182 ± 0.999 | 15.615 ± 1.044 | 18.870 ± 0.555 | 14.353 ± 0.614 | 14.747 ± 0.494 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 10.697 ± 0.744 | - | 10.764 ± 0.117 | 9.154 ± 0.125 | 17.189 ± 1.361 | 11.278 ± 1.673 | 13.887 ± 3.316 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 10.142 ± 0.434 | - | 10.260 ± 0.587 | 10.185 ± 0.427 | 9.183 ± 0.009 | 9.469 ± 0.168 | 9.880 ± 0.467 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 32.702 ± 0.216 | - | 32.820 ± 0.517 | 64.109 ± 2.969 | 54.581 ± 8.312 | 102.616 ± 42.639 | 84.065 ± 4.976 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 18.069 ± 0.388 | - | 17.643 ± 3.784 | 28.229 ± 3.643 | 18.339 ± 1.234 | 88.369 ± 25.531 | 28.101 ± 3.920 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 6.909 ± 0.029 | - | 6.084 ± 0.233 | 11.077 ± 0.459 | 24.423 ± 1.057 | 6.602 ± 0.659 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 5.317 ± 0.178 | - | 5.257 ± 0.058 | 6.789 ± 0.115 | 10.406 ± 0.088 | 6.287 ± 0.030 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.045 ± 0.087 | - | 1.067 ± 0.129 | 13.452 ± 1.797 | 5.780 ± 2.958 | 0.823 ± 0.088 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.723 ± 0.043 | - | 0.726 ± 0.017 | 10.787 ± 0.506 | 1.163 ± 0.080 | 0.581 ± 0.036 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 0.867 ± 0.089 | - | 0.909 ± 0.075 | 1.419 ± 0.021 | 1.203 ± 0.135 | 0.979 ± 0.069 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 1.294 ± 0.038 | - | 1.508 ± 0.209 | 0.999 ± 0.073 | 0.718 ± 0.074 | 0.677 ± 0.011 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 6.005 ± 0.271 | - | 6.033 ± 0.025 | 10.669 ± 0.244 | 23.986 ± 4.918 | 7.340 ± 1.034 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 5.303 ± 0.146 | - | 6.111 ± 0.457 | 6.793 ± 0.510 | 10.524 ± 0.152 | 5.856 ± 0.634 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 0.887 ± 0.085 | - | 1.049 ± 0.332 | 12.015 ± 0.686 | 3.014 ± 0.432 | 0.887 ± 0.157 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.740 ± 0.042 | - | 0.732 ± 0.031 | 11.062 ± 2.310 | 1.209 ± 0.088 | 0.569 ± 0.002 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 3.301 ± 0.544 | - | 3.337 ± 0.197 | 5.585 ± 0.161 | 4.684 ± 1.016 | 3.527 ± 0.519 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 5.695 ± 0.443 | - | 5.075 ± 0.036 | 3.618 ± 0.115 | 2.459 ± 0.083 | 2.689 ± 0.037 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 5.444 ± 0.126 | - | 6.506 ± 0.542 | 5.957 ± 0.414 | 52.514 ± 4.651 | 6.665 ± 0.281 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 5.234 ± 0.251 | - | 5.370 ± 0.425 | 5.412 ± 0.265 | 19.388 ± 0.081 | 6.847 ± 0.180 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.908 ± 0.060 | - | 0.783 ± 0.047 | 3.443 ± 0.086 | 2.779 ± 0.431 | 0.981 ± 0.154 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.832 ± 0.026 | - | 0.814 ± 0.053 | 2.338 ± 0.073 | 1.505 ± 0.122 | 0.546 ± 0.003 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 0.778 ± 0.055 | - | 0.769 ± 0.076 | 0.635 ± 0.016 | 1.291 ± 0.116 | 1.288 ± 0.600 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 1.372 ± 0.070 | - | 1.347 ± 0.021 | 0.875 ± 0.078 | 0.828 ± 0.083 | 0.672 ± 0.019 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 4.720 ± 0.529 | - | 4.751 ± 0.775 | 5.205 ± 0.822 | 46.180 ± 10.977 | 6.532 ± 0.229 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 6.084 ± 0.653 | - | 6.251 ± 0.183 | 5.126 ± 0.049 | 14.799 ± 0.254 | 7.049 ± 0.651 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.755 ± 0.110 | - | 0.811 ± 0.090 | 3.459 ± 0.084 | 2.805 ± 0.219 | 0.866 ± 0.143 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.695 ± 0.034 | - | 0.733 ± 0.031 | 2.350 ± 0.198 | 1.377 ± 0.087 | 0.568 ± 0.030 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 0.858 ± 0.084 | - | 0.934 ± 0.081 | 0.563 ± 0.025 | 1.195 ± 0.030 | 0.900 ± 0.085 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 1.714 ± 0.042 | - | 1.825 ± 0.262 | 0.700 ± 0.026 | 0.832 ± 0.040 | 0.649 ± 0.018 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.782 ± 0.034 | 0.749 ± 0.023 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.665 ± 0.013 | 0.667 ± 0.033 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 25.668 ± 1.096 | - | 24.969 ± 0.063 | 31.211 ± 3.550 | 46.858 ± 4.848 | 28.098 ± 1.871 | 34.716 ± 3.063 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 16.602 ± 3.518 | - | 14.138 ± 2.170 | 15.201 ± 4.190 | 20.614 ± 1.313 | 24.627 ± 0.881 | 11.511 ± 0.499 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 16.715 ± 0.209 | - | 16.713 ± 0.107 | 17.330 ± 1.448 | 51.245 ± 5.234 | 43.700 ± 5.460 | 49.994 ± 4.157 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 11.825 ± 1.330 | - | 12.048 ± 0.575 | 10.992 ± 1.277 | 17.262 ± 0.378 | 38.388 ± 0.257 | 16.379 ± 3.964 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 19.858 ± 0.169 | - | 19.931 ± 0.438 | 15.145 ± 0.357 | 38.852 ± 10.072 | 28.827 ± 3.021 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 28.575 ± 7.577 | - | 29.211 ± 5.536 | 16.143 ± 1.201 | 21.636 ± 1.788 | 23.908 ± 0.850 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 11.242 ± 0.255 | - | 11.330 ± 0.523 | 9.665 ± 0.643 | 19.034 ± 5.312 | 12.987 ± 10.767 | 21.075 ± 0.449 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 10.223 ± 0.214 | - | 10.297 ± 0.179 | 10.075 ± 0.741 | 10.387 ± 1.741 | 9.914 ± 0.780 | 9.816 ± 0.870 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 21.955 ± 0.517 | - | 21.945 ± 2.670 | 31.362 ± 1.440 | 28.734 ± 1.506 | 36.505 ± 6.765 | 33.515 ± 7.121 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 11.409 ± 1.353 | - | 11.727 ± 0.595 | 11.976 ± 3.711 | 14.809 ± 0.464 | 27.776 ± 2.291 | 14.270 ± 1.177 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 11.021 ± 0.975 | - | 10.726 ± 0.129 | 11.012 ± 0.762 | 28.355 ± 9.384 | 32.074 ± 6.637 | 29.368 ± 6.299 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 10.211 ± 0.376 | - | 10.873 ± 0.602 | 11.278 ± 0.705 | 10.225 ± 0.922 | 23.958 ± 0.604 | 11.749 ± 0.498 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 16.991 ± 1.943 | - | 14.350 ± 1.133 | 18.871 ± 1.550 | 28.935 ± 4.621 | 20.195 ± 3.926 | 20.961 ± 1.112 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 18.861 ± 5.335 | - | 14.769 ± 0.344 | 15.111 ± 0.857 | 18.879 ± 0.217 | 14.520 ± 2.373 | 14.888 ± 0.670 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 62.368 ± 6.981 | - | 61.668 ± 6.678 | 61.525 ± 7.007 | 15.736 ± 0.840 | 41.733 ± 3.894 | 46.979 ± 8.490 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 28.506 ± 2.390 | - | 26.787 ± 1.454 | 24.980 ± 3.428 | 7.192 ± 0.963 | 40.642 ± 1.440 | 15.602 ± 0.672 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 19.954 ± 0.714 | 19.957 ± 1.623 | unsupported | 15.317 ± 1.555 | 32.459 ± 10.895 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 24.852 ± 3.362 | 21.561 ± 1.184 | unsupported | 16.780 ± 0.813 | 20.203 ± 0.371 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.653 ± 0.027 | - | 0.647 ± 0.033 | 0.624 ± 0.020 | 1.102 ± 0.666 | 0.715 ± 0.032 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.562 ± 0.088 | - | 0.595 ± 0.090 | 0.680 ± 0.048 | 0.564 ± 0.002 | 1.765 ± 0.727 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.645 ± 0.057 | - | 0.641 ± 0.047 | 0.621 ± 0.038 | 0.806 ± 0.138 | 0.753 ± 0.090 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.574 ± 0.040 | - | 0.616 ± 0.085 | 0.726 ± 0.086 | 0.654 ± 0.006 | 1.154 ± 0.300 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 0.942 ± 0.049 | - | unsupported | 1.077 ± 0.084 | 1.191 ± 0.997 | 1.144 ± 0.358 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.976 ± 0.165 | - | unsupported | 1.337 ± 0.176 | 0.822 ± 0.023 | 1.507 ± 0.593 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.350 ± 0.044 | - | 0.428 ± 0.025 | 0.265 ± 0.001 | 0.792 ± 0.444 | 1.130 ± 0.062 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.310 ± 0.068 | - | 0.568 ± 0.356 | 0.306 ± 0.007 | 0.229 ± 0.044 | 1.499 ± 3.174 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.607 ± 0.052 | - | 0.632 ± 0.036 | 1.043 ± 0.141 | 0.834 ± 0.066 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.587 ± 0.021 | - | 0.581 ± 0.048 | 1.008 ± 0.098 | 0.676 ± 0.045 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.642 ± 0.029 | - | 0.652 ± 0.049 | 0.648 ± 0.070 | 1.186 ± 0.130 | 1.157 ± 0.229 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.565 ± 0.076 | - | 0.589 ± 0.086 | 0.811 ± 0.114 | 0.669 ± 0.046 | 1.632 ± 0.901 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.562 ± 0.050 | - | 0.727 ± 0.120 | 0.352 ± 0.013 | 0.833 ± 0.242 | 0.359 ± 0.020 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.719 ± 0.089 | - | 0.869 ± 0.287 | 0.451 ± 0.044 | 0.380 ± 0.006 | 1.264 ± 2.301 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.118 ± 0.143 | - | 0.002 ± 0.000 | 1.078 ± 0.058 | 1.592 ± 0.269 | 1.328 ± 0.205 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.982 ± 0.047 | - | 0.002 ± 0.000 | 1.640 ± 0.307 | 1.027 ± 0.023 | 1.954 ± 0.388 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.051 ± 0.000 | - | 0.055 ± 0.000 | 0.112 ± 0.004 | 0.086 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.081 ± 0.000 | - | 0.087 ± 0.000 | 0.151 ± 0.001 | 0.131 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.176 ± 0.001 | - | 0.192 ± 0.001 | 0.261 ± 0.029 | 0.301 ± 0.089 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.023 ± 0.003 | - | 0.033 ± 0.000 | 0.037 ± 0.001 | 0.105 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.002 ± 0.001 | - | 0.005 ± 0.000 | 0.005 ± 0.001 | 0.033 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.001 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.047 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.152 ± 0.000 | - | 0.227 ± 0.012 | 0.243 ± 0.028 | 0.296 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.009 ± 0.001 | - | 0.015 ± 0.000 | 0.016 ± 0.001 | 0.053 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.038 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.051 ± 0.000 | - | 0.054 ± 0.000 | 0.125 ± 0.008 | 0.101 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.081 ± 0.005 | - | 0.087 ± 0.000 | 0.165 ± 0.013 | 0.150 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.176 ± 0.001 | - | 0.223 ± 0.004 | 0.260 ± 0.014 | 0.252 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.023 ± 0.001 | - | 0.035 ± 0.003 | 0.067 ± 0.008 | 0.079 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.040 ± 0.024 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.035 ± 0.016 | 0.052 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.032 ± 0.002 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.031 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.034 ± 0.021 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.035 ± 0.002 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.021 ± 0.001 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.021 ± 0.001 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.034 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.039 ± 0.008 | 0.054 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.172 ± 0.002 | - | 0.236 ± 0.009 | 0.212 ± 0.029 | 0.192 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.022 ± 0.001 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.010 ± 0.001 | - | 0.016 ± 0.001 | 0.037 ± 0.001 | 0.064 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.032 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.002 ± 0.001 | - | 0.005 ± 0.000 | 0.041 ± 0.016 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.232 ± 0.001 | - | 0.236 ± 0.001 | 0.421 ± 0.034 | 0.823 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.363 ± 0.011 | - | 1.382 ± 0.062 | 1.703 ± 0.145 | 2.256 ± 0.087 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 5.667 ± 0.260 | - | 5.651 ± 0.014 | 6.335 ± 0.172 | 7.051 ± 0.114 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 5.788 ± 0.026 | - | 5.795 ± 0.059 | 3.325 ± 0.216 | 3.967 ± 0.265 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.084 ± 0.002 | - | 0.087 ± 0.000 | 0.088 ± 0.004 | 0.169 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.001 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.002 ± 0.003 | - | 0.006 ± 0.001 | 0.007 ± 0.001 | 0.013 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.017 ± 0.000 | - | 0.020 ± 0.000 | 0.022 ± 0.000 | 0.106 ± 0.069 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.002 ± 0.001 | - | 0.006 ± 0.001 | 0.006 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.023 ± 0.000 | - | 0.025 ± 0.000 | 0.027 ± 0.000 | 0.080 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 46.304 ± 0.688 | - | 46.209 ± 1.011 | 27.417 ± 1.389 | 32.272 ± 0.993 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.002 | 0.008 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.649 ± 0.003 | - | 0.654 ± 0.001 | 0.697 ± 0.058 | 0.869 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.012 ± 0.000 | - | 0.015 ± 0.001 | 0.016 ± 0.000 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.044 ± 0.000 | - | 0.047 ± 0.000 | 0.052 ± 0.001 | 0.104 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.242 ± 0.007 | - | 0.235 ± 0.002 | 0.436 ± 0.035 | 0.866 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 1.364 ± 0.008 | - | 1.369 ± 0.013 | 1.704 ± 0.029 | 1.290 ± 0.365 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 5.637 ± 0.102 | - | 5.717 ± 0.032 | 5.954 ± 0.169 | 3.456 ± 0.111 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 5.795 ± 0.058 | - | 6.292 ± 0.101 | 3.029 ± 0.005 | 3.453 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.084 ± 0.000 | - | 0.087 ± 0.001 | 0.082 ± 0.001 | 0.157 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.004 ± 0.002 | - | 0.007 ± 0.001 | 0.007 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.008 ± 0.001 | - | 0.011 ± 0.003 | 0.010 ± 0.000 | 0.021 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.007 ± 0.001 | - | 0.008 ± 0.002 | 0.009 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.018 ± 0.002 | - | 0.020 ± 0.000 | 0.022 ± 0.002 | 0.084 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.008 ± 0.001 | - | 0.011 ± 0.002 | 0.012 ± 0.003 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.023 ± 0.001 | - | 0.026 ± 0.000 | 0.031 ± 0.003 | 0.082 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 49.890 ± 2.788 | - | 47.884 ± 3.668 | 24.200 ± 0.112 | 14.737 ± 0.203 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.004 ± 0.001 | - | 0.006 ± 0.002 | 0.008 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.681 ± 0.061 | - | 0.657 ± 0.006 | 0.629 ± 0.007 | 0.478 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.015 ± 0.004 | - | 0.015 ± 0.000 | 0.017 ± 0.004 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.045 ± 0.000 | - | 0.048 ± 0.000 | 0.056 ± 0.005 | 0.109 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.220 ± 0.000 | - | 0.224 ± 0.007 | 0.232 ± 0.008 | 0.589 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.136 ± 0.008 | - | 1.181 ± 0.250 | 1.379 ± 0.082 | 1.551 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 4.467 ± 0.047 | - | 4.483 ± 0.067 | 4.776 ± 0.290 | 5.695 ± 0.729 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 0.824 ± 0.004 | - | 0.866 ± 0.019 | 0.980 ± 0.057 | 1.494 ± 0.169 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.002 ± 0.002 | - | 0.005 ± 0.001 | 0.004 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.058 ± 0.000 | - | 0.061 ± 0.001 | 0.066 ± 0.007 | 0.119 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.006 ± 0.002 | - | 0.008 ± 0.000 | 0.011 ± 0.002 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.002 ± 0.006 | - | 0.005 ± 0.001 | 0.004 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.005 ± 0.001 | - | 0.007 ± 0.002 | 0.008 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.014 ± 0.000 | - | 0.017 ± 0.000 | 0.021 ± 0.004 | 0.036 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.007 ± 0.001 | - | 0.008 ± 0.000 | 0.011 ± 0.002 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.026 ± 0.006 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 6.595 ± 0.099 | - | 6.866 ± 0.025 | 7.512 ± 0.310 | 9.722 ± 2.197 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.001 | 0.006 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.452 ± 0.000 | - | 0.457 ± 0.010 | 0.514 ± 0.024 | 0.615 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.010 ± 0.003 | - | 0.012 ± 0.000 | 0.015 ± 0.002 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.036 ± 0.000 | - | 0.039 ± 0.000 | 0.041 ± 0.002 | 0.110 ± 0.077 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.220 ± 0.000 | - | 0.224 ± 0.002 | 0.244 ± 0.025 | 0.516 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.141 ± 0.016 | - | 1.142 ± 0.011 | 1.264 ± 0.059 | 0.961 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 4.451 ± 0.022 | - | 4.530 ± 0.132 | 5.110 ± 0.085 | 2.595 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 0.824 ± 0.006 | - | 0.876 ± 0.015 | 1.009 ± 0.092 | 0.955 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.058 ± 0.000 | - | 0.061 ± 0.002 | 0.086 ± 0.013 | 0.126 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.006 ± 0.000 | - | 0.009 ± 0.001 | 0.008 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.001 | 0.004 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.002 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.015 ± 0.003 | - | 0.019 ± 0.004 | 0.021 ± 0.001 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.007 ± 0.001 | - | 0.010 ± 0.002 | 0.009 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.021 ± 0.003 | - | 0.029 ± 0.004 | 0.022 ± 0.001 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 6.561 ± 0.013 | - | 6.999 ± 0.190 | 7.904 ± 0.120 | 3.887 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.001 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.453 ± 0.006 | - | 0.458 ± 0.000 | 0.514 ± 0.073 | 0.387 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.012 ± 0.001 | - | 0.012 ± 0.003 | 0.013 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.037 ± 0.004 | - | 0.054 ± 0.007 | 0.040 ± 0.001 | 0.098 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.140 ± 0.000 | - | 0.146 ± 0.000 | 0.121 ± 0.010 | 0.665 ± 0.111 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 0.685 ± 0.015 | - | 0.695 ± 0.002 | 0.717 ± 0.046 | 1.540 ± 0.342 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 2.201 ± 0.009 | - | 2.218 ± 0.012 | 2.302 ± 0.165 | 5.567 ± 0.294 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.467 ± 0.003 | - | 0.475 ± 0.001 | 0.472 ± 0.016 | 0.943 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.001 ± 0.001 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.026 ± 0.001 | - | 0.031 ± 0.001 | 0.029 ± 0.002 | 0.102 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.002 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.008 ± 0.002 | 0.007 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.002 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.003 ± 0.001 | - | 0.008 ± 0.002 | 0.007 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.009 ± 0.002 | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.027 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.008 ± 0.002 | 0.005 ± 0.000 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.009 ± 0.002 | 0.007 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.011 ± 0.003 | - | 0.015 ± 0.003 | 0.013 ± 0.000 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 3.722 ± 0.018 | - | 3.770 ± 0.009 | 3.726 ± 0.042 | 7.330 ± 0.218 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.007 ± 0.002 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.193 ± 0.001 | - | 0.200 ± 0.001 | 0.204 ± 0.001 | 0.561 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.008 ± 0.001 | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.018 ± 0.000 | - | 0.023 ± 0.000 | 0.022 ± 0.000 | 0.089 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.140 ± 0.000 | - | 0.146 ± 0.000 | 0.117 ± 0.000 | 0.528 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 0.684 ± 0.006 | - | 0.694 ± 0.004 | 0.680 ± 0.005 | 0.943 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 2.195 ± 0.015 | - | 2.223 ± 0.015 | 2.217 ± 0.013 | 2.586 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.412 ± 0.002 | - | 0.417 ± 0.001 | 0.420 ± 0.011 | 0.949 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.003 ± 0.003 | - | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.023 ± 0.000 | - | 0.026 ± 0.000 | 0.028 ± 0.002 | 0.116 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.004 ± 0.001 | - | 0.008 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.004 ± 0.001 | - | 0.008 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.009 ± 0.001 | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.004 ± 0.001 | - | 0.009 ± 0.002 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.011 ± 0.002 | - | 0.015 ± 0.001 | 0.013 ± 0.001 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 3.429 ± 0.070 | - | 3.514 ± 0.176 | 3.344 ± 0.097 | 3.872 ± 0.310 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.007 ± 0.002 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.176 ± 0.001 | - | 0.181 ± 0.000 | 0.184 ± 0.010 | 0.322 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.009 ± 0.002 | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.019 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.018 ± 0.003 | - | 0.023 ± 0.000 | 0.022 ± 0.000 | 0.092 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.084 ± 0.000 | - | 0.087 ± 0.001 | 0.153 ± 0.009 | 0.134 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.191 ± 0.002 | - | 0.196 ± 0.001 | 0.277 ± 0.072 | 0.206 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.390 ± 0.001 | - | 0.403 ± 0.000 | 0.532 ± 0.105 | 0.408 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.052 ± 0.001 | - | 0.058 ± 0.000 | 0.068 ± 0.019 | 0.116 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.003 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.039 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.009 ± 0.002 | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.045 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.008 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.001 | 0.008 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.001 | 0.008 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.001 | 0.009 ± 0.000 | 0.042 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.008 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.040 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.401 ± 0.002 | - | 0.421 ± 0.002 | 0.502 ± 0.084 | 0.551 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.029 ± 0.000 | - | 0.033 ± 0.000 | 0.041 ± 0.010 | 0.080 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.001 | 0.011 ± 0.083 | 0.039 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.005 ± 0.001 | - | 0.007 ± 0.002 | 0.011 ± 0.002 | 0.040 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.083 ± 0.004 | - | 0.087 ± 0.000 | 0.184 ± 0.005 | 0.158 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.191 ± 0.006 | - | 0.196 ± 0.001 | 0.227 ± 0.004 | 0.280 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.390 ± 0.002 | - | 0.441 ± 0.007 | 0.422 ± 0.006 | 0.343 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.051 ± 0.000 | - | 0.057 ± 0.000 | 0.109 ± 0.008 | 0.128 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.050 ± 0.018 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.006 ± 0.000 | - | 0.008 ± 0.002 | 0.057 ± 0.006 | 0.062 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.001 | 0.048 ± 0.006 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.010 ± 0.001 | - | 0.011 ± 0.001 | 0.055 ± 0.006 | 0.065 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.050 ± 0.002 | 0.021 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.050 ± 0.004 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.033 ± 0.018 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.001 | 0.033 ± 0.001 | 0.068 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.001 | 0.050 ± 0.002 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.049 ± 0.006 | 0.015 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.004 ± 0.000 | - | 0.007 ± 0.001 | 0.053 ± 0.004 | 0.068 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.399 ± 0.001 | - | 0.458 ± 0.008 | 0.377 ± 0.009 | 0.497 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.065 ± 0.012 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.029 ± 0.000 | - | 0.033 ± 0.001 | 0.106 ± 0.003 | 0.122 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.001 | 0.071 ± 0.009 | 0.061 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.005 ± 0.000 | - | 0.009 ± 0.001 | 0.076 ± 0.012 | 0.074 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.181 ± 0.000 | - | 0.184 ± 0.001 | 0.332 ± 0.031 | 0.335 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.454 ± 0.001 | - | 0.460 ± 0.003 | 0.746 ± 0.067 | 0.644 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.126 ± 0.010 | - | 1.142 ± 0.100 | 1.778 ± 0.090 | 1.273 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.275 ± 0.004 | - | 0.283 ± 0.001 | 0.328 ± 0.008 | 0.364 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.012 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.011 ± 0.002 | - | 0.014 ± 0.000 | 0.021 ± 0.002 | 0.049 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.003 ± 0.000 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.014 ± 0.002 | - | 0.017 ± 0.002 | 0.017 ± 0.000 | 0.058 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.004 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.003 ± 0.001 | - | 0.006 ± 0.002 | 0.005 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.006 ± 0.001 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.015 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.001 | 0.004 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.007 ± 0.000 | - | 0.008 ± 0.001 | 0.009 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 2.227 ± 0.091 | - | 2.215 ± 0.016 | 2.610 ± 0.013 | 2.912 ± 0.182 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.003 ± 0.001 | - | 0.006 ± 0.001 | 0.006 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.081 ± 0.003 | - | 0.085 ± 0.000 | 0.151 ± 0.008 | 0.178 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.006 ± 0.001 | - | 0.007 ± 0.002 | 0.008 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.010 ± 0.003 | - | 0.013 ± 0.000 | 0.016 ± 0.000 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.242 ± 0.004 | - | 0.248 ± 0.001 | 0.258 ± 0.032 | 0.324 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.558 ± 0.001 | - | 0.568 ± 0.013 | 0.596 ± 0.065 | 0.590 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.182 ± 0.056 | - | 1.227 ± 0.008 | 1.372 ± 0.047 | 0.726 ± 0.068 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.275 ± 0.013 | - | 0.288 ± 0.008 | 0.388 ± 0.034 | 0.348 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.019 ± 0.011 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.011 ± 0.001 | - | 0.014 ± 0.000 | 0.028 ± 0.003 | 0.070 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.008 ± 0.002 | 0.017 ± 0.003 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.013 ± 0.003 | - | 0.016 ± 0.000 | 0.031 ± 0.016 | 0.073 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.008 ± 0.001 | 0.017 ± 0.003 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.010 ± 0.002 | 0.019 ± 0.023 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.004 ± 0.001 | - | 0.010 ± 0.002 | 0.018 ± 0.003 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.007 ± 0.002 | - | 0.013 ± 0.003 | 0.019 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.003 ± 0.001 | - | 0.008 ± 0.000 | 0.018 ± 0.003 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.019 ± 0.002 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.008 ± 0.001 | - | 0.012 ± 0.002 | 0.026 ± 0.026 | 0.016 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 2.193 ± 0.012 | - | 2.250 ± 0.012 | 2.731 ± 1.631 | 1.408 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.004 ± 0.001 | - | 0.010 ± 0.003 | 0.018 ± 0.005 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.085 ± 0.002 | - | 0.091 ± 0.001 | 0.112 ± 0.002 | 0.140 ± 0.060 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.007 ± 0.001 | - | 0.011 ± 0.000 | 0.023 ± 0.011 | 0.013 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.015 ± 0.003 | - | 0.039 ± 0.010 | 0.028 ± 0.010 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.127 ± 0.001 | - | 0.114 ± 0.002 | 0.105 ± 0.014 | 0.178 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.193 ± 0.029 | - | 0.170 ± 0.002 | 0.181 ± 0.025 | 0.249 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.337 ± 0.001 | - | 0.339 ± 0.034 | 0.333 ± 0.007 | 0.534 ± 0.086 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.043 ± 0.001 | - | 0.050 ± 0.000 | 0.060 ± 0.001 | 0.101 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.007 | 0.009 ± 0.001 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.004 ± 0.001 | - | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.062 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.003 | - | 0.005 ± 0.002 | 0.010 ± 0.002 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.001 | - | 0.005 ± 0.001 | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.002 | 0.010 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.001 | - | 0.005 ± 0.001 | 0.010 ± 0.003 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.006 ± 0.002 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.431 ± 0.012 | - | 0.398 ± 0.034 | 0.374 ± 0.018 | 0.554 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.025 ± 0.000 | - | 0.031 ± 0.000 | 0.036 ± 0.000 | 0.083 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.001 | - | 0.008 ± 0.001 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.005 ± 0.001 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.127 ± 0.001 | - | 0.109 ± 0.002 | 0.116 ± 0.004 | 0.182 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.188 ± 0.001 | - | 0.167 ± 0.001 | 0.153 ± 0.013 | 0.240 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.338 ± 0.019 | - | 0.342 ± 0.011 | 0.281 ± 0.019 | 0.344 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.044 ± 0.008 | - | 0.050 ± 0.000 | 0.062 ± 0.005 | 0.140 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.007 ± 0.007 | 0.011 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.005 ± 0.000 | - | 0.008 ± 0.002 | 0.013 ± 0.001 | 0.061 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.010 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.002 ± 0.001 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.003 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.020 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.001 | 0.012 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.003 ± 0.001 | - | 0.007 ± 0.001 | 0.011 ± 0.002 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.335 ± 0.001 | - | 0.399 ± 0.008 | 0.306 ± 0.018 | 0.445 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.032 ± 0.004 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.025 ± 0.000 | - | 0.031 ± 0.000 | 0.062 ± 0.010 | 0.126 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.003 ± 0.001 | - | 0.006 ± 0.001 | 0.028 ± 0.014 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.008 ± 0.004 | 0.031 ± 0.005 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.064 ± 0.002 | - | 1.082 ± 0.130 | 1.109 ± 0.014 | 2.630 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 4.057 ± 0.790 | - | 3.905 ± 0.007 | 4.030 ± 0.034 | 5.453 ± 0.095 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 13.488 ± 0.302 | - | 13.445 ± 0.691 | 14.806 ± 1.633 | 15.298 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 3.915 ± 0.017 | - | 5.181 ± 0.035 | 5.604 ± 0.228 | 4.553 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.163 ± 0.003 | - | 0.166 ± 0.001 | 0.186 ± 0.005 | 0.231 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.150 ± 0.002 | - | 0.153 ± 0.001 | 0.261 ± 0.086 | 0.238 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.016 ± 0.000 | - | 0.019 ± 0.000 | 0.020 ± 0.001 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.013 ± 0.000 | - | 0.016 ± 0.004 | 0.017 ± 0.000 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.042 ± 0.000 | - | 0.045 ± 0.000 | 0.047 ± 0.000 | 0.058 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.007 ± 0.001 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.017 ± 0.000 | - | 0.020 ± 0.000 | 0.021 ± 0.000 | 0.033 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.057 ± 0.000 | - | 0.060 ± 0.000 | 0.062 ± 0.001 | 0.075 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 31.288 ± 3.751 | - | 42.767 ± 4.392 | 44.172 ± 2.978 | 36.043 ± 0.221 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.010 ± 0.003 | - | 0.013 ± 0.000 | 0.015 ± 0.000 | 0.032 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.293 ± 0.072 | - | 1.268 ± 0.012 | 2.079 ± 0.800 | 1.452 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.032 ± 0.000 | - | 0.035 ± 0.000 | 0.037 ± 0.000 | 0.054 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.109 ± 0.001 | - | 0.112 ± 0.000 | 0.116 ± 0.000 | 0.168 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.067 ± 0.013 | - | 1.068 ± 0.012 | 1.078 ± 0.013 | 2.641 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 3.904 ± 0.046 | - | 3.911 ± 0.016 | 3.885 ± 0.015 | 3.328 ± 1.395 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 13.516 ± 4.913 | - | 13.477 ± 0.088 | 13.462 ± 0.073 | 8.048 ± 0.189 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 3.919 ± 0.039 | - | 5.168 ± 0.019 | 4.422 ± 0.039 | 4.608 ± 0.161 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.163 ± 0.004 | - | 0.166 ± 0.002 | 0.167 ± 0.003 | 0.244 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.007 ± 0.001 | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.035 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.150 ± 0.005 | - | 0.153 ± 0.000 | 0.154 ± 0.002 | 0.228 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.016 ± 0.003 | - | 0.018 ± 0.000 | 0.019 ± 0.001 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.006 ± 0.003 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.015 ± 0.004 | - | 0.016 ± 0.000 | 0.017 ± 0.000 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.042 ± 0.000 | - | 0.045 ± 0.000 | 0.045 ± 0.000 | 0.058 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.008 ± 0.001 | - | 0.010 ± 0.002 | 0.009 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.017 ± 0.004 | - | 0.020 ± 0.000 | 0.020 ± 0.000 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.073 ± 0.007 | - | 0.083 ± 0.007 | 0.060 ± 0.000 | 0.074 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 31.239 ± 0.131 | - | 41.651 ± 0.416 | 35.639 ± 0.222 | 21.376 ± 2.284 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.013 ± 0.003 | - | 0.013 ± 0.000 | 0.013 ± 0.000 | 0.033 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 1.264 ± 0.010 | - | 1.264 ± 0.012 | 1.267 ± 0.011 | 0.846 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.032 ± 0.000 | - | 0.035 ± 0.000 | 0.036 ± 0.000 | 0.054 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.141 ± 0.005 | - | 0.137 ± 0.041 | 0.111 ± 0.000 | 0.185 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.688 ± 0.006 | - | 0.688 ± 0.003 | 0.592 ± 0.024 | 1.764 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 2.726 ± 0.028 | - | 2.724 ± 0.016 | 2.783 ± 0.049 | 3.686 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 10.431 ± 0.176 | - | 9.618 ± 0.034 | 10.376 ± 0.334 | 11.047 ± 0.114 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.323 ± 0.014 | - | 1.341 ± 0.054 | 1.468 ± 0.020 | 1.429 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.006 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.108 ± 0.002 | - | 0.111 ± 0.002 | 0.126 ± 0.010 | 0.161 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.002 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.084 ± 0.001 | - | 0.087 ± 0.001 | 0.101 ± 0.015 | 0.129 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.012 ± 0.000 | - | 0.017 ± 0.000 | 0.016 ± 0.003 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.005 ± 0.001 | - | 0.009 ± 0.000 | 0.009 ± 0.002 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.012 ± 0.003 | - | 0.015 ± 0.000 | 0.013 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.030 ± 0.000 | - | 0.035 ± 0.000 | 0.033 ± 0.002 | 0.043 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.006 ± 0.001 | - | 0.010 ± 0.000 | 0.008 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.013 ± 0.003 | - | 0.018 ± 0.000 | 0.017 ± 0.005 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.040 ± 0.000 | - | 0.045 ± 0.005 | 0.046 ± 0.007 | 0.064 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 10.584 ± 0.195 | - | 10.616 ± 0.169 | 12.045 ± 0.492 | 11.178 ± 0.465 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.009 ± 0.002 | - | 0.012 ± 0.000 | 0.010 ± 0.000 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 0.840 ± 0.007 | - | 0.846 ± 0.010 | 0.948 ± 0.053 | 0.939 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.024 ± 0.000 | - | 0.029 ± 0.000 | 0.027 ± 0.000 | 0.040 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.079 ± 0.001 | - | 0.084 ± 0.001 | 0.086 ± 0.006 | 0.171 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.521 ± 0.010 | - | 0.537 ± 0.012 | 0.521 ± 0.001 | 1.828 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 2.480 ± 0.028 | - | 2.483 ± 0.017 | 2.469 ± 0.019 | 1.620 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 9.599 ± 0.059 | - | 9.682 ± 0.056 | 9.626 ± 0.297 | 5.782 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 1.324 ± 0.009 | - | 1.334 ± 0.019 | 1.405 ± 0.271 | 1.439 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.006 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.107 ± 0.002 | - | 0.110 ± 0.001 | 0.112 ± 0.002 | 0.180 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.006 ± 0.012 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.084 ± 0.001 | - | 0.087 ± 0.000 | 0.088 ± 0.001 | 0.146 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.011 ± 0.003 | - | 0.014 ± 0.001 | 0.016 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.002 | 0.007 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.009 ± 0.002 | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.027 ± 0.001 | - | 0.030 ± 0.000 | 0.031 ± 0.000 | 0.041 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.005 ± 0.005 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.012 ± 0.003 | - | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.037 ± 0.000 | - | 0.039 ± 0.000 | 0.040 ± 0.000 | 0.053 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 10.575 ± 0.046 | - | 10.646 ± 0.044 | 10.574 ± 0.027 | 5.955 ± 0.160 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.008 ± 0.002 | - | 0.009 ± 0.002 | 0.009 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 0.841 ± 0.007 | - | 0.846 ± 0.006 | 0.849 ± 0.010 | 0.579 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.022 ± 0.000 | - | 0.025 ± 0.000 | 0.025 ± 0.000 | 0.040 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.073 ± 0.001 | - | 0.076 ± 0.000 | 0.076 ± 0.000 | 0.144 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.193 ± 0.001 | - | unsupported | 0.235 ± 0.020 | 1.269 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.249 ± 0.009 | - | unsupported | 1.572 ± 0.109 | 2.424 ± 0.075 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 3.515 ± 0.005 | - | unsupported | 4.313 ± 0.328 | 4.945 ± 0.108 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.705 ± 0.013 | - | unsupported | 0.938 ± 0.142 | 0.775 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.001 ± 0.001 | - | unsupported | 0.005 ± 0.002 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.034 ± 0.000 | - | unsupported | 0.043 ± 0.002 | 0.053 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.028 ± 0.000 | - | unsupported | 0.039 ± 0.006 | 0.040 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | unsupported | 0.009 ± 0.002 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.011 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.015 ± 0.000 | - | unsupported | 0.022 ± 0.007 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 5.647 ± 0.204 | - | unsupported | 6.635 ± 0.303 | 6.104 ± 0.127 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.266 ± 0.000 | - | unsupported | 0.332 ± 0.061 | 0.304 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.010 ± 0.000 | - | unsupported | 0.015 ± 0.000 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.028 ± 0.000 | - | unsupported | 0.042 ± 0.010 | 0.046 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.193 ± 0.000 | - | unsupported | 0.202 ± 0.000 | 1.225 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 1.253 ± 0.011 | - | unsupported | 1.282 ± 0.081 | 1.021 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 3.521 ± 0.006 | - | unsupported | 3.588 ± 0.208 | 2.511 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 0.707 ± 0.014 | - | unsupported | 0.707 ± 0.004 | 0.740 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.034 ± 0.001 | - | unsupported | 0.038 ± 0.001 | 0.044 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.028 ± 0.000 | - | unsupported | 0.032 ± 0.001 | 0.038 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.005 ± 0.001 | - | unsupported | 0.008 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.002 ± 0.001 | - | unsupported | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.011 ± 0.000 | - | unsupported | 0.014 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.006 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.015 ± 0.000 | - | unsupported | 0.018 ± 0.000 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 5.658 ± 0.054 | - | unsupported | 5.681 ± 0.233 | 3.030 ± 0.059 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.266 ± 0.000 | - | unsupported | 0.273 ± 0.000 | 0.290 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.010 ± 0.000 | - | unsupported | 0.014 ± 0.000 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.028 ± 0.000 | - | unsupported | 0.032 ± 0.000 | 0.045 ± 0.002 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.744 ± 0.075 | - | unsupported | 0.892 ± 0.056 | 1.066 ± 0.068 | 3.330 ± 0.782 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.051 ± 0.004 | - | unsupported | 0.090 ± 0.016 | 0.180 ± 0.029 | 0.077 ± 0.006 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.104 ± 0.000 | - | unsupported | 0.145 ± 0.029 | 0.269 ± 0.045 | 0.182 ± 0.212 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.242 ± 0.000 | - | unsupported | 0.308 ± 0.012 | 0.401 ± 0.036 | 0.771 ± 0.071 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.749 ± 0.044 | - | unsupported | 0.486 ± 0.059 | 1.005 ± 0.147 | 8.021 ± 6.034 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.055 ± 0.002 | - | unsupported | 0.104 ± 0.032 | 0.166 ± 0.022 | 0.267 ± 0.161 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.108 ± 0.000 | - | unsupported | 0.131 ± 0.018 | 0.224 ± 0.041 | 0.238 ± 0.030 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.254 ± 0.008 | - | unsupported | 0.215 ± 0.025 | 0.540 ± 0.161 | 0.886 ± 0.444 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.209 ± 0.002 | - | unsupported | 0.196 ± 0.012 | 0.475 ± 0.046 | 0.505 ± 0.028 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.054 ± 0.001 | - | unsupported | 0.046 ± 0.010 | 0.095 ± 0.021 | 0.067 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.075 ± 0.006 | - | unsupported | 0.050 ± 0.010 | 0.099 ± 0.019 | 0.093 ± 0.006 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.093 ± 0.001 | - | unsupported | 0.078 ± 0.000 | 0.176 ± 0.010 | 0.246 ± 0.013 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.203 ± 0.001 | - | unsupported | 0.187 ± 0.018 | 0.631 ± 0.157 | 0.639 ± 0.513 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.061 ± 0.000 | - | unsupported | 0.041 ± 0.001 | 0.114 ± 0.028 | 0.076 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.090 ± 0.000 | - | unsupported | 0.051 ± 0.007 | 0.118 ± 0.008 | 0.097 ± 0.006 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.094 ± 0.000 | - | unsupported | 0.078 ± 0.001 | 0.215 ± 0.094 | 0.292 ± 0.102 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.112 ± 0.002 | - | 0.150 ± 0.001 | 0.084 ± 0.006 | 0.197 ± 0.013 | 0.336 ± 0.151 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.043 ± 0.000 | - | 0.052 ± 0.008 | 0.023 ± 0.005 | 0.030 ± 0.003 | 0.033 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.047 ± 0.002 | - | 0.067 ± 0.012 | 0.026 ± 0.002 | 0.034 ± 0.009 | 0.055 ± 0.008 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.067 ± 0.005 | - | 0.082 ± 0.003 | 0.035 ± 0.001 | 0.066 ± 0.005 | 0.102 ± 0.010 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.099 ± 0.001 | - | 0.313 ± 0.104 | 0.078 ± 0.003 | 0.210 ± 0.084 | 0.266 ± 0.052 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.043 ± 0.000 | - | 0.045 ± 0.004 | 0.026 ± 0.016 | 0.029 ± 0.004 | 0.056 ± 0.040 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.044 ± 0.000 | - | 0.049 ± 0.004 | 0.026 ± 0.001 | 0.045 ± 0.008 | 0.057 ± 0.008 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.058 ± 0.000 | - | 0.104 ± 0.010 | 0.037 ± 0.012 | 0.101 ± 0.053 | 0.112 ± 0.010 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 7.503 ± 0.530 | - | 9.954 ± 1.046 | 26.596 ± 2.746 | 17.616 ± 1.942 | 50.076 ± 13.989 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 7.025 ± 0.041 | - | 8.799 ± 1.201 | 13.074 ± 0.218 | 9.807 ± 0.053 | 346.863 ± 33.762 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 6.319 ± 0.052 | - | 6.141 ± 0.082 | 10.279 ± 3.287 | 6.057 ± 0.064 | 21.904 ± 2.329 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 6.302 ± 0.076 | - | 6.344 ± 0.236 | 5.536 ± 0.105 | 9.122 ± 1.562 | 243.144 ± 49.729 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 6.177 ± 0.695 | - | 6.149 ± 0.069 | 7.751 ± 0.594 | 7.888 ± 0.719 | 6.350 ± 0.745 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 6.084 ± 0.172 | - | 6.093 ± 0.051 | 6.128 ± 0.050 | 7.103 ± 0.103 | 447.508 ± 373.272 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 7.428 ± 0.782 | - | 6.728 ± 0.131 | 8.118 ± 0.326 | 7.403 ± 0.437 | 6.564 ± 0.102 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 6.709 ± 0.026 | - | 6.905 ± 0.266 | 6.775 ± 0.080 | 6.346 ± 0.031 | 257.796 ± 29.341 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 17.692 ± 1.206 | - | 20.257 ± 3.017 | 23.146 ± 1.661 | 26.882 ± 2.836 | 14.088 ± 1.031 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 17.710 ± 1.101 | - | 17.859 ± 0.023 | 18.044 ± 0.531 | 22.540 ± 0.264 | 656.661 ± 24.185 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.246 ± 0.011 | 0.252 ± 0.001 | unsupported | 0.296 ± 0.011 | 0.379 ± 0.027 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 3.061 ± 0.203 | 3.122 ± 0.038 | unsupported | 3.567 ± 0.246 | 5.276 ± 0.645 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.248 ± 0.015 | 0.256 ± 0.005 | unsupported | 0.289 ± 0.054 | 0.410 ± 0.038 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 3.081 ± 0.071 | 3.121 ± 0.098 | unsupported | 3.166 ± 0.165 | 4.001 ± 0.343 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 7.026 ± 0.104 | - | 7.395 ± 0.076 | 11.578 ± 1.375 | 8.658 ± 1.887 | 27.749 ± 4.489 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 7.160 ± 0.628 | - | 7.425 ± 0.165 | 7.085 ± 0.161 | 8.733 ± 0.288 | 333.955 ± 41.462 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 7.624 ± 0.078 | 7.487 ± 0.021 | 5.667 ± 0.478 | 25.046 ± 2.575 | 8.768 ± 0.490 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 9.180 ± 0.603 | 10.613 ± 2.673 | 4.269 ± 0.066 | 27.470 ± 1.666 | 201.138 ± 167.601 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 6.550 ± 0.584 | - | 6.980 ± 0.066 | 13.418 ± 0.640 | 14.191 ± 6.026 | 31.326 ± 9.941 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 8.033 ± 0.991 | - | 8.334 ± 0.761 | 8.355 ± 1.274 | 12.134 ± 8.950 | 227.936 ± 13.668 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.583 ± 0.034 | - | 0.681 ± 0.046 | 2.424 ± 0.171 | 5.771 ± 0.545 | 5.842 ± 0.128 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.731 ± 0.047 | - | 0.623 ± 0.031 | 2.010 ± 0.020 | 7.139 ± 5.162 | 7.551 ± 3.797 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 9.440 ± 0.048 | - | 9.045 ± 0.565 | 13.893 ± 4.611 | 11.084 ± 0.173 | 17.944 ± 0.495 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 9.306 ± 0.108 | - | 9.655 ± 0.529 | 9.128 ± 0.200 | 13.599 ± 1.193 | 968.446 ± 80.422 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 8.963 ± 0.083 | - | 8.901 ± 0.215 | 13.618 ± 2.723 | 12.158 ± 1.393 | 18.168 ± 1.783 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 10.530 ± 0.805 | - | 10.503 ± 0.825 | 9.054 ± 0.102 | 13.828 ± 2.205 | 987.748 ± 846.997 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.292 ± 0.075 | 1.294 ± 0.006 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 12.781 ± 1.603 | 12.135 ± 0.813 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.348 ± 0.132 | 1.323 ± 0.050 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 13.794 ± 1.376 | 12.573 ± 0.291 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.306 ± 0.005 | 1.316 ± 0.013 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 10.566 ± 0.140 | 10.600 ± 0.039 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.305 ± 0.011 | 1.317 ± 0.033 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 10.775 ± 0.085 | 10.934 ± 0.080 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 6.389 ± 0.168 | - | 6.098 ± 0.055 | 9.905 ± 2.209 | 6.083 ± 0.057 | 21.060 ± 1.155 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 6.319 ± 0.052 | - | 6.222 ± 0.068 | 5.501 ± 0.029 | 10.929 ± 5.560 | 221.327 ± 36.965 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 29.482 ± 0.169 | 30.191 ± 0.983 | 41.512 ± 7.223 | 30.849 ± 3.726 | 64.746 ± 2.561 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 31.835 ± 2.533 | 30.587 ± 2.293 | 29.816 ± 0.267 | 44.665 ± 4.211 | 1692.513 ± 153.135 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 12.491 ± 0.410 | - | 17.811 ± 0.405 | 14.309 ± 1.642 | 70.096 ± 1.984 | 32.445 ± 0.513 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 12.182 ± 0.198 | - | 17.572 ± 2.125 | 10.800 ± 0.113 | 45.415 ± 7.184 | 31.429 ± 12.199 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.164 ± 0.014 | 0.163 ± 0.007 | unsupported | 0.054 ± 0.003 | 0.039 ± 0.004 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.841 ± 0.063 | 2.823 ± 0.107 | unsupported | 0.840 ± 0.113 | 0.547 ± 0.006 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.111 ± 0.001 | 0.115 ± 0.017 | unsupported | 0.056 ± 0.018 | 0.041 ± 0.015 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 1.349 ± 0.010 | 1.355 ± 0.006 | unsupported | 0.521 ± 0.102 | 0.281 ± 0.051 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.069 ± 0.000 | 0.069 ± 0.001 | unsupported | 0.085 ± 0.006 | 0.179 ± 0.009 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.101 ± 0.005 | 1.113 ± 0.081 | unsupported | 1.353 ± 0.058 | 2.331 ± 0.090 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.052 ± 0.005 | 0.052 ± 0.010 | unsupported | 0.076 ± 0.024 | 0.142 ± 0.002 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.599 ± 0.011 | 0.597 ± 0.002 | unsupported | 0.675 ± 0.032 | 1.293 ± 0.138 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.245 ± 0.004 | 0.243 ± 0.001 | unsupported | 0.289 ± 0.006 | 0.087 ± 0.015 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 4.239 ± 0.380 | 4.672 ± 0.232 | unsupported | 4.677 ± 0.072 | 0.884 ± 0.025 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.173 ± 0.010 | 0.167 ± 0.018 | unsupported | 0.198 ± 0.007 | 0.066 ± 0.018 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 2.041 ± 0.077 | 2.040 ± 0.012 | unsupported | 2.293 ± 0.004 | 0.489 ± 0.023 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.041 ± 0.001 | 0.060 ± 0.004 | unsupported | 0.044 ± 0.001 | 0.085 ± 0.004 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.515 ± 0.014 | 0.620 ± 0.139 | unsupported | 0.534 ± 0.009 | 0.762 ± 0.036 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.099 ± 0.006 | 0.129 ± 0.026 | unsupported | 0.055 ± 0.007 | 0.075 ± 0.011 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.606 ± 0.010 | 0.661 ± 0.077 | unsupported | 0.301 ± 0.001 | 0.426 ± 0.011 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.081 ± 0.000 | 0.097 ± 0.002 | unsupported | 0.088 ± 0.003 | 0.123 ± 0.001 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.221 ± 0.128 | 1.404 ± 0.162 | unsupported | 1.339 ± 0.020 | 1.174 ± 0.059 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.148 ± 0.020 | 0.158 ± 0.008 | unsupported | 0.083 ± 0.015 | 0.103 ± 0.021 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.906 ± 0.007 | 0.946 ± 0.013 | unsupported | 0.690 ± 0.011 | 0.679 ± 0.006 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.112 ± 0.002 | 0.134 ± 0.011 | unsupported | unsupported | 0.139 ± 0.015 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 1.839 ± 0.170 | 1.785 ± 0.153 | unsupported | unsupported | 1.557 ± 0.027 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.133 ± 0.014 | 0.163 ± 0.027 | unsupported | unsupported | 0.106 ± 0.023 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 1.182 ± 0.058 | 1.270 ± 0.022 | unsupported | unsupported | 0.898 ± 0.018 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.010 ± 0.000 | 0.015 ± 0.000 | unsupported | 0.007 ± 0.000 | 0.064 ± 0.007 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.116 ± 0.009 | 0.120 ± 0.005 | unsupported | 0.057 ± 0.009 | 0.217 ± 0.006 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.021 ± 0.001 | 0.032 ± 0.015 | unsupported | 0.024 ± 0.002 | 0.035 ± 0.005 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.083 ± 0.006 | 0.090 ± 0.002 | unsupported | 0.063 ± 0.001 | 0.161 ± 0.015 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.022 ± 0.000 | 0.034 ± 0.001 | unsupported | 0.025 ± 0.001 | 0.087 ± 0.008 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.285 ± 0.012 | 0.296 ± 0.006 | unsupported | 0.297 ± 0.006 | 0.571 ± 0.014 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.050 ± 0.002 | 0.069 ± 0.006 | unsupported | 0.096 ± 0.012 | 0.084 ± 0.005 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.318 ± 0.015 | 0.377 ± 0.050 | unsupported | 0.281 ± 0.002 | 0.378 ± 0.013 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.045 ± 0.001 | 0.045 ± 0.005 | unsupported | 0.077 ± 0.002 | 0.082 ± 0.004 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.772 ± 0.023 | 0.777 ± 0.028 | unsupported | 1.204 ± 0.067 | 0.939 ± 0.034 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.034 ± 0.002 | 0.035 ± 0.005 | unsupported | 0.075 ± 0.013 | 0.095 ± 0.019 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.392 ± 0.007 | 0.394 ± 0.006 | unsupported | 0.725 ± 0.031 | 0.435 ± 0.022 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.048 ± 0.002 | 0.047 ± 0.000 | unsupported | 0.083 ± 0.025 | 0.081 ± 0.004 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.802 ± 0.016 | 0.814 ± 0.022 | unsupported | 1.201 ± 0.052 | 0.850 ± 0.019 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.036 ± 0.001 | 0.037 ± 0.002 | unsupported | 0.077 ± 0.007 | 0.087 ± 0.022 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.406 ± 0.001 | 0.409 ± 0.008 | unsupported | 0.703 ± 0.052 | 0.462 ± 0.020 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.080 ± 0.002 | 0.095 ± 0.001 | unsupported | 0.095 ± 0.001 | 0.102 ± 0.002 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.395 ± 0.156 | 1.487 ± 0.070 | unsupported | 1.643 ± 0.040 | 1.160 ± 0.056 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.136 ± 0.022 | 0.166 ± 0.016 | unsupported | 0.084 ± 0.008 | 0.112 ± 0.013 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 0.900 ± 0.004 | 0.957 ± 0.031 | unsupported | 0.821 ± 0.004 | 0.733 ± 0.098 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.203 ± 0.000 | 0.203 ± 0.001 | unsupported | 0.276 ± 0.013 | 0.289 ± 0.004 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 3.379 ± 0.412 | 3.239 ± 0.014 | unsupported | 4.344 ± 0.155 | 4.057 ± 0.020 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.129 ± 0.001 | 0.129 ± 0.004 | unsupported | 0.190 ± 0.012 | 0.199 ± 0.013 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 1.702 ± 0.010 | 1.702 ± 0.088 | unsupported | 2.456 ± 0.131 | 2.102 ± 0.033 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.483 ± 0.076 | 0.703 ± 0.026 | unsupported | 0.117 ± 0.003 | 0.109 ± 0.003 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 8.613 ± 0.142 | 10.230 ± 1.062 | unsupported | 2.022 ± 0.103 | 0.965 ± 0.126 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.465 ± 0.038 | 0.659 ± 0.044 | unsupported | 0.141 ± 0.035 | 0.074 ± 0.008 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 6.839 ± 2.074 | 7.309 ± 0.240 | unsupported | 1.085 ± 0.047 | 0.533 ± 0.032 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 18.359 ± 3.014 | - | unsupported | 15.725 ± 1.259 | - | 15.062 ± 1.918 | 19.004 ± 2.952 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 14.473 ± 0.483 | - | unsupported | 21.536 ± 2.034 | - | 33.366 ± 18.617 | 20.868 ± 2.692 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 11.791 ± 1.487 | - | unsupported | 9.581 ± 0.074 | - | 12.371 ± 0.402 | 11.204 ± 0.774 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 9.375 ± 0.674 | - | unsupported | 11.352 ± 0.721 | - | 15.357 ± 5.433 | 13.420 ± 1.497 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 10.539 ± 0.766 | - | unsupported | 10.560 ± 1.009 | - | 10.584 ± 2.944 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 9.186 ± 0.058 | - | unsupported | 10.893 ± 0.543 | - | 12.417 ± 3.547 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 16.952 ± 0.652 | - | unsupported | 15.966 ± 0.241 | - | 25.920 ± 21.837 | 21.108 ± 1.948 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 15.995 ± 2.629 | - | unsupported | 15.073 ± 1.000 | - | 22.249 ± 10.112 | 23.718 ± 2.605 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 10.973 ± 0.661 | - | unsupported | 9.510 ± 0.258 | - | 48.447 ± 1.375 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 7.854 ± 0.322 | - | unsupported | 11.573 ± 0.835 | - | 35.483 ± 5.019 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 12.015 ± 1.674 | - | unsupported | 10.466 ± 0.289 | - | 48.141 ± 0.346 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 8.569 ± 0.088 | - | unsupported | 10.991 ± 1.863 | - | 54.516 ± 15.993 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 18.682 ± 2.127 | - | unsupported | 15.857 ± 0.764 | - | 22.451 ± 4.234 | 21.844 ± 3.156 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 14.436 ± 0.651 | - | unsupported | 18.638 ± 4.800 | - | 26.586 ± 7.363 | 22.506 ± 1.927 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 11.425 ± 0.678 | - | unsupported | 10.523 ± 0.533 | - | 12.339 ± 2.474 | 12.339 ± 2.413 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 9.358 ± 0.473 | - | unsupported | 9.729 ± 1.268 | - | 13.108 ± 0.821 | 13.384 ± 1.304 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.208 ± 0.022 | unsupported | unsupported | 0.247 ± 0.002 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 2.783 ± 0.211 | unsupported | unsupported | 3.008 ± 0.178 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.174 ± 0.001 | unsupported | unsupported | 0.253 ± 0.009 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 2.181 ± 0.126 | unsupported | unsupported | 2.452 ± 0.572 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 18.465 ± 4.216 | - | unsupported | 16.129 ± 1.662 | - | 17.182 ± 1.133 | 23.991 ± 5.089 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 14.275 ± 0.124 | - | unsupported | 20.369 ± 3.668 | - | 23.291 ± 3.879 | 22.845 ± 2.261 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 6.051 ± 1.265 | - | 0.002 ± 0.000 | 4.711 ± 0.388 | 7.167 ± 4.496 | 14.382 ± 2.365 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 6.588 ± 1.266 | - | 0.002 ± 0.000 | 5.205 ± 0.614 | 4.488 ± 0.425 | 12.552 ± 0.135 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 10.710 ± 1.774 | - | 22.525 ± 2.255 | 7.512 ± 0.685 | 13.597 ± 3.968 | 7.525 ± 0.342 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 7.149 ± 0.166 | - | 16.382 ± 0.097 | 8.054 ± 0.533 | 7.841 ± 1.056 | 7.912 ± 0.920 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 11.524 ± 0.993 | - | 11.313 ± 1.150 | 8.878 ± 0.512 | 21.307 ± 5.397 | 12.855 ± 0.869 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 8.380 ± 0.634 | - | 11.988 ± 1.264 | 9.526 ± 0.483 | 8.338 ± 0.623 | 8.909 ± 1.381 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 6.098 ± 2.582 | - | 4.561 ± 0.875 | 9.821 ± 0.361 | 15.452 ± 3.574 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 4.559 ± 0.141 | - | 5.168 ± 0.213 | 8.525 ± 0.484 | 7.102 ± 0.217 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 8.973 ± 0.479 | - | 0.002 ± 0.000 | 10.180 ± 0.177 | 16.112 ± 9.013 | 9.803 ± 1.092 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 9.436 ± 0.136 | - | 0.003 ± 0.000 | 10.209 ± 0.849 | 9.477 ± 0.686 | 11.022 ± 0.361 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 36.743 ± 0.243 | - | 0.002 ± 0.001 | 40.691 ± 2.992 | 81.879 ± 15.586 | 198.013 ± 17.536 | 37.273 ± 1.717 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 16.306 ± 0.418 | - | 0.002 ± 0.001 | 42.657 ± 2.962 | 25.150 ± 0.504 | 235.365 ± 59.922 | 23.016 ± 2.189 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 8.073 ± 0.596 | - | 7.086 ± 0.474 | 7.271 ± 0.128 | 8.234 ± 1.388 | 9.829 ± 0.776 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 5.446 ± 0.749 | - | 5.108 ± 0.438 | 6.002 ± 0.337 | 4.650 ± 0.099 | 7.517 ± 0.024 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 7.370 ± 0.312 | - | 6.958 ± 0.041 | 6.774 ± 0.125 | 7.119 ± 1.252 | 10.719 ± 0.615 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 4.803 ± 0.126 | - | 4.877 ± 0.118 | 5.512 ± 0.092 | 4.693 ± 0.582 | 7.903 ± 1.461 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.002 ± 0.002 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.000 ± 0.001 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 4.646791 | 290424.44 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 4.321000 | 270062.50 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 4.833416 | 302088.50 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 2.992333 | 187020.81 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 3.781250 | 236328.12 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 4.281458 | 267591.12 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 12.515667 | 782229.19 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 11.985166 | 749072.88 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 10.639667 | 664979.19 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 10.667250 | 666703.12 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 6.071875 | 379492.19 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 4.729542 | 295596.38 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 3.934625 | 245914.06 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 4.028541 | 251783.81 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 84.409708 | 5275606.75 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 57.070625 | 3566914.06 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 48.972833 | 3060802.06 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 49.958875 | 3122429.69 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 6.555125 | 409695.31 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 4.624041 | 289002.56 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 3.972750 | 248296.88 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 4.103417 | 256463.56 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 64.008875 | 4000554.69 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 50.656541 | 3166033.81 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 49.300917 | 3081307.31 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 49.930209 | 3120638.06 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 20.671459 | 1291966.19 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 20.712000 | 1294500.00 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 204.499708 | 12781231.75 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 194.158167 | 12134885.44 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 21.575833 | 1348489.56 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 21.168625 | 1323039.06 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 220.705542 | 13794096.38 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 201.161417 | 12572588.56 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 20.899417 | 1306213.56 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 21.060791 | 1316299.44 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 169.055208 | 10565950.50 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 169.599083 | 10599942.69 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 20.887541 | 1305471.31 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 21.066875 | 1316679.69 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 172.403458 | 10775216.12 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 174.951542 | 10934471.38 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.476459 | 38694.67 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.475709 | 54307.95 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 10.482459 | 163788.42 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 10.433500 | 163023.44 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 4.379667 | 547458.38 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 6.719458 | 839932.25 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 22.730625 | 2841328.12 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 22.580625 | 2822578.12 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.643542 | 41305.34 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.576208 | 55878.25 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 7.097167 | 110893.23 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.343083 | 114735.67 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.244166 | 280520.75 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.165875 | 520734.38 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 10.794666 | 1349333.25 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 10.841417 | 1355177.12 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.485250 | 179457.03 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.439625 | 84994.14 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.428333 | 69192.70 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.446000 | 69468.75 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.651666 | 2331458.25 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.820625 | 1352578.12 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 8.811042 | 1101380.25 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 8.900333 | 1112541.62 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.103917 | 142248.70 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.871000 | 76109.38 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.306542 | 51664.72 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3.324458 | 51944.66 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.343959 | 1292994.88 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.396125 | 674515.62 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.794375 | 599296.88 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.775417 | 596927.12 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.562375 | 86912.11 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 18.465250 | 288519.53 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 15.670125 | 244845.70 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 15.578792 | 243418.62 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.073375 | 884171.88 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 37.418959 | 4677369.88 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 33.908459 | 4238557.38 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 37.377208 | 4672151.00 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.250833 | 66419.27 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 12.661875 | 197841.80 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 11.062750 | 172855.47 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 10.714083 | 167407.55 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.911334 | 488916.75 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 18.342000 | 2292750.00 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 16.324667 | 2040583.38 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 16.320583 | 2040072.88 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.449750 | 85152.34 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 2.826250 | 44160.16 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.653041 | 41453.77 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.870166 | 60471.34 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.093250 | 761656.25 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.274375 | 534296.88 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.119666 | 514958.25 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.956667 | 619583.38 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.818958 | 75296.22 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.539666 | 55307.28 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.344958 | 99139.97 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 8.282375 | 129412.11 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.405084 | 425635.50 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.411459 | 301432.38 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.850583 | 606322.88 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.284084 | 660510.50 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.843167 | 122549.48 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.633083 | 88016.92 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.199625 | 81244.14 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.228000 | 97312.50 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 9.394791 | 1174348.88 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.708000 | 1338500.00 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.764500 | 1220562.50 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.229750 | 1403718.75 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.609083 | 103266.92 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.334375 | 83349.61 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 9.497541 | 148399.08 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 10.141750 | 158464.84 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.432750 | 679093.75 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.520500 | 690062.50 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.247834 | 905979.25 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.571792 | 946474.00 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.908542 | 139195.97 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 7.195125 | 112423.83 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.572792 | 133949.88 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.456583 | 1557072.88 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 14.712583 | 1839072.88 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 14.277875 | 1784734.38 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.806834 | 106356.78 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.543209 | 133487.64 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 10.406417 | 162600.27 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.186417 | 898302.12 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.458500 | 1182312.50 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 10.160959 | 1270119.88 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.077542 | 63711.59 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.465834 | 7278.66 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.663875 | 10373.05 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 0.982000 | 15343.75 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 1.738834 | 217354.25 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.459917 | 57489.62 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.926875 | 115859.38 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.959625 | 119953.12 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.243000 | 35046.88 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 1.556292 | 24317.06 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.342833 | 20981.77 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.065417 | 32272.14 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.285958 | 160744.75 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.500042 | 62505.25 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.660417 | 82552.12 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 0.718333 | 89791.62 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.557667 | 86838.55 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 1.597541 | 24961.58 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.394416 | 21787.75 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.153250 | 33644.53 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 4.568833 | 571104.12 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 2.375625 | 296953.12 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.279875 | 284984.38 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 2.371292 | 296411.50 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.382833 | 84106.77 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.141792 | 95965.50 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.207416 | 50115.88 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.423334 | 69114.59 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.022625 | 377828.12 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.248542 | 281067.75 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.540792 | 317599.00 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.015125 | 376890.62 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.242416 | 81912.75 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.900667 | 76572.92 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.856292 | 44629.56 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.876041 | 44938.14 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.513375 | 939171.88 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.631167 | 1203895.88 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.173208 | 771651.00 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.215875 | 776984.38 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.083750 | 95058.59 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.831459 | 75491.55 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.206083 | 34470.05 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.242833 | 35044.27 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.480625 | 435078.12 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.796792 | 724599.00 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.136958 | 392119.75 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.148000 | 393500.00 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.166833 | 80731.77 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.334875 | 83357.42 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.103917 | 48498.70 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.979333 | 46552.08 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.798208 | 849776.00 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.608250 | 1201031.25 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.412375 | 801546.88 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.513209 | 814151.12 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.547291 | 86676.42 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.941875 | 77216.80 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.300209 | 35940.77 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.378250 | 37160.16 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.695291 | 461911.38 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.627083 | 703385.38 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.247750 | 405968.75 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.269458 | 408682.25 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.539167 | 102174.48 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.051083 | 94548.17 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.115584 | 79931.00 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.087875 | 95123.05 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 9.283541 | 1160442.62 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 13.142625 | 1642828.12 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 11.157542 | 1394692.75 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.895792 | 1486974.00 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.150541 | 111727.20 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.380875 | 84076.17 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.719959 | 136249.36 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 10.607958 | 165749.34 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.865250 | 733156.25 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.567083 | 820885.38 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.196750 | 899593.75 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.658750 | 957343.75 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 18.505375 | 289146.48 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.694834 | 276481.78 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 13.002250 | 203160.16 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 12.991834 | 202997.41 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 32.454167 | 4056770.88 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 34.751417 | 4343927.12 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 27.032250 | 3379031.25 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 25.911041 | 3238880.12 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.708250 | 198566.41 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 12.133958 | 189593.09 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.243500 | 128804.69 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 8.258625 | 129041.02 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 16.812166 | 2101520.75 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 19.644208 | 2455526.00 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 13.613084 | 1701635.50 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 13.616500 | 1702062.50 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.998000 | 109343.75 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.500750 | 117199.22 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 30.921208 | 483143.88 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 45.014500 | 703351.56 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.718959 | 964869.88 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 16.172458 | 2021557.25 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 68.902459 | 8612807.38 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 81.843667 | 10230458.38 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.726167 | 73846.36 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 9.019458 | 140929.03 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 29.789833 | 465466.14 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 42.169292 | 658895.19 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.262791 | 532848.88 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.678584 | 1084823.00 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 54.715916 | 6839489.50 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 58.472750 | 7309093.75 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 3.955333 | 247208.31 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 3.323209 | 207700.56 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 48.131958 | 3008247.38 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 44.521875 | 2782617.19 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 4.041958 | 252622.38 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 2.788292 | 174268.25 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 39.238208 | 2452388.00 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 34.895958 | 2180997.38 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.012083 | 755.19 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.001292 | 80.75 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.012334 | 770.88 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.001208 | 75.50 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000125 | 7.81 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.011583 | 723.94 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.027958 | 1747.38 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.011541 | 721.31 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.005459 | 341.19 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.009958 | 622.38 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.003417 | 213.56 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000166 | 10.38 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.010208 | 638.00 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.001459 | 91.19 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000208 | 13.00 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.009958 | 622.38 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002792 | 174.50 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.010333 | 645.81 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002375 | 148.44 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `jax-cpu` is 63.6x faster than the slowest successful cell (`julia-base`, 75.337 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `pytorch-cpu` is 22.5x faster than the slowest successful cell (`julia-base`, 75.337 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-direct` is 23.2x faster than the slowest successful cell (`julia-base`, 75.337 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-trace` is 21.8x faster than the slowest successful cell (`julia-base`, 75.337 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `jax-cpu` is 24.4x faster than the slowest successful cell (`julia-base`, 501.867 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `pytorch-cpu` is 49.5x faster than the slowest successful cell (`julia-base`, 501.867 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-direct` is 49.7x faster than the slowest successful cell (`julia-base`, 501.867 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-trace` is 49.7x faster than the slowest successful cell (`julia-base`, 501.867 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 13.7x faster than the slowest successful cell (`julia-base`, 12.346 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 26.4x faster than the slowest successful cell (`julia-base`, 154.765 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 21.5x faster than the slowest successful cell (`julia-base`, 154.765 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-direct` is 20.9x faster than the slowest successful cell (`julia-base`, 154.765 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-trace` is 21.4x faster than the slowest successful cell (`julia-base`, 154.765 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 57.9x faster than the slowest successful cell (`julia-base`, 117.880 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 25.2x faster than the slowest successful cell (`julia-base`, 117.880 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-direct` is 25.5x faster than the slowest successful cell (`julia-base`, 117.880 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-trace` is 25.0x faster than the slowest successful cell (`julia-base`, 117.880 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `jax-cpu` is 106.6x faster than the slowest successful cell (`julia-base`, 618.355 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `pytorch-cpu` is 91.3x faster than the slowest successful cell (`julia-base`, 618.355 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-direct` is 90.8x faster than the slowest successful cell (`julia-base`, 618.355 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-trace` is 91.5x faster than the slowest successful cell (`julia-base`, 618.355 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/einsum_concrete/einsum_ij_jk_ik` (f64, threads=1, shape=`1024x1024`): `tenferro-direct` is 10.8x faster than the slowest successful cell (`jax-cpu`, 88.353 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 16.3x faster than the slowest successful cell (`pytorch-cpu`, 13.452 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 13.452 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 13.452 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 18.6x faster than the slowest successful cell (`pytorch-cpu`, 10.787 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 10.787 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 10.787 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 12.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 12.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 12.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 19.4x faster than the slowest successful cell (`pytorch-cpu`, 11.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 11.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 15.1x faster than the slowest successful cell (`pytorch-cpu`, 11.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 734.6x faster than the slowest successful cell (`jax-cpu`, 1.592 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 868.6x faster than the slowest successful cell (`julia-base`, 1.954 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.007 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 17.8x faster than the slowest successful cell (`jax-cpu`, 0.033 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.007 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x8x8`): `tenferro-direct` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.008 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x2x2`): `tenferro-direct` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.008 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `pytorch-cpu` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 37.9x faster than the slowest successful cell (`jax-cpu`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 17.8x faster than the slowest successful cell (`jax-cpu`, 0.038 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 55.7x faster than the slowest successful cell (`pytorch-cpu`, 0.040 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 10.4x faster than the slowest successful cell (`pytorch-cpu`, 0.040 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 29.8x faster than the slowest successful cell (`jax-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-trace` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 45.1x faster than the slowest successful cell (`pytorch-cpu`, 0.032 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 36.7x faster than the slowest successful cell (`pytorch-cpu`, 0.031 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 45.3x faster than the slowest successful cell (`pytorch-cpu`, 0.034 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 37.8x faster than the slowest successful cell (`pytorch-cpu`, 0.035 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 17.4x faster than the slowest successful cell (`pytorch-cpu`, 0.021 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 25.1x faster than the slowest successful cell (`pytorch-cpu`, 0.021 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 35.7x faster than the slowest successful cell (`pytorch-cpu`, 0.034 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 40.8x faster than the slowest successful cell (`jax-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 12.8x faster than the slowest successful cell (`jax-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 19.9x faster than the slowest successful cell (`pytorch-cpu`, 0.022 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 23.4x faster than the slowest successful cell (`pytorch-cpu`, 0.032 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 18.6x faster than the slowest successful cell (`pytorch-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`4x2x2`): `tenferro-direct` is 15.6x faster than the slowest successful cell (`jax-cpu`, 0.029 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 14.7x faster than the slowest successful cell (`jax-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 11.9x faster than the slowest successful cell (`jax-cpu`, 0.040 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 27.2x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 12.3x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 22.1x faster than the slowest successful cell (`pytorch-cpu`, 0.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 22.7x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 22.7x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.033 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 23.9x faster than the slowest successful cell (`jax-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 22.1x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 18.5x faster than the slowest successful cell (`pytorch-cpu`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 17.8x faster than the slowest successful cell (`jax-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 25.0x faster than the slowest successful cell (`pytorch-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 21.0x faster than the slowest successful cell (`pytorch-cpu`, 0.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 10.4x faster than the slowest successful cell (`pytorch-cpu`, 0.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 15.5x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x32x32,rhs=1`): `tenferro-direct` is 14.0x faster than the slowest successful cell (`jax-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x32x32,rhs=1`): `tenferro-direct` is 12.2x faster than the slowest successful cell (`jax-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`4x2x2,rhs=1`): `tenferro-direct` is 12.2x faster than the slowest successful cell (`jax-cpu`, 0.020 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x2x2,rhs=1`): `tenferro-direct` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 0.032 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_lu_factor` (f64, threads=4, shape=`1024x16x16,rhs=1`): `pytorch-cpu` is 16.5x faster than the slowest successful cell (`julia-base`, 8.021 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_lu_factor` (f64, threads=4, shape=`1024x16x16,rhs=1`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`julia-base`, 8.021 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `jax-cpu` is 35.4x faster than the slowest successful cell (`julia-base`, 346.863 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `pytorch-cpu` is 26.5x faster than the slowest successful cell (`julia-base`, 346.863 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-direct` is 49.4x faster than the slowest successful cell (`julia-base`, 346.863 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-trace` is 39.4x faster than the slowest successful cell (`julia-base`, 346.863 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 26.7x faster than the slowest successful cell (`julia-base`, 243.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 43.9x faster than the slowest successful cell (`julia-base`, 243.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 38.6x faster than the slowest successful cell (`julia-base`, 243.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 38.3x faster than the slowest successful cell (`julia-base`, 243.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `jax-cpu` is 63.0x faster than the slowest successful cell (`julia-base`, 447.508 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `pytorch-cpu` is 73.0x faster than the slowest successful cell (`julia-base`, 447.508 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-direct` is 73.6x faster than the slowest successful cell (`julia-base`, 447.508 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-trace` is 73.4x faster than the slowest successful cell (`julia-base`, 447.508 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `jax-cpu` is 40.6x faster than the slowest successful cell (`julia-base`, 257.796 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `pytorch-cpu` is 38.0x faster than the slowest successful cell (`julia-base`, 257.796 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-direct` is 38.4x faster than the slowest successful cell (`julia-base`, 257.796 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-trace` is 37.3x faster than the slowest successful cell (`julia-base`, 257.796 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `jax-cpu` is 29.1x faster than the slowest successful cell (`julia-base`, 656.661 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 36.4x faster than the slowest successful cell (`julia-base`, 656.661 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-direct` is 37.1x faster than the slowest successful cell (`julia-base`, 656.661 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-trace` is 36.8x faster than the slowest successful cell (`julia-base`, 656.661 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 38.2x faster than the slowest successful cell (`julia-base`, 333.955 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 47.1x faster than the slowest successful cell (`julia-base`, 333.955 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-direct` is 46.6x faster than the slowest successful cell (`julia-base`, 333.955 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-trace` is 45.0x faster than the slowest successful cell (`julia-base`, 333.955 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 47.1x faster than the slowest successful cell (`julia-base`, 201.138 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-eager` is 21.9x faster than the slowest successful cell (`julia-base`, 201.138 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-trace` is 19.0x faster than the slowest successful cell (`julia-base`, 201.138 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 18.8x faster than the slowest successful cell (`julia-base`, 227.936 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 27.3x faster than the slowest successful cell (`julia-base`, 227.936 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 28.4x faster than the slowest successful cell (`julia-base`, 227.936 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 27.4x faster than the slowest successful cell (`julia-base`, 227.936 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 10.0x faster than the slowest successful cell (`julia-base`, 5.842 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`julia-base`, 7.551 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`julia-base`, 7.551 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `jax-cpu` is 71.2x faster than the slowest successful cell (`julia-base`, 968.446 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 106.1x faster than the slowest successful cell (`julia-base`, 968.446 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 104.1x faster than the slowest successful cell (`julia-base`, 968.446 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 100.3x faster than the slowest successful cell (`julia-base`, 968.446 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `jax-cpu` is 71.4x faster than the slowest successful cell (`julia-base`, 987.748 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 109.1x faster than the slowest successful cell (`julia-base`, 987.748 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 93.8x faster than the slowest successful cell (`julia-base`, 987.748 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 94.0x faster than the slowest successful cell (`julia-base`, 987.748 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 20.3x faster than the slowest successful cell (`julia-base`, 221.327 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 40.2x faster than the slowest successful cell (`julia-base`, 221.327 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 35.0x faster than the slowest successful cell (`julia-base`, 221.327 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 35.6x faster than the slowest successful cell (`julia-base`, 221.327 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 37.9x faster than the slowest successful cell (`julia-base`, 1692.513 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 56.8x faster than the slowest successful cell (`julia-base`, 1692.513 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-eager` is 53.2x faster than the slowest successful cell (`julia-base`, 1692.513 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-trace` is 55.3x faster than the slowest successful cell (`julia-base`, 1692.513 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 10.6x faster than the slowest successful cell (`tenferro-eager`, 10.230 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 13.7x faster than the slowest successful cell (`tenferro-eager`, 7.309 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 6274.9x faster than the slowest successful cell (`julia-base`, 14.382 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 5906.6x faster than the slowest successful cell (`julia-base`, 12.552 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 7734.8x faster than the slowest successful cell (`jax-cpu`, 16.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 4335.9x faster than the slowest successful cell (`julia-base`, 11.022 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 93182.5x faster than the slowest successful cell (`julia-base`, 198.013 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `strided-jl` is 10.2x faster than the slowest successful cell (`julia-base`, 235.365 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-direct` is 14.4x faster than the slowest successful cell (`julia-base`, 235.365 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 99101.1x faster than the slowest successful cell (`julia-base`, 235.365 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 218.4x faster than the slowest successful cell (`tenferro-direct`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 144.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 124.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 63.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 47.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 129.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 7738.4 GB/s at 0.002 ms. The conservative physical lower bound is 0.168 ms, so the cell is 77.4x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 7452.9 GB/s at 0.002 ms. The conservative physical lower bound is 0.042 ms, so the cell is 18.6x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 117118.4 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1171.2x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 126322.6 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 315.8x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 128869.6 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1288.7x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 105600.1 GB/s at 0.003 ms. The conservative physical lower bound is 0.671 ms, so the cell is 264.0x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 63161.3 GB/s at 0.002 ms. The conservative physical lower bound is 1.342 ms, so the cell is 631.6x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 56512.7 GB/s at 0.002 ms. The conservative physical lower bound is 0.336 ms, so the cell is 141.3x faster than the bound.

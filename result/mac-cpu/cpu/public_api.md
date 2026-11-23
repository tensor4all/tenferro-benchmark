# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20261123_180210/run.yaml`
- Timestamp: `2026-11-23T18:02:10.535982+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20261123_180210`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

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

- CSV: `data/results/mac-cpu/cpu/public_api/20261123_180210/cpu_public_api_t1_20261123_180210.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20261123_180210/cpu_public_api_t4_20261123_180210.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 4.365 ± 0.010 | - | 4.370 ± 0.008 | 3.750 ± 0.054 | 1.506 ± 0.030 | 4.275 ± 0.050 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.802 ± 0.037 | - | 2.807 ± 0.045 | 4.242 ± 0.135 | 1.400 ± 0.024 | 77.000 ± 20.097 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 9.888 ± 0.518 | - | 9.344 ± 0.028 | 10.315 ± 0.405 | 10.397 ± 0.080 | 10.555 ± 0.564 | 10.636 ± 13.023 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 9.619 ± 0.035 | - | 9.801 ± 0.221 | 9.678 ± 0.294 | 9.725 ± 0.165 | 10.556 ± 0.130 | 10.171 ± 0.521 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 42.358 ± 0.239 | - | 44.826 ± 1.138 | 18.649 ± 0.064 | 33.400 ± 0.086 | 46.026 ± 1.040 | 48.195 ± 0.098 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 14.572 ± 0.225 | - | 14.893 ± 6.327 | 8.977 ± 0.780 | 17.399 ± 0.959 | 44.177 ± 5.084 | 18.408 ± 1.893 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 62.508 ± 1.829 | - | 62.463 ± 0.181 | 9.810 ± 0.157 | 47.086 ± 1.010 | 56.029 ± 1.554 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 23.175 ± 0.235 | - | 23.636 ± 1.930 | 9.877 ± 0.534 | 16.803 ± 1.007 | 26.683 ± 1.540 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 65.698 ± 0.600 | - | unsupported | 10.290 ± 0.398 | 45.323 ± 0.123 | 56.144 ± 0.882 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 23.691 ± 0.488 | - | unsupported | 10.240 ± 0.321 | 17.086 ± 1.228 | 28.906 ± 4.468 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 6.722 ± 0.042 | - | 6.715 ± 0.196 | 10.943 ± 0.156 | 19.856 ± 0.433 | 8.525 ± 0.058 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 8.226 ± 0.105 | - | 8.240 ± 0.162 | 10.851 ± 0.299 | 19.066 ± 0.312 | 454.474 ± 173.742 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 32.335 ± 1.024 | - | 33.775 ± 0.170 | 34.329 ± 0.746 | 42.617 ± 1.449 | 43.663 ± 4.121 | 45.740 ± 0.376 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 12.502 ± 0.044 | - | 12.503 ± 0.030 | 12.697 ± 0.590 | 21.288 ± 0.519 | 45.343 ± 2.688 | 15.785 ± 0.517 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 44.797 ± 0.204 | - | 44.885 ± 0.559 | 45.460 ± 0.355 | 84.671 ± 0.505 | 69.840 ± 1.690 | 76.588 ± 3.973 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 15.164 ± 0.066 | - | 15.175 ± 0.401 | 19.797 ± 1.109 | 29.484 ± 4.267 | 75.470 ± 2.158 | 28.317 ± 8.011 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 7.080 ± 0.092 | - | 7.099 ± 0.143 | 8.204 ± 0.283 | 8.178 ± 0.105 | 8.314 ± 0.098 | 8.326 ± 0.104 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 7.673 ± 0.227 | - | 7.619 ± 0.034 | 7.656 ± 0.118 | 7.663 ± 0.039 | 9.456 ± 1.287 | 7.915 ± 0.107 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 0.923 ± 0.201 | - | 6.724 ± 0.021 | 5.916 ± 0.053 | 4.467 ± 0.075 | 8.194 ± 0.016 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 0.976 ± 0.007 | - | 3.632 ± 0.036 | 3.125 ± 0.145 | 2.966 ± 0.084 | 11.265 ± 6.397 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 8.924 ± 0.053 | - | 8.786 ± 0.073 | 6.964 ± 0.085 | 6.530 ± 0.096 | 8.761 ± 0.101 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 5.619 ± 0.094 | - | 5.516 ± 0.053 | 6.856 ± 0.249 | 6.906 ± 0.088 | 160.759 ± 24.457 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 5.474 ± 0.027 | - | 5.508 ± 0.021 | 4.899 ± 0.131 | 2.414 ± 0.083 | 5.433 ± 0.204 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 4.157 ± 0.045 | - | 4.149 ± 0.037 | 5.828 ± 0.178 | 2.519 ± 0.082 | 155.161 ± 15.336 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 7.884 ± 0.125 | - | 7.674 ± 0.062 | 6.887 ± 0.147 | 6.369 ± 0.056 | 6.845 ± 0.038 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 7.871 ± 0.080 | - | 7.879 ± 0.220 | 6.894 ± 0.114 | 6.290 ± 0.190 | 591.142 ± 23.185 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 65.172 ± 0.200 | - | 65.274 ± 0.767 | 9.973 ± 0.111 | 45.040 ± 0.246 | 58.461 ± 0.478 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 23.572 ± 0.101 | - | 23.762 ± 0.166 | 10.147 ± 0.769 | 16.607 ± 0.526 | 28.469 ± 4.625 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 53.949 ± 3.814 | - | unsupported | 8.889 ± 0.040 | 47.028 ± 0.603 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 20.277 ± 3.424 | - | unsupported | 9.762 ± 0.102 | 16.561 ± 0.187 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 9.076 ± 0.070 | - | 9.096 ± 0.049 | 10.340 ± 0.376 | 13.066 ± 1.116 | 10.542 ± 4.238 | 11.864 ± 0.355 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 9.921 ± 0.409 | - | 9.626 ± 0.024 | 10.731 ± 0.578 | 11.079 ± 2.511 | 11.631 ± 1.283 | 9.834 ± 0.066 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 16.115 ± 0.121 | - | 16.082 ± 0.103 | 16.735 ± 0.638 | 22.360 ± 0.846 | 14.249 ± 6.593 | 16.556 ± 0.534 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 20.046 ± 1.543 | - | 18.627 ± 0.530 | 14.981 ± 0.582 | 21.763 ± 2.826 | 18.442 ± 9.748 | 16.738 ± 0.440 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 25.464 ± 0.183 | - | 33.134 ± 0.141 | 37.899 ± 3.550 | 31.294 ± 2.388 | 48.487 ± 0.309 | 49.898 ± 4.248 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 13.427 ± 0.201 | - | 17.480 ± 3.451 | 17.918 ± 0.325 | 14.703 ± 0.171 | 46.954 ± 1.116 | 20.066 ± 1.322 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 4.841 ± 0.052 | - | 4.836 ± 0.037 | 6.576 ± 0.532 | 7.064 ± 0.033 | 5.752 ± 0.546 | 6.453 ± 0.295 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 5.276 ± 0.029 | - | 5.305 ± 0.120 | 6.707 ± 0.149 | 7.642 ± 0.532 | 6.832 ± 0.366 | 6.113 ± 1.775 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 12.688 ± 2.154 | - | 11.506 ± 0.115 | 15.010 ± 0.125 | 13.799 ± 0.279 | 11.483 ± 1.401 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 10.670 ± 0.110 | - | 11.191 ± 1.132 | 15.044 ± 7.700 | 13.392 ± 0.619 | 11.882 ± 0.246 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 30.707 ± 0.514 | - | 30.604 ± 0.179 | 30.832 ± 0.045 | 26.591 ± 0.540 | 34.393 ± 0.477 | 37.145 ± 0.733 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 11.785 ± 3.640 | - | 11.767 ± 0.055 | 16.172 ± 4.227 | 14.412 ± 0.217 | 36.757 ± 0.772 | 13.684 ± 1.948 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 16.235 ± 0.744 | - | 16.057 ± 1.000 | 16.168 ± 0.126 | 22.087 ± 0.930 | 16.367 ± 0.198 | 17.223 ± 1.262 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 15.155 ± 1.615 | - | 15.135 ± 0.140 | 15.978 ± 0.309 | 21.749 ± 0.456 | 18.197 ± 1.199 | 17.082 ± 0.934 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 15.895 ± 0.086 | - | 15.928 ± 0.071 | 25.915 ± 0.152 | 16.638 ± 0.119 | 37.525 ± 4.273 | 36.361 ± 3.480 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 8.238 ± 0.065 | - | 8.116 ± 0.702 | 10.591 ± 0.185 | 8.341 ± 0.112 | 34.678 ± 2.766 | 14.022 ± 2.702 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 25.820 ± 0.171 | - | 25.869 ± 0.432 | 29.367 ± 0.278 | 17.659 ± 2.474 | 24.960 ± 0.487 | 25.658 ± 3.853 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 10.447 ± 0.022 | - | 10.442 ± 0.044 | 10.930 ± 0.162 | 8.550 ± 0.050 | 24.364 ± 0.760 | 9.628 ± 3.384 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 29.683 ± 0.636 | - | 29.679 ± 0.041 | 38.530 ± 0.753 | 25.791 ± 0.438 | 41.861 ± 0.623 | 42.467 ± 0.528 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 11.586 ± 0.178 | - | 11.561 ± 0.093 | 18.317 ± 2.148 | 13.267 ± 0.179 | 41.994 ± 0.578 | 24.109 ± 3.704 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 15.165 ± 0.263 | - | 15.264 ± 0.201 | 23.255 ± 0.303 | 19.656 ± 1.082 | 22.054 ± 0.276 | 22.664 ± 3.460 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 7.401 ± 1.619 | - | 7.449 ± 0.119 | 9.449 ± 0.043 | 9.091 ± 0.115 | 22.632 ± 0.450 | 9.856 ± 2.422 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.229 ± 0.002 | 0.249 ± 0.006 | unsupported | unsupported | 0.250 ± 0.003 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.253 ± 0.008 | 0.273 ± 0.012 | unsupported | unsupported | 0.211 ± 0.009 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 14.167 ± 0.108 | - | 14.406 ± 0.093 | 16.148 ± 0.455 | 22.793 ± 1.817 | 14.345 ± 3.276 | 14.674 ± 0.106 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 15.062 ± 0.676 | - | 15.182 ± 0.503 | 18.538 ± 0.369 | 22.688 ± 0.215 | 18.787 ± 2.034 | 18.387 ± 2.342 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 14.854 ± 2.095 | - | 17.058 ± 0.026 | 16.491 ± 0.623 | 21.947 ± 0.263 | 16.801 ± 2.799 | 16.645 ± 1.572 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 15.021 ± 0.496 | - | 15.017 ± 0.538 | 18.230 ± 0.448 | 23.105 ± 0.710 | 16.490 ± 2.142 | 16.060 ± 0.842 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 16.065 ± 0.475 | - | 16.130 ± 1.340 | 17.149 ± 0.430 | 21.861 ± 0.064 | 16.342 ± 0.730 | 16.526 ± 0.134 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 16.764 ± 1.335 | - | 16.572 ± 0.142 | 14.509 ± 0.081 | 21.641 ± 0.043 | 15.308 ± 0.405 | 16.657 ± 0.145 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 10.053 ± 0.294 | - | 9.349 ± 0.095 | 10.349 ± 0.086 | 13.245 ± 0.203 | 10.551 ± 5.444 | 12.113 ± 0.153 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 9.635 ± 0.292 | - | 9.687 ± 0.215 | 10.525 ± 0.267 | 10.310 ± 0.419 | 9.914 ± 5.699 | 9.917 ± 1.010 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 33.682 ± 0.661 | - | 35.374 ± 0.104 | 66.897 ± 1.284 | 44.042 ± 0.759 | 72.997 ± 0.604 | 71.645 ± 7.077 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 13.002 ± 0.770 | - | 17.631 ± 0.623 | 24.610 ± 0.534 | 18.801 ± 0.200 | 70.321 ± 2.676 | 29.535 ± 8.190 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 6.379 ± 0.168 | - | 6.353 ± 0.111 | 10.994 ± 0.020 | 23.170 ± 0.665 | 4.613 ± 0.014 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 5.568 ± 0.186 | - | 5.641 ± 0.175 | 6.641 ± 0.046 | 10.810 ± 0.126 | 6.013 ± 1.006 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 0.807 ± 0.003 | - | 0.808 ± 0.007 | 12.838 ± 0.427 | 2.033 ± 0.032 | 0.556 ± 0.019 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.748 ± 0.035 | - | 0.766 ± 0.017 | 10.250 ± 0.420 | 1.326 ± 0.083 | 0.759 ± 0.103 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 0.743 ± 0.010 | - | 0.743 ± 0.010 | 1.540 ± 0.022 | 0.868 ± 0.007 | 0.691 ± 0.016 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 1.416 ± 0.027 | - | 1.582 ± 0.093 | 0.922 ± 0.029 | 0.810 ± 0.042 | 0.800 ± 0.055 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 6.239 ± 0.074 | - | 6.257 ± 0.035 | 10.996 ± 0.375 | 22.047 ± 0.373 | 4.634 ± 0.036 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 5.641 ± 0.174 | - | 5.609 ± 0.044 | 6.771 ± 0.066 | 10.816 ± 0.046 | 4.885 ± 0.022 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 0.814 ± 0.011 | - | 0.821 ± 0.021 | 14.782 ± 0.219 | 2.027 ± 0.057 | 0.581 ± 0.022 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.857 ± 0.016 | - | 0.781 ± 0.030 | 10.062 ± 0.272 | 1.337 ± 0.043 | 0.762 ± 0.046 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 2.959 ± 0.044 | - | 2.980 ± 0.033 | 5.699 ± 0.041 | 3.065 ± 0.038 | 2.733 ± 0.033 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 5.305 ± 0.311 | - | 5.254 ± 0.018 | 3.488 ± 0.110 | 2.781 ± 0.045 | 3.158 ± 0.084 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 5.889 ± 0.063 | - | 5.874 ± 0.054 | 5.593 ± 0.047 | 41.723 ± 2.298 | 4.698 ± 0.022 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 6.273 ± 0.372 | - | 5.435 ± 0.021 | 5.197 ± 0.083 | 20.037 ± 0.094 | 6.035 ± 0.644 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.681 ± 0.010 | - | 0.686 ± 0.012 | 3.963 ± 0.122 | 2.057 ± 0.028 | 0.629 ± 0.027 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.750 ± 0.052 | - | 0.751 ± 0.015 | 2.241 ± 0.067 | 1.433 ± 0.067 | 0.742 ± 0.054 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 0.691 ± 0.021 | - | 0.690 ± 0.025 | 0.750 ± 0.025 | 1.122 ± 0.045 | 0.868 ± 0.070 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 1.422 ± 0.027 | - | 1.419 ± 0.022 | 0.673 ± 0.005 | 0.816 ± 0.019 | 0.781 ± 0.033 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 5.457 ± 0.122 | - | 5.590 ± 0.188 | 4.630 ± 0.059 | 30.612 ± 0.285 | 4.703 ± 0.089 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 5.804 ± 0.200 | - | 5.783 ± 0.460 | 5.024 ± 0.020 | 15.285 ± 0.052 | 5.889 ± 0.098 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.570 ± 0.030 | - | 0.588 ± 0.196 | 3.917 ± 0.045 | 2.087 ± 0.090 | 0.574 ± 0.015 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.738 ± 0.014 | - | 0.772 ± 0.039 | 2.272 ± 0.211 | 1.316 ± 0.046 | 0.754 ± 0.036 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 0.707 ± 0.075 | - | 0.690 ± 0.026 | 0.665 ± 0.020 | 0.959 ± 0.010 | 0.678 ± 0.021 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 1.440 ± 0.110 | - | 1.414 ± 0.032 | 0.649 ± 0.009 | 0.819 ± 0.033 | 0.765 ± 0.021 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.633 ± 0.012 | 0.640 ± 0.008 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.745 ± 0.040 | 0.673 ± 0.161 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 26.741 ± 0.047 | - | 26.717 ± 0.964 | 31.407 ± 0.163 | 35.467 ± 0.227 | 27.602 ± 4.690 | 29.017 ± 0.857 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 10.882 ± 0.070 | - | 10.911 ± 2.772 | 15.860 ± 0.813 | 19.756 ± 0.184 | 24.967 ± 3.870 | 16.676 ± 1.971 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 18.175 ± 0.133 | - | 18.227 ± 0.160 | 17.984 ± 0.062 | 35.450 ± 0.098 | 41.127 ± 0.404 | 40.871 ± 4.600 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 10.894 ± 1.605 | - | 12.265 ± 1.070 | 10.052 ± 0.328 | 17.754 ± 0.137 | 41.756 ± 0.401 | 19.401 ± 3.215 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 23.226 ± 1.403 | - | 26.885 ± 0.397 | 17.592 ± 0.461 | 22.935 ± 1.299 | 22.757 ± 4.525 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 20.709 ± 0.502 | - | 20.280 ± 0.244 | 19.726 ± 0.399 | 24.421 ± 2.811 | 23.103 ± 4.165 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 9.134 ± 0.034 | - | 9.183 ± 0.080 | 10.407 ± 0.204 | 14.864 ± 0.117 | 9.809 ± 4.714 | 16.989 ± 0.075 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 9.779 ± 0.636 | - | 9.618 ± 0.018 | 11.492 ± 0.197 | 10.473 ± 0.331 | 11.457 ± 2.578 | 11.174 ± 1.879 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 23.582 ± 0.036 | - | 23.591 ± 0.205 | 27.251 ± 0.202 | 25.237 ± 0.241 | 28.717 ± 3.747 | 29.821 ± 0.364 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 10.008 ± 0.051 | - | 11.838 ± 0.612 | 13.147 ± 1.657 | 13.021 ± 0.259 | 29.056 ± 0.572 | 11.869 ± 0.533 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 13.165 ± 0.681 | - | 11.953 ± 0.787 | 11.765 ± 0.073 | 18.200 ± 1.577 | 25.733 ± 0.323 | 25.603 ± 4.185 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 10.058 ± 0.442 | - | 10.561 ± 0.486 | 9.681 ± 0.048 | 11.471 ± 1.099 | 25.656 ± 1.595 | 11.577 ± 0.555 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 16.460 ± 1.982 | - | 17.298 ± 1.480 | 16.984 ± 0.128 | 21.994 ± 0.608 | 16.369 ± 11.803 | 16.830 ± 0.308 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 16.607 ± 0.267 | - | 16.749 ± 0.989 | 15.050 ± 0.545 | 21.642 ± 0.074 | 15.525 ± 12.222 | 15.249 ± 1.718 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 66.298 ± 1.504 | - | 63.391 ± 0.134 | 65.499 ± 0.171 | 12.674 ± 0.056 | 42.726 ± 1.603 | 45.507 ± 0.969 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 26.594 ± 4.139 | - | 29.132 ± 8.561 | 32.434 ± 2.532 | 6.754 ± 0.068 | 41.303 ± 1.188 | 22.098 ± 4.584 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 26.253 ± 0.223 | 26.878 ± 0.906 | unsupported | 17.572 ± 0.438 | 20.808 ± 0.223 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 22.311 ± 2.657 | 23.913 ± 0.253 | unsupported | 20.319 ± 0.263 | 22.464 ± 0.366 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.557 ± 0.002 | - | 0.570 ± 0.004 | 0.593 ± 0.061 | 0.563 ± 0.016 | 0.772 ± 0.202 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.631 ± 0.093 | - | 0.645 ± 0.024 | 0.654 ± 0.031 | 0.569 ± 0.002 | 0.691 ± 0.010 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.557 ± 0.001 | - | 0.559 ± 0.015 | 0.670 ± 0.029 | 0.587 ± 0.007 | 0.852 ± 0.139 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.587 ± 0.003 | - | 0.584 ± 0.034 | 0.612 ± 0.004 | 0.672 ± 0.008 | 0.735 ± 0.068 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 0.807 ± 0.010 | - | unsupported | 1.048 ± 0.041 | 0.854 ± 0.008 | 1.169 ± 4.000 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.907 ± 0.056 | - | unsupported | 0.952 ± 0.033 | 0.895 ± 0.019 | 0.994 ± 2.476 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.320 ± 0.001 | - | 0.388 ± 0.004 | 0.293 ± 0.010 | 0.319 ± 0.012 | 0.461 ± 0.050 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.286 ± 0.006 | - | 0.340 ± 0.028 | 0.284 ± 0.001 | 0.230 ± 0.025 | 0.645 ± 0.371 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.500 ± 0.003 | - | 0.558 ± 0.019 | 0.996 ± 0.042 | 0.650 ± 0.049 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.510 ± 0.009 | - | 0.648 ± 0.035 | 1.140 ± 0.232 | 0.734 ± 0.048 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.565 ± 0.007 | - | 0.578 ± 0.016 | 0.655 ± 0.145 | 0.591 ± 0.010 | 1.193 ± 3.907 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.577 ± 0.068 | - | 0.619 ± 0.043 | 0.656 ± 0.032 | 0.715 ± 0.014 | 1.344 ± 4.062 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.527 ± 0.004 | - | 0.595 ± 0.012 | 0.361 ± 0.014 | 0.383 ± 0.028 | 0.491 ± 0.125 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.571 ± 0.008 | - | 0.648 ± 0.005 | 0.407 ± 0.036 | 0.400 ± 0.013 | 0.350 ± 0.342 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 0.920 ± 0.007 | - | 0.002 ± 0.000 | 1.078 ± 0.037 | 0.969 ± 0.036 | 1.528 ± 0.438 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.958 ± 0.046 | - | 0.001 ± 0.000 | 0.985 ± 0.027 | 1.003 ± 0.002 | 1.402 ± 0.048 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.046 ± 0.000 | - | 0.047 ± 0.000 | 0.104 ± 0.000 | 0.083 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.093 ± 0.001 | - | 0.097 ± 0.000 | 0.151 ± 0.000 | 0.141 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.210 ± 0.001 | - | 0.211 ± 0.001 | 0.247 ± 0.000 | 0.230 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.047 ± 0.000 | - | 0.048 ± 0.000 | 0.037 ± 0.000 | 0.089 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.049 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.059 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.365 ± 0.000 | - | 0.366 ± 0.000 | 0.231 ± 0.001 | 0.257 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.018 ± 0.000 | - | 0.019 ± 0.000 | 0.016 ± 0.001 | 0.069 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.044 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.047 ± 0.000 | - | 0.047 ± 0.000 | 0.151 ± 0.011 | 0.119 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.097 ± 0.001 | - | 0.098 ± 0.000 | 0.182 ± 0.013 | 0.146 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.208 ± 0.009 | - | 0.209 ± 0.000 | 0.259 ± 0.011 | 0.241 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.048 ± 0.000 | - | 0.050 ± 0.000 | 0.088 ± 0.012 | 0.064 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.046 ± 0.018 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.060 ± 0.012 | 0.054 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.053 ± 0.014 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.001 | 0.057 ± 0.018 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.053 ± 0.021 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.053 ± 0.014 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.042 ± 0.010 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.049 ± 0.011 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.054 ± 0.030 | 0.008 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.048 ± 0.015 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.218 ± 0.006 | - | 0.220 ± 0.001 | 0.216 ± 0.033 | 0.188 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.051 ± 0.012 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.018 ± 0.004 | - | 0.020 ± 0.000 | 0.074 ± 0.009 | 0.075 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.049 ± 0.016 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.044 ± 0.023 | 0.057 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.276 ± 0.004 | - | 0.277 ± 0.003 | 0.406 ± 0.003 | 0.813 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.389 ± 0.019 | - | 1.398 ± 0.009 | 1.603 ± 0.012 | 2.253 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 4.713 ± 0.032 | - | 4.714 ± 0.052 | 6.113 ± 0.360 | 6.639 ± 0.163 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 2.332 ± 0.014 | - | 2.359 ± 0.024 | 3.167 ± 0.024 | 3.770 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.077 ± 0.001 | - | 0.078 ± 0.006 | 0.089 ± 0.006 | 0.134 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.001 | 0.009 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.016 ± 0.002 | - | 0.016 ± 0.000 | 0.022 ± 0.000 | 0.061 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.008 ± 0.001 | - | 0.008 ± 0.002 | 0.011 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.028 ± 0.000 | 0.066 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 18.800 ± 0.141 | - | 18.788 ± 0.114 | 25.153 ± 0.168 | 30.054 ± 0.125 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.609 ± 0.001 | - | 0.607 ± 0.007 | 0.666 ± 0.019 | 0.845 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.014 ± 0.003 | - | 0.013 ± 0.000 | 0.017 ± 0.000 | 0.026 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.038 ± 0.000 | - | 0.039 ± 0.000 | 0.051 ± 0.000 | 0.095 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.265 ± 0.002 | - | 0.266 ± 0.001 | 0.421 ± 0.001 | 0.769 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 0.713 ± 0.005 | - | 0.712 ± 0.006 | 1.673 ± 0.038 | 1.064 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 2.323 ± 0.011 | - | 2.328 ± 0.009 | 6.548 ± 0.108 | 4.008 ± 0.259 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 2.242 ± 0.022 | - | 2.245 ± 0.042 | 3.183 ± 0.042 | 3.682 ± 0.092 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.074 ± 0.001 | - | 0.076 ± 0.001 | 0.089 ± 0.003 | 0.204 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.002 | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.008 ± 0.001 | - | 0.008 ± 0.001 | 0.011 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.006 ± 0.001 | - | 0.007 ± 0.002 | 0.010 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.015 ± 0.004 | - | 0.016 ± 0.001 | 0.023 ± 0.001 | 0.092 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.008 ± 0.002 | - | 0.011 ± 0.002 | 0.011 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.019 ± 0.004 | - | 0.021 ± 0.000 | 0.029 ± 0.000 | 0.099 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 7.489 ± 1.915 | - | 7.461 ± 1.824 | 25.078 ± 0.420 | 15.607 ± 0.277 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.002 | 0.008 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.328 ± 0.005 | - | 0.331 ± 0.001 | 0.714 ± 0.041 | 0.531 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.014 ± 0.004 | - | 0.014 ± 0.000 | 0.017 ± 0.000 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.037 ± 0.002 | - | 0.039 ± 0.001 | 0.056 ± 0.016 | 0.133 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.256 ± 0.002 | - | 0.255 ± 0.001 | 0.230 ± 0.005 | 0.517 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.155 ± 0.015 | - | 1.132 ± 0.010 | 1.183 ± 0.030 | 1.450 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 4.122 ± 0.046 | - | 4.112 ± 0.010 | 4.629 ± 0.033 | 5.010 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.366 ± 0.021 | - | 1.374 ± 0.012 | 0.912 ± 0.005 | 0.983 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.061 ± 0.001 | - | 0.063 ± 0.000 | 0.065 ± 0.001 | 0.101 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.012 ± 0.000 | - | 0.013 ± 0.000 | 0.018 ± 0.001 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.016 ± 0.001 | - | 0.017 ± 0.000 | 0.022 ± 0.000 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 10.943 ± 0.078 | - | 11.119 ± 0.240 | 7.251 ± 0.016 | 7.619 ± 0.231 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.483 ± 0.003 | - | 0.485 ± 0.001 | 0.487 ± 0.001 | 0.541 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.010 ± 0.000 | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.031 ± 0.001 | - | 0.032 ± 0.001 | 0.040 ± 0.001 | 0.082 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.257 ± 0.006 | - | 0.257 ± 0.014 | 0.239 ± 0.001 | 0.585 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.157 ± 0.011 | - | 1.156 ± 0.020 | 1.226 ± 0.020 | 1.023 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 2.030 ± 0.005 | - | 2.032 ± 0.002 | 4.876 ± 0.095 | 2.800 ± 0.059 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.307 ± 0.012 | - | 1.305 ± 0.012 | 0.958 ± 0.042 | 1.087 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.059 ± 0.000 | - | 0.061 ± 0.000 | 0.063 ± 0.006 | 0.146 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.007 ± 0.001 | - | 0.009 ± 0.002 | 0.009 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.002 | 0.008 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.014 ± 0.004 | - | 0.015 ± 0.000 | 0.019 ± 0.000 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.001 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.007 ± 0.002 | - | 0.008 ± 0.002 | 0.010 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.017 ± 0.004 | - | 0.018 ± 0.000 | 0.024 ± 0.000 | 0.031 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 5.387 ± 0.709 | - | 5.369 ± 0.015 | 7.633 ± 0.294 | 4.332 ± 0.155 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.270 ± 0.001 | - | 0.272 ± 0.003 | 0.478 ± 0.011 | 0.334 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.013 ± 0.003 | - | 0.012 ± 0.000 | 0.014 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.032 ± 0.000 | - | 0.034 ± 0.000 | 0.042 ± 0.000 | 0.112 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.202 ± 0.002 | - | 0.191 ± 0.003 | 0.127 ± 0.001 | 0.543 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 1.114 ± 0.033 | - | 1.113 ± 0.018 | 0.736 ± 0.001 | 1.517 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 3.702 ± 0.040 | - | 3.703 ± 0.022 | 2.371 ± 0.030 | 5.229 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.735 ± 0.015 | - | 0.735 ± 0.026 | 0.453 ± 0.002 | 1.024 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.046 ± 0.001 | - | 0.047 ± 0.000 | 0.030 ± 0.001 | 0.109 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.012 ± 0.000 | - | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.015 ± 0.000 | - | 0.016 ± 0.000 | 0.014 ± 0.004 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 5.908 ± 0.044 | - | 5.940 ± 0.072 | 3.679 ± 0.077 | 7.806 ± 0.141 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.366 ± 0.009 | - | 0.367 ± 0.002 | 0.197 ± 0.003 | 0.533 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.009 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.003 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.030 ± 0.000 | - | 0.031 ± 0.000 | 0.023 ± 0.000 | 0.083 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.194 ± 0.001 | - | 0.185 ± 0.001 | 0.123 ± 0.002 | 0.528 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 1.063 ± 0.018 | - | 1.060 ± 0.011 | 0.750 ± 0.022 | 0.982 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 1.843 ± 0.033 | - | 1.839 ± 0.011 | 2.395 ± 0.009 | 2.961 ± 0.655 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.731 ± 0.021 | - | 0.736 ± 0.004 | 0.455 ± 0.002 | 1.098 ± 0.061 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.045 ± 0.000 | - | 0.047 ± 0.000 | 0.030 ± 0.001 | 0.157 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.002 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.002 | 0.007 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.013 ± 0.003 | - | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.002 | 0.008 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.015 ± 0.004 | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 2.897 ± 0.031 | - | 2.906 ± 0.019 | 3.601 ± 0.062 | 4.111 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.001 | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.211 ± 0.001 | - | 0.214 ± 0.001 | 0.197 ± 0.005 | 0.363 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.011 ± 0.002 | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.029 ± 0.000 | - | 0.031 ± 0.000 | 0.026 ± 0.005 | 0.108 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.067 ± 0.003 | - | 0.066 ± 0.000 | 0.148 ± 0.000 | 0.152 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.158 ± 0.019 | - | 0.152 ± 0.004 | 0.233 ± 0.002 | 0.204 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.429 ± 0.007 | - | 0.429 ± 0.003 | 0.477 ± 0.006 | 0.403 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.126 ± 0.000 | - | 0.127 ± 0.002 | 0.067 ± 0.002 | 0.115 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.042 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.051 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.043 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.042 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.978 ± 0.012 | - | 0.978 ± 0.017 | 0.464 ± 0.009 | 0.631 ± 0.066 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.057 ± 0.010 | - | 0.053 ± 0.001 | 0.038 ± 0.000 | 0.089 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.065 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.041 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.073 ± 0.006 | - | 0.071 ± 0.004 | 0.173 ± 0.017 | 0.164 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.161 ± 0.004 | - | 0.158 ± 0.003 | 0.252 ± 0.002 | 0.256 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.451 ± 0.012 | - | 0.449 ± 0.001 | 0.442 ± 0.008 | 0.313 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.171 ± 0.005 | - | 0.175 ± 0.006 | 0.113 ± 0.010 | 0.141 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.053 ± 0.006 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.011 ± 0.002 | - | 0.011 ± 0.001 | 0.053 ± 0.011 | 0.063 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.052 ± 0.019 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.012 ± 0.003 | - | 0.013 ± 0.000 | 0.060 ± 0.004 | 0.081 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.050 ± 0.005 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.001 | 0.051 ± 0.012 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.001 | 0.067 ± 0.025 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.001 | 0.053 ± 0.015 | 0.061 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.051 ± 0.011 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.004 ± 0.001 | - | 0.005 ± 0.000 | 0.058 ± 0.015 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.054 ± 0.004 | 0.013 ± 0.061 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.548 ± 0.011 | - | 0.563 ± 0.017 | 0.399 ± 0.016 | 0.492 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.004 ± 0.001 | - | 0.005 ± 0.000 | 0.069 ± 0.009 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.055 ± 0.001 | - | 0.057 ± 0.001 | 0.112 ± 0.011 | 0.130 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.086 ± 0.021 | 0.008 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.007 ± 0.001 | - | 0.008 ± 0.002 | 0.075 ± 0.010 | 0.041 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.157 ± 0.000 | - | 0.158 ± 0.001 | 0.270 ± 0.005 | 0.344 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.360 ± 0.001 | - | 0.361 ± 0.001 | 0.593 ± 0.003 | 0.634 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.011 ± 0.004 | - | 1.016 ± 0.035 | 1.453 ± 0.012 | 1.398 ± 0.068 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.381 ± 0.001 | - | 0.380 ± 0.001 | 0.320 ± 0.002 | 0.392 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.019 ± 0.000 | - | 0.020 ± 0.000 | 0.018 ± 0.000 | 0.053 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.021 ± 0.000 | - | 0.022 ± 0.000 | 0.016 ± 0.000 | 0.051 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 3.041 ± 0.011 | - | 3.019 ± 0.019 | 2.573 ± 0.050 | 2.620 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.143 ± 0.001 | - | 0.144 ± 0.001 | 0.131 ± 0.001 | 0.151 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.009 ± 0.000 | - | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.158 ± 0.001 | - | 0.159 ± 0.001 | 0.251 ± 0.015 | 0.398 ± 0.072 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.361 ± 0.012 | - | 0.362 ± 0.002 | 0.529 ± 0.009 | 0.657 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 0.530 ± 0.024 | - | 0.532 ± 0.021 | 1.275 ± 0.029 | 0.773 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.383 ± 0.006 | - | 0.383 ± 0.002 | 0.365 ± 0.017 | 0.406 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.001 | 0.039 ± 0.015 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.020 ± 0.006 | - | 0.021 ± 0.000 | 0.043 ± 0.026 | 0.087 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.024 ± 0.018 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.022 ± 0.000 | - | 0.024 ± 0.000 | 0.044 ± 0.010 | 0.075 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.001 | 0.024 ± 0.024 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.001 | 0.037 ± 0.018 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.001 | 0.027 ± 0.016 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.001 | 0.039 ± 0.015 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.024 ± 0.011 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.029 ± 0.014 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.006 ± 0.000 | - | 0.007 ± 0.002 | 0.029 ± 0.017 | 0.018 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 1.596 ± 0.158 | - | 1.591 ± 0.001 | 2.698 ± 0.129 | 1.606 ± 0.107 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.032 ± 0.013 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.144 ± 0.001 | - | 0.145 ± 0.001 | 0.128 ± 0.012 | 0.166 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.002 | 0.038 ± 0.011 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.011 ± 0.002 | - | 0.011 ± 0.000 | 0.038 ± 0.010 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.090 ± 0.006 | - | 0.094 ± 0.002 | 0.122 ± 0.001 | 0.182 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.142 ± 0.001 | - | 0.153 ± 0.003 | 0.178 ± 0.007 | 0.227 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.349 ± 0.002 | - | 0.356 ± 0.002 | 0.354 ± 0.001 | 0.482 ± 0.075 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.081 ± 0.002 | - | 0.084 ± 0.000 | 0.059 ± 0.001 | 0.102 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.006 ± 0.002 | 0.009 ± 0.001 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.060 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.624 ± 0.002 | - | 0.633 ± 0.013 | 0.371 ± 0.014 | 0.497 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.033 ± 0.000 | - | 0.038 ± 0.000 | 0.036 ± 0.000 | 0.113 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.090 ± 0.001 | - | 0.098 ± 0.001 | 0.111 ± 0.009 | 0.182 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.142 ± 0.001 | - | 0.154 ± 0.005 | 0.157 ± 0.001 | 0.298 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.209 ± 0.001 | - | 0.227 ± 0.005 | 0.292 ± 0.011 | 0.373 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.124 ± 0.003 | - | 0.126 ± 0.002 | 0.064 ± 0.004 | 0.152 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.003 | 0.010 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.006 ± 0.001 | - | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.071 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.010 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.002 | 0.011 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.002 | 0.011 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.355 ± 0.005 | - | 0.400 ± 0.019 | 0.294 ± 0.016 | 0.505 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.051 ± 0.006 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.034 ± 0.000 | - | 0.040 ± 0.000 | 0.074 ± 0.007 | 0.147 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.002 | 0.046 ± 0.012 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.005 ± 0.001 | - | 0.009 ± 0.000 | 0.046 ± 0.010 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 0.760 ± 0.032 | - | 0.682 ± 0.016 | 1.206 ± 0.083 | 2.714 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 2.780 ± 0.014 | - | 2.596 ± 0.035 | 4.395 ± 0.119 | 5.620 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 9.149 ± 0.072 | - | 8.672 ± 0.051 | 14.928 ± 1.021 | 15.908 ± 0.233 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 4.567 ± 0.032 | - | 4.576 ± 0.076 | 4.879 ± 0.461 | 5.013 ± 0.064 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.141 ± 0.006 | - | 0.145 ± 0.008 | 0.185 ± 0.005 | 0.282 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.121 ± 0.001 | - | 0.122 ± 0.001 | 0.166 ± 0.011 | 0.260 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.010 ± 0.000 | - | 0.011 ± 0.000 | 0.026 ± 0.005 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.010 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.000 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.027 ± 0.003 | - | 0.027 ± 0.001 | 0.049 ± 0.001 | 0.061 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.013 ± 0.000 | - | 0.013 ± 0.000 | 0.022 ± 0.000 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.036 ± 0.001 | - | 0.035 ± 0.000 | 0.065 ± 0.003 | 0.078 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 35.633 ± 0.088 | - | 35.654 ± 0.169 | 36.551 ± 1.780 | 39.693 ± 0.186 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.015 ± 0.000 | 0.033 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.098 ± 0.023 | - | 1.093 ± 0.008 | 1.397 ± 0.009 | 1.728 ± 0.129 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.023 ± 0.000 | - | 0.023 ± 0.000 | 0.039 ± 0.001 | 0.056 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.071 ± 0.001 | - | 0.069 ± 0.001 | 0.121 ± 0.002 | 0.171 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 0.753 ± 0.005 | - | 0.687 ± 0.011 | 1.167 ± 0.018 | 2.736 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 1.324 ± 0.022 | - | 1.331 ± 0.012 | 4.243 ± 0.035 | 2.375 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 4.370 ± 0.083 | - | 4.387 ± 0.208 | 14.553 ± 0.041 | 8.466 ± 0.229 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 4.763 ± 0.107 | - | 4.686 ± 0.026 | 4.552 ± 0.078 | 5.193 ± 0.634 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.001 | 0.007 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.148 ± 0.019 | - | 0.146 ± 0.004 | 0.178 ± 0.006 | 0.295 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.006 ± 0.001 | - | 0.007 ± 0.001 | 0.011 ± 0.001 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.128 ± 0.005 | - | 0.129 ± 0.001 | 0.162 ± 0.002 | 0.244 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.011 ± 0.000 | - | 0.012 ± 0.000 | 0.021 ± 0.001 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.005 ± 0.001 | - | 0.006 ± 0.002 | 0.010 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.012 ± 0.003 | - | 0.011 ± 0.000 | 0.019 ± 0.000 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.028 ± 0.000 | - | 0.028 ± 0.001 | 0.050 ± 0.001 | 0.062 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.006 ± 0.001 | - | 0.007 ± 0.002 | 0.011 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.014 ± 0.003 | - | 0.014 ± 0.000 | 0.023 ± 0.000 | 0.035 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.037 ± 0.000 | - | 0.037 ± 0.001 | 0.065 ± 0.001 | 0.080 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 17.342 ± 2.560 | - | 17.357 ± 2.734 | 35.882 ± 0.836 | 21.585 ± 1.413 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.010 ± 0.002 | - | 0.009 ± 0.001 | 0.016 ± 0.001 | 0.035 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 0.613 ± 0.029 | - | 0.621 ± 0.026 | 1.372 ± 0.066 | 0.984 ± 0.151 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.024 ± 0.002 | - | 0.025 ± 0.000 | 0.039 ± 0.000 | 0.057 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.072 ± 0.001 | - | 0.070 ± 0.001 | 0.121 ± 0.002 | 0.200 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.502 ± 0.010 | - | 0.510 ± 0.013 | 0.566 ± 0.006 | 1.926 ± 0.059 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 2.112 ± 0.028 | - | 2.108 ± 0.039 | 2.667 ± 0.014 | 4.010 ± 0.141 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 7.406 ± 0.108 | - | 7.440 ± 0.196 | 10.385 ± 0.126 | 11.653 ± 0.282 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 2.079 ± 0.022 | - | 2.083 ± 0.039 | 1.446 ± 0.026 | 1.480 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.002 ± 0.001 | - | 0.003 ± 0.000 | 0.006 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.105 ± 0.001 | - | 0.106 ± 0.001 | 0.123 ± 0.004 | 0.164 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.075 ± 0.000 | - | 0.076 ± 0.000 | 0.100 ± 0.006 | 0.141 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.016 ± 0.001 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.013 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.021 ± 0.000 | - | 0.023 ± 0.000 | 0.033 ± 0.001 | 0.045 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | 0.011 ± 0.000 | 0.016 ± 0.000 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.029 ± 0.000 | - | 0.030 ± 0.000 | 0.044 ± 0.001 | 0.056 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 16.490 ± 0.083 | - | 16.539 ± 0.162 | 11.957 ± 0.155 | 11.638 ± 0.123 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.027 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 0.825 ± 0.007 | - | 0.823 ± 0.017 | 0.909 ± 0.025 | 0.997 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.018 ± 0.000 | - | 0.019 ± 0.000 | 0.027 ± 0.000 | 0.044 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.057 ± 0.001 | - | 0.058 ± 0.000 | 0.082 ± 0.001 | 0.155 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.480 ± 0.013 | - | 0.480 ± 0.001 | 0.568 ± 0.007 | 1.972 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 2.031 ± 0.016 | - | 2.023 ± 0.012 | 2.671 ± 0.027 | 1.757 ± 0.179 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 3.579 ± 0.013 | - | 3.585 ± 0.014 | 10.417 ± 0.080 | 6.364 ± 0.147 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 1.968 ± 0.017 | - | 2.067 ± 0.026 | 1.429 ± 0.017 | 1.615 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.002 | 0.007 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.102 ± 0.001 | - | 0.103 ± 0.001 | 0.121 ± 0.005 | 0.208 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.002 | 0.009 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.077 ± 0.004 | - | 0.079 ± 0.002 | 0.097 ± 0.010 | 0.140 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.011 ± 0.002 | - | 0.010 ± 0.000 | 0.017 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.002 | 0.008 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.010 ± 0.002 | - | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.021 ± 0.001 | - | 0.023 ± 0.000 | 0.034 ± 0.002 | 0.043 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.001 | 0.009 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.012 ± 0.003 | - | 0.012 ± 0.000 | 0.017 ± 0.001 | 0.028 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.028 ± 0.000 | - | 0.029 ± 0.000 | 0.045 ± 0.000 | 0.057 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 8.187 ± 1.915 | - | 8.085 ± 0.815 | 11.531 ± 0.191 | 6.542 ± 0.185 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 0.433 ± 0.006 | - | 0.428 ± 0.007 | 0.910 ± 0.019 | 0.622 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.018 ± 0.005 | - | 0.020 ± 0.000 | 0.027 ± 0.000 | 0.041 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.055 ± 0.000 | - | 0.056 ± 0.000 | 0.084 ± 0.001 | 0.152 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.306 ± 0.001 | - | unsupported | 0.219 ± 0.008 | 1.326 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.417 ± 0.015 | - | unsupported | 1.356 ± 0.013 | 2.509 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 4.059 ± 0.020 | - | unsupported | 3.869 ± 0.056 | 5.082 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 1.111 ± 0.011 | - | unsupported | 0.762 ± 0.011 | 0.792 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.044 ± 0.001 | - | unsupported | 0.041 ± 0.001 | 0.048 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.035 ± 0.001 | - | unsupported | 0.035 ± 0.000 | 0.042 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.013 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.006 ± 0.001 | - | unsupported | 0.010 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.017 ± 0.000 | - | unsupported | 0.020 ± 0.000 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 8.886 ± 0.034 | - | unsupported | 6.096 ± 0.061 | 6.210 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.345 ± 0.001 | - | unsupported | 0.295 ± 0.017 | 0.314 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.012 ± 0.000 | - | unsupported | 0.015 ± 0.000 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.032 ± 0.000 | - | unsupported | 0.034 ± 0.000 | 0.048 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.318 ± 0.004 | - | unsupported | 0.210 ± 0.000 | 1.346 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 1.485 ± 0.038 | - | unsupported | 1.294 ± 0.012 | 1.078 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 2.145 ± 0.016 | - | unsupported | 3.683 ± 0.010 | 2.633 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.171 ± 0.027 | - | unsupported | 0.736 ± 0.011 | 0.740 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.047 ± 0.001 | - | unsupported | 0.040 ± 0.007 | 0.047 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.038 ± 0.000 | - | unsupported | 0.034 ± 0.000 | 0.040 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.006 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.006 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.014 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.035 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.007 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.018 ± 0.000 | - | unsupported | 0.019 ± 0.000 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 4.520 ± 0.166 | - | unsupported | 5.875 ± 0.017 | 3.312 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.209 ± 0.014 | - | unsupported | 0.281 ± 0.000 | 0.277 ± 0.061 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.013 ± 0.000 | - | unsupported | 0.015 ± 0.000 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.034 ± 0.000 | - | unsupported | 0.033 ± 0.000 | 0.048 ± 0.001 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.686 ± 0.001 | - | unsupported | 0.757 ± 0.001 | 1.005 ± 0.025 | 2.666 ± 0.046 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.049 ± 0.000 | - | unsupported | 0.063 ± 0.001 | 0.127 ± 0.008 | 0.058 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.091 ± 0.007 | - | unsupported | 0.115 ± 0.000 | 0.178 ± 0.010 | 0.136 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.209 ± 0.006 | - | unsupported | 0.259 ± 0.001 | 0.332 ± 0.003 | 0.546 ± 0.019 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.413 ± 0.006 | - | unsupported | 0.456 ± 0.004 | 0.624 ± 0.067 | 2.700 ± 0.381 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.053 ± 0.001 | - | unsupported | 0.074 ± 0.002 | 0.154 ± 0.031 | 0.064 ± 0.007 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.095 ± 0.003 | - | unsupported | 0.107 ± 0.002 | 0.211 ± 0.004 | 0.150 ± 0.023 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.214 ± 0.002 | - | unsupported | 0.210 ± 0.008 | 0.562 ± 0.128 | 0.641 ± 0.033 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.408 ± 0.002 | - | unsupported | 0.179 ± 0.001 | 0.485 ± 0.046 | 0.455 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.030 ± 0.000 | - | unsupported | 0.041 ± 0.000 | 0.087 ± 0.018 | 0.059 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.040 ± 0.000 | - | unsupported | 0.048 ± 0.000 | 0.115 ± 0.014 | 0.086 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.126 ± 0.000 | - | unsupported | 0.077 ± 0.000 | 0.181 ± 0.018 | 0.223 ± 0.016 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.279 ± 0.004 | - | unsupported | 0.186 ± 0.000 | 0.548 ± 0.030 | 0.469 ± 0.039 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.032 ± 0.001 | - | unsupported | 0.043 ± 0.000 | 0.108 ± 0.020 | 0.061 ± 0.015 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.042 ± 0.001 | - | unsupported | 0.051 ± 0.000 | 0.128 ± 0.013 | 0.088 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.130 ± 0.002 | - | unsupported | 0.081 ± 0.001 | 0.189 ± 0.015 | 0.224 ± 0.011 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.190 ± 0.002 | - | 0.192 ± 0.002 | 0.075 ± 0.002 | 0.177 ± 0.002 | 0.238 ± 0.004 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.019 ± 0.000 | - | 0.022 ± 0.002 | 0.023 ± 0.001 | 0.025 ± 0.001 | 0.029 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.023 ± 0.000 | - | 0.025 ± 0.000 | 0.025 ± 0.000 | 0.031 ± 0.001 | 0.044 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.060 ± 0.001 | - | 0.062 ± 0.001 | 0.035 ± 0.000 | 0.073 ± 0.019 | 0.098 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.191 ± 0.003 | - | 0.194 ± 0.003 | 0.089 ± 0.003 | 0.224 ± 0.026 | 0.259 ± 0.029 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.020 ± 0.001 | - | 0.022 ± 0.003 | 0.023 ± 0.001 | 0.024 ± 0.001 | 0.028 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.024 ± 0.000 | - | 0.026 ± 0.000 | 0.027 ± 0.001 | 0.031 ± 0.001 | 0.048 ± 0.008 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.061 ± 0.001 | - | 0.064 ± 0.001 | 0.037 ± 0.000 | 0.076 ± 0.006 | 0.103 ± 0.007 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 35.774 ± 0.882 | - | 35.661 ± 0.549 | 22.364 ± 1.486 | 13.397 ± 0.577 | 35.315 ± 7.103 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 21.614 ± 1.516 | - | 20.101 ± 0.800 | 13.329 ± 0.488 | 12.255 ± 0.984 | 337.557 ± 35.454 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 23.317 ± 1.291 | - | 19.754 ± 0.207 | 6.063 ± 0.071 | 7.428 ± 0.768 | 19.145 ± 4.255 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 15.563 ± 0.999 | - | 13.231 ± 0.104 | 6.338 ± 0.361 | 6.867 ± 0.370 | 236.230 ± 43.034 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 6.429 ± 0.096 | - | 6.184 ± 0.052 | 6.713 ± 0.157 | 7.469 ± 0.098 | 6.367 ± 0.024 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 7.202 ± 0.053 | - | 7.032 ± 0.122 | 6.426 ± 0.049 | 7.840 ± 0.153 | 311.307 ± 162.980 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 7.253 ± 0.079 | - | 7.316 ± 0.536 | 7.606 ± 0.337 | 6.681 ± 0.073 | 7.149 ± 0.209 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 9.152 ± 0.133 | - | 9.126 ± 0.047 | 7.114 ± 0.071 | 7.216 ± 0.170 | 251.345 ± 19.031 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 19.144 ± 0.103 | - | 19.295 ± 0.854 | 20.858 ± 1.234 | 27.797 ± 0.774 | 12.384 ± 0.079 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 19.200 ± 0.193 | - | 18.885 ± 0.136 | 18.946 ± 0.103 | 23.935 ± 0.177 | 631.571 ± 137.685 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.329 ± 0.003 | 0.344 ± 0.013 | unsupported | 0.287 ± 0.001 | 0.332 ± 0.004 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 6.724 ± 0.119 | 5.987 ± 0.038 | unsupported | 3.491 ± 0.020 | 4.169 ± 0.103 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.391 ± 0.020 | 0.402 ± 0.007 | unsupported | 0.288 ± 0.001 | 0.345 ± 0.003 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 5.039 ± 0.894 | 4.796 ± 0.078 | unsupported | 3.405 ± 0.158 | 3.833 ± 0.135 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 33.133 ± 0.264 | - | 33.170 ± 0.437 | 7.875 ± 0.128 | 8.407 ± 0.625 | 26.239 ± 0.379 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 19.389 ± 0.382 | - | 19.507 ± 0.211 | 8.407 ± 0.236 | 8.596 ± 0.373 | 381.331 ± 53.808 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 16.827 ± 0.189 | 16.271 ± 0.100 | 4.642 ± 0.018 | 29.823 ± 0.451 | 7.354 ± 1.039 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 11.265 ± 0.190 | 10.666 ± 0.114 | 4.970 ± 0.140 | 29.084 ± 0.227 | 139.506 ± 26.849 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 24.203 ± 0.843 | - | 23.383 ± 1.026 | 9.514 ± 0.183 | 9.834 ± 0.080 | 22.148 ± 4.772 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 16.323 ± 0.478 | - | 16.906 ± 1.087 | 9.537 ± 0.148 | 11.233 ± 0.264 | 228.781 ± 28.662 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.644 ± 0.014 | - | 0.690 ± 0.013 | 2.064 ± 0.027 | 5.616 ± 0.161 | 5.687 ± 0.051 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.608 ± 0.010 | - | 0.654 ± 0.013 | 2.210 ± 0.045 | 3.415 ± 0.150 | 7.924 ± 2.342 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 16.947 ± 0.186 | - | 16.730 ± 0.085 | 9.433 ± 0.060 | 11.337 ± 0.048 | 19.401 ± 0.308 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 16.355 ± 0.308 | - | 16.085 ± 0.182 | 10.165 ± 0.084 | 12.987 ± 1.963 | 944.787 ± 57.333 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 16.920 ± 0.114 | - | 16.660 ± 0.125 | 9.427 ± 0.016 | 11.348 ± 0.018 | 19.741 ± 0.723 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 16.218 ± 0.104 | - | 16.076 ± 0.166 | 10.185 ± 0.353 | 12.533 ± 0.341 | 943.060 ± 32.818 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.077 ± 0.006 | 1.098 ± 0.046 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 25.657 ± 0.225 | 25.605 ± 0.699 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.112 ± 0.062 | 1.075 ± 0.017 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 16.257 ± 1.493 | 16.344 ± 0.228 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.074 ± 0.007 | 1.081 ± 0.005 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 25.745 ± 0.436 | 25.675 ± 0.706 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.112 ± 0.015 | 1.079 ± 0.047 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 16.158 ± 0.289 | 16.331 ± 0.325 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 23.281 ± 1.189 | - | 20.780 ± 0.198 | 6.044 ± 0.038 | 7.220 ± 0.032 | 18.548 ± 2.167 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 15.684 ± 0.250 | - | 13.247 ± 0.023 | 6.341 ± 0.085 | 6.798 ± 0.154 | 241.166 ± 29.830 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 53.983 ± 0.542 | 56.417 ± 1.564 | 31.676 ± 0.452 | 33.130 ± 0.382 | 64.911 ± 0.657 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 57.061 ± 0.692 | 56.119 ± 3.141 | 32.816 ± 0.236 | 32.028 ± 0.276 | 1688.377 ± 141.168 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 35.596 ± 0.836 | - | 34.489 ± 0.115 | 11.347 ± 0.184 | 67.419 ± 3.820 | 28.431 ± 0.519 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 19.425 ± 0.231 | - | 19.410 ± 0.665 | 12.178 ± 0.108 | 41.538 ± 4.438 | 16.083 ± 1.085 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.173 ± 0.006 | 0.173 ± 0.001 | unsupported | 0.053 ± 0.001 | 0.033 ± 0.001 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.764 ± 0.006 | 2.769 ± 0.011 | unsupported | 0.837 ± 0.016 | 0.498 ± 0.014 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.174 ± 0.002 | 0.176 ± 0.003 | unsupported | 0.051 ± 0.003 | 0.060 ± 0.004 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 1.380 ± 0.004 | 1.456 ± 0.101 | unsupported | 0.445 ± 0.004 | 0.314 ± 0.008 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.079 ± 0.000 | 0.079 ± 0.001 | unsupported | 0.085 ± 0.002 | 0.187 ± 0.003 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.197 ± 0.005 | 1.195 ± 0.003 | unsupported | 1.362 ± 0.028 | 2.240 ± 0.006 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.050 ± 0.001 | 0.050 ± 0.003 | unsupported | 0.066 ± 0.004 | 0.137 ± 0.013 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.711 ± 0.045 | 0.687 ± 0.033 | unsupported | 0.706 ± 0.004 | 1.289 ± 0.102 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.262 ± 0.002 | 0.263 ± 0.001 | unsupported | 0.297 ± 0.001 | 0.089 ± 0.001 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 4.016 ± 0.016 | 4.215 ± 0.137 | unsupported | 4.679 ± 0.095 | 0.873 ± 0.001 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.160 ± 0.001 | 0.162 ± 0.002 | unsupported | 0.177 ± 0.001 | 0.074 ± 0.017 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 2.230 ± 0.014 | 2.281 ± 0.091 | unsupported | 2.388 ± 0.013 | 0.492 ± 0.028 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.046 ± 0.001 | 0.064 ± 0.002 | unsupported | 0.046 ± 0.001 | 0.083 ± 0.003 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.629 ± 0.017 | 0.653 ± 0.024 | unsupported | 0.532 ± 0.002 | 0.706 ± 0.020 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.051 ± 0.001 | 0.070 ± 0.002 | unsupported | 0.052 ± 0.002 | 0.101 ± 0.002 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.749 ± 0.055 | 0.804 ± 0.024 | unsupported | 0.324 ± 0.008 | 0.468 ± 0.016 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.089 ± 0.001 | 0.103 ± 0.001 | unsupported | 0.089 ± 0.002 | 0.112 ± 0.003 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.275 ± 0.035 | 1.313 ± 0.037 | unsupported | 1.334 ± 0.010 | 1.157 ± 0.029 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.094 ± 0.001 | 0.111 ± 0.002 | unsupported | 0.072 ± 0.001 | 0.121 ± 0.002 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 1.077 ± 0.005 | 1.123 ± 0.019 | unsupported | 0.723 ± 0.006 | 0.704 ± 0.023 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.123 ± 0.001 | 0.142 ± 0.001 | unsupported | unsupported | 0.130 ± 0.002 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 1.932 ± 0.037 | 1.979 ± 0.038 | unsupported | unsupported | 1.578 ± 0.014 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.128 ± 0.001 | 0.148 ± 0.001 | unsupported | unsupported | 0.133 ± 0.008 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 1.396 ± 0.009 | 1.442 ± 0.032 | unsupported | unsupported | 0.959 ± 0.037 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.012 ± 0.001 | 0.016 ± 0.000 | unsupported | 0.007 ± 0.000 | 0.050 ± 0.004 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.105 ± 0.002 | 0.110 ± 0.002 | unsupported | 0.056 ± 0.000 | 0.226 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.013 ± 0.000 | 0.018 ± 0.000 | unsupported | 0.038 ± 0.002 | 0.062 ± 0.010 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.091 ± 0.018 | 0.116 ± 0.009 | unsupported | 0.062 ± 0.004 | 0.169 ± 0.017 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.025 ± 0.001 | 0.036 ± 0.001 | unsupported | 0.026 ± 0.003 | 0.080 ± 0.002 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.349 ± 0.014 | 0.355 ± 0.009 | unsupported | 0.305 ± 0.008 | 0.610 ± 0.040 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.028 ± 0.002 | 0.040 ± 0.002 | unsupported | 0.111 ± 0.002 | 0.079 ± 0.003 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.387 ± 0.011 | 0.438 ± 0.025 | unsupported | 0.333 ± 0.015 | 0.503 ± 0.028 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.048 ± 0.001 | 0.050 ± 0.001 | unsupported | 0.076 ± 0.001 | 0.088 ± 0.002 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.748 ± 0.014 | 0.749 ± 0.006 | unsupported | 1.227 ± 0.016 | 0.865 ± 0.008 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.029 ± 0.001 | 0.030 ± 0.001 | unsupported | 0.077 ± 0.002 | 0.082 ± 0.004 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.440 ± 0.016 | 0.448 ± 0.020 | unsupported | 0.640 ± 0.005 | 0.487 ± 0.022 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.051 ± 0.001 | 0.051 ± 0.001 | unsupported | 0.078 ± 0.002 | 0.089 ± 0.001 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.785 ± 0.009 | 0.764 ± 0.013 | unsupported | 1.266 ± 0.019 | 0.941 ± 0.050 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.030 ± 0.000 | 0.030 ± 0.000 | unsupported | 0.078 ± 0.001 | 0.089 ± 0.026 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.457 ± 0.016 | 0.458 ± 0.010 | unsupported | 0.662 ± 0.009 | 0.519 ± 0.014 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.087 ± 0.000 | 0.102 ± 0.002 | unsupported | 0.099 ± 0.001 | 0.113 ± 0.003 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.172 ± 0.084 | 1.325 ± 0.129 | unsupported | 1.582 ± 0.002 | 1.129 ± 0.002 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.092 ± 0.001 | 0.109 ± 0.002 | unsupported | 0.076 ± 0.002 | 0.133 ± 0.001 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 1.102 ± 0.020 | 1.138 ± 0.077 | unsupported | 0.856 ± 0.009 | 0.693 ± 0.011 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.219 ± 0.001 | 0.220 ± 0.001 | unsupported | 0.277 ± 0.002 | 0.299 ± 0.003 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 3.356 ± 0.020 | 3.352 ± 0.012 | unsupported | 4.260 ± 0.033 | 4.411 ± 0.141 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.133 ± 0.001 | 0.134 ± 0.002 | unsupported | 0.183 ± 0.004 | 0.189 ± 0.009 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 1.891 ± 0.038 | 1.865 ± 0.025 | unsupported | 2.245 ± 0.016 | 2.149 ± 0.017 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.528 ± 0.009 | 0.643 ± 0.012 | unsupported | 0.117 ± 0.000 | 0.110 ± 0.002 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 9.392 ± 0.936 | 10.333 ± 0.130 | unsupported | 2.015 ± 0.008 | 1.001 ± 0.036 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.600 ± 0.007 | 0.683 ± 0.040 | unsupported | 0.139 ± 0.004 | 0.089 ± 0.007 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 6.614 ± 0.140 | 7.623 ± 0.058 | unsupported | 1.168 ± 0.041 | 0.850 ± 0.235 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 16.298 ± 0.631 | - | unsupported | 16.185 ± 0.772 | - | 16.040 ± 0.182 | 16.673 ± 1.081 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 16.187 ± 1.372 | - | unsupported | 16.937 ± 0.054 | - | 16.104 ± 0.938 | 14.537 ± 0.427 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 11.856 ± 0.569 | - | unsupported | 9.174 ± 0.068 | - | 9.193 ± 0.054 | 9.387 ± 0.134 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 9.997 ± 0.172 | - | unsupported | 9.608 ± 0.151 | - | 11.580 ± 0.297 | 9.500 ± 0.096 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 11.071 ± 0.409 | - | unsupported | 9.111 ± 0.032 | - | 9.032 ± 0.082 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 10.012 ± 0.288 | - | unsupported | 9.678 ± 0.652 | - | 9.600 ± 0.446 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 18.573 ± 0.417 | - | unsupported | 15.998 ± 0.475 | - | 15.349 ± 0.718 | 15.067 ± 1.343 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 15.736 ± 0.197 | - | unsupported | 15.226 ± 0.576 | - | 17.451 ± 0.987 | 16.909 ± 1.574 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 50.128 ± 3.632 | - | unsupported | 8.742 ± 0.036 | - | 47.690 ± 0.341 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 23.246 ± 0.560 | - | unsupported | 9.215 ± 0.071 | - | 23.962 ± 2.318 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 49.521 ± 0.590 | - | unsupported | 9.574 ± 0.081 | - | 46.652 ± 1.885 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 24.125 ± 1.666 | - | unsupported | 10.410 ± 0.523 | - | 21.406 ± 4.383 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 18.643 ± 1.194 | - | unsupported | 15.760 ± 0.906 | - | 16.087 ± 0.549 | 16.688 ± 0.343 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 16.531 ± 0.758 | - | unsupported | 14.835 ± 0.133 | - | 14.964 ± 0.593 | 17.950 ± 1.817 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 11.597 ± 0.237 | - | unsupported | 9.348 ± 0.284 | - | 9.090 ± 0.030 | 11.125 ± 0.054 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 9.892 ± 0.196 | - | unsupported | 9.610 ± 0.044 | - | 11.357 ± 0.137 | 11.139 ± 0.350 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.519 ± 0.003 | unsupported | unsupported | 0.268 ± 0.004 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 9.098 ± 0.519 | unsupported | unsupported | 3.006 ± 0.114 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.530 ± 0.022 | unsupported | unsupported | 0.288 ± 0.003 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 6.435 ± 0.109 | unsupported | unsupported | 2.870 ± 0.045 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 18.375 ± 0.520 | - | unsupported | 16.197 ± 0.112 | - | 16.182 ± 0.695 | 16.305 ± 0.523 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 16.227 ± 0.742 | - | unsupported | 16.580 ± 0.034 | - | 14.011 ± 0.018 | 14.465 ± 0.122 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 7.644 ± 0.204 | - | 0.001 ± 0.000 | 5.019 ± 0.404 | 4.754 ± 0.083 | 12.823 ± 0.233 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 8.150 ± 0.299 | - | 0.001 ± 0.000 | 4.212 ± 0.029 | 4.409 ± 0.048 | 13.083 ± 0.083 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 8.032 ± 0.061 | - | 8.167 ± 0.277 | 8.086 ± 0.127 | 8.281 ± 0.171 | 7.555 ± 0.688 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 7.659 ± 0.065 | - | 7.562 ± 0.884 | 7.269 ± 0.036 | 7.950 ± 0.031 | 8.582 ± 1.333 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 9.739 ± 0.285 | - | 9.738 ± 0.081 | 10.022 ± 1.063 | 15.913 ± 0.051 | 9.004 ± 1.231 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 8.821 ± 0.203 | - | 12.762 ± 0.707 | 8.468 ± 0.089 | 8.930 ± 0.130 | 12.634 ± 2.068 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 4.794 ± 0.148 | - | 4.743 ± 0.034 | 9.101 ± 0.248 | 8.086 ± 0.069 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 4.717 ± 0.052 | - | 4.913 ± 0.105 | 8.389 ± 0.011 | 7.952 ± 0.130 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 9.513 ± 0.205 | - | 0.001 ± 0.000 | 11.786 ± 0.674 | 9.737 ± 0.510 | 9.090 ± 0.539 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 9.432 ± 0.052 | - | 0.001 ± 0.000 | 9.279 ± 0.340 | 9.421 ± 0.107 | 11.134 ± 1.287 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 39.680 ± 0.590 | - | 0.001 ± 0.000 | 41.037 ± 0.487 | 48.456 ± 0.709 | 220.629 ± 87.424 | 25.972 ± 1.353 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 17.116 ± 0.590 | - | 0.002 ± 0.000 | 39.594 ± 0.114 | 26.322 ± 0.401 | 201.774 ± 9.434 | 16.012 ± 2.963 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 7.772 ± 0.036 | - | 7.359 ± 0.065 | 7.634 ± 0.216 | 5.281 ± 0.042 | 8.404 ± 0.189 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 5.238 ± 0.021 | - | 5.262 ± 0.026 | 5.618 ± 0.037 | 5.146 ± 0.102 | 10.903 ± 1.788 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 7.431 ± 0.199 | - | 7.372 ± 0.059 | 7.257 ± 0.246 | 5.256 ± 0.059 | 8.429 ± 3.677 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 5.177 ± 0.291 | - | 5.010 ± 0.094 | 5.436 ± 0.061 | 5.221 ± 0.183 | 9.270 ± 3.936 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.001 ± 0.001 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.001 ± 0.001 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 3.996333 | 249770.81 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 3.669083 | 229317.69 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 3.979792 | 248737.00 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 3.369625 | 210601.56 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 4.043042 | 252690.12 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 4.373000 | 273312.50 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 10.131750 | 633234.38 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 10.234291 | 639643.19 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 11.925750 | 745359.38 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 10.762250 | 672640.62 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 5.318958 | 332434.88 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 4.589083 | 286817.69 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 5.267958 | 329247.38 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 5.498834 | 343677.12 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 66.707959 | 4169247.44 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 55.863500 | 3491468.75 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 107.591083 | 6724442.69 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 95.795792 | 5987237.00 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 5.523000 | 345187.50 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 4.613209 | 288325.56 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 6.260167 | 391260.44 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 6.436875 | 402304.69 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 61.320708 | 3832544.25 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 54.487375 | 3405460.94 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 80.624500 | 5039031.25 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 76.728500 | 4795531.25 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 17.226333 | 1076645.81 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 17.572375 | 1098273.44 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 410.504500 | 25656531.25 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 409.680166 | 25605010.38 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 17.792792 | 1112049.50 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 17.202958 | 1075184.88 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 260.107375 | 16256710.94 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 261.497041 | 16343565.06 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 17.189042 | 1074315.12 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 17.298541 | 1081158.81 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 411.919708 | 25744981.75 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 410.804375 | 25675273.44 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 17.787708 | 1111731.75 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 17.262667 | 1078916.69 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 258.526666 | 16157916.62 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 261.293416 | 16330838.50 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.140375 | 33443.36 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.366834 | 52606.78 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 11.043208 | 172550.12 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 11.087583 | 173243.48 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 3.983459 | 497932.38 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 6.693083 | 836635.38 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 22.108666 | 2763583.25 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 22.148708 | 2768588.50 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 3.808541 | 59508.45 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.232708 | 50511.06 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 11.141541 | 174086.58 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 11.239083 | 175610.67 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.514458 | 314307.25 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.563292 | 445411.50 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 11.043000 | 1380375.00 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 11.646916 | 1455864.50 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.986000 | 187281.25 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.465500 | 85398.44 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.039875 | 78748.05 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.028416 | 78569.00 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 17.920792 | 2240099.00 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.892625 | 1361578.12 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.573042 | 1196630.25 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 9.556709 | 1194588.62 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.748625 | 136697.27 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.242125 | 66283.20 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.181333 | 49708.33 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3.184958 | 49764.97 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.310458 | 1288807.25 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.649792 | 706224.00 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.691917 | 711489.62 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.497416 | 687177.00 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.666667 | 88541.67 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 19.022417 | 297225.27 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 16.793291 | 262395.17 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 16.840250 | 263128.91 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.981750 | 872718.75 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 37.435291 | 4679411.38 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 32.128792 | 4016099.00 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 33.719708 | 4214963.50 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.719875 | 73748.05 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 11.319375 | 176865.23 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 10.265375 | 160396.48 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 10.337291 | 161520.17 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.933041 | 491630.12 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 19.100750 | 2387593.75 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 17.836833 | 2229604.12 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 18.246500 | 2280812.50 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.283625 | 82556.64 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 2.914625 | 45541.02 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.921375 | 45646.48 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.118083 | 64345.05 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.650250 | 706281.25 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.258125 | 532265.62 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.030708 | 628838.50 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.226209 | 653276.12 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.481333 | 101270.83 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.350416 | 52350.25 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.260167 | 50940.11 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.479750 | 69996.09 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.742459 | 467807.38 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.589709 | 323713.62 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.990541 | 748817.62 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 6.434041 | 804255.12 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.188958 | 112327.47 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.725875 | 89466.80 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.704083 | 89126.30 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.609750 | 103277.34 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 9.257083 | 1157135.38 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.671750 | 1333968.75 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.201375 | 1275171.88 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.504125 | 1313015.62 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.772875 | 121451.17 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.633250 | 72394.53 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.023542 | 94117.84 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.111208 | 111112.62 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.635959 | 704494.88 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.782875 | 722859.38 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.612250 | 1076531.25 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 8.985084 | 1123135.50 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.313917 | 129904.95 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 7.878125 | 123095.70 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.092750 | 142074.22 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.625333 | 1578166.62 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 15.453833 | 1931729.12 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 15.831292 | 1978911.50 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.495833 | 132747.39 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.217041 | 128391.27 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 9.488292 | 148254.56 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.672667 | 959083.38 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 11.166958 | 1395869.75 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 11.536625 | 1442078.12 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 3.196125 | 49939.45 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.474458 | 7413.41 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.738125 | 11533.20 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.036875 | 16201.17 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 1.808167 | 226020.88 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.445625 | 55703.12 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.839125 | 104890.62 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.879875 | 109984.38 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 3.998083 | 62470.05 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.427833 | 37934.89 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.816209 | 12753.27 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.125208 | 17581.38 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.348167 | 168520.88 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.493125 | 61640.62 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.728958 | 91119.75 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 0.931291 | 116411.38 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.097167 | 79643.23 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 1.694042 | 26469.41 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.590709 | 24854.83 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.287958 | 35749.34 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 4.883625 | 610453.12 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 2.438083 | 304760.38 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.792291 | 349036.38 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 2.840417 | 355052.12 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.060250 | 79066.41 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 7.114292 | 111160.81 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.782083 | 27845.05 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.552709 | 39886.08 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.021917 | 502739.62 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.662458 | 332807.25 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.098834 | 387354.25 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.506916 | 438364.50 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.627416 | 87928.38 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.865208 | 76018.88 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.064667 | 47885.42 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.186750 | 49792.97 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.919416 | 864927.00 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.814041 | 1226755.12 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.985292 | 748161.50 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.992750 | 749093.75 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.217916 | 81529.94 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.916875 | 76826.17 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.872000 | 29250.00 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.905792 | 29778.00 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.892625 | 486578.12 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.116584 | 639573.00 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.517084 | 439635.50 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.584833 | 448104.12 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.695292 | 88988.94 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.015083 | 78360.67 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.283250 | 51300.78 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.276708 | 51198.56 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.528375 | 941046.88 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.131167 | 1266395.88 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.277834 | 784729.25 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.115500 | 764437.50 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.664416 | 88506.50 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.009917 | 78279.95 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.904500 | 29757.81 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.928958 | 30139.97 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.153583 | 519197.88 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.293625 | 661703.12 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.658791 | 457348.88 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.660208 | 457526.00 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.261292 | 113457.69 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.320709 | 98761.08 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.539833 | 86559.89 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.525375 | 101958.98 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 9.033917 | 1129239.62 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 12.655042 | 1581880.25 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.376542 | 1172067.75 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.601833 | 1325229.12 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.535458 | 133366.53 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.876375 | 76193.36 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.910541 | 92352.20 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6.950000 | 108593.75 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.544458 | 693057.25 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.850375 | 856296.88 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.816458 | 1102057.25 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 9.103334 | 1137916.75 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 19.163292 | 299426.44 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.743000 | 277234.38 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 14.005375 | 218833.98 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 14.098459 | 220288.42 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 35.288375 | 4411046.88 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 34.081000 | 4260125.00 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 26.845041 | 3355630.12 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 26.813208 | 3351651.00 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.125667 | 189463.55 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 11.735833 | 183372.39 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.526084 | 133220.06 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 8.598875 | 134357.42 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 17.191250 | 2148906.25 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 17.963166 | 2245395.75 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 15.129625 | 1891203.12 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 14.922250 | 1865281.25 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.021375 | 109708.98 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.486250 | 116972.66 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 33.768750 | 527636.72 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 41.175333 | 643364.58 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 8.007292 | 1000911.50 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 16.123709 | 2015463.63 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 75.132667 | 9391583.38 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 82.662292 | 10332786.50 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.714833 | 89294.27 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 8.906166 | 139158.84 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 38.420042 | 600313.16 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 43.696875 | 682763.67 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.802750 | 850343.75 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 9.342125 | 1167765.62 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 52.911750 | 6613968.75 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 60.986625 | 7623328.12 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 4.287375 | 267960.94 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 8.298959 | 518684.94 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 48.089667 | 3005604.19 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 145.572250 | 9098265.62 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 4.612041 | 288252.56 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 8.486334 | 530395.88 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 45.927875 | 2870492.19 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 102.966125 | 6435382.81 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.012291 | 768.19 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.001250 | 78.12 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.017250 | 1078.12 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.001292 | 80.75 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.011708 | 731.75 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.012875 | 804.69 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000125 | 7.81 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.012417 | 776.06 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.017917 | 1119.81 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.010084 | 630.25 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.001750 | 109.38 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000125 | 7.81 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.011416 | 713.50 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.002500 | 156.25 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.010334 | 645.88 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.001833 | 114.56 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.011750 | 734.38 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002458 | 153.62 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `jax-cpu` is 55.0x faster than the slowest successful cell (`julia-base`, 77.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `pytorch-cpu` is 18.2x faster than the slowest successful cell (`julia-base`, 77.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-direct` is 27.5x faster than the slowest successful cell (`julia-base`, 77.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-trace` is 27.4x faster than the slowest successful cell (`julia-base`, 77.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `jax-cpu` is 23.8x faster than the slowest successful cell (`julia-base`, 454.474 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `pytorch-cpu` is 41.9x faster than the slowest successful cell (`julia-base`, 454.474 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-direct` is 55.2x faster than the slowest successful cell (`julia-base`, 454.474 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-trace` is 55.2x faster than the slowest successful cell (`julia-base`, 454.474 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 11.5x faster than the slowest successful cell (`julia-base`, 11.265 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 23.3x faster than the slowest successful cell (`julia-base`, 160.759 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 23.4x faster than the slowest successful cell (`julia-base`, 160.759 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-direct` is 28.6x faster than the slowest successful cell (`julia-base`, 160.759 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-trace` is 29.1x faster than the slowest successful cell (`julia-base`, 160.759 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 61.6x faster than the slowest successful cell (`julia-base`, 155.161 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 26.6x faster than the slowest successful cell (`julia-base`, 155.161 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-direct` is 37.3x faster than the slowest successful cell (`julia-base`, 155.161 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-trace` is 37.4x faster than the slowest successful cell (`julia-base`, 155.161 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `jax-cpu` is 94.0x faster than the slowest successful cell (`julia-base`, 591.142 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `pytorch-cpu` is 85.7x faster than the slowest successful cell (`julia-base`, 591.142 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-direct` is 75.1x faster than the slowest successful cell (`julia-base`, 591.142 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-trace` is 75.0x faster than the slowest successful cell (`julia-base`, 591.142 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 23.1x faster than the slowest successful cell (`pytorch-cpu`, 12.838 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 12.838 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 12.838 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 10.250 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 13.7x faster than the slowest successful cell (`pytorch-cpu`, 10.250 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 13.4x faster than the slowest successful cell (`pytorch-cpu`, 10.250 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 25.5x faster than the slowest successful cell (`pytorch-cpu`, 14.782 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 18.2x faster than the slowest successful cell (`pytorch-cpu`, 14.782 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 18.0x faster than the slowest successful cell (`pytorch-cpu`, 14.782 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 13.2x faster than the slowest successful cell (`pytorch-cpu`, 10.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 10.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 10.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 940.2x faster than the slowest successful cell (`julia-base`, 1.528 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 1019.6x faster than the slowest successful cell (`julia-base`, 1.402 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 17.2x faster than the slowest successful cell (`jax-cpu`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-trace` is 12.3x faster than the slowest successful cell (`jax-cpu`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `pytorch-cpu` is 14.4x faster than the slowest successful cell (`jax-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 37.0x faster than the slowest successful cell (`jax-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 21.6x faster than the slowest successful cell (`jax-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 18.7x faster than the slowest successful cell (`jax-cpu`, 0.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-trace` is 12.7x faster than the slowest successful cell (`jax-cpu`, 0.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 26.0x faster than the slowest successful cell (`pytorch-cpu`, 0.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 17.1x faster than the slowest successful cell (`pytorch-cpu`, 0.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 16.4x faster than the slowest successful cell (`pytorch-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 31.7x faster than the slowest successful cell (`pytorch-cpu`, 0.053 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 14.8x faster than the slowest successful cell (`pytorch-cpu`, 0.053 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 29.5x faster than the slowest successful cell (`pytorch-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 16.3x faster than the slowest successful cell (`pytorch-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 28.8x faster than the slowest successful cell (`pytorch-cpu`, 0.053 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 0.053 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 31.7x faster than the slowest successful cell (`pytorch-cpu`, 0.053 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 14.6x faster than the slowest successful cell (`pytorch-cpu`, 0.053 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 20.9x faster than the slowest successful cell (`pytorch-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 10.0x faster than the slowest successful cell (`pytorch-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 32.4x faster than the slowest successful cell (`pytorch-cpu`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 13.6x faster than the slowest successful cell (`pytorch-cpu`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 30.1x faster than the slowest successful cell (`pytorch-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 14.4x faster than the slowest successful cell (`pytorch-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 21.6x faster than the slowest successful cell (`pytorch-cpu`, 0.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 28.6x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 13.2x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 23.0x faster than the slowest successful cell (`pytorch-cpu`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 18.7x faster than the slowest successful cell (`jax-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 14.4x faster than the slowest successful cell (`jax-cpu`, 0.043 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.043 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 12.3x faster than the slowest successful cell (`jax-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 22.4x faster than the slowest successful cell (`jax-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-trace` is 16.9x faster than the slowest successful cell (`jax-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 16.5x faster than the slowest successful cell (`pytorch-cpu`, 0.053 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 13.7x faster than the slowest successful cell (`pytorch-cpu`, 0.053 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 17.7x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 15.1x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 23.3x faster than the slowest successful cell (`pytorch-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 15.3x faster than the slowest successful cell (`pytorch-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 16.2x faster than the slowest successful cell (`jax-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 15.1x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 15.2x faster than the slowest successful cell (`pytorch-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 18.3x faster than the slowest successful cell (`pytorch-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 13.1x faster than the slowest successful cell (`pytorch-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `jax-cpu` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 22.8x faster than the slowest successful cell (`pytorch-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 14.4x faster than the slowest successful cell (`pytorch-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 19.4x faster than the slowest successful cell (`pytorch-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.024 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 15.2x faster than the slowest successful cell (`pytorch-cpu`, 0.037 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.032 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x32x32,rhs=1`): `tenferro-direct` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x32x32,rhs=1`): `tenferro-direct` is 11.9x faster than the slowest successful cell (`jax-cpu`, 0.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x2x2,rhs=1`): `tenferro-direct` is 18.5x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x4x4,rhs=1`): `tenferro-direct` is 15.3x faster than the slowest successful cell (`pytorch-cpu`, 0.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svdvals` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 12.6x faster than the slowest successful cell (`jax-cpu`, 0.035 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `jax-cpu` is 27.5x faster than the slowest successful cell (`julia-base`, 337.557 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `pytorch-cpu` is 25.3x faster than the slowest successful cell (`julia-base`, 337.557 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-direct` is 15.6x faster than the slowest successful cell (`julia-base`, 337.557 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-trace` is 16.8x faster than the slowest successful cell (`julia-base`, 337.557 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 34.4x faster than the slowest successful cell (`julia-base`, 236.230 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 37.3x faster than the slowest successful cell (`julia-base`, 236.230 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 15.2x faster than the slowest successful cell (`julia-base`, 236.230 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 17.9x faster than the slowest successful cell (`julia-base`, 236.230 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `jax-cpu` is 39.7x faster than the slowest successful cell (`julia-base`, 311.307 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `pytorch-cpu` is 48.4x faster than the slowest successful cell (`julia-base`, 311.307 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-direct` is 43.2x faster than the slowest successful cell (`julia-base`, 311.307 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-trace` is 44.3x faster than the slowest successful cell (`julia-base`, 311.307 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `jax-cpu` is 34.8x faster than the slowest successful cell (`julia-base`, 251.345 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `pytorch-cpu` is 35.3x faster than the slowest successful cell (`julia-base`, 251.345 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-direct` is 27.5x faster than the slowest successful cell (`julia-base`, 251.345 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-trace` is 27.5x faster than the slowest successful cell (`julia-base`, 251.345 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `jax-cpu` is 26.4x faster than the slowest successful cell (`julia-base`, 631.571 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 33.3x faster than the slowest successful cell (`julia-base`, 631.571 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-direct` is 32.9x faster than the slowest successful cell (`julia-base`, 631.571 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-trace` is 33.4x faster than the slowest successful cell (`julia-base`, 631.571 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 44.4x faster than the slowest successful cell (`julia-base`, 381.331 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 45.4x faster than the slowest successful cell (`julia-base`, 381.331 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-direct` is 19.7x faster than the slowest successful cell (`julia-base`, 381.331 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-trace` is 19.5x faster than the slowest successful cell (`julia-base`, 381.331 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 28.1x faster than the slowest successful cell (`julia-base`, 139.506 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-eager` is 12.4x faster than the slowest successful cell (`julia-base`, 139.506 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-trace` is 13.1x faster than the slowest successful cell (`julia-base`, 139.506 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 20.4x faster than the slowest successful cell (`julia-base`, 228.781 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 24.0x faster than the slowest successful cell (`julia-base`, 228.781 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 14.0x faster than the slowest successful cell (`julia-base`, 228.781 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 13.5x faster than the slowest successful cell (`julia-base`, 228.781 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 13.0x faster than the slowest successful cell (`julia-base`, 7.924 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`julia-base`, 7.924 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `jax-cpu` is 72.7x faster than the slowest successful cell (`julia-base`, 944.787 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 92.9x faster than the slowest successful cell (`julia-base`, 944.787 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 57.8x faster than the slowest successful cell (`julia-base`, 944.787 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 58.7x faster than the slowest successful cell (`julia-base`, 944.787 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `jax-cpu` is 75.2x faster than the slowest successful cell (`julia-base`, 943.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 92.6x faster than the slowest successful cell (`julia-base`, 943.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 58.1x faster than the slowest successful cell (`julia-base`, 943.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 58.7x faster than the slowest successful cell (`julia-base`, 943.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 35.5x faster than the slowest successful cell (`julia-base`, 241.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 38.0x faster than the slowest successful cell (`julia-base`, 241.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 15.4x faster than the slowest successful cell (`julia-base`, 241.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 18.2x faster than the slowest successful cell (`julia-base`, 241.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 52.7x faster than the slowest successful cell (`julia-base`, 1688.377 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 51.5x faster than the slowest successful cell (`julia-base`, 1688.377 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-eager` is 29.6x faster than the slowest successful cell (`julia-base`, 1688.377 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-trace` is 30.1x faster than the slowest successful cell (`julia-base`, 1688.377 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 10.3x faster than the slowest successful cell (`tenferro-eager`, 10.333 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 9925.0x faster than the slowest successful cell (`julia-base`, 12.823 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 9807.2x faster than the slowest successful cell (`julia-base`, 13.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 8834.8x faster than the slowest successful cell (`pytorch-cpu`, 11.786 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 8617.5x faster than the slowest successful cell (`julia-base`, 11.134 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 165389.1x faster than the slowest successful cell (`julia-base`, 220.629 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `strided-jl` is 12.6x faster than the slowest successful cell (`julia-base`, 201.774 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-direct` is 11.8x faster than the slowest successful cell (`julia-base`, 201.774 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 127463.2x faster than the slowest successful cell (`julia-base`, 201.774 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 13.3x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 161.0x faster than the slowest successful cell (`tenferro-direct`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 140.0x faster than the slowest successful cell (`tenferro-direct`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 126.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 89.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 129.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 146.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 10319.4 GB/s at 0.002 ms. The conservative physical lower bound is 0.168 ms, so the cell is 103.2x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 12195.7 GB/s at 0.001 ms. The conservative physical lower bound is 0.042 ms, so the cell is 30.5x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 207767.4 GB/s at 0.001 ms. The conservative physical lower bound is 2.684 ms, so the cell is 2077.7x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 201226.0 GB/s at 0.001 ms. The conservative physical lower bound is 0.671 ms, so the cell is 503.1x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 201226.0 GB/s at 0.001 ms. The conservative physical lower bound is 2.684 ms, so the cell is 2012.3x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 207767.4 GB/s at 0.001 ms. The conservative physical lower bound is 0.671 ms, so the cell is 519.4x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 100613.0 GB/s at 0.001 ms. The conservative physical lower bound is 1.342 ms, so the cell is 1006.1x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 84786.9 GB/s at 0.002 ms. The conservative physical lower bound is 0.336 ms, so the cell is 212.0x faster than the bound.

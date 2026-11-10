# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/amd-cpu/cpu/public_api/20261110_190512/run.yaml`
- Timestamp: `2026-11-10T19:05:12.155925+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/amd-cpu/cpu/public_api/20261110_190512`.

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

- CSV: `data/results/amd-cpu/cpu/public_api/20261110_190512/cpu_public_api_t1_20261110_190512.csv`
- CSV: `data/results/amd-cpu/cpu/public_api/20261110_190512/cpu_public_api_t4_20261110_190512.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 3.514 ± 0.015 | - | 3.598 ± 0.019 | 3.739 ± 0.041 | 3.601 ± 0.024 | 5.524 ± 2.036 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 3.825 ± 0.012 | - | 3.724 ± 0.032 | 1.883 ± 0.011 | 2.602 ± 0.126 | 4.774 ± 2.204 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 105.934 ± 5.992 | - | 105.805 ± 8.545 | 100.041 ± 2.040 | 100.637 ± 13.820 | 112.672 ± 5.868 | 118.722 ± 6.004 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 36.736 ± 2.280 | - | 119.575 ± 11.109 | 35.299 ± 0.861 | 50.556 ± 1.995 | 113.591 ± 6.363 | 52.527 ± 2.761 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 88.059 ± 2.997 | - | 92.617 ± 4.405 | 85.475 ± 0.405 | 81.909 ± 3.876 | 100.062 ± 3.544 | 105.468 ± 3.363 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 33.005 ± 3.559 | - | 107.297 ± 11.726 | 26.114 ± 0.107 | 40.389 ± 1.540 | 99.882 ± 3.540 | 46.613 ± 0.668 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 40.328 ± 0.125 | - | 40.461 ± 0.115 | 40.127 ± 0.165 | 59.134 ± 0.162 | 43.271 ± 0.672 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 40.719 ± 0.084 | - | 40.811 ± 0.210 | 25.958 ± 0.055 | 25.537 ± 0.226 | 21.112 ± 0.114 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 41.352 ± 0.203 | - | unsupported | 40.657 ± 0.056 | 59.803 ± 0.305 | 41.291 ± 2.043 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 41.543 ± 0.080 | - | unsupported | 26.025 ± 0.307 | 26.633 ± 0.995 | 20.875 ± 3.283 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 9.056 ± 0.111 | - | 9.037 ± 0.041 | 9.591 ± 0.098 | 9.692 ± 0.060 | 9.553 ± 0.229 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 11.836 ± 0.033 | - | 11.758 ± 0.027 | 11.662 ± 0.071 | 14.730 ± 0.039 | 14.749 ± 0.032 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 60.074 ± 0.569 | - | 59.347 ± 0.215 | 93.451 ± 3.400 | 84.277 ± 0.723 | 65.450 ± 3.144 | 65.147 ± 3.558 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 25.767 ± 0.191 | - | 71.998 ± 0.005 | 37.847 ± 0.400 | 41.901 ± 0.674 | 63.588 ± 2.714 | 29.485 ± 9.502 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 90.921 ± 0.161 | - | 90.873 ± 0.269 | 124.127 ± 3.219 | 139.440 ± 2.719 | 114.827 ± 3.295 | 117.328 ± 1.276 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 38.834 ± 0.239 | - | 107.998 ± 0.007 | 51.979 ± 0.346 | 60.521 ± 1.268 | 114.536 ± 1.521 | 49.279 ± 0.962 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 61.311 ± 3.037 | - | 61.536 ± 3.016 | 60.728 ± 0.528 | 60.236 ± 2.322 | 64.988 ± 3.606 | 69.990 ± 4.013 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 20.217 ± 0.323 | - | 83.669 ± 0.664 | 20.715 ± 0.872 | 30.283 ± 0.452 | 67.521 ± 1.335 | 29.194 ± 2.527 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.228 ± 0.039 | - | 15.397 ± 0.097 | 18.199 ± 0.082 | 15.797 ± 0.481 | 3.460 ± 0.004 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 0.641 ± 0.097 | - | 59.996 ± 0.004 | 9.464 ± 0.064 | 2.696 ± 0.144 | 3.471 ± 0.015 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 6.951 ± 0.108 | - | 7.097 ± 0.033 | 7.993 ± 0.048 | 6.712 ± 0.055 | 8.093 ± 0.083 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 125.994 ± 11.969 | - | 93.995 ± 12.013 | 4.984 ± 0.094 | 6.338 ± 0.388 | 6.351 ± 0.665 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 4.574 ± 0.113 | - | 4.701 ± 0.152 | 4.710 ± 0.059 | 4.527 ± 0.039 | 4.299 ± 0.090 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 88.061 ± 12.017 | - | 120.990 ± 14.555 | 2.913 ± 0.029 | 3.297 ± 0.019 | 3.781 ± 0.113 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 5.196 ± 0.071 | - | 5.225 ± 0.035 | 6.314 ± 0.052 | 4.389 ± 0.027 | 4.297 ± 0.046 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 9.555 ± 0.288 | - | 9.435 ± 0.025 | 4.995 ± 0.060 | 5.367 ± 0.015 | 5.084 ± 0.084 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 40.576 ± 0.298 | - | 40.486 ± 0.294 | 40.240 ± 0.137 | 59.276 ± 0.534 | 40.639 ± 0.118 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 40.672 ± 0.134 | - | 41.060 ± 5.941 | 26.031 ± 0.175 | 25.437 ± 0.933 | 20.442 ± 0.039 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 41.957 ± 0.240 | - | unsupported | 45.172 ± 0.303 | 44.927 ± 0.178 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 44.911 ± 1.015 | - | unsupported | 22.133 ± 0.140 | 27.320 ± 1.025 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 102.950 ± 7.502 | - | 103.397 ± 12.063 | 99.283 ± 3.386 | 110.719 ± 10.377 | 116.592 ± 9.853 | 155.930 ± 12.213 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 34.848 ± 0.584 | - | 117.777 ± 2.544 | 35.221 ± 2.622 | 56.347 ± 1.657 | 113.599 ± 5.670 | 66.736 ± 2.949 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 116.713 ± 6.089 | - | 116.464 ± 7.821 | 114.633 ± 5.235 | 128.972 ± 11.326 | 124.615 ± 9.222 | 172.996 ± 1.471 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 39.478 ± 0.383 | - | 129.623 ± 0.345 | 39.788 ± 2.472 | 66.005 ± 1.093 | 122.356 ± 8.399 | 73.534 ± 3.583 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 13.647 ± 1.118 | - | 38.928 ± 0.315 | 54.429 ± 1.392 | 42.488 ± 0.430 | 63.243 ± 7.123 | 75.348 ± 12.574 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 7.758 ± 0.123 | - | 53.999 ± 48.587 | 24.630 ± 1.117 | 21.840 ± 0.653 | 63.636 ± 0.708 | 27.215 ± 9.106 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 7.912 ± 0.162 | - | 8.138 ± 0.053 | 30.166 ± 1.610 | 32.908 ± 0.128 | 38.596 ± 9.033 | 44.081 ± 10.603 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 6.525 ± 0.117 | - | 23.997 ± 0.005 | 10.760 ± 0.615 | 17.383 ± 0.592 | 38.056 ± 7.431 | 23.548 ± 9.567 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 17.037 ± 0.162 | - | 17.108 ± 0.198 | 41.969 ± 0.718 | 40.513 ± 0.268 | 19.513 ± 0.511 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 14.639 ± 0.038 | - | 36.000 ± 0.004 | 16.895 ± 0.195 | 22.675 ± 0.778 | 18.173 ± 0.158 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 13.439 ± 0.022 | - | 13.598 ± 0.031 | 50.032 ± 0.717 | 92.984 ± 0.089 | 50.681 ± 11.704 | 56.600 ± 9.947 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 5.890 ± 0.012 | - | 35.998 ± 0.004 | 18.623 ± 0.729 | 47.263 ± 0.972 | 51.060 ± 19.439 | 33.023 ± 10.764 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 118.353 ± 4.474 | - | 111.302 ± 2.995 | 106.167 ± 4.771 | 128.770 ± 8.724 | 128.945 ± 7.255 | 173.388 ± 5.679 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 39.596 ± 0.267 | - | 129.784 ± 0.618 | 38.852 ± 1.514 | 68.706 ± 6.670 | 130.560 ± 5.296 | 72.770 ± 1.824 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 7.969 ± 0.012 | - | 8.097 ± 0.055 | 38.586 ± 0.890 | 33.717 ± 1.847 | 47.870 ± 9.881 | 53.028 ± 10.210 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 3.829 ± 0.027 | - | 24.000 ± 0.007 | 13.973 ± 0.227 | 17.088 ± 1.409 | 57.655 ± 6.998 | 31.498 ± 10.567 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 10.952 ± 0.036 | - | 10.906 ± 0.211 | 21.229 ± 0.903 | 19.185 ± 0.162 | 30.755 ± 7.037 | 31.700 ± 9.389 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 4.632 ± 0.044 | - | 23.998 ± 0.004 | 7.387 ± 0.283 | 10.283 ± 0.739 | 30.599 ± 0.231 | 14.308 ± 9.369 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 13.870 ± 0.027 | - | 13.958 ± 0.039 | 42.126 ± 0.304 | 53.900 ± 1.106 | 60.943 ± 18.123 | 68.393 ± 7.551 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 6.869 ± 0.018 | - | 36.002 ± 0.014 | 15.548 ± 0.564 | 27.782 ± 1.009 | 60.431 ± 16.696 | 27.182 ± 14.072 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 7.020 ± 0.016 | - | 6.968 ± 0.031 | 18.545 ± 0.858 | 30.840 ± 1.057 | 36.461 ± 9.037 | 36.792 ± 8.240 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 3.409 ± 0.018 | - | 23.998 ± 0.017 | 6.707 ± 0.206 | 16.344 ± 0.145 | 35.820 ± 3.725 | 17.274 ± 9.299 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.287 ± 0.009 | 0.305 ± 0.004 | unsupported | unsupported | 0.373 ± 0.009 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.208 ± 0.263 | 111.993 ± 1.562 | unsupported | unsupported | 0.315 ± 0.017 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 118.370 ± 6.674 | - | 117.362 ± 6.986 | 113.127 ± 1.745 | 113.618 ± 0.960 | 131.419 ± 8.138 | 178.424 ± 4.160 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 39.030 ± 2.355 | - | 130.067 ± 12.518 | 38.495 ± 0.120 | 61.621 ± 5.334 | 128.419 ± 4.656 | 75.474 ± 12.774 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 117.171 ± 7.675 | - | 117.503 ± 7.969 | 112.825 ± 1.186 | 112.972 ± 0.589 | 132.822 ± 5.422 | 177.964 ± 5.576 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 38.796 ± 0.283 | - | 129.924 ± 1.185 | 38.555 ± 0.332 | 61.891 ± 5.274 | 131.629 ± 3.035 | 73.043 ± 1.100 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 113.318 ± 3.553 | - | 114.737 ± 3.311 | 112.320 ± 5.430 | 115.928 ± 1.003 | 129.469 ± 0.855 | 174.383 ± 5.878 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 39.652 ± 0.208 | - | 129.936 ± 0.864 | 40.962 ± 1.600 | 64.361 ± 1.115 | 129.878 ± 3.466 | 72.235 ± 2.332 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 102.334 ± 8.038 | - | 103.073 ± 4.590 | 99.338 ± 4.837 | 110.168 ± 3.772 | 117.408 ± 8.505 | 153.131 ± 5.452 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 35.127 ± 0.335 | - | 119.010 ± 2.271 | 34.670 ± 1.381 | 56.150 ± 2.384 | 116.885 ± 5.882 | 67.067 ± 0.959 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 45.958 ± 1.810 | - | 45.566 ± 0.210 | 33.469 ± 0.774 | 56.656 ± 0.158 | 108.179 ± 0.878 | 96.434 ± 10.382 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 20.106 ± 0.030 | - | 59.997 ± 0.005 | 12.564 ± 0.080 | 29.534 ± 0.251 | 107.083 ± 1.327 | 37.033 ± 8.440 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 11.527 ± 1.388 | - | 11.522 ± 0.062 | 7.660 ± 0.157 | 37.868 ± 0.031 | 8.022 ± 0.190 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 7.004 ± 0.154 | - | 23.999 ± 0.006 | 7.108 ± 0.299 | 18.980 ± 0.050 | 7.918 ± 0.125 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.489 ± 0.008 | - | 1.485 ± 0.011 | 21.719 ± 0.226 | 2.099 ± 0.016 | 0.901 ± 0.041 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.686 ± 0.009 | - | 24.000 ± 0.006 | 10.056 ± 0.215 | 1.237 ± 2.226 | 0.915 ± 0.019 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 1.582 ± 0.120 | - | 1.574 ± 0.207 | 4.648 ± 0.035 | 3.415 ± 0.010 | 0.980 ± 0.161 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 0.929 ± 0.178 | - | 23.998 ± 0.002 | 2.352 ± 0.043 | 2.556 ± 0.018 | 1.485 ± 0.429 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 11.062 ± 0.019 | - | 11.077 ± 0.026 | 7.530 ± 0.054 | 37.881 ± 0.202 | 8.071 ± 0.106 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 6.830 ± 0.079 | - | 23.999 ± 0.006 | 6.923 ± 0.187 | 18.947 ± 0.488 | 7.909 ± 0.134 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.418 ± 0.017 | - | 1.426 ± 0.016 | 21.792 ± 0.063 | 2.093 ± 0.025 | 0.897 ± 0.010 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.690 ± 0.031 | - | 23.996 ± 0.007 | 9.365 ± 0.402 | 1.559 ± 0.214 | 0.941 ± 0.012 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 8.502 ± 0.323 | - | 8.286 ± 0.204 | 13.892 ± 0.032 | 13.527 ± 0.016 | 7.410 ± 0.025 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 4.269 ± 0.006 | - | 35.998 ± 0.007 | 7.011 ± 0.037 | 6.940 ± 0.845 | 7.592 ± 0.163 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 7.532 ± 0.109 | - | 7.689 ± 0.298 | 7.529 ± 0.201 | 27.445 ± 0.471 | 7.709 ± 0.157 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 7.238 ± 0.169 | - | 24.000 ± 0.004 | 7.228 ± 0.096 | 14.008 ± 0.028 | 7.981 ± 0.451 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.616 ± 0.020 | - | 0.611 ± 0.039 | 2.039 ± 0.137 | 1.648 ± 0.030 | 0.617 ± 0.031 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.312 ± 0.108 | - | 23.997 ± 0.012 | 0.955 ± 0.097 | 1.040 ± 0.073 | 0.661 ± 0.038 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.104 ± 0.098 | - | 1.167 ± 0.139 | 0.567 ± 0.028 | 1.068 ± 0.020 | 0.782 ± 0.106 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.663 ± 0.052 | - | 23.999 ± 0.003 | 0.231 ± 0.017 | 0.725 ± 0.330 | 0.867 ± 0.328 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 7.274 ± 0.061 | - | 7.501 ± 0.210 | 7.604 ± 0.184 | 27.709 ± 0.126 | 7.758 ± 0.088 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 7.212 ± 0.060 | - | 24.001 ± 0.008 | 7.141 ± 0.265 | 13.967 ± 0.065 | 7.765 ± 0.073 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.619 ± 0.019 | - | 0.606 ± 0.017 | 1.904 ± 0.058 | 1.591 ± 0.020 | 0.566 ± 0.059 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.322 ± 0.082 | - | 23.997 ± 0.002 | 0.962 ± 0.041 | 1.371 ± 0.151 | 0.631 ± 0.083 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.141 ± 0.096 | - | 1.195 ± 0.184 | 0.594 ± 0.032 | 1.048 ± 0.007 | 0.728 ± 0.034 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.674 ± 0.054 | - | 23.998 ± 0.197 | 0.247 ± 0.024 | 0.688 ± 0.083 | 0.820 ± 0.093 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.610 ± 0.014 | 0.567 ± 0.017 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.246 ± 0.011 | 23.999 ± 0.063 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 30.866 ± 0.322 | - | 34.177 ± 0.185 | 49.875 ± 1.959 | 56.293 ± 0.526 | 60.205 ± 12.598 | 68.132 ± 7.599 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 14.755 ± 0.105 | - | 47.995 ± 1.005 | 17.846 ± 0.414 | 31.493 ± 0.509 | 53.661 ± 8.219 | 33.565 ± 9.336 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 111.342 ± 0.483 | - | 113.333 ± 7.896 | 116.903 ± 0.148 | 202.165 ± 10.944 | 227.381 ± 5.672 | 228.272 ± 5.188 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 35.970 ± 0.373 | - | 131.218 ± 2.047 | 36.554 ± 0.639 | 86.253 ± 5.649 | 222.079 ± 2.128 | 92.139 ± 1.338 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 112.035 ± 6.356 | - | 116.652 ± 7.140 | 131.654 ± 4.821 | 134.008 ± 2.148 | 134.341 ± 5.184 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 42.016 ± 2.525 | - | 132.748 ± 12.425 | 44.025 ± 0.496 | 69.145 ± 1.990 | 133.836 ± 4.960 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 100.768 ± 7.912 | - | 105.717 ± 8.855 | 104.079 ± 4.694 | 95.830 ± 1.004 | 119.192 ± 10.038 | 156.492 ± 6.055 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 36.986 ± 1.256 | - | 130.941 ± 16.020 | 36.717 ± 1.681 | 58.549 ± 2.641 | 114.419 ± 14.769 | 67.424 ± 5.106 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 10.791 ± 0.017 | - | 10.986 ± 0.047 | 52.021 ± 0.771 | 93.440 ± 0.162 | 50.224 ± 6.007 | 54.823 ± 9.905 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 4.998 ± 0.007 | - | 23.998 ± 0.010 | 18.888 ± 0.301 | 47.358 ± 1.281 | 52.836 ± 13.023 | 23.074 ± 13.838 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 102.583 ± 4.567 | - | 104.661 ± 8.787 | 150.638 ± 0.353 | 112.157 ± 7.175 | 178.730 ± 4.735 | 180.721 ± 7.231 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 34.535 ± 0.705 | - | 119.498 ± 1.909 | 54.918 ± 0.413 | 58.793 ± 4.273 | 176.378 ± 5.848 | 73.831 ± 0.434 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 110.693 ± 6.152 | - | 116.132 ± 7.174 | 114.119 ± 5.510 | 129.169 ± 13.206 | 123.952 ± 7.339 | 175.802 ± 2.344 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 40.084 ± 0.231 | - | 129.801 ± 0.896 | 40.119 ± 1.322 | 67.569 ± 2.746 | 130.421 ± 2.565 | 75.262 ± 2.006 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 27.065 ± 0.041 | - | 27.608 ± 0.099 | 68.405 ± 1.195 | 30.373 ± 0.337 | 59.455 ± 11.287 | 64.285 ± 19.643 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 11.844 ± 0.041 | - | 47.997 ± 0.001 | 24.043 ± 0.224 | 15.833 ± 0.155 | 60.303 ± 9.437 | 26.231 ± 13.142 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 111.549 ± 6.748 | 111.668 ± 7.082 | unsupported | 135.906 ± 4.625 | 136.014 ± 1.003 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 42.654 ± 2.721 | 142.640 ± 5.748 | unsupported | 44.052 ± 0.721 | 69.817 ± 0.683 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.707 ± 0.033 | - | 0.693 ± 0.065 | 0.633 ± 0.019 | 0.644 ± 0.049 | 6.574 ± 3.926 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.283 ± 0.047 | - | 12.000 ± 0.007 | 0.305 ± 0.021 | 0.677 ± 0.117 | 7.784 ± 5.252 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.633 ± 0.112 | - | 0.618 ± 0.029 | 0.617 ± 0.019 | 0.644 ± 0.089 | 6.902 ± 4.334 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.405 ± 0.076 | - | 12.000 ± 0.009 | 0.294 ± 0.075 | 0.656 ± 0.181 | 1.011 ± 6.273 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.021 ± 0.145 | - | unsupported | 0.967 ± 0.068 | 0.899 ± 0.009 | 6.242 ± 5.748 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.587 ± 0.161 | - | unsupported | 0.623 ± 0.136 | 0.939 ± 0.102 | 7.855 ± 4.525 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.264 ± 0.010 | - | 0.327 ± 0.014 | 0.638 ± 0.010 | 0.261 ± 0.012 | 1.063 ± 0.051 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.284 ± 0.118 | - | 47.999 ± 0.010 | 0.319 ± 0.012 | 0.253 ± 0.014 | 1.109 ± 0.124 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.667 ± 0.146 | - | 0.619 ± 0.075 | 0.877 ± 0.095 | 0.886 ± 0.038 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.325 ± 0.173 | - | 11.999 ± 0.007 | 0.551 ± 0.241 | 0.842 ± 0.026 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.770 ± 0.093 | - | 0.652 ± 0.025 | 0.779 ± 0.060 | 0.735 ± 0.035 | 1.168 ± 6.054 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.285 ± 0.094 | - | 11.995 ± 0.014 | 0.313 ± 0.045 | 0.740 ± 0.148 | 6.537 ± 5.903 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.601 ± 0.010 | - | 0.656 ± 0.006 | 0.305 ± 0.013 | 0.302 ± 0.018 | 0.441 ± 0.878 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.826 ± 0.014 | - | 35.966 ± 0.328 | 0.364 ± 0.012 | 0.307 ± 0.020 | 1.136 ± 0.136 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.301 ± 0.057 | - | 0.002 ± 0.000 | 1.321 ± 0.031 | 1.261 ± 0.367 | 7.911 ± 5.680 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.703 ± 0.091 | - | 0.002 ± 0.000 | 0.871 ± 0.180 | 1.051 ± 0.149 | 8.282 ± 0.663 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.042 ± 0.000 | - | 0.044 ± 0.000 | 0.058 ± 0.000 | 0.077 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.072 ± 0.000 | - | 0.075 ± 0.001 | 0.080 ± 0.001 | 0.108 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.190 ± 0.010 | - | 0.197 ± 0.003 | 0.231 ± 0.011 | 0.250 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.067 ± 0.002 | - | 0.069 ± 0.000 | 0.109 ± 0.002 | 0.091 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.004 ± 0.001 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.001 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.551 ± 0.011 | - | 0.549 ± 0.010 | 0.911 ± 0.004 | 0.568 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.031 ± 0.001 | 0.043 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.042 ± 0.000 | - | 0.044 ± 0.000 | 0.069 ± 0.000 | 0.058 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.073 ± 0.000 | - | 0.076 ± 0.001 | 0.100 ± 0.001 | 0.105 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.187 ± 0.002 | - | 0.191 ± 0.011 | 0.244 ± 0.011 | 0.267 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.069 ± 0.001 | - | 0.070 ± 0.000 | 0.142 ± 0.002 | 0.095 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.010 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.012 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.556 ± 0.015 | - | 0.585 ± 3.770 | 1.113 ± 0.015 | 0.997 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.020 ± 0.000 | - | 0.022 ± 0.000 | 0.045 ± 0.001 | 0.035 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.414 ± 0.008 | - | 0.414 ± 0.010 | 0.533 ± 0.010 | 0.631 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.647 ± 0.004 | - | 1.649 ± 0.008 | 1.697 ± 0.021 | 2.091 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 5.032 ± 0.012 | - | 5.021 ± 0.008 | 5.171 ± 0.013 | 7.138 ± 0.053 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 2.974 ± 0.008 | - | 2.966 ± 0.005 | 2.729 ± 0.032 | 2.167 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.088 ± 0.001 | - | 0.090 ± 0.001 | 0.094 ± 0.002 | 0.135 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.006 ± 0.000 | - | 0.012 ± 0.003 | 0.012 ± 0.001 | 0.036 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.016 ± 0.000 | - | 0.018 ± 0.000 | 0.022 ± 0.001 | 0.045 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.021 ± 0.000 | - | 0.023 ± 0.000 | 0.026 ± 0.000 | 0.067 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 23.808 ± 0.050 | - | 23.765 ± 0.069 | 21.563 ± 0.122 | 18.463 ± 0.282 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.687 ± 0.006 | - | 0.689 ± 0.005 | 0.699 ± 0.014 | 0.793 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.019 ± 0.000 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.041 ± 0.000 | - | 0.043 ± 0.000 | 0.047 ± 0.001 | 0.094 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.419 ± 0.024 | - | 0.421 ± 0.017 | 0.789 ± 0.260 | 0.671 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 1.666 ± 0.023 | - | 1.699 ± 0.126 | 1.681 ± 0.009 | 2.090 ± 0.173 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 5.069 ± 0.029 | - | 5.073 ± 0.026 | 7.398 ± 0.011 | 5.089 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 5.536 ± 0.007 | - | 5.543 ± 0.016 | 1.934 ± 0.015 | 2.881 ± 0.088 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.089 ± 0.002 | - | 0.091 ± 0.002 | 0.139 ± 0.003 | 0.117 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.017 ± 0.000 | - | 0.018 ± 0.000 | 0.021 ± 0.000 | 0.052 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.022 ± 0.000 | - | 0.024 ± 0.001 | 0.026 ± 0.001 | 0.061 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 44.299 ± 0.097 | - | 44.152 ± 0.150 | 15.520 ± 0.455 | 22.442 ± 5.995 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.005 ± 0.003 | - | 0.007 ± 0.000 | 0.014 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.693 ± 0.027 | - | 0.704 ± 3.771 | 1.039 ± 0.004 | 0.896 ± 0.312 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.015 ± 0.000 | - | 0.017 ± 0.000 | 0.019 ± 0.000 | 0.027 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.041 ± 0.000 | - | 0.044 ± 0.000 | 0.047 ± 0.000 | 0.086 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.480 ± 0.012 | - | 0.483 ± 0.012 | 0.330 ± 0.009 | 0.304 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.718 ± 0.005 | - | 1.722 ± 0.013 | 1.479 ± 0.010 | 1.469 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 4.522 ± 0.018 | - | 4.513 ± 0.016 | 4.381 ± 0.015 | 5.005 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.094 ± 0.021 | - | 1.085 ± 0.021 | 1.096 ± 0.023 | 1.140 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.002 ± 0.001 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.057 ± 0.000 | - | 0.058 ± 0.001 | 0.066 ± 0.002 | 0.112 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.015 ± 0.000 | - | 0.017 ± 0.000 | 0.019 ± 0.001 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.023 ± 0.001 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 8.516 ± 0.146 | - | 8.590 ± 0.136 | 8.556 ± 0.066 | 8.866 ± 0.072 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.436 ± 0.008 | - | 0.439 ± 0.009 | 0.484 ± 0.011 | 0.577 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.014 ± 0.000 | - | 0.017 ± 0.000 | 0.016 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.037 ± 0.000 | - | 0.039 ± 0.001 | 0.040 ± 0.001 | 0.078 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.458 ± 0.024 | - | 0.468 ± 0.018 | 0.539 ± 0.011 | 0.278 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.707 ± 0.023 | - | 1.738 ± 0.123 | 1.463 ± 0.004 | 0.944 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 4.542 ± 0.132 | - | 4.532 ± 0.114 | 6.233 ± 0.011 | 2.967 ± 1.348 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 2.686 ± 0.056 | - | 2.681 ± 0.021 | 1.014 ± 0.011 | 1.341 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.122 ± 0.031 | - | 0.118 ± 0.010 | 0.098 ± 0.001 | 0.084 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.003 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.013 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.015 ± 0.000 | - | 0.017 ± 0.000 | 0.018 ± 0.000 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.003 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.016 ± 0.004 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.019 ± 0.000 | - | 0.022 ± 0.000 | 0.022 ± 0.001 | 0.031 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 21.075 ± 0.106 | - | 21.151 ± 0.064 | 7.948 ± 0.173 | 10.487 ± 0.308 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.890 ± 0.009 | - | 0.900 ± 0.014 | 0.722 ± 0.007 | 0.350 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.015 ± 0.000 | - | 0.017 ± 0.000 | 0.024 ± 0.003 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.037 ± 0.000 | - | 0.040 ± 0.001 | 0.040 ± 0.000 | 0.056 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.181 ± 0.008 | - | 0.182 ± 0.013 | 0.169 ± 0.018 | 0.302 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 0.832 ± 0.009 | - | 0.836 ± 0.006 | 0.729 ± 0.013 | 1.465 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 2.415 ± 0.008 | - | 2.416 ± 0.008 | 2.292 ± 0.019 | 5.044 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.423 ± 0.013 | - | 0.426 ± 0.008 | 0.442 ± 0.009 | 1.158 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.025 ± 0.000 | - | 0.027 ± 0.000 | 0.032 ± 0.001 | 0.090 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.014 ± 0.001 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.005 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.001 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.011 ± 0.000 | - | 0.012 ± 0.000 | 0.016 ± 0.003 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 3.378 ± 0.011 | - | 3.383 ± 0.007 | 3.492 ± 0.018 | 8.796 ± 0.058 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.186 ± 0.007 | - | 0.189 ± 0.008 | 0.196 ± 0.002 | 0.572 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.020 ± 0.000 | - | 0.022 ± 0.000 | 0.024 ± 0.001 | 0.074 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.182 ± 0.010 | - | 0.185 ± 0.009 | 0.284 ± 0.008 | 0.291 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 0.841 ± 0.020 | - | 0.846 ± 1.882 | 0.721 ± 0.010 | 0.940 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 2.429 ± 0.028 | - | 2.436 ± 0.020 | 2.764 ± 0.004 | 2.918 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 1.686 ± 0.016 | - | 1.677 ± 0.014 | 0.594 ± 0.008 | 1.317 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.026 ± 0.000 | - | 0.027 ± 0.000 | 0.040 ± 0.001 | 0.097 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.015 ± 0.000 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 13.436 ± 0.096 | - | 13.460 ± 0.103 | 4.577 ± 0.023 | 10.445 ± 3.064 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.189 ± 0.001 | - | 0.191 ± 0.008 | 0.243 ± 0.011 | 0.336 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.015 ± 0.000 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.020 ± 0.000 | - | 0.022 ± 0.000 | 0.024 ± 0.000 | 0.065 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.140 ± 0.003 | - | 0.144 ± 0.002 | 0.147 ± 0.003 | 0.088 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.178 ± 0.001 | - | 0.180 ± 0.010 | 0.205 ± 0.011 | 0.182 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.395 ± 0.009 | - | 0.406 ± 0.008 | 0.416 ± 0.010 | 0.453 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.106 ± 0.002 | - | 0.110 ± 0.005 | 0.135 ± 0.003 | 0.174 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.016 ± 0.002 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.011 ± 0.001 | - | 0.013 ± 0.001 | 0.017 ± 0.000 | 0.046 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.036 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.037 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.872 ± 0.033 | - | 0.852 ± 0.010 | 1.026 ± 0.007 | 1.009 ± 0.068 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.057 ± 0.001 | - | 0.059 ± 0.001 | 0.061 ± 0.001 | 0.077 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.034 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.041 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.142 ± 0.001 | - | 0.142 ± 0.001 | 0.124 ± 0.002 | 0.096 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.190 ± 0.009 | - | 0.180 ± 0.001 | 0.165 ± 0.002 | 0.171 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.403 ± 0.021 | - | 0.403 ± 0.017 | 0.288 ± 0.012 | 0.508 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.106 ± 0.003 | - | 0.111 ± 0.004 | 0.166 ± 0.002 | 0.167 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.015 ± 0.001 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.010 ± 0.001 | - | 0.012 ± 0.000 | 0.023 ± 0.000 | 0.036 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.015 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.026 ± 0.001 | 0.027 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.015 ± 0.002 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.018 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.018 ± 0.001 | 0.018 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.841 ± 0.012 | - | 0.856 ± 0.024 | 0.490 ± 0.005 | 0.657 ± 0.124 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.018 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.058 ± 0.001 | - | 0.060 ± 0.000 | 0.052 ± 0.001 | 0.119 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.019 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.021 ± 0.001 | 0.023 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.188 ± 0.013 | - | 0.199 ± 0.024 | 0.171 ± 0.009 | 0.227 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.421 ± 0.010 | - | 0.432 ± 0.007 | 0.411 ± 0.011 | 0.603 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.014 ± 0.003 | - | 1.036 ± 0.006 | 1.043 ± 0.008 | 1.666 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.392 ± 0.021 | - | 0.400 ± 0.010 | 0.449 ± 0.008 | 0.491 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.019 ± 0.000 | 0.041 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.015 ± 0.000 | - | 0.017 ± 0.000 | 0.022 ± 0.000 | 0.041 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 3.061 ± 0.012 | - | 3.757 ± 0.023 | 3.480 ± 0.018 | 3.765 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.094 ± 0.002 | - | 0.095 ± 0.001 | 0.128 ± 0.003 | 0.153 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.198 ± 0.009 | - | 0.203 ± 0.008 | 0.246 ± 0.007 | 0.229 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.429 ± 0.020 | - | 0.433 ± 0.011 | 0.581 ± 0.010 | 0.631 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.011 ± 0.019 | - | 1.016 ± 0.010 | 1.524 ± 0.012 | 0.996 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.820 ± 0.024 | - | 0.815 ± 0.022 | 0.412 ± 0.009 | 0.997 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.032 ± 0.000 | 0.033 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.016 ± 0.001 | - | 0.017 ± 0.001 | 0.035 ± 0.001 | 0.033 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 6.451 ± 0.052 | - | 7.191 ± 0.059 | 3.129 ± 0.015 | 7.997 ± 1.340 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.095 ± 0.001 | - | 0.096 ± 0.002 | 0.210 ± 0.010 | 0.147 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.019 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.153 ± 0.001 | - | 0.126 ± 0.001 | 0.075 ± 0.001 | 0.183 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.179 ± 0.001 | - | 0.130 ± 0.003 | 0.116 ± 0.001 | 0.267 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.429 ± 0.008 | - | 0.379 ± 0.010 | 0.336 ± 0.008 | 0.581 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.093 ± 0.002 | - | 0.098 ± 0.003 | 0.111 ± 0.002 | 0.154 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.008 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.008 ± 0.000 | - | 0.013 ± 0.000 | 0.018 ± 0.001 | 0.043 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.014 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.728 ± 0.010 | - | 0.758 ± 0.009 | 0.825 ± 0.009 | 0.857 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.053 ± 0.001 | - | 0.059 ± 0.001 | 0.057 ± 0.001 | 0.094 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.014 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.138 ± 0.001 | - | 0.129 ± 0.004 | 0.098 ± 0.000 | 0.182 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.164 ± 0.009 | - | 5.996 ± 0.014 | 0.133 ± 0.007 | 0.262 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.418 ± 0.015 | - | 23.992 ± 0.050 | 0.299 ± 0.013 | 0.652 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.092 ± 0.002 | - | 5.997 ± 0.011 | 0.148 ± 0.005 | 0.128 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.001 | 0.010 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.009 ± 0.001 | - | 0.015 ± 0.001 | 0.026 ± 0.001 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.001 | 0.009 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.018 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.019 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.728 ± 0.012 | - | 24.004 ± 0.045 | 0.447 ± 0.018 | 0.595 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.019 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.053 ± 0.001 | - | 0.061 ± 0.000 | 0.048 ± 0.001 | 0.101 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.019 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.011 ± 0.000 | 0.021 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.372 ± 0.009 | - | 1.383 ± 0.011 | 1.180 ± 0.008 | 1.608 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 4.549 ± 0.015 | - | 4.556 ± 0.035 | 3.974 ± 0.007 | 4.009 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 13.519 ± 0.135 | - | 13.493 ± 0.098 | 12.084 ± 0.080 | 11.763 ± 0.059 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 4.077 ± 0.027 | - | 4.065 ± 0.025 | 4.696 ± 0.026 | 3.437 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.158 ± 0.005 | - | 0.160 ± 0.002 | 0.182 ± 0.011 | 0.176 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.144 ± 0.009 | - | 0.144 ± 0.005 | 0.162 ± 0.003 | 0.162 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.016 ± 0.000 | - | 0.018 ± 0.000 | 0.021 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.015 ± 0.000 | - | 0.017 ± 0.000 | 0.019 ± 0.001 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.043 ± 0.001 | - | 0.045 ± 0.000 | 0.045 ± 0.001 | 0.047 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.020 ± 0.000 | - | 0.022 ± 0.000 | 0.023 ± 0.001 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.057 ± 0.002 | - | 0.059 ± 0.000 | 0.057 ± 0.001 | 0.060 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 32.446 ± 0.082 | - | 33.735 ± 1.459 | 37.178 ± 0.169 | 27.526 ± 0.086 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.016 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.227 ± 0.006 | - | 1.233 ± 0.009 | 1.342 ± 0.012 | 1.194 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.038 ± 0.001 | - | 0.041 ± 0.001 | 0.039 ± 0.000 | 0.045 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.109 ± 0.002 | - | 0.112 ± 0.001 | 0.103 ± 0.001 | 0.118 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.368 ± 0.016 | - | 1.591 ± 0.826 | 1.149 ± 0.027 | 1.582 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 6.635 ± 0.334 | - | 6.672 ± 0.257 | 3.938 ± 0.013 | 1.976 ± 0.066 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 13.524 ± 0.090 | - | 13.533 ± 0.197 | 14.569 ± 0.222 | 5.993 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 7.539 ± 0.044 | - | 7.571 ± 0.036 | 4.003 ± 0.024 | 4.619 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.237 ± 0.015 | - | 0.241 ± 0.013 | 0.271 ± 0.015 | 0.191 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.012 ± 0.001 | - | 0.020 ± 0.013 | 0.011 ± 0.001 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.174 ± 0.013 | - | 0.182 ± 0.010 | 0.233 ± 0.010 | 0.155 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.030 ± 0.002 | - | 0.019 ± 0.001 | 0.019 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.014 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.021 ± 0.012 | - | 0.033 ± 0.024 | 0.018 ± 0.001 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.043 ± 0.001 | - | 0.046 ± 0.002 | 0.044 ± 0.002 | 0.048 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.015 ± 0.001 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.036 ± 0.015 | - | 0.040 ± 0.020 | 0.022 ± 0.002 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.058 ± 0.000 | - | 0.060 ± 0.001 | 0.056 ± 0.001 | 0.062 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 60.133 ± 0.098 | - | 61.482 ± 0.363 | 31.274 ± 0.229 | 61.981 ± 47.581 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.022 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 1.841 ± 0.018 | - | 1.844 ± 0.022 | 1.990 ± 0.016 | 0.602 ± 0.119 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.066 ± 0.027 | - | 0.071 ± 0.033 | 0.038 ± 0.001 | 0.045 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.110 ± 0.001 | - | 0.113 ± 0.002 | 0.102 ± 0.002 | 0.110 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 1.042 ± 0.009 | - | 1.050 ± 0.006 | 0.824 ± 0.011 | 0.997 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 3.871 ± 0.049 | - | 3.876 ± 0.006 | 3.338 ± 0.015 | 3.098 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 11.577 ± 0.034 | - | 11.579 ± 0.011 | 10.423 ± 0.147 | 9.601 ± 0.087 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.875 ± 0.017 | - | 1.873 ± 0.015 | 1.997 ± 0.009 | 1.870 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.105 ± 0.005 | - | 0.109 ± 0.003 | 0.126 ± 0.004 | 0.160 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.001 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.087 ± 0.006 | - | 0.087 ± 0.003 | 0.100 ± 0.002 | 0.138 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.019 ± 0.001 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.013 ± 0.000 | - | 0.016 ± 0.000 | 0.017 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.034 ± 0.000 | - | 0.036 ± 0.000 | 0.037 ± 0.001 | 0.039 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.017 ± 0.000 | - | 0.020 ± 0.000 | 0.020 ± 0.000 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.045 ± 0.000 | - | 0.048 ± 0.000 | 0.047 ± 0.001 | 0.048 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 14.933 ± 0.063 | - | 14.867 ± 0.013 | 15.928 ± 0.113 | 14.226 ± 0.100 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.013 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 0.821 ± 0.004 | - | 0.825 ± 0.006 | 0.917 ± 0.006 | 0.931 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.033 ± 0.000 | - | 0.035 ± 0.001 | 0.033 ± 0.000 | 0.036 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.089 ± 0.000 | - | 0.092 ± 0.001 | 0.087 ± 0.002 | 0.115 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 1.004 ± 0.015 | - | 1.015 ± 0.017 | 0.789 ± 0.389 | 0.987 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 3.935 ± 0.011 | - | 3.940 ± 0.022 | 3.284 ± 0.012 | 1.528 ± 0.288 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 11.654 ± 0.128 | - | 11.602 ± 0.054 | 12.972 ± 0.121 | 4.917 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 2.796 ± 0.108 | - | 2.819 ± 0.020 | 2.355 ± 0.006 | 2.544 ± 0.492 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.180 ± 0.010 | - | 0.192 ± 0.011 | 0.189 ± 0.005 | 0.127 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.119 ± 0.011 | - | 0.125 ± 0.006 | 0.146 ± 0.006 | 0.182 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.014 ± 0.001 | - | 0.016 ± 0.001 | 0.017 ± 0.002 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.016 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.034 ± 0.000 | - | 0.037 ± 0.027 | 0.036 ± 0.000 | 0.039 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.014 ± 0.000 | 0.016 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.018 ± 0.000 | - | 0.020 ± 0.000 | 0.020 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.048 ± 0.021 | - | 0.048 ± 0.001 | 0.046 ± 0.000 | 0.048 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 22.330 ± 0.047 | - | 22.352 ± 0.059 | 18.287 ± 0.030 | 21.988 ± 3.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.018 ± 0.002 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 1.415 ± 0.014 | - | 1.427 ± 0.011 | 1.409 ± 0.015 | 0.453 ± 0.079 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.033 ± 0.000 | - | 0.036 ± 0.000 | 0.032 ± 0.000 | 0.036 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.090 ± 0.000 | - | 0.093 ± 0.003 | 0.086 ± 0.000 | 0.106 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.364 ± 0.014 | - | unsupported | 0.263 ± 0.008 | 0.531 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.670 ± 0.016 | - | unsupported | 1.423 ± 0.016 | 2.027 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 4.138 ± 0.337 | - | unsupported | 3.926 ± 0.008 | 5.214 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.805 ± 0.017 | - | unsupported | 0.927 ± 0.010 | 0.927 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.038 ± 0.001 | - | unsupported | 0.050 ± 0.003 | 0.057 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.031 ± 0.001 | - | unsupported | 0.040 ± 0.002 | 0.047 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.013 ± 0.000 | - | unsupported | 0.018 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.008 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.018 ± 0.000 | - | unsupported | 0.022 ± 0.000 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 6.370 ± 0.025 | - | unsupported | 7.317 ± 0.049 | 7.272 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.297 ± 0.007 | - | unsupported | 0.331 ± 0.009 | 0.379 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.014 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.034 ± 0.000 | - | unsupported | 0.037 ± 0.001 | 0.048 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.365 ± 0.009 | - | unsupported | 0.402 ± 0.012 | 0.543 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 1.680 ± 0.033 | - | unsupported | 1.411 ± 0.007 | 1.029 ± 0.173 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 4.161 ± 0.017 | - | unsupported | 5.625 ± 0.028 | 2.674 ± 0.104 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 0.983 ± 0.014 | - | unsupported | 1.179 ± 0.012 | 1.382 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.039 ± 0.002 | - | unsupported | 0.065 ± 0.005 | 0.054 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.032 ± 0.000 | - | unsupported | 0.056 ± 0.003 | 0.060 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.006 ± 0.000 | - | unsupported | 0.010 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.010 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.007 ± 0.000 | - | unsupported | 0.014 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.014 ± 0.000 | - | unsupported | 0.018 ± 0.000 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.008 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.018 ± 0.000 | - | unsupported | 0.021 ± 0.000 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 7.743 ± 0.026 | - | unsupported | 9.195 ± 0.121 | 11.110 ± 0.078 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.005 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.298 ± 0.008 | - | unsupported | 0.329 ± 0.020 | 0.194 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.014 ± 0.000 | - | unsupported | 0.024 ± 0.002 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.034 ± 0.000 | - | unsupported | 0.038 ± 0.001 | 0.048 ± 0.001 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.839 ± 0.004 | - | unsupported | 0.779 ± 0.010 | 2.668 ± 0.050 | 2.100 ± 0.091 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.108 ± 0.002 | - | unsupported | 0.049 ± 0.001 | 0.091 ± 0.002 | 0.112 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.142 ± 0.003 | - | unsupported | 0.070 ± 0.001 | 0.133 ± 0.004 | 0.181 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.283 ± 0.007 | - | unsupported | 0.200 ± 0.003 | 0.362 ± 0.013 | 0.554 ± 0.217 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.833 ± 0.005 | - | unsupported | 0.387 ± 0.014 | 0.881 ± 0.149 | 2.061 ± 0.946 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.103 ± 0.009 | - | unsupported | 0.049 ± 0.007 | 0.070 ± 0.002 | 0.099 ± 0.006 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.139 ± 0.002 | - | unsupported | 0.059 ± 0.005 | 0.152 ± 0.037 | 0.158 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.295 ± 0.020 | - | unsupported | 0.124 ± 0.002 | 0.415 ± 0.018 | 0.612 ± 0.061 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.247 ± 0.001 | - | unsupported | 0.304 ± 0.010 | 0.628 ± 0.010 | 0.296 ± 0.010 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.028 ± 0.001 | - | unsupported | 0.033 ± 0.001 | 0.129 ± 0.004 | 0.061 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.050 ± 0.000 | - | unsupported | 0.050 ± 0.001 | 0.172 ± 0.004 | 0.089 ± 0.007 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.163 ± 0.001 | - | unsupported | 0.138 ± 0.006 | 0.286 ± 0.016 | 0.150 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.249 ± 0.003 | - | unsupported | 0.430 ± 0.009 | 0.707 ± 0.075 | 0.296 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.029 ± 0.000 | - | unsupported | 0.055 ± 0.002 | 0.125 ± 0.005 | 0.062 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.051 ± 0.000 | - | unsupported | 0.076 ± 0.002 | 0.157 ± 0.010 | 0.092 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.165 ± 0.002 | - | unsupported | 0.180 ± 0.002 | 0.267 ± 0.021 | 0.153 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.155 ± 0.000 | - | 0.160 ± 0.003 | 0.162 ± 0.009 | 0.292 ± 0.018 | 0.169 ± 0.010 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.063 ± 0.000 | - | 0.063 ± 0.000 | 0.056 ± 0.001 | 0.056 ± 0.002 | 0.046 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.073 ± 0.001 | - | 0.075 ± 0.000 | 0.068 ± 0.001 | 0.068 ± 0.002 | 0.062 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.097 ± 0.000 | - | 0.100 ± 0.001 | 0.098 ± 0.001 | 0.124 ± 0.007 | 0.101 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.156 ± 0.001 | - | 0.160 ± 0.002 | 0.206 ± 0.001 | 0.378 ± 0.061 | 0.172 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.071 ± 0.000 | - | 0.065 ± 0.000 | 0.095 ± 0.004 | 0.058 ± 0.002 | 0.047 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.073 ± 0.000 | - | 0.077 ± 0.001 | 0.108 ± 0.001 | 0.071 ± 0.001 | 0.065 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.099 ± 0.001 | - | 0.103 ± 0.001 | 0.141 ± 0.002 | 0.140 ± 0.012 | 0.104 ± 0.003 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 36.153 ± 0.182 | - | 38.814 ± 0.142 | 54.238 ± 0.414 | 60.282 ± 0.225 | 48.429 ± 5.750 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 39.202 ± 0.664 | - | 38.366 ± 0.269 | 20.288 ± 0.109 | 53.007 ± 4.502 | 40.453 ± 8.186 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 21.239 ± 0.494 | - | 20.507 ± 0.399 | 21.922 ± 0.765 | 20.364 ± 0.440 | 18.076 ± 0.322 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 245.943 ± 11.239 | - | 325.992 ± 11.032 | 9.411 ± 0.248 | 12.309 ± 0.041 | 10.658 ± 5.092 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 6.126 ± 0.023 | - | 6.131 ± 0.008 | 6.553 ± 0.036 | 6.027 ± 0.066 | 5.987 ± 0.173 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 13.735 ± 0.027 | - | 13.778 ± 0.066 | 9.070 ± 0.215 | 8.933 ± 0.149 | 8.511 ± 0.106 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 7.047 ± 0.016 | - | 7.074 ± 0.010 | 7.177 ± 0.049 | 7.163 ± 0.038 | 6.980 ± 0.031 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 10.508 ± 0.053 | - | 10.580 ± 0.021 | 9.697 ± 0.159 | 10.818 ± 0.052 | 10.384 ± 0.126 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 17.227 ± 0.121 | - | 17.224 ± 0.036 | 15.682 ± 0.094 | 31.790 ± 0.307 | 11.027 ± 0.676 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 30.991 ± 0.058 | - | 31.456 ± 0.123 | 12.717 ± 0.090 | 26.542 ± 1.028 | 9.270 ± 0.054 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.406 ± 0.008 | 0.426 ± 0.017 | unsupported | 0.485 ± 0.007 | 0.590 ± 0.039 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 6.367 ± 0.037 | 6.514 ± 0.173 | unsupported | 6.968 ± 0.043 | 7.686 ± 0.070 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.639 ± 0.008 | 29.543 ± 0.933 | unsupported | 0.409 ± 0.001 | 1.397 ± 0.264 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 2267.990 ± 2.039 | 2296.482 ± 1.437 | unsupported | 4.607 ± 0.036 | 8.739 ± 0.121 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 34.508 ± 0.441 | - | 34.381 ± 0.343 | 39.471 ± 1.414 | 35.284 ± 0.434 | 24.071 ± 2.032 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 222.577 ± 11.932 | - | 254.967 ± 13.607 | 18.615 ± 0.279 | 20.987 ± 0.461 | 14.234 ± 0.124 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 14.446 ± 0.261 | 14.172 ± 0.081 | 7.577 ± 0.073 | 35.624 ± 0.340 | 6.209 ± 0.111 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 2278.029 ± 20.892 | 2280.034 ± 19.937 | 5.115 ± 0.032 | 34.881 ± 0.492 | 5.202 ± 0.045 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 21.099 ± 0.519 | - | 21.431 ± 0.205 | 28.751 ± 1.120 | 27.559 ± 0.143 | 25.942 ± 10.753 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 234.198 ± 11.885 | - | 244.880 ± 14.933 | 13.767 ± 0.240 | 20.965 ± 23.639 | 31.651 ± 20.649 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.805 ± 0.048 | - | 0.712 ± 0.043 | 0.975 ± 0.030 | 4.816 ± 0.039 | 2.306 ± 0.007 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.266 ± 0.042 | - | 23.998 ± 0.009 | 0.947 ± 0.026 | 12.228 ± 2.053 | 2.310 ± 0.011 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 16.809 ± 0.301 | - | 16.619 ± 0.049 | 17.576 ± 0.292 | 16.352 ± 0.072 | 17.460 ± 0.554 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 200.933 ± 22.830 | - | 205.495 ± 24.257 | 14.069 ± 0.167 | 17.717 ± 1.242 | 17.529 ± 1.438 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 16.615 ± 0.256 | - | 16.829 ± 0.062 | 17.627 ± 0.166 | 16.304 ± 0.051 | 15.889 ± 1.415 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 195.160 ± 14.888 | - | 202.529 ± 28.170 | 13.933 ± 0.086 | 18.118 ± 0.273 | 16.904 ± 0.898 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 0.978 ± 0.028 | 0.981 ± 0.009 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 17.453 ± 0.133 | 17.399 ± 0.061 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.314 ± 0.003 | 1.331 ± 0.005 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 18.444 ± 1.186 | 18.468 ± 0.290 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 0.913 ± 0.010 | 0.921 ± 0.010 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 16.537 ± 0.104 | 16.559 ± 0.029 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.252 ± 0.004 | 1.270 ± 0.009 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 17.751 ± 0.045 | 17.550 ± 0.030 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 20.871 ± 0.448 | - | 20.786 ± 0.619 | 21.951 ± 0.556 | 19.957 ± 0.049 | 18.063 ± 1.510 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 237.004 ± 14.821 | - | 317.997 ± 14.998 | 9.550 ± 0.217 | 12.306 ± 0.340 | 10.846 ± 2.163 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 47.877 ± 0.505 | 48.522 ± 0.954 | 50.796 ± 0.359 | 49.632 ± 0.118 | 48.977 ± 2.642 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 2314.964 ± 13.369 | 2315.040 ± 5.731 | 32.822 ± 0.069 | 42.984 ± 1.262 | 44.903 ± 2.644 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 30.385 ± 0.087 | - | 30.203 ± 0.134 | 25.948 ± 0.114 | 128.382 ± 4.488 | 25.722 ± 0.191 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 48.902 ± 1.225 | - | 52.265 ± 1.721 | 20.216 ± 2.231 | 114.154 ± 3.249 | 17.975 ± 0.287 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.079 ± 0.000 | 0.079 ± 0.001 | unsupported | 0.132 ± 0.030 | 0.121 ± 0.024 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 1.257 ± 0.003 | 1.255 ± 0.012 | unsupported | 2.006 ± 0.027 | 0.333 ± 0.529 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.039 ± 0.000 | 23.984 ± 0.203 | unsupported | 0.064 ± 0.045 | 0.023 ± 0.006 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.568 ± 0.002 | 23.999 ± 0.126 | unsupported | 0.851 ± 0.076 | 0.882 ± 0.268 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.038 ± 0.000 | 0.038 ± 0.000 | unsupported | 0.037 ± 0.001 | 0.176 ± 0.018 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 0.590 ± 0.002 | 0.612 ± 0.001 | unsupported | 1.200 ± 0.007 | 2.089 ± 0.029 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.019 ± 0.000 | 12.000 ± 0.141 | unsupported | 0.062 ± 0.002 | 0.116 ± 0.002 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.281 ± 0.001 | 11.999 ± 0.001 | unsupported | 0.501 ± 0.045 | 0.883 ± 0.092 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.105 ± 0.000 | 0.106 ± 0.001 | unsupported | 0.111 ± 0.000 | 0.158 ± 0.003 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 1.672 ± 0.012 | 2.364 ± 1.203 | unsupported | 2.501 ± 0.015 | 1.572 ± 0.082 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.050 ± 0.001 | 12.921 ± 1.922 | unsupported | 0.072 ± 0.001 | 0.104 ± 0.010 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 0.763 ± 0.002 | 11.999 ± 0.002 | unsupported | 0.991 ± 0.014 | 0.784 ± 0.016 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.049 ± 0.003 | 0.070 ± 0.006 | unsupported | 0.083 ± 0.001 | 0.156 ± 0.003 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.596 ± 0.010 | 0.627 ± 0.027 | unsupported | 1.352 ± 0.009 | 1.663 ± 0.027 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.075 ± 0.001 | 65.169 ± 0.513 | unsupported | 0.053 ± 0.002 | 0.141 ± 0.001 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.343 ± 0.005 | 95.999 ± 0.125 | unsupported | 0.364 ± 0.146 | 0.960 ± 0.020 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.071 ± 0.002 | 0.088 ± 0.002 | unsupported | 0.045 ± 0.000 | 0.217 ± 0.004 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 0.927 ± 0.039 | 1.442 ± 0.829 | unsupported | 1.481 ± 0.006 | 2.463 ± 0.033 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.090 ± 0.002 | 88.870 ± 9.044 | unsupported | 0.044 ± 0.002 | 0.195 ± 0.008 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.480 ± 0.013 | 119.999 ± 1.503 | unsupported | 0.333 ± 0.011 | 1.402 ± 0.031 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.235 ± 0.006 | 0.260 ± 0.021 | unsupported | unsupported | 0.221 ± 0.019 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 3.307 ± 0.036 | 3.389 ± 0.075 | unsupported | unsupported | 3.297 ± 0.050 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.169 ± 0.008 | 112.736 ± 0.657 | unsupported | unsupported | 0.223 ± 0.005 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 1.353 ± 0.020 | 142.376 ± 106.298 | unsupported | unsupported | 1.787 ± 0.081 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.008 ± 0.000 | 0.014 ± 0.000 | unsupported | 0.010 ± 0.000 | 0.042 ± 0.002 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.089 ± 0.001 | 0.097 ± 0.003 | unsupported | 0.060 ± 0.002 | 0.256 ± 0.013 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.010 ± 0.000 | 23.984 ± 0.016 | unsupported | 0.011 ± 0.000 | 0.029 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.050 ± 0.003 | 23.999 ± 0.002 | unsupported | 0.038 ± 0.002 | 0.225 ± 0.003 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.026 ± 0.004 | 0.041 ± 0.001 | unsupported | 0.030 ± 0.001 | 0.154 ± 0.003 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.341 ± 0.013 | 0.358 ± 0.025 | unsupported | 1.221 ± 0.057 | 1.421 ± 0.034 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.036 ± 0.002 | 32.546 ± 0.375 | unsupported | 0.085 ± 0.003 | 0.126 ± 0.004 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.193 ± 0.004 | 47.999 ± 0.124 | unsupported | 0.220 ± 0.003 | 0.773 ± 0.163 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.026 ± 0.000 | 0.026 ± 0.000 | unsupported | 0.071 ± 0.004 | 0.148 ± 0.006 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.401 ± 0.003 | 0.450 ± 0.003 | unsupported | 1.211 ± 0.013 | 1.371 ± 0.118 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.014 ± 0.000 | 12.000 ± 0.016 | unsupported | 0.061 ± 0.003 | 0.102 ± 0.006 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.193 ± 0.003 | 11.999 ± 0.001 | unsupported | 0.447 ± 0.014 | 0.706 ± 0.057 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.027 ± 0.001 | 0.027 ± 0.000 | unsupported | 0.074 ± 0.006 | 0.151 ± 0.003 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.415 ± 0.002 | 0.447 ± 0.002 | unsupported | 1.234 ± 0.010 | 1.425 ± 0.037 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.014 ± 0.001 | 12.000 ± 0.016 | unsupported | 0.059 ± 0.002 | 0.101 ± 0.015 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.201 ± 0.007 | 11.999 ± 0.000 | unsupported | 0.452 ± 0.013 | 0.861 ± 0.265 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.065 ± 0.001 | 0.080 ± 0.001 | unsupported | 0.050 ± 0.001 | 0.213 ± 0.004 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 0.974 ± 0.036 | 0.959 ± 0.023 | unsupported | 0.871 ± 0.004 | 2.382 ± 0.033 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.087 ± 0.007 | 88.643 ± 0.460 | unsupported | 0.044 ± 0.001 | 0.181 ± 0.002 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 0.494 ± 0.004 | 119.875 ± 0.251 | unsupported | 0.324 ± 0.073 | 1.640 ± 0.116 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.092 ± 0.000 | 0.092 ± 0.000 | unsupported | 0.148 ± 0.003 | 0.272 ± 0.004 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 1.456 ± 0.004 | 1.464 ± 0.152 | unsupported | 2.320 ± 0.008 | 3.439 ± 0.050 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.040 ± 0.001 | 12.000 ± 0.141 | unsupported | 0.114 ± 0.002 | 0.156 ± 0.002 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 0.608 ± 0.002 | 11.999 ± 0.001 | unsupported | 0.887 ± 0.008 | 1.729 ± 0.094 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.519 ± 0.004 | 1.087 ± 0.007 | unsupported | 0.221 ± 0.004 | 0.157 ± 0.004 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 10.839 ± 0.256 | 19.379 ± 0.590 | unsupported | 4.812 ± 0.073 | 1.485 ± 0.906 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.402 ± 0.002 | 104.078 ± 0.921 | unsupported | 0.234 ± 0.007 | 0.102 ± 0.005 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 6.448 ± 0.193 | 141.375 ± 7.751 | unsupported | 1.863 ± 0.059 | 0.730 ± 0.058 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 23.981 ± 1.269 | - | unsupported | 22.254 ± 0.133 | - | 22.677 ± 0.665 | 71.089 ± 0.722 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 19.905 ± 0.094 | - | unsupported | 19.786 ± 0.183 | - | 24.014 ± 0.687 | 27.678 ± 0.530 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 17.384 ± 0.132 | - | unsupported | 16.844 ± 0.385 | - | 17.401 ± 0.376 | 16.087 ± 0.445 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 13.762 ± 0.304 | - | unsupported | 19.303 ± 1.374 | - | 17.585 ± 1.160 | 14.024 ± 0.400 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 18.038 ± 0.097 | - | unsupported | 17.123 ± 0.471 | - | 22.016 ± 9.370 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 13.629 ± 0.152 | - | unsupported | 18.397 ± 3.046 | - | 18.719 ± 0.215 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 21.443 ± 0.227 | - | unsupported | 23.013 ± 0.626 | - | 25.709 ± 0.862 | 74.032 ± 0.860 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 20.251 ± 0.708 | - | unsupported | 20.111 ± 0.166 | - | 26.070 ± 1.490 | 27.048 ± 1.016 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 41.393 ± 0.087 | - | unsupported | 44.650 ± 0.902 | - | 39.768 ± 0.088 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 45.249 ± 0.876 | - | unsupported | 21.859 ± 0.266 | - | 19.699 ± 0.303 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 41.222 ± 0.119 | - | unsupported | 44.496 ± 0.240 | - | 39.514 ± 0.176 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 44.335 ± 1.650 | - | unsupported | 21.844 ± 0.231 | - | 19.625 ± 0.086 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 23.400 ± 0.270 | - | unsupported | 22.516 ± 0.638 | - | 24.156 ± 0.669 | 71.152 ± 1.837 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 19.961 ± 0.476 | - | unsupported | 19.718 ± 0.187 | - | 26.724 ± 3.190 | 27.375 ± 1.936 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 17.119 ± 0.117 | - | unsupported | 16.761 ± 0.362 | - | 17.052 ± 0.698 | 56.151 ± 0.667 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 13.898 ± 0.626 | - | unsupported | 13.983 ± 0.127 | - | 19.837 ± 1.658 | 27.239 ± 3.464 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.592 ± 0.001 | unsupported | unsupported | 0.767 ± 0.003 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 9.144 ± 0.008 | unsupported | unsupported | 10.828 ± 0.038 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 1.258 ± 0.007 | unsupported | unsupported | 0.515 ± 0.002 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 183.188 ± 2.941 | unsupported | unsupported | 5.378 ± 0.010 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 23.873 ± 0.562 | - | unsupported | 21.993 ± 1.039 | - | 24.471 ± 0.563 | 70.975 ± 0.359 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 20.477 ± 0.193 | - | unsupported | 19.792 ± 0.147 | - | 24.744 ± 1.834 | 27.887 ± 2.077 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 90.640 ± 3.349 | - | 0.001 ± 0.000 | 88.282 ± 1.016 | 84.610 ± 5.213 | 110.838 ± 1.102 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 37.397 ± 0.120 | - | 0.001 ± 0.000 | 28.619 ± 0.871 | 50.615 ± 2.649 | 108.354 ± 2.754 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 56.999 ± 0.476 | - | 175.321 ± 10.546 | 60.421 ± 1.373 | 62.812 ± 2.030 | 69.728 ± 3.411 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 20.326 ± 0.157 | - | 226.177 ± 8.705 | 20.385 ± 0.353 | 32.832 ± 1.903 | 69.700 ± 0.755 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 170.518 ± 1.917 | - | 170.404 ± 0.454 | 174.858 ± 0.270 | 175.646 ± 3.152 | 203.945 ± 5.718 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 57.389 ± 0.646 | - | 179.154 ± 0.377 | 56.610 ± 0.291 | 94.479 ± 5.091 | 201.081 ± 9.628 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 56.685 ± 3.765 | - | 56.245 ± 3.579 | 57.834 ± 5.928 | 60.489 ± 2.778 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 36.698 ± 1.599 | - | 58.736 ± 1.985 | 21.162 ± 0.142 | 32.788 ± 0.390 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 113.935 ± 6.550 | - | 0.002 ± 0.000 | 100.632 ± 1.489 | 101.659 ± 1.727 | 128.590 ± 10.573 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 36.311 ± 2.503 | - | 0.002 ± 0.000 | 34.822 ± 0.355 | 39.548 ± 6.718 | 127.001 ± 5.232 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 78.071 ± 1.794 | - | 0.001 ± 0.000 | 71.610 ± 0.443 | 65.308 ± 1.224 | 177.689 ± 11.124 | 116.351 ± 5.487 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 25.392 ± 0.640 | - | 0.001 ± 0.000 | 70.489 ± 0.358 | 34.163 ± 0.283 | 176.841 ± 2.059 | 44.012 ± 1.058 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 47.622 ± 2.611 | - | 47.541 ± 2.547 | 58.483 ± 1.381 | 51.525 ± 2.310 | 68.735 ± 3.038 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 19.891 ± 0.167 | - | 66.526 ± 9.280 | 19.439 ± 0.427 | 27.713 ± 2.022 | 67.521 ± 1.984 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 49.273 ± 2.110 | - | 47.132 ± 1.814 | 53.886 ± 1.924 | 49.903 ± 2.069 | 68.535 ± 2.573 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 18.578 ± 0.177 | - | 65.573 ± 12.634 | 18.114 ± 0.895 | 26.274 ± 2.979 | 67.597 ± 6.238 | - |
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
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 5.964078 | 372754.88 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 4.597608 | 287350.50 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 4.885461 | 305341.31 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 5.043056 | 315191.00 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 3.330671 | 208166.94 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 1791.885916 | 111992869.75 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 9.761620 | 610101.25 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 9.079637 | 567477.31 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 3.935821 | 245988.81 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 383.988658 | 23999291.12 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 9.433070 | 589566.88 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 7.765448 | 485340.50 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 6.497533 | 406095.81 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 6.814259 | 425891.19 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 122.972127 | 7685757.94 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 111.483480 | 6967717.50 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 101.874145 | 6367134.06 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 104.226784 | 6514174.00 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 22.345352 | 1396584.50 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 6.550453 | 409403.31 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 10.218777 | 638673.56 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 472.690996 | 29543187.25 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 139.824236 | 8739014.75 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 73.706963 | 4606685.19 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 36287.834214 | 2267989638.38 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 36743.717779 | 2296482361.19 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 15.649198 | 978074.88 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 15.700764 | 981297.75 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 279.246949 | 17452934.31 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 278.384804 | 17399050.25 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 21.031728 | 1314483.00 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 21.295506 | 1330969.12 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 295.098546 | 18443659.12 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 295.482222 | 18467638.88 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 14.610731 | 913170.69 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 14.741033 | 921314.56 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 264.587567 | 16536722.94 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 264.938914 | 16558682.12 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 20.026851 | 1251678.19 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 20.313412 | 1269588.25 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 284.020969 | 17751310.56 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 280.807183 | 17550448.94 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.722544 | 120664.75 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.445216 | 131956.50 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.084356 | 79443.06 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.041643 | 78775.67 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 2.665249 | 333156.12 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 16.048135 | 2006016.87 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.059090 | 1257386.25 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.036437 | 1254554.62 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 1.467518 | 22929.97 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.090655 | 63916.48 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.508980 | 39202.81 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1534.980548 | 23984071.06 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.055185 | 881898.12 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.811537 | 851442.12 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.540968 | 567621.00 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 191.992255 | 23999031.88 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.235816 | 175559.62 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 2.361792 | 36903.00 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.407139 | 37611.55 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.442681 | 38166.89 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 16.712433 | 2089054.12 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.596840 | 1199605.00 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.720821 | 590102.62 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.895739 | 611967.38 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.419916 | 115936.19 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.960925 | 61889.45 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.243589 | 19431.08 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.976105 | 11999626.64 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.067443 | 883430.38 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.009927 | 501240.88 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.245773 | 280721.62 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.992508 | 11999063.50 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.142583 | 158477.86 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.084416 | 110694.00 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.724227 | 105066.05 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.792327 | 106130.11 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.573637 | 1571704.62 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 20.004251 | 2500531.38 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.373363 | 1671670.38 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 18.915962 | 2364495.25 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.645668 | 103838.56 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.627034 | 72297.41 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.209409 | 50147.02 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 826.967869 | 12921372.95 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.273676 | 784209.50 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.928961 | 991120.12 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.106550 | 763318.75 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.992127 | 11999015.88 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.974636 | 155853.69 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.280454 | 82507.09 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.159165 | 49361.95 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.459487 | 69679.48 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.301387 | 1662673.38 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.812647 | 1351580.88 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.767460 | 595932.50 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.014203 | 626775.38 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.999814 | 140622.09 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.409785 | 53277.89 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.788812 | 74825.19 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4170.831510 | 65169242.34 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.680044 | 960005.50 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.914997 | 364374.62 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.745167 | 343145.88 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 767.995351 | 95999418.88 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.873228 | 216769.19 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 2.856260 | 44629.06 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.566965 | 71358.83 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.641845 | 88153.83 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 19.703250 | 2462906.25 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.849953 | 1481244.12 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.418117 | 927264.62 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.538435 | 1442304.38 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.470623 | 194853.48 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.786638 | 43541.22 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.743835 | 89747.42 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5687.692754 | 88870199.28 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.218317 | 1402289.62 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.660753 | 332594.12 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.840478 | 480059.75 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 959.992091 | 119999011.38 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.153030 | 221141.09 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 15.025658 | 234775.91 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 16.625355 | 259771.17 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 26.377863 | 3297232.88 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 26.453154 | 3306644.25 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 27.114196 | 3389274.50 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 14.250836 | 222669.31 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 10.789915 | 168592.42 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7215.124507 | 112736320.42 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 14.297705 | 1787213.12 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 10.823627 | 1352953.38 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1139.004887 | 142375610.88 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.667943 | 41686.61 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.619291 | 9676.42 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.514737 | 8042.77 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 0.871622 | 13619.09 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 2.049944 | 256243.00 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.481799 | 60224.88 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.715371 | 89421.38 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.779677 | 97459.62 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 1.839134 | 28736.47 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.685658 | 10713.41 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.667151 | 10424.23 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1534.995093 | 23984298.33 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.796031 | 224503.88 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.306752 | 38344.00 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.401751 | 50218.88 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 191.995520 | 23999440.00 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.856952 | 154014.88 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 1.899032 | 29672.38 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.664587 | 26009.17 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.600093 | 40626.45 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.368916 | 1421114.50 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.768533 | 1221066.62 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.727160 | 340895.00 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 2.865784 | 358223.00 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.048075 | 125751.17 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.436315 | 84942.42 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.305881 | 36029.39 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2082.967512 | 32546367.38 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.180449 | 772556.12 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.758686 | 219835.75 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.541327 | 192665.88 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 383.992007 | 47999000.88 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.457535 | 147773.98 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.563780 | 71309.06 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.651197 | 25799.95 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.688282 | 26379.41 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.970741 | 1371342.62 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.686531 | 1210816.38 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.205934 | 400741.75 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.600043 | 450005.38 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.520218 | 101878.41 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.904913 | 61014.27 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.894315 | 13973.67 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.981231 | 11999706.73 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.648847 | 706105.88 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.577671 | 447208.88 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.541557 | 192694.62 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.994430 | 11999303.75 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.690818 | 151419.03 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.758627 | 74353.55 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.711556 | 26743.06 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.721761 | 26902.52 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.398971 | 1424871.38 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.873266 | 1234158.25 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.323436 | 415429.50 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.573775 | 446721.88 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.484746 | 101324.16 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.753721 | 58651.89 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.919642 | 14369.41 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.978048 | 11999657.00 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.889391 | 861173.88 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.618531 | 452316.38 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.608796 | 201099.50 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.994611 | 11999326.38 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.608389 | 212631.08 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.185003 | 49765.67 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.180877 | 65326.20 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.131244 | 80175.69 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 19.055197 | 2381899.62 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 6.966703 | 870837.88 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.791846 | 973980.75 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.675285 | 959410.62 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.559128 | 180611.38 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.831805 | 44246.95 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.558333 | 86848.95 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5673.131008 | 88642672.00 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 13.121296 | 1640162.00 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.590740 | 323842.50 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.950470 | 493808.75 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 958.997546 | 119874693.25 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 17.400554 | 271883.66 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 9.443574 | 147555.84 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.859335 | 91552.11 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.892904 | 92076.62 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 27.511040 | 3438880.00 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 18.560259 | 2320032.38 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 11.649448 | 1456181.00 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.714654 | 1464331.75 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.977242 | 155894.41 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 7.326414 | 114475.22 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.584861 | 40388.45 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.980043 | 11999688.17 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 13.830118 | 1728764.75 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.099891 | 887486.38 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.860167 | 607520.88 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.993839 | 11999229.88 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.057628 | 157150.44 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 14.167179 | 221362.17 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 33.192343 | 518630.36 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 69.595918 | 1087436.22 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.883773 | 1485471.62 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 38.497843 | 4812230.38 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 86.713236 | 10839154.50 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 155.030494 | 19378811.75 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.534769 | 102105.77 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 14.998617 | 234353.39 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 25.734125 | 402095.70 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6660.969991 | 104077656.11 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.836503 | 729562.88 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 14.906902 | 1863362.75 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 51.581224 | 6447653.00 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1131.003022 | 141375377.75 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 12.279674 | 767479.62 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 9.473460 | 592091.25 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 173.251786 | 10828236.62 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 146.303860 | 9143991.25 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 8.241140 | 515071.25 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 20.130242 | 1258140.12 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 86.055733 | 5378483.31 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 2931.008641 | 183188040.06 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.017706 | 1106.62 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002083 | 130.19 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.017595 | 1099.69 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002273 | 142.06 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000091 | 5.69 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.016234 | 1014.62 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.007832 | 489.50 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000101 | 6.31 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.016544 | 1034.00 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.008282 | 517.62 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000210 | 13.12 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.017205 | 1075.31 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.005918 | 369.88 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000420 | 26.25 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.017356 | 1084.75 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.006049 | 378.06 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000080 | 5.00 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.015744 | 984.00 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003094 | 193.38 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000090 | 5.62 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.015773 | 985.81 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003515 | 219.69 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/norm_fro` (c64, threads=1, shape=`2048x1536`): `tenferro-direct` is 14.8x faster than the slowest successful cell (`pytorch-cpu`, 18.199 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `jax-cpu` is 22.3x faster than the slowest successful cell (`tenferro-trace`, 59.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `julia-base` is 17.3x faster than the slowest successful cell (`tenferro-trace`, 59.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 93.6x faster than the slowest successful cell (`tenferro-trace`, 59.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 19.9x faster than the slowest successful cell (`tenferro-direct`, 125.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `julia-base` is 19.8x faster than the slowest successful cell (`tenferro-direct`, 125.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 25.3x faster than the slowest successful cell (`tenferro-direct`, 125.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 36.7x faster than the slowest successful cell (`tenferro-trace`, 120.990 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `julia-base` is 32.0x faster than the slowest successful cell (`tenferro-trace`, 120.990 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 41.5x faster than the slowest successful cell (`tenferro-trace`, 120.990 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/exp` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 15.1x faster than the slowest successful cell (`julia-base`, 57.655 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log1p` (f64, threads=4, shape=`4194304`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`julia-base`, 35.820 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/masked_log_softmax_axis1` (f64, threads=4, shape=`1024x64`): `jax-cpu` is 355.3x faster than the slowest successful cell (`tenferro-eager`, 111.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/masked_log_softmax_axis1` (f64, threads=4, shape=`1024x64`): `tenferro-direct` is 538.0x faster than the slowest successful cell (`tenferro-eager`, 111.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 21.719 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 24.1x faster than the slowest successful cell (`pytorch-cpu`, 21.719 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 14.6x faster than the slowest successful cell (`pytorch-cpu`, 21.719 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 14.6x faster than the slowest successful cell (`pytorch-cpu`, 21.719 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 19.4x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 26.2x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 35.0x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 16.2x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 10.2x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 25.8x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 10.4x faster than the slowest successful cell (`pytorch-cpu`, 21.792 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 24.3x faster than the slowest successful cell (`pytorch-cpu`, 21.792 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 21.792 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 15.3x faster than the slowest successful cell (`pytorch-cpu`, 21.792 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 15.4x faster than the slowest successful cell (`tenferro-trace`, 23.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 25.5x faster than the slowest successful cell (`tenferro-trace`, 23.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 34.8x faster than the slowest successful cell (`tenferro-trace`, 23.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 23.1x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 36.3x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 25.1x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 76.8x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 33.1x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 27.7x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 103.9x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 36.2x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 17.5x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 38.0x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 25.0x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 74.5x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 34.9x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 29.2x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 97.2x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 35.6x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_squares_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 97.6x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sin` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 10.6x faster than the slowest successful cell (`julia-base`, 52.836 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=1, shape=`1048576+1048576`): `jax-cpu` is 10.2x faster than the slowest successful cell (`julia-base`, 6.574 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=1, shape=`1048576+1048576`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`julia-base`, 6.574 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `jax-cpu` is 17.7x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `pytorch-cpu` is 39.3x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `tenferro-direct` is 42.4x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=1, shape=`4194304 -> 2097152`): `jax-cpu` is 10.7x faster than the slowest successful cell (`julia-base`, 6.902 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=1, shape=`4194304 -> 2097152`): `pytorch-cpu` is 11.2x faster than the slowest successful cell (`julia-base`, 6.902 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=1, shape=`4194304 -> 2097152`): `tenferro-direct` is 10.9x faster than the slowest successful cell (`julia-base`, 6.902 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=1, shape=`4194304 -> 2097152`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`julia-base`, 6.902 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `jax-cpu` is 18.3x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `julia-base` is 11.9x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `pytorch-cpu` is 40.9x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `tenferro-direct` is 29.6x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 12.6x faster than the slowest successful cell (`julia-base`, 7.855 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 13.4x faster than the slowest successful cell (`julia-base`, 7.855 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `jax-cpu` is 189.4x faster than the slowest successful cell (`tenferro-trace`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `julia-base` is 43.3x faster than the slowest successful cell (`tenferro-trace`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 150.2x faster than the slowest successful cell (`tenferro-trace`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `tenferro-direct` is 168.7x faster than the slowest successful cell (`tenferro-trace`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `jax-cpu` is 14.2x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 21.8x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 36.9x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `jax-cpu` is 16.2x faster than the slowest successful cell (`tenferro-trace`, 11.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 38.3x faster than the slowest successful cell (`tenferro-trace`, 11.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 42.1x faster than the slowest successful cell (`tenferro-trace`, 11.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `jax-cpu` is 117.1x faster than the slowest successful cell (`tenferro-trace`, 35.966 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `julia-base` is 31.7x faster than the slowest successful cell (`tenferro-trace`, 35.966 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 98.7x faster than the slowest successful cell (`tenferro-trace`, 35.966 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `tenferro-direct` is 43.5x faster than the slowest successful cell (`tenferro-trace`, 35.966 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 4847.5x faster than the slowest successful cell (`julia-base`, 7.911 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-direct` is 11.8x faster than the slowest successful cell (`julia-base`, 8.282 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 5170.0x faster than the slowest successful cell (`julia-base`, 8.282 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 14.8x faster than the slowest successful cell (`jax-cpu`, 0.023 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.024 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x4x4,rhs=1`): `jax-cpu` is 22.9x faster than the slowest successful cell (`tenferro-trace`, 5.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x4x4,rhs=1`): `pytorch-cpu` is 45.0x faster than the slowest successful cell (`tenferro-trace`, 5.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x4x4,rhs=1`): `tenferro-direct` is 36.5x faster than the slowest successful cell (`tenferro-trace`, 5.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `jax-cpu` is 36.8x faster than the slowest successful cell (`tenferro-trace`, 23.992 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `pytorch-cpu` is 80.3x faster than the slowest successful cell (`tenferro-trace`, 23.992 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `tenferro-direct` is 57.4x faster than the slowest successful cell (`tenferro-trace`, 23.992 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x128x128,rhs=1`): `jax-cpu` is 46.8x faster than the slowest successful cell (`tenferro-trace`, 5.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x128x128,rhs=1`): `pytorch-cpu` is 40.6x faster than the slowest successful cell (`tenferro-trace`, 5.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x128x128,rhs=1`): `tenferro-direct` is 65.4x faster than the slowest successful cell (`tenferro-trace`, 5.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `jax-cpu` is 40.3x faster than the slowest successful cell (`tenferro-trace`, 24.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `pytorch-cpu` is 53.8x faster than the slowest successful cell (`tenferro-trace`, 24.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `tenferro-direct` is 33.0x faster than the slowest successful cell (`tenferro-trace`, 24.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 26.5x faster than the slowest successful cell (`tenferro-trace`, 325.992 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `julia-base` is 30.6x faster than the slowest successful cell (`tenferro-trace`, 325.992 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 34.6x faster than the slowest successful cell (`tenferro-trace`, 325.992 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`256x128`): `jax-cpu` is 21.2x faster than the slowest successful cell (`tenferro-eager`, 29.543 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`256x128`): `pytorch-cpu` is 72.2x faster than the slowest successful cell (`tenferro-eager`, 29.543 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`256x128`): `tenferro-direct` is 46.3x faster than the slowest successful cell (`tenferro-eager`, 29.543 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`768x384`): `jax-cpu` is 262.8x faster than the slowest successful cell (`tenferro-eager`, 2296.482 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 498.5x faster than the slowest successful cell (`tenferro-eager`, 2296.482 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 12.1x faster than the slowest successful cell (`tenferro-trace`, 254.967 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `julia-base` is 17.9x faster than the slowest successful cell (`tenferro-trace`, 254.967 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 13.7x faster than the slowest successful cell (`tenferro-trace`, 254.967 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `jax-cpu` is 65.4x faster than the slowest successful cell (`tenferro-trace`, 2280.034 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `julia-base` is 438.3x faster than the slowest successful cell (`tenferro-trace`, 2280.034 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 445.8x faster than the slowest successful cell (`tenferro-trace`, 2280.034 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 11.7x faster than the slowest successful cell (`tenferro-trace`, 244.880 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 17.8x faster than the slowest successful cell (`tenferro-trace`, 244.880 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `julia-base` is 10.4x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 25.3x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 90.1x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `jax-cpu` is 11.6x faster than the slowest successful cell (`tenferro-trace`, 205.495 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `julia-base` is 11.7x faster than the slowest successful cell (`tenferro-trace`, 205.495 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 14.6x faster than the slowest successful cell (`tenferro-trace`, 205.495 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `jax-cpu` is 11.2x faster than the slowest successful cell (`tenferro-trace`, 202.529 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `julia-base` is 12.0x faster than the slowest successful cell (`tenferro-trace`, 202.529 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 14.5x faster than the slowest successful cell (`tenferro-trace`, 202.529 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 25.8x faster than the slowest successful cell (`tenferro-trace`, 317.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `julia-base` is 29.3x faster than the slowest successful cell (`tenferro-trace`, 317.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 33.3x faster than the slowest successful cell (`tenferro-trace`, 317.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 53.9x faster than the slowest successful cell (`tenferro-trace`, 2315.040 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `julia-base` is 51.6x faster than the slowest successful cell (`tenferro-trace`, 2315.040 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 70.5x faster than the slowest successful cell (`tenferro-trace`, 2315.040 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 1046.0x faster than the slowest successful cell (`tenferro-eager`, 23.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 375.2x faster than the slowest successful cell (`tenferro-eager`, 23.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 611.8x faster than the slowest successful cell (`tenferro-eager`, 23.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 27.2x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 28.2x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 42.3x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 103.5x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 193.9x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 617.6x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 13.6x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 23.9x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 42.7x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 124.4x faster than the slowest successful cell (`tenferro-eager`, 12.921 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 178.7x faster than the slowest successful cell (`tenferro-eager`, 12.921 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 257.7x faster than the slowest successful cell (`tenferro-eager`, 12.921 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 15.3x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 12.1x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 15.7x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 463.4x faster than the slowest successful cell (`tenferro-eager`, 65.169 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 1223.2x faster than the slowest successful cell (`tenferro-eager`, 65.169 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 871.0x faster than the slowest successful cell (`tenferro-eager`, 65.169 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 100.0x faster than the slowest successful cell (`tenferro-eager`, 95.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 263.5x faster than the slowest successful cell (`tenferro-eager`, 95.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 279.8x faster than the slowest successful cell (`tenferro-eager`, 95.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 456.1x faster than the slowest successful cell (`tenferro-eager`, 88.870 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 2041.1x faster than the slowest successful cell (`tenferro-eager`, 88.870 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 990.2x faster than the slowest successful cell (`tenferro-eager`, 88.870 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 85.6x faster than the slowest successful cell (`tenferro-eager`, 119.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 360.8x faster than the slowest successful cell (`tenferro-eager`, 119.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 250.0x faster than the slowest successful cell (`tenferro-eager`, 119.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 506.3x faster than the slowest successful cell (`tenferro-eager`, 112.736 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 668.7x faster than the slowest successful cell (`tenferro-eager`, 112.736 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 79.7x faster than the slowest successful cell (`tenferro-eager`, 142.376 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 105.2x faster than the slowest successful cell (`tenferro-eager`, 142.376 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 834.6x faster than the slowest successful cell (`tenferro-eager`, 23.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 2238.8x faster than the slowest successful cell (`tenferro-eager`, 23.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 2300.9x faster than the slowest successful cell (`tenferro-eager`, 23.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 106.9x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 625.9x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 477.9x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 258.8x faster than the slowest successful cell (`tenferro-eager`, 32.546 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 383.2x faster than the slowest successful cell (`tenferro-eager`, 32.546 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 903.3x faster than the slowest successful cell (`tenferro-eager`, 32.546 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 62.1x faster than the slowest successful cell (`tenferro-eager`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 218.3x faster than the slowest successful cell (`tenferro-eager`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 249.1x faster than the slowest successful cell (`tenferro-eager`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 117.8x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 196.7x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 858.7x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 17.0x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 26.8x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 62.3x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 118.4x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 204.6x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 835.1x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 13.9x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 26.5x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 59.7x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 490.8x faster than the slowest successful cell (`tenferro-eager`, 88.643 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 2003.4x faster than the slowest successful cell (`tenferro-eager`, 88.643 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 1020.7x faster than the slowest successful cell (`tenferro-eager`, 88.643 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 73.1x faster than the slowest successful cell (`tenferro-eager`, 119.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 370.2x faster than the slowest successful cell (`tenferro-eager`, 119.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 242.8x faster than the slowest successful cell (`tenferro-eager`, 119.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 77.0x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 104.8x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 297.1x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 13.5x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 19.8x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 13.0x faster than the slowest successful cell (`tenferro-eager`, 19.379 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 1019.3x faster than the slowest successful cell (`tenferro-eager`, 104.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 444.1x faster than the slowest successful cell (`tenferro-eager`, 104.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 258.8x faster than the slowest successful cell (`tenferro-eager`, 104.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 193.8x faster than the slowest successful cell (`tenferro-eager`, 141.375 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 75.9x faster than the slowest successful cell (`tenferro-eager`, 141.375 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 21.9x faster than the slowest successful cell (`tenferro-eager`, 141.375 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/output_reuse/solve_read_into` (f64, threads=4, shape=`768x768,rhs=16`): `pytorch-cpu` is 34.1x faster than the slowest successful cell (`tenferro-direct`, 183.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 74789.3x faster than the slowest successful cell (`julia-base`, 110.838 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 74623.7x faster than the slowest successful cell (`julia-base`, 108.354 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/cast_f64_f32` (f64->f32, threads=4, shape=`33554432`): `pytorch-cpu` is 11.1x faster than the slowest successful cell (`tenferro-trace`, 226.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/cast_f64_f32` (f64->f32, threads=4, shape=`33554432`): `tenferro-direct` is 11.1x faster than the slowest successful cell (`tenferro-trace`, 226.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 80268.3x faster than the slowest successful cell (`julia-base`, 128.590 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 77819.4x faster than the slowest successful cell (`julia-base`, 127.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 123224.0x faster than the slowest successful cell (`julia-base`, 177.689 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 124360.6x faster than the slowest successful cell (`julia-base`, 176.841 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 169.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 172.3x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 82.7x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 41.7x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 196.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 164.3x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 10275.1 GB/s at 0.002 ms. The conservative physical lower bound is 0.168 ms, so the cell is 102.8x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 10467.6 GB/s at 0.002 ms. The conservative physical lower bound is 0.042 ms, so the cell is 26.2x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 181130.5 GB/s at 0.001 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1811.3x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 184872.9 GB/s at 0.001 ms. The conservative physical lower bound is 0.671 ms, so the cell is 462.2x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 167562.7 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1675.6x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 164482.5 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 411.2x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 93077.5 GB/s at 0.001 ms. The conservative physical lower bound is 1.342 ms, so the cell is 930.8x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 94386.6 GB/s at 0.001 ms. The conservative physical lower bound is 0.336 ms, so the cell is 236.0x faster than the bound.

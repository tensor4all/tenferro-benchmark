# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20260806_153419/run.yaml`
- Timestamp: `20260806_153419`

Latest run: `PUBLICATION_GATE_PROFILE=full ./scripts/run_cpu_public_api.sh 1 4`.

- Sampling: `full` profile, 15 measured runs, 3 warmups per row

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20260806_153419`.

- tenferro-rs commit: `11a5b5a3c30b6258919557b4b69b429b6e686d75`

## CPU Information

- Model: `Apple M4`
- Vendor: `Apple`
- Logical CPUs: `10`
- Physical CPUs: `10`
- Sockets: `1`
- Cores per socket: `10`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Performance: 4 physical / 4 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 4 CPUs/L2); Efficiency: 6 physical / 6 logical (L1i 128 KiB, L1d 64 KiB, L2 4 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5.2-arm64-arm-64bit`

## Thread Environments

### Threads: 1

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260806_153419/run_t1.yaml`
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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`
- JULIA_NUM_THREADS: `1`

### Threads: 4

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260806_153419/run_t4.yaml`
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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4`
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
- PyTorch complex conjugation uses `torch.conj_physical` to match tenferro-rs physical output rather than the lazy conjugate view from `torch.conj`.
- `dot_general_with_conj` instead uses PyTorch's lazy conjugate view so conjugation can be handled by the contraction, matching tenferro-rs' conjugation flags; trace is unsupported because there is no equivalent public traced API. Julia's `dot_general_with_conj` row materializes the conjugate before the GEMM instead, since Julia has no lazy conj-without-transpose spelling that BLAS can fuse.
- `pad` has no natural Base spelling and stays missing for Julia.
- `dynamic_update_slice` reports trace mode as `unsupported` because tenferro-rs does not currently expose a corresponding `TracedTensor` API.
- `full_piv_lu` and `full_piv_lu_solve` are excluded because PyTorch has no direct public full-pivot equivalent; substituting `torch.linalg.solve` would compare different algorithms.
- `svd_full` remains in the table even when the selected tenferro-rs provider reports it as unsupported.
- Julia is column-major, like tenferro-rs, so the `julia-base`/`strided-jl` columns need no PyTorch/JAX-style layout reconstruction to keep the same logical fixture values.
- Julia warmup runs move JIT compilation outside the measured region, the same way PyTorch/JAX warmups do.
- Julia factorization rows materialize their factors inside the timed call (`cholesky` returns the factor matrix, `lu` returns P/L/U, `qr` returns Q/R). A Julia `Factorization` keeps its factors packed in LAPACK's working storage, so timing the compact object would compare strictly less work than the tenferro-rs and PyTorch columns, which return separate materialized tensors.
- Julia dense linalg rows run through Julia's own BLAS/LAPACK (libblastrampoline, by default OpenBLAS), recorded as `julia.blas_provider` in the run metadata. When that differs from the provider tenferro-rs and PyTorch link against (Accelerate on macOS), those rows partly compare BLAS implementations rather than framework overhead; read them together with the recorded providers.
- The Julia `lstsq` row uses `qr(a) \ rhs` rather than `a \ rhs`: the bare backslash runs a column-pivoted, rank-revealing QR (the LAPACK `gelsy` algorithm), while the PyTorch row selects the `gels` driver, so the unpivoted spelling is the like-for-like comparison.
- `julia-base` uses the natural Base/LinearAlgebra spelling and `strided-jl` the natural Strided.jl (`@strided`) spelling; each is populated only for rows where that spelling naturally applies (reductions and dense linalg have no natural Strided.jl spelling, so `strided-jl` covers elementwise/chain/transpose rows, the elementwise `cpu/output_reuse` `_into` rows, and the elementwise `cpu/complex` rows conj/mul/div/exp/log).
- Strided.jl (https://github.com/Jutho/Strided.jl) is prior art for tenferro-rs' strided-rs kernel layer; the `strided-jl` column credits that lineage directly in the report.

## Threads: 1 4

- CSV: `data/results/mac-cpu/cpu/public_api/20260806_153419/cpu_public_api_t1_20260806_153419.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20260806_153419/cpu_public_api_t4_20260806_153419.csv`
- Source table: `data/results/mac-cpu/cpu/public_api/20260806_153419/cpu_public_api_20260806_153419.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 2.664 ± 0.051 | 2.733 ± 0.048 | 3.098 ± 0.048 | 0.799 ± 0.021 | 3.079 ± 0.150 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.769 ± 0.039 | 2.688 ± 0.039 | 2.721 ± 0.034 | 0.793 ± 0.021 | 1.418 ± 0.298 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 6.409 ± 1.022 | 6.571 ± 0.452 | 6.462 ± 0.292 | 5.630 ± 0.374 | 6.533 ± 0.148 | 6.552 ± 0.282 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 5.642 ± 3.716 | 8.346 ± 2.138 | 5.498 ± 0.083 | 5.465 ± 0.100 | 6.314 ± 0.186 | 11.110 ± 8.383 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 7.539 ± 0.047 | 7.555 ± 0.103 | 11.388 ± 2.147 | 6.460 ± 0.448 | 21.250 ± 0.206 | 21.155 ± 0.337 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 4.420 ± 0.228 | 4.554 ± 0.251 | 4.278 ± 0.097 | 6.394 ± 0.434 | 24.843 ± 1.987 | 7.874 ± 6.600 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 6.573 ± 0.505 | 6.721 ± 0.146 | 6.233 ± 0.232 | 7.276 ± 0.482 | 42.762 ± 0.081 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 5.827 ± 0.642 | 6.243 ± 0.257 | 5.954 ± 0.187 | 7.427 ± 0.777 | 12.609 ± 0.337 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 6.820 ± 0.126 | unsupported | 6.308 ± 0.167 | 8.877 ± 1.254 | 43.011 ± 0.566 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 6.287 ± 0.185 | unsupported | 5.858 ± 0.149 | 7.873 ± 0.433 | 13.122 ± 0.351 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 6.155 ± 0.026 | 6.136 ± 0.015 | 6.134 ± 0.012 | 11.052 ± 0.141 | 5.505 ± 0.016 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 6.163 ± 0.017 | 6.212 ± 0.065 | 6.134 ± 0.015 | 11.699 ± 2.925 | 6.012 ± 0.056 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 23.513 ± 0.219 | 23.882 ± 0.407 | 24.047 ± 0.096 | 7.523 ± 0.151 | 27.284 ± 0.698 | 27.357 ± 0.283 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 9.395 ± 1.127 | 9.589 ± 1.367 | 7.476 ± 0.185 | 8.131 ± 0.966 | 29.379 ± 4.339 | 8.544 ± 1.805 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 25.997 ± 0.160 | 25.995 ± 0.023 | 28.103 ± 0.031 | 12.600 ± 0.515 | 39.166 ± 1.396 | 38.794 ± 2.542 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 10.231 ± 3.246 | 8.303 ± 0.240 | 9.185 ± 0.092 | 12.680 ± 0.475 | 43.624 ± 3.038 | 12.462 ± 2.746 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 5.269 ± 0.601 | 4.624 ± 0.305 | 4.565 ± 0.055 | 4.435 ± 0.232 | 4.497 ± 0.118 | 4.762 ± 0.292 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 4.358 ± 0.246 | 4.512 ± 0.363 | 4.269 ± 0.100 | 4.382 ± 0.166 | 4.574 ± 0.176 | 6.421 ± 11.507 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 4.910 ± 0.299 | 4.330 ± 0.065 | 3.977 ± 0.040 | 1.209 ± 0.056 | 6.234 ± 0.017 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 2.322 ± 0.568 | 1.745 ± 0.166 | 1.114 ± 0.022 | 1.206 ± 0.030 | 6.753 ± 0.121 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 4.576 ± 0.037 | 4.641 ± 0.034 | 4.806 ± 0.049 | 4.134 ± 0.560 | 6.348 ± 0.063 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 4.544 ± 0.044 | 4.552 ± 0.038 | 4.671 ± 0.046 | 3.816 ± 0.175 | 3.422 ± 0.525 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 3.403 ± 0.012 | 3.371 ± 0.030 | 3.137 ± 0.024 | 1.287 ± 0.009 | 3.834 ± 0.027 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.587 ± 0.036 | 3.576 ± 0.036 | 3.293 ± 0.046 | 1.300 ± 0.016 | 1.688 ± 0.258 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 3.731 ± 0.122 | 3.872 ± 0.017 | 3.797 ± 0.040 | 2.797 ± 0.083 | 4.814 ± 0.727 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 3.866 ± 0.020 | 3.923 ± 0.031 | 3.762 ± 0.016 | 2.780 ± 0.053 | 3.226 ± 0.102 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 6.273 ± 0.127 | 6.802 ± 0.204 | 6.179 ± 0.249 | 7.480 ± 0.638 | 42.485 ± 0.625 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 5.624 ± 0.184 | 6.170 ± 0.146 | 5.657 ± 0.197 | 8.099 ± 1.021 | 13.099 ± 0.118 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 5.118 ± 0.816 | unsupported | 5.436 ± 0.315 | 6.968 ± 0.119 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 5.090 ± 0.458 | unsupported | 4.713 ± 0.142 | 7.210 ± 0.295 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 6.508 ± 0.265 | 6.308 ± 0.246 | 6.402 ± 0.206 | 5.428 ± 0.072 | 6.713 ± 0.175 | 8.557 ± 0.085 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 5.418 ± 0.259 | 5.444 ± 0.098 | 5.566 ± 0.240 | 5.894 ± 0.590 | 6.308 ± 0.134 | 6.492 ± 4.617 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 8.545 ± 0.202 | 8.687 ± 0.359 | 8.777 ± 0.182 | 11.543 ± 0.173 | 9.079 ± 0.349 | 9.523 ± 0.389 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 8.868 ± 0.500 | 8.623 ± 0.219 | 8.665 ± 0.710 | 11.530 ± 0.190 | 8.791 ± 0.116 | 19.961 ± 17.663 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 18.113 ± 0.084 | 18.191 ± 0.058 | 22.144 ± 0.068 | 5.190 ± 0.219 | 23.826 ± 0.656 | 24.298 ± 0.093 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 6.469 ± 0.157 | 6.523 ± 0.045 | 7.619 ± 0.409 | 5.488 ± 0.378 | 24.699 ± 1.498 | 6.970 ± 0.635 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 3.000 ± 0.057 | 3.023 ± 0.026 | 2.945 ± 0.054 | 3.759 ± 0.039 | 3.150 ± 0.232 | 3.374 ± 0.135 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 3.034 ± 0.117 | 3.043 ± 0.115 | 2.970 ± 0.058 | 3.805 ± 0.030 | 3.165 ± 0.364 | 2.999 ± 0.517 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 7.437 ± 0.027 | 7.483 ± 0.035 | 8.569 ± 0.043 | 7.836 ± 0.016 | 7.869 ± 0.960 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 7.370 ± 0.052 | 7.403 ± 0.010 | 7.362 ± 0.172 | 7.832 ± 0.031 | 7.872 ± 0.213 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 17.467 ± 0.148 | 17.256 ± 0.139 | 20.914 ± 0.037 | 4.116 ± 0.015 | 22.403 ± 0.132 | 22.567 ± 0.084 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 5.526 ± 0.489 | 5.400 ± 0.038 | 6.648 ± 0.495 | 4.705 ± 0.454 | 25.501 ± 2.424 | 6.884 ± 1.046 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 8.589 ± 0.164 | 8.654 ± 0.117 | 8.906 ± 0.142 | 10.766 ± 0.314 | 9.147 ± 0.253 | 9.489 ± 0.201 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 8.500 ± 0.140 | 8.586 ± 0.251 | 8.656 ± 0.203 | 11.629 ± 0.170 | 10.890 ± 1.852 | 10.059 ± 3.884 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 17.519 ± 0.112 | 17.870 ± 0.099 | 14.847 ± 0.024 | 2.884 ± 0.008 | 20.846 ± 0.107 | 20.850 ± 0.255 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 5.142 ± 0.890 | 5.316 ± 0.397 | 5.149 ± 1.793 | 3.432 ± 0.382 | 22.027 ± 1.082 | 9.288 ± 3.037 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 8.887 ± 0.081 | 8.906 ± 0.027 | 17.699 ± 0.027 | 3.348 ± 0.007 | 13.874 ± 0.585 | 13.897 ± 0.062 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 2.653 ± 0.992 | 2.777 ± 0.028 | 5.472 ± 0.207 | 3.527 ± 0.328 | 15.127 ± 0.260 | 4.003 ± 0.200 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 16.282 ± 0.090 | 16.648 ± 0.077 | 26.182 ± 0.019 | 4.596 ± 0.011 | 25.173 ± 0.129 | 25.041 ± 0.108 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 5.241 ± 0.156 | 5.299 ± 0.222 | 7.877 ± 0.274 | 5.279 ± 0.687 | 26.348 ± 2.626 | 7.376 ± 0.421 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 10.815 ± 0.049 | 10.811 ± 0.039 | 13.574 ± 0.040 | 3.254 ± 0.805 | 13.176 ± 0.033 | 13.242 ± 0.080 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 3.345 ± 0.513 | 3.213 ± 0.117 | 4.561 ± 1.862 | 3.441 ± 0.382 | 14.741 ± 0.780 | 3.918 ± 0.158 |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 8.922 ± 0.309 | 8.980 ± 0.214 | 8.630 ± 0.136 | 11.648 ± 0.327 | 9.114 ± 0.258 | 9.506 ± 0.526 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 8.582 ± 0.251 | 8.480 ± 0.173 | 8.696 ± 0.094 | 11.614 ± 0.110 | 8.907 ± 0.211 | 8.803 ± 3.362 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 8.923 ± 0.200 | 9.091 ± 0.528 | 8.619 ± 0.140 | 11.570 ± 0.215 | 8.975 ± 0.171 | 9.528 ± 0.245 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 8.515 ± 0.109 | 8.479 ± 0.118 | 8.608 ± 0.234 | 11.621 ± 0.114 | 8.954 ± 0.354 | 9.275 ± 3.111 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 8.700 ± 0.142 | 8.716 ± 0.159 | 8.824 ± 0.241 | 11.434 ± 0.785 | 9.272 ± 0.351 | 9.653 ± 0.303 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 8.591 ± 0.090 | 8.668 ± 0.225 | 8.804 ± 0.166 | 11.663 ± 0.189 | 8.929 ± 0.276 | 9.725 ± 5.418 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 6.167 ± 0.233 | 6.420 ± 0.302 | 6.350 ± 0.175 | 5.486 ± 0.216 | 6.696 ± 0.372 | 8.519 ± 0.100 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 5.375 ± 0.078 | 5.416 ± 0.132 | 5.726 ± 0.471 | 5.603 ± 0.148 | 6.489 ± 0.355 | 5.512 ± 3.237 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 20.092 ± 0.042 | 20.121 ± 0.039 | 45.742 ± 0.963 | 5.854 ± 0.029 | 37.684 ± 1.343 | 37.926 ± 0.105 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 6.026 ± 0.688 | 5.929 ± 0.034 | 13.028 ± 0.233 | 6.388 ± 0.614 | 37.204 ± 1.194 | 11.339 ± 0.425 |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 5.873 ± 0.062 | 5.839 ± 0.029 | 9.009 ± 0.062 | 0.576 ± 0.016 | 0.423 ± 0.021 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 6.122 ± 0.009 | 6.131 ± 0.011 | 6.174 ± 0.055 | 0.580 ± 0.014 | 0.426 ± 0.017 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 3.730 ± 0.013 | 3.738 ± 0.005 | 4.267 ± 0.040 | 1.956 ± 0.007 | 2.195 ± 0.167 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 3.740 ± 0.014 | 3.754 ± 0.004 | 1.737 ± 0.013 | 1.954 ± 0.012 | 2.086 ± 0.020 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 3.988 ± 0.092 | 4.142 ± 0.230 | 3.932 ± 0.109 | 5.052 ± 0.021 | 3.948 ± 0.076 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 8.211 ± 0.766 | 7.503 ± 0.012 | 3.697 ± 0.004 | 5.064 ± 0.054 | 3.915 ± 0.038 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 4.225 ± 0.223 | 3.936 ± 0.132 | 3.883 ± 0.056 | 4.000 ± 0.047 | 3.933 ± 0.079 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 4.975 ± 0.661 | 4.776 ± 0.023 | 3.701 ± 0.008 | 3.960 ± 0.018 | 3.884 ± 0.052 | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 19.348 ± 0.310 | 19.414 ± 0.233 | 22.493 ± 0.045 | 6.300 ± 0.062 | 15.745 ± 1.726 | 15.531 ± 0.227 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 4.985 ± 0.297 | 5.019 ± 0.516 | 5.804 ± 0.502 | 6.640 ± 0.235 | 16.495 ± 1.106 | 4.734 ± 1.161 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 12.415 ± 0.099 | 12.617 ± 0.042 | 12.695 ± 0.021 | 8.662 ± 1.805 | 29.729 ± 0.344 | 29.941 ± 0.153 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 5.415 ± 0.181 | 5.436 ± 0.054 | 5.473 ± 0.219 | 6.623 ± 0.036 | 30.058 ± 0.362 | 7.896 ± 3.108 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 12.019 ± 0.166 | 12.065 ± 0.101 | 9.391 ± 0.165 | 12.212 ± 0.785 | 12.253 ± 0.174 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 11.817 ± 0.105 | 11.823 ± 0.189 | 8.906 ± 0.280 | 12.057 ± 0.268 | 12.177 ± 0.106 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 6.454 ± 0.204 | 6.373 ± 0.196 | 6.436 ± 0.296 | 5.427 ± 0.078 | 6.691 ± 0.261 | 12.654 ± 0.050 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 5.907 ± 0.331 | 5.842 ± 0.141 | 5.713 ± 0.370 | 5.627 ± 0.108 | 6.586 ± 1.139 | 5.651 ± 3.204 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 16.164 ± 0.068 | 16.361 ± 0.134 | 17.987 ± 0.038 | 4.087 ± 0.028 | 20.175 ± 0.045 | 20.168 ± 0.076 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 5.056 ± 0.086 | 5.120 ± 0.284 | 5.685 ± 0.291 | 4.062 ± 0.032 | 21.129 ± 0.725 | 6.024 ± 1.012 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 8.322 ± 0.019 | 8.341 ± 0.018 | 8.340 ± 0.037 | 5.515 ± 0.185 | 17.807 ± 0.467 | 17.907 ± 0.349 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 5.420 ± 0.132 | 5.477 ± 0.312 | 5.476 ± 0.264 | 5.490 ± 0.127 | 17.878 ± 0.251 | 5.723 ± 3.743 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 8.618 ± 0.164 | 8.640 ± 0.177 | 8.788 ± 0.431 | 11.541 ± 0.131 | 8.968 ± 0.233 | 9.690 ± 0.502 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 8.514 ± 0.141 | 8.656 ± 0.379 | 8.817 ± 0.276 | 11.697 ± 1.333 | 8.899 ± 0.170 | 12.181 ± 11.760 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 22.733 ± 0.031 | 22.740 ± 0.026 | 42.631 ± 0.090 | 2.705 ± 0.014 | 26.942 ± 0.614 | 26.924 ± 0.161 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 6.929 ± 0.313 | 7.066 ± 0.983 | 12.863 ± 0.317 | 2.716 ± 0.148 | 29.029 ± 1.446 | 8.319 ± 1.201 |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.299 ± 0.009 | 0.731 ± 0.070 | 0.323 ± 0.029 | 0.354 ± 0.047 | 0.458 ± 0.129 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.311 ± 0.006 | 0.717 ± 0.035 | 0.255 ± 0.015 | 0.314 ± 0.033 | 0.399 ± 0.032 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.311 ± 0.035 | 1.104 ± 0.040 | 0.304 ± 0.018 | 0.257 ± 0.004 | 0.427 ± 0.031 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.324 ± 0.025 | 1.144 ± 0.077 | 0.260 ± 0.013 | 0.264 ± 0.019 | 0.408 ± 0.034 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 0.531 ± 0.039 | unsupported | 0.514 ± 0.033 | 0.469 ± 0.117 | 0.616 ± 0.036 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.504 ± 0.020 | unsupported | 0.499 ± 0.016 | 0.458 ± 0.018 | 0.571 ± 0.027 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 3.290 ± 0.853 | 2.911 ± 0.005 | 0.180 ± 0.001 | 0.066 ± 0.003 | 0.297 ± 0.013 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 1.060 ± 0.181 | 0.904 ± 0.090 | 0.180 ± 0.001 | 0.079 ± 0.010 | 0.292 ± 0.096 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.493 ± 0.010 | 0.915 ± 0.038 | 0.548 ± 0.021 | 0.252 ± 0.007 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.504 ± 0.007 | 0.888 ± 0.031 | 0.492 ± 0.006 | 0.251 ± 0.005 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.780 ± 0.000 | 1.112 ± 0.009 | 0.333 ± 0.121 | 0.266 ± 0.012 | 0.848 ± 0.012 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.791 ± 0.004 | 1.062 ± 0.009 | 0.263 ± 0.009 | 0.258 ± 0.006 | 0.843 ± 0.007 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 2.532 ± 0.004 | 2.661 ± 0.005 | 0.285 ± 0.005 | 0.247 ± 0.015 | 0.310 ± 0.041 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 2.543 ± 0.207 | 2.649 ± 0.054 | 0.264 ± 0.007 | 0.231 ± 0.008 | 0.316 ± 0.062 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 0.795 ± 0.020 | 1.502 ± 0.022 | 0.563 ± 0.028 | 0.487 ± 0.021 | 0.697 ± 0.271 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.797 ± 0.006 | 1.543 ± 0.051 | 0.505 ± 0.013 | 0.478 ± 0.015 | 0.682 ± 0.027 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 4.983 ± 0.537 | 5.616 ± 0.163 | 14.182 ± 0.310 | 6.877 ± 0.191 | 27.741 ± 5.117 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 5.151 ± 0.555 | 5.715 ± 0.046 | 7.512 ± 0.385 | 6.835 ± 0.213 | 12.773 ± 15.907 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 5.171 ± 0.101 | 5.311 ± 0.072 | 4.262 ± 0.022 | 4.171 ± 0.095 | 14.578 ± 0.257 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 5.275 ± 0.069 | 5.394 ± 0.041 | 4.360 ± 0.181 | 4.198 ± 0.115 | 6.938 ± 0.818 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 4.736 ± 0.083 | 4.735 ± 0.062 | 4.718 ± 0.016 | 5.451 ± 0.232 | 4.265 ± 0.021 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 4.724 ± 0.045 | 4.749 ± 0.037 | 4.720 ± 0.051 | 5.465 ± 0.085 | 4.635 ± 0.451 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 4.906 ± 0.039 | 4.990 ± 0.241 | 4.912 ± 0.046 | 4.441 ± 0.010 | 4.783 ± 0.070 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 4.885 ± 0.027 | 4.955 ± 0.046 | 4.934 ± 0.054 | 4.592 ± 0.049 | 5.452 ± 0.112 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 12.831 ± 0.607 | 14.418 ± 4.682 | 13.345 ± 0.019 | 12.772 ± 0.172 | 11.923 ± 0.099 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 12.491 ± 0.132 | 13.085 ± 0.226 | 13.326 ± 0.339 | 12.876 ± 0.174 | 7.983 ± 0.175 | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 7.520 ± 0.069 | 6.730 ± 0.075 | 5.212 ± 0.134 | 4.947 ± 0.024 | 20.450 ± 0.105 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 7.146 ± 0.024 | 6.409 ± 0.248 | 4.623 ± 0.084 | 4.563 ± 0.037 | 8.670 ± 1.488 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | 5.372 ± 0.044 | 5.153 ± 0.034 | 3.076 ± 0.010 | 12.812 ± 2.941 | 5.474 ± 0.012 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | 5.479 ± 0.073 | 5.483 ± 0.239 | 2.956 ± 0.057 | 13.054 ± 0.157 | 4.117 ± 1.230 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 5.443 ± 0.400 | 5.774 ± 0.215 | 6.732 ± 0.039 | 5.042 ± 0.132 | 17.048 ± 4.233 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 4.994 ± 0.095 | 5.348 ± 0.086 | 6.184 ± 0.077 | 5.765 ± 0.644 | 9.196 ± 3.788 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.416 ± 0.024 | 0.436 ± 0.015 | 1.360 ± 0.002 | 1.161 ± 0.058 | 4.161 ± 0.049 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.621 ± 0.013 | 0.657 ± 0.114 | 1.360 ± 0.006 | 1.159 ± 0.027 | 4.588 ± 0.105 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 6.462 ± 0.018 | 6.523 ± 0.071 | 6.350 ± 0.019 | 6.331 ± 0.045 | 14.092 ± 0.140 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 6.607 ± 0.345 | 6.950 ± 0.527 | 6.365 ± 0.071 | 6.475 ± 0.155 | 12.032 ± 1.871 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 6.430 ± 0.025 | 6.462 ± 0.053 | 6.340 ± 0.014 | 6.317 ± 0.059 | 14.018 ± 0.108 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 6.588 ± 0.680 | 6.754 ± 0.436 | 6.469 ± 0.069 | 6.756 ± 0.511 | 11.327 ± 0.270 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 5.093 ± 0.032 | 5.633 ± 0.170 | 4.311 ± 0.044 | 4.150 ± 0.037 | 14.559 ± 0.139 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 5.216 ± 0.128 | 5.668 ± 0.056 | 4.371 ± 0.251 | 4.172 ± 0.054 | 6.832 ± 0.383 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | unsupported | unsupported | 17.915 ± 0.062 | 17.948 ± 0.593 | 48.046 ± 4.788 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | unsupported | unsupported | 18.840 ± 0.209 | 18.521 ± 0.339 | 35.992 ± 2.691 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 6.428 ± 0.479 | 9.696 ± 0.804 | 4.485 ± 0.034 | 15.062 ± 0.153 | 21.913 ± 0.070 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 6.789 ± 0.130 | 9.566 ± 0.149 | 4.971 ± 0.115 | 15.989 ± 0.220 | 9.358 ± 0.234 | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 24.974 ± 0.029 | unsupported | 8.695 ± 0.203 | - | 8.709 ± 0.215 | 9.471 ± 0.600 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 8.595 ± 0.185 | unsupported | 9.187 ± 0.420 | - | 8.691 ± 0.149 | 8.786 ± 0.364 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 10.543 ± 0.037 | unsupported | 6.342 ± 0.296 | - | 6.600 ± 0.467 | 6.884 ± 0.614 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 5.515 ± 0.270 | unsupported | 5.579 ± 0.611 | - | 6.224 ± 0.169 | 5.338 ± 0.167 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 6.206 ± 0.363 | unsupported | 6.167 ± 0.353 | - | 6.163 ± 0.221 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 5.496 ± 0.305 | unsupported | 5.592 ± 0.462 | - | 5.986 ± 0.185 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 29.481 ± 0.365 | unsupported | 9.621 ± 0.994 | - | 8.762 ± 0.158 | 9.118 ± 0.422 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 8.551 ± 0.114 | unsupported | 8.561 ± 0.259 | - | 8.627 ± 0.154 | 8.466 ± 0.147 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 4.966 ± 0.085 | unsupported | 5.218 ± 0.336 | - | 38.250 ± 0.335 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 4.553 ± 0.028 | unsupported | 4.945 ± 0.651 | - | 11.943 ± 0.152 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 5.695 ± 0.055 | unsupported | 6.006 ± 0.225 | - | 38.140 ± 0.056 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 5.188 ± 0.196 | unsupported | 5.321 ± 0.326 | - | 13.416 ± 2.288 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 25.748 ± 1.103 | unsupported | 8.731 ± 0.326 | - | 8.739 ± 0.096 | 9.275 ± 0.428 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 8.531 ± 0.148 | unsupported | 9.258 ± 0.218 | - | 8.759 ± 0.177 | 8.556 ± 0.161 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 16.661 ± 0.030 | unsupported | 7.254 ± 0.974 | - | 6.240 ± 0.251 | 8.422 ± 0.188 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 5.417 ± 0.105 | unsupported | 5.428 ± 0.454 | - | 6.244 ± 0.192 | 5.362 ± 0.094 |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 25.887 ± 1.532 | unsupported | 8.759 ± 0.194 | - | 9.012 ± 0.486 | 9.385 ± 0.386 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 8.584 ± 0.227 | unsupported | 9.438 ± 0.617 | - | 8.832 ± 0.287 | 8.521 ± 0.186 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 3.745 ± 0.140 | 3.666 ± 0.015 | 3.652 ± 0.005 | 3.666 ± 0.004 | 9.218 ± 0.143 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 3.493 ± 0.177 | 3.542 ± 0.013 | 3.655 ± 0.005 | 3.663 ± 0.008 | 9.148 ± 0.052 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 4.983 ± 0.188 | 11.193 ± 0.410 | 5.103 ± 0.121 | 4.598 ± 0.177 | 6.322 ± 1.251 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 4.238 ± 0.107 | 9.707 ± 0.229 | 4.254 ± 0.084 | 4.243 ± 0.111 | 5.190 ± 0.214 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 9.382 ± 0.360 | 9.864 ± 0.742 | 7.802 ± 3.353 | 7.401 ± 0.820 | 5.636 ± 3.905 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 9.893 ± 1.987 | 15.758 ± 5.428 | 7.810 ± 1.315 | 7.247 ± 0.554 | 5.662 ± 0.439 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 3.006 ± 0.182 | 9.234 ± 0.317 | 6.342 ± 0.085 | 4.527 ± 0.114 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 3.261 ± 0.191 | 8.584 ± 0.250 | 4.345 ± 0.116 | 4.253 ± 0.104 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 6.362 ± 0.485 | 6.850 ± 1.019 | 6.420 ± 0.242 | 5.304 ± 0.078 | 7.064 ± 0.582 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 6.067 ± 0.176 | 5.503 ± 0.136 | 5.509 ± 0.137 | 5.243 ± 0.179 | 6.277 ± 0.298 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 21.169 ± 0.267 | 21.700 ± 0.880 | 19.201 ± 0.114 | 7.261 ± 0.115 | 52.767 ± 2.944 | 20.858 ± 0.669 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 11.074 ± 0.328 | 11.321 ± 0.300 | 19.674 ± 0.371 | 7.264 ± 0.184 | 53.265 ± 3.576 | 13.918 ± 10.120 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 4.625 ± 0.151 | 7.975 ± 0.395 | 4.890 ± 0.119 | 2.728 ± 0.074 | 5.759 ± 0.243 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 4.745 ± 0.211 | 7.580 ± 0.191 | 3.047 ± 0.061 | 2.733 ± 0.053 | 6.125 ± 0.616 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 4.840 ± 0.140 | 8.406 ± 0.333 | 4.609 ± 0.141 | 2.905 ± 0.295 | 5.362 ± 0.192 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 4.954 ± 0.153 | 8.014 ± 0.375 | 2.972 ± 0.059 | 2.725 ± 0.037 | 5.614 ± 0.516 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.002 ± 0.000 | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.002 ± 0.000 | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 9.089 ± 3.083 | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 7.928 ± 0.239 | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.741 ± 0.060 | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.693 ± 0.031 | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 3.283 ± 0.257 | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 2.944 ± 0.100 | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 15.7x faster than the slowest successful cell (`pytorch-cpu`, 9.009 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 21.3x faster than the slowest successful cell (`pytorch-cpu`, 9.009 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 6.174 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 6.174 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/tanh` (f64, threads=1, shape=`8388608`): `jax-cpu` is 15.8x faster than the slowest successful cell (`pytorch-cpu`, 42.631 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/tanh` (f64, threads=4, shape=`8388608`): `jax-cpu` is 10.7x faster than the slowest successful cell (`julia-base`, 29.029 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=1, shape=`262144`): `jax-cpu` is 49.5x faster than the slowest successful cell (`tenferro-eager`, 3.290 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=1, shape=`262144`): `julia-base` is 11.1x faster than the slowest successful cell (`tenferro-eager`, 3.290 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=1, shape=`262144`): `pytorch-cpu` is 18.3x faster than the slowest successful cell (`tenferro-eager`, 3.290 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `jax-cpu` is 13.4x faster than the slowest successful cell (`tenferro-eager`, 1.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=1, shape=`262144`): `jax-cpu` is 10.8x faster than the slowest successful cell (`tenferro-trace`, 2.661 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `jax-cpu` is 11.5x faster than the slowest successful cell (`tenferro-trace`, 2.649 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 10.0x faster than the slowest successful cell (`tenferro-trace`, 2.649 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `pytorch-cpu` is 16799.8x faster than the slowest successful cell (`tenferro-eager`, 9.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `pytorch-cpu` is 12684.6x faster than the slowest successful cell (`tenferro-eager`, 7.928 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `pytorch-cpu` is 1366.2x faster than the slowest successful cell (`tenferro-eager`, 0.741 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `pytorch-cpu` is 1189.0x faster than the slowest successful cell (`tenferro-eager`, 0.693 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `pytorch-cpu` is 5252.3x faster than the slowest successful cell (`tenferro-eager`, 3.283 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `pytorch-cpu` is 4157.9x faster than the slowest successful cell (`tenferro-eager`, 2.944 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

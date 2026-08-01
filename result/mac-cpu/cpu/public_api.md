# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20260801_220212/run.yaml`
- Timestamp: `20260801_220212`

Latest run: `PUBLICATION_GATE_PROFILE=full ./scripts/run_cpu_public_api.sh 1 4`.

- Sampling: `full` profile, 15 measured runs, 3 warmups per row

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20260801_220212`.

- tenferro-rs commit: `0ee2d0dc2f8d21ff62ea682f90f34e4319108ace`

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

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260801_220212/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260801_220212/run_t4.yaml`
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

- CSV: `data/results/mac-cpu/cpu/public_api/20260801_220212/cpu_public_api_t1_20260801_220212.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20260801_220212/cpu_public_api_t4_20260801_220212.csv`
- Source table: `data/results/mac-cpu/cpu/public_api/20260801_220212/cpu_public_api_20260801_220212.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 2.688 ± 0.044 | 2.673 ± 0.031 | 3.441 ± 0.152 | 0.820 ± 0.005 | 3.263 ± 0.135 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.697 ± 0.047 | 2.668 ± 0.029 | 3.191 ± 0.049 | 0.796 ± 0.016 | 1.803 ± 0.323 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 6.853 ± 0.568 | 6.054 ± 0.103 | 8.724 ± 1.820 | 5.531 ± 0.023 | 9.472 ± 2.091 | 8.114 ± 2.863 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 5.333 ± 0.078 | 5.337 ± 0.189 | 5.320 ± 0.079 | 5.332 ± 0.030 | 6.142 ± 0.059 | 16.188 ± 8.237 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 7.517 ± 0.008 | 7.531 ± 0.017 | 10.580 ± 0.054 | 9.573 ± 0.863 | 27.732 ± 2.227 | 29.761 ± 0.037 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 4.179 ± 0.027 | 4.207 ± 0.009 | 4.671 ± 0.339 | 7.595 ± 1.159 | 23.874 ± 0.170 | 7.450 ± 5.308 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 6.131 ± 0.707 | 6.069 ± 0.124 | 6.356 ± 0.319 | 10.852 ± 1.574 | 47.331 ± 1.720 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 5.669 ± 0.738 | 5.647 ± 0.191 | 8.863 ± 0.435 | 9.786 ± 1.136 | 13.333 ± 2.551 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 6.569 ± 0.182 | unsupported | 6.964 ± 0.253 | 12.675 ± 0.966 | 47.515 ± 0.267 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 5.985 ± 0.166 | unsupported | 9.165 ± 0.348 | 12.015 ± 1.290 | 18.524 ± 1.821 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 6.070 ± 0.011 | 6.125 ± 0.009 | 6.850 ± 0.021 | 13.800 ± 0.849 | 5.941 ± 0.029 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 6.170 ± 0.024 | 6.176 ± 0.008 | 7.674 ± 0.058 | 11.853 ± 0.017 | 7.275 ± 0.025 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 23.375 ± 0.095 | 23.754 ± 1.500 | 27.472 ± 0.897 | 11.438 ± 1.184 | 35.633 ± 0.174 | 33.431 ± 2.169 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 7.101 ± 0.329 | 8.032 ± 0.648 | 10.179 ± 1.032 | 9.259 ± 2.105 | 33.644 ± 3.375 | 9.690 ± 3.184 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 25.962 ± 0.023 | 26.029 ± 1.392 | 31.334 ± 0.015 | 19.636 ± 1.970 | 47.427 ± 2.440 | 45.475 ± 1.855 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 11.180 ± 1.117 | 11.238 ± 0.163 | 13.626 ± 0.422 | 16.676 ± 1.868 | 49.444 ± 3.359 | 16.601 ± 2.599 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 4.797 ± 0.065 | 4.330 ± 0.042 | 5.136 ± 0.456 | 4.417 ± 0.084 | 5.216 ± 0.688 | 4.675 ± 0.488 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 4.190 ± 0.025 | 4.218 ± 0.032 | 4.351 ± 0.163 | 4.216 ± 0.016 | 4.676 ± 0.249 | 7.326 ± 10.957 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 4.766 ± 0.021 | 4.294 ± 0.018 | 4.396 ± 0.015 | 1.204 ± 0.023 | 6.501 ± 0.002 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 2.148 ± 0.021 | 1.794 ± 0.160 | 1.227 ± 0.007 | 1.235 ± 0.105 | 8.787 ± 0.002 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 4.505 ± 0.030 | 4.601 ± 0.052 | 5.287 ± 0.022 | 5.079 ± 0.239 | 6.842 ± 0.083 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 4.605 ± 0.024 | 4.625 ± 0.030 | 6.077 ± 0.454 | 4.275 ± 0.228 | 5.042 ± 0.036 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 3.400 ± 0.020 | 3.354 ± 0.027 | 3.654 ± 0.089 | 1.380 ± 0.014 | 4.000 ± 0.020 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.544 ± 0.032 | 3.565 ± 0.023 | 4.118 ± 0.447 | 1.289 ± 0.017 | 1.972 ± 0.026 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 3.812 ± 0.034 | 3.837 ± 0.028 | 3.921 ± 0.192 | 4.364 ± 0.268 | 5.105 ± 0.029 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 3.853 ± 0.018 | 3.869 ± 0.018 | 5.078 ± 0.440 | 3.625 ± 0.034 | 4.985 ± 0.025 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 5.913 ± 0.199 | 5.988 ± 0.146 | 7.118 ± 0.331 | 15.398 ± 1.004 | 46.052 ± 1.125 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 5.595 ± 0.181 | 5.633 ± 0.174 | 7.822 ± 0.256 | 13.445 ± 2.048 | 20.498 ± 0.873 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 5.211 ± 0.650 | unsupported | 8.201 ± 0.130 | 9.556 ± 2.162 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 4.834 ± 0.670 | unsupported | 9.438 ± 0.053 | 8.712 ± 2.096 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 6.045 ± 0.122 | 6.003 ± 0.020 | 6.014 ± 0.086 | 6.382 ± 0.405 | 14.057 ± 2.236 | 14.752 ± 0.416 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 6.797 ± 0.622 | 7.889 ± 0.578 | 5.665 ± 0.011 | 5.680 ± 0.092 | 14.087 ± 0.405 | 6.231 ± 5.946 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 8.568 ± 0.431 | 8.531 ± 0.041 | 8.505 ± 0.022 | 11.548 ± 0.063 | 8.602 ± 0.185 | 9.149 ± 0.487 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 8.341 ± 0.044 | 8.614 ± 0.394 | 8.335 ± 0.050 | 11.410 ± 0.099 | 8.578 ± 0.229 | 11.594 ± 13.273 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 18.729 ± 0.832 | 19.055 ± 1.073 | 22.092 ± 0.117 | 9.985 ± 1.371 | 27.718 ± 0.316 | 27.445 ± 1.481 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 10.590 ± 2.441 | 11.006 ± 0.120 | 12.105 ± 1.590 | 7.678 ± 0.966 | 27.302 ± 1.205 | 8.047 ± 1.250 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 2.971 ± 0.026 | 2.998 ± 0.018 | 3.181 ± 0.097 | 4.130 ± 0.103 | 3.350 ± 0.211 | 5.153 ± 0.436 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 2.951 ± 0.044 | 3.434 ± 0.102 | 4.483 ± 0.340 | 4.132 ± 0.036 | 3.290 ± 0.286 | 3.741 ± 1.526 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 7.474 ± 0.127 | 7.803 ± 0.099 | 9.338 ± 0.292 | 9.990 ± 1.477 | 9.849 ± 0.055 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 10.046 ± 0.536 | 7.308 ± 0.512 | 9.762 ± 0.976 | 8.372 ± 0.291 | 8.872 ± 0.068 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 17.412 ± 0.701 | 18.322 ± 0.160 | 20.862 ± 0.014 | 7.518 ± 1.416 | 29.269 ± 0.754 | 27.616 ± 0.058 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 11.226 ± 0.869 | 11.323 ± 1.168 | 13.966 ± 0.010 | 7.554 ± 0.900 | 27.495 ± 1.988 | 8.011 ± 1.406 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 8.509 ± 0.061 | 8.505 ± 0.061 | 8.500 ± 0.093 | 12.661 ± 0.252 | 9.775 ± 0.095 | 11.888 ± 0.522 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 9.630 ± 0.179 | 9.855 ± 0.586 | 9.352 ± 0.036 | 11.589 ± 0.231 | 9.975 ± 0.192 | 9.398 ± 4.271 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 18.368 ± 0.629 | 18.075 ± 0.894 | 14.813 ± 0.011 | 4.827 ± 0.439 | 35.724 ± 3.655 | 32.107 ± 0.083 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 7.746 ± 0.490 | 9.539 ± 0.817 | 18.336 ± 0.026 | 4.820 ± 0.017 | 22.533 ± 0.172 | 6.141 ± 0.926 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 8.928 ± 0.334 | 9.627 ± 0.226 | 17.671 ± 0.027 | 5.160 ± 0.648 | 16.668 ± 0.849 | 17.003 ± 0.083 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 3.446 ± 0.762 | 3.458 ± 0.006 | 8.630 ± 0.059 | 4.909 ± 0.519 | 15.451 ± 0.542 | 4.137 ± 0.183 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 17.217 ± 0.892 | 16.937 ± 0.521 | 26.136 ± 0.013 | 7.269 ± 1.175 | 38.808 ± 0.093 | 35.330 ± 0.240 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 9.193 ± 1.076 | 10.380 ± 1.188 | 20.580 ± 0.021 | 7.251 ± 0.791 | 29.335 ± 0.894 | 8.692 ± 1.881 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 11.785 ± 0.333 | 10.803 ± 0.872 | 13.569 ± 0.030 | 5.099 ± 0.018 | 16.121 ± 0.026 | 15.710 ± 0.495 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 4.716 ± 0.776 | 4.714 ± 0.015 | 6.737 ± 0.989 | 4.526 ± 0.017 | 14.733 ± 0.036 | 4.072 ± 0.521 |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 8.554 ± 0.061 | 8.591 ± 0.045 | 8.522 ± 0.075 | 11.450 ± 1.674 | 10.157 ± 0.834 | 12.845 ± 0.585 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 16.429 ± 3.082 | 13.137 ± 1.039 | 10.983 ± 0.070 | 12.855 ± 0.072 | 10.244 ± 1.104 | 9.415 ± 3.628 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 8.619 ± 0.037 | 9.600 ± 0.257 | 8.600 ± 0.157 | 12.928 ± 1.091 | 12.053 ± 1.099 | 15.440 ± 0.854 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 13.928 ± 3.823 | 13.202 ± 0.257 | 8.753 ± 0.427 | 12.859 ± 0.129 | 10.937 ± 1.082 | 9.563 ± 5.383 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 8.561 ± 0.022 | 8.588 ± 0.043 | 8.467 ± 0.028 | 11.708 ± 0.687 | 9.785 ± 0.177 | 11.609 ± 0.330 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 9.418 ± 0.200 | 9.636 ± 0.058 | 8.381 ± 0.156 | 11.523 ± 0.065 | 9.715 ± 0.249 | 8.531 ± 4.466 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 6.117 ± 0.152 | 6.021 ± 0.047 | 6.005 ± 0.022 | 6.267 ± 0.866 | 14.013 ± 3.556 | 14.660 ± 0.362 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 5.657 ± 0.108 | 5.806 ± 0.110 | 5.414 ± 0.060 | 5.618 ± 0.041 | 14.107 ± 0.201 | 5.562 ± 4.716 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 20.067 ± 0.015 | 20.094 ± 0.030 | 44.861 ± 0.061 | 7.750 ± 1.302 | 41.621 ± 1.566 | 44.222 ± 0.720 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 6.155 ± 0.470 | 6.094 ± 0.691 | 21.727 ± 0.021 | 7.067 ± 1.180 | 37.574 ± 1.492 | 10.782 ± 0.511 |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 6.105 ± 0.002 | 6.114 ± 0.001 | 8.919 ± 0.114 | 0.814 ± 0.099 | 0.407 ± 0.016 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 7.986 ± 0.015 | 8.001 ± 0.471 | 9.977 ± 0.380 | 0.629 ± 0.029 | 0.405 ± 0.008 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 3.714 ± 0.008 | 3.745 ± 0.046 | 4.520 ± 0.138 | 1.956 ± 0.013 | 2.073 ± 0.002 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 4.919 ± 0.036 | 4.906 ± 0.068 | 1.904 ± 0.094 | 1.957 ± 0.013 | 2.073 ± 0.002 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 3.976 ± 0.018 | 4.034 ± 0.200 | 4.070 ± 0.148 | 7.536 ± 0.046 | 3.832 ± 0.008 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 8.588 ± 0.807 | 8.931 ± 0.020 | 3.710 ± 0.009 | 6.327 ± 0.352 | 3.840 ± 0.084 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 3.993 ± 0.184 | 3.995 ± 0.151 | 3.905 ± 0.061 | 5.769 ± 0.058 | 3.799 ± 0.010 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 5.315 ± 0.307 | 5.572 ± 0.281 | 3.710 ± 0.008 | 4.647 ± 0.025 | 3.796 ± 0.005 | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 19.212 ± 0.044 | 19.171 ± 0.124 | 22.419 ± 0.082 | 9.650 ± 0.052 | 16.146 ± 2.371 | 19.336 ± 4.137 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 5.498 ± 0.268 | 6.358 ± 0.388 | 9.692 ± 1.190 | 9.487 ± 0.765 | 15.924 ± 1.001 | 4.594 ± 1.553 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 19.634 ± 0.052 | 16.739 ± 0.055 | 12.716 ± 0.018 | 16.994 ± 1.667 | 34.523 ± 1.503 | 31.206 ± 0.992 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 10.499 ± 2.652 | 8.603 ± 0.045 | 7.229 ± 0.052 | 17.325 ± 0.144 | 33.519 ± 2.751 | 9.242 ± 4.168 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 13.027 ± 0.073 | 13.643 ± 0.107 | 10.107 ± 0.490 | 20.534 ± 5.369 | 13.389 ± 0.129 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 12.397 ± 0.860 | 12.717 ± 1.315 | 20.529 ± 3.134 | 17.530 ± 3.626 | 13.389 ± 1.146 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 6.148 ± 0.053 | 6.184 ± 0.194 | 6.138 ± 0.107 | 6.321 ± 0.293 | 14.039 ± 0.099 | 16.729 ± 0.143 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 11.100 ± 1.296 | 15.277 ± 0.757 | 5.683 ± 0.064 | 5.685 ± 0.034 | 13.997 ± 0.135 | 5.547 ± 5.467 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 16.254 ± 0.046 | 16.181 ± 0.120 | 17.931 ± 0.012 | 6.999 ± 0.027 | 29.727 ± 2.772 | 26.426 ± 2.150 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 9.443 ± 0.248 | 9.574 ± 0.138 | 16.410 ± 0.025 | 7.013 ± 0.030 | 25.623 ± 1.600 | 7.152 ± 0.983 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 14.499 ± 0.098 | 14.590 ± 1.336 | 8.389 ± 0.029 | 11.075 ± 2.513 | 25.718 ± 2.596 | 22.816 ± 0.411 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 10.259 ± 1.286 | 9.136 ± 0.363 | 5.705 ± 0.036 | 10.350 ± 0.579 | 22.440 ± 1.392 | 7.501 ± 4.102 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 8.516 ± 0.141 | 8.526 ± 0.066 | 8.586 ± 0.190 | 11.514 ± 0.054 | 8.604 ± 0.036 | 9.279 ± 0.669 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 8.599 ± 0.140 | 8.945 ± 0.540 | 8.343 ± 0.166 | 11.460 ± 0.038 | 9.769 ± 0.196 | 12.970 ± 16.082 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 23.052 ± 0.021 | 23.316 ± 0.027 | 42.605 ± 0.013 | 5.175 ± 0.066 | 33.051 ± 0.154 | 32.934 ± 0.525 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 14.235 ± 1.639 | 14.035 ± 1.828 | 23.975 ± 1.565 | 5.835 ± 0.729 | 32.935 ± 3.874 | 9.651 ± 1.932 |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.299 ± 0.004 | 0.321 ± 0.016 | 0.317 ± 0.010 | 0.317 ± 0.034 | 0.423 ± 0.034 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.305 ± 0.015 | 0.337 ± 0.021 | 0.379 ± 0.012 | 0.315 ± 0.044 | 0.381 ± 0.032 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.304 ± 0.027 | 0.318 ± 0.029 | 0.302 ± 0.004 | 0.259 ± 0.007 | 0.388 ± 0.007 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.315 ± 0.028 | 0.353 ± 0.041 | 0.297 ± 0.006 | 0.257 ± 0.005 | 0.367 ± 0.026 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 0.486 ± 0.017 | unsupported | 0.504 ± 0.013 | 0.474 ± 0.030 | 0.565 ± 0.019 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.507 ± 0.023 | unsupported | 0.737 ± 0.010 | 0.448 ± 0.017 | 0.545 ± 0.028 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 3.256 ± 0.684 | 2.764 ± 0.004 | 0.182 ± 0.001 | 0.082 ± 0.007 | 0.313 ± 0.005 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 1.132 ± 0.344 | 0.830 ± 0.105 | 0.309 ± 0.001 | 0.077 ± 0.005 | 0.285 ± 0.094 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.487 ± 0.004 | 0.563 ± 0.019 | 0.534 ± 0.002 | 0.255 ± 0.009 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.497 ± 0.003 | 0.575 ± 0.033 | 0.535 ± 0.011 | 0.257 ± 0.008 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.779 ± 0.001 | 0.787 ± 0.001 | 0.335 ± 0.122 | 0.265 ± 0.015 | 0.928 ± 0.003 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.798 ± 0.002 | 0.818 ± 0.004 | 0.379 ± 0.012 | 0.261 ± 0.007 | 0.839 ± 0.005 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 2.530 ± 0.067 | 2.476 ± 0.008 | 0.283 ± 0.003 | 0.258 ± 0.010 | 0.303 ± 0.004 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 2.728 ± 0.295 | 2.595 ± 0.043 | 0.454 ± 0.002 | 0.250 ± 0.007 | 0.280 ± 0.083 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 0.783 ± 0.001 | 0.792 ± 0.003 | 0.559 ± 0.013 | 0.494 ± 0.053 | 0.686 ± 0.017 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.807 ± 0.010 | 0.821 ± 0.006 | 0.636 ± 0.018 | 0.483 ± 0.020 | 0.671 ± 0.024 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 4.839 ± 0.833 | 4.795 ± 0.116 | 22.616 ± 2.406 | 7.223 ± 0.047 | 32.172 ± 5.181 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 4.720 ± 0.488 | 4.618 ± 0.044 | 13.817 ± 0.104 | 8.295 ± 0.322 | 16.223 ± 13.306 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 5.040 ± 0.042 | 5.309 ± 0.093 | 5.007 ± 0.012 | 4.041 ± 0.032 | 17.674 ± 0.126 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 5.083 ± 0.041 | 5.316 ± 0.123 | 5.270 ± 0.485 | 4.018 ± 0.046 | 11.687 ± 0.548 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 4.715 ± 0.026 | 4.710 ± 0.054 | 6.563 ± 0.028 | 5.484 ± 0.058 | 5.234 ± 0.043 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 4.718 ± 0.063 | 4.724 ± 0.047 | 7.240 ± 0.941 | 5.718 ± 0.008 | 6.560 ± 0.035 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 4.879 ± 0.009 | 4.881 ± 0.010 | 6.546 ± 0.617 | 4.442 ± 0.020 | 5.542 ± 0.013 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 4.903 ± 0.017 | 4.918 ± 0.008 | 7.926 ± 0.028 | 4.681 ± 0.065 | 8.133 ± 0.031 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 12.344 ± 0.014 | 13.011 ± 0.250 | 18.480 ± 1.306 | 12.858 ± 0.172 | 13.777 ± 0.024 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 12.400 ± 0.058 | 13.168 ± 0.013 | 20.172 ± 1.682 | 12.913 ± 0.196 | 12.266 ± 0.064 | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 7.446 ± 0.028 | 6.942 ± 0.383 | 6.212 ± 0.018 | 4.907 ± 0.019 | 23.719 ± 0.101 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 7.052 ± 0.028 | 6.158 ± 0.057 | 5.789 ± 0.011 | 4.675 ± 0.140 | 12.135 ± 0.195 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | 5.121 ± 0.031 | 5.272 ± 0.058 | 3.518 ± 0.011 | 12.622 ± 0.410 | 6.367 ± 0.020 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | 5.244 ± 0.017 | 5.367 ± 0.076 | 3.384 ± 0.170 | 12.661 ± 0.182 | 5.373 ± 0.028 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 4.666 ± 0.030 | 4.959 ± 0.033 | 7.339 ± 0.013 | 4.938 ± 0.234 | 20.129 ± 4.240 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 4.671 ± 0.045 | 4.961 ± 0.053 | 6.623 ± 0.035 | 5.091 ± 0.032 | 12.493 ± 4.103 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.431 ± 0.014 | 0.437 ± 0.009 | 1.584 ± 0.001 | 1.167 ± 0.034 | 4.854 ± 0.006 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.616 ± 0.005 | 0.631 ± 0.003 | 1.518 ± 0.002 | 1.124 ± 0.026 | 6.418 ± 0.005 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 6.401 ± 0.013 | 6.419 ± 0.036 | 7.431 ± 0.018 | 6.392 ± 0.038 | 16.407 ± 0.122 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 6.539 ± 0.029 | 6.589 ± 0.052 | 8.094 ± 0.020 | 6.311 ± 0.040 | 15.269 ± 0.147 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 6.396 ± 0.040 | 6.405 ± 0.024 | 6.991 ± 0.454 | 6.389 ± 0.067 | 16.443 ± 0.144 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 6.503 ± 0.035 | 6.575 ± 0.039 | 7.514 ± 0.065 | 6.309 ± 0.035 | 16.140 ± 1.668 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 5.012 ± 0.041 | 5.237 ± 0.131 | 5.002 ± 0.049 | 4.067 ± 0.045 | 17.549 ± 0.666 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 5.042 ± 0.039 | 5.229 ± 0.039 | 5.274 ± 0.022 | 4.161 ± 0.027 | 11.090 ± 1.036 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | unsupported | unsupported | 19.400 ± 0.124 | 18.180 ± 0.056 | 57.496 ± 0.610 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | unsupported | unsupported | 19.958 ± 0.722 | 18.218 ± 0.084 | 48.936 ± 1.544 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 5.736 ± 0.227 | 5.751 ± 0.303 | 5.598 ± 0.018 | 14.737 ± 0.372 | 26.210 ± 0.063 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 6.640 ± 0.018 | 6.698 ± 0.014 | 5.654 ± 0.140 | 15.529 ± 0.351 | 13.707 ± 2.053 | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 24.971 ± 0.048 | unsupported | 9.478 ± 0.096 | - | 8.696 ± 0.063 | 11.582 ± 0.576 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 8.442 ± 0.166 | unsupported | 9.356 ± 0.033 | - | 8.432 ± 0.032 | 8.314 ± 0.041 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 10.454 ± 0.017 | unsupported | 10.088 ± 0.209 | - | 10.131 ± 0.111 | 12.071 ± 4.065 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 5.607 ± 0.186 | unsupported | 5.670 ± 0.025 | - | 13.836 ± 0.124 | 5.336 ± 0.155 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 5.808 ± 0.117 | unsupported | 9.941 ± 1.677 | - | 9.686 ± 0.072 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 5.409 ± 0.069 | unsupported | 5.658 ± 0.176 | - | 9.463 ± 1.663 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 29.184 ± 0.057 | unsupported | 9.547 ± 0.061 | - | 11.776 ± 0.282 | 15.027 ± 0.525 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 12.787 ± 1.641 | unsupported | 9.570 ± 0.184 | - | 9.425 ± 0.118 | 8.518 ± 0.183 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 4.925 ± 0.090 | unsupported | 7.778 ± 0.193 | - | 53.543 ± 0.066 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 8.222 ± 0.027 | unsupported | 9.721 ± 0.303 | - | 18.133 ± 4.932 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 5.787 ± 0.310 | unsupported | 9.903 ± 0.258 | - | 49.578 ± 0.189 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 8.584 ± 0.173 | unsupported | 9.842 ± 0.072 | - | 25.504 ± 7.208 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 24.982 ± 0.019 | unsupported | 9.491 ± 0.080 | - | 11.142 ± 0.037 | 13.637 ± 1.060 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 9.956 ± 1.103 | unsupported | 9.415 ± 0.061 | - | 9.416 ± 0.091 | 8.497 ± 0.204 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 16.612 ± 0.005 | unsupported | 8.116 ± 0.054 | - | 14.553 ± 0.182 | 14.146 ± 0.130 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 6.326 ± 0.665 | unsupported | 5.425 ± 0.064 | - | 13.797 ± 4.077 | 5.357 ± 0.182 |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 24.977 ± 0.030 | unsupported | 9.512 ± 0.061 | - | 9.634 ± 0.065 | 12.371 ± 1.111 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 8.443 ± 0.112 | unsupported | 9.391 ± 0.132 | - | 8.438 ± 0.017 | 8.314 ± 0.037 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 3.649 ± 0.004 | 3.657 ± 0.003 | 3.651 ± 0.001 | 3.679 ± 0.028 | 9.087 ± 0.105 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 3.665 ± 0.086 | 4.727 ± 1.081 | 3.656 ± 0.010 | 3.662 ± 0.004 | 9.068 ± 0.047 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 4.833 ± 0.017 | 4.899 ± 0.143 | 4.858 ± 0.019 | 4.364 ± 0.036 | 5.011 ± 0.112 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 4.537 ± 0.194 | 4.539 ± 0.156 | 4.339 ± 0.037 | 4.291 ± 0.184 | 4.951 ± 0.011 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 9.259 ± 2.668 | 9.325 ± 6.588 | 7.785 ± 0.557 | 7.591 ± 0.544 | 4.631 ± 0.575 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 11.876 ± 4.617 | 12.275 ± 6.436 | 7.338 ± 0.571 | 7.556 ± 0.532 | 5.127 ± 0.454 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 2.980 ± 0.049 | 3.013 ± 0.024 | 7.066 ± 0.200 | 4.377 ± 0.030 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 7.268 ± 0.372 | 7.000 ± 0.063 | 4.343 ± 0.019 | 4.381 ± 0.071 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 5.842 ± 0.087 | 5.815 ± 0.036 | 6.004 ± 0.059 | 5.033 ± 0.023 | 6.191 ± 0.414 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 7.114 ± 0.355 | 10.537 ± 0.265 | 5.300 ± 0.042 | 5.105 ± 0.030 | 6.931 ± 1.682 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 20.915 ± 0.062 | 21.245 ± 0.062 | 19.276 ± 0.242 | 7.402 ± 0.863 | 52.646 ± 1.037 | 20.858 ± 0.092 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 11.276 ± 0.866 | 11.886 ± 1.145 | 27.265 ± 16.289 | 6.923 ± 0.080 | 55.184 ± 3.565 | 14.396 ± 7.791 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 4.673 ± 0.048 | 4.506 ± 0.026 | 8.404 ± 0.114 | 2.913 ± 0.274 | 5.403 ± 0.031 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 9.506 ± 0.934 | 9.495 ± 0.702 | 3.507 ± 0.040 | 2.887 ± 0.048 | 5.635 ± 0.062 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 4.755 ± 0.165 | 4.798 ± 0.053 | 8.082 ± 0.091 | 2.885 ± 0.126 | 5.171 ± 0.043 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 9.549 ± 0.326 | 9.434 ± 1.696 | 3.132 ± 0.035 | 2.901 ± 0.171 | 5.353 ± 0.076 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 8.919 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 21.9x faster than the slowest successful cell (`pytorch-cpu`, 8.919 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 9.977 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 24.7x faster than the slowest successful cell (`pytorch-cpu`, 9.977 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=1, shape=`262144`): `jax-cpu` is 39.5x faster than the slowest successful cell (`tenferro-eager`, 3.256 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=1, shape=`262144`): `julia-base` is 10.4x faster than the slowest successful cell (`tenferro-eager`, 3.256 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=1, shape=`262144`): `pytorch-cpu` is 17.8x faster than the slowest successful cell (`tenferro-eager`, 3.256 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `jax-cpu` is 14.7x faster than the slowest successful cell (`tenferro-eager`, 1.132 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `jax-cpu` is 10.9x faster than the slowest successful cell (`tenferro-eager`, 2.728 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-eager` is 11.3x faster than the slowest successful cell (`julia-base`, 4.854 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 11.1x faster than the slowest successful cell (`julia-base`, 4.854 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-eager` is 10.4x faster than the slowest successful cell (`julia-base`, 6.418 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 10.2x faster than the slowest successful cell (`julia-base`, 6.418 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/output_reuse/dot_general_read_into` (f64, threads=1, shape=`1024x1024`): `tenferro-eager` is 10.9x faster than the slowest successful cell (`julia-base`, 53.543 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

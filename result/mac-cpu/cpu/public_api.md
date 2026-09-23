# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20260923_111502/run.yaml`
- Timestamp: `20260923_111502`

Latest run: `PUBLICATION_GATE_PROFILE=full ./scripts/run_cpu_public_api.sh 1 4`.

- Sampling: `full` profile, 15 measured runs, 3 warmups per row

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20260923_111502`.

- tenferro-rs commit: `f6279624e964c253ad21dec6521d82df6b00f7db`

## CPU Information

- Model: `Apple M5 Max`
- Vendor: `Apple`
- Logical CPUs: `18`
- Physical CPUs: `18`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Super: 6 physical / 6 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 6 CPUs/L2); Performance: 12 physical / 12 logical (L1i 128 KiB, L1d 64 KiB, L2 8 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5.1-arm64-arm-64bit-Mach-O`

## Thread Environments

### Threads: 1

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260923_111502/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260923_111502/run_t4.yaml`
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

- CSV: `data/results/mac-cpu/cpu/public_api/20260923_111502/cpu_public_api_t1_20260923_111502.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20260923_111502/cpu_public_api_t4_20260923_111502.csv`
- Source table: `data/results/mac-cpu/cpu/public_api/20260923_111502/cpu_public_api_20260923_111502.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 2.408 ± 0.053 | - | 2.438 ± 0.045 | 2.747 ± 0.043 | 0.953 ± 0.024 | 2.899 ± 0.368 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.398 ± 0.054 | - | 2.450 ± 0.046 | 2.456 ± 0.036 | 0.766 ± 0.013 | 1.198 ± 0.386 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 11.358 ± 0.181 | - | 11.052 ± 0.361 | 4.542 ± 0.048 | 4.562 ± 0.124 | 4.597 ± 0.251 | 4.423 ± 0.103 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 3.006 ± 0.179 | - | 2.951 ± 0.065 | 1.588 ± 0.052 | 1.584 ± 0.032 | 4.648 ± 0.290 | 6.553 ± 6.894 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 4.477 ± 0.020 | - | 4.556 ± 0.140 | 7.864 ± 0.154 | 23.058 ± 0.137 | 18.608 ± 0.052 | 18.198 ± 0.034 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 1.573 ± 0.034 | - | 1.785 ± 0.070 | 2.810 ± 0.102 | 7.708 ± 0.102 | 18.755 ± 0.295 | 7.028 ± 5.497 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 5.731 ± 0.977 | - | 5.846 ± 0.192 | 5.519 ± 0.164 | 36.543 ± 0.068 | 39.978 ± 0.373 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 5.914 ± 1.027 | - | 5.667 ± 0.222 | 5.614 ± 0.198 | 9.425 ± 0.203 | 10.686 ± 0.145 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 5.819 ± 0.234 | - | unsupported | 5.691 ± 0.185 | 36.696 ± 0.070 | 39.958 ± 0.889 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 5.716 ± 0.189 | - | unsupported | 5.736 ± 0.121 | 9.819 ± 0.214 | 10.692 ± 0.254 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 5.855 ± 0.049 | - | 5.804 ± 0.022 | 5.873 ± 0.066 | 9.952 ± 0.037 | 5.136 ± 0.037 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 5.861 ± 0.038 | - | 5.930 ± 0.030 | 5.870 ± 0.053 | 9.932 ± 0.015 | 4.901 ± 0.064 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 19.935 ± 0.664 | - | 20.367 ± 0.910 | 20.360 ± 4.482 | 23.985 ± 0.050 | 24.685 ± 3.210 | 24.527 ± 0.101 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 5.476 ± 0.758 | - | 5.281 ± 0.029 | 5.969 ± 0.105 | 6.529 ± 0.189 | 25.409 ± 0.703 | 6.263 ± 1.107 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 23.639 ± 0.131 | - | 23.629 ± 0.067 | 25.678 ± 0.098 | 49.597 ± 0.049 | 35.292 ± 0.098 | 34.380 ± 0.107 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 6.357 ± 0.197 | - | 6.527 ± 0.261 | 6.920 ± 0.207 | 13.701 ± 0.287 | 35.343 ± 0.173 | 9.645 ± 2.116 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 3.426 ± 0.302 | - | 3.375 ± 0.076 | 4.222 ± 0.033 | 3.612 ± 0.112 | 3.728 ± 0.193 | 3.489 ± 0.207 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 1.406 ± 0.285 | - | 1.332 ± 0.043 | 1.436 ± 0.025 | 1.367 ± 0.099 | 3.764 ± 0.209 | 3.562 ± 8.481 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 4.872 ± 0.157 | - | 3.987 ± 0.074 | 3.642 ± 0.071 | 1.951 ± 0.081 | 5.559 ± 0.104 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 2.272 ± 0.061 | - | 1.410 ± 0.017 | 0.932 ± 0.015 | 0.747 ± 0.044 | 5.894 ± 0.021 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 4.775 ± 0.047 | - | 4.781 ± 0.055 | 5.014 ± 0.017 | 3.643 ± 0.028 | 5.988 ± 0.081 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 4.700 ± 0.045 | - | 4.828 ± 0.042 | 4.837 ± 0.028 | 3.599 ± 0.021 | 2.745 ± 0.091 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 3.018 ± 0.030 | - | 3.222 ± 0.059 | 3.137 ± 0.052 | 1.432 ± 0.026 | 3.653 ± 0.047 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.301 ± 0.094 | - | 3.418 ± 0.112 | 3.291 ± 0.082 | 1.312 ± 0.025 | 1.513 ± 0.049 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 3.834 ± 0.035 | - | 3.902 ± 0.054 | 3.809 ± 0.039 | 2.761 ± 0.023 | 4.430 ± 0.019 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 3.883 ± 0.021 | - | 3.986 ± 0.033 | 3.811 ± 0.021 | 2.775 ± 0.037 | 2.890 ± 0.076 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 5.476 ± 0.184 | - | 5.823 ± 0.194 | 5.583 ± 0.162 | 36.560 ± 0.104 | 39.686 ± 0.144 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 5.543 ± 0.187 | - | 5.773 ± 0.190 | 5.553 ± 0.202 | 9.517 ± 0.232 | 10.496 ± 0.205 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 5.100 ± 0.974 | - | unsupported | 4.506 ± 0.084 | 32.792 ± 0.196 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 5.017 ± 0.955 | - | unsupported | 4.477 ± 0.037 | 8.795 ± 0.185 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 4.282 ± 0.126 | - | 4.267 ± 0.101 | 4.298 ± 0.094 | 8.656 ± 1.293 | 4.377 ± 0.090 | 8.035 ± 0.396 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 1.562 ± 0.041 | - | 1.622 ± 0.116 | 1.570 ± 0.049 | 2.491 ± 0.045 | 4.438 ± 0.118 | 3.038 ± 2.603 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 6.762 ± 0.329 | - | 6.531 ± 0.128 | 6.588 ± 0.386 | 11.553 ± 0.523 | 6.570 ± 0.384 | 8.637 ± 0.833 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 2.513 ± 0.065 | - | 2.570 ± 0.151 | 2.528 ± 0.080 | 3.521 ± 0.123 | 6.792 ± 0.397 | 5.462 ± 7.075 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 16.582 ± 0.063 | - | 42.559 ± 1.670 | 17.850 ± 0.081 | 17.852 ± 0.047 | 20.841 ± 0.060 | 21.133 ± 0.223 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 5.473 ± 0.525 | - | 17.874 ± 0.956 | 4.979 ± 0.363 | 4.732 ± 0.213 | 20.883 ± 0.074 | 5.721 ± 0.372 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 2.788 ± 0.040 | - | 2.858 ± 0.036 | 2.210 ± 0.068 | 3.717 ± 0.147 | 2.242 ± 0.131 | 3.152 ± 0.113 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 1.543 ± 0.060 | - | 1.591 ± 0.074 | 0.913 ± 0.077 | 1.827 ± 0.036 | 2.207 ± 0.126 | 1.223 ± 0.691 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 5.117 ± 0.027 | - | 5.120 ± 0.085 | 8.554 ± 0.021 | 8.669 ± 0.031 | 5.810 ± 0.080 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 2.055 ± 0.016 | - | 2.066 ± 0.019 | 2.915 ± 0.071 | 2.667 ± 0.025 | 5.830 ± 0.085 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 13.956 ± 1.319 | - | 14.459 ± 0.421 | 16.930 ± 0.026 | 12.295 ± 0.760 | 20.593 ± 0.233 | 20.738 ± 0.153 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 4.089 ± 0.027 | - | 4.107 ± 0.033 | 4.494 ± 0.149 | 3.284 ± 0.011 | 20.416 ± 0.118 | 5.230 ± 0.227 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 6.240 ± 0.150 | - | 6.389 ± 0.230 | 6.481 ± 0.299 | 11.447 ± 0.186 | 6.488 ± 0.208 | 8.627 ± 0.235 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 2.517 ± 0.158 | - | 2.576 ± 0.097 | 2.504 ± 0.088 | 3.497 ± 0.153 | 6.576 ± 0.072 | 3.406 ± 2.405 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 15.042 ± 0.375 | - | 14.990 ± 0.519 | 12.174 ± 0.046 | 10.313 ± 0.048 | 16.668 ± 0.374 | 16.719 ± 1.663 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 4.104 ± 0.033 | - | 4.109 ± 0.171 | 4.353 ± 0.065 | 2.648 ± 0.072 | 16.660 ± 0.111 | 5.672 ± 0.355 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 7.887 ± 0.452 | - | 7.865 ± 0.590 | 14.650 ± 0.118 | 10.935 ± 0.031 | 12.686 ± 0.064 | 12.640 ± 0.101 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 2.201 ± 0.015 | - | 2.214 ± 0.009 | 3.875 ± 0.490 | 2.848 ± 0.142 | 12.755 ± 0.094 | 3.342 ± 0.949 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 15.251 ± 0.182 | - | 15.179 ± 0.105 | 20.708 ± 0.043 | 14.100 ± 0.034 | 22.504 ± 0.278 | 22.359 ± 0.158 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 4.081 ± 0.118 | - | 4.076 ± 0.089 | 5.343 ± 0.367 | 3.833 ± 0.174 | 22.601 ± 0.146 | 6.220 ± 0.450 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 10.136 ± 0.023 | - | 10.167 ± 0.017 | 10.928 ± 0.030 | 9.911 ± 0.019 | 12.196 ± 0.075 | 12.187 ± 0.107 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 2.708 ± 0.179 | - | 2.721 ± 0.023 | 2.900 ± 0.099 | 2.671 ± 0.069 | 12.234 ± 0.272 | 3.100 ± 0.198 |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 6.503 ± 0.023 | - | 6.503 ± 0.037 | 6.456 ± 0.153 | 11.419 ± 0.292 | 6.353 ± 0.183 | 8.456 ± 0.357 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 2.441 ± 0.135 | - | 2.461 ± 0.175 | 2.639 ± 0.122 | 3.492 ± 0.059 | 6.509 ± 0.103 | 3.232 ± 2.641 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 6.527 ± 0.032 | - | 6.545 ± 0.041 | 6.491 ± 0.105 | 11.272 ± 0.209 | 6.465 ± 0.201 | 8.411 ± 0.213 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 2.505 ± 0.028 | - | 2.549 ± 0.109 | 2.491 ± 0.056 | 3.494 ± 0.114 | 6.556 ± 0.127 | 3.227 ± 2.357 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 6.399 ± 0.109 | - | 6.434 ± 0.148 | 6.622 ± 0.172 | 11.311 ± 0.262 | 6.526 ± 0.155 | 8.513 ± 0.526 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 2.499 ± 0.102 | - | 2.472 ± 0.054 | 2.488 ± 0.031 | 3.478 ± 0.121 | 6.629 ± 0.064 | 3.651 ± 8.960 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 4.258 ± 0.108 | - | 4.282 ± 0.119 | 4.292 ± 0.096 | 8.765 ± 1.170 | 4.327 ± 0.092 | 7.797 ± 0.165 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 1.567 ± 0.025 | - | 1.592 ± 0.021 | 1.558 ± 0.121 | 2.498 ± 0.048 | 4.447 ± 0.107 | 3.573 ± 9.420 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 17.595 ± 0.061 | - | 17.616 ± 0.030 | 37.917 ± 0.080 | 17.584 ± 0.041 | 32.945 ± 1.191 | 31.663 ± 0.266 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 4.924 ± 0.600 | - | 4.858 ± 0.150 | 11.217 ± 1.810 | 4.936 ± 0.131 | 32.136 ± 1.734 | 8.315 ± 0.620 |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 5.506 ± 0.170 | - | 5.473 ± 0.099 | 9.339 ± 0.107 | 1.026 ± 0.055 | 0.364 ± 0.004 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 5.616 ± 0.123 | - | 5.795 ± 0.120 | 4.550 ± 0.034 | 0.447 ± 0.026 | 0.365 ± 0.014 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 3.172 ± 0.116 | - | 3.160 ± 0.145 | 3.860 ± 0.053 | 1.675 ± 0.110 | 1.897 ± 0.034 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 3.185 ± 0.127 | - | 3.228 ± 0.075 | 1.043 ± 0.008 | 0.706 ± 0.036 | 1.899 ± 0.029 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 3.243 ± 0.148 | - | 3.352 ± 0.117 | 3.190 ± 0.047 | 17.185 ± 0.269 | 3.060 ± 0.068 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 5.939 ± 0.015 | - | 6.083 ± 0.084 | 0.968 ± 0.057 | 5.387 ± 0.030 | 3.053 ± 0.051 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 3.344 ± 0.290 | - | 3.281 ± 0.144 | 2.915 ± 0.054 | 14.037 ± 0.290 | 3.036 ± 0.143 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 4.570 ± 0.336 | - | 4.412 ± 0.025 | 0.996 ± 0.131 | 4.523 ± 0.057 | 3.075 ± 0.109 | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 15.444 ± 1.488 | - | 15.982 ± 1.291 | 19.251 ± 0.073 | 14.717 ± 0.195 | 14.547 ± 2.977 | 14.162 ± 0.779 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 4.384 ± 0.486 | - | 4.083 ± 0.185 | 6.355 ± 0.093 | 4.348 ± 0.036 | 13.374 ± 1.708 | 4.164 ± 1.317 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 11.352 ± 0.079 | - | 11.353 ± 0.031 | 11.323 ± 0.038 | 22.685 ± 0.191 | 22.669 ± 0.189 | 22.934 ± 0.449 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 3.084 ± 0.022 | - | 3.085 ± 0.017 | 3.153 ± 0.098 | 6.059 ± 0.036 | 23.375 ± 0.846 | 6.332 ± 2.317 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 7.774 ± 0.026 | - | 7.787 ± 0.054 | 9.155 ± 0.346 | 14.056 ± 0.098 | 7.778 ± 0.028 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 3.308 ± 0.076 | - | 3.325 ± 0.055 | 3.570 ± 0.071 | 5.970 ± 0.132 | 7.849 ± 0.151 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 5.082 ± 0.051 | - | 5.062 ± 0.045 | 5.028 ± 0.018 | 9.862 ± 0.028 | 4.454 ± 0.083 | 11.728 ± 0.142 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 3.245 ± 0.053 | - | 3.105 ± 0.036 | 1.720 ± 0.023 | 3.292 ± 0.036 | 4.504 ± 0.056 | 3.454 ± 2.531 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 13.103 ± 0.318 | - | 13.229 ± 0.612 | 14.156 ± 0.043 | 11.301 ± 0.136 | 17.386 ± 0.498 | 17.515 ± 0.346 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 3.749 ± 0.060 | - | 3.803 ± 0.070 | 3.818 ± 0.444 | 3.029 ± 0.079 | 17.282 ± 0.270 | 4.489 ± 0.257 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 7.403 ± 0.148 | - | 7.405 ± 0.194 | 7.410 ± 0.075 | 7.758 ± 1.166 | 15.308 ± 0.303 | 15.429 ± 0.177 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 2.033 ± 0.031 | - | 2.030 ± 0.040 | 2.041 ± 0.117 | 2.471 ± 0.056 | 15.517 ± 0.177 | 4.161 ± 2.622 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 6.360 ± 0.160 | - | 6.485 ± 0.150 | 6.676 ± 0.136 | 11.405 ± 0.173 | 6.487 ± 0.202 | 8.370 ± 0.200 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 2.423 ± 0.105 | - | 2.452 ± 0.101 | 2.506 ± 0.068 | 3.631 ± 0.195 | 6.563 ± 0.061 | 3.298 ± 2.368 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 21.176 ± 0.072 | - | 21.214 ± 0.052 | 35.406 ± 0.051 | 8.394 ± 0.026 | 23.376 ± 0.327 | 23.370 ± 0.114 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 5.384 ± 0.038 | - | 5.384 ± 0.030 | 9.360 ± 0.025 | 2.256 ± 0.011 | 23.395 ± 0.044 | 5.951 ± 0.206 |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.261 ± 0.015 | - | 0.530 ± 0.027 | 0.208 ± 0.009 | 0.216 ± 0.004 | 0.281 ± 0.010 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.276 ± 0.007 | - | 0.412 ± 0.021 | 0.120 ± 0.002 | 0.222 ± 0.004 | 0.286 ± 0.008 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.238 ± 0.014 | - | 0.775 ± 0.008 | 0.269 ± 0.019 | 0.261 ± 0.003 | 0.273 ± 0.008 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.254 ± 0.004 | - | 0.575 ± 0.027 | 0.096 ± 0.015 | 0.120 ± 0.007 | 0.282 ± 0.005 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 0.361 ± 0.005 | - | unsupported | 0.409 ± 0.012 | 0.361 ± 0.007 | 0.397 ± 0.070 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.389 ± 0.009 | - | unsupported | 0.158 ± 0.004 | 0.309 ± 0.014 | 0.402 ± 0.017 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.384 ± 0.131 | - | 0.466 ± 0.118 | 0.159 ± 0.006 | 0.181 ± 0.005 | 0.288 ± 0.030 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.273 ± 0.053 | - | 0.309 ± 0.106 | 0.176 ± 0.021 | 0.072 ± 0.010 | 0.292 ± 0.107 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.330 ± 0.005 | - | 0.595 ± 0.026 | 0.388 ± 0.008 | 0.286 ± 0.007 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.343 ± 0.010 | - | 0.478 ± 0.036 | 0.180 ± 0.001 | 0.126 ± 0.007 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.493 ± 0.021 | - | 0.752 ± 0.232 | 0.311 ± 0.064 | 0.266 ± 0.004 | 0.712 ± 0.058 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.478 ± 0.008 | - | 0.810 ± 0.222 | 0.103 ± 0.015 | 0.114 ± 0.012 | 0.610 ± 0.185 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.482 ± 0.050 | - | 0.592 ± 0.014 | 0.239 ± 0.010 | 0.226 ± 0.009 | 0.255 ± 0.041 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.495 ± 0.032 | - | 0.549 ± 0.024 | 0.346 ± 0.018 | 0.221 ± 0.009 | 0.270 ± 0.103 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 0.599 ± 0.004 | - | 1.190 ± 0.211 | 0.427 ± 0.071 | 0.418 ± 0.026 | 0.705 ± 0.054 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.668 ± 0.279 | - | 0.953 ± 0.062 | 0.152 ± 0.031 | 0.296 ± 0.019 | 0.719 ± 0.076 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 4.197 ± 0.635 | - | 4.584 ± 0.137 | 14.871 ± 0.176 | 8.981 ± 0.560 | 25.582 ± 3.527 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 4.308 ± 0.774 | - | 4.420 ± 0.127 | 7.997 ± 0.136 | 6.013 ± 0.132 | 10.366 ± 12.301 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 4.109 ± 0.055 | - | 4.346 ± 0.048 | 4.139 ± 0.029 | 4.072 ± 0.060 | 13.553 ± 0.460 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 4.130 ± 0.070 | - | 4.166 ± 0.051 | 3.981 ± 0.053 | 3.780 ± 0.058 | 5.259 ± 0.336 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 4.435 ± 0.051 | - | 4.397 ± 0.032 | 4.454 ± 0.082 | 5.399 ± 0.112 | 3.967 ± 0.050 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 4.440 ± 0.061 | - | 4.436 ± 0.044 | 4.444 ± 0.092 | 5.391 ± 0.119 | 4.173 ± 0.044 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 4.588 ± 0.019 | - | 4.601 ± 0.019 | 4.600 ± 0.052 | 4.259 ± 0.042 | 4.407 ± 0.031 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 4.527 ± 0.113 | - | 4.679 ± 0.038 | 4.660 ± 0.103 | 4.395 ± 0.037 | 4.535 ± 0.067 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 14.977 ± 0.063 | - | 14.174 ± 0.049 | 14.309 ± 0.044 | 12.545 ± 0.062 | 10.793 ± 0.081 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 14.200 ± 0.069 | - | 14.721 ± 0.068 | 14.292 ± 0.045 | 12.451 ± 0.071 | 6.490 ± 0.155 | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 6.041 ± 0.037 | - | 6.405 ± 0.071 | 5.818 ± 0.067 | 5.528 ± 0.038 | 19.146 ± 0.165 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 6.031 ± 0.020 | - | 6.377 ± 0.036 | 5.279 ± 0.020 | 5.100 ± 0.030 | 6.122 ± 0.161 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 5.656 ± 0.062 | 5.534 ± 0.030 | 3.232 ± 0.038 | 13.561 ± 0.049 | 5.163 ± 0.032 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 5.675 ± 0.051 | 5.610 ± 0.057 | 3.172 ± 0.034 | 13.134 ± 0.087 | 3.101 ± 0.057 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 4.181 ± 0.043 | - | 4.193 ± 0.034 | 6.013 ± 0.232 | 7.012 ± 0.470 | 15.833 ± 3.296 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 4.005 ± 0.040 | - | 4.171 ± 0.040 | 5.330 ± 0.118 | 5.024 ± 0.138 | 6.923 ± 2.407 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.360 ± 0.003 | - | 0.365 ± 0.014 | 1.250 ± 0.044 | 2.776 ± 0.203 | 3.693 ± 0.064 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.549 ± 0.017 | - | 0.601 ± 0.026 | 1.223 ± 0.045 | 0.823 ± 0.038 | 3.928 ± 0.024 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 6.578 ± 0.026 | - | 6.500 ± 0.082 | 6.563 ± 0.242 | 7.312 ± 0.086 | 12.867 ± 0.297 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 6.590 ± 0.113 | - | 6.756 ± 0.080 | 6.484 ± 0.040 | 6.570 ± 0.101 | 9.468 ± 0.163 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 6.577 ± 0.022 | - | 6.470 ± 0.051 | 6.416 ± 0.031 | 7.315 ± 0.075 | 12.776 ± 0.158 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 6.512 ± 0.034 | - | 6.632 ± 0.017 | 6.476 ± 0.039 | 6.537 ± 0.066 | 9.446 ± 0.147 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 4.113 ± 0.049 | - | 4.337 ± 0.034 | 4.128 ± 0.048 | 4.072 ± 0.047 | 13.494 ± 0.151 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 4.131 ± 0.090 | - | 4.150 ± 0.050 | 3.969 ± 0.043 | 3.790 ± 0.023 | 5.020 ± 0.665 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 18.357 ± 0.069 | 18.331 ± 0.060 | 18.428 ± 0.052 | 18.856 ± 0.064 | 44.044 ± 0.227 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 18.845 ± 0.101 | 18.828 ± 0.126 | 18.785 ± 0.078 | 18.708 ± 0.131 | 29.247 ± 0.415 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 5.394 ± 0.070 | - | 8.002 ± 0.523 | 3.950 ± 0.076 | 25.085 ± 0.148 | 21.091 ± 0.072 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 5.338 ± 0.111 | - | 6.835 ± 0.443 | 3.776 ± 0.054 | 12.450 ± 0.199 | 7.430 ± 0.147 | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 23.142 ± 1.706 | - | unsupported | 6.498 ± 0.157 | - | 6.525 ± 0.271 | 8.249 ± 0.320 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 5.987 ± 0.144 | - | unsupported | 2.503 ± 0.035 | - | 6.599 ± 0.241 | 3.087 ± 0.237 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 8.061 ± 0.581 | - | unsupported | 4.519 ± 0.049 | - | 4.454 ± 0.098 | 4.344 ± 0.051 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 1.876 ± 0.054 | - | unsupported | 1.564 ± 0.037 | - | 4.371 ± 0.063 | 1.742 ± 0.160 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 3.796 ± 0.050 | - | unsupported | 4.357 ± 0.131 | - | 3.814 ± 0.074 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 1.552 ± 0.074 | - | unsupported | 1.539 ± 0.010 | - | 3.844 ± 0.052 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 21.838 ± 1.331 | - | unsupported | 6.416 ± 0.119 | - | 6.364 ± 0.155 | 8.487 ± 0.398 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 4.955 ± 0.218 | - | unsupported | 2.488 ± 0.035 | - | 6.498 ± 0.021 | 3.062 ± 0.229 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 4.465 ± 0.094 | - | unsupported | 4.501 ± 0.770 | - | 35.886 ± 0.099 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 4.467 ± 0.099 | - | unsupported | 4.438 ± 0.033 | - | 9.352 ± 0.327 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 4.977 ± 0.051 | - | unsupported | 5.015 ± 0.058 | - | 35.818 ± 0.070 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 5.022 ± 0.047 | - | unsupported | 4.962 ± 0.088 | - | 9.553 ± 0.133 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 23.148 ± 0.994 | - | unsupported | 6.468 ± 0.170 | - | 6.341 ± 0.074 | 8.353 ± 0.410 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 6.093 ± 0.058 | - | unsupported | 2.494 ± 0.032 | - | 6.493 ± 0.033 | 3.226 ± 0.096 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 10.168 ± 0.220 | - | unsupported | 4.343 ± 0.043 | - | 4.305 ± 0.131 | 7.801 ± 0.186 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 2.743 ± 0.091 | - | unsupported | 1.537 ± 0.019 | - | 4.367 ± 0.022 | 2.753 ± 0.520 |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 23.103 ± 0.826 | - | unsupported | 6.500 ± 0.120 | - | 6.411 ± 0.153 | 8.480 ± 0.533 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 6.078 ± 0.133 | - | unsupported | 2.497 ± 0.021 | - | 6.479 ± 0.035 | 3.186 ± 0.217 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 2.001 ± 0.071 | - | 1.990 ± 0.094 | 2.024 ± 0.079 | 1.874 ± 0.028 | 8.316 ± 0.193 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 1.856 ± 0.024 | - | 1.121 ± 0.014 | 0.926 ± 0.163 | 1.109 ± 0.023 | 8.561 ± 0.467 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 3.533 ± 0.047 | - | 7.776 ± 0.095 | 3.558 ± 0.031 | 4.606 ± 0.273 | 3.608 ± 0.099 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 1.287 ± 0.075 | - | 2.907 ± 0.135 | 1.334 ± 0.041 | 1.387 ± 0.067 | 3.630 ± 0.100 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 3.875 ± 0.095 | - | 3.848 ± 0.102 | 3.985 ± 0.044 | 6.148 ± 0.106 | 2.035 ± 0.120 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 2.206 ± 0.070 | - | 2.216 ± 0.072 | 2.172 ± 0.087 | 2.268 ± 0.150 | 2.090 ± 0.138 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 2.083 ± 0.056 | - | 6.295 ± 0.083 | 4.386 ± 0.066 | 3.502 ± 0.062 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 2.146 ± 0.107 | - | 3.755 ± 0.165 | 1.813 ± 0.040 | 1.291 ± 0.019 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 3.886 ± 0.073 | - | 4.154 ± 0.056 | 4.329 ± 0.117 | 3.651 ± 0.157 | 3.975 ± 0.194 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 3.920 ± 0.082 | - | 1.573 ± 0.049 | 1.547 ± 0.027 | 1.695 ± 0.081 | 4.025 ± 0.204 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 28.158 ± 0.079 | - | 23.142 ± 0.079 | 18.632 ± 0.602 | 13.078 ± 0.525 | 51.099 ± 2.832 | 30.805 ± 0.146 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 11.459 ± 1.271 | - | 7.938 ± 0.790 | 19.202 ± 0.177 | 5.418 ± 0.748 | 50.900 ± 2.757 | 21.235 ± 6.609 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 3.191 ± 0.087 | - | 5.435 ± 0.223 | 3.524 ± 0.092 | 2.212 ± 0.030 | 4.214 ± 0.079 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 1.433 ± 0.027 | - | 2.508 ± 0.385 | 1.371 ± 0.030 | 0.903 ± 0.104 | 4.231 ± 0.118 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 3.254 ± 0.085 | - | 5.377 ± 0.045 | 3.275 ± 0.072 | 2.197 ± 0.023 | 3.946 ± 0.120 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 1.306 ± 0.037 | - | 2.463 ± 0.662 | 1.198 ± 0.036 | 0.814 ± 0.015 | 4.020 ± 0.201 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.000 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.000 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.000 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.000 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.000 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.000 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.000 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.000 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---:|---|---:|---:|---:|
| broadcast_in_dim_view | 1 | PyTorch Python (ms) | 16 | 0.005500 | 343.75 |
| broadcast_in_dim_view | 1 | tenferro-rs direct API (ms) | 16 | 0.001375 | 85.94 |
| broadcast_in_dim_view | 4 | PyTorch Python (ms) | 16 | 0.007833 | 489.56 |
| broadcast_in_dim_view | 4 | tenferro-rs direct API (ms) | 16 | 0.001417 | 88.56 |
| reshape_view | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| reshape_view | 1 | PyTorch Python (ms) | 16 | 0.005167 | 322.94 |
| reshape_view | 1 | tenferro-rs direct API (ms) | 16 | 0.002708 | 169.25 |
| reshape_view | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| reshape_view | 4 | PyTorch Python (ms) | 16 | 0.007583 | 473.94 |
| reshape_view | 4 | tenferro-rs direct API (ms) | 16 | 0.003166 | 197.88 |
| slice_view | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| slice_view | 1 | PyTorch Python (ms) | 16 | 0.004751 | 296.94 |
| slice_view | 1 | tenferro-rs direct API (ms) | 16 | 0.002250 | 140.62 |
| slice_view | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| slice_view | 4 | PyTorch Python (ms) | 16 | 0.006501 | 406.31 |
| slice_view | 4 | tenferro-rs direct API (ms) | 16 | 0.002334 | 145.88 |
| transpose_view | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| transpose_view | 1 | PyTorch Python (ms) | 16 | 0.004834 | 302.12 |
| transpose_view | 1 | tenferro-rs direct API (ms) | 16 | 0.002167 | 135.44 |
| transpose_view | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| transpose_view | 4 | PyTorch Python (ms) | 16 | 0.006460 | 403.75 |
| transpose_view | 4 | tenferro-rs direct API (ms) | 16 | 0.001917 | 119.81 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/div` (c64, threads=4, shape=`8388608`): `tenferro-direct` is 11.9x faster than the slowest successful cell (`julia-base`, 18.755 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/div` (c64, threads=4, shape=`8388608`): `tenferro-trace` is 10.5x faster than the slowest successful cell (`julia-base`, 18.755 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 25.7x faster than the slowest successful cell (`pytorch-cpu`, 9.339 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 13.0x faster than the slowest successful cell (`tenferro-trace`, 5.795 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 15.9x faster than the slowest successful cell (`tenferro-trace`, 5.795 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/tanh` (f64, threads=4, shape=`8388608`): `jax-cpu` is 10.4x faster than the slowest successful cell (`julia-base`, 23.395 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`julia-base`, 3.693 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`julia-base`, 3.693 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 107.7x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 158.0x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 99.0x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 135.3x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 100.7x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 134.7x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

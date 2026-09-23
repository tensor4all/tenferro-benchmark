# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20260923_210012/run.yaml`
- Timestamp: `20260923_210012`

Latest run: `PUBLICATION_GATE_PROFILE=full ./scripts/run_cpu_public_api.sh 1 4`.

- Sampling: `full` profile, 15 measured runs, 3 warmups per row

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20260923_210012`.

- tenferro-rs commit: `8a1839febeb3c868a502e26397ca10761bbc568d`

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

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260923_210012/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260923_210012/run_t4.yaml`
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

- CSV: `data/results/mac-cpu/cpu/public_api/20260923_210012/cpu_public_api_t1_20260923_210012.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20260923_210012/cpu_public_api_t4_20260923_210012.csv`
- Source table: `data/results/mac-cpu/cpu/public_api/20260923_210012/cpu_public_api_20260923_210012.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 2.422 ± 0.039 | - | 2.458 ± 0.094 | 2.851 ± 0.110 | 0.952 ± 0.024 | 2.920 ± 0.354 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.498 ± 0.104 | - | 2.511 ± 0.086 | 2.493 ± 0.037 | 0.753 ± 0.024 | 1.203 ± 0.395 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 4.472 ± 0.129 | - | 4.434 ± 0.208 | 4.525 ± 0.034 | 4.467 ± 0.065 | 4.435 ± 0.036 | 4.475 ± 0.151 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 1.658 ± 0.073 | - | 1.616 ± 0.069 | 1.564 ± 0.043 | 1.619 ± 0.094 | 4.458 ± 0.070 | 5.398 ± 6.306 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 5.672 ± 0.078 | - | 5.735 ± 0.121 | 8.328 ± 0.318 | 23.349 ± 0.913 | 19.351 ± 0.296 | 18.203 ± 0.154 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 1.576 ± 0.061 | - | 1.801 ± 0.210 | 2.796 ± 0.061 | 7.702 ± 0.073 | 18.569 ± 0.034 | 8.572 ± 8.821 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 5.701 ± 1.293 | - | 5.658 ± 0.213 | 5.428 ± 0.194 | 37.021 ± 0.145 | 40.291 ± 0.486 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 6.032 ± 1.307 | - | 6.114 ± 0.278 | 5.517 ± 0.181 | 9.590 ± 0.253 | 10.982 ± 0.309 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 5.870 ± 0.194 | - | unsupported | 5.703 ± 0.216 | 36.640 ± 0.209 | 40.452 ± 0.657 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 5.831 ± 0.420 | - | unsupported | 5.658 ± 0.167 | 9.892 ± 0.111 | 10.923 ± 0.528 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 5.875 ± 0.043 | - | 5.891 ± 0.061 | 5.887 ± 0.066 | 9.927 ± 0.011 | 5.163 ± 0.060 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 5.853 ± 0.075 | - | 5.976 ± 0.083 | 5.881 ± 0.037 | 9.968 ± 0.086 | 4.860 ± 0.024 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 20.508 ± 0.560 | - | 20.471 ± 0.602 | 20.121 ± 0.319 | 24.472 ± 0.648 | 26.073 ± 0.708 | 24.998 ± 1.395 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 5.596 ± 0.407 | - | 5.668 ± 0.039 | 5.972 ± 0.094 | 6.354 ± 0.074 | 25.139 ± 0.941 | 6.454 ± 0.868 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 23.642 ± 0.167 | - | 23.632 ± 0.154 | 25.705 ± 0.345 | 50.111 ± 3.730 | 35.556 ± 0.471 | 34.486 ± 0.286 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 6.651 ± 0.220 | - | 6.595 ± 0.301 | 7.164 ± 0.296 | 13.738 ± 0.227 | 35.216 ± 0.147 | 9.673 ± 2.411 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 4.057 ± 0.214 | - | 4.000 ± 0.066 | 4.238 ± 0.057 | 3.583 ± 0.138 | 3.685 ± 0.146 | 3.749 ± 0.135 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 1.374 ± 0.063 | - | 1.378 ± 0.050 | 1.452 ± 0.020 | 1.304 ± 0.082 | 3.677 ± 0.655 | 3.588 ± 9.165 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 4.793 ± 0.041 | - | 3.970 ± 0.071 | 3.700 ± 0.060 | 1.954 ± 0.015 | 5.710 ± 0.130 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 1.987 ± 0.105 | - | 1.248 ± 0.197 | 0.936 ± 0.014 | 0.730 ± 0.048 | 5.885 ± 0.011 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 4.780 ± 0.013 | - | 4.757 ± 0.026 | 5.020 ± 0.036 | 3.640 ± 0.026 | 6.151 ± 0.156 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 4.705 ± 0.034 | - | 4.838 ± 0.041 | 4.859 ± 0.033 | 3.606 ± 0.033 | 2.793 ± 0.090 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 3.001 ± 0.030 | - | 3.062 ± 0.032 | 3.171 ± 0.036 | 1.434 ± 0.007 | 3.638 ± 0.118 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.232 ± 0.081 | - | 3.307 ± 0.156 | 3.301 ± 0.083 | 1.327 ± 0.039 | 1.500 ± 0.024 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 3.852 ± 0.051 | - | 3.971 ± 0.026 | 3.840 ± 0.063 | 2.755 ± 0.018 | 4.566 ± 0.177 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 3.950 ± 0.075 | - | 3.983 ± 0.106 | 3.838 ± 0.085 | 2.759 ± 0.034 | 2.871 ± 0.157 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 5.564 ± 0.202 | - | 5.804 ± 0.191 | 5.387 ± 0.132 | 36.779 ± 0.526 | 40.028 ± 0.194 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 5.639 ± 0.220 | - | 5.906 ± 0.498 | 5.667 ± 0.147 | 9.907 ± 0.240 | 10.539 ± 0.306 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 4.972 ± 0.937 | - | unsupported | 5.064 ± 0.198 | 32.420 ± 0.124 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 5.341 ± 0.834 | - | unsupported | 4.942 ± 0.560 | 9.091 ± 0.097 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 4.387 ± 0.058 | - | 4.380 ± 0.075 | 4.340 ± 0.035 | 8.534 ± 0.753 | 4.334 ± 0.092 | 7.851 ± 0.101 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 1.563 ± 0.036 | - | 1.598 ± 0.082 | 1.656 ± 0.081 | 2.486 ± 0.024 | 4.449 ± 0.135 | 2.852 ± 2.904 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 6.725 ± 0.248 | - | 6.429 ± 0.090 | 6.555 ± 0.385 | 11.712 ± 0.433 | 6.562 ± 0.402 | 8.504 ± 0.531 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 4.546 ± 0.152 | - | 4.255 ± 0.825 | 2.548 ± 0.164 | 3.758 ± 0.317 | 6.639 ± 0.128 | 7.647 ± 9.189 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 16.658 ± 0.072 | - | 42.073 ± 1.059 | 17.869 ± 0.110 | 17.962 ± 0.399 | 21.057 ± 1.792 | 21.530 ± 1.190 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 5.538 ± 0.152 | - | 17.371 ± 0.705 | 5.122 ± 0.130 | 4.744 ± 0.053 | 20.930 ± 0.147 | 5.972 ± 0.404 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 2.719 ± 0.030 | - | 2.746 ± 0.049 | 2.214 ± 0.058 | 3.796 ± 0.012 | 2.302 ± 0.165 | 3.111 ± 0.207 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 1.533 ± 0.120 | - | 1.554 ± 0.018 | 0.923 ± 0.070 | 1.912 ± 0.027 | 2.271 ± 0.135 | 1.254 ± 1.491 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 5.117 ± 0.036 | - | 5.135 ± 0.049 | 8.509 ± 0.102 | 8.687 ± 0.048 | 5.830 ± 0.063 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 2.057 ± 0.032 | - | 2.059 ± 0.023 | 2.989 ± 0.121 | 2.699 ± 0.050 | 5.799 ± 0.078 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 16.116 ± 0.427 | - | 16.093 ± 1.042 | 16.904 ± 0.071 | 12.474 ± 0.137 | 20.372 ± 0.203 | 20.513 ± 0.289 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 4.275 ± 0.196 | - | 4.258 ± 0.112 | 4.590 ± 0.331 | 3.527 ± 0.152 | 20.298 ± 0.327 | 5.310 ± 0.425 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 6.506 ± 0.079 | - | 6.627 ± 0.250 | 6.561 ± 0.131 | 13.051 ± 1.290 | 6.389 ± 0.126 | 8.444 ± 0.627 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 2.537 ± 0.214 | - | 2.520 ± 0.327 | 2.691 ± 0.143 | 3.472 ± 0.143 | 6.580 ± 0.086 | 3.638 ± 3.229 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 15.938 ± 0.348 | - | 15.861 ± 0.489 | 12.192 ± 0.057 | 10.300 ± 0.034 | 17.271 ± 0.821 | 17.181 ± 0.489 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 4.014 ± 0.046 | - | 4.148 ± 0.169 | 4.461 ± 0.086 | 2.835 ± 0.318 | 17.266 ± 0.428 | 5.861 ± 0.371 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 8.485 ± 0.302 | - | 8.620 ± 0.315 | 14.606 ± 0.045 | 10.921 ± 0.050 | 12.732 ± 0.268 | 12.762 ± 0.339 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 2.211 ± 0.055 | - | 2.214 ± 0.032 | 4.036 ± 0.405 | 2.923 ± 0.176 | 12.638 ± 0.061 | 3.233 ± 0.348 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 15.367 ± 1.625 | - | 15.409 ± 0.192 | 20.777 ± 0.169 | 14.043 ± 0.045 | 22.423 ± 0.182 | 22.618 ± 0.435 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 4.200 ± 0.150 | - | 4.315 ± 0.084 | 7.026 ± 0.134 | 4.172 ± 0.251 | 22.442 ± 0.130 | 6.500 ± 0.625 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 10.305 ± 0.441 | - | 10.298 ± 0.178 | 10.902 ± 0.029 | 9.912 ± 0.037 | 12.203 ± 0.272 | 12.192 ± 0.098 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 2.743 ± 0.076 | - | 2.815 ± 0.095 | 2.944 ± 0.165 | 2.829 ± 0.132 | 12.195 ± 0.036 | 3.137 ± 0.655 |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 6.914 ± 0.228 | - | 6.888 ± 0.258 | 6.481 ± 0.073 | 11.436 ± 0.143 | 6.461 ± 0.170 | 8.775 ± 0.319 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 2.511 ± 0.061 | - | 2.421 ± 0.030 | 2.949 ± 0.245 | 3.453 ± 0.099 | 6.562 ± 0.058 | 3.249 ± 2.405 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 7.003 ± 0.387 | - | 6.610 ± 0.086 | 6.423 ± 0.064 | 11.503 ± 0.112 | 6.545 ± 0.092 | 8.903 ± 0.261 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 2.494 ± 0.010 | - | 2.522 ± 0.042 | 3.015 ± 0.591 | 3.398 ± 0.124 | 6.573 ± 0.090 | 3.095 ± 2.659 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 6.709 ± 0.182 | - | 6.593 ± 0.103 | 6.582 ± 0.198 | 11.720 ± 0.219 | 6.481 ± 0.103 | 8.723 ± 0.962 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 2.508 ± 0.087 | - | 2.491 ± 0.133 | 2.676 ± 0.140 | 3.429 ± 0.178 | 6.606 ± 0.084 | 3.537 ± 9.086 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 4.303 ± 0.031 | - | 4.298 ± 0.070 | 4.327 ± 0.027 | 9.283 ± 1.215 | 4.379 ± 0.097 | 7.809 ± 0.046 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 1.559 ± 0.022 | - | 1.579 ± 0.015 | 1.671 ± 0.174 | 2.522 ± 0.076 | 4.450 ± 0.095 | 3.152 ± 2.774 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 18.044 ± 0.075 | - | 17.780 ± 0.116 | 38.118 ± 0.260 | 17.545 ± 0.054 | 31.856 ± 2.761 | 31.379 ± 0.581 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 5.015 ± 0.547 | - | 4.977 ± 0.102 | 11.319 ± 1.706 | 4.985 ± 0.259 | 31.913 ± 1.508 | 8.502 ± 0.609 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 4.233 ± 0.184 | - | 4.100 ± 0.045 | 7.538 ± 0.230 | 10.131 ± 0.142 | 2.922 ± 0.047 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 1.721 ± 0.213 | - | 1.685 ± 0.162 | 1.996 ± 0.007 | 2.990 ± 0.013 | 2.986 ± 0.101 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 0.882 ± 0.042 | - | 0.723 ± 0.059 | 9.755 ± 1.479 | 0.954 ± 0.078 | 0.455 ± 0.180 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.293 ± 0.021 | - | 0.268 ± 0.034 | 4.837 ± 0.488 | 0.343 ± 0.042 | 0.382 ± 0.020 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 0.679 ± 0.078 | - | 0.677 ± 0.037 | 0.974 ± 0.038 | 0.388 ± 0.013 | 0.486 ± 0.005 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 0.245 ± 0.045 | - | 0.260 ± 0.017 | 0.296 ± 0.020 | 0.238 ± 0.010 | 0.498 ± 0.013 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 4.149 ± 0.030 | - | 4.180 ± 0.071 | 7.469 ± 0.142 | 10.145 ± 0.192 | 2.886 ± 0.020 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 1.513 ± 0.101 | - | 1.591 ± 0.262 | 1.996 ± 0.012 | 3.017 ± 0.048 | 2.986 ± 0.113 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 0.544 ± 0.019 | - | 0.535 ± 0.012 | 8.845 ± 0.061 | 1.033 ± 0.111 | 0.364 ± 0.009 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.161 ± 0.018 | - | 0.168 ± 0.011 | 4.634 ± 0.041 | 0.356 ± 0.039 | 0.360 ± 0.008 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 2.645 ± 0.108 | - | 2.569 ± 0.092 | 3.891 ± 0.225 | 1.560 ± 0.021 | 1.922 ± 0.030 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 1.067 ± 0.095 | - | 0.970 ± 0.052 | 1.056 ± 0.032 | 0.881 ± 0.059 | 1.910 ± 0.075 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 3.305 ± 0.272 | - | 3.363 ± 0.052 | 3.159 ± 0.019 | 18.148 ± 0.307 | 3.022 ± 0.022 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 1.560 ± 0.346 | - | 1.452 ± 0.271 | 1.110 ± 0.147 | 5.392 ± 0.044 | 3.134 ± 0.152 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.370 ± 0.004 | - | 0.382 ± 0.003 | 1.513 ± 0.108 | 1.005 ± 0.064 | 0.404 ± 0.018 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.128 ± 0.007 | - | 0.138 ± 0.011 | 0.676 ± 0.019 | 0.313 ± 0.086 | 0.377 ± 0.014 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 0.453 ± 0.018 | - | 0.474 ± 0.016 | 0.341 ± 0.002 | 0.386 ± 0.006 | 0.625 ± 0.342 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.179 ± 0.006 | - | 0.206 ± 0.012 | 0.123 ± 0.015 | 0.246 ± 0.016 | 0.498 ± 0.010 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 3.295 ± 0.207 | - | 3.146 ± 0.072 | 2.920 ± 0.022 | 13.890 ± 0.266 | 2.911 ± 0.043 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 1.722 ± 0.156 | - | 1.832 ± 0.135 | 1.065 ± 0.040 | 4.478 ± 0.024 | 3.058 ± 0.052 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.377 ± 0.012 | - | 0.365 ± 0.007 | 1.276 ± 0.037 | 0.829 ± 0.030 | 0.385 ± 0.027 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.167 ± 0.011 | - | 0.144 ± 0.006 | 0.621 ± 0.009 | 0.342 ± 0.037 | 0.382 ± 0.013 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 0.473 ± 0.022 | - | 0.474 ± 0.030 | 0.305 ± 0.004 | 0.375 ± 0.006 | 0.494 ± 0.028 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.168 ± 0.008 | - | 0.202 ± 0.015 | 0.120 ± 0.015 | 0.249 ± 0.040 | 0.485 ± 0.009 | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 16.070 ± 0.671 | - | 15.765 ± 0.739 | 19.250 ± 0.091 | 14.658 ± 0.358 | 13.568 ± 3.444 | 13.580 ± 1.011 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 4.404 ± 0.830 | - | 4.251 ± 1.962 | 6.432 ± 0.082 | 6.506 ± 1.816 | 13.710 ± 0.975 | 3.853 ± 1.607 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 12.206 ± 0.130 | - | 12.138 ± 0.316 | 11.301 ± 0.052 | 22.960 ± 0.397 | 24.291 ± 0.351 | 23.263 ± 0.828 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 3.077 ± 0.031 | - | 3.068 ± 0.013 | 3.137 ± 0.101 | 6.154 ± 0.039 | 23.506 ± 0.220 | 6.571 ± 2.512 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 8.388 ± 1.905 | - | 8.384 ± 0.060 | 9.094 ± 0.159 | 14.040 ± 0.056 | 7.842 ± 0.179 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 3.719 ± 0.115 | - | 3.749 ± 0.107 | 3.686 ± 0.086 | 5.928 ± 1.504 | 7.826 ± 0.068 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 5.220 ± 0.115 | - | 5.252 ± 0.281 | 5.028 ± 0.073 | 9.917 ± 0.122 | 4.424 ± 0.082 | 11.769 ± 0.147 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 3.204 ± 0.047 | - | 3.200 ± 0.072 | 1.796 ± 0.074 | 3.294 ± 0.032 | 4.459 ± 0.125 | 3.261 ± 2.418 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 14.688 ± 0.119 | - | 14.747 ± 1.510 | 14.137 ± 0.030 | 11.180 ± 0.234 | 17.361 ± 0.210 | 17.620 ± 0.276 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 3.969 ± 0.122 | - | 4.082 ± 0.207 | 5.036 ± 0.029 | 3.337 ± 0.394 | 17.291 ± 0.337 | 4.401 ± 0.435 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 8.015 ± 0.268 | - | 7.948 ± 0.071 | 7.565 ± 0.128 | 8.487 ± 1.129 | 15.838 ± 0.356 | 15.902 ± 0.180 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 2.018 ± 0.027 | - | 2.037 ± 0.047 | 2.036 ± 0.049 | 2.483 ± 0.067 | 15.638 ± 0.065 | 4.129 ± 2.698 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 6.470 ± 0.081 | - | 6.502 ± 0.044 | 6.493 ± 0.060 | 11.602 ± 0.094 | 6.467 ± 0.150 | 8.622 ± 0.506 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 2.506 ± 0.086 | - | 2.524 ± 0.099 | 2.603 ± 0.355 | 3.533 ± 0.260 | 6.600 ± 0.119 | 3.260 ± 2.618 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 21.278 ± 0.083 | - | 21.207 ± 0.077 | 35.378 ± 0.087 | 8.376 ± 0.055 | 23.354 ± 0.206 | 23.374 ± 0.232 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 5.495 ± 0.150 | - | 5.519 ± 0.106 | 9.505 ± 0.325 | 2.470 ± 0.474 | 23.838 ± 0.122 | 6.508 ± 0.641 |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.282 ± 0.022 | - | 0.560 ± 0.020 | 0.215 ± 0.017 | 0.223 ± 0.006 | 0.290 ± 0.015 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.108 ± 0.017 | - | 0.293 ± 0.021 | 0.122 ± 0.047 | 0.232 ± 0.008 | 0.300 ± 0.075 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.259 ± 0.051 | - | 0.864 ± 0.083 | 0.266 ± 0.014 | 0.264 ± 0.004 | 0.280 ± 0.014 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.150 ± 0.014 | - | 0.474 ± 0.010 | 0.083 ± 0.002 | 0.123 ± 0.038 | 0.295 ± 0.008 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 0.402 ± 0.096 | - | unsupported | 0.407 ± 0.020 | 0.372 ± 0.007 | 0.411 ± 0.179 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.177 ± 0.009 | - | unsupported | 0.156 ± 0.001 | 0.302 ± 0.009 | 0.405 ± 0.004 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.338 ± 0.174 | - | 0.336 ± 0.152 | 0.161 ± 0.006 | 0.183 ± 0.009 | 0.282 ± 0.019 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.278 ± 0.018 | - | 0.311 ± 0.034 | 0.176 ± 0.001 | 0.080 ± 0.011 | 0.286 ± 0.148 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.359 ± 0.024 | - | 0.671 ± 0.106 | 0.399 ± 0.006 | 0.284 ± 0.001 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.172 ± 0.016 | - | 0.365 ± 0.018 | 0.170 ± 0.002 | 0.149 ± 0.034 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.279 ± 0.011 | - | 0.587 ± 0.123 | 0.276 ± 0.062 | 0.269 ± 0.010 | 0.729 ± 0.227 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.098 ± 0.004 | - | 0.282 ± 0.016 | 0.113 ± 0.011 | 0.114 ± 0.015 | 0.618 ± 0.009 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.355 ± 0.040 | - | 0.474 ± 0.033 | 0.230 ± 0.005 | 0.230 ± 0.008 | 0.229 ± 0.037 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.489 ± 0.011 | - | 0.551 ± 0.027 | 0.344 ± 0.007 | 0.216 ± 0.006 | 0.220 ± 0.053 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 0.513 ± 0.021 | - | 1.091 ± 0.145 | 0.419 ± 0.003 | 0.417 ± 0.003 | 0.676 ± 0.046 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.220 ± 0.015 | - | 0.581 ± 0.031 | 0.141 ± 0.040 | 0.287 ± 0.029 | 0.666 ± 0.019 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.792 ± 0.046 | - | unsupported | 0.473 ± 0.008 | 0.563 ± 0.016 | 1.087 ± 0.054 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.032 ± 0.001 | - | unsupported | 0.034 ± 0.001 | 0.070 ± 0.003 | 0.041 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.067 ± 0.001 | - | unsupported | 0.060 ± 0.003 | 0.095 ± 0.003 | 0.071 ± 0.012 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.274 ± 0.140 | - | unsupported | 0.159 ± 0.004 | 0.198 ± 0.004 | 0.246 ± 0.007 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.695 ± 0.059 | - | unsupported | 0.200 ± 0.014 | 0.301 ± 0.010 | 1.009 ± 0.147 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.046 ± 0.000 | - | unsupported | 0.108 ± 0.038 | 0.067 ± 0.002 | 0.043 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.086 ± 0.002 | - | unsupported | 0.086 ± 0.032 | 0.098 ± 0.002 | 0.086 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.316 ± 0.021 | - | unsupported | 0.139 ± 0.009 | 0.106 ± 0.005 | 0.247 ± 0.007 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.195 ± 0.004 | - | unsupported | 0.119 ± 0.005 | 0.268 ± 0.014 | 0.182 ± 0.006 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.049 ± 0.001 | - | unsupported | 0.023 ± 0.000 | 0.040 ± 0.001 | 0.026 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.058 ± 0.000 | - | unsupported | 0.027 ± 0.000 | 0.050 ± 0.002 | 0.044 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.086 ± 0.001 | - | unsupported | 0.050 ± 0.003 | 0.090 ± 0.008 | 0.092 ± 0.005 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.178 ± 0.004 | - | unsupported | 0.125 ± 0.018 | 0.228 ± 0.008 | 0.182 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.038 ± 0.001 | - | unsupported | 0.024 ± 0.000 | 0.035 ± 0.003 | 0.026 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.045 ± 0.004 | - | unsupported | 0.028 ± 0.000 | 0.046 ± 0.002 | 0.047 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.069 ± 0.000 | - | unsupported | 0.048 ± 0.002 | 0.083 ± 0.002 | 0.081 ± 0.005 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.062 ± 0.003 | - | 0.117 ± 0.002 | 0.048 ± 0.000 | 0.108 ± 0.002 | 0.113 ± 0.003 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.025 ± 0.000 | - | 0.033 ± 0.002 | 0.012 ± 0.000 | 0.013 ± 0.000 | 0.021 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.026 ± 0.003 | - | 0.035 ± 0.000 | 0.015 ± 0.000 | 0.017 ± 0.000 | 0.033 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.031 ± 0.000 | - | 0.049 ± 0.000 | 0.020 ± 0.000 | 0.033 ± 0.001 | 0.058 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.057 ± 0.000 | - | 0.108 ± 0.019 | 0.048 ± 0.001 | 0.090 ± 0.008 | 0.112 ± 0.005 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.022 ± 0.000 | - | 0.048 ± 0.009 | 0.017 ± 0.002 | 0.013 ± 0.001 | 0.022 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.023 ± 0.000 | - | 0.054 ± 0.006 | 0.014 ± 0.001 | 0.016 ± 0.000 | 0.035 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.028 ± 0.000 | - | 0.069 ± 0.008 | 0.019 ± 0.001 | 0.045 ± 0.007 | 0.061 ± 0.002 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 4.264 ± 0.373 | - | 5.080 ± 0.483 | 14.805 ± 0.113 | 8.825 ± 0.201 | 26.077 ± 2.957 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 4.474 ± 0.822 | - | 4.503 ± 0.226 | 10.602 ± 5.197 | 6.103 ± 0.181 | 10.802 ± 12.418 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 4.098 ± 0.041 | - | 4.119 ± 0.033 | 4.311 ± 0.105 | 4.076 ± 0.019 | 13.641 ± 0.418 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 4.380 ± 0.200 | - | 4.011 ± 0.190 | 3.966 ± 0.052 | 4.050 ± 0.649 | 5.050 ± 0.274 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 4.462 ± 0.205 | - | 4.472 ± 0.432 | 4.471 ± 0.042 | 5.298 ± 0.064 | 4.013 ± 0.052 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 4.460 ± 0.070 | - | 4.466 ± 0.048 | 4.439 ± 0.051 | 5.444 ± 0.097 | 4.133 ± 0.057 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 4.616 ± 0.067 | - | 4.650 ± 0.127 | 4.614 ± 0.033 | 4.271 ± 0.062 | 4.431 ± 0.042 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 4.623 ± 0.071 | - | 4.656 ± 0.041 | 4.661 ± 0.034 | 4.239 ± 0.036 | 4.506 ± 0.040 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 14.927 ± 0.537 | - | 15.083 ± 0.194 | 14.115 ± 0.738 | 12.581 ± 0.099 | 10.840 ± 0.124 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 14.005 ± 0.340 | - | 14.774 ± 0.362 | 14.918 ± 0.126 | 12.705 ± 0.615 | 6.516 ± 0.061 | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 6.073 ± 0.073 | - | 5.194 ± 0.218 | 6.439 ± 0.722 | 5.479 ± 0.031 | 19.096 ± 0.153 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 6.202 ± 0.159 | - | 5.248 ± 0.124 | 5.472 ± 0.164 | 5.110 ± 0.136 | 6.448 ± 0.262 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 5.635 ± 0.037 | 5.549 ± 0.049 | 3.241 ± 0.035 | 13.559 ± 0.035 | 5.208 ± 0.033 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 5.695 ± 0.058 | 5.585 ± 0.029 | 3.165 ± 0.023 | 13.264 ± 0.184 | 3.082 ± 0.044 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 4.154 ± 0.359 | - | 4.250 ± 0.235 | 7.101 ± 0.461 | 6.650 ± 0.051 | 15.499 ± 3.476 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 4.180 ± 0.304 | - | 4.346 ± 0.326 | 5.171 ± 0.065 | 5.507 ± 0.544 | 7.396 ± 2.693 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.362 ± 0.003 | - | 0.366 ± 0.020 | 1.270 ± 0.043 | 2.858 ± 0.227 | 3.744 ± 0.111 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.156 ± 0.033 | - | 0.176 ± 0.036 | 1.273 ± 0.013 | 0.816 ± 0.045 | 3.920 ± 0.006 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 6.601 ± 0.046 | - | 6.504 ± 0.106 | 6.680 ± 0.650 | 7.319 ± 0.057 | 12.944 ± 0.201 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 6.674 ± 0.104 | - | 6.740 ± 0.105 | 6.683 ± 0.129 | 6.525 ± 0.103 | 9.479 ± 0.070 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 6.560 ± 0.025 | - | 6.445 ± 0.038 | 6.814 ± 0.402 | 7.294 ± 0.076 | 12.896 ± 0.066 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 6.580 ± 0.112 | - | 6.733 ± 0.063 | 6.486 ± 0.055 | 6.569 ± 0.122 | 9.411 ± 0.136 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 4.111 ± 0.043 | - | 4.109 ± 0.036 | 4.476 ± 0.409 | 4.093 ± 0.016 | 13.485 ± 0.142 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 4.272 ± 0.190 | - | 4.083 ± 0.192 | 3.972 ± 0.252 | 3.900 ± 0.186 | 5.127 ± 0.329 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 18.337 ± 0.193 | 18.630 ± 0.179 | 18.578 ± 0.256 | 18.739 ± 0.041 | 44.442 ± 0.388 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 18.859 ± 0.261 | 18.920 ± 0.273 | 18.848 ± 0.337 | 18.683 ± 0.178 | 29.169 ± 0.155 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 5.768 ± 0.438 | - | 7.797 ± 0.291 | 3.923 ± 0.052 | 24.834 ± 0.138 | 20.770 ± 0.135 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 5.606 ± 0.336 | - | 7.142 ± 0.326 | 3.780 ± 0.023 | 12.516 ± 0.478 | 7.188 ± 0.223 | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 6.409 ± 0.105 | - | unsupported | 6.485 ± 0.069 | - | 6.518 ± 0.211 | 8.303 ± 0.618 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 2.610 ± 0.178 | - | unsupported | 2.525 ± 0.145 | - | 6.471 ± 0.042 | 3.155 ± 0.245 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 4.341 ± 0.039 | - | unsupported | 4.498 ± 0.020 | - | 4.334 ± 0.032 | 4.336 ± 0.041 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 1.637 ± 0.051 | - | unsupported | 1.563 ± 0.046 | - | 4.332 ± 0.036 | 1.752 ± 0.184 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 3.878 ± 0.092 | - | unsupported | 4.337 ± 0.088 | - | 3.787 ± 0.081 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 1.556 ± 0.019 | - | unsupported | 1.528 ± 0.009 | - | 3.821 ± 0.044 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 6.483 ± 0.119 | - | unsupported | 6.428 ± 0.076 | - | 6.421 ± 0.130 | 8.707 ± 0.544 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 2.562 ± 0.288 | - | unsupported | 2.467 ± 0.023 | - | 6.458 ± 0.064 | 3.222 ± 0.120 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 4.444 ± 0.050 | - | unsupported | 4.620 ± 0.157 | - | 35.864 ± 0.208 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 4.512 ± 0.069 | - | unsupported | 4.630 ± 0.081 | - | 9.746 ± 0.277 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 4.671 ± 0.017 | - | unsupported | 4.876 ± 0.135 | - | 35.790 ± 0.104 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 4.917 ± 0.100 | - | unsupported | 5.107 ± 0.271 | - | 9.534 ± 0.252 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 6.510 ± 0.115 | - | unsupported | 6.443 ± 0.049 | - | 6.413 ± 0.096 | 8.333 ± 0.531 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 2.505 ± 0.143 | - | unsupported | 2.474 ± 0.016 | - | 6.455 ± 0.031 | 3.202 ± 0.129 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 4.321 ± 0.036 | - | unsupported | 4.333 ± 0.039 | - | 4.330 ± 0.040 | 7.947 ± 0.284 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 1.577 ± 0.056 | - | unsupported | 1.526 ± 0.009 | - | 4.377 ± 0.053 | 2.695 ± 0.763 |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 6.394 ± 0.080 | - | unsupported | 6.513 ± 0.120 | - | 6.435 ± 0.110 | 8.341 ± 0.463 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 2.487 ± 0.106 | - | unsupported | 2.466 ± 0.022 | - | 6.462 ± 0.046 | 3.140 ± 0.069 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 3.304 ± 0.075 | - | 2.021 ± 0.015 | 2.051 ± 0.086 | 1.929 ± 0.088 | 8.571 ± 0.105 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 1.900 ± 0.026 | - | 1.974 ± 0.054 | 1.078 ± 0.008 | 0.931 ± 0.067 | 8.519 ± 0.115 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 3.501 ± 0.021 | - | 7.863 ± 0.128 | 3.541 ± 0.019 | 4.858 ± 0.808 | 3.677 ± 0.428 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 1.355 ± 0.071 | - | 2.898 ± 0.069 | 1.317 ± 0.059 | 1.386 ± 0.078 | 3.687 ± 0.076 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 3.775 ± 0.053 | - | 3.768 ± 0.091 | 3.969 ± 0.042 | 5.980 ± 0.025 | 2.044 ± 0.108 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 1.937 ± 0.342 | - | 2.260 ± 0.060 | 2.133 ± 0.015 | 2.118 ± 0.160 | 2.166 ± 0.221 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 2.138 ± 0.109 | - | 6.409 ± 0.091 | 5.506 ± 1.432 | 3.479 ± 0.029 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 2.187 ± 0.080 | - | 3.851 ± 0.080 | 1.744 ± 0.015 | 1.347 ± 0.046 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 3.785 ± 0.033 | - | 4.260 ± 0.142 | 4.306 ± 0.131 | 3.531 ± 0.071 | 3.875 ± 0.031 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 3.845 ± 0.090 | - | 1.607 ± 0.111 | 1.543 ± 0.079 | 1.783 ± 0.117 | 4.017 ± 0.131 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 27.749 ± 0.472 | - | 22.984 ± 0.656 | 18.186 ± 0.474 | 13.189 ± 0.188 | 53.414 ± 2.983 | 30.703 ± 0.615 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 12.162 ± 0.805 | - | 7.345 ± 0.329 | 18.511 ± 0.535 | 5.338 ± 0.282 | 50.858 ± 0.843 | 17.027 ± 7.444 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 3.131 ± 0.128 | - | 5.342 ± 0.057 | 3.563 ± 0.103 | 2.205 ± 0.018 | 4.208 ± 0.116 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 1.411 ± 0.102 | - | 2.473 ± 0.550 | 1.369 ± 0.027 | 0.866 ± 0.051 | 4.258 ± 0.158 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 2.939 ± 0.084 | - | 5.366 ± 0.086 | 3.187 ± 0.085 | 2.200 ± 0.009 | 3.888 ± 0.202 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 1.358 ± 0.059 | - | 2.807 ± 0.411 | 1.238 ± 0.020 | 0.905 ± 0.061 | 4.026 ± 0.127 | - |
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
| broadcast_in_dim_view | 1 | PyTorch Python (ms) | 16 | 0.005625 | 351.56 |
| broadcast_in_dim_view | 1 | tenferro-rs direct API (ms) | 16 | 0.001375 | 85.94 |
| broadcast_in_dim_view | 4 | PyTorch Python (ms) | 16 | 0.005791 | 361.94 |
| broadcast_in_dim_view | 4 | tenferro-rs direct API (ms) | 16 | 0.001375 | 85.94 |
| reshape_view | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| reshape_view | 1 | PyTorch Python (ms) | 16 | 0.005291 | 330.69 |
| reshape_view | 1 | tenferro-rs direct API (ms) | 16 | 0.002625 | 164.06 |
| reshape_view | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| reshape_view | 4 | PyTorch Python (ms) | 16 | 0.005792 | 362.00 |
| reshape_view | 4 | tenferro-rs direct API (ms) | 16 | 0.002667 | 166.69 |
| slice_view | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| slice_view | 1 | PyTorch Python (ms) | 16 | 0.004708 | 294.25 |
| slice_view | 1 | tenferro-rs direct API (ms) | 16 | 0.002416 | 151.00 |
| slice_view | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| slice_view | 4 | PyTorch Python (ms) | 16 | 0.004917 | 307.31 |
| slice_view | 4 | tenferro-rs direct API (ms) | 16 | 0.002458 | 153.62 |
| transpose_view | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| transpose_view | 1 | PyTorch Python (ms) | 16 | 0.004666 | 291.62 |
| transpose_view | 1 | tenferro-rs direct API (ms) | 16 | 0.001708 | 106.75 |
| transpose_view | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| transpose_view | 4 | PyTorch Python (ms) | 16 | 0.004916 | 307.25 |
| transpose_view | 4 | tenferro-rs direct API (ms) | 16 | 0.001875 | 117.19 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/div` (c64, threads=4, shape=`8388608`): `tenferro-direct` is 11.8x faster than the slowest successful cell (`julia-base`, 18.569 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/div` (c64, threads=4, shape=`8388608`): `tenferro-trace` is 10.3x faster than the slowest successful cell (`julia-base`, 18.569 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 9.755 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 21.5x faster than the slowest successful cell (`pytorch-cpu`, 9.755 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 9.755 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 9.755 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 14.1x faster than the slowest successful cell (`pytorch-cpu`, 4.837 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 4.837 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 16.5x faster than the slowest successful cell (`pytorch-cpu`, 4.837 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 18.1x faster than the slowest successful cell (`pytorch-cpu`, 4.837 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 24.3x faster than the slowest successful cell (`pytorch-cpu`, 8.845 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 16.3x faster than the slowest successful cell (`pytorch-cpu`, 8.845 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 16.5x faster than the slowest successful cell (`pytorch-cpu`, 8.845 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 4.634 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 4.634 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 28.7x faster than the slowest successful cell (`pytorch-cpu`, 4.634 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 27.6x faster than the slowest successful cell (`pytorch-cpu`, 4.634 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 10.4x faster than the slowest successful cell (`julia-base`, 3.744 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 10.2x faster than the slowest successful cell (`julia-base`, 3.744 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 25.2x faster than the slowest successful cell (`julia-base`, 3.920 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 22.3x faster than the slowest successful cell (`julia-base`, 3.920 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 110.3x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 120.7x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 98.0x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 102.3x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 97.3x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 102.3x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

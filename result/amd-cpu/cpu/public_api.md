# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`
- Run metadata: `data/results/amd-cpu/cpu/public_api/20261022_185903/run.yaml`
- Timestamp: `20261022_185903`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Sampling: `quick` profile, 7 measured runs, 3 warmups per row

This file is generated from sequential CPU public API runs under `data/results/amd-cpu/cpu/public_api/20261022_185903`.

- tenferro-rs commit: `2007d101d23b5baf7deba4afb8c7b92da347dbba`

## CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`

## Thread Environments

### Threads: 1

- Run metadata: `data/results/amd-cpu/cpu/public_api/20261022_185903/run_t1.yaml`
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

- Run metadata: `data/results/amd-cpu/cpu/public_api/20261022_185903/run_t4.yaml`
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

- CSV: `data/results/amd-cpu/cpu/public_api/20261022_185903/cpu_public_api_t1_20261022_185903.csv`
- CSV: `data/results/amd-cpu/cpu/public_api/20261022_185903/cpu_public_api_t4_20261022_185903.csv`
- Source table: `data/results/amd-cpu/cpu/public_api/20261022_185903/cpu_public_api_20261022_185903.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 4.249 ± 0.038 | - | 4.344 ± 0.037 | 4.504 ± 0.005 | 3.878 ± 0.015 | 3.633 ± 1.205 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.449 ± 0.036 | - | 2.581 ± 0.042 | 2.293 ± 0.012 | 3.421 ± 0.142 | 4.115 ± 0.461 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 25.663 ± 0.367 | - | 25.512 ± 0.239 | 24.718 ± 0.588 | 24.799 ± 0.181 | 28.037 ± 0.199 | 30.602 ± 0.291 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 17.397 ± 0.976 | - | 16.475 ± 0.649 | 16.969 ± 1.027 | 20.526 ± 0.363 | 29.425 ± 1.118 | 18.506 ± 0.610 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 57.108 ± 0.187 | - | 57.175 ± 0.155 | 42.856 ± 0.171 | 39.065 ± 0.255 | 70.190 ± 0.481 | 76.230 ± 0.237 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 26.657 ± 0.644 | - | 27.310 ± 0.534 | 17.098 ± 0.159 | 20.394 ± 0.130 | 72.650 ± 0.132 | 37.465 ± 0.354 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 43.267 ± 0.624 | - | 43.204 ± 0.253 | 42.137 ± 0.097 | 67.606 ± 0.262 | 45.159 ± 0.316 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 29.452 ± 0.074 | - | 29.520 ± 0.130 | 29.094 ± 0.308 | 29.646 ± 0.582 | 22.731 ± 0.236 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 44.385 ± 0.657 | - | unsupported | 42.476 ± 0.648 | 66.687 ± 0.360 | 45.020 ± 3.054 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 30.162 ± 0.096 | - | unsupported | 29.410 ± 0.243 | 30.102 ± 0.236 | 23.198 ± 1.820 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 10.903 ± 0.052 | - | 10.949 ± 0.023 | 11.546 ± 0.009 | 11.892 ± 0.008 | 11.876 ± 0.041 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 14.501 ± 0.067 | - | 14.552 ± 0.061 | 14.006 ± 0.085 | 18.220 ± 1.863 | 17.544 ± 0.080 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 70.277 ± 0.145 | - | 70.303 ± 0.365 | 85.409 ± 0.500 | 79.048 ± 0.959 | 56.897 ± 1.235 | 56.838 ± 0.783 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 31.293 ± 0.363 | - | 31.392 ± 0.103 | 36.310 ± 0.084 | 40.951 ± 0.803 | 56.168 ± 0.263 | 25.372 ± 0.898 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 117.044 ± 0.277 | - | 117.180 ± 0.467 | 131.002 ± 0.337 | 152.225 ± 0.882 | 120.416 ± 0.463 | 124.790 ± 0.249 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 49.834 ± 0.549 | - | 49.807 ± 0.197 | 56.438 ± 0.258 | 65.407 ± 0.759 | 120.277 ± 0.105 | 51.223 ± 0.509 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 19.588 ± 0.166 | - | 17.749 ± 0.162 | 16.374 ± 0.158 | 16.418 ± 0.229 | 17.059 ± 0.348 | 18.464 ± 0.234 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 12.517 ± 0.455 | - | 11.888 ± 0.322 | 13.142 ± 0.363 | 12.839 ± 0.486 | 18.907 ± 0.768 | 14.005 ± 0.196 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 2.557 ± 0.118 | - | 20.895 ± 0.113 | 24.197 ± 0.109 | 4.369 ± 0.308 | 4.046 ± 0.008 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 1.292 ± 0.028 | - | 17.138 ± 2.830 | 11.282 ± 0.014 | 3.826 ± 0.353 | 4.048 ± 0.015 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 8.835 ± 0.017 | - | 8.917 ± 0.008 | 9.126 ± 0.024 | 7.926 ± 0.138 | 9.174 ± 0.589 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 5.319 ± 0.063 | - | 5.563 ± 0.119 | 5.777 ± 0.054 | 7.043 ± 0.136 | 9.553 ± 0.077 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 5.385 ± 0.280 | - | 5.499 ± 0.117 | 5.401 ± 0.032 | 5.112 ± 0.048 | 4.838 ± 0.076 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.839 ± 0.125 | - | 3.920 ± 0.065 | 3.639 ± 0.042 | 4.748 ± 0.268 | 4.751 ± 0.448 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 9.094 ± 3.120 | - | 6.463 ± 0.015 | 7.705 ± 0.100 | 5.397 ± 0.047 | 5.248 ± 0.085 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 5.618 ± 0.064 | - | 5.678 ± 0.160 | 5.860 ± 0.016 | 6.255 ± 0.078 | 7.480 ± 1.736 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 43.908 ± 0.843 | - | 45.809 ± 7.912 | 42.160 ± 0.210 | 66.155 ± 0.326 | 44.028 ± 0.196 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 29.359 ± 0.089 | - | 30.200 ± 1.142 | 29.125 ± 0.123 | 30.063 ± 0.336 | 22.358 ± 0.204 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 48.615 ± 0.476 | - | unsupported | 51.162 ± 0.243 | 60.484 ± 1.136 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 24.926 ± 0.568 | - | unsupported | 26.467 ± 0.202 | 35.322 ± 0.209 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 103.275 ± 0.253 | - | 116.298 ± 0.603 | 24.194 ± 0.162 | 25.948 ± 0.223 | 25.865 ± 0.216 | 36.801 ± 0.193 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 16.502 ± 0.717 | - | 15.746 ± 0.456 | 16.785 ± 0.322 | 17.257 ± 0.489 | 28.320 ± 1.173 | 20.972 ± 0.662 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 66.722 ± 26.826 | - | 119.857 ± 8.074 | 31.290 ± 0.610 | 31.994 ± 0.399 | 32.518 ± 0.609 | 42.150 ± 0.196 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 23.591 ± 0.474 | - | 23.593 ± 0.613 | 23.052 ± 0.445 | 23.504 ± 0.612 | 37.297 ± 1.663 | 27.558 ± 1.106 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 26.929 ± 0.481 | - | 55.675 ± 0.262 | 37.402 ± 0.170 | 38.557 ± 0.138 | 69.066 ± 1.424 | 74.427 ± 3.295 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 12.710 ± 0.397 | - | 25.716 ± 0.124 | 20.087 ± 0.532 | 19.824 ± 0.054 | 70.538 ± 0.082 | 30.518 ± 2.412 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 8.458 ± 0.213 | - | 8.242 ± 0.432 | 10.346 ± 0.109 | 9.832 ± 0.042 | 11.310 ± 1.090 | 14.409 ± 1.382 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 7.039 ± 0.076 | - | 6.885 ± 0.042 | 8.485 ± 0.391 | 8.200 ± 0.115 | 12.219 ± 0.865 | 10.115 ± 4.524 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 18.654 ± 0.592 | - | 20.828 ± 0.124 | 24.582 ± 0.045 | 20.034 ± 0.343 | 17.087 ± 0.216 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 14.558 ± 0.065 | - | 14.436 ± 0.301 | 15.987 ± 0.250 | 16.590 ± 0.172 | 18.670 ± 0.286 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 28.361 ± 0.047 | - | 28.414 ± 0.040 | 32.242 ± 0.084 | 83.455 ± 0.175 | 44.574 ± 1.500 | 49.392 ± 1.666 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 11.912 ± 0.186 | - | 11.984 ± 0.372 | 15.542 ± 0.354 | 42.246 ± 0.023 | 43.476 ± 8.107 | 25.337 ± 1.882 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 111.461 ± 0.492 | - | 126.176 ± 1.069 | 31.120 ± 0.644 | 31.233 ± 0.348 | 32.459 ± 0.144 | 42.190 ± 0.190 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 23.532 ± 0.560 | - | 23.422 ± 0.219 | 23.081 ± 0.342 | 24.797 ± 0.311 | 34.183 ± 1.198 | 28.862 ± 1.304 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 20.562 ± 0.093 | - | 20.915 ± 0.047 | 20.008 ± 0.039 | 16.766 ± 0.164 | 39.254 ± 1.361 | 44.340 ± 1.333 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 8.467 ± 0.056 | - | 8.744 ± 0.020 | 10.111 ± 7.491 | 8.887 ± 0.073 | 39.533 ± 1.089 | 20.932 ± 0.950 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 20.422 ± 0.041 | - | 20.665 ± 0.027 | 20.840 ± 0.070 | 16.561 ± 0.176 | 26.831 ± 1.471 | 29.474 ± 3.113 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 8.806 ± 0.982 | - | 8.242 ± 0.024 | 9.123 ± 0.122 | 8.085 ± 0.101 | 26.858 ± 0.746 | 13.294 ± 3.216 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 26.326 ± 0.036 | - | 25.376 ± 0.037 | 26.540 ± 0.172 | 39.961 ± 0.079 | 60.918 ± 10.530 | 66.347 ± 0.822 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 10.502 ± 0.085 | - | 10.540 ± 0.028 | 12.790 ± 0.275 | 21.335 ± 0.073 | 60.526 ± 1.559 | 31.883 ± 3.273 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 13.228 ± 0.026 | - | 13.343 ± 0.034 | 21.279 ± 0.090 | 24.899 ± 0.223 | 35.260 ± 1.223 | 38.959 ± 2.655 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 5.822 ± 0.027 | - | 5.760 ± 0.042 | 9.345 ± 0.186 | 13.511 ± 0.565 | 35.141 ± 0.152 | 17.744 ± 2.627 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.312 ± 0.010 | 0.353 ± 0.014 | unsupported | unsupported | 0.374 ± 0.003 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.411 ± 0.039 | 0.557 ± 0.007 | unsupported | unsupported | 0.315 ± 0.025 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 112.628 ± 0.673 | - | 131.635 ± 0.754 | 31.281 ± 0.360 | 32.222 ± 0.409 | 32.917 ± 0.338 | 44.917 ± 0.184 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 24.306 ± 0.526 | - | 24.392 ± 0.602 | 23.149 ± 1.188 | 24.783 ± 1.561 | 35.172 ± 0.736 | 27.327 ± 0.684 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 114.869 ± 1.104 | - | 131.176 ± 0.394 | 30.858 ± 0.271 | 32.364 ± 0.363 | 32.985 ± 0.399 | 44.997 ± 0.345 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 23.889 ± 0.616 | - | 23.840 ± 0.945 | 23.836 ± 0.774 | 23.404 ± 0.712 | 35.937 ± 0.661 | 27.388 ± 1.002 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 31.024 ± 0.209 | - | 120.961 ± 0.558 | 31.066 ± 0.248 | 31.734 ± 0.333 | 32.798 ± 0.227 | 42.274 ± 0.316 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 24.000 ± 2.291 | - | 26.182 ± 3.510 | 23.381 ± 0.754 | 25.753 ± 0.568 | 34.418 ± 0.626 | 27.048 ± 1.186 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 103.389 ± 0.726 | - | 114.720 ± 0.524 | 24.275 ± 0.136 | 25.865 ± 0.410 | 25.987 ± 0.563 | 36.846 ± 0.484 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 16.785 ± 1.060 | - | 16.216 ± 0.500 | 16.953 ± 0.490 | 17.926 ± 0.528 | 28.604 ± 0.953 | 20.703 ± 1.424 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 57.836 ± 0.235 | - | 58.211 ± 0.178 | 51.434 ± 0.115 | 62.702 ± 0.235 | 114.872 ± 3.632 | 112.065 ± 4.523 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 23.639 ± 0.087 | - | 24.014 ± 0.275 | 19.874 ± 0.079 | 31.082 ± 0.088 | 115.013 ± 0.106 | 46.778 ± 3.188 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 14.856 ± 0.415 | - | 15.169 ± 0.470 | 10.502 ± 0.265 | 30.397 ± 0.364 | 12.241 ± 0.173 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 6.444 ± 0.081 | - | 6.526 ± 0.107 | 6.287 ± 0.027 | 15.177 ± 0.161 | 13.513 ± 0.309 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.733 ± 0.047 | - | 1.804 ± 0.128 | 37.098 ± 1.656 | 2.559 ± 0.024 | 1.230 ± 0.226 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.787 ± 0.021 | - | 0.791 ± 0.010 | 12.150 ± 0.261 | 1.430 ± 0.068 | 1.555 ± 0.024 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 1.763 ± 0.094 | - | 1.759 ± 0.133 | 4.026 ± 0.024 | 2.459 ± 0.894 | 1.352 ± 0.076 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 0.894 ± 0.256 | - | 0.921 ± 0.077 | 2.304 ± 0.024 | 2.068 ± 0.198 | 1.579 ± 0.090 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 14.447 ± 0.291 | - | 16.298 ± 0.632 | 10.268 ± 0.091 | 30.273 ± 0.253 | 12.043 ± 0.210 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 6.421 ± 0.175 | - | 6.412 ± 0.121 | 6.263 ± 0.078 | 15.210 ± 0.622 | 13.640 ± 0.450 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.695 ± 0.043 | - | 1.755 ± 0.100 | 33.356 ± 0.371 | 2.631 ± 0.021 | 1.193 ± 0.088 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.799 ± 0.031 | - | 0.798 ± 0.042 | 11.237 ± 0.972 | 1.449 ± 0.483 | 1.678 ± 0.165 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 7.248 ± 0.211 | - | 7.393 ± 0.160 | 16.017 ± 0.083 | 12.284 ± 0.389 | 6.005 ± 0.138 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 3.939 ± 0.232 | - | 3.886 ± 0.215 | 8.738 ± 0.122 | 8.914 ± 4.008 | 6.772 ± 0.297 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 10.317 ± 0.529 | - | 10.398 ± 0.157 | 9.291 ± 0.209 | 28.498 ± 0.312 | 9.670 ± 0.233 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 6.373 ± 0.069 | - | 6.452 ± 0.177 | 6.284 ± 0.185 | 14.898 ± 0.149 | 10.867 ± 0.277 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 1.004 ± 0.073 | - | 1.077 ± 0.113 | 2.004 ± 0.078 | 2.214 ± 0.040 | 0.945 ± 0.114 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.398 ± 0.037 | - | 0.361 ± 0.039 | 0.836 ± 0.062 | 1.256 ± 0.066 | 1.109 ± 0.048 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.359 ± 0.035 | - | 1.388 ± 0.044 | 0.719 ± 0.019 | 3.961 ± 0.331 | 1.023 ± 0.106 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.556 ± 0.039 | - | 0.565 ± 0.041 | 0.330 ± 0.068 | 2.031 ± 0.267 | 1.210 ± 0.071 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 10.219 ± 0.191 | - | 10.309 ± 0.350 | 9.355 ± 0.257 | 28.326 ± 0.240 | 9.712 ± 0.291 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 6.435 ± 0.069 | - | 6.383 ± 0.031 | 6.493 ± 0.380 | 14.791 ± 0.332 | 10.738 ± 0.211 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.986 ± 0.139 | - | 0.999 ± 0.150 | 1.989 ± 0.108 | 2.185 ± 0.014 | 0.867 ± 0.129 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.444 ± 0.040 | - | 0.396 ± 0.035 | 0.826 ± 0.020 | 1.455 ± 1.447 | 1.067 ± 0.033 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.105 ± 0.122 | - | 1.158 ± 0.174 | 0.724 ± 0.016 | 3.983 ± 0.404 | 0.940 ± 0.079 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.550 ± 0.066 | - | 0.582 ± 0.056 | 0.403 ± 0.122 | 1.940 ± 0.235 | 1.243 ± 0.079 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 1.235 ± 0.030 | 1.132 ± 0.155 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.376 ± 0.037 | 0.372 ± 0.036 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 39.958 ± 1.729 | - | 40.256 ± 0.896 | 30.784 ± 0.114 | 43.206 ± 0.265 | 47.045 ± 0.796 | 54.870 ± 3.753 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 17.616 ± 0.050 | - | 17.731 ± 0.147 | 14.110 ± 0.295 | 25.264 ± 0.113 | 48.263 ± 1.727 | 26.404 ± 1.884 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 138.184 ± 0.570 | - | 137.795 ± 0.515 | 48.779 ± 0.142 | 153.550 ± 0.178 | 154.460 ± 0.455 | 154.445 ± 0.322 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 20.778 ± 2.376 | - | 22.406 ± 2.737 | 23.252 ± 2.001 | 77.556 ± 0.274 | 156.082 ± 0.590 | 73.394 ± 2.152 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 118.780 ± 0.191 | - | 131.567 ± 0.689 | 39.339 ± 0.168 | 34.516 ± 0.693 | 33.948 ± 0.222 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 26.229 ± 0.378 | - | 26.915 ± 0.863 | 25.002 ± 0.934 | 26.269 ± 1.542 | 36.063 ± 0.550 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 106.235 ± 1.533 | - | 118.623 ± 0.490 | 26.412 ± 0.325 | 28.289 ± 0.417 | 27.554 ± 0.416 | 32.194 ± 0.235 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 16.588 ± 0.259 | - | 16.939 ± 0.576 | 16.973 ± 0.548 | 17.995 ± 0.594 | 29.652 ± 0.827 | 19.030 ± 0.424 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 24.444 ± 0.035 | - | 24.464 ± 0.034 | 34.320 ± 0.160 | 82.775 ± 0.035 | 42.965 ± 0.792 | 44.874 ± 1.450 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 10.234 ± 0.269 | - | 10.443 ± 0.357 | 16.448 ± 0.191 | 41.928 ± 0.034 | 42.877 ± 1.646 | 22.791 ± 1.469 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 125.788 ± 0.505 | - | 125.649 ± 0.360 | 88.188 ± 0.493 | 37.622 ± 0.244 | 105.454 ± 0.375 | 108.942 ± 0.189 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 18.644 ± 0.797 | - | 18.007 ± 1.247 | 40.264 ± 1.085 | 20.024 ± 0.216 | 105.713 ± 0.316 | 53.005 ± 1.355 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 31.318 ± 0.253 | - | 124.091 ± 0.956 | 30.838 ± 0.086 | 31.756 ± 0.530 | 32.522 ± 0.359 | 42.199 ± 0.235 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 24.006 ± 0.889 | - | 23.332 ± 0.580 | 23.141 ± 0.378 | 24.537 ± 1.068 | 34.961 ± 1.106 | 25.992 ± 0.932 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 51.552 ± 0.207 | - | 51.979 ± 0.038 | 52.852 ± 0.081 | 14.618 ± 0.095 | 58.053 ± 4.103 | 75.063 ± 20.840 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 20.546 ± 0.423 | - | 20.552 ± 0.075 | 25.215 ± 0.239 | 11.046 ± 4.358 | 56.947 ± 1.768 | 28.673 ± 2.475 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 126.297 ± 0.458 | 130.639 ± 0.875 | unsupported | 39.314 ± 0.133 | 34.065 ± 0.518 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 26.522 ± 0.633 | 38.466 ± 16.102 | unsupported | 24.543 ± 0.438 | 24.136 ± 0.807 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.775 ± 0.103 | - | 0.953 ± 0.077 | 0.666 ± 0.089 | 0.682 ± 0.056 | 1.963 ± 0.766 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.288 ± 0.036 | - | 0.305 ± 0.025 | 0.279 ± 0.035 | 0.729 ± 0.117 | 1.201 ± 1.412 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.608 ± 0.037 | - | 0.666 ± 0.024 | 0.799 ± 0.125 | 0.789 ± 0.211 | 1.147 ± 0.309 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.331 ± 0.079 | - | 0.298 ± 0.028 | 0.279 ± 0.034 | 0.652 ± 0.315 | 1.213 ± 0.385 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.234 ± 0.030 | - | unsupported | 1.079 ± 0.118 | 1.041 ± 0.176 | 2.427 ± 0.709 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.547 ± 0.132 | - | unsupported | 0.592 ± 0.078 | 0.831 ± 0.123 | 2.830 ± 1.026 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.390 ± 0.013 | - | 0.489 ± 0.029 | 0.692 ± 0.014 | 0.436 ± 0.026 | 1.150 ± 0.163 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.364 ± 0.090 | - | 0.418 ± 0.033 | 0.349 ± 0.023 | 0.369 ± 0.047 | 1.193 ± 0.062 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.658 ± 0.063 | - | 0.738 ± 0.047 | 0.855 ± 0.041 | 0.931 ± 0.052 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.296 ± 0.051 | - | 0.295 ± 0.042 | 0.458 ± 0.051 | 0.628 ± 0.132 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.809 ± 0.195 | - | 0.743 ± 0.024 | 0.598 ± 0.027 | 0.708 ± 0.112 | 1.408 ± 1.715 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.298 ± 0.057 | - | 0.305 ± 0.030 | 0.258 ± 0.057 | 0.401 ± 0.033 | 2.663 ± 1.862 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.817 ± 0.092 | - | 0.954 ± 0.025 | 0.433 ± 0.017 | 0.478 ± 0.015 | 1.263 ± 0.193 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.871 ± 0.023 | - | 1.066 ± 0.056 | 0.598 ± 0.019 | 0.501 ± 0.039 | 1.214 ± 0.823 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.612 ± 0.031 | - | 0.003 ± 0.000 | 1.612 ± 0.032 | 1.213 ± 0.019 | 3.577 ± 0.368 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.648 ± 0.054 | - | 0.003 ± 0.000 | 0.696 ± 0.130 | 0.801 ± 0.055 | 3.630 ± 0.360 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.050 ± 0.000 | - | 0.055 ± 0.003 | 0.076 ± 0.004 | 0.122 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.093 ± 0.001 | - | 0.102 ± 0.002 | 0.123 ± 0.004 | 0.183 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.237 ± 0.010 | - | 0.263 ± 0.010 | 0.280 ± 0.014 | 0.350 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.081 ± 0.003 | - | 0.091 ± 0.002 | 0.149 ± 0.003 | 0.121 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.079 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.002 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.003 | 0.066 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.677 ± 0.013 | - | 0.711 ± 0.013 | 1.162 ± 0.014 | 0.675 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.001 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.025 ± 0.000 | - | 0.032 ± 0.000 | 0.039 ± 0.001 | 0.066 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.067 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.050 ± 0.000 | - | 0.055 ± 0.000 | 0.085 ± 0.003 | 0.106 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.092 ± 0.000 | - | 0.102 ± 0.004 | 0.124 ± 0.002 | 0.163 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.235 ± 0.021 | - | 0.350 ± 0.029 | 0.341 ± 0.017 | 0.371 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.081 ± 0.002 | - | 0.097 ± 0.002 | 0.180 ± 0.019 | 0.138 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.002 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.001 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.001 | - | 0.005 ± 0.006 | 0.010 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.001 | - | 0.005 ± 0.000 | 0.010 ± 0.001 | 0.014 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.004 ± 0.001 | - | 0.005 ± 0.000 | 0.011 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.690 ± 0.017 | - | 0.821 ± 0.039 | 1.379 ± 0.022 | 1.021 ± 0.078 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.026 ± 0.001 | - | 0.032 ± 0.000 | 0.059 ± 0.000 | 0.085 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.627 ± 0.011 | - | 0.637 ± 0.015 | 0.733 ± 0.004 | 0.939 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 2.131 ± 0.023 | - | 2.150 ± 0.014 | 2.141 ± 0.010 | 2.728 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 6.473 ± 0.012 | - | 6.520 ± 0.018 | 6.523 ± 0.026 | 8.969 ± 0.161 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 3.659 ± 0.025 | - | 3.683 ± 0.019 | 3.227 ± 0.014 | 2.641 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.001 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.110 ± 0.002 | - | 0.115 ± 0.002 | 0.121 ± 0.008 | 0.169 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.001 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.008 ± 0.000 | - | 0.013 ± 0.000 | 0.017 ± 0.006 | 0.033 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.001 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.013 ± 0.000 | 0.016 ± 0.001 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.021 ± 0.000 | - | 0.026 ± 0.003 | 0.029 ± 0.001 | 0.080 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.001 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | 0.015 ± 0.000 | 0.018 ± 0.001 | 0.034 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.027 ± 0.000 | - | 0.032 ± 0.000 | 0.036 ± 0.007 | 0.088 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 29.141 ± 0.064 | - | 29.305 ± 0.064 | 25.333 ± 0.121 | 21.408 ± 0.154 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.015 ± 0.000 | 0.031 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.867 ± 0.026 | - | 0.878 ± 0.004 | 0.877 ± 0.002 | 0.990 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.018 ± 0.001 | - | 0.023 ± 0.000 | 0.026 ± 0.001 | 0.043 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.052 ± 0.000 | - | 0.057 ± 0.002 | 0.061 ± 0.001 | 0.124 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 1.027 ± 0.004 | - | 1.041 ± 0.006 | 0.718 ± 0.426 | 0.976 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 3.410 ± 0.035 | - | 3.446 ± 0.013 | 2.130 ± 0.002 | 2.396 ± 0.058 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 9.936 ± 0.031 | - | 6.697 ± 0.032 | 9.367 ± 0.059 | 6.026 ± 0.229 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 3.602 ± 0.013 | - | 3.628 ± 0.283 | 2.246 ± 0.024 | 3.504 ± 0.494 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.011 ± 0.000 | 0.013 ± 0.001 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.110 ± 0.002 | - | 0.115 ± 0.002 | 0.183 ± 0.021 | 0.211 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | 0.014 ± 0.000 | 0.012 ± 0.004 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.013 ± 0.000 | - | 0.020 ± 0.000 | 0.016 ± 0.001 | 0.032 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.006 ± 0.001 | - | 0.013 ± 0.001 | 0.017 ± 0.002 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.013 ± 0.000 | - | 0.021 ± 0.001 | 0.016 ± 0.001 | 0.032 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.032 ± 0.000 | - | 0.040 ± 0.002 | 0.030 ± 0.003 | 0.078 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.008 ± 0.001 | - | 0.014 ± 0.001 | 0.017 ± 0.001 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.016 ± 0.000 | - | 0.028 ± 0.017 | 0.018 ± 0.001 | 0.033 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.041 ± 0.001 | - | 0.050 ± 0.004 | 0.036 ± 0.001 | 0.064 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 29.070 ± 0.163 | - | 28.876 ± 0.563 | 17.555 ± 0.229 | 28.476 ± 3.936 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.011 ± 0.000 | - | 0.019 ± 0.000 | 0.022 ± 0.001 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.882 ± 0.014 | - | 0.894 ± 0.009 | 1.261 ± 0.014 | 1.569 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.018 ± 0.000 | - | 0.036 ± 0.021 | 0.026 ± 0.001 | 0.045 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.079 ± 0.001 | - | 0.090 ± 0.010 | 0.061 ± 0.001 | 0.143 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.683 ± 0.035 | - | 0.681 ± 0.020 | 0.524 ± 0.018 | 0.453 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 2.235 ± 0.030 | - | 2.251 ± 0.031 | 1.954 ± 0.003 | 1.958 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 5.822 ± 0.049 | - | 5.818 ± 0.010 | 5.702 ± 0.014 | 6.508 ± 0.058 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.325 ± 0.018 | - | 1.345 ± 0.002 | 1.362 ± 0.018 | 1.462 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.003 ± 0.001 | - | 0.008 ± 0.007 | 0.008 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.070 ± 0.004 | - | 0.079 ± 0.006 | 0.090 ± 0.006 | 0.138 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | 0.012 ± 0.000 | 0.015 ± 0.001 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.013 ± 0.000 | 0.014 ± 0.001 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.018 ± 0.000 | - | 0.024 ± 0.000 | 0.025 ± 0.001 | 0.041 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.001 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | 0.015 ± 0.000 | 0.017 ± 0.000 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.024 ± 0.000 | - | 0.029 ± 0.000 | 0.031 ± 0.001 | 0.047 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 10.431 ± 0.034 | - | 10.566 ± 0.042 | 10.655 ± 0.062 | 11.218 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.007 ± 0.000 | - | 0.011 ± 0.001 | 0.012 ± 0.000 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.553 ± 0.012 | - | 0.559 ± 0.013 | 0.630 ± 0.019 | 0.755 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.019 ± 0.000 | - | 0.024 ± 0.000 | 0.023 ± 0.001 | 0.037 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.046 ± 0.000 | - | 0.051 ± 0.000 | 0.052 ± 0.001 | 0.127 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 1.103 ± 0.008 | - | 1.112 ± 0.007 | 0.807 ± 0.054 | 0.495 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 3.585 ± 0.009 | - | 3.624 ± 0.027 | 1.932 ± 0.008 | 1.326 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 8.992 ± 0.008 | - | 5.948 ± 0.021 | 8.572 ± 0.016 | 3.324 ± 0.432 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.295 ± 0.026 | - | 1.305 ± 0.030 | 1.158 ± 0.037 | 1.738 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.004 ± 0.001 | - | 0.010 ± 0.001 | 0.012 ± 0.002 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.115 ± 0.006 | - | 0.127 ± 0.005 | 0.136 ± 0.003 | 0.162 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | 0.013 ± 0.000 | 0.011 ± 0.001 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.011 ± 0.000 | - | 0.020 ± 0.000 | 0.015 ± 0.001 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.006 ± 0.000 | - | 0.013 ± 0.001 | 0.014 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.013 ± 0.001 | - | 0.021 ± 0.001 | 0.015 ± 0.001 | 0.028 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.029 ± 0.000 | - | 0.037 ± 0.000 | 0.026 ± 0.001 | 0.041 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.009 ± 0.001 | - | 0.014 ± 0.001 | 0.015 ± 0.002 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.017 ± 0.000 | - | 0.025 ± 0.000 | 0.017 ± 0.001 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.037 ± 0.000 | - | 0.046 ± 0.001 | 0.032 ± 0.001 | 0.049 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 10.077 ± 0.068 | - | 10.226 ± 0.032 | 9.117 ± 0.123 | 15.251 ± 2.069 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.012 ± 0.007 | - | 0.018 ± 0.000 | 0.018 ± 0.004 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.887 ± 0.012 | - | 0.909 ± 0.012 | 0.949 ± 0.002 | 0.471 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.030 ± 0.000 | - | 0.039 ± 0.000 | 0.024 ± 0.001 | 0.039 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.072 ± 0.001 | - | 0.079 ± 0.003 | 0.053 ± 0.003 | 0.077 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.269 ± 0.012 | - | 0.272 ± 0.012 | 0.236 ± 0.018 | 0.447 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 1.098 ± 0.003 | - | 1.109 ± 0.002 | 0.950 ± 0.002 | 1.962 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 3.092 ± 0.006 | - | 3.115 ± 0.004 | 2.887 ± 0.011 | 6.460 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.515 ± 0.008 | - | 0.530 ± 0.022 | 0.561 ± 0.016 | 1.455 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.002 ± 0.002 | - | 0.005 ± 0.002 | 0.010 ± 0.001 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.031 ± 0.000 | - | 0.035 ± 0.004 | 0.040 ± 0.001 | 0.137 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.008 ± 0.001 | 0.013 ± 0.001 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.004 | 0.018 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.013 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.010 ± 0.000 | - | 0.015 ± 0.000 | 0.020 ± 0.002 | 0.035 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.007 ± 0.005 | 0.010 ± 0.001 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.014 ± 0.001 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.013 ± 0.000 | - | 0.018 ± 0.001 | 0.022 ± 0.006 | 0.042 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 4.092 ± 0.023 | - | 4.117 ± 0.018 | 4.328 ± 0.023 | 11.074 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.232 ± 0.010 | - | 0.238 ± 0.012 | 0.245 ± 0.017 | 0.760 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.010 ± 0.000 | - | 0.014 ± 0.000 | 0.017 ± 0.001 | 0.031 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.025 ± 0.000 | - | 0.029 ± 0.000 | 0.032 ± 0.000 | 0.100 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.455 ± 0.019 | - | 0.465 ± 0.021 | 0.396 ± 0.016 | 0.511 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 1.610 ± 0.011 | - | 1.628 ± 0.020 | 0.951 ± 0.029 | 1.929 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 4.231 ± 0.402 | - | 4.305 ± 0.025 | 3.665 ± 0.012 | 3.691 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.690 ± 0.010 | - | 0.711 ± 0.018 | 0.693 ± 0.016 | 1.718 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.003 ± 0.002 | - | 0.008 ± 0.001 | 0.015 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.039 ± 0.000 | - | 0.048 ± 0.001 | 0.057 ± 0.004 | 0.128 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.010 ± 0.001 | 0.016 ± 0.002 | 0.019 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.006 ± 0.006 | - | 0.013 ± 0.001 | 0.013 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.005 ± 0.004 | - | 0.010 ± 0.001 | 0.015 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.007 ± 0.002 | - | 0.014 ± 0.000 | 0.019 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.014 ± 0.001 | - | 0.022 ± 0.000 | 0.020 ± 0.000 | 0.044 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.005 ± 0.001 | - | 0.010 ± 0.001 | 0.016 ± 0.002 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.009 ± 0.000 | - | 0.015 ± 0.001 | 0.020 ± 0.003 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.018 ± 0.000 | - | 0.025 ± 0.000 | 0.023 ± 0.002 | 0.053 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 5.412 ± 0.020 | - | 5.548 ± 0.039 | 5.216 ± 0.030 | 13.975 ± 2.196 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.006 ± 0.002 | - | 0.012 ± 0.001 | 0.017 ± 0.001 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.300 ± 0.020 | - | 0.308 ± 0.022 | 0.319 ± 0.020 | 0.466 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.014 ± 0.000 | - | 0.021 ± 0.001 | 0.025 ± 0.003 | 0.032 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.034 ± 0.000 | - | 0.042 ± 0.001 | 0.033 ± 0.000 | 0.114 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.160 ± 0.005 | - | 0.162 ± 0.004 | 0.188 ± 0.002 | 0.131 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.221 ± 0.013 | - | 0.227 ± 0.002 | 0.253 ± 0.001 | 0.218 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.519 ± 0.012 | - | 0.555 ± 0.014 | 0.542 ± 0.021 | 0.599 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.136 ± 0.002 | - | 0.149 ± 0.003 | 0.173 ± 0.006 | 0.205 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.014 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.012 ± 0.000 | - | 0.017 ± 0.001 | 0.025 ± 0.001 | 0.057 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.004 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.014 ± 0.000 | - | 0.020 ± 0.001 | 0.029 ± 0.001 | 0.052 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.001 | 0.023 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.003 | 0.016 ± 0.002 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.052 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.003 | 0.016 ± 0.001 | 0.026 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.021 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.051 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 1.091 ± 0.011 | - | 1.111 ± 0.031 | 1.281 ± 0.012 | 1.263 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.007 ± 0.002 | - | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.077 ± 0.001 | - | 0.085 ± 0.002 | 0.087 ± 0.001 | 0.131 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.002 | 0.017 ± 0.008 | 0.056 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.006 ± 0.000 | - | 0.011 ± 0.000 | 0.020 ± 0.003 | 0.051 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.200 ± 0.022 | - | 0.209 ± 0.005 | 0.170 ± 0.003 | 0.134 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.303 ± 0.024 | - | 0.318 ± 0.017 | 0.233 ± 0.020 | 0.252 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.801 ± 0.008 | - | 0.871 ± 0.013 | 0.383 ± 0.019 | 0.689 ± 0.092 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.195 ± 0.022 | - | 0.207 ± 0.007 | 0.226 ± 0.026 | 0.231 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.023 ± 0.003 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.019 ± 0.001 | - | 0.027 ± 0.000 | 0.038 ± 0.001 | 0.063 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.010 ± 0.001 | 0.024 ± 0.003 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.022 ± 0.001 | - | 0.031 ± 0.001 | 0.042 ± 0.006 | 0.059 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.010 ± 0.001 | 0.023 ± 0.002 | 0.023 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.005 ± 0.001 | - | 0.010 ± 0.001 | 0.025 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.005 ± 0.001 | - | 0.010 ± 0.001 | 0.026 ± 0.000 | 0.025 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.007 ± 0.002 | - | 0.012 ± 0.001 | 0.028 ± 0.001 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.006 ± 0.003 | - | 0.010 ± 0.001 | 0.026 ± 0.000 | 0.027 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.006 ± 0.001 | - | 0.010 ± 0.000 | 0.026 ± 0.008 | 0.028 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.008 ± 0.001 | - | 0.013 ± 0.000 | 0.028 ± 0.000 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 1.522 ± 0.022 | - | 1.635 ± 0.018 | 0.704 ± 0.023 | 0.989 ± 0.143 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.006 ± 0.001 | - | 0.011 ± 0.000 | 0.027 ± 0.001 | 0.021 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.123 ± 0.001 | - | 0.136 ± 0.008 | 0.079 ± 0.003 | 0.168 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.006 ± 0.000 | - | 0.012 ± 0.001 | 0.028 ± 0.001 | 0.023 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.010 ± 0.000 | - | 0.016 ± 0.000 | 0.031 ± 0.002 | 0.054 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.251 ± 0.009 | - | 0.262 ± 0.003 | 0.233 ± 0.013 | 0.325 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.599 ± 0.026 | - | 0.614 ± 0.017 | 0.557 ± 0.091 | 0.868 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.352 ± 0.023 | - | 1.377 ± 0.039 | 1.398 ± 0.031 | 2.320 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.492 ± 0.025 | - | 0.514 ± 0.031 | 0.568 ± 0.022 | 0.637 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.018 ± 0.000 | - | 0.022 ± 0.001 | 0.026 ± 0.001 | 0.072 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.021 ± 0.015 | - | 0.023 ± 0.000 | 0.029 ± 0.001 | 0.071 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.024 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.023 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.001 | 0.007 ± 0.000 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.011 ± 0.001 | 0.032 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 3.942 ± 0.019 | - | 4.695 ± 0.069 | 4.456 ± 1.270 | 4.709 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.001 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.128 ± 0.001 | - | 0.137 ± 0.002 | 0.176 ± 0.014 | 0.224 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.009 ± 0.001 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.012 ± 0.000 | - | 0.016 ± 0.011 | 0.016 ± 0.000 | 0.039 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.250 ± 0.027 | - | 0.255 ± 0.027 | 0.327 ± 0.015 | 0.349 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.612 ± 0.022 | - | 0.618 ± 0.018 | 0.785 ± 0.004 | 0.872 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.370 ± 0.024 | - | 1.436 ± 0.036 | 2.067 ± 0.008 | 1.361 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.573 ± 0.037 | - | 0.601 ± 0.029 | 0.510 ± 0.022 | 0.927 ± 0.076 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.018 ± 0.001 | - | 0.022 ± 0.000 | 0.044 ± 0.000 | 0.073 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.031 ± 0.000 | - | 0.039 ± 0.001 | 0.047 ± 0.001 | 0.074 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.001 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.006 ± 0.003 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.009 ± 0.007 | - | 0.009 ± 0.000 | 0.016 ± 0.001 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.021 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.007 ± 0.003 | - | 0.008 ± 0.000 | 0.013 ± 0.002 | 0.022 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.007 ± 0.006 | - | 0.011 ± 0.007 | 0.018 ± 0.001 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 4.512 ± 0.096 | - | 5.647 ± 0.136 | 3.736 ± 0.050 | 7.296 ± 2.865 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.007 ± 0.001 | - | 0.008 ± 0.000 | 0.012 ± 0.001 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.129 ± 0.002 | - | 0.136 ± 0.006 | 0.291 ± 0.016 | 0.235 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.011 ± 0.004 | - | 0.010 ± 0.000 | 0.016 ± 0.001 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.012 ± 0.000 | - | 0.016 ± 0.000 | 0.026 ± 0.001 | 0.040 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.164 ± 0.006 | - | 0.152 ± 0.006 | 0.099 ± 0.002 | 0.264 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.246 ± 0.001 | - | 0.224 ± 0.043 | 0.140 ± 0.003 | 0.370 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.565 ± 0.013 | - | 0.564 ± 0.013 | 0.406 ± 0.016 | 0.821 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.117 ± 0.001 | - | 0.131 ± 0.042 | 0.145 ± 0.004 | 0.213 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.001 | 0.016 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.011 ± 0.000 | - | 0.018 ± 0.003 | 0.029 ± 0.001 | 0.084 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.002 | 0.021 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.001 | 0.019 ± 0.001 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.022 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.001 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.019 ± 0.001 | 0.022 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.915 ± 0.002 | - | 0.995 ± 0.009 | 1.025 ± 0.008 | 1.037 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.070 ± 0.001 | - | 0.080 ± 0.020 | 0.080 ± 0.001 | 0.163 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.012 ± 0.001 | 0.021 ± 0.000 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.162 ± 0.002 | - | 0.148 ± 0.006 | 0.132 ± 0.002 | 0.300 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.219 ± 0.027 | - | 0.223 ± 0.040 | 0.173 ± 0.021 | 0.394 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.553 ± 0.029 | - | 0.634 ± 0.026 | 0.382 ± 0.020 | 0.883 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.117 ± 0.003 | - | 0.167 ± 0.044 | 0.206 ± 0.023 | 0.201 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.027 ± 0.002 | 0.021 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.011 ± 0.000 | - | 0.018 ± 0.003 | 0.043 ± 0.004 | 0.060 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.002 | - | 0.014 ± 0.044 | 0.026 ± 0.004 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.004 ± 0.001 | - | 0.008 ± 0.000 | 0.027 ± 0.004 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.027 ± 0.005 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.027 ± 0.002 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.005 ± 0.003 | - | 0.018 ± 0.011 | 0.030 ± 0.004 | 0.027 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.027 ± 0.004 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.027 ± 0.003 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.007 ± 0.005 | - | 0.009 ± 0.000 | 0.030 ± 0.004 | 0.023 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.931 ± 0.012 | - | 1.076 ± 0.019 | 0.585 ± 0.021 | 0.799 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.005 ± 0.002 | - | 0.008 ± 0.000 | 0.029 ± 0.003 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.070 ± 0.005 | - | 0.085 ± 0.021 | 0.073 ± 0.001 | 0.202 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.006 ± 0.004 | - | 0.009 ± 0.000 | 0.030 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.006 ± 0.001 | - | 0.011 ± 0.001 | 0.033 ± 0.001 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.822 ± 0.026 | - | 1.825 ± 0.011 | 1.573 ± 0.011 | 2.090 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 6.044 ± 0.003 | - | 6.055 ± 0.005 | 5.164 ± 0.033 | 5.154 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 17.757 ± 0.014 | - | 17.813 ± 0.044 | 15.939 ± 0.028 | 14.707 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 5.011 ± 0.074 | - | 4.974 ± 0.024 | 5.549 ± 0.032 | 4.196 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.015 ± 0.001 | 0.029 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.204 ± 0.017 | - | 0.213 ± 0.015 | 0.231 ± 0.008 | 0.252 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.009 ± 0.000 | - | 0.015 ± 0.000 | 0.021 ± 0.006 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.178 ± 0.006 | - | 0.189 ± 0.006 | 0.206 ± 0.013 | 0.232 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.021 ± 0.002 | - | 0.033 ± 0.024 | 0.031 ± 0.001 | 0.043 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.008 ± 0.000 | - | 0.014 ± 0.000 | 0.019 ± 0.002 | 0.033 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.037 ± 0.003 | - | 0.026 ± 0.005 | 0.030 ± 0.001 | 0.044 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.055 ± 0.000 | - | 0.061 ± 0.000 | 0.063 ± 0.002 | 0.075 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.010 ± 0.000 | - | 0.015 ± 0.001 | 0.019 ± 0.001 | 0.035 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.026 ± 0.000 | - | 0.033 ± 0.002 | 0.035 ± 0.008 | 0.051 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.073 ± 0.001 | - | 0.079 ± 0.007 | 0.079 ± 0.006 | 0.088 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 39.970 ± 0.220 | - | 40.688 ± 0.157 | 44.047 ± 0.353 | 33.093 ± 0.078 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.017 ± 0.000 | - | 0.023 ± 0.001 | 0.025 ± 0.001 | 0.043 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.559 ± 0.045 | - | 1.577 ± 0.045 | 1.647 ± 0.011 | 1.569 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.049 ± 0.000 | - | 0.056 ± 0.001 | 0.056 ± 0.000 | 0.072 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.141 ± 0.003 | - | 0.150 ± 0.019 | 0.142 ± 0.006 | 0.160 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 3.057 ± 0.011 | - | 3.081 ± 0.015 | 1.555 ± 0.011 | 2.244 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 9.822 ± 0.031 | - | 6.009 ± 0.012 | 5.086 ± 0.029 | 2.881 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 17.712 ± 0.072 | - | 17.975 ± 0.071 | 18.582 ± 0.183 | 7.719 ± 0.086 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 4.699 ± 0.020 | - | 4.821 ± 0.031 | 4.629 ± 0.031 | 4.888 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.007 ± 0.000 | - | 0.016 ± 0.000 | 0.019 ± 0.001 | 0.032 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.315 ± 0.014 | - | 0.323 ± 0.033 | 0.328 ± 0.023 | 0.273 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.014 ± 0.000 | - | 0.025 ± 0.000 | 0.020 ± 0.005 | 0.037 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.271 ± 0.022 | - | 0.287 ± 0.030 | 0.289 ± 0.017 | 0.234 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.021 ± 0.000 | - | 0.026 ± 0.000 | 0.032 ± 0.006 | 0.044 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.013 ± 0.001 | - | 0.023 ± 0.004 | 0.025 ± 0.004 | 0.035 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.032 ± 0.000 | - | 0.043 ± 0.003 | 0.030 ± 0.001 | 0.043 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.054 ± 0.007 | - | 0.061 ± 0.026 | 0.063 ± 0.001 | 0.075 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.016 ± 0.000 | - | 0.025 ± 0.004 | 0.028 ± 0.003 | 0.038 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.042 ± 0.002 | - | 0.054 ± 0.005 | 0.035 ± 0.010 | 0.050 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.073 ± 0.001 | - | 0.080 ± 0.010 | 0.080 ± 0.000 | 0.094 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 37.802 ± 0.767 | - | 39.363 ± 0.260 | 36.166 ± 0.373 | 72.946 ± 22.268 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.028 ± 0.000 | - | 0.039 ± 0.004 | 0.038 ± 0.004 | 0.046 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 2.352 ± 0.023 | - | 2.404 ± 0.040 | 2.345 ± 0.023 | 0.919 ± 0.065 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.080 ± 0.000 | - | 0.091 ± 0.006 | 0.056 ± 0.005 | 0.070 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.141 ± 0.002 | - | 0.147 ± 0.006 | 0.141 ± 0.005 | 0.191 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 1.361 ± 0.010 | - | 1.371 ± 0.011 | 1.098 ± 0.009 | 1.331 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 5.137 ± 0.005 | - | 5.413 ± 0.013 | 4.370 ± 0.017 | 4.059 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 15.215 ± 0.029 | - | 15.872 ± 0.009 | 13.874 ± 0.055 | 12.224 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 2.306 ± 0.017 | - | 2.337 ± 0.010 | 2.477 ± 0.013 | 2.437 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.004 ± 0.001 | - | 0.009 ± 0.001 | 0.013 ± 0.002 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.148 ± 0.023 | - | 0.145 ± 0.007 | 0.164 ± 0.010 | 0.231 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.008 ± 0.000 | - | 0.014 ± 0.000 | 0.018 ± 0.001 | 0.032 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.107 ± 0.008 | - | 0.117 ± 0.003 | 0.133 ± 0.018 | 0.206 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.017 ± 0.002 | - | 0.024 ± 0.000 | 0.027 ± 0.001 | 0.039 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.006 ± 0.000 | - | 0.012 ± 0.004 | 0.016 ± 0.001 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.018 ± 0.000 | - | 0.024 ± 0.000 | 0.027 ± 0.001 | 0.038 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.043 ± 0.000 | - | 0.051 ± 0.000 | 0.052 ± 0.004 | 0.076 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.008 ± 0.000 | - | 0.013 ± 0.003 | 0.017 ± 0.001 | 0.033 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.023 ± 0.001 | - | 0.030 ± 0.001 | 0.031 ± 0.007 | 0.043 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.058 ± 0.002 | - | 0.066 ± 0.001 | 0.065 ± 0.001 | 0.074 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 18.382 ± 0.184 | - | 18.442 ± 0.028 | 19.414 ± 0.150 | 18.599 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.013 ± 0.000 | - | 0.018 ± 0.003 | 0.021 ± 0.001 | 0.035 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 1.058 ± 0.009 | - | 1.078 ± 0.010 | 1.200 ± 0.042 | 1.201 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.042 ± 0.001 | - | 0.050 ± 0.001 | 0.048 ± 0.001 | 0.058 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.115 ± 0.002 | - | 0.126 ± 0.004 | 0.117 ± 0.004 | 0.146 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 2.264 ± 0.033 | - | 2.261 ± 0.022 | 1.052 ± 0.008 | 1.495 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 8.366 ± 0.392 | - | 8.397 ± 3.290 | 4.317 ± 0.009 | 2.240 ± 0.382 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 15.125 ± 0.027 | - | 15.324 ± 0.017 | 16.803 ± 0.165 | 6.403 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 2.637 ± 0.030 | - | 2.687 ± 0.041 | 2.766 ± 0.628 | 3.156 ± 0.060 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.006 ± 0.000 | - | 0.013 ± 0.000 | 0.018 ± 0.001 | 0.032 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.220 ± 0.024 | - | 0.229 ± 0.032 | 0.248 ± 0.016 | 0.220 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.012 ± 0.000 | - | 0.020 ± 0.000 | 0.018 ± 0.000 | 0.033 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.167 ± 0.021 | - | 0.181 ± 0.008 | 0.199 ± 0.022 | 0.210 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.017 ± 0.002 | - | 0.022 ± 0.000 | 0.028 ± 0.000 | 0.039 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.011 ± 0.001 | - | 0.018 ± 0.001 | 0.023 ± 0.004 | 0.028 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.028 ± 0.000 | - | 0.036 ± 0.001 | 0.027 ± 0.001 | 0.038 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.043 ± 0.001 | - | 0.048 ± 0.001 | 0.053 ± 0.001 | 0.062 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.013 ± 0.000 | - | 0.020 ± 0.004 | 0.024 ± 0.004 | 0.031 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.037 ± 0.000 | - | 0.047 ± 0.004 | 0.031 ± 0.000 | 0.043 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.057 ± 0.002 | - | 0.063 ± 0.005 | 0.066 ± 0.001 | 0.076 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 20.921 ± 0.207 | - | 21.188 ± 0.226 | 21.735 ± 0.092 | 24.581 ± 12.067 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.021 ± 0.000 | - | 0.030 ± 0.001 | 0.031 ± 0.005 | 0.036 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 1.652 ± 0.027 | - | 1.667 ± 0.009 | 1.750 ± 0.016 | 0.616 ± 0.129 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.069 ± 0.002 | - | 0.081 ± 0.008 | 0.047 ± 0.002 | 0.063 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.114 ± 0.002 | - | 0.120 ± 0.008 | 0.118 ± 0.004 | 0.151 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.474 ± 0.007 | - | unsupported | 0.353 ± 0.016 | 0.766 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 2.313 ± 0.019 | - | unsupported | 1.938 ± 0.005 | 2.728 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 5.544 ± 0.009 | - | unsupported | 5.307 ± 0.013 | 6.845 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.987 ± 0.018 | - | unsupported | 1.146 ± 0.017 | 1.111 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.003 ± 0.004 | - | unsupported | 0.009 ± 0.001 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.047 ± 0.001 | - | unsupported | 0.061 ± 0.002 | 0.077 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | unsupported | 0.012 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.039 ± 0.000 | - | unsupported | 0.053 ± 0.003 | 0.070 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | unsupported | 0.015 ± 0.001 | 0.027 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | unsupported | 0.015 ± 0.001 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.017 ± 0.000 | - | unsupported | 0.025 ± 0.001 | 0.038 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.011 ± 0.013 | 0.021 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | unsupported | 0.018 ± 0.004 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.023 ± 0.000 | - | unsupported | 0.031 ± 0.001 | 0.045 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 7.686 ± 0.029 | - | unsupported | 9.001 ± 0.052 | 8.626 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | unsupported | 0.012 ± 0.003 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.372 ± 0.010 | - | unsupported | 0.420 ± 0.015 | 0.476 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.019 ± 0.000 | - | unsupported | 0.026 ± 0.008 | 0.046 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.044 ± 0.000 | - | unsupported | 0.052 ± 0.004 | 0.086 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.809 ± 0.008 | - | unsupported | 0.585 ± 0.017 | 0.841 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 3.603 ± 0.021 | - | unsupported | 1.932 ± 0.006 | 1.465 ± 0.248 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 8.373 ± 0.009 | - | unsupported | 7.685 ± 0.024 | 3.573 ± 0.147 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.226 ± 0.025 | - | unsupported | 1.445 ± 0.013 | 1.457 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.004 ± 0.002 | - | unsupported | 0.014 ± 0.001 | 0.016 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.064 ± 0.000 | - | unsupported | 0.090 ± 0.003 | 0.075 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | unsupported | 0.017 ± 0.006 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.055 ± 0.001 | - | unsupported | 0.079 ± 0.005 | 0.091 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.011 ± 0.000 | - | unsupported | 0.016 ± 0.001 | 0.025 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.005 ± 0.000 | - | unsupported | 0.016 ± 0.005 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.013 ± 0.000 | - | unsupported | 0.031 ± 0.010 | 0.026 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.026 ± 0.000 | - | unsupported | 0.027 ± 0.001 | 0.036 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.006 ± 0.001 | - | unsupported | 0.016 ± 0.004 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.016 ± 0.000 | - | unsupported | 0.027 ± 0.007 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.035 ± 0.000 | - | unsupported | 0.032 ± 0.001 | 0.044 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 9.748 ± 0.058 | - | unsupported | 11.213 ± 0.012 | 11.574 ± 0.169 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.009 ± 0.000 | - | unsupported | 0.018 ± 0.004 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.496 ± 0.013 | - | unsupported | 0.425 ± 0.015 | 0.263 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.030 ± 0.000 | - | unsupported | 0.026 ± 0.000 | 0.037 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.067 ± 0.001 | - | unsupported | 0.053 ± 0.001 | 0.075 ± 0.008 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 1.054 ± 0.006 | - | unsupported | 0.953 ± 0.001 | 2.680 ± 0.024 | 2.524 ± 0.017 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.133 ± 0.003 | - | unsupported | 0.059 ± 0.002 | 0.130 ± 0.008 | 0.096 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.138 ± 0.001 | - | unsupported | 0.084 ± 0.001 | 0.206 ± 0.026 | 0.173 ± 0.005 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.336 ± 0.008 | - | unsupported | 0.243 ± 0.013 | 0.538 ± 0.030 | 0.655 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 1.058 ± 0.021 | - | unsupported | 0.500 ± 0.028 | 1.972 ± 0.023 | 2.488 ± 0.782 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.136 ± 0.002 | - | unsupported | 0.063 ± 0.002 | 0.120 ± 0.006 | 0.102 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.142 ± 0.003 | - | unsupported | 0.082 ± 0.003 | 0.200 ± 0.008 | 0.179 ± 0.014 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.338 ± 0.014 | - | unsupported | 0.162 ± 0.002 | 0.555 ± 0.049 | 0.674 ± 0.047 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.309 ± 0.008 | - | unsupported | 0.374 ± 0.017 | 0.804 ± 0.019 | 0.469 ± 0.019 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.034 ± 0.000 | - | unsupported | 0.046 ± 0.004 | 0.227 ± 0.019 | 0.078 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.059 ± 0.000 | - | unsupported | 0.063 ± 0.005 | 0.229 ± 0.015 | 0.127 ± 0.005 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.200 ± 0.003 | - | unsupported | 0.168 ± 0.013 | 0.399 ± 0.029 | 0.233 ± 0.016 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.324 ± 0.015 | - | unsupported | 0.567 ± 0.013 | 0.875 ± 0.064 | 0.413 ± 0.011 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.034 ± 0.000 | - | unsupported | 0.075 ± 0.012 | 0.207 ± 0.009 | 0.074 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.059 ± 0.004 | - | unsupported | 0.099 ± 0.004 | 0.251 ± 0.026 | 0.115 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.199 ± 0.002 | - | unsupported | 0.227 ± 0.013 | 0.411 ± 0.018 | 0.209 ± 0.005 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.207 ± 0.012 | - | 0.290 ± 0.018 | 0.195 ± 0.013 | 0.387 ± 0.016 | 0.302 ± 0.009 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.080 ± 0.000 | - | 0.086 ± 0.001 | 0.074 ± 0.006 | 0.077 ± 0.002 | 0.064 ± 0.004 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.091 ± 0.000 | - | 0.101 ± 0.001 | 0.088 ± 0.004 | 0.102 ± 0.006 | 0.097 ± 0.003 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.121 ± 0.001 | - | 0.151 ± 0.001 | 0.123 ± 0.004 | 0.173 ± 0.019 | 0.164 ± 0.009 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.194 ± 0.025 | - | 0.338 ± 0.024 | 0.260 ± 0.015 | 0.448 ± 0.103 | 0.263 ± 0.013 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.080 ± 0.000 | - | 0.087 ± 0.003 | 0.126 ± 0.002 | 0.080 ± 0.003 | 0.061 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.091 ± 0.001 | - | 0.102 ± 0.004 | 0.143 ± 0.005 | 0.105 ± 0.005 | 0.087 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.123 ± 0.000 | - | 0.199 ± 0.004 | 0.180 ± 0.006 | 0.229 ± 0.009 | 0.141 ± 0.002 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 43.666 ± 0.178 | - | 47.693 ± 0.143 | 60.909 ± 0.205 | 62.844 ± 0.639 | 40.535 ± 2.802 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 20.786 ± 0.177 | - | 34.064 ± 0.187 | 25.539 ± 0.352 | 54.107 ± 4.544 | 35.478 ± 5.070 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 26.448 ± 0.283 | - | 25.319 ± 0.274 | 25.347 ± 0.176 | 21.605 ± 0.109 | 20.394 ± 2.431 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 12.908 ± 0.450 | - | 12.227 ± 0.155 | 11.545 ± 0.186 | 14.291 ± 0.862 | 12.841 ± 4.650 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 7.462 ± 0.022 | - | 7.478 ± 0.014 | 7.968 ± 0.126 | 8.071 ± 0.018 | 7.928 ± 0.408 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 11.386 ± 0.129 | - | 11.341 ± 0.070 | 10.408 ± 0.035 | 11.556 ± 0.095 | 11.940 ± 0.434 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 8.741 ± 0.025 | - | 8.770 ± 0.013 | 8.878 ± 0.159 | 9.625 ± 0.095 | 9.104 ± 0.018 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 13.211 ± 0.332 | - | 13.389 ± 0.088 | 11.810 ± 0.031 | 14.184 ± 0.071 | 13.804 ± 0.129 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 21.066 ± 0.045 | - | 22.056 ± 0.233 | 19.074 ± 0.170 | 39.866 ± 0.125 | 14.558 ± 0.116 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 15.506 ± 0.499 | - | 17.180 ± 0.233 | 14.687 ± 0.065 | 33.966 ± 0.905 | 12.838 ± 0.587 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.490 ± 0.006 | 0.516 ± 0.008 | unsupported | 0.611 ± 0.004 | 0.754 ± 0.023 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 7.852 ± 0.019 | 7.923 ± 0.015 | unsupported | 8.572 ± 0.055 | 9.453 ± 0.020 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.388 ± 0.001 | 0.414 ± 0.002 | unsupported | 0.510 ± 0.009 | 1.293 ± 0.063 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 5.997 ± 0.108 | 6.241 ± 0.142 | unsupported | 5.677 ± 0.033 | 9.261 ± 0.052 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 41.151 ± 0.347 | - | 41.559 ± 0.252 | 42.400 ± 0.259 | 31.489 ± 0.056 | 27.579 ± 0.126 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 24.060 ± 2.659 | - | 21.061 ± 0.117 | 21.568 ± 0.149 | 19.801 ± 0.185 | 18.365 ± 0.371 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 17.824 ± 0.107 | 17.712 ± 0.074 | 9.256 ± 0.118 | 46.678 ± 0.174 | 9.724 ± 0.138 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 11.386 ± 0.341 | 11.046 ± 0.118 | 6.078 ± 0.121 | 41.554 ± 1.947 | 8.837 ± 0.090 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 25.880 ± 0.939 | - | 26.577 ± 0.292 | 36.559 ± 0.390 | 38.578 ± 0.230 | 27.456 ± 10.510 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 13.010 ± 0.409 | - | 13.351 ± 0.438 | 16.127 ± 0.305 | 31.822 ± 3.427 | 22.292 ± 10.282 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 1.220 ± 0.051 | - | 1.036 ± 0.212 | 1.184 ± 0.060 | 5.165 ± 0.101 | 2.693 ± 0.015 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.636 ± 0.043 | - | 0.674 ± 0.057 | 1.181 ± 0.054 | 3.889 ± 0.155 | 2.699 ± 0.005 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 19.927 ± 0.669 | - | 20.902 ± 0.270 | 19.836 ± 0.200 | 19.978 ± 0.088 | 20.045 ± 2.715 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 15.604 ± 0.836 | - | 15.976 ± 0.108 | 16.549 ± 0.163 | 20.235 ± 1.938 | 24.006 ± 0.533 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 20.391 ± 0.545 | - | 20.773 ± 0.148 | 19.746 ± 0.087 | 19.966 ± 0.039 | 19.273 ± 1.411 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 15.979 ± 0.455 | - | 16.118 ± 0.746 | 16.498 ± 0.246 | 21.252 ± 1.398 | 24.183 ± 0.110 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.151 ± 0.007 | 1.178 ± 0.004 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 21.328 ± 0.037 | 21.469 ± 0.028 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.329 ± 0.011 | 1.366 ± 0.019 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 19.643 ± 0.156 | 19.876 ± 0.126 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.079 ± 0.016 | 1.101 ± 0.005 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 20.319 ± 0.072 | 20.458 ± 0.034 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.247 ± 0.060 | 1.262 ± 0.024 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 18.310 ± 0.154 | 18.386 ± 0.340 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 26.523 ± 0.135 | - | 25.817 ± 0.551 | 25.321 ± 0.076 | 21.643 ± 0.394 | 20.303 ± 0.513 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 13.034 ± 0.288 | - | 12.462 ± 3.759 | 11.582 ± 0.201 | 14.181 ± 0.268 | 12.944 ± 0.659 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 59.146 ± 0.783 | 59.763 ± 1.456 | 61.601 ± 0.123 | 67.197 ± 0.674 | 66.175 ± 6.365 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 37.769 ± 1.712 | 37.913 ± 1.232 | 37.625 ± 0.301 | 54.318 ± 2.470 | 51.178 ± 0.737 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 41.728 ± 0.858 | - | 58.424 ± 2.413 | 29.662 ± 0.224 | 80.621 ± 0.639 | 33.350 ± 0.186 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 30.743 ± 0.540 | - | 43.013 ± 1.104 | 42.946 ± 5.525 | 63.808 ± 8.276 | 21.686 ± 0.740 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.139 ± 0.000 | 0.140 ± 0.001 | unsupported | 0.152 ± 0.029 | 0.134 ± 0.003 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.214 ± 0.009 | 2.214 ± 0.005 | unsupported | 2.044 ± 0.032 | 0.416 ± 0.026 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.117 ± 0.009 | 0.123 ± 0.006 | unsupported | 0.076 ± 0.004 | 0.128 ± 0.011 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.906 ± 0.005 | 0.907 ± 0.010 | unsupported | 0.749 ± 0.009 | 0.506 ± 0.377 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.079 ± 0.001 | 0.078 ± 0.001 | unsupported | 0.131 ± 0.012 | 0.257 ± 0.002 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.232 ± 0.001 | 1.242 ± 0.004 | unsupported | 1.358 ± 0.018 | 1.827 ± 0.007 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.062 ± 0.006 | 0.046 ± 0.000 | unsupported | 0.067 ± 0.025 | 0.171 ± 0.002 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.564 ± 0.003 | 0.569 ± 0.014 | unsupported | 0.722 ± 0.010 | 0.945 ± 0.025 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.196 ± 0.000 | 0.195 ± 0.001 | unsupported | 0.278 ± 0.001 | 0.211 ± 0.001 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 3.113 ± 0.003 | 3.116 ± 0.026 | unsupported | 4.005 ± 0.012 | 1.001 ± 0.007 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.084 ± 0.006 | 0.086 ± 0.030 | unsupported | 0.106 ± 0.002 | 0.154 ± 0.013 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 1.182 ± 0.006 | 1.187 ± 0.010 | unsupported | 1.519 ± 0.038 | 0.658 ± 0.009 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.054 ± 0.002 | 0.095 ± 0.005 | unsupported | 0.156 ± 0.002 | 0.183 ± 0.005 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.595 ± 0.007 | 0.694 ± 0.021 | unsupported | 1.336 ± 0.031 | 1.004 ± 0.030 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.150 ± 0.016 | 0.279 ± 0.010 | unsupported | 0.067 ± 0.001 | 0.195 ± 0.007 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.441 ± 0.031 | 0.632 ± 0.019 | unsupported | 0.370 ± 0.004 | 0.790 ± 0.027 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.100 ± 0.002 | 0.132 ± 0.003 | unsupported | 0.132 ± 0.001 | 0.224 ± 0.025 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.265 ± 0.015 | 1.532 ± 0.207 | unsupported | 1.633 ± 0.011 | 1.365 ± 0.022 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.202 ± 0.017 | 0.320 ± 0.008 | unsupported | 0.062 ± 0.002 | 0.227 ± 0.003 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.794 ± 0.015 | 0.976 ± 0.110 | unsupported | 0.485 ± 0.009 | 1.107 ± 0.151 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.139 ± 0.003 | 0.181 ± 0.002 | unsupported | unsupported | 0.247 ± 0.023 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 1.853 ± 0.013 | 2.058 ± 0.133 | unsupported | unsupported | 1.791 ± 0.029 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.295 ± 0.013 | 0.445 ± 0.012 | unsupported | unsupported | 0.247 ± 0.016 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 1.285 ± 0.062 | 1.467 ± 0.077 | unsupported | unsupported | 1.254 ± 0.058 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.011 ± 0.000 | 0.020 ± 0.001 | unsupported | 0.014 ± 0.000 | 0.080 ± 0.000 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.097 ± 0.004 | 0.112 ± 0.004 | unsupported | 0.091 ± 0.007 | 0.235 ± 0.005 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.023 ± 0.001 | 0.033 ± 0.006 | unsupported | 0.015 ± 0.000 | 0.048 ± 0.003 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.084 ± 0.029 | 0.151 ± 0.018 | unsupported | 0.076 ± 0.029 | 0.165 ± 0.020 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.028 ± 0.003 | 0.053 ± 0.004 | unsupported | 0.096 ± 0.001 | 0.168 ± 0.003 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.314 ± 0.019 | 0.417 ± 0.024 | unsupported | 0.993 ± 0.008 | 0.802 ± 0.025 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.081 ± 0.013 | 0.152 ± 0.003 | unsupported | 0.096 ± 0.002 | 0.165 ± 0.011 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.244 ± 0.035 | 0.346 ± 0.041 | unsupported | 0.265 ± 0.044 | 0.645 ± 0.065 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.054 ± 0.001 | 0.054 ± 0.000 | unsupported | 0.107 ± 0.002 | 0.198 ± 0.002 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.850 ± 0.002 | 0.877 ± 0.006 | unsupported | 1.397 ± 0.016 | 0.909 ± 0.003 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.035 ± 0.001 | 0.036 ± 0.001 | unsupported | 0.093 ± 0.003 | 0.144 ± 0.012 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.393 ± 0.001 | 0.394 ± 0.004 | unsupported | 0.701 ± 0.005 | 0.621 ± 0.041 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.056 ± 0.001 | 0.057 ± 0.000 | unsupported | 0.115 ± 0.001 | 0.173 ± 0.003 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.883 ± 0.002 | 0.906 ± 0.006 | unsupported | 1.432 ± 0.018 | 0.927 ± 0.018 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.036 ± 0.001 | 0.036 ± 0.000 | unsupported | 0.097 ± 0.001 | 0.156 ± 0.001 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.409 ± 0.002 | 0.411 ± 0.001 | unsupported | 0.713 ± 0.016 | 0.618 ± 0.023 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.101 ± 0.001 | 0.135 ± 0.005 | unsupported | 0.130 ± 0.002 | 0.241 ± 0.002 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.276 ± 0.018 | 1.443 ± 0.079 | unsupported | 1.231 ± 0.009 | 1.280 ± 0.023 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.208 ± 0.029 | 0.326 ± 0.011 | unsupported | 0.060 ± 0.001 | 0.219 ± 0.002 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 0.817 ± 0.057 | 0.997 ± 0.046 | unsupported | 0.469 ± 0.001 | 0.930 ± 0.030 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.184 ± 0.001 | 0.184 ± 0.000 | unsupported | 0.271 ± 0.001 | 0.340 ± 0.023 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 2.922 ± 0.004 | 2.915 ± 0.015 | unsupported | 3.880 ± 0.012 | 3.147 ± 0.008 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.081 ± 0.002 | 0.082 ± 0.001 | unsupported | 0.198 ± 0.004 | 0.216 ± 0.016 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 1.147 ± 0.203 | 1.138 ± 0.010 | unsupported | 1.558 ± 0.043 | 1.740 ± 0.005 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.757 ± 0.012 | 1.390 ± 0.011 | unsupported | 0.259 ± 0.002 | 0.214 ± 0.009 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 15.664 ± 0.088 | 19.746 ± 0.094 | unsupported | 4.094 ± 0.034 | 1.458 ± 0.016 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.636 ± 0.026 | 1.333 ± 0.008 | unsupported | 0.259 ± 0.002 | 0.163 ± 0.004 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 8.306 ± 0.239 | 13.550 ± 0.175 | unsupported | 2.109 ± 0.015 | 0.905 ± 0.029 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 26.176 ± 0.567 | - | unsupported | 24.906 ± 0.388 | - | 25.405 ± 0.219 | 27.969 ± 0.213 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 21.807 ± 0.251 | - | unsupported | 21.584 ± 0.472 | - | 27.580 ± 0.785 | 23.674 ± 0.891 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 18.164 ± 0.572 | - | unsupported | 17.503 ± 0.259 | - | 16.905 ± 0.250 | 17.383 ± 0.392 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 14.961 ± 0.237 | - | unsupported | 15.394 ± 0.589 | - | 18.274 ± 0.431 | 15.773 ± 0.517 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 13.158 ± 0.322 | - | unsupported | 16.901 ± 0.259 | - | 12.368 ± 0.432 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 15.091 ± 0.598 | - | unsupported | 15.252 ± 0.581 | - | 12.884 ± 0.214 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 26.083 ± 0.496 | - | unsupported | 25.459 ± 1.238 | - | 25.353 ± 0.435 | 27.854 ± 0.109 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 21.350 ± 0.201 | - | unsupported | 22.018 ± 0.372 | - | 26.615 ± 0.795 | 22.380 ± 0.447 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 49.053 ± 0.291 | - | unsupported | 51.614 ± 0.320 | - | 48.039 ± 0.091 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 24.523 ± 0.108 | - | unsupported | 26.268 ± 0.071 | - | 23.613 ± 0.227 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 48.947 ± 0.269 | - | unsupported | 50.984 ± 0.234 | - | 47.540 ± 0.280 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 24.434 ± 0.087 | - | unsupported | 26.268 ± 0.167 | - | 23.379 ± 0.105 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 26.726 ± 0.649 | - | unsupported | 24.696 ± 0.516 | - | 26.365 ± 0.400 | 28.074 ± 0.270 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 21.655 ± 0.086 | - | unsupported | 21.937 ± 0.204 | - | 26.376 ± 0.797 | 23.012 ± 0.674 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 18.033 ± 0.158 | - | unsupported | 16.847 ± 0.148 | - | 17.511 ± 0.196 | 22.354 ± 0.286 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 14.905 ± 0.335 | - | unsupported | 15.448 ± 0.366 | - | 18.629 ± 0.318 | 15.649 ± 0.178 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.747 ± 0.002 | unsupported | unsupported | 0.901 ± 0.005 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 10.976 ± 0.019 | unsupported | unsupported | 12.712 ± 0.011 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.503 ± 0.012 | unsupported | unsupported | 0.616 ± 0.003 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 5.985 ± 0.055 | unsupported | unsupported | 6.652 ± 0.032 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 26.508 ± 0.340 | - | unsupported | 24.614 ± 0.287 | - | 25.779 ± 0.689 | 28.040 ± 0.365 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 21.760 ± 0.274 | - | unsupported | 21.620 ± 0.209 | - | 26.772 ± 0.282 | 23.207 ± 0.521 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 21.256 ± 0.214 | - | 0.003 ± 0.000 | 18.667 ± 0.248 | 17.894 ± 0.096 | 26.242 ± 0.180 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 16.549 ± 0.064 | - | 0.003 ± 0.000 | 14.193 ± 0.118 | 14.798 ± 0.217 | 27.187 ± 0.382 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 18.187 ± 0.679 | - | 44.682 ± 0.279 | 17.237 ± 0.160 | 17.441 ± 0.190 | 18.106 ± 0.282 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 12.052 ± 0.085 | - | 30.180 ± 0.746 | 12.127 ± 0.595 | 12.842 ± 0.686 | 19.538 ± 0.912 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 39.864 ± 0.325 | - | 39.521 ± 0.566 | 34.817 ± 0.266 | 37.723 ± 0.095 | 36.652 ± 0.496 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 22.764 ± 2.360 | - | 23.742 ± 1.713 | 22.715 ± 0.570 | 25.113 ± 0.987 | 38.642 ± 0.435 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 13.549 ± 0.179 | - | 13.471 ± 0.079 | 20.625 ± 0.166 | 16.618 ± 0.234 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 10.325 ± 0.538 | - | 10.368 ± 0.834 | 12.175 ± 0.408 | 12.242 ± 0.508 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 26.336 ± 0.241 | - | 0.003 ± 0.000 | 24.624 ± 0.186 | 23.743 ± 0.285 | 27.214 ± 0.308 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 16.390 ± 0.272 | - | 0.003 ± 0.000 | 16.482 ± 0.466 | 19.154 ± 0.763 | 29.100 ± 1.175 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 49.311 ± 0.848 | - | 0.003 ± 0.000 | 32.302 ± 0.258 | 44.860 ± 0.931 | 892.837 ± 7.336 | 641.981 ± 0.377 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 20.131 ± 0.555 | - | 0.003 ± 0.000 | 32.089 ± 0.529 | 24.948 ± 0.573 | 893.120 ± 5.666 | 106.186 ± 3.076 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 12.499 ± 0.197 | - | 13.106 ± 0.087 | 15.364 ± 0.144 | 12.703 ± 0.200 | 17.485 ± 0.523 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 7.431 ± 0.494 | - | 7.534 ± 0.261 | 9.984 ± 0.319 | 9.156 ± 0.340 | 17.866 ± 0.497 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 12.337 ± 0.253 | - | 12.694 ± 0.331 | 16.971 ± 0.105 | 12.726 ± 0.071 | 17.704 ± 0.337 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 8.275 ± 1.049 | - | 8.319 ± 0.681 | 9.804 ± 0.088 | 9.210 ± 0.162 | 17.916 ± 0.734 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 5.976148 | 373509.25 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 4.989619 | 311851.19 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 5.645864 | 352866.50 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 5.046985 | 315436.56 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 6.581441 | 411340.06 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 8.918008 | 557375.50 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 19.765306 | 1235331.62 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 18.117999 | 1132374.94 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 6.019952 | 376247.00 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 5.954049 | 372128.06 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 12.068617 | 754288.56 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 9.771315 | 610707.19 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 7.841934 | 490120.88 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 8.256859 | 516053.69 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 151.247155 | 9452947.19 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 137.146522 | 8571657.62 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 125.626695 | 7851668.44 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 126.762507 | 7922656.69 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 20.688412 | 1293025.75 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 8.154953 | 509684.56 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 6.214127 | 388382.94 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 6.617241 | 413577.56 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 148.179212 | 9261200.75 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 90.836071 | 5677254.44 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 95.952718 | 5997044.88 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 99.861816 | 6241363.50 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 18.409041 | 1150565.06 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 18.849253 | 1178078.31 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 341.254793 | 21328424.56 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 343.497573 | 21468598.31 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 21.269422 | 1329338.88 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 21.863773 | 1366485.81 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 314.294225 | 19643389.06 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 318.013015 | 19875813.44 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 17.265283 | 1079080.19 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 17.615518 | 1100969.88 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 325.100458 | 20318778.62 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 327.334758 | 20458422.38 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 19.952250 | 1247015.62 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 20.198631 | 1262414.44 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 292.953154 | 18309572.12 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 294.180137 | 18386258.56 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.552707 | 133636.05 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 9.736914 | 152139.28 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.902581 | 139102.83 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.929281 | 139520.02 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 3.330666 | 416333.25 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 16.348501 | 2043562.62 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 17.712258 | 2214032.25 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 17.715244 | 2214405.50 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.193286 | 128020.09 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.843403 | 75678.17 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 7.488940 | 117014.69 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.856760 | 122761.88 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.051644 | 506455.50 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.994936 | 749367.00 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.245926 | 905740.75 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.257908 | 907238.50 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 16.459977 | 257187.14 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.412476 | 131444.94 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.054763 | 78980.67 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.007084 | 78235.69 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.615520 | 1826940.00 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.867929 | 1358491.12 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.859338 | 1232417.25 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 9.937615 | 1242201.88 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.945841 | 171028.77 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.295490 | 67117.03 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.937761 | 61527.52 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.936539 | 45883.42 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.561575 | 945196.88 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.772409 | 721551.12 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.513408 | 564176.00 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.549494 | 568686.75 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.512498 | 211132.78 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.765361 | 277583.77 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 12.532181 | 195815.33 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 12.508608 | 195447.00 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 8.008740 | 1001092.50 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 32.040014 | 4005001.75 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 24.904825 | 3113103.12 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 24.925433 | 3115679.12 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.875440 | 154303.75 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.757311 | 105582.98 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.353739 | 83652.17 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.491466 | 85804.16 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.264821 | 658102.62 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.154690 | 1519336.25 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.459924 | 1182490.50 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 9.494419 | 1186802.38 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.692014 | 182687.72 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 9.981211 | 155956.42 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.452350 | 53942.97 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.052446 | 94569.47 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 8.030290 | 1003786.25 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.690612 | 1336326.50 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.760754 | 595094.25 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.548315 | 693539.38 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.449982 | 194530.97 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.297944 | 67155.38 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 9.624862 | 150388.47 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 17.825049 | 278516.39 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.316357 | 789544.62 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.963880 | 370485.00 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.529649 | 441206.12 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.052806 | 631600.75 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.362818 | 224419.03 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.463602 | 132243.78 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.426595 | 100415.55 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.445918 | 131967.47 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.918660 | 1364832.50 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 13.065067 | 1633133.38 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.116729 | 1264591.12 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 12.255495 | 1531936.88 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 14.505335 | 226645.86 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.940075 | 61563.67 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 12.955456 | 202429.00 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 20.490806 | 320168.84 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.855674 | 1106959.25 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.883710 | 485463.75 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.351333 | 793916.62 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.810731 | 976341.38 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 15.814130 | 247095.78 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.921436 | 139397.44 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 11.583910 | 180998.59 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.329575 | 1791196.88 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 14.821158 | 1852644.75 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 16.465519 | 2058189.88 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 15.787464 | 246679.12 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 18.882701 | 295042.20 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 28.503624 | 445369.12 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.030550 | 1253818.75 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 10.280151 | 1285018.88 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 11.735137 | 1466892.12 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.093350 | 79583.59 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.884565 | 13821.33 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.731396 | 11428.06 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.302864 | 20357.25 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 1.881407 | 235175.88 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.727301 | 90912.62 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.778554 | 97319.25 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.896535 | 112066.88 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 3.044632 | 47572.38 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.935188 | 14612.31 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.444380 | 22568.44 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.125464 | 33210.38 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.319527 | 164940.88 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.609159 | 76144.88 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.668620 | 83577.50 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.210442 | 151305.25 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.753100 | 168017.19 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.133253 | 95832.08 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.820610 | 28447.03 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.376017 | 52750.27 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.412056 | 801507.00 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.940394 | 992549.25 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.511913 | 313989.12 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.339288 | 417411.00 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.550602 | 164853.16 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.122965 | 95671.33 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.156607 | 80571.98 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 9.700935 | 151577.11 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.159305 | 644913.12 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.117578 | 264697.25 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.950656 | 243832.00 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.764598 | 345574.75 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.678499 | 198101.55 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.878677 | 107479.33 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.459914 | 54061.16 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.479009 | 54359.52 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.274718 | 909339.75 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.175365 | 1396920.62 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.797168 | 849646.00 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.013241 | 876655.12 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.201700 | 143776.56 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.961723 | 93151.92 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.237683 | 34963.80 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.284590 | 35696.72 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.968438 | 621054.75 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.610948 | 701368.50 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.146382 | 393297.75 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.155299 | 394412.38 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.053745 | 172714.77 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.381037 | 115328.70 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.598963 | 56233.80 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.655418 | 57115.91 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.419639 | 927454.88 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.455449 | 1431931.12 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.063124 | 882890.50 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.246728 | 905841.00 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.007006 | 156359.47 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.228573 | 97321.45 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.295151 | 35861.73 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.315809 | 36184.52 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.943610 | 617951.25 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.701327 | 712665.88 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.274191 | 409273.88 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.287265 | 410908.12 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 15.406117 | 240720.58 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.340742 | 130324.09 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.442024 | 100656.62 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.665738 | 135402.16 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.239721 | 1279965.12 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.847150 | 1230893.75 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.204694 | 1275586.75 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.547402 | 1443425.25 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 14.033423 | 219272.23 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.825531 | 59773.92 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 13.340476 | 208444.94 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 20.845676 | 325713.69 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.441350 | 930168.75 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.754257 | 469282.12 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.534495 | 816811.88 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.977032 | 997129.00 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 21.762349 | 340036.70 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.313055 | 270516.48 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 11.752115 | 183626.80 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 11.754469 | 183663.58 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 25.176300 | 3147037.50 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 31.036847 | 3879605.88 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 23.379385 | 2922423.12 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 23.318181 | 2914772.62 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.827117 | 216048.70 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 12.683207 | 198175.11 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.169745 | 80777.27 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.217885 | 81529.45 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 13.923176 | 1740397.00 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.465911 | 1558238.88 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.176445 | 1147055.62 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 9.100242 | 1137530.25 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.700991 | 214077.98 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 16.587589 | 259181.08 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 48.426796 | 756668.69 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 88.969616 | 1390150.25 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.667669 | 1458458.62 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 32.749560 | 4093695.00 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 125.310063 | 15663757.88 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 157.968817 | 19746102.12 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.451497 | 163304.64 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 16.605459 | 259460.30 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 40.703220 | 635987.81 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 85.335155 | 1333361.80 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.236968 | 904621.00 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 16.869413 | 2108676.62 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 66.450671 | 8306333.88 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 108.398988 | 13549873.50 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 14.423059 | 901441.19 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 11.946648 | 746665.50 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 203.390877 | 12711929.81 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 175.612714 | 10975794.62 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 9.855913 | 615994.56 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 8.054975 | 503435.94 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 106.438574 | 6652410.88 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 95.764737 | 5985296.06 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.024346 | 1521.62 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002434 | 152.12 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.039944 | 2496.50 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002515 | 157.19 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000130 | 8.12 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.022442 | 1402.62 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.007714 | 482.12 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000201 | 12.56 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.036088 | 2255.50 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.007745 | 484.06 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000120 | 7.50 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.022833 | 1427.06 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.005501 | 343.81 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000190 | 11.88 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.037620 | 2351.25 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.005400 | 337.50 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000121 | 7.56 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.020599 | 1287.44 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003657 | 228.56 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000101 | 6.31 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.033272 | 2079.50 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003326 | 207.88 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 13.3x faster than the slowest successful cell (`tenferro-trace`, 17.138 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 37.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 30.2x faster than the slowest successful cell (`pytorch-cpu`, 37.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 21.4x faster than the slowest successful cell (`pytorch-cpu`, 37.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 20.6x faster than the slowest successful cell (`pytorch-cpu`, 37.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 12.150 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 12.150 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 33.356 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 28.0x faster than the slowest successful cell (`pytorch-cpu`, 33.356 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 19.7x faster than the slowest successful cell (`pytorch-cpu`, 33.356 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 19.0x faster than the slowest successful cell (`pytorch-cpu`, 33.356 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 14.1x faster than the slowest successful cell (`pytorch-cpu`, 11.237 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 14.1x faster than the slowest successful cell (`pytorch-cpu`, 11.237 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 10.3x faster than the slowest successful cell (`julia-base`, 2.663 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 1293.6x faster than the slowest successful cell (`julia-base`, 3.577 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 1275.8x faster than the slowest successful cell (`julia-base`, 3.630 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 12.9x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 19.3x faster than the slowest successful cell (`jax-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 13.1x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x8x8`): `tenferro-direct` is 11.3x faster than the slowest successful cell (`jax-cpu`, 0.013 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`3x2x2`): `tenferro-direct` is 11.7x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`3x4x4`): `tenferro-direct` is 12.4x faster than the slowest successful cell (`jax-cpu`, 0.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x2x2`): `tenferro-direct` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.011 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x4x4`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.013 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `pytorch-cpu` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 35.9x faster than the slowest successful cell (`jax-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`jax-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 24.7x faster than the slowest successful cell (`jax-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 13.2x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 13.6x faster than the slowest successful cell (`jax-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 14.2x faster than the slowest successful cell (`jax-cpu`, 0.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 11.9x faster than the slowest successful cell (`jax-cpu`, 0.022 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.022 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 13.0x faster than the slowest successful cell (`jax-cpu`, 0.025 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`3x4x4,rhs=1`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.022 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x2x2,rhs=1`): `tenferro-direct` is 15.0x faster than the slowest successful cell (`pytorch-cpu`, 0.027 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x4x4,rhs=1`): `tenferro-direct` is 14.6x faster than the slowest successful cell (`pytorch-cpu`, 0.026 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`3x4x4,rhs=1`): `tenferro-direct` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 0.027 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`4x4x4,rhs=1`): `tenferro-direct` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.027 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 13.5x faster than the slowest successful cell (`tenferro-eager`, 19.746 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 15.0x faster than the slowest successful cell (`tenferro-eager`, 13.550 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 9352.0x faster than the slowest successful cell (`julia-base`, 26.242 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 9977.1x faster than the slowest successful cell (`julia-base`, 27.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 9364.7x faster than the slowest successful cell (`julia-base`, 27.214 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 10411.5x faster than the slowest successful cell (`julia-base`, 29.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `jax-cpu` is 19.9x faster than the slowest successful cell (`julia-base`, 892.837 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `pytorch-cpu` is 27.6x faster than the slowest successful cell (`julia-base`, 892.837 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-direct` is 18.1x faster than the slowest successful cell (`julia-base`, 892.837 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 332527.6x faster than the slowest successful cell (`julia-base`, 892.837 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `jax-cpu` is 35.8x faster than the slowest successful cell (`julia-base`, 893.120 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `pytorch-cpu` is 27.8x faster than the slowest successful cell (`julia-base`, 893.120 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-direct` is 44.4x faster than the slowest successful cell (`julia-base`, 893.120 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 338945.1x faster than the slowest successful cell (`julia-base`, 893.120 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 10.0x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 175.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 173.5x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 178.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 195.9x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 160.9x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 346.7x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 6064.7 GB/s at 0.003 ms. The conservative physical lower bound is 0.168 ms, so the cell is 60.6x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 5894.2 GB/s at 0.003 ms. The conservative physical lower bound is 0.042 ms, so the cell is 14.7x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 95664.8 GB/s at 0.003 ms. The conservative physical lower bound is 2.684 ms, so the cell is 956.6x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 98508.4 GB/s at 0.003 ms. The conservative physical lower bound is 0.671 ms, so the cell is 246.3x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 92372.8 GB/s at 0.003 ms. The conservative physical lower bound is 2.684 ms, so the cell is 923.7x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 96041.3 GB/s at 0.003 ms. The conservative physical lower bound is 0.671 ms, so the cell is 240.1x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 49988.0 GB/s at 0.003 ms. The conservative physical lower bound is 1.342 ms, so the cell is 499.9x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 50936.5 GB/s at 0.003 ms. The conservative physical lower bound is 0.336 ms, so the cell is 127.3x faster than the bound.

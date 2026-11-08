# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/amd-cpu/cpu/public_api/20261108_191828/run.yaml`
- Timestamp: `2026-11-08T19:18:28.042560+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/amd-cpu/cpu/public_api/20261108_191828`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

## CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
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

- CSV: `data/results/amd-cpu/cpu/public_api/20261108_191828/cpu_public_api_t1_20261108_191828.csv`
- CSV: `data/results/amd-cpu/cpu/public_api/20261108_191828/cpu_public_api_t4_20261108_191828.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 4.180 ± 0.042 | - | 4.070 ± 0.030 | 4.500 ± 0.063 | 3.919 ± 0.045 | 3.716 ± 2.168 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 4.597 ± 0.056 | - | 4.348 ± 0.030 | 2.262 ± 0.049 | 3.994 ± 0.933 | 3.584 ± 0.755 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 124.441 ± 3.185 | - | 124.317 ± 2.818 | 123.983 ± 3.739 | 125.591 ± 1.274 | 141.039 ± 4.043 | 144.516 ± 3.405 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 41.035 ± 0.425 | - | 143.033 ± 1.861 | 41.668 ± 1.259 | 66.522 ± 2.627 | 143.536 ± 4.369 | 61.934 ± 4.628 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 107.154 ± 0.676 | - | 111.205 ± 2.099 | 92.946 ± 2.018 | 90.178 ± 3.935 | 124.302 ± 1.472 | 132.728 ± 2.363 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 42.528 ± 0.233 | - | 118.843 ± 13.514 | 31.042 ± 0.626 | 44.594 ± 0.713 | 125.831 ± 0.826 | 63.057 ± 1.585 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 43.859 ± 0.336 | - | 43.705 ± 0.367 | 42.965 ± 0.462 | 66.743 ± 0.512 | 46.421 ± 0.201 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 45.190 ± 0.334 | - | 44.820 ± 0.543 | 29.871 ± 0.343 | 29.448 ± 0.363 | 23.316 ± 0.787 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 44.597 ± 0.121 | - | unsupported | 43.339 ± 0.231 | 67.420 ± 0.618 | 44.380 ± 0.409 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 46.149 ± 1.056 | - | unsupported | 30.164 ± 0.224 | 30.844 ± 1.026 | 23.196 ± 4.219 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 10.891 ± 0.097 | - | 10.923 ± 0.120 | 11.659 ± 0.222 | 11.864 ± 0.044 | 11.926 ± 0.235 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 14.252 ± 0.087 | - | 14.241 ± 0.080 | 14.159 ± 0.188 | 17.486 ± 0.125 | 17.613 ± 0.034 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 71.037 ± 0.058 | - | 72.187 ± 0.145 | 108.512 ± 0.244 | 104.511 ± 2.852 | 82.582 ± 1.119 | 83.188 ± 8.838 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 31.055 ± 0.412 | - | 83.992 ± 0.021 | 43.011 ± 0.133 | 52.904 ± 1.528 | 83.332 ± 0.597 | 38.089 ± 9.708 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 116.161 ± 1.822 | - | 116.497 ± 0.062 | 156.684 ± 0.271 | 177.008 ± 3.449 | 154.646 ± 2.985 | 150.984 ± 0.401 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 49.519 ± 0.478 | - | 131.996 ± 0.015 | 63.168 ± 0.389 | 72.655 ± 1.311 | 222.394 ± 1.180 | 63.530 ± 2.167 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 73.233 ± 1.356 | - | 71.092 ± 3.088 | 68.337 ± 1.969 | 72.248 ± 0.170 | 76.744 ± 2.684 | 77.142 ± 2.434 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 24.381 ± 0.128 | - | 94.212 ± 0.566 | 23.051 ± 0.806 | 36.034 ± 0.911 | 76.957 ± 2.479 | 34.767 ± 0.462 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 2.271 ± 0.136 | - | 20.737 ± 0.161 | 24.169 ± 0.161 | 16.693 ± 0.435 | 4.416 ± 0.076 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 1.225 ± 0.094 | - | 60.000 ± 10.994 | 11.291 ± 0.077 | 2.580 ± 0.207 | 4.135 ± 0.043 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 8.691 ± 0.035 | - | 8.623 ± 0.031 | 9.925 ± 0.042 | 7.558 ± 0.048 | 8.978 ± 0.081 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 140.005 ± 11.627 | - | 127.939 ± 23.856 | 5.743 ± 0.047 | 6.919 ± 0.377 | 7.376 ± 0.215 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 5.462 ± 0.193 | - | 5.670 ± 0.291 | 5.473 ± 0.124 | 5.204 ± 0.066 | 4.789 ± 0.042 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 101.020 ± 13.222 | - | 146.027 ± 12.046 | 3.597 ± 0.056 | 4.629 ± 0.975 | 4.309 ± 0.170 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 6.466 ± 0.049 | - | 6.440 ± 0.175 | 7.619 ± 0.053 | 5.397 ± 0.019 | 5.391 ± 0.098 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 11.629 ± 0.156 | - | 11.630 ± 0.322 | 5.883 ± 0.281 | 6.067 ± 0.101 | 6.180 ± 0.597 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 43.699 ± 0.270 | - | 46.197 ± 3.660 | 42.820 ± 0.267 | 66.698 ± 0.166 | 43.994 ± 0.127 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 45.759 ± 1.846 | - | 49.758 ± 10.577 | 29.728 ± 0.301 | 29.741 ± 0.762 | 22.341 ± 0.146 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 49.698 ± 0.791 | - | unsupported | 50.109 ± 0.133 | 62.205 ± 0.619 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 54.651 ± 0.471 | - | unsupported | 25.963 ± 0.503 | 34.885 ± 0.112 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 122.589 ± 5.046 | - | 121.207 ± 4.233 | 122.518 ± 4.159 | 125.576 ± 2.700 | 143.733 ± 6.592 | 153.024 ± 3.985 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 41.156 ± 0.368 | - | 143.332 ± 1.656 | 40.635 ± 1.563 | 65.079 ± 0.254 | 138.349 ± 3.264 | 66.225 ± 1.195 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 132.100 ± 3.871 | - | 131.004 ± 4.840 | 134.105 ± 1.510 | 138.659 ± 3.177 | 149.447 ± 4.900 | 165.467 ± 1.887 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 45.277 ± 4.898 | - | 144.074 ± 9.681 | 45.199 ± 1.626 | 71.602 ± 3.103 | 147.844 ± 6.193 | 71.016 ± 1.427 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 27.230 ± 0.429 | - | 58.322 ± 0.451 | 76.361 ± 0.689 | 51.329 ± 0.897 | 80.141 ± 8.708 | 86.063 ± 1.380 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 12.390 ± 0.311 | - | 119.996 ± 0.982 | 33.079 ± 0.950 | 26.055 ± 0.456 | 80.041 ± 0.380 | 32.375 ± 11.413 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 10.387 ± 0.141 | - | 9.818 ± 0.551 | 37.923 ± 5.848 | 39.685 ± 1.242 | 43.751 ± 8.923 | 41.382 ± 11.491 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 6.817 ± 0.299 | - | 23.998 ± 0.004 | 12.598 ± 0.717 | 19.649 ± 0.162 | 45.405 ± 11.407 | 14.461 ± 21.500 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 23.085 ± 0.167 | - | 23.282 ± 0.575 | 44.253 ± 0.908 | 44.493 ± 5.064 | 23.397 ± 0.926 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 14.038 ± 0.441 | - | 35.999 ± 0.004 | 20.168 ± 0.625 | 24.617 ± 0.813 | 21.599 ± 0.400 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 28.275 ± 0.247 | - | 28.444 ± 0.178 | 55.345 ± 1.196 | 107.158 ± 0.996 | 68.401 ± 14.229 | 70.443 ± 13.400 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 11.880 ± 0.243 | - | 47.998 ± 0.003 | 21.799 ± 0.143 | 54.082 ± 0.143 | 68.468 ± 20.351 | 41.062 ± 12.408 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 134.832 ± 4.240 | - | 133.365 ± 2.739 | 130.679 ± 4.589 | 137.339 ± 2.985 | 150.825 ± 4.456 | 163.969 ± 2.931 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 45.101 ± 0.726 | - | 153.742 ± 12.455 | 72.726 ± 4.473 | 74.663 ± 1.967 | 148.265 ± 3.910 | 68.306 ± 2.454 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 20.460 ± 0.084 | - | 20.770 ± 0.037 | 43.376 ± 0.531 | 41.085 ± 0.819 | 62.915 ± 12.554 | 67.414 ± 12.769 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 8.432 ± 0.029 | - | 35.999 ± 0.003 | 16.509 ± 0.321 | 21.402 ± 0.506 | 71.735 ± 10.237 | 37.810 ± 11.048 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 20.383 ± 0.234 | - | 20.385 ± 3.692 | 31.831 ± 0.543 | 27.716 ± 0.313 | 37.693 ± 9.460 | 40.933 ± 0.137 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 8.082 ± 0.016 | - | 35.998 ± 0.005 | 11.280 ± 0.048 | 14.326 ± 0.238 | 37.169 ± 0.435 | 15.999 ± 10.361 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 25.711 ± 0.072 | - | 25.316 ± 0.036 | 49.452 ± 0.202 | 65.168 ± 0.383 | 82.951 ± 18.899 | 96.440 ± 8.517 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 10.931 ± 0.032 | - | 48.000 ± 0.009 | 18.825 ± 0.242 | 33.039 ± 0.755 | 83.499 ± 13.619 | 36.807 ± 14.006 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 13.188 ± 0.022 | - | 13.324 ± 0.048 | 32.504 ± 0.525 | 37.061 ± 0.437 | 46.240 ± 8.137 | 49.613 ± 1.171 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 5.824 ± 0.013 | - | 35.999 ± 0.006 | 11.578 ± 0.377 | 18.869 ± 0.303 | 46.291 ± 0.724 | 20.666 ± 11.363 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.449 ± 0.015 | 0.495 ± 0.024 | unsupported | unsupported | 0.501 ± 0.021 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.385 ± 0.005 | 51.749 ± 0.502 | unsupported | unsupported | 0.376 ± 0.012 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 140.690 ± 8.937 | - | 139.009 ± 6.074 | 129.203 ± 2.892 | 137.874 ± 1.164 | 158.089 ± 2.602 | 168.557 ± 3.293 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 45.993 ± 1.095 | - | 155.354 ± 2.199 | 45.416 ± 1.116 | 72.444 ± 2.083 | 151.138 ± 5.081 | 70.956 ± 1.027 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 137.949 ± 5.673 | - | 134.639 ± 4.489 | 132.409 ± 2.107 | 140.159 ± 1.577 | 153.167 ± 4.782 | 168.800 ± 1.562 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 45.433 ± 1.236 | - | 154.274 ± 2.391 | 46.122 ± 1.024 | 72.781 ± 0.916 | 155.569 ± 4.143 | 99.612 ± 6.896 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 135.066 ± 4.630 | - | 135.349 ± 4.947 | 130.217 ± 3.613 | 139.009 ± 3.123 | 150.368 ± 7.304 | 167.408 ± 4.961 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 46.081 ± 3.580 | - | 143.787 ± 10.524 | 44.997 ± 0.936 | 73.969 ± 2.408 | 149.776 ± 1.715 | 69.793 ± 1.782 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 124.685 ± 4.497 | - | 124.085 ± 2.721 | 121.695 ± 0.634 | 124.160 ± 1.692 | 142.126 ± 4.721 | 154.329 ± 2.154 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 40.573 ± 0.345 | - | 141.181 ± 3.298 | 40.915 ± 2.524 | 65.355 ± 1.721 | 138.495 ± 4.704 | 66.250 ± 2.580 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 57.689 ± 0.083 | - | 58.196 ± 0.441 | 62.373 ± 0.527 | 73.970 ± 0.236 | 125.880 ± 10.695 | 122.700 ± 0.925 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 23.966 ± 0.430 | - | 71.999 ± 0.012 | 21.978 ± 0.084 | 37.316 ± 0.331 | 124.982 ± 1.021 | 49.792 ± 10.265 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 16.946 ± 0.611 | - | 16.965 ± 1.794 | 12.654 ± 0.429 | 26.071 ± 0.849 | 13.766 ± 1.662 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 7.628 ± 0.598 | - | 35.997 ± 0.007 | 6.360 ± 0.050 | 14.845 ± 0.879 | 14.549 ± 0.547 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.891 ± 0.035 | - | 2.062 ± 0.110 | 23.940 ± 0.278 | 2.491 ± 0.032 | 1.476 ± 0.191 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.798 ± 0.014 | - | 24.001 ± 0.018 | 11.123 ± 0.919 | 1.819 ± 0.562 | 1.519 ± 0.238 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 2.592 ± 0.083 | - | 2.607 ± 0.062 | 4.065 ± 0.016 | 2.950 ± 0.214 | 1.840 ± 0.643 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 1.335 ± 0.057 | - | 23.998 ± 0.007 | 2.309 ± 0.010 | 1.602 ± 0.052 | 1.402 ± 0.426 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 16.922 ± 0.520 | - | 17.569 ± 1.782 | 12.101 ± 0.130 | 26.308 ± 0.569 | 13.339 ± 0.405 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 6.812 ± 0.219 | - | 35.998 ± 0.004 | 6.378 ± 0.495 | 14.343 ± 0.210 | 15.202 ± 2.231 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 2.071 ± 0.262 | - | 1.928 ± 0.058 | 23.020 ± 0.252 | 2.488 ± 0.032 | 1.349 ± 0.121 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.797 ± 0.022 | - | 24.000 ± 0.006 | 10.978 ± 0.845 | 1.511 ± 0.013 | 1.292 ± 0.322 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 10.336 ± 0.265 | - | 11.675 ± 1.876 | 16.297 ± 0.139 | 13.499 ± 0.925 | 7.590 ± 0.947 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 5.186 ± 0.130 | - | 36.001 ± 1.000 | 8.809 ± 0.110 | 7.761 ± 0.325 | 7.472 ± 0.282 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 12.755 ± 0.774 | - | 12.579 ± 0.373 | 11.245 ± 0.240 | 20.048 ± 0.540 | 12.444 ± 1.481 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 6.492 ± 0.319 | - | 35.996 ± 0.010 | 6.748 ± 0.185 | 11.106 ± 0.076 | 12.167 ± 1.561 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 1.070 ± 0.067 | - | 1.198 ± 0.146 | 2.110 ± 0.053 | 1.995 ± 0.032 | 1.017 ± 0.057 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.569 ± 0.032 | - | 23.998 ± 0.005 | 1.046 ± 0.028 | 1.221 ± 0.024 | 1.281 ± 0.269 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.258 ± 0.243 | - | 1.254 ± 0.281 | 0.796 ± 0.036 | 3.156 ± 0.321 | 0.981 ± 0.101 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.821 ± 0.090 | - | 23.998 ± 0.003 | 0.388 ± 0.027 | 1.716 ± 0.085 | 1.320 ± 0.150 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 11.265 ± 0.282 | - | 11.905 ± 0.428 | 11.290 ± 0.329 | 20.351 ± 0.472 | 11.240 ± 0.393 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 6.174 ± 0.276 | - | 35.998 ± 12.003 | 6.453 ± 0.492 | 11.606 ± 0.432 | 11.407 ± 1.301 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 1.026 ± 0.010 | - | 1.067 ± 0.097 | 2.130 ± 0.057 | 2.027 ± 0.036 | 0.951 ± 0.125 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.533 ± 0.299 | - | 23.998 ± 0.006 | 1.042 ± 0.056 | 1.232 ± 0.042 | 1.263 ± 0.189 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.203 ± 0.095 | - | 1.247 ± 0.097 | 0.860 ± 0.071 | 3.682 ± 0.512 | 1.035 ± 0.269 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.735 ± 0.095 | - | 24.000 ± 0.012 | 0.440 ± 0.042 | 1.464 ± 0.108 | 1.289 ± 0.293 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 1.171 ± 0.095 | 1.237 ± 0.049 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.407 ± 0.066 | 23.999 ± 0.001 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 37.782 ± 0.281 | - | 36.678 ± 0.172 | 54.416 ± 1.097 | 72.892 ± 2.869 | 77.392 ± 9.797 | 86.370 ± 12.023 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 17.698 ± 0.079 | - | 60.002 ± 2.986 | 20.555 ± 19.087 | 41.240 ± 4.555 | 71.127 ± 7.286 | 43.104 ± 10.687 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 143.953 ± 1.876 | - | 142.421 ± 3.810 | 143.477 ± 3.987 | 247.606 ± 5.025 | 262.383 ± 5.241 | 264.033 ± 1.958 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 46.394 ± 0.757 | - | 165.142 ± 11.693 | 46.137 ± 0.635 | 124.067 ± 2.076 | 266.209 ± 3.142 | 105.020 ± 5.034 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 138.505 ± 2.991 | - | 136.539 ± 3.372 | 144.955 ± 6.782 | 156.467 ± 5.894 | 155.115 ± 7.137 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 47.706 ± 0.533 | - | 156.114 ± 2.193 | 53.006 ± 1.081 | 79.912 ± 0.651 | 151.261 ± 1.127 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 128.674 ± 3.476 | - | 126.489 ± 3.618 | 125.748 ± 2.636 | 135.012 ± 4.664 | 142.217 ± 6.558 | 148.148 ± 0.815 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 41.180 ± 1.474 | - | 141.476 ± 3.895 | 41.055 ± 0.983 | 65.345 ± 2.415 | 141.637 ± 3.871 | 64.229 ± 0.726 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 24.455 ± 0.118 | - | 24.505 ± 0.086 | 56.618 ± 1.072 | 106.269 ± 0.856 | 73.740 ± 8.414 | 69.822 ± 12.045 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 10.224 ± 0.045 | - | 47.998 ± 0.006 | 22.198 ± 0.457 | 54.049 ± 0.244 | 79.244 ± 16.271 | 29.415 ± 15.911 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 132.203 ± 2.616 | - | 133.706 ± 2.043 | 186.114 ± 1.723 | 134.520 ± 4.497 | 213.203 ± 0.831 | 219.360 ± 6.080 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 42.964 ± 0.490 | - | 150.630 ± 12.588 | 68.016 ± 0.981 | 68.100 ± 1.148 | 214.501 ± 4.208 | 90.553 ± 2.024 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 131.233 ± 3.299 | - | 134.747 ± 3.161 | 131.430 ± 3.363 | 139.151 ± 2.748 | 149.002 ± 5.016 | 167.011 ± 4.282 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 45.564 ± 1.311 | - | 153.165 ± 11.466 | 46.324 ± 1.164 | 73.227 ± 0.669 | 146.995 ± 1.410 | 70.250 ± 0.686 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 51.306 ± 7.729 | - | 51.363 ± 0.028 | 74.673 ± 0.827 | 38.698 ± 0.737 | 78.564 ± 12.069 | 86.210 ± 23.886 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 20.461 ± 0.093 | - | 71.999 ± 0.004 | 28.513 ± 0.431 | 20.121 ± 0.730 | 79.514 ± 13.405 | 34.269 ± 17.598 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 137.345 ± 5.193 | 136.325 ± 1.495 | unsupported | 145.308 ± 5.431 | 153.659 ± 1.913 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 46.934 ± 1.078 | 155.568 ± 4.226 | unsupported | 52.669 ± 2.007 | 81.939 ± 1.953 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.870 ± 0.081 | - | 0.936 ± 0.219 | 0.822 ± 0.110 | 0.907 ± 0.121 | 7.490 ± 3.892 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.440 ± 0.037 | - | 11.999 ± 0.004 | 0.325 ± 0.026 | 1.075 ± 0.047 | 1.614 ± 5.947 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.926 ± 0.394 | - | 0.974 ± 0.186 | 0.861 ± 0.088 | 1.421 ± 0.355 | 7.601 ± 4.186 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 1.999 ± 0.033 | - | 11.994 ± 0.015 | 0.322 ± 0.033 | 0.853 ± 0.151 | 1.568 ± 1.319 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.364 ± 0.150 | - | unsupported | 1.338 ± 0.203 | 1.420 ± 0.300 | 8.061 ± 3.621 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.796 ± 0.081 | - | unsupported | 0.724 ± 0.132 | 1.113 ± 0.138 | 8.054 ± 3.843 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.408 ± 0.025 | - | 0.478 ± 0.011 | 0.686 ± 0.021 | 0.424 ± 0.045 | 1.209 ± 0.062 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 1.709 ± 1.428 | - | 19.448 ± 5.750 | 0.345 ± 0.015 | 0.386 ± 0.046 | 1.203 ± 0.087 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 1.094 ± 0.147 | - | 0.878 ± 0.125 | 1.115 ± 0.126 | 1.049 ± 0.035 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.641 ± 0.038 | - | 12.000 ± 0.018 | 0.524 ± 0.025 | 0.753 ± 0.140 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 1.067 ± 0.122 | - | 0.850 ± 0.080 | 0.791 ± 0.079 | 0.930 ± 0.050 | 2.333 ± 7.180 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.458 ± 0.035 | - | 11.999 ± 0.006 | 0.316 ± 0.122 | 0.586 ± 0.103 | 7.673 ± 6.961 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.781 ± 0.013 | - | 0.901 ± 0.006 | 0.481 ± 0.009 | 0.500 ± 0.096 | 0.520 ± 0.778 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 1.995 ± 0.021 | - | 16.537 ± 7.249 | 0.605 ± 0.014 | 0.527 ± 0.076 | 1.350 ± 0.302 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.751 ± 0.061 | - | 0.002 ± 0.000 | 1.757 ± 0.016 | 1.720 ± 0.187 | 8.777 ± 5.659 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 2.058 ± 0.273 | - | 0.002 ± 0.000 | 0.911 ± 0.329 | 1.028 ± 0.132 | 9.198 ± 1.229 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.050 ± 0.000 | - | 0.052 ± 0.000 | 0.075 ± 0.000 | 0.095 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.092 ± 0.000 | - | 0.095 ± 0.005 | 0.125 ± 0.001 | 0.148 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.235 ± 0.022 | - | 0.238 ± 0.001 | 0.277 ± 0.012 | 0.323 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.081 ± 0.000 | - | 0.084 ± 0.001 | 0.146 ± 0.003 | 0.136 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.002 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.036 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.002 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.026 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.671 ± 0.010 | - | 0.670 ± 0.010 | 1.151 ± 0.008 | 0.669 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.026 ± 0.000 | - | 0.027 ± 0.000 | 0.039 ± 0.002 | 0.065 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.034 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.062 ± 0.000 | - | 0.071 ± 0.000 | 0.085 ± 0.002 | 0.084 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.093 ± 0.001 | - | 0.095 ± 0.001 | 0.123 ± 0.002 | 0.154 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.253 ± 0.028 | - | 0.249 ± 0.020 | 0.340 ± 0.011 | 0.344 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.114 ± 0.002 | - | 0.115 ± 0.002 | 0.177 ± 0.004 | 0.126 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.009 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.001 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.667 ± 0.021 | - | 0.693 ± 4.496 | 1.374 ± 0.018 | 1.126 ± 0.112 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.034 ± 0.000 | - | 0.045 ± 0.000 | 0.058 ± 0.002 | 0.065 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.590 ± 0.010 | - | 0.593 ± 0.009 | 0.704 ± 0.058 | 0.896 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 2.054 ± 0.004 | - | 2.062 ± 0.002 | 2.144 ± 0.018 | 2.720 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 6.435 ± 0.017 | - | 6.451 ± 0.032 | 6.635 ± 0.046 | 8.929 ± 0.098 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 3.863 ± 0.215 | - | 3.650 ± 0.094 | 3.211 ± 0.020 | 2.653 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.001 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.109 ± 0.001 | - | 0.111 ± 0.000 | 0.121 ± 0.006 | 0.202 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.024 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.016 ± 0.001 | 0.030 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.024 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.016 ± 0.000 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.020 ± 0.000 | - | 0.022 ± 0.002 | 0.029 ± 0.001 | 0.097 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.030 ± 0.020 | 0.033 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.026 ± 0.000 | - | 0.029 ± 0.000 | 0.035 ± 0.000 | 0.105 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 29.294 ± 0.412 | - | 29.263 ± 0.331 | 25.639 ± 0.252 | 20.557 ± 0.077 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.033 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.863 ± 0.003 | - | 0.868 ± 0.003 | 0.871 ± 0.002 | 0.994 ± 0.093 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.017 ± 0.001 | - | 0.020 ± 0.000 | 0.025 ± 0.001 | 0.043 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.051 ± 0.000 | - | 0.054 ± 0.001 | 0.061 ± 0.001 | 0.141 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.603 ± 0.039 | - | 0.602 ± 4.498 | 0.705 ± 0.411 | 0.887 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 2.106 ± 0.011 | - | 2.108 ± 0.022 | 2.103 ± 0.013 | 2.440 ± 0.234 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 6.522 ± 0.021 | - | 6.523 ± 0.034 | 9.305 ± 0.071 | 7.655 ± 1.037 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 6.654 ± 0.043 | - | 6.719 ± 0.371 | 2.259 ± 0.044 | 3.450 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.013 ± 0.001 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.110 ± 0.001 | - | 0.133 ± 0.021 | 0.176 ± 0.006 | 0.187 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.001 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.016 ± 0.001 | 0.031 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.016 ± 0.001 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.016 ± 0.000 | 0.032 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.022 ± 0.000 | - | 0.023 ± 0.000 | 0.029 ± 0.001 | 0.063 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.004 ± 0.004 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.018 ± 0.000 | 0.034 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.027 ± 0.000 | - | 0.031 ± 0.000 | 0.035 ± 0.001 | 0.073 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 53.287 ± 0.212 | - | 53.347 ± 0.818 | 17.892 ± 0.788 | 26.992 ± 0.905 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.021 ± 0.001 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.880 ± 0.034 | - | 0.885 ± 0.033 | 1.244 ± 0.010 | 1.714 ± 0.858 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.018 ± 0.000 | - | 0.020 ± 0.000 | 0.025 ± 0.000 | 0.053 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.054 ± 0.000 | - | 0.055 ± 0.000 | 0.061 ± 0.002 | 0.128 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.660 ± 0.011 | - | 0.656 ± 0.008 | 0.508 ± 0.012 | 0.468 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 2.163 ± 0.004 | - | 2.167 ± 0.009 | 1.918 ± 0.004 | 1.975 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 5.747 ± 0.022 | - | 5.751 ± 0.021 | 5.676 ± 0.025 | 6.421 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.317 ± 0.014 | - | 1.311 ± 0.007 | 1.324 ± 0.009 | 1.589 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.020 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.070 ± 0.002 | - | 0.074 ± 0.012 | 0.089 ± 0.005 | 0.129 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | 0.010 ± 0.002 | 0.015 ± 0.001 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.025 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.014 ± 0.001 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.018 ± 0.000 | - | 0.021 ± 0.000 | 0.026 ± 0.003 | 0.042 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.016 ± 0.001 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.024 ± 0.000 | - | 0.026 ± 0.000 | 0.031 ± 0.001 | 0.048 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 10.353 ± 0.046 | - | 10.319 ± 0.017 | 10.433 ± 0.060 | 11.419 ± 0.799 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.545 ± 0.011 | - | 0.557 ± 0.012 | 0.623 ± 0.013 | 0.735 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.018 ± 0.000 | - | 0.020 ± 0.002 | 0.023 ± 0.001 | 0.036 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.046 ± 0.000 | - | 0.048 ± 0.000 | 0.053 ± 0.006 | 0.117 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.635 ± 0.019 | - | 0.648 ± 4.470 | 0.787 ± 0.040 | 0.444 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 2.189 ± 0.037 | - | 2.175 ± 0.024 | 1.896 ± 0.017 | 1.280 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 5.795 ± 0.052 | - | 5.787 ± 0.033 | 8.508 ± 0.009 | 3.683 ± 0.075 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 3.224 ± 0.191 | - | 3.355 ± 0.147 | 1.154 ± 0.021 | 1.994 ± 0.161 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.012 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.159 ± 0.038 | - | 0.163 ± 0.026 | 0.134 ± 0.006 | 0.149 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.015 ± 0.000 | 0.035 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.006 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.014 ± 0.001 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.019 ± 0.000 | - | 0.023 ± 0.000 | 0.025 ± 0.000 | 0.040 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.015 ± 0.004 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.017 ± 0.001 | 0.036 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.027 ± 0.000 | - | 0.033 ± 0.000 | 0.031 ± 0.001 | 0.049 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 25.357 ± 0.573 | - | 24.918 ± 0.194 | 9.094 ± 0.253 | 14.032 ± 2.293 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.017 ± 0.001 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 1.080 ± 0.023 | - | 1.096 ± 0.027 | 0.934 ± 0.002 | 0.446 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.023 ± 0.001 | 0.036 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.056 ± 0.000 | - | 0.054 ± 0.000 | 0.053 ± 0.002 | 0.097 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.269 ± 0.011 | - | 0.275 ± 0.015 | 0.238 ± 0.012 | 0.464 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 1.076 ± 0.002 | - | 1.082 ± 0.002 | 0.947 ± 0.004 | 1.959 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 3.076 ± 0.007 | - | 3.079 ± 0.010 | 2.885 ± 0.008 | 6.409 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.516 ± 0.008 | - | 0.516 ± 0.008 | 0.550 ± 0.013 | 1.446 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.010 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.031 ± 0.000 | - | 0.033 ± 0.000 | 0.042 ± 0.007 | 0.137 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.019 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.001 | 0.012 ± 0.001 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.001 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.019 ± 0.001 | 0.043 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.010 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.001 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.021 ± 0.001 | 0.041 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 4.083 ± 0.018 | - | 4.080 ± 0.007 | 4.302 ± 0.014 | 11.162 ± 0.260 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.001 | 0.018 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.231 ± 0.009 | - | 0.234 ± 0.001 | 0.243 ± 0.004 | 0.756 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.010 ± 0.000 | - | 0.012 ± 0.001 | 0.016 ± 0.001 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.025 ± 0.000 | - | 0.027 ± 0.000 | 0.032 ± 0.002 | 0.089 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.269 ± 0.028 | - | 0.277 ± 0.005 | 0.396 ± 0.012 | 0.467 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 1.086 ± 0.028 | - | 1.089 ± 0.012 | 0.943 ± 0.003 | 1.269 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 3.107 ± 0.006 | - | 3.144 ± 0.009 | 3.647 ± 0.015 | 3.774 ± 0.064 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 2.038 ± 0.056 | - | 1.994 ± 0.008 | 0.693 ± 0.016 | 1.887 ± 0.174 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.014 ± 0.002 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.031 ± 0.000 | - | 0.038 ± 0.000 | 0.055 ± 0.003 | 0.138 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.015 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.015 ± 0.001 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.004 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.019 ± 0.001 | 0.034 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.015 ± 0.001 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.019 ± 0.002 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.021 ± 0.000 | 0.040 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 15.785 ± 0.022 | - | 15.654 ± 0.058 | 5.165 ± 0.016 | 14.998 ± 1.981 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.017 ± 0.004 | 0.023 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.251 ± 0.010 | - | 0.263 ± 0.022 | 0.317 ± 0.014 | 0.427 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.023 ± 0.001 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.025 ± 0.000 | - | 0.028 ± 0.000 | 0.032 ± 0.000 | 0.076 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.152 ± 0.002 | - | 0.157 ± 0.004 | 0.194 ± 0.003 | 0.147 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.213 ± 0.010 | - | 0.219 ± 0.010 | 0.259 ± 0.002 | 0.209 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.507 ± 0.009 | - | 0.506 ± 0.009 | 0.546 ± 0.016 | 0.585 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.135 ± 0.004 | - | 0.141 ± 0.004 | 0.203 ± 0.037 | 0.266 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.013 ± 0.001 | 0.019 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.012 ± 0.001 | - | 0.014 ± 0.000 | 0.025 ± 0.001 | 0.051 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.001 | 0.013 ± 0.002 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.014 ± 0.000 | - | 0.016 ± 0.001 | 0.028 ± 0.004 | 0.088 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.013 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.016 ± 0.008 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.001 | 0.025 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.017 ± 0.001 | 0.034 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.015 ± 0.001 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.018 ± 0.000 | 0.045 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 1.089 ± 0.018 | - | 1.074 ± 0.049 | 1.330 ± 0.023 | 1.285 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.000 | 0.023 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.076 ± 0.003 | - | 0.079 ± 0.006 | 0.088 ± 0.001 | 0.181 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.016 ± 0.001 | 0.034 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.020 ± 0.006 | 0.051 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.156 ± 0.006 | - | 0.172 ± 0.014 | 0.173 ± 0.021 | 0.127 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.212 ± 0.015 | - | 0.221 ± 0.010 | 0.233 ± 0.015 | 0.212 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.542 ± 0.028 | - | 0.535 ± 0.022 | 0.400 ± 0.027 | 0.639 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.136 ± 0.002 | - | 0.165 ± 0.006 | 0.222 ± 0.020 | 0.226 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.023 ± 0.003 | 0.022 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.013 ± 0.004 | - | 0.016 ± 0.000 | 0.037 ± 0.001 | 0.031 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.022 ± 0.004 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.015 ± 0.000 | - | 0.023 ± 0.061 | 0.042 ± 0.002 | 0.042 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.022 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.024 ± 0.001 | 0.019 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.025 ± 0.001 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.027 ± 0.001 | 0.023 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.025 ± 0.001 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.025 ± 0.001 | 0.025 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.027 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 1.059 ± 0.037 | - | 1.106 ± 0.071 | 0.704 ± 0.028 | 0.825 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.026 ± 0.002 | 0.023 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.085 ± 0.000 | - | 0.085 ± 0.005 | 0.076 ± 0.000 | 0.217 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.027 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.010 ± 0.000 | - | 0.011 ± 0.001 | 0.030 ± 0.000 | 0.028 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.258 ± 0.010 | - | 0.262 ± 0.014 | 0.231 ± 0.013 | 0.322 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.589 ± 0.010 | - | 0.602 ± 0.014 | 0.525 ± 0.015 | 0.855 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.346 ± 0.003 | - | 1.339 ± 0.019 | 1.363 ± 0.011 | 2.322 ± 0.067 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.488 ± 0.006 | - | 0.502 ± 0.012 | 0.584 ± 0.021 | 0.627 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.017 ± 0.000 | - | 0.019 ± 0.000 | 0.027 ± 0.001 | 0.065 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.029 ± 0.001 | 0.061 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.001 | 0.031 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.021 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.001 | 0.030 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 3.875 ± 0.008 | - | 4.634 ± 0.092 | 4.407 ± 0.059 | 4.713 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.128 ± 0.001 | - | 0.132 ± 0.001 | 0.175 ± 0.006 | 0.221 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.016 ± 0.000 | 0.037 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.256 ± 0.002 | - | 0.258 ± 0.011 | 0.329 ± 0.008 | 0.323 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.604 ± 0.028 | - | 0.606 ± 4.475 | 0.771 ± 0.008 | 0.859 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.327 ± 0.021 | - | 1.343 ± 0.044 | 2.034 ± 0.026 | 1.326 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 1.026 ± 0.028 | - | 1.030 ± 0.018 | 0.508 ± 0.033 | 0.995 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.018 ± 0.000 | - | 0.034 ± 0.000 | 0.044 ± 0.004 | 0.052 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.002 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.020 ± 0.001 | - | 0.028 ± 0.001 | 0.047 ± 0.007 | 0.061 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.001 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.000 | 0.035 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 8.015 ± 0.110 | - | 8.867 ± 0.073 | 3.735 ± 0.076 | 8.046 ± 0.997 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.182 ± 0.052 | - | 0.153 ± 0.052 | 0.291 ± 0.015 | 0.215 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.015 ± 0.000 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.026 ± 0.001 | 0.037 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.157 ± 0.002 | - | 0.156 ± 0.004 | 0.099 ± 0.001 | 0.280 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.200 ± 0.001 | - | 0.198 ± 0.050 | 0.139 ± 0.001 | 0.392 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.539 ± 0.008 | - | 0.564 ± 0.005 | 0.403 ± 0.012 | 0.817 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.118 ± 0.002 | - | 0.134 ± 0.042 | 0.146 ± 0.003 | 0.225 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.017 ± 0.001 | 0.018 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.011 ± 0.000 | - | 0.017 ± 0.003 | 0.030 ± 0.003 | 0.076 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.017 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.001 | 0.017 ± 0.001 | 0.017 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.020 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.019 ± 0.000 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.913 ± 0.012 | - | 0.968 ± 0.003 | 1.015 ± 0.026 | 1.051 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.019 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.069 ± 0.006 | - | 0.079 ± 0.020 | 0.080 ± 0.002 | 0.166 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.021 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.011 ± 0.000 | 0.021 ± 0.001 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.193 ± 0.024 | - | 0.148 ± 0.004 | 0.130 ± 0.002 | 0.263 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.211 ± 0.004 | - | 0.207 ± 0.043 | 0.172 ± 0.016 | 0.373 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.553 ± 0.022 | - | 5.996 ± 0.017 | 0.385 ± 0.012 | 0.855 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.118 ± 0.020 | - | 0.174 ± 0.041 | 0.201 ± 0.015 | 0.230 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.025 ± 0.003 | 0.020 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.012 ± 0.000 | - | 0.019 ± 0.003 | 0.042 ± 0.004 | 0.036 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.025 ± 0.001 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.010 ± 0.000 | 0.026 ± 0.003 | 0.018 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.010 ± 0.000 | 0.025 ± 0.003 | 0.018 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.025 ± 0.003 | 0.018 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.014 ± 0.000 | 0.028 ± 0.002 | 0.023 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.025 ± 0.002 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.026 ± 0.002 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.013 ± 0.000 | 0.028 ± 0.002 | 0.022 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.912 ± 0.017 | - | 24.006 ± 0.347 | 0.576 ± 0.020 | 0.731 ± 0.058 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.013 ± 0.000 | 0.027 ± 0.000 | 0.023 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.072 ± 0.004 | - | 0.092 ± 0.026 | 0.071 ± 0.001 | 0.172 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.028 ± 0.001 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.008 ± 0.000 | - | 0.013 ± 0.000 | 0.031 ± 0.001 | 0.027 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.754 ± 0.009 | - | 1.759 ± 0.007 | 1.507 ± 0.016 | 2.027 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 5.926 ± 0.035 | - | 5.917 ± 0.010 | 5.120 ± 0.073 | 5.183 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 17.704 ± 0.096 | - | 17.646 ± 0.049 | 16.132 ± 0.348 | 14.609 ± 0.080 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 4.931 ± 0.029 | - | 4.938 ± 0.019 | 5.775 ± 0.210 | 4.202 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.004 ± 0.001 | - | 0.007 ± 0.000 | 0.014 ± 0.002 | 0.039 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.203 ± 0.012 | - | 0.209 ± 0.020 | 0.243 ± 0.018 | 0.238 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.019 ± 0.001 | 0.034 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.180 ± 0.012 | - | 0.184 ± 0.010 | 0.208 ± 0.012 | 0.220 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.020 ± 0.000 | - | 0.023 ± 0.000 | 0.031 ± 0.001 | 0.045 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.002 | 0.034 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.020 ± 0.000 | - | 0.022 ± 0.000 | 0.030 ± 0.004 | 0.045 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.054 ± 0.000 | - | 0.057 ± 0.000 | 0.062 ± 0.002 | 0.075 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.019 ± 0.001 | 0.033 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.026 ± 0.000 | - | 0.029 ± 0.002 | 0.034 ± 0.001 | 0.053 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.073 ± 0.001 | - | 0.078 ± 0.004 | 0.079 ± 0.001 | 0.088 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 39.266 ± 0.103 | - | 40.795 ± 1.809 | 44.910 ± 1.764 | 33.050 ± 0.181 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.016 ± 0.000 | - | 0.019 ± 0.000 | 0.025 ± 0.000 | 0.042 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.547 ± 0.033 | - | 1.545 ± 0.023 | 1.644 ± 0.029 | 1.538 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.049 ± 0.000 | - | 0.052 ± 0.003 | 0.056 ± 0.005 | 0.102 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.140 ± 0.001 | - | 0.143 ± 0.003 | 0.140 ± 0.005 | 0.162 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.738 ± 0.030 | - | 1.747 ± 0.024 | 1.468 ± 0.016 | 2.004 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 6.030 ± 0.017 | - | 6.025 ± 0.026 | 5.061 ± 0.007 | 2.755 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 17.714 ± 0.139 | - | 17.712 ± 0.083 | 18.503 ± 0.088 | 7.533 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 9.079 ± 0.046 | - | 9.178 ± 0.231 | 4.601 ± 0.015 | 5.064 ± 0.124 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.020 ± 0.003 | 0.033 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.300 ± 0.027 | - | 0.305 ± 0.017 | 0.324 ± 0.014 | 0.258 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.019 ± 0.002 | 0.033 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.220 ± 0.011 | - | 0.245 ± 0.025 | 0.285 ± 0.014 | 0.239 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.021 ± 0.000 | - | 0.025 ± 0.000 | 0.031 ± 0.001 | 0.045 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.024 ± 0.001 | 0.034 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.020 ± 0.000 | - | 0.038 ± 0.019 | 0.029 ± 0.001 | 0.042 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.056 ± 0.003 | - | 0.061 ± 0.001 | 0.062 ± 0.002 | 0.072 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.026 ± 0.005 | 0.034 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.027 ± 0.000 | - | 0.031 ± 0.000 | 0.034 ± 0.000 | 0.049 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.073 ± 0.000 | - | 0.080 ± 0.001 | 0.078 ± 0.001 | 0.088 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 71.921 ± 0.316 | - | 73.365 ± 0.551 | 35.923 ± 0.226 | 74.994 ± 38.532 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.016 ± 0.000 | - | 0.019 ± 0.000 | 0.037 ± 0.003 | 0.041 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 2.271 ± 0.043 | - | 2.291 ± 0.032 | 2.316 ± 0.012 | 0.870 ± 0.157 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.053 ± 0.001 | - | 0.055 ± 0.001 | 0.055 ± 0.001 | 0.069 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.141 ± 0.001 | - | 0.147 ± 0.009 | 0.139 ± 0.003 | 0.161 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 1.300 ± 0.008 | - | 1.306 ± 0.010 | 1.058 ± 0.001 | 1.277 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 5.084 ± 0.009 | - | 5.096 ± 0.012 | 4.284 ± 0.019 | 3.988 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 15.092 ± 0.032 | - | 15.084 ± 0.073 | 13.750 ± 0.026 | 12.198 ± 0.059 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 2.343 ± 0.021 | - | 2.311 ± 0.081 | 2.440 ± 0.019 | 2.505 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.001 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.137 ± 0.006 | - | 0.144 ± 0.007 | 0.165 ± 0.008 | 0.187 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.017 ± 0.001 | 0.036 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.107 ± 0.002 | - | 0.111 ± 0.007 | 0.139 ± 0.010 | 0.171 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.017 ± 0.000 | - | 0.020 ± 0.003 | 0.027 ± 0.007 | 0.046 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.015 ± 0.001 | 0.034 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.018 ± 0.000 | - | 0.020 ± 0.000 | 0.026 ± 0.001 | 0.037 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.043 ± 0.000 | - | 0.047 ± 0.004 | 0.051 ± 0.001 | 0.068 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.016 ± 0.001 | 0.034 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.023 ± 0.000 | - | 0.026 ± 0.000 | 0.030 ± 0.001 | 0.053 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.057 ± 0.000 | - | 0.060 ± 0.000 | 0.065 ± 0.002 | 0.073 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 18.471 ± 0.333 | - | 18.192 ± 0.137 | 19.227 ± 0.547 | 19.228 ± 0.219 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.020 ± 0.000 | 0.033 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 1.057 ± 0.009 | - | 1.072 ± 0.018 | 1.153 ± 0.011 | 1.131 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.042 ± 0.000 | - | 0.046 ± 0.002 | 0.047 ± 0.001 | 0.074 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.114 ± 0.000 | - | 0.117 ± 0.002 | 0.117 ± 0.002 | 0.140 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 1.280 ± 0.023 | - | 1.287 ± 0.023 | 1.012 ± 0.029 | 1.267 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 5.124 ± 0.011 | - | 5.144 ± 0.015 | 4.244 ± 0.117 | 2.072 ± 0.337 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 15.150 ± 0.138 | - | 15.085 ± 0.061 | 16.593 ± 0.038 | 6.213 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 3.365 ± 0.026 | - | 3.398 ± 0.026 | 2.752 ± 0.032 | 3.105 ± 0.954 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.001 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.228 ± 0.016 | - | 0.249 ± 0.030 | 0.253 ± 0.020 | 0.184 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.008 ± 0.000 | - | 0.012 ± 0.000 | 0.017 ± 0.001 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.148 ± 0.007 | - | 0.179 ± 0.024 | 0.196 ± 0.008 | 0.183 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.018 ± 0.001 | - | 0.026 ± 0.000 | 0.027 ± 0.002 | 0.045 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.021 ± 0.002 | 0.028 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.021 ± 0.000 | - | 0.024 ± 0.000 | 0.026 ± 0.003 | 0.040 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.052 ± 0.001 | - | 0.055 ± 0.000 | 0.052 ± 0.008 | 0.060 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.010 ± 0.000 | - | 0.011 ± 0.000 | 0.023 ± 0.006 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.026 ± 0.000 | - | 0.027 ± 0.000 | 0.030 ± 0.001 | 0.043 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.061 ± 0.002 | - | 0.065 ± 0.000 | 0.065 ± 0.005 | 0.072 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 27.039 ± 0.195 | - | 26.815 ± 0.166 | 21.672 ± 0.077 | 24.263 ± 6.308 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.013 ± 0.000 | - | 0.019 ± 0.000 | 0.029 ± 0.004 | 0.034 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 1.755 ± 0.047 | - | 1.796 ± 0.036 | 1.736 ± 0.005 | 0.651 ± 0.083 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.043 ± 0.000 | - | 0.046 ± 0.000 | 0.046 ± 0.002 | 0.059 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.116 ± 0.002 | - | 0.125 ± 0.007 | 0.116 ± 0.007 | 0.140 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.466 ± 0.018 | - | unsupported | 0.353 ± 0.012 | 0.741 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 2.255 ± 0.049 | - | unsupported | 1.904 ± 0.004 | 2.669 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 5.517 ± 0.009 | - | unsupported | 5.282 ± 0.008 | 6.769 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 1.000 ± 0.011 | - | unsupported | 1.141 ± 0.007 | 1.110 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.009 ± 0.001 | 0.015 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.049 ± 0.001 | - | unsupported | 0.063 ± 0.005 | 0.073 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | unsupported | 0.011 ± 0.001 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.040 ± 0.001 | - | unsupported | 0.054 ± 0.005 | 0.066 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | unsupported | 0.015 ± 0.007 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.010 ± 0.001 | 0.016 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | unsupported | 0.015 ± 0.001 | 0.024 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.017 ± 0.000 | - | unsupported | 0.026 ± 0.001 | 0.036 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.010 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | unsupported | 0.017 ± 0.001 | 0.028 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.023 ± 0.000 | - | unsupported | 0.031 ± 0.001 | 0.043 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 7.883 ± 0.092 | - | unsupported | 8.919 ± 0.027 | 8.695 ± 0.090 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | unsupported | 0.012 ± 0.000 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.401 ± 0.009 | - | unsupported | 0.425 ± 0.014 | 0.484 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.019 ± 0.000 | - | unsupported | 0.024 ± 0.002 | 0.036 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.044 ± 0.000 | - | unsupported | 0.052 ± 0.001 | 0.071 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.454 ± 0.009 | - | unsupported | 0.572 ± 0.015 | 0.765 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 2.268 ± 0.038 | - | unsupported | 1.895 ± 0.005 | 1.394 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 5.593 ± 0.026 | - | unsupported | 7.578 ± 0.012 | 3.502 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.188 ± 0.035 | - | unsupported | 1.439 ± 0.012 | 1.419 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | unsupported | 0.013 ± 0.001 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.049 ± 0.002 | - | unsupported | 0.087 ± 0.005 | 0.073 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | unsupported | 0.016 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.040 ± 0.000 | - | unsupported | 0.079 ± 0.004 | 0.087 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.007 ± 0.000 | - | unsupported | 0.015 ± 0.000 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.015 ± 0.001 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.008 ± 0.000 | - | unsupported | 0.023 ± 0.004 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.018 ± 0.000 | - | unsupported | 0.026 ± 0.003 | 0.036 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | unsupported | 0.015 ± 0.000 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.011 ± 0.000 | - | unsupported | 0.025 ± 0.002 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.023 ± 0.001 | - | unsupported | 0.030 ± 0.001 | 0.046 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 9.404 ± 0.135 | - | unsupported | 11.139 ± 0.035 | 11.454 ± 0.349 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | unsupported | 0.017 ± 0.001 | 0.022 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.375 ± 0.009 | - | unsupported | 0.421 ± 0.011 | 0.315 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.019 ± 0.000 | - | unsupported | 0.024 ± 0.006 | 0.052 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.044 ± 0.000 | - | unsupported | 0.051 ± 0.003 | 0.068 ± 0.003 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 1.050 ± 0.016 | - | unsupported | 0.952 ± 0.001 | 3.420 ± 0.081 | 2.588 ± 0.022 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.133 ± 0.001 | - | unsupported | 0.061 ± 0.004 | 0.107 ± 0.013 | 0.131 ± 0.005 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.136 ± 0.003 | - | unsupported | 0.084 ± 0.003 | 0.199 ± 0.022 | 0.229 ± 0.023 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.340 ± 0.009 | - | unsupported | 0.243 ± 0.012 | 0.497 ± 0.017 | 0.749 ± 0.092 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 1.044 ± 0.013 | - | unsupported | 0.486 ± 0.013 | 1.076 ± 0.048 | 2.476 ± 0.840 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.135 ± 0.007 | - | unsupported | 0.060 ± 0.002 | 0.109 ± 0.015 | 0.101 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.143 ± 0.004 | - | unsupported | 0.079 ± 0.001 | 0.185 ± 0.013 | 0.180 ± 0.017 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.339 ± 0.020 | - | unsupported | 0.159 ± 0.002 | 0.538 ± 0.111 | 0.679 ± 0.045 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.312 ± 0.010 | - | unsupported | 0.371 ± 0.011 | 0.802 ± 0.008 | 0.484 ± 0.014 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.034 ± 0.000 | - | unsupported | 0.045 ± 0.001 | 0.189 ± 0.014 | 0.078 ± 0.010 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.059 ± 0.000 | - | unsupported | 0.061 ± 0.000 | 0.233 ± 0.016 | 0.126 ± 0.006 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.201 ± 0.003 | - | unsupported | 0.166 ± 0.002 | 0.379 ± 0.014 | 0.236 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.303 ± 0.012 | - | unsupported | 0.559 ± 0.009 | 0.844 ± 0.058 | 0.423 ± 0.014 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.035 ± 0.000 | - | unsupported | 0.073 ± 0.003 | 0.204 ± 0.008 | 0.075 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.060 ± 0.003 | - | unsupported | 0.098 ± 0.001 | 0.242 ± 0.022 | 0.115 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.201 ± 0.005 | - | unsupported | 0.226 ± 0.005 | 0.414 ± 0.022 | 0.210 ± 0.007 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.193 ± 0.001 | - | 0.195 ± 0.010 | 0.197 ± 0.016 | 0.385 ± 0.016 | 0.319 ± 0.019 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.084 ± 0.003 | - | 0.087 ± 0.001 | 0.073 ± 0.001 | 0.077 ± 0.005 | 0.063 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.093 ± 0.000 | - | 0.095 ± 0.000 | 0.086 ± 0.001 | 0.102 ± 0.002 | 0.095 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.125 ± 0.000 | - | 0.127 ± 0.001 | 0.122 ± 0.001 | 0.174 ± 0.015 | 0.168 ± 0.006 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.194 ± 0.001 | - | 0.199 ± 0.022 | 0.255 ± 0.010 | 0.464 ± 0.031 | 0.251 ± 0.009 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.082 ± 0.003 | - | 0.108 ± 0.005 | 0.124 ± 0.003 | 0.085 ± 0.027 | 0.060 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.093 ± 0.001 | - | 0.111 ± 0.019 | 0.143 ± 0.002 | 0.105 ± 0.005 | 0.086 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.124 ± 0.001 | - | 0.127 ± 0.015 | 0.177 ± 0.013 | 0.231 ± 0.050 | 0.143 ± 0.006 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 43.071 ± 0.477 | - | 49.785 ± 0.341 | 60.887 ± 0.498 | 70.320 ± 1.272 | 45.581 ± 13.505 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 46.464 ± 0.697 | - | 49.564 ± 1.010 | 24.821 ± 0.220 | 50.983 ± 19.163 | 46.988 ± 7.725 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 25.238 ± 0.428 | - | 24.702 ± 0.426 | 27.005 ± 0.079 | 21.859 ± 0.100 | 20.855 ± 0.243 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 249.851 ± 12.329 | - | 313.973 ± 59.007 | 11.314 ± 0.433 | 13.924 ± 0.062 | 12.549 ± 0.100 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 7.477 ± 0.083 | - | 7.448 ± 0.018 | 7.932 ± 0.066 | 8.060 ± 0.159 | 8.054 ± 0.227 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 16.675 ± 0.308 | - | 16.435 ± 0.129 | 10.364 ± 0.325 | 11.692 ± 0.032 | 11.609 ± 0.060 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 8.684 ± 0.103 | - | 8.758 ± 0.047 | 8.789 ± 0.061 | 9.530 ± 0.180 | 9.195 ± 0.055 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 12.910 ± 0.085 | - | 12.928 ± 0.023 | 11.765 ± 0.054 | 14.308 ± 0.821 | 13.768 ± 0.085 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 20.944 ± 0.096 | - | 21.524 ± 0.061 | 19.427 ± 0.230 | 38.674 ± 0.090 | 13.973 ± 0.893 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 38.909 ± 0.197 | - | 38.953 ± 0.083 | 15.139 ± 0.386 | 32.468 ± 1.745 | 12.931 ± 0.541 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.490 ± 0.008 | 0.508 ± 0.003 | unsupported | 0.605 ± 0.011 | 0.734 ± 0.052 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 7.780 ± 0.073 | 7.885 ± 0.023 | unsupported | 8.733 ± 0.145 | 9.670 ± 0.138 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.777 ± 0.004 | 0.800 ± 0.015 | unsupported | 0.519 ± 0.012 | 1.251 ± 1.379 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 2275.688 ± 1.209 | 2308.212 ± 1.888 | unsupported | 5.665 ± 0.083 | 9.549 ± 0.245 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 44.082 ± 3.250 | - | 41.348 ± 0.791 | 42.882 ± 0.513 | 32.025 ± 0.185 | 27.450 ± 0.303 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 236.400 ± 20.485 | - | 282.313 ± 12.266 | 21.606 ± 0.222 | 19.801 ± 0.156 | 17.106 ± 0.416 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 17.755 ± 0.756 | 17.302 ± 0.045 | 9.213 ± 0.094 | 46.212 ± 0.451 | 9.551 ± 0.072 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 2291.091 ± 9.822 | 2285.042 ± 21.422 | 6.212 ± 0.199 | 43.096 ± 2.470 | 9.376 ± 0.607 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 24.925 ± 0.314 | - | 25.153 ± 0.343 | 33.325 ± 0.298 | 32.056 ± 0.123 | 31.065 ± 14.497 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 250.164 ± 26.556 | - | 249.842 ± 12.052 | 16.034 ± 0.251 | 25.055 ± 1.223 | 23.365 ± 4.216 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 1.358 ± 0.131 | - | 1.340 ± 0.120 | 1.588 ± 0.180 | 20.325 ± 0.939 | 2.719 ± 0.039 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.517 ± 0.050 | - | 23.999 ± 0.011 | 1.504 ± 0.031 | 13.044 ± 2.342 | 2.750 ± 0.054 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 19.770 ± 0.405 | - | 19.472 ± 0.189 | 20.989 ± 0.185 | 19.993 ± 0.284 | 19.188 ± 0.100 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 220.738 ± 14.474 | - | 238.013 ± 52.372 | 16.451 ± 0.155 | 20.391 ± 0.900 | 21.186 ± 2.523 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 19.649 ± 0.364 | - | 19.647 ± 0.107 | 21.248 ± 0.290 | 19.887 ± 0.092 | 19.184 ± 1.859 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 228.048 ± 8.383 | - | 244.186 ± 12.183 | 16.481 ± 0.049 | 23.011 ± 5.652 | 19.699 ± 1.014 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.179 ± 0.014 | 1.189 ± 0.006 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 21.308 ± 0.089 | 21.377 ± 0.148 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.627 ± 0.083 | 1.602 ± 0.013 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 23.393 ± 0.043 | 23.267 ± 0.165 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.095 ± 0.010 | 1.106 ± 0.007 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 20.302 ± 0.107 | 20.277 ± 0.125 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.520 ± 0.045 | 1.547 ± 0.043 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 22.381 ± 0.092 | 22.350 ± 0.080 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 24.983 ± 0.654 | - | 24.147 ± 0.579 | 26.437 ± 0.471 | 21.996 ± 0.257 | 20.712 ± 0.383 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 243.827 ± 65.040 | - | 324.015 ± 18.999 | 11.351 ± 0.277 | 14.077 ± 0.220 | 12.735 ± 2.314 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 59.076 ± 1.446 | 59.096 ± 0.529 | 61.133 ± 0.827 | 65.701 ± 0.759 | 62.439 ± 1.464 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 2335.368 ± 15.654 | 2333.833 ± 20.644 | 38.635 ± 0.929 | 54.214 ± 3.400 | 54.963 ± 4.428 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 33.673 ± 0.449 | - | 33.682 ± 0.297 | 30.554 ± 0.283 | 126.711 ± 4.133 | 28.979 ± 0.162 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 52.730 ± 1.713 | - | 59.198 ± 4.651 | 24.969 ± 4.851 | 124.433 ± 11.881 | 19.929 ± 1.099 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.140 ± 0.000 | 0.139 ± 0.001 | unsupported | 0.156 ± 0.030 | 0.036 ± 0.004 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.234 ± 0.781 | 2.222 ± 0.009 | unsupported | 2.435 ± 0.013 | 1.216 ± 0.136 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.075 ± 0.007 | 2.185 ± 3.915 | unsupported | 0.080 ± 0.011 | 0.136 ± 0.012 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.896 ± 0.002 | 23.999 ± 0.000 | unsupported | 1.014 ± 0.021 | 0.427 ± 0.381 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.079 ± 0.000 | 0.079 ± 0.000 | unsupported | 0.133 ± 0.001 | 0.248 ± 0.004 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.252 ± 0.003 | 1.247 ± 0.008 | unsupported | 1.732 ± 0.016 | 2.618 ± 0.029 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.037 ± 0.000 | 12.000 ± 0.000 | unsupported | 0.089 ± 0.033 | 0.157 ± 0.004 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.560 ± 0.002 | 11.999 ± 0.001 | unsupported | 0.731 ± 0.016 | 1.086 ± 0.047 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.197 ± 0.000 | 0.198 ± 0.000 | unsupported | 0.282 ± 0.002 | 0.185 ± 0.026 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 3.136 ± 0.007 | 3.117 ± 0.004 | unsupported | 4.400 ± 0.025 | 1.837 ± 0.025 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.076 ± 0.000 | 12.562 ± 0.703 | unsupported | 0.107 ± 0.002 | 0.130 ± 0.001 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 1.178 ± 0.009 | 11.999 ± 0.000 | unsupported | 1.498 ± 0.007 | 0.970 ± 0.352 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.054 ± 0.003 | 0.087 ± 0.005 | unsupported | 0.155 ± 0.001 | 0.178 ± 0.002 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.640 ± 0.045 | 0.702 ± 0.041 | unsupported | 0.906 ± 0.009 | 1.876 ± 0.027 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.085 ± 0.001 | 35.993 ± 0.016 | unsupported | 0.069 ± 0.002 | 0.178 ± 0.004 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.439 ± 0.051 | 96.000 ± 0.002 | unsupported | 0.364 ± 0.004 | 1.192 ± 0.039 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.124 ± 0.004 | 0.151 ± 0.001 | unsupported | 0.131 ± 0.003 | 0.217 ± 0.002 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.674 ± 0.018 | 1.740 ± 0.059 | unsupported | 1.239 ± 0.020 | 2.305 ± 0.104 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.144 ± 0.005 | 43.859 ± 4.851 | unsupported | 0.064 ± 0.004 | 0.206 ± 0.002 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 1.148 ± 0.254 | 119.875 ± 56.753 | unsupported | 0.480 ± 0.009 | 1.402 ± 0.080 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.290 ± 0.002 | 0.336 ± 0.015 | unsupported | unsupported | 0.240 ± 0.005 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 4.329 ± 0.034 | 4.432 ± 0.054 | unsupported | unsupported | 3.114 ± 0.025 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.287 ± 0.008 | 51.874 ± 0.266 | unsupported | unsupported | 0.236 ± 0.017 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 2.101 ± 0.115 | 144.000 ± 0.127 | unsupported | unsupported | 1.882 ± 0.205 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.011 ± 0.001 | 0.019 ± 0.001 | unsupported | 0.013 ± 0.001 | 0.049 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.101 ± 0.001 | 0.111 ± 0.002 | unsupported | 0.091 ± 0.005 | 0.247 ± 0.004 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.015 ± 0.000 | 5.968 ± 0.062 | unsupported | 0.015 ± 0.000 | 0.037 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.113 ± 0.007 | 24.000 ± 0.001 | unsupported | 0.072 ± 0.009 | 0.215 ± 0.003 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.029 ± 0.005 | 0.048 ± 0.003 | unsupported | 0.098 ± 0.001 | 0.181 ± 0.002 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.369 ± 0.043 | 0.379 ± 0.043 | unsupported | 0.572 ± 0.031 | 1.699 ± 0.043 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.045 ± 0.002 | 17.984 ± 0.141 | unsupported | 0.101 ± 0.002 | 0.152 ± 0.006 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.230 ± 0.020 | 48.000 ± 0.002 | unsupported | 0.283 ± 0.041 | 1.100 ± 0.063 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.054 ± 0.001 | 0.055 ± 0.000 | unsupported | 0.116 ± 0.001 | 0.192 ± 0.008 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.858 ± 0.003 | 0.869 ± 0.010 | unsupported | 1.765 ± 0.017 | 1.727 ± 0.017 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.096 ± 0.023 | 12.000 ± 1.172 | unsupported | 0.096 ± 0.003 | 0.130 ± 0.005 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.391 ± 0.003 | 11.999 ± 0.000 | unsupported | 0.698 ± 0.007 | 0.928 ± 0.184 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.056 ± 0.001 | 0.057 ± 0.001 | unsupported | 0.121 ± 0.002 | 0.170 ± 0.004 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.890 ± 0.012 | 0.895 ± 0.001 | unsupported | 1.829 ± 0.017 | 1.741 ± 0.031 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.028 ± 0.001 | 12.000 ± 0.187 | unsupported | 0.101 ± 0.003 | 0.131 ± 0.012 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.406 ± 0.002 | 11.999 ± 0.000 | unsupported | 0.733 ± 0.021 | 0.947 ± 0.024 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.124 ± 0.001 | 0.152 ± 0.003 | unsupported | 0.137 ± 0.004 | 0.215 ± 0.003 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.701 ± 0.020 | 1.761 ± 0.093 | unsupported | 1.229 ± 0.021 | 2.171 ± 0.035 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.140 ± 0.001 | 43.875 ± 0.109 | unsupported | 0.062 ± 0.001 | 0.199 ± 0.001 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 1.112 ± 0.178 | 118.876 ± 1.376 | unsupported | 0.467 ± 0.002 | 1.272 ± 0.024 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.183 ± 0.001 | 0.183 ± 0.000 | unsupported | 0.271 ± 0.002 | 0.311 ± 0.003 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 2.907 ± 0.003 | 2.899 ± 0.014 | unsupported | 4.261 ± 0.022 | 3.962 ± 0.038 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.072 ± 0.000 | 12.000 ± 0.009 | unsupported | 0.206 ± 0.003 | 0.205 ± 0.011 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 1.124 ± 0.007 | 11.999 ± 0.001 | unsupported | 1.581 ± 0.023 | 2.184 ± 0.404 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.759 ± 0.036 | 1.339 ± 0.015 | unsupported | 0.271 ± 0.007 | 0.233 ± 0.002 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 18.610 ± 0.524 | 29.567 ± 1.076 | unsupported | 5.624 ± 0.521 | 2.301 ± 0.029 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.849 ± 0.032 | 92.718 ± 0.663 | unsupported | 0.263 ± 0.009 | 0.160 ± 0.014 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 8.767 ± 0.333 | 154.248 ± 1.750 | unsupported | 2.226 ± 0.063 | 1.426 ± 0.274 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 28.670 ± 0.425 | - | unsupported | 28.073 ± 1.376 | - | 32.365 ± 1.407 | 40.859 ± 0.837 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 21.623 ± 0.090 | - | unsupported | 23.408 ± 1.311 | - | 31.251 ± 1.559 | 22.876 ± 2.165 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 20.759 ± 0.265 | - | unsupported | 27.183 ± 0.234 | - | 22.515 ± 0.394 | 25.871 ± 1.896 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 15.929 ± 1.665 | - | unsupported | 16.592 ± 0.627 | - | 24.053 ± 2.775 | 15.365 ± 0.621 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 14.178 ± 0.187 | - | unsupported | 23.542 ± 0.457 | - | 13.125 ± 0.340 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 16.148 ± 0.445 | - | unsupported | 17.246 ± 1.645 | - | 13.050 ± 0.329 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 27.034 ± 3.569 | - | unsupported | 30.940 ± 1.229 | - | 31.916 ± 0.349 | 40.476 ± 1.315 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 21.575 ± 0.430 | - | unsupported | 22.045 ± 0.703 | - | 30.233 ± 1.023 | 21.574 ± 0.343 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 48.485 ± 0.174 | - | unsupported | 49.748 ± 0.178 | - | 47.691 ± 0.227 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 53.708 ± 2.250 | - | unsupported | 25.975 ± 0.446 | - | 23.612 ± 0.164 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 48.282 ± 0.202 | - | unsupported | 49.771 ± 0.281 | - | 47.555 ± 0.135 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 53.441 ± 0.509 | - | unsupported | 26.090 ± 0.824 | - | 23.381 ± 0.251 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 28.835 ± 0.818 | - | unsupported | 27.955 ± 0.672 | - | 34.526 ± 1.509 | 41.388 ± 0.691 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 22.724 ± 2.645 | - | unsupported | 22.254 ± 0.544 | - | 30.219 ± 0.605 | 22.548 ± 2.851 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 22.368 ± 0.466 | - | unsupported | 23.649 ± 1.530 | - | 24.768 ± 0.318 | 29.081 ± 1.613 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 15.168 ± 0.510 | - | unsupported | 16.761 ± 1.730 | - | 23.288 ± 0.117 | 15.274 ± 0.668 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.743 ± 0.006 | unsupported | unsupported | 0.901 ± 0.014 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 11.030 ± 0.041 | unsupported | unsupported | 12.829 ± 0.145 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 1.473 ± 0.016 | unsupported | unsupported | 0.621 ± 0.010 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 195.793 ± 8.301 | unsupported | unsupported | 6.613 ± 0.273 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 28.849 ± 1.156 | - | unsupported | 29.116 ± 0.736 | - | 31.449 ± 0.569 | 40.665 ± 0.330 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 21.926 ± 2.326 | - | unsupported | 23.030 ± 1.065 | - | 30.237 ± 0.307 | 23.247 ± 2.509 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 107.762 ± 3.793 | - | 0.002 ± 0.000 | 107.427 ± 1.571 | 106.803 ± 2.733 | 131.172 ± 1.802 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 42.533 ± 0.839 | - | 0.002 ± 0.000 | 34.978 ± 0.237 | 54.431 ± 0.236 | 131.804 ± 3.823 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 67.893 ± 1.702 | - | 210.031 ± 8.599 | 68.994 ± 1.044 | 68.545 ± 2.282 | 76.269 ± 2.757 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 22.752 ± 0.306 | - | 258.248 ± 2.251 | 23.003 ± 0.344 | 36.337 ± 0.743 | 77.213 ± 1.668 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 216.045 ± 2.332 | - | 216.523 ± 4.247 | 216.464 ± 1.286 | 216.176 ± 2.987 | 244.337 ± 8.100 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 70.724 ± 0.909 | - | 216.019 ± 10.452 | 70.818 ± 0.656 | 110.498 ± 2.770 | 249.251 ± 4.024 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 67.682 ± 2.101 | - | 67.526 ± 2.495 | 73.785 ± 3.179 | 66.889 ± 2.923 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 42.105 ± 0.480 | - | 69.368 ± 2.533 | 25.259 ± 0.719 | 36.388 ± 1.003 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 136.115 ± 8.470 | - | 0.002 ± 0.000 | 124.771 ± 2.765 | 122.459 ± 4.514 | 154.283 ± 3.657 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 40.534 ± 0.442 | - | 0.002 ± 0.000 | 40.375 ± 0.648 | 49.484 ± 6.354 | 154.950 ± 2.606 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 85.873 ± 10.742 | - | 0.002 ± 0.000 | 78.351 ± 1.242 | 88.809 ± 5.584 | 214.121 ± 9.934 | 128.643 ± 2.219 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 28.144 ± 0.281 | - | 0.002 ± 0.000 | 78.938 ± 2.734 | 42.227 ± 0.885 | 211.491 ± 1.458 | 51.056 ± 1.053 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 61.061 ± 1.934 | - | 60.454 ± 1.736 | 64.310 ± 1.024 | 65.596 ± 3.594 | 79.375 ± 1.502 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 21.427 ± 0.479 | - | 70.740 ± 0.478 | 23.902 ± 0.292 | 36.059 ± 4.672 | 81.885 ± 1.924 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 60.065 ± 1.911 | - | 59.844 ± 1.382 | 64.198 ± 2.188 | 65.507 ± 2.003 | 80.491 ± 4.280 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 20.438 ± 0.271 | - | 71.091 ± 1.047 | 23.078 ± 0.353 | 32.960 ± 0.389 | 81.625 ± 1.824 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 8.019225 | 501201.56 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 7.186980 | 449186.25 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 7.917352 | 494834.50 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 6.023977 | 376498.56 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 6.167753 | 385484.56 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 827.991887 | 51749492.94 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 18.731235 | 1170702.19 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 19.796701 | 1237293.81 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 6.505543 | 406596.44 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 383.987481 | 23999217.56 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 11.742223 | 733888.94 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 9.686898 | 605431.12 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 7.837710 | 489856.88 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 8.133521 | 508345.06 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 154.717266 | 9669829.12 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 139.732721 | 8733295.06 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 124.483381 | 7780211.31 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 126.160521 | 7885032.56 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 20.013679 | 1250854.94 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 8.302791 | 518924.44 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 12.425275 | 776579.69 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 12.795807 | 799737.94 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 152.786094 | 9549130.88 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 90.632818 | 5664551.12 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 36411.002783 | 2275687673.94 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 36931.391544 | 2308211971.50 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 18.869761 | 1179360.06 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 19.018399 | 1188649.94 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 340.920235 | 21307514.69 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 342.025964 | 21376622.75 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 26.026355 | 1626647.19 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 25.626639 | 1601664.94 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 374.282928 | 23392683.00 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 372.278331 | 23267395.69 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 17.521919 | 1095119.94 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 17.690604 | 1105662.75 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 324.831443 | 20301965.19 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 324.428091 | 20276755.69 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 24.315843 | 1519740.19 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 24.744022 | 1546501.38 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 358.092280 | 22380767.50 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 357.606908 | 22350431.75 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.333284 | 36457.56 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 10.001965 | 156280.70 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.963596 | 140056.19 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.921577 | 139399.64 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 9.730497 | 1216312.12 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 19.483559 | 2435444.88 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 17.875445 | 2234430.62 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 17.778324 | 2222290.50 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.709524 | 136086.31 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.088045 | 79500.70 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.771387 | 74552.92 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 139.865178 | 2185393.41 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.414244 | 426780.50 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.109962 | 1013745.25 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.171817 | 896477.12 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 191.993337 | 23999167.12 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 15.854726 | 247730.09 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.482219 | 132534.67 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.069450 | 79210.16 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.055623 | 78994.11 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 20.940216 | 2617527.00 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 13.852280 | 1731535.00 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.016971 | 1252121.38 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 9.972839 | 1246604.88 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.067243 | 157300.67 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.702850 | 89107.03 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.388552 | 37321.12 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.977423 | 11999647.23 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.686191 | 1085773.88 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.850385 | 731298.12 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.478793 | 559849.12 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.994585 | 11999323.12 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.825717 | 184776.83 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 18.059742 | 282183.47 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 12.604980 | 196952.81 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 12.643963 | 197561.92 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.698639 | 1837329.88 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 35.200699 | 4400087.38 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 25.087903 | 3135987.88 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 24.939126 | 3117390.75 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.301053 | 129703.95 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.867789 | 107309.20 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.867978 | 76062.16 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 803.986539 | 12562289.67 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.759904 | 969988.00 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 11.984917 | 1498114.62 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.422219 | 1177777.38 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.994246 | 11999280.75 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.402117 | 178158.08 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 9.940690 | 155323.28 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.479313 | 54364.27 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.557149 | 86830.45 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 15.011723 | 1876465.38 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.244840 | 905605.00 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.121767 | 640220.88 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.617071 | 702133.88 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.360816 | 177512.75 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.412857 | 68950.89 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.425096 | 84767.12 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2303.564141 | 35993189.70 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.535123 | 1191890.38 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.908326 | 363540.75 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.514353 | 439294.12 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 767.996107 | 95999513.38 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.916660 | 217447.81 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.370501 | 130789.08 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 7.923455 | 123803.98 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.643715 | 150683.05 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.441323 | 2305165.38 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.911166 | 1238895.75 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.395516 | 1674439.50 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 13.920665 | 1740083.12 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.183794 | 205996.78 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.084386 | 63818.53 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 9.242655 | 144416.48 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2807.007402 | 43859490.66 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.215104 | 1401888.00 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.841994 | 480249.25 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.185603 | 1148200.38 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 959.003293 | 119875411.62 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 15.350425 | 239850.39 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 18.563548 | 290055.44 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 21.513142 | 336142.84 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 24.910104 | 3113763.00 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 34.635860 | 4329482.50 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 35.454477 | 4431809.62 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 15.092422 | 235819.09 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 18.342683 | 286604.42 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3319.967234 | 51874488.03 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 15.056805 | 1882100.62 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 16.804141 | 2100517.62 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1151.997394 | 143999674.25 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 3.165668 | 49463.56 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.839477 | 13116.83 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.680871 | 10638.61 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.190922 | 18608.16 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 1.972120 | 246515.00 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.730804 | 91350.50 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.804091 | 100511.38 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.888829 | 111103.62 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.388392 | 37318.62 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.928799 | 14512.48 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.981488 | 15335.75 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 381.970322 | 5968286.28 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.719385 | 214923.12 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.573007 | 71625.88 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.902299 | 112787.38 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 191.996376 | 23999547.00 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.583125 | 180986.33 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.259370 | 97802.66 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.866232 | 29159.88 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.087010 | 48234.53 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.593547 | 1699193.38 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.578995 | 572374.38 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.951868 | 368983.50 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.033901 | 379237.62 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.700450 | 151569.53 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.495425 | 101491.02 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.890915 | 45170.55 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1150.997051 | 17984328.92 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.798420 | 1099802.50 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.263995 | 282999.38 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.836468 | 229558.50 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 383.996094 | 47999511.75 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.286578 | 191977.78 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.404478 | 115694.97 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.473551 | 54274.23 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.504580 | 54759.06 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.816352 | 1727044.00 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.122826 | 1765353.25 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.860241 | 857530.12 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.949146 | 868643.25 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.309758 | 129839.97 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.115298 | 95551.53 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.131232 | 95800.50 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.979404 | 11999678.19 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.423828 | 927978.50 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.587515 | 698439.38 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.129055 | 391131.88 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.993613 | 11999201.62 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.889441 | 170147.52 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.738381 | 120912.20 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.595750 | 56183.59 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.616689 | 56510.77 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.931658 | 1741457.25 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.631114 | 1828889.25 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.119695 | 889961.88 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.163566 | 895445.75 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.396968 | 131202.62 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.471059 | 101110.30 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.785124 | 27892.56 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.984320 | 11999755.00 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.572535 | 946566.88 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.862296 | 732787.00 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.248719 | 406089.88 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.994955 | 11999369.38 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.775836 | 215247.44 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.785936 | 137280.25 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 7.924627 | 123822.30 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.717833 | 151841.14 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 17.370555 | 2171319.38 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.834002 | 1229250.25 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.607310 | 1700913.75 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 14.089180 | 1761147.50 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.715530 | 198680.16 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.986352 | 62286.75 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.942403 | 139725.05 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2807.983064 | 43874735.38 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.173633 | 1271704.12 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.737720 | 467215.00 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.899844 | 1112480.50 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 951.005487 | 118875685.88 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 19.930722 | 311417.53 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.343365 | 270990.08 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 11.686657 | 182604.02 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 11.709189 | 182956.08 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 31.693312 | 3961664.00 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 34.084747 | 4260593.38 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 23.256946 | 2907118.25 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 23.192336 | 2899042.00 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.093260 | 204582.19 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 13.182867 | 205982.30 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.639522 | 72492.53 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.976518 | 11999633.09 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 17.471586 | 2183948.25 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.646509 | 1580813.62 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.993844 | 1124230.50 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.994384 | 11999298.00 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.940700 | 233448.44 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.354707 | 271167.30 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 48.569789 | 758902.95 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 85.724367 | 1339443.23 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.406528 | 2300816.00 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 44.991521 | 5623940.12 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 148.883836 | 18610479.50 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 236.535444 | 29566930.50 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.241089 | 160017.02 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 16.855838 | 263372.47 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 54.347215 | 849175.23 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5933.972454 | 92718319.59 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.409059 | 1426132.38 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 17.804183 | 2225522.88 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 70.137281 | 8767160.12 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1233.981232 | 154247654.00 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 14.419549 | 901221.81 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 11.885627 | 742851.69 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 205.267307 | 12829206.69 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 176.477190 | 11029824.38 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 9.928027 | 620501.69 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 23.561867 | 1472616.69 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 105.813196 | 6613324.75 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 3132.680486 | 195792530.38 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.023444 | 1465.25 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002454 | 153.38 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.023664 | 1479.00 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002725 | 170.31 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000120 | 7.50 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.021801 | 1362.56 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.007634 | 477.12 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000140 | 8.75 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.021831 | 1364.44 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.006873 | 429.56 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000120 | 7.50 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.022392 | 1399.50 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.005831 | 364.44 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000131 | 8.19 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.022381 | 1398.81 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.005670 | 354.38 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000110 | 6.88 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.020389 | 1274.31 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003416 | 213.50 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000110 | 6.88 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.020508 | 1281.75 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003507 | 219.19 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/norm_fro` (c64, threads=1, shape=`2048x1536`): `tenferro-direct` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 24.169 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `jax-cpu` is 23.3x faster than the slowest successful cell (`tenferro-trace`, 60.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `julia-base` is 14.5x faster than the slowest successful cell (`tenferro-trace`, 60.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 49.0x faster than the slowest successful cell (`tenferro-trace`, 60.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 20.2x faster than the slowest successful cell (`tenferro-direct`, 140.005 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `julia-base` is 19.0x faster than the slowest successful cell (`tenferro-direct`, 140.005 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 24.4x faster than the slowest successful cell (`tenferro-direct`, 140.005 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 31.5x faster than the slowest successful cell (`tenferro-trace`, 146.027 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `julia-base` is 33.9x faster than the slowest successful cell (`tenferro-trace`, 146.027 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 40.6x faster than the slowest successful cell (`tenferro-trace`, 146.027 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/masked_log_softmax_axis1` (f64, threads=4, shape=`1024x64`): `jax-cpu` is 137.4x faster than the slowest successful cell (`tenferro-eager`, 51.749 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/masked_log_softmax_axis1` (f64, threads=4, shape=`1024x64`): `tenferro-direct` is 134.2x faster than the slowest successful cell (`tenferro-eager`, 51.749 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 16.2x faster than the slowest successful cell (`pytorch-cpu`, 23.940 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 23.940 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 23.940 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 13.2x faster than the slowest successful cell (`tenferro-trace`, 24.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 15.8x faster than the slowest successful cell (`tenferro-trace`, 24.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 30.1x faster than the slowest successful cell (`tenferro-trace`, 24.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 15.0x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 17.1x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 18.0x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 17.1x faster than the slowest successful cell (`pytorch-cpu`, 23.020 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 23.020 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 23.020 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 15.9x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 18.6x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 30.1x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 19.7x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 18.7x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 22.9x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 42.2x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 14.0x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 18.2x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 61.9x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 29.2x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 19.5x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 19.0x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 23.0x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 45.0x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 16.4x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 18.6x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 54.5x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 32.7x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_squares_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 59.0x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `jax-cpu` is 11.2x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `pytorch-cpu` is 37.0x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `tenferro-direct` is 27.3x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `jax-cpu` is 14.1x faster than the slowest successful cell (`tenferro-trace`, 11.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `pytorch-cpu` is 37.3x faster than the slowest successful cell (`tenferro-trace`, 11.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 11.1x faster than the slowest successful cell (`julia-base`, 8.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`julia-base`, 8.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `jax-cpu` is 50.4x faster than the slowest successful cell (`tenferro-trace`, 19.448 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `julia-base` is 16.2x faster than the slowest successful cell (`tenferro-trace`, 19.448 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 56.4x faster than the slowest successful cell (`tenferro-trace`, 19.448 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `tenferro-direct` is 11.4x faster than the slowest successful cell (`tenferro-trace`, 19.448 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `jax-cpu` is 15.9x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 22.9x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 18.7x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `jax-cpu` is 20.5x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 38.0x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 26.2x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `jax-cpu` is 31.4x faster than the slowest successful cell (`tenferro-trace`, 16.537 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `julia-base` is 12.3x faster than the slowest successful cell (`tenferro-trace`, 16.537 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 27.3x faster than the slowest successful cell (`tenferro-trace`, 16.537 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 4708.5x faster than the slowest successful cell (`julia-base`, 8.777 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `pytorch-cpu` is 10.1x faster than the slowest successful cell (`julia-base`, 9.198 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 4135.8x faster than the slowest successful cell (`julia-base`, 9.198 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 14.8x faster than the slowest successful cell (`jax-cpu`, 0.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.011 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`3x2x2`): `tenferro-direct` is 13.3x faster than the slowest successful cell (`jax-cpu`, 0.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`3x4x4`): `tenferro-direct` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x4x4`): `tenferro-direct` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 14.0x faster than the slowest successful cell (`jax-cpu`, 0.026 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 12.5x faster than the slowest successful cell (`jax-cpu`, 0.034 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=1, shape=`1x2x2`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.022 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.045 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.020 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.024 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x2x2,rhs=1`): `tenferro-direct` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.018 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `pytorch-cpu` is 15.6x faster than the slowest successful cell (`tenferro-trace`, 5.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `tenferro-direct` is 10.8x faster than the slowest successful cell (`tenferro-trace`, 5.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x4x4,rhs=1`): `tenferro-direct` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.025 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `jax-cpu` is 32.8x faster than the slowest successful cell (`tenferro-trace`, 24.006 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `pytorch-cpu` is 41.7x faster than the slowest successful cell (`tenferro-trace`, 24.006 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `tenferro-direct` is 26.3x faster than the slowest successful cell (`tenferro-trace`, 24.006 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 22.5x faster than the slowest successful cell (`tenferro-trace`, 313.973 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `julia-base` is 25.0x faster than the slowest successful cell (`tenferro-trace`, 313.973 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 27.8x faster than the slowest successful cell (`tenferro-trace`, 313.973 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`768x384`): `jax-cpu` is 241.7x faster than the slowest successful cell (`tenferro-eager`, 2308.212 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 407.5x faster than the slowest successful cell (`tenferro-eager`, 2308.212 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 14.3x faster than the slowest successful cell (`tenferro-trace`, 282.313 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `julia-base` is 16.5x faster than the slowest successful cell (`tenferro-trace`, 282.313 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 13.1x faster than the slowest successful cell (`tenferro-trace`, 282.313 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `jax-cpu` is 53.2x faster than the slowest successful cell (`tenferro-eager`, 2291.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `julia-base` is 244.3x faster than the slowest successful cell (`tenferro-eager`, 2291.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 368.8x faster than the slowest successful cell (`tenferro-eager`, 2291.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `julia-base` is 10.7x faster than the slowest successful cell (`tenferro-direct`, 250.164 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 15.6x faster than the slowest successful cell (`tenferro-direct`, 250.164 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `pytorch-cpu` is 12.8x faster than the slowest successful cell (`jax-cpu`, 20.325 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 15.0x faster than the slowest successful cell (`jax-cpu`, 20.325 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 15.2x faster than the slowest successful cell (`jax-cpu`, 20.325 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 16.0x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 46.4x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `jax-cpu` is 11.7x faster than the slowest successful cell (`tenferro-trace`, 238.013 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `julia-base` is 11.2x faster than the slowest successful cell (`tenferro-trace`, 238.013 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 14.5x faster than the slowest successful cell (`tenferro-trace`, 238.013 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `jax-cpu` is 10.6x faster than the slowest successful cell (`tenferro-trace`, 244.186 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `julia-base` is 12.4x faster than the slowest successful cell (`tenferro-trace`, 244.186 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 14.8x faster than the slowest successful cell (`tenferro-trace`, 244.186 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 23.0x faster than the slowest successful cell (`tenferro-trace`, 324.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `julia-base` is 25.4x faster than the slowest successful cell (`tenferro-trace`, 324.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 28.5x faster than the slowest successful cell (`tenferro-trace`, 324.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 43.1x faster than the slowest successful cell (`tenferro-eager`, 2335.368 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `julia-base` is 42.5x faster than the slowest successful cell (`tenferro-eager`, 2335.368 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 60.4x faster than the slowest successful cell (`tenferro-eager`, 2335.368 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 16.1x faster than the slowest successful cell (`tenferro-eager`, 2.185 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 27.5x faster than the slowest successful cell (`tenferro-eager`, 2.185 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 29.3x faster than the slowest successful cell (`tenferro-eager`, 2.185 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 56.2x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 23.7x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 26.8x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 76.3x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 134.7x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 321.5x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 11.1x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 16.4x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 21.4x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 96.9x faster than the slowest successful cell (`tenferro-eager`, 12.562 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 117.1x faster than the slowest successful cell (`tenferro-eager`, 12.562 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 165.2x faster than the slowest successful cell (`tenferro-eager`, 12.562 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 12.4x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 202.8x faster than the slowest successful cell (`tenferro-eager`, 35.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 522.0x faster than the slowest successful cell (`tenferro-eager`, 35.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 424.6x faster than the slowest successful cell (`tenferro-eager`, 35.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 80.5x faster than the slowest successful cell (`tenferro-eager`, 96.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 264.1x faster than the slowest successful cell (`tenferro-eager`, 96.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 218.5x faster than the slowest successful cell (`tenferro-eager`, 96.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 212.9x faster than the slowest successful cell (`tenferro-eager`, 43.859 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 687.2x faster than the slowest successful cell (`tenferro-eager`, 43.859 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 303.7x faster than the slowest successful cell (`tenferro-eager`, 43.859 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 85.5x faster than the slowest successful cell (`tenferro-eager`, 119.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 249.6x faster than the slowest successful cell (`tenferro-eager`, 119.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 104.4x faster than the slowest successful cell (`tenferro-eager`, 119.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 220.0x faster than the slowest successful cell (`tenferro-eager`, 51.874 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 181.0x faster than the slowest successful cell (`tenferro-eager`, 51.874 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 76.5x faster than the slowest successful cell (`tenferro-eager`, 144.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 68.6x faster than the slowest successful cell (`tenferro-eager`, 144.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 159.9x faster than the slowest successful cell (`tenferro-eager`, 5.968 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 411.3x faster than the slowest successful cell (`tenferro-eager`, 5.968 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 389.2x faster than the slowest successful cell (`tenferro-eager`, 5.968 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 111.7x faster than the slowest successful cell (`tenferro-eager`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 335.1x faster than the slowest successful cell (`tenferro-eager`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 212.8x faster than the slowest successful cell (`tenferro-eager`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 118.7x faster than the slowest successful cell (`tenferro-eager`, 17.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 177.2x faster than the slowest successful cell (`tenferro-eager`, 17.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 398.1x faster than the slowest successful cell (`tenferro-eager`, 17.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 43.6x faster than the slowest successful cell (`tenferro-eager`, 48.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 169.6x faster than the slowest successful cell (`tenferro-eager`, 48.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 209.1x faster than the slowest successful cell (`tenferro-eager`, 48.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 92.4x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 125.6x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 125.3x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 12.9x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 17.2x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 30.7x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 91.5x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 118.7x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 430.2x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 12.7x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 16.4x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 29.5x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 220.8x faster than the slowest successful cell (`tenferro-eager`, 43.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 704.4x faster than the slowest successful cell (`tenferro-eager`, 43.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 314.0x faster than the slowest successful cell (`tenferro-eager`, 43.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 93.5x faster than the slowest successful cell (`tenferro-eager`, 118.876 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 254.4x faster than the slowest successful cell (`tenferro-eager`, 118.876 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 106.9x faster than the slowest successful cell (`tenferro-eager`, 118.876 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 58.7x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 58.3x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 165.5x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 12.9x faster than the slowest successful cell (`tenferro-eager`, 29.567 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 579.4x faster than the slowest successful cell (`tenferro-eager`, 92.718 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 352.0x faster than the slowest successful cell (`tenferro-eager`, 92.718 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 109.2x faster than the slowest successful cell (`tenferro-eager`, 92.718 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 108.2x faster than the slowest successful cell (`tenferro-eager`, 154.248 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 69.3x faster than the slowest successful cell (`tenferro-eager`, 154.248 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 17.6x faster than the slowest successful cell (`tenferro-eager`, 154.248 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/output_reuse/solve_read_into` (f64, threads=4, shape=`768x768,rhs=16`): `pytorch-cpu` is 29.6x faster than the slowest successful cell (`tenferro-direct`, 195.793 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 73983.2x faster than the slowest successful cell (`julia-base`, 131.172 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 53275.6x faster than the slowest successful cell (`julia-base`, 131.804 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/cast_f64_f32` (f64->f32, threads=4, shape=`33554432`): `pytorch-cpu` is 11.2x faster than the slowest successful cell (`tenferro-trace`, 258.248 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/cast_f64_f32` (f64->f32, threads=4, shape=`33554432`): `tenferro-direct` is 11.4x faster than the slowest successful cell (`tenferro-trace`, 258.248 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 85570.1x faster than the slowest successful cell (`julia-base`, 154.283 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 86419.3x faster than the slowest successful cell (`julia-base`, 154.950 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 127986.3x faster than the slowest successful cell (`julia-base`, 214.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 126413.9x faster than the slowest successful cell (`julia-base`, 211.491 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 170.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 151.6x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 174.9x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 174.9x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 182.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 183.1x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 8996.3 GB/s at 0.002 ms. The conservative physical lower bound is 0.168 ms, so the cell is 90.0x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 7540.0 GB/s at 0.002 ms. The conservative physical lower bound is 0.042 ms, so the cell is 18.9x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 151401.8 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1514.0x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 108502.6 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 271.3x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 148882.7 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1488.8x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 149713.0 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 374.3x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 80225.8 GB/s at 0.002 ms. The conservative physical lower bound is 1.342 ms, so the cell is 802.3x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 80225.8 GB/s at 0.002 ms. The conservative physical lower bound is 0.336 ms, so the cell is 200.6x faster than the bound.

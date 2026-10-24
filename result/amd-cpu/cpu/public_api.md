# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`
- Run metadata: `data/results/amd-cpu/cpu/public_api/20261024_184901/run.yaml`
- Timestamp: `20261024_184901`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Sampling: `quick` profile, 7 measured runs, 3 warmups per row

This file is generated from sequential CPU public API runs under `data/results/amd-cpu/cpu/public_api/20261024_184901`.

- tenferro-rs commit: `fa39e8d78d9c58e36575084b67448a8112ec1d6c`

## CPU Information

- Model: `AMD EPYC 9V74 80-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`

## Thread Environments

### Threads: 1

- Run metadata: `data/results/amd-cpu/cpu/public_api/20261024_184901/run_t1.yaml`
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

- Run metadata: `data/results/amd-cpu/cpu/public_api/20261024_184901/run_t4.yaml`
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

- CSV: `data/results/amd-cpu/cpu/public_api/20261024_184901/cpu_public_api_t1_20261024_184901.csv`
- CSV: `data/results/amd-cpu/cpu/public_api/20261024_184901/cpu_public_api_t4_20261024_184901.csv`
- Source table: `data/results/amd-cpu/cpu/public_api/20261024_184901/cpu_public_api_20261024_184901.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 3.526 ± 0.015 | - | 3.584 ± 0.028 | 3.696 ± 0.027 | 3.529 ± 0.008 | 3.605 ± 1.279 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.106 ± 0.013 | - | 2.143 ± 0.017 | 1.889 ± 0.038 | 2.614 ± 0.138 | 4.065 ± 1.576 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 21.399 ± 0.366 | - | 21.211 ± 0.152 | 20.736 ± 0.168 | 20.380 ± 0.108 | 21.578 ± 0.060 | 23.674 ± 0.283 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 16.143 ± 0.275 | - | 15.781 ± 0.269 | 16.246 ± 0.334 | 16.450 ± 0.178 | 22.631 ± 1.970 | 17.273 ± 0.261 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 44.762 ± 0.174 | - | 47.502 ± 0.236 | 37.367 ± 0.124 | 30.900 ± 0.301 | 52.852 ± 0.126 | 58.996 ± 0.103 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 20.490 ± 0.098 | - | 21.572 ± 6.755 | 15.641 ± 0.234 | 16.828 ± 0.099 | 53.549 ± 0.098 | 26.625 ± 0.250 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 40.216 ± 0.098 | - | 40.233 ± 0.150 | 39.628 ± 0.032 | 56.537 ± 0.031 | 41.498 ± 0.470 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 25.192 ± 0.078 | - | 25.198 ± 0.024 | 25.607 ± 0.220 | 25.664 ± 0.122 | 20.554 ± 0.218 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 41.184 ± 0.133 | - | unsupported | 40.463 ± 0.260 | 56.943 ± 0.237 | 41.032 ± 0.304 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 25.907 ± 0.020 | - | unsupported | 26.163 ± 0.259 | 25.991 ± 0.592 | 20.681 ± 1.534 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 9.137 ± 0.041 | - | 9.066 ± 0.022 | 9.484 ± 0.076 | 9.606 ± 0.024 | 9.537 ± 0.077 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 11.881 ± 0.009 | - | 11.947 ± 0.036 | 11.642 ± 0.082 | 14.670 ± 0.201 | 14.739 ± 0.016 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 54.833 ± 0.154 | - | 54.772 ± 7.812 | 72.991 ± 0.089 | 62.544 ± 0.076 | 42.082 ± 2.253 | 42.760 ± 1.462 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 25.077 ± 0.059 | - | 25.444 ± 0.065 | 32.414 ± 0.154 | 31.809 ± 0.264 | 40.381 ± 0.135 | 20.851 ± 1.132 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 88.147 ± 0.191 | - | 87.706 ± 0.141 | 105.136 ± 0.180 | 117.341 ± 0.566 | 92.257 ± 0.304 | 93.719 ± 0.263 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 39.037 ± 0.074 | - | 39.014 ± 0.057 | 47.283 ± 0.069 | 54.549 ± 1.556 | 96.074 ± 0.353 | 40.074 ± 1.843 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 16.187 ± 0.162 | - | 16.276 ± 0.181 | 14.896 ± 0.137 | 14.129 ± 0.221 | 15.201 ± 0.248 | 16.491 ± 0.045 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 12.243 ± 0.364 | - | 11.807 ± 0.225 | 12.235 ± 0.111 | 12.067 ± 0.110 | 15.078 ± 0.396 | 12.320 ± 0.046 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.412 ± 0.038 | - | 16.034 ± 0.053 | 18.196 ± 0.070 | 5.148 ± 0.026 | 3.460 ± 0.010 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 0.938 ± 0.045 | - | 12.599 ± 0.071 | 9.431 ± 0.011 | 2.993 ± 0.158 | 3.456 ± 0.011 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 7.067 ± 0.029 | - | 7.457 ± 0.099 | 7.880 ± 0.014 | 6.625 ± 0.074 | 8.080 ± 0.034 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 4.617 ± 0.131 | - | 4.792 ± 0.048 | 4.887 ± 0.025 | 6.283 ± 0.019 | 5.854 ± 0.678 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 4.597 ± 0.152 | - | 4.725 ± 0.058 | 4.701 ± 0.027 | 4.534 ± 0.038 | 4.201 ± 0.031 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.297 ± 0.126 | - | 3.254 ± 0.138 | 2.914 ± 0.033 | 3.311 ± 0.085 | 2.987 ± 0.054 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 5.173 ± 0.074 | - | 5.154 ± 0.011 | 6.303 ± 0.055 | 4.340 ± 0.012 | 4.258 ± 0.060 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 4.679 ± 0.148 | - | 4.676 ± 0.126 | 4.977 ± 0.021 | 5.268 ± 0.025 | 5.087 ± 0.027 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 40.395 ± 0.359 | - | 39.948 ± 0.058 | 39.955 ± 0.125 | 56.585 ± 0.118 | 40.714 ± 0.065 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 25.108 ± 0.117 | - | 25.074 ± 0.156 | 25.884 ± 0.060 | 25.619 ± 0.249 | 20.313 ± 0.097 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 43.173 ± 0.379 | - | unsupported | 46.920 ± 0.216 | 45.099 ± 0.391 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 22.232 ± 0.074 | - | unsupported | 23.625 ± 0.056 | 26.434 ± 0.576 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 20.706 ± 0.190 | - | 70.144 ± 0.138 | 20.543 ± 0.257 | 22.094 ± 0.396 | 21.008 ± 0.257 | 38.416 ± 0.104 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 16.303 ± 0.190 | - | 15.603 ± 0.196 | 16.134 ± 0.225 | 16.459 ± 0.231 | 21.748 ± 0.614 | 21.134 ± 0.424 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 26.646 ± 0.208 | - | 83.038 ± 1.032 | 26.986 ± 0.306 | 30.572 ± 0.483 | 26.218 ± 0.605 | 53.808 ± 2.067 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 23.037 ± 0.303 | - | 22.161 ± 0.102 | 22.918 ± 0.271 | 23.125 ± 0.157 | 29.034 ± 0.684 | 27.682 ± 1.012 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 13.679 ± 0.060 | - | 37.548 ± 0.269 | 23.226 ± 0.166 | 30.383 ± 0.063 | 52.087 ± 1.241 | 57.576 ± 3.132 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 7.932 ± 0.307 | - | 18.304 ± 0.215 | 14.686 ± 0.176 | 15.633 ± 0.065 | 52.436 ± 0.109 | 24.834 ± 2.006 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 7.707 ± 0.100 | - | 7.801 ± 0.089 | 9.483 ± 0.225 | 9.529 ± 0.048 | 8.909 ± 0.708 | 16.203 ± 1.375 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 6.768 ± 0.069 | - | 6.955 ± 0.052 | 8.201 ± 0.215 | 7.570 ± 0.062 | 9.688 ± 0.860 | 8.891 ± 7.281 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 18.393 ± 0.144 | - | 23.662 ± 0.779 | 23.543 ± 0.302 | 20.135 ± 0.305 | 16.212 ± 0.283 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 15.250 ± 0.210 | - | 15.434 ± 0.136 | 16.673 ± 0.151 | 17.052 ± 0.216 | 17.362 ± 0.518 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 13.391 ± 0.012 | - | 13.440 ± 0.011 | 28.510 ± 0.035 | 70.781 ± 0.034 | 32.557 ± 1.223 | 38.745 ± 1.708 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 5.902 ± 0.051 | - | 5.994 ± 0.014 | 13.779 ± 0.163 | 35.655 ± 0.099 | 32.885 ± 4.593 | 18.972 ± 1.881 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 26.828 ± 0.123 | - | 100.303 ± 0.295 | 26.637 ± 0.395 | 28.971 ± 0.439 | 26.281 ± 0.289 | 52.510 ± 0.157 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 22.786 ± 0.525 | - | 22.914 ± 0.386 | 22.768 ± 0.470 | 23.144 ± 0.134 | 29.387 ± 1.113 | 27.285 ± 1.829 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 7.930 ± 0.128 | - | 7.992 ± 0.018 | 17.105 ± 0.072 | 9.812 ± 0.034 | 29.000 ± 1.218 | 33.827 ± 1.305 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 3.840 ± 0.017 | - | 3.884 ± 0.046 | 8.787 ± 0.195 | 5.436 ± 0.041 | 29.862 ± 1.447 | 17.569 ± 1.469 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 10.763 ± 0.028 | - | 10.777 ± 0.020 | 12.039 ± 0.058 | 9.239 ± 0.039 | 20.244 ± 1.218 | 22.388 ± 2.951 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 4.642 ± 0.031 | - | 4.662 ± 0.017 | 6.064 ± 0.070 | 4.916 ± 0.010 | 20.094 ± 0.624 | 10.501 ± 2.670 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 13.773 ± 0.024 | - | 13.799 ± 0.029 | 22.368 ± 0.194 | 31.699 ± 0.267 | 41.737 ± 9.081 | 47.036 ± 0.836 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 6.926 ± 0.056 | - | 6.899 ± 0.089 | 11.124 ± 0.156 | 15.984 ± 0.146 | 41.643 ± 1.720 | 22.374 ± 1.629 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 6.852 ± 0.011 | - | 6.885 ± 0.048 | 10.008 ± 0.096 | 19.301 ± 0.067 | 25.800 ± 1.118 | 27.690 ± 2.607 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 3.426 ± 0.022 | - | 3.451 ± 0.014 | 5.217 ± 0.085 | 10.267 ± 0.179 | 25.666 ± 0.141 | 13.186 ± 2.516 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.175 ± 0.010 | 0.204 ± 0.007 | unsupported | unsupported | 0.294 ± 0.008 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.281 ± 0.026 | 0.394 ± 0.041 | unsupported | unsupported | 0.184 ± 0.012 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 28.476 ± 0.151 | - | 107.018 ± 1.067 | 27.582 ± 0.287 | 30.592 ± 0.208 | 27.406 ± 0.278 | 53.815 ± 0.124 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 22.861 ± 0.289 | - | 23.210 ± 0.343 | 22.895 ± 0.326 | 22.967 ± 0.316 | 28.401 ± 0.631 | 27.492 ± 0.389 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 28.324 ± 0.166 | - | 106.478 ± 0.568 | 27.550 ± 0.276 | 30.689 ± 0.587 | 27.094 ± 0.765 | 53.544 ± 0.095 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 22.867 ± 0.578 | - | 22.894 ± 0.505 | 22.805 ± 0.591 | 22.925 ± 0.186 | 28.244 ± 0.652 | 27.528 ± 0.250 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 26.919 ± 0.119 | - | 78.946 ± 3.247 | 26.170 ± 0.353 | 30.671 ± 0.404 | 26.205 ± 0.105 | 51.115 ± 0.346 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 22.767 ± 0.701 | - | 22.306 ± 0.193 | 22.774 ± 0.283 | 23.142 ± 0.126 | 28.078 ± 0.610 | 26.897 ± 0.644 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 20.630 ± 0.121 | - | 70.029 ± 1.083 | 20.555 ± 0.271 | 22.319 ± 0.525 | 20.807 ± 0.057 | 38.534 ± 0.164 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 16.235 ± 0.344 | - | 15.517 ± 0.081 | 16.066 ± 0.194 | 16.297 ± 0.189 | 22.079 ± 0.607 | 21.227 ± 0.797 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 42.520 ± 0.162 | - | 43.130 ± 0.059 | 24.391 ± 0.036 | 46.315 ± 0.087 | 97.530 ± 5.842 | 86.273 ± 4.298 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 19.932 ± 0.029 | - | 20.060 ± 0.316 | 10.766 ± 0.068 | 23.148 ± 0.075 | 97.436 ± 0.200 | 34.778 ± 2.435 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 11.514 ± 0.840 | - | 11.727 ± 0.556 | 8.770 ± 0.058 | 37.656 ± 0.584 | 9.212 ± 0.120 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 7.390 ± 0.248 | - | 7.467 ± 0.032 | 7.171 ± 0.259 | 18.583 ± 0.029 | 9.329 ± 0.046 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.480 ± 0.010 | - | 1.482 ± 0.012 | 26.742 ± 0.154 | 2.107 ± 0.027 | 1.031 ± 0.051 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.716 ± 0.027 | - | 0.713 ± 0.013 | 9.156 ± 0.672 | 1.458 ± 0.387 | 1.072 ± 0.012 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 1.765 ± 0.030 | - | 1.749 ± 0.012 | 4.631 ± 0.009 | 3.371 ± 0.007 | 1.086 ± 0.309 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 1.092 ± 0.036 | - | 1.127 ± 0.117 | 2.356 ± 0.023 | 2.534 ± 0.505 | 0.922 ± 0.549 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 10.990 ± 0.018 | - | 11.028 ± 0.040 | 8.834 ± 0.049 | 37.538 ± 0.111 | 9.093 ± 0.206 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 7.388 ± 0.057 | - | 7.496 ± 0.032 | 7.153 ± 0.070 | 18.613 ± 0.027 | 9.317 ± 0.098 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.437 ± 0.029 | - | 1.456 ± 0.026 | 26.225 ± 0.057 | 2.040 ± 0.010 | 0.971 ± 0.048 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.705 ± 0.016 | - | 0.700 ± 0.028 | 9.084 ± 0.250 | 1.164 ± 0.056 | 1.061 ± 0.040 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 7.494 ± 0.051 | - | 7.628 ± 0.191 | 13.851 ± 0.068 | 13.420 ± 0.050 | 6.245 ± 0.120 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 4.501 ± 0.050 | - | 4.605 ± 0.445 | 6.991 ± 0.018 | 7.581 ± 3.135 | 6.209 ± 0.127 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 8.266 ± 0.090 | - | 8.158 ± 0.031 | 7.887 ± 0.051 | 26.493 ± 0.063 | 7.661 ± 0.183 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 7.626 ± 0.072 | - | 7.683 ± 0.407 | 7.251 ± 0.138 | 13.326 ± 0.034 | 8.187 ± 0.160 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.839 ± 0.025 | - | 0.841 ± 0.026 | 2.283 ± 0.125 | 1.546 ± 0.016 | 0.581 ± 0.055 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.394 ± 0.177 | - | 0.481 ± 0.205 | 1.078 ± 0.030 | 0.951 ± 0.045 | 0.662 ± 0.074 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.425 ± 0.018 | - | 1.448 ± 0.031 | 0.700 ± 0.017 | 1.068 ± 0.004 | 0.777 ± 0.100 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.745 ± 0.088 | - | 0.774 ± 0.027 | 0.316 ± 0.035 | 0.650 ± 0.081 | 0.878 ± 0.249 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 8.164 ± 0.078 | - | 8.192 ± 0.241 | 7.925 ± 0.055 | 26.427 ± 0.032 | 7.601 ± 0.173 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 7.719 ± 0.060 | - | 7.543 ± 0.090 | 7.161 ± 0.192 | 13.372 ± 0.025 | 8.134 ± 0.177 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.842 ± 0.042 | - | 0.856 ± 0.033 | 2.414 ± 0.176 | 1.542 ± 0.033 | 0.573 ± 0.069 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.464 ± 0.178 | - | 0.422 ± 0.117 | 1.089 ± 0.037 | 0.911 ± 0.055 | 0.672 ± 0.086 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.446 ± 0.026 | - | 1.431 ± 0.015 | 0.706 ± 0.019 | 1.082 ± 0.014 | 0.741 ± 0.065 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.754 ± 0.042 | - | 0.780 ± 0.080 | 0.327 ± 0.029 | 0.635 ± 0.060 | 1.092 ± 0.090 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.888 ± 0.010 | 0.878 ± 0.010 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.356 ± 0.018 | 0.321 ± 0.016 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 28.073 ± 0.059 | - | 26.901 ± 0.094 | 26.610 ± 0.054 | 33.333 ± 0.215 | 34.738 ± 0.773 | 42.359 ± 0.936 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 13.084 ± 0.068 | - | 12.792 ± 0.160 | 12.530 ± 0.490 | 19.095 ± 0.079 | 35.371 ± 0.954 | 21.170 ± 2.047 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 96.849 ± 0.275 | - | 109.847 ± 0.206 | 41.210 ± 0.195 | 84.428 ± 0.178 | 135.157 ± 0.058 | 135.277 ± 0.189 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 21.627 ± 1.067 | - | 21.822 ± 0.707 | 21.925 ± 0.972 | 42.640 ± 0.374 | 134.857 ± 0.536 | 67.609 ± 1.307 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 30.361 ± 0.195 | - | 116.776 ± 0.552 | 38.866 ± 9.473 | 41.991 ± 5.791 | 27.338 ± 0.184 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 24.534 ± 0.661 | - | 25.133 ± 0.437 | 24.204 ± 0.235 | 31.540 ± 0.994 | 28.711 ± 0.810 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 22.427 ± 0.350 | - | 71.700 ± 0.422 | 21.425 ± 0.116 | 21.787 ± 0.636 | 21.833 ± 0.173 | 41.352 ± 0.024 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 16.125 ± 0.162 | - | 15.616 ± 0.250 | 16.113 ± 0.373 | 16.277 ± 0.223 | 23.026 ± 0.756 | 21.050 ± 0.395 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 10.767 ± 0.028 | - | 10.806 ± 0.023 | 29.792 ± 0.297 | 71.141 ± 0.061 | 31.482 ± 0.770 | 35.728 ± 1.308 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 4.996 ± 0.009 | - | 5.074 ± 0.008 | 14.348 ± 0.112 | 35.842 ± 0.071 | 31.474 ± 1.893 | 17.354 ± 1.666 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 85.573 ± 0.344 | - | 97.843 ± 0.807 | 70.954 ± 0.154 | 28.481 ± 0.147 | 88.062 ± 0.204 | 92.476 ± 0.247 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 17.217 ± 0.529 | - | 17.528 ± 0.622 | 34.945 ± 1.376 | 17.842 ± 0.157 | 88.037 ± 0.644 | 45.899 ± 3.262 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 26.649 ± 0.240 | - | 87.132 ± 0.604 | 27.719 ± 0.850 | 30.854 ± 0.334 | 26.126 ± 0.285 | 51.275 ± 0.235 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 22.901 ± 0.206 | - | 22.279 ± 0.186 | 22.799 ± 0.192 | 23.072 ± 0.099 | 28.078 ± 1.006 | 27.096 ± 0.458 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 27.213 ± 0.042 | - | 27.272 ± 0.044 | 46.890 ± 0.138 | 9.639 ± 0.051 | 40.509 ± 1.317 | 47.000 ± 6.581 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 12.118 ± 0.352 | - | 11.968 ± 0.058 | 19.796 ± 0.087 | 5.362 ± 0.113 | 41.322 ± 1.491 | 21.311 ± 1.637 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 101.171 ± 4.082 | 112.368 ± 8.427 | unsupported | 33.833 ± 7.039 | 42.409 ± 7.981 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 25.013 ± 0.241 | 25.132 ± 0.768 | unsupported | 24.014 ± 0.397 | 32.176 ± 1.110 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.777 ± 0.039 | - | 0.815 ± 0.024 | 0.690 ± 0.025 | 0.743 ± 0.048 | 1.084 ± 1.221 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.295 ± 0.045 | - | 0.342 ± 0.039 | 0.266 ± 0.058 | 0.691 ± 0.043 | 1.998 ± 0.603 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.701 ± 0.038 | - | 0.687 ± 0.030 | 0.676 ± 0.030 | 0.737 ± 0.075 | 2.179 ± 0.703 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.347 ± 0.131 | - | 0.308 ± 0.043 | 0.273 ± 0.034 | 0.686 ± 0.149 | 2.053 ± 0.437 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.129 ± 0.021 | - | unsupported | 1.020 ± 0.038 | 1.013 ± 0.014 | 1.683 ± 0.985 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.617 ± 0.088 | - | unsupported | 0.545 ± 0.143 | 0.880 ± 0.109 | 2.534 ± 0.412 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.273 ± 0.014 | - | 0.325 ± 0.032 | 0.642 ± 0.007 | 0.278 ± 0.013 | 1.000 ± 0.112 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.298 ± 0.018 | - | 0.342 ± 0.008 | 0.320 ± 0.003 | 0.173 ± 0.009 | 1.017 ± 0.242 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.693 ± 0.014 | - | 0.711 ± 0.039 | 1.022 ± 0.020 | 0.880 ± 0.013 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.298 ± 0.193 | - | 0.302 ± 0.033 | 0.404 ± 0.075 | 0.849 ± 0.016 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.819 ± 0.018 | - | 0.804 ± 0.027 | 0.742 ± 0.032 | 0.817 ± 0.078 | 1.174 ± 1.392 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.298 ± 0.031 | - | 0.316 ± 0.020 | 0.288 ± 0.056 | 0.641 ± 0.182 | 2.351 ± 1.431 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.618 ± 0.022 | - | 0.702 ± 0.034 | 0.298 ± 0.012 | 0.286 ± 0.015 | 0.344 ± 0.793 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.682 ± 0.027 | - | 0.723 ± 0.023 | 0.380 ± 0.008 | 0.310 ± 0.016 | 0.354 ± 0.860 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.316 ± 0.017 | - | 0.004 ± 0.001 | 1.344 ± 0.030 | 1.115 ± 0.071 | 2.808 ± 1.059 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.719 ± 0.025 | - | 0.002 ± 0.000 | 0.745 ± 0.115 | 0.991 ± 0.117 | 2.930 ± 0.862 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.041 ± 0.000 | - | 0.045 ± 0.000 | 0.058 ± 0.000 | 0.079 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.073 ± 0.000 | - | 0.078 ± 0.000 | 0.081 ± 0.001 | 0.130 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.187 ± 0.003 | - | 0.215 ± 0.009 | 0.231 ± 0.001 | 0.276 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.067 ± 0.001 | - | 0.073 ± 0.000 | 0.109 ± 0.002 | 0.107 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.040 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.030 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.539 ± 0.010 | - | 0.563 ± 0.011 | 0.895 ± 0.006 | 0.609 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.019 ± 0.000 | - | 0.024 ± 0.001 | 0.031 ± 0.000 | 0.060 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.041 ± 0.001 | - | 0.045 ± 0.000 | 0.069 ± 0.001 | 0.060 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.073 ± 0.000 | - | 0.079 ± 0.003 | 0.099 ± 0.000 | 0.110 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.187 ± 0.003 | - | 0.246 ± 0.009 | 0.245 ± 0.001 | 0.264 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.066 ± 0.000 | - | 0.073 ± 0.001 | 0.143 ± 0.001 | 0.108 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.001 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.002 ± 0.002 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.002 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.551 ± 0.022 | - | 0.630 ± 0.017 | 1.119 ± 0.006 | 0.911 ± 0.116 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.002 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.019 ± 0.000 | - | 0.024 ± 0.002 | 0.045 ± 0.000 | 0.036 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.002 ± 0.002 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.004 ± 0.002 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.417 ± 0.009 | - | 0.425 ± 0.010 | 0.527 ± 0.009 | 0.636 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.646 ± 0.008 | - | 1.662 ± 0.011 | 1.688 ± 0.006 | 2.103 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 5.022 ± 0.041 | - | 5.037 ± 0.010 | 5.167 ± 0.004 | 7.101 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 2.978 ± 0.007 | - | 3.002 ± 0.008 | 2.675 ± 0.014 | 2.121 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.088 ± 0.001 | - | 0.092 ± 0.001 | 0.093 ± 0.002 | 0.120 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.012 ± 0.001 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.016 ± 0.000 | - | 0.019 ± 0.000 | 0.022 ± 0.001 | 0.062 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.021 ± 0.000 | - | 0.024 ± 0.001 | 0.026 ± 0.000 | 0.071 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 23.835 ± 0.056 | - | 23.965 ± 0.091 | 21.148 ± 0.048 | 17.553 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.688 ± 0.011 | - | 0.697 ± 0.012 | 0.696 ± 0.008 | 0.756 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.014 ± 0.000 | - | 0.017 ± 0.000 | 0.019 ± 0.001 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.041 ± 0.000 | - | 0.045 ± 0.001 | 0.046 ± 0.001 | 0.098 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.711 ± 0.011 | - | 0.718 ± 0.022 | 0.800 ± 0.288 | 0.690 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 2.490 ± 0.012 | - | 2.514 ± 0.083 | 1.674 ± 0.008 | 1.987 ± 0.137 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 7.457 ± 0.003 | - | 6.007 ± 2.423 | 7.385 ± 0.011 | 5.299 ± 0.495 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 3.091 ± 0.024 | - | 3.108 ± 0.016 | 1.931 ± 0.032 | 2.853 ± 0.765 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.089 ± 0.002 | - | 0.093 ± 0.003 | 0.141 ± 0.003 | 0.119 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.009 ± 0.001 | 0.007 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.009 ± 0.000 | - | 0.014 ± 0.000 | 0.011 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.005 ± 0.002 | - | 0.008 ± 0.001 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.009 ± 0.000 | - | 0.014 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.024 ± 0.000 | - | 0.029 ± 0.001 | 0.021 ± 0.000 | 0.056 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.006 ± 0.000 | - | 0.010 ± 0.001 | 0.011 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.012 ± 0.000 | - | 0.017 ± 0.000 | 0.012 ± 0.001 | 0.018 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.032 ± 0.000 | - | 0.036 ± 0.000 | 0.026 ± 0.000 | 0.045 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 24.579 ± 0.098 | - | 24.764 ± 0.509 | 15.238 ± 0.336 | 23.705 ± 2.569 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.009 ± 0.001 | - | 0.012 ± 0.001 | 0.014 ± 0.002 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.695 ± 0.016 | - | 0.709 ± 0.019 | 1.040 ± 0.008 | 1.262 ± 1.060 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.022 ± 0.004 | - | 0.026 ± 0.001 | 0.019 ± 0.000 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.060 ± 0.000 | - | 0.065 ± 0.000 | 0.047 ± 0.000 | 0.075 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.491 ± 0.019 | - | 0.491 ± 0.016 | 0.330 ± 0.007 | 0.321 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.693 ± 0.004 | - | 1.704 ± 0.005 | 1.477 ± 0.005 | 1.494 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 4.499 ± 0.008 | - | 4.516 ± 0.012 | 4.366 ± 0.038 | 4.998 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.097 ± 0.011 | - | 1.102 ± 0.015 | 1.101 ± 0.006 | 1.133 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.002 ± 0.001 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.057 ± 0.000 | - | 0.062 ± 0.015 | 0.067 ± 0.003 | 0.094 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.001 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.015 ± 0.000 | - | 0.018 ± 0.000 | 0.019 ± 0.000 | 0.027 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.006 ± 0.003 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.019 ± 0.000 | - | 0.022 ± 0.000 | 0.022 ± 0.001 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 8.676 ± 0.042 | - | 8.753 ± 0.070 | 8.672 ± 0.065 | 8.759 ± 0.074 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.439 ± 0.009 | - | 0.449 ± 0.006 | 0.485 ± 0.011 | 0.579 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.014 ± 0.000 | - | 0.018 ± 0.000 | 0.016 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.036 ± 0.000 | - | 0.040 ± 0.000 | 0.040 ± 0.000 | 0.078 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.758 ± 0.008 | - | 0.775 ± 0.024 | 0.535 ± 0.007 | 0.282 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 2.553 ± 0.009 | - | 2.574 ± 0.014 | 1.496 ± 0.013 | 0.967 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 6.671 ± 0.008 | - | 6.737 ± 1.741 | 6.220 ± 0.008 | 2.928 ± 0.516 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.050 ± 0.021 | - | 1.075 ± 0.010 | 1.007 ± 0.007 | 1.314 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.003 ± 0.001 | - | 0.007 ± 0.000 | 0.007 ± 0.001 | 0.010 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.090 ± 0.005 | - | 0.095 ± 0.002 | 0.098 ± 0.004 | 0.088 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.008 ± 0.000 | - | 0.013 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.005 ± 0.001 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.010 ± 0.002 | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.022 ± 0.000 | - | 0.026 ± 0.002 | 0.018 ± 0.001 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.002 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.012 ± 0.000 | - | 0.017 ± 0.002 | 0.016 ± 0.002 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.028 ± 0.000 | - | 0.033 ± 0.001 | 0.023 ± 0.000 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 8.244 ± 0.066 | - | 8.395 ± 0.032 | 7.821 ± 0.121 | 9.573 ± 1.086 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.010 ± 0.004 | - | 0.012 ± 0.000 | 0.011 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.702 ± 0.012 | - | 0.715 ± 0.011 | 0.717 ± 0.011 | 0.347 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.022 ± 0.000 | - | 0.027 ± 0.000 | 0.024 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.054 ± 0.000 | - | 0.059 ± 0.002 | 0.040 ± 0.001 | 0.057 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.182 ± 0.007 | - | 0.186 ± 0.007 | 0.172 ± 0.002 | 0.302 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 0.828 ± 0.005 | - | 0.841 ± 0.006 | 0.721 ± 0.007 | 1.458 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 2.419 ± 0.009 | - | 2.433 ± 0.003 | 2.273 ± 0.010 | 4.991 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.446 ± 0.004 | - | 0.432 ± 0.011 | 0.449 ± 0.008 | 1.107 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.002 ± 0.002 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.025 ± 0.000 | - | 0.028 ± 0.000 | 0.031 ± 0.001 | 0.093 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.014 ± 0.001 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.011 ± 0.000 | - | 0.014 ± 0.000 | 0.016 ± 0.000 | 0.027 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 3.587 ± 0.100 | - | 3.429 ± 0.041 | 3.524 ± 0.016 | 8.632 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.186 ± 0.001 | - | 0.191 ± 0.008 | 0.196 ± 0.002 | 0.573 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.020 ± 0.000 | - | 0.023 ± 0.000 | 0.024 ± 0.000 | 0.064 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.326 ± 0.014 | - | 0.331 ± 0.014 | 0.284 ± 0.010 | 0.286 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 1.147 ± 0.017 | - | 1.158 ± 0.004 | 0.719 ± 0.013 | 0.943 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 3.100 ± 0.012 | - | 3.176 ± 0.021 | 2.762 ± 0.011 | 3.072 ± 0.194 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.561 ± 0.014 | - | 0.572 ± 0.013 | 0.593 ± 0.010 | 1.308 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.002 | - | 0.006 ± 0.000 | 0.008 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.031 ± 0.001 | - | 0.036 ± 0.001 | 0.040 ± 0.001 | 0.099 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.001 | 0.009 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.005 ± 0.001 | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.003 ± 0.001 | - | 0.007 ± 0.001 | 0.009 ± 0.003 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.011 ± 0.002 | - | 0.014 ± 0.000 | 0.013 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.001 | 0.010 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.007 ± 0.003 | - | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.014 ± 0.000 | - | 0.018 ± 0.001 | 0.015 ± 0.000 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 4.391 ± 0.035 | - | 4.484 ± 0.036 | 4.525 ± 0.292 | 10.240 ± 0.073 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.231 ± 0.003 | - | 0.237 ± 0.015 | 0.245 ± 0.011 | 0.343 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.010 ± 0.002 | - | 0.015 ± 0.001 | 0.015 ± 0.000 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.026 ± 0.000 | - | 0.030 ± 0.000 | 0.024 ± 0.000 | 0.053 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.171 ± 0.001 | - | 0.146 ± 0.001 | 0.145 ± 0.002 | 0.107 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.220 ± 0.014 | - | 0.189 ± 0.020 | 0.204 ± 0.007 | 0.184 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.431 ± 0.009 | - | 0.442 ± 0.008 | 0.416 ± 0.011 | 0.469 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.108 ± 0.003 | - | 0.115 ± 0.003 | 0.132 ± 0.003 | 0.184 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.009 ± 0.000 | - | 0.013 ± 0.000 | 0.015 ± 0.001 | 0.051 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.011 ± 0.000 | - | 0.015 ± 0.000 | 0.016 ± 0.001 | 0.054 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.038 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.004 ± 0.000 | - | 0.007 ± 0.002 | 0.011 ± 0.000 | 0.042 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.841 ± 0.018 | - | 0.873 ± 0.016 | 1.026 ± 0.008 | 1.025 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.004 ± 0.002 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.057 ± 0.001 | - | 0.063 ± 0.001 | 0.061 ± 0.002 | 0.101 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.038 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.013 ± 0.000 | 0.045 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.157 ± 0.001 | - | 0.167 ± 0.002 | 0.124 ± 0.001 | 0.103 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.231 ± 0.018 | - | 0.254 ± 0.001 | 0.164 ± 0.001 | 0.176 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.583 ± 0.036 | - | 0.642 ± 0.022 | 0.289 ± 0.012 | 0.516 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.150 ± 0.009 | - | 0.160 ± 0.004 | 0.171 ± 0.004 | 0.146 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.015 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.014 ± 0.000 | - | 0.018 ± 0.001 | 0.023 ± 0.001 | 0.024 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.004 ± 0.001 | - | 0.007 ± 0.000 | 0.015 ± 0.003 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.017 ± 0.001 | - | 0.021 ± 0.001 | 0.026 ± 0.002 | 0.027 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.004 ± 0.002 | - | 0.007 ± 0.000 | 0.015 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.001 | 0.017 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.005 ± 0.000 | - | 0.008 ± 0.001 | 0.018 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.001 | 0.017 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.004 ± 0.001 | - | 0.008 ± 0.000 | 0.017 ± 0.002 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.018 ± 0.000 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 1.186 ± 0.014 | - | 1.266 ± 0.028 | 0.485 ± 0.020 | 0.722 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.005 ± 0.002 | - | 0.008 ± 0.000 | 0.018 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.089 ± 0.001 | - | 0.096 ± 0.005 | 0.052 ± 0.001 | 0.109 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.005 ± 0.002 | - | 0.009 ± 0.000 | 0.018 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.008 ± 0.001 | - | 0.012 ± 0.000 | 0.020 ± 0.000 | 0.032 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.195 ± 0.002 | - | 0.199 ± 0.001 | 0.171 ± 0.007 | 0.243 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.430 ± 0.011 | - | 0.437 ± 0.009 | 0.412 ± 0.009 | 0.640 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.019 ± 0.007 | - | 1.063 ± 0.010 | 1.044 ± 0.005 | 1.709 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.409 ± 0.012 | - | 0.395 ± 0.009 | 0.453 ± 0.011 | 0.514 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.013 ± 0.000 | - | 0.016 ± 0.000 | 0.019 ± 0.000 | 0.059 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.015 ± 0.000 | - | 0.018 ± 0.001 | 0.022 ± 0.000 | 0.056 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.024 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 3.142 ± 0.015 | - | 3.820 ± 0.013 | 3.445 ± 0.015 | 3.818 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.094 ± 0.001 | - | 0.099 ± 0.001 | 0.128 ± 0.002 | 0.174 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.009 ± 0.000 | - | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.196 ± 0.019 | - | 0.206 ± 0.024 | 0.246 ± 0.008 | 0.231 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.427 ± 0.021 | - | 0.442 ± 0.020 | 0.584 ± 0.012 | 0.611 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.024 ± 0.004 | - | 1.106 ± 0.029 | 1.534 ± 0.011 | 0.984 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.463 ± 0.016 | - | 0.472 ± 0.016 | 0.418 ± 0.015 | 0.927 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.013 ± 0.000 | - | 0.016 ± 0.000 | 0.032 ± 0.000 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.023 ± 0.000 | - | 0.029 ± 0.001 | 0.035 ± 0.001 | 0.149 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.004 ± 0.003 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.004 ± 0.002 | - | 0.008 ± 0.003 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.005 ± 0.004 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.006 ± 0.003 | - | 0.009 ± 0.000 | 0.013 ± 0.003 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 3.591 ± 0.018 | - | 4.491 ± 0.175 | 3.096 ± 0.029 | 8.192 ± 0.905 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.005 ± 0.001 | - | 0.006 ± 0.001 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.093 ± 0.000 | - | 0.099 ± 0.002 | 0.210 ± 0.008 | 0.159 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.005 ± 0.001 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.010 ± 0.008 | - | 0.013 ± 0.001 | 0.019 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.176 ± 0.002 | - | 0.128 ± 0.002 | 0.075 ± 0.000 | 0.204 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.192 ± 0.002 | - | 0.142 ± 0.002 | 0.116 ± 0.001 | 0.287 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.451 ± 0.016 | - | 0.385 ± 0.028 | 0.335 ± 0.010 | 0.597 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.091 ± 0.001 | - | 0.104 ± 0.003 | 0.110 ± 0.002 | 0.161 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.008 ± 0.000 | - | 0.013 ± 0.001 | 0.019 ± 0.000 | 0.050 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.001 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.726 ± 0.012 | - | 0.765 ± 0.007 | 0.819 ± 0.014 | 0.854 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.052 ± 0.000 | - | 0.060 ± 0.001 | 0.058 ± 0.001 | 0.100 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.014 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.138 ± 0.004 | - | 0.126 ± 0.002 | 0.098 ± 0.001 | 0.201 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.179 ± 0.017 | - | 0.131 ± 0.008 | 0.133 ± 0.001 | 0.264 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.429 ± 0.020 | - | 0.451 ± 0.020 | 0.300 ± 0.012 | 0.639 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.091 ± 0.002 | - | 0.102 ± 0.007 | 0.152 ± 0.004 | 0.149 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.016 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.008 ± 0.000 | - | 0.013 ± 0.000 | 0.026 ± 0.002 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.016 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.017 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.002 ± 0.001 | - | 0.006 ± 0.000 | 0.017 ± 0.002 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.002 ± 0.002 | - | 0.007 ± 0.000 | 0.017 ± 0.002 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.004 ± 0.002 | - | 0.007 ± 0.000 | 0.019 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.003 ± 0.001 | - | 0.006 ± 0.000 | 0.017 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.003 ± 0.001 | - | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.019 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.735 ± 0.022 | - | 0.812 ± 0.029 | 0.444 ± 0.014 | 0.607 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.003 ± 0.001 | - | 0.007 ± 0.000 | 0.019 ± 0.001 | 0.010 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.053 ± 0.001 | - | 0.059 ± 0.000 | 0.047 ± 0.001 | 0.110 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.005 ± 0.003 | - | 0.007 ± 0.000 | 0.019 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.007 ± 0.002 | - | 0.009 ± 0.000 | 0.022 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.402 ± 0.007 | - | 1.411 ± 0.010 | 1.196 ± 0.005 | 1.604 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 4.579 ± 0.031 | - | 4.569 ± 0.015 | 3.983 ± 0.007 | 4.040 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 13.615 ± 0.098 | - | 13.611 ± 0.048 | 11.983 ± 0.024 | 11.747 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 4.060 ± 0.026 | - | 4.063 ± 0.022 | 4.676 ± 0.029 | 3.467 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.001 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.157 ± 0.006 | - | 0.164 ± 0.006 | 0.181 ± 0.013 | 0.175 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.140 ± 0.006 | - | 0.145 ± 0.003 | 0.163 ± 0.005 | 0.168 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.017 ± 0.000 | - | 0.020 ± 0.000 | 0.020 ± 0.001 | 0.026 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.015 ± 0.000 | - | 0.019 ± 0.001 | 0.018 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.043 ± 0.001 | - | 0.046 ± 0.000 | 0.044 ± 0.001 | 0.049 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.011 ± 0.001 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.020 ± 0.001 | - | 0.023 ± 0.001 | 0.023 ± 0.001 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.058 ± 0.001 | - | 0.061 ± 0.001 | 0.056 ± 0.001 | 0.063 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 32.313 ± 0.075 | - | 33.758 ± 0.219 | 37.056 ± 0.218 | 27.057 ± 0.084 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.013 ± 0.000 | - | 0.016 ± 0.000 | 0.016 ± 0.001 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.220 ± 0.004 | - | 1.251 ± 0.019 | 1.332 ± 0.017 | 1.193 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.038 ± 0.000 | - | 0.042 ± 0.001 | 0.038 ± 0.001 | 0.045 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.110 ± 0.001 | - | 0.114 ± 0.003 | 0.103 ± 0.000 | 0.141 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 2.147 ± 0.016 | - | 2.157 ± 0.020 | 1.142 ± 0.003 | 1.581 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 6.966 ± 0.055 | - | 6.987 ± 0.014 | 3.916 ± 0.012 | 2.000 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 13.616 ± 0.045 | - | 13.720 ± 0.024 | 14.360 ± 0.014 | 6.022 ± 0.252 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 4.080 ± 0.055 | - | 4.089 ± 0.016 | 3.993 ± 0.011 | 4.434 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.005 ± 0.000 | - | 0.010 ± 0.001 | 0.010 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.247 ± 0.011 | - | 0.254 ± 0.016 | 0.267 ± 0.012 | 0.195 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.010 ± 0.000 | - | 0.015 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.216 ± 0.015 | - | 0.221 ± 0.013 | 0.232 ± 0.014 | 0.181 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.024 ± 0.000 | - | 0.029 ± 0.000 | 0.020 ± 0.001 | 0.024 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.012 ± 0.004 | - | 0.014 ± 0.001 | 0.014 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.023 ± 0.000 | - | 0.028 ± 0.000 | 0.018 ± 0.001 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.063 ± 0.001 | - | 0.068 ± 0.001 | 0.044 ± 0.001 | 0.048 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.012 ± 0.001 | - | 0.016 ± 0.000 | 0.015 ± 0.001 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.030 ± 0.001 | - | 0.035 ± 0.000 | 0.022 ± 0.000 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.084 ± 0.003 | - | 0.089 ± 0.004 | 0.056 ± 0.001 | 0.061 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 32.706 ± 0.302 | - | 34.078 ± 0.665 | 31.393 ± 0.110 | 47.670 ± 20.383 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.020 ± 0.001 | - | 0.025 ± 0.001 | 0.023 ± 0.003 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 1.865 ± 0.017 | - | 1.894 ± 0.012 | 1.995 ± 0.011 | 0.609 ± 0.125 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.057 ± 0.001 | - | 0.062 ± 0.003 | 0.038 ± 0.000 | 0.046 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.162 ± 0.010 | - | 0.166 ± 0.037 | 0.102 ± 0.001 | 0.110 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 1.027 ± 0.008 | - | 1.037 ± 0.005 | 0.843 ± 0.014 | 0.965 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 3.892 ± 0.016 | - | 3.875 ± 0.018 | 3.357 ± 0.083 | 3.052 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 11.601 ± 0.017 | - | 11.619 ± 0.021 | 10.449 ± 0.056 | 9.572 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.893 ± 0.024 | - | 1.890 ± 0.012 | 2.018 ± 0.010 | 1.815 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.108 ± 0.008 | - | 0.112 ± 0.004 | 0.124 ± 0.003 | 0.136 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.088 ± 0.006 | - | 0.091 ± 0.002 | 0.098 ± 0.002 | 0.124 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.014 ± 0.000 | - | 0.017 ± 0.000 | 0.019 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.014 ± 0.000 | - | 0.017 ± 0.000 | 0.018 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.034 ± 0.000 | - | 0.037 ± 0.000 | 0.037 ± 0.001 | 0.039 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.017 ± 0.000 | - | 0.021 ± 0.001 | 0.020 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.045 ± 0.000 | - | 0.049 ± 0.000 | 0.047 ± 0.001 | 0.048 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 15.093 ± 0.080 | - | 15.115 ± 0.056 | 15.826 ± 0.027 | 14.202 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.013 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 0.825 ± 0.012 | - | 0.833 ± 0.006 | 0.909 ± 0.003 | 0.898 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.033 ± 0.000 | - | 0.036 ± 0.000 | 0.035 ± 0.021 | 0.037 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.089 ± 0.000 | - | 0.093 ± 0.004 | 0.087 ± 0.001 | 0.099 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 1.568 ± 0.013 | - | 1.578 ± 0.022 | 0.799 ± 0.381 | 0.954 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 5.863 ± 0.012 | - | 5.873 ± 0.027 | 3.279 ± 0.008 | 1.544 ± 0.266 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 11.556 ± 0.041 | - | 11.686 ± 0.025 | 12.924 ± 0.044 | 4.916 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 2.228 ± 0.014 | - | 2.252 ± 0.036 | 2.352 ± 0.021 | 2.570 ± 0.071 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.005 ± 0.000 | - | 0.009 ± 0.001 | 0.010 ± 0.001 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.163 ± 0.005 | - | 0.174 ± 0.013 | 0.189 ± 0.006 | 0.127 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.009 ± 0.000 | - | 0.013 ± 0.000 | 0.010 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.128 ± 0.006 | - | 0.134 ± 0.014 | 0.145 ± 0.007 | 0.186 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.020 ± 0.000 | - | 0.024 ± 0.001 | 0.018 ± 0.001 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.008 ± 0.000 | - | 0.012 ± 0.001 | 0.012 ± 0.002 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.020 ± 0.000 | - | 0.025 ± 0.001 | 0.016 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.050 ± 0.001 | - | 0.054 ± 0.001 | 0.036 ± 0.001 | 0.038 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.009 ± 0.001 | - | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.026 ± 0.000 | - | 0.031 ± 0.000 | 0.020 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.066 ± 0.000 | - | 0.070 ± 0.001 | 0.047 ± 0.000 | 0.047 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 17.558 ± 0.071 | - | 17.855 ± 0.071 | 18.420 ± 0.137 | 20.237 ± 1.465 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.015 ± 0.001 | - | 0.020 ± 0.000 | 0.018 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 1.269 ± 0.017 | - | 1.279 ± 0.012 | 1.397 ± 0.019 | 0.526 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.049 ± 0.000 | - | 0.054 ± 0.002 | 0.033 ± 0.000 | 0.036 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.132 ± 0.001 | - | 0.135 ± 0.002 | 0.086 ± 0.001 | 0.092 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.361 ± 0.009 | - | unsupported | 0.266 ± 0.003 | 0.529 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.654 ± 0.012 | - | unsupported | 1.453 ± 0.009 | 1.970 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 4.119 ± 0.010 | - | unsupported | 3.993 ± 0.049 | 5.126 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.831 ± 0.007 | - | unsupported | 0.937 ± 0.025 | 0.936 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.002 ± 0.001 | - | unsupported | 0.006 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.038 ± 0.002 | - | unsupported | 0.047 ± 0.002 | 0.057 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.031 ± 0.000 | - | unsupported | 0.041 ± 0.003 | 0.049 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | unsupported | 0.010 ± 0.001 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.007 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | unsupported | 0.011 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.013 ± 0.000 | - | unsupported | 0.018 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.007 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.018 ± 0.001 | - | unsupported | 0.022 ± 0.001 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 6.588 ± 0.048 | - | unsupported | 7.217 ± 0.041 | 7.172 ± 0.060 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.298 ± 0.008 | - | unsupported | 0.330 ± 0.009 | 0.377 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.014 ± 0.000 | - | unsupported | 0.017 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.033 ± 0.000 | - | unsupported | 0.038 ± 0.000 | 0.049 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.549 ± 0.012 | - | unsupported | 0.404 ± 0.003 | 0.544 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 2.608 ± 0.011 | - | unsupported | 1.431 ± 0.034 | 1.208 ± 0.188 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 6.111 ± 0.012 | - | unsupported | 5.613 ± 0.005 | 2.702 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.030 ± 0.013 | - | unsupported | 1.187 ± 0.008 | 1.376 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.003 ± 0.002 | - | unsupported | 0.008 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.051 ± 0.003 | - | unsupported | 0.065 ± 0.003 | 0.054 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | unsupported | 0.010 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.043 ± 0.001 | - | unsupported | 0.056 ± 0.001 | 0.059 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.008 ± 0.000 | - | unsupported | 0.010 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | unsupported | 0.010 ± 0.001 | 0.009 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.009 ± 0.000 | - | unsupported | 0.015 ± 0.002 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.019 ± 0.000 | - | unsupported | 0.018 ± 0.001 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.012 ± 0.000 | - | unsupported | 0.017 ± 0.000 | 0.016 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.026 ± 0.000 | - | unsupported | 0.022 ± 0.001 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 8.126 ± 0.048 | - | unsupported | 9.216 ± 0.064 | 11.179 ± 0.144 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.393 ± 0.014 | - | unsupported | 0.333 ± 0.014 | 0.241 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.022 ± 0.000 | - | unsupported | 0.026 ± 0.002 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.049 ± 0.000 | - | unsupported | 0.038 ± 0.001 | 0.048 ± 0.005 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.827 ± 0.012 | - | unsupported | 0.782 ± 0.009 | 2.007 ± 0.035 | 2.125 ± 0.380 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.102 ± 0.001 | - | unsupported | 0.047 ± 0.002 | 0.079 ± 0.002 | 0.081 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.136 ± 0.001 | - | unsupported | 0.070 ± 0.001 | 0.138 ± 0.004 | 0.142 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.291 ± 0.007 | - | unsupported | 0.202 ± 0.011 | 0.372 ± 0.017 | 0.375 ± 0.276 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.841 ± 0.005 | - | unsupported | 0.388 ± 0.012 | 1.502 ± 0.035 | 2.023 ± 0.888 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.109 ± 0.001 | - | unsupported | 0.048 ± 0.026 | 0.073 ± 0.009 | 0.083 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.145 ± 0.005 | - | unsupported | 0.068 ± 0.014 | 0.148 ± 0.008 | 0.145 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.302 ± 0.023 | - | unsupported | 0.123 ± 0.007 | 0.431 ± 0.028 | 0.623 ± 0.245 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.247 ± 0.013 | - | unsupported | 0.301 ± 0.009 | 0.626 ± 0.006 | 0.297 ± 0.008 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.027 ± 0.001 | - | unsupported | 0.033 ± 0.001 | 0.136 ± 0.003 | 0.061 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.050 ± 0.000 | - | unsupported | 0.050 ± 0.004 | 0.163 ± 0.004 | 0.090 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.162 ± 0.002 | - | unsupported | 0.138 ± 0.001 | 0.275 ± 0.017 | 0.150 ± 0.006 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.245 ± 0.009 | - | unsupported | 0.437 ± 0.010 | 0.715 ± 0.067 | 0.296 ± 0.008 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.027 ± 0.000 | - | unsupported | 0.056 ± 0.002 | 0.128 ± 0.011 | 0.061 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.050 ± 0.000 | - | unsupported | 0.076 ± 0.001 | 0.158 ± 0.008 | 0.091 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.163 ± 0.009 | - | unsupported | 0.180 ± 0.010 | 0.274 ± 0.016 | 0.151 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.157 ± 0.004 | - | 0.213 ± 0.004 | 0.160 ± 0.000 | 0.289 ± 0.007 | 0.170 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.065 ± 0.001 | - | 0.068 ± 0.000 | 0.056 ± 0.001 | 0.056 ± 0.001 | 0.046 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.073 ± 0.004 | - | 0.082 ± 0.001 | 0.068 ± 0.001 | 0.069 ± 0.001 | 0.063 ± 0.004 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.097 ± 0.001 | - | 0.117 ± 0.001 | 0.097 ± 0.000 | 0.123 ± 0.002 | 0.103 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.154 ± 0.001 | - | 0.233 ± 0.011 | 0.207 ± 0.004 | 0.398 ± 0.069 | 0.233 ± 0.007 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.069 ± 0.000 | - | 0.066 ± 0.001 | 0.096 ± 0.002 | 0.058 ± 0.003 | 0.047 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.073 ± 0.000 | - | 0.081 ± 0.001 | 0.109 ± 0.011 | 0.072 ± 0.001 | 0.064 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.097 ± 0.000 | - | 0.183 ± 0.003 | 0.140 ± 0.002 | 0.142 ± 0.012 | 0.133 ± 0.016 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 36.571 ± 0.210 | - | 39.012 ± 0.312 | 53.363 ± 0.671 | 59.198 ± 0.285 | 40.048 ± 3.101 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 17.412 ± 0.291 | - | 19.185 ± 0.117 | 20.321 ± 0.160 | 46.765 ± 4.353 | 29.815 ± 8.390 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 21.795 ± 0.398 | - | 21.417 ± 0.386 | 21.500 ± 0.170 | 19.567 ± 0.286 | 17.904 ± 0.771 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 11.186 ± 0.360 | - | 10.599 ± 0.124 | 9.747 ± 0.163 | 12.075 ± 0.087 | 10.871 ± 0.543 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 6.135 ± 0.017 | - | 6.176 ± 0.017 | 6.498 ± 0.011 | 5.994 ± 0.044 | 5.880 ± 0.239 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 9.577 ± 0.085 | - | 9.602 ± 0.089 | 9.023 ± 0.039 | 8.930 ± 0.412 | 8.580 ± 0.316 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 7.076 ± 0.011 | - | 7.128 ± 0.050 | 7.178 ± 0.030 | 7.336 ± 0.036 | 6.959 ± 0.028 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 10.663 ± 0.060 | - | 10.815 ± 0.045 | 9.711 ± 0.071 | 10.644 ± 0.144 | 10.364 ± 0.062 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 17.115 ± 0.080 | - | 17.109 ± 0.090 | 15.562 ± 0.134 | 32.248 ± 0.113 | 10.868 ± 0.176 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 12.132 ± 0.143 | - | 12.326 ± 0.145 | 12.475 ± 0.051 | 27.471 ± 2.211 | 9.207 ± 0.151 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.405 ± 0.006 | 0.416 ± 0.004 | unsupported | 0.494 ± 0.005 | 0.585 ± 0.017 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 6.445 ± 0.043 | 7.385 ± 0.013 | unsupported | 6.985 ± 0.067 | 7.422 ± 0.086 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.319 ± 0.001 | 0.337 ± 0.002 | unsupported | 0.411 ± 0.002 | 1.402 ± 0.018 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 5.228 ± 0.050 | 5.312 ± 0.076 | unsupported | 4.611 ± 0.135 | 8.599 ± 0.377 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 34.744 ± 0.444 | - | 34.639 ± 0.275 | 36.403 ± 0.275 | 34.319 ± 0.134 | 24.167 ± 2.136 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 18.355 ± 0.436 | - | 18.022 ± 0.325 | 18.660 ± 0.023 | 19.676 ± 1.350 | 15.526 ± 1.102 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 14.852 ± 0.078 | 14.601 ± 0.021 | 7.502 ± 0.037 | 35.092 ± 0.479 | 6.140 ± 0.199 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 9.375 ± 0.057 | 9.449 ± 0.098 | 5.072 ± 0.063 | 34.787 ± 1.455 | 4.592 ± 0.112 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 21.406 ± 0.161 | - | 22.140 ± 0.179 | 32.327 ± 0.306 | 33.656 ± 0.265 | 25.860 ± 12.219 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 11.176 ± 0.302 | - | 11.791 ± 0.222 | 13.833 ± 0.142 | 24.958 ± 2.224 | 16.121 ± 16.811 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.821 ± 0.019 | - | 0.884 ± 0.028 | 0.938 ± 0.016 | 5.834 ± 0.041 | 2.305 ± 0.008 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.526 ± 0.045 | - | 0.402 ± 0.204 | 0.939 ± 0.013 | 4.708 ± 0.386 | 2.308 ± 0.006 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 16.719 ± 0.095 | - | 16.797 ± 0.108 | 17.435 ± 0.271 | 16.027 ± 0.030 | 17.352 ± 0.841 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 13.716 ± 0.234 | - | 13.400 ± 0.066 | 14.246 ± 0.049 | 16.211 ± 0.580 | 19.347 ± 1.534 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 16.901 ± 0.174 | - | 16.858 ± 0.064 | 17.196 ± 0.067 | 16.005 ± 0.027 | 15.815 ± 1.645 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 13.502 ± 0.115 | - | 13.376 ± 0.063 | 14.238 ± 0.027 | 18.112 ± 0.434 | 18.561 ± 0.900 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 0.962 ± 0.006 | 0.986 ± 0.007 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 17.526 ± 0.025 | 17.631 ± 0.041 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.174 ± 0.049 | 1.191 ± 0.010 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 17.032 ± 0.079 | 17.100 ± 0.171 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 0.901 ± 0.003 | 0.918 ± 0.007 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 16.640 ± 0.146 | 16.785 ± 0.044 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.082 ± 0.005 | 1.098 ± 0.012 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 15.954 ± 0.175 | 16.076 ± 0.111 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 21.751 ± 0.273 | - | 21.271 ± 0.250 | 21.227 ± 0.123 | 19.535 ± 0.076 | 17.871 ± 1.444 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 10.915 ± 0.319 | - | 10.305 ± 0.124 | 9.597 ± 0.389 | 12.531 ± 0.650 | 11.340 ± 1.031 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 48.190 ± 0.362 | 48.356 ± 0.251 | 50.997 ± 0.414 | 49.140 ± 0.369 | 48.882 ± 6.315 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 31.396 ± 0.145 | 31.801 ± 1.626 | 32.682 ± 0.650 | 41.769 ± 0.585 | 40.683 ± 4.723 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 31.081 ± 0.054 | - | 43.924 ± 0.424 | 25.663 ± 0.121 | 76.422 ± 0.584 | 24.720 ± 0.055 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 23.860 ± 0.350 | - | 33.707 ± 0.673 | 21.109 ± 1.768 | 61.740 ± 1.887 | 15.589 ± 0.174 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.078 ± 0.001 | 0.079 ± 0.000 | unsupported | 0.127 ± 0.023 | 0.104 ± 0.007 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 1.238 ± 0.003 | 1.262 ± 0.005 | unsupported | 1.717 ± 0.018 | 0.330 ± 0.002 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.071 ± 0.007 | 0.075 ± 0.005 | unsupported | 0.064 ± 0.002 | 0.021 ± 0.004 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.571 ± 0.027 | 0.573 ± 0.005 | unsupported | 0.833 ± 0.016 | 0.421 ± 0.127 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.038 ± 0.000 | 0.038 ± 0.001 | unsupported | 0.091 ± 0.013 | 0.169 ± 0.003 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 0.589 ± 0.001 | 0.700 ± 0.003 | unsupported | 0.905 ± 0.082 | 1.303 ± 0.030 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.025 ± 0.005 | 0.026 ± 0.001 | unsupported | 0.058 ± 0.019 | 0.121 ± 0.001 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.288 ± 0.001 | 0.303 ± 0.004 | unsupported | 0.493 ± 0.008 | 0.739 ± 0.024 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.104 ± 0.000 | 0.104 ± 0.001 | unsupported | 0.154 ± 0.004 | 0.136 ± 0.001 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 1.654 ± 0.003 | 1.681 ± 0.006 | unsupported | 2.255 ± 0.037 | 0.783 ± 0.016 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.056 ± 0.001 | 0.056 ± 0.001 | unsupported | 0.070 ± 0.002 | 0.105 ± 0.002 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 0.762 ± 0.001 | 0.768 ± 0.099 | unsupported | 0.995 ± 0.010 | 0.503 ± 0.008 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.045 ± 0.003 | 0.067 ± 0.004 | unsupported | 0.128 ± 0.001 | 0.156 ± 0.002 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.544 ± 0.013 | 0.611 ± 0.032 | unsupported | 0.687 ± 0.040 | 0.902 ± 0.007 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.124 ± 0.006 | 0.181 ± 0.010 | unsupported | 0.052 ± 0.001 | 0.143 ± 0.003 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.410 ± 0.014 | 0.552 ± 0.021 | unsupported | 0.292 ± 0.119 | 0.655 ± 0.029 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.069 ± 0.003 | 0.083 ± 0.001 | unsupported | 0.090 ± 0.001 | 0.192 ± 0.004 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 0.915 ± 0.009 | 1.380 ± 0.508 | unsupported | 0.748 ± 0.005 | 1.682 ± 0.006 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.155 ± 0.008 | 0.198 ± 0.020 | unsupported | 0.044 ± 0.001 | 0.188 ± 0.003 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.574 ± 0.030 | 0.842 ± 0.198 | unsupported | 0.338 ± 0.006 | 1.103 ± 0.043 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.106 ± 0.002 | 0.131 ± 0.004 | unsupported | unsupported | 0.207 ± 0.004 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 1.382 ± 0.008 | 1.760 ± 0.286 | unsupported | unsupported | 2.040 ± 0.039 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.216 ± 0.004 | 0.266 ± 0.017 | unsupported | unsupported | 0.224 ± 0.005 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 0.900 ± 0.017 | 1.231 ± 0.190 | unsupported | unsupported | 1.296 ± 0.025 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.009 ± 0.000 | 0.014 ± 0.000 | unsupported | 0.010 ± 0.000 | 0.039 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.080 ± 0.002 | 0.093 ± 0.003 | unsupported | 0.060 ± 0.002 | 0.271 ± 0.013 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.017 ± 0.001 | 0.033 ± 0.009 | unsupported | 0.011 ± 0.000 | 0.034 ± 0.005 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.054 ± 0.001 | 0.078 ± 0.005 | unsupported | 0.038 ± 0.002 | 0.164 ± 0.002 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.023 ± 0.005 | 0.039 ± 0.005 | unsupported | 0.080 ± 0.004 | 0.127 ± 0.017 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.296 ± 0.004 | 0.359 ± 0.006 | unsupported | 0.478 ± 0.031 | 0.662 ± 0.018 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.059 ± 0.007 | 0.100 ± 0.005 | unsupported | 0.087 ± 0.002 | 0.115 ± 0.009 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.213 ± 0.014 | 0.294 ± 0.004 | unsupported | 0.255 ± 0.007 | 0.544 ± 0.030 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.026 ± 0.001 | 0.026 ± 0.000 | unsupported | 0.075 ± 0.002 | 0.143 ± 0.002 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.399 ± 0.002 | 0.573 ± 0.004 | unsupported | 0.892 ± 0.007 | 0.641 ± 0.006 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.021 ± 0.000 | 0.021 ± 0.000 | unsupported | 0.058 ± 0.002 | 0.098 ± 0.005 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.200 ± 0.002 | 0.232 ± 0.003 | unsupported | 0.456 ± 0.017 | 0.425 ± 0.005 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.027 ± 0.000 | 0.028 ± 0.000 | unsupported | 0.073 ± 0.002 | 0.143 ± 0.013 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.415 ± 0.002 | 0.593 ± 0.004 | unsupported | 0.896 ± 0.018 | 0.698 ± 0.007 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.021 ± 0.001 | 0.023 ± 0.001 | unsupported | 0.059 ± 0.002 | 0.098 ± 0.005 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.208 ± 0.002 | 0.241 ± 0.007 | unsupported | 0.469 ± 0.032 | 0.655 ± 0.098 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.065 ± 0.001 | 0.085 ± 0.000 | unsupported | 0.096 ± 0.001 | 0.199 ± 0.017 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 0.916 ± 0.012 | 1.006 ± 0.020 | unsupported | 0.801 ± 0.005 | 1.622 ± 0.013 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.150 ± 0.007 | 0.197 ± 0.019 | unsupported | 0.042 ± 0.001 | 0.185 ± 0.006 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 0.570 ± 0.080 | 0.702 ± 0.021 | unsupported | 0.300 ± 0.001 | 1.111 ± 0.111 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.092 ± 0.000 | 0.092 ± 0.001 | unsupported | 0.146 ± 0.002 | 0.266 ± 0.015 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 1.452 ± 0.003 | 1.486 ± 0.004 | unsupported | 2.009 ± 0.107 | 2.675 ± 0.010 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.047 ± 0.001 | 0.047 ± 0.000 | unsupported | 0.113 ± 0.001 | 0.157 ± 0.002 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 0.615 ± 0.003 | 0.619 ± 0.017 | unsupported | 0.893 ± 0.020 | 1.450 ± 0.136 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.538 ± 0.002 | 1.125 ± 0.003 | unsupported | 0.222 ± 0.002 | 0.153 ± 0.003 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 10.469 ± 0.207 | 14.128 ± 0.166 | unsupported | 3.446 ± 0.044 | 0.636 ± 0.008 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.469 ± 0.005 | 1.120 ± 0.106 | unsupported | 0.212 ± 0.003 | 0.104 ± 0.002 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 6.566 ± 0.129 | 10.698 ± 0.099 | unsupported | 1.826 ± 0.018 | 0.444 ± 0.002 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 22.555 ± 0.044 | - | unsupported | 22.189 ± 0.103 | - | 24.722 ± 0.209 | 40.440 ± 0.115 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 21.019 ± 0.592 | - | unsupported | 20.898 ± 0.531 | - | 25.051 ± 0.495 | 37.858 ± 9.965 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 17.176 ± 0.410 | - | unsupported | 16.638 ± 0.170 | - | 17.962 ± 0.570 | 17.542 ± 0.295 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 13.895 ± 0.465 | - | unsupported | 13.959 ± 0.096 | - | 17.385 ± 0.446 | 13.987 ± 0.500 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 14.635 ± 0.159 | - | unsupported | 16.326 ± 0.259 | - | 14.799 ± 0.113 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 14.262 ± 0.114 | - | unsupported | 13.809 ± 0.029 | - | 14.901 ± 0.078 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 22.466 ± 0.131 | - | unsupported | 23.994 ± 0.261 | - | 23.554 ± 0.199 | 41.500 ± 0.159 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 20.249 ± 0.150 | - | unsupported | 20.339 ± 0.108 | - | 23.818 ± 0.287 | 27.331 ± 3.389 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 42.676 ± 0.877 | - | unsupported | 46.227 ± 0.138 | - | 39.364 ± 0.126 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 21.780 ± 0.119 | - | unsupported | 23.254 ± 0.137 | - | 19.498 ± 0.111 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 42.201 ± 0.311 | - | unsupported | 46.200 ± 0.430 | - | 39.223 ± 0.100 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 21.800 ± 0.094 | - | unsupported | 23.387 ± 0.215 | - | 19.402 ± 0.256 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 22.465 ± 0.226 | - | unsupported | 23.983 ± 0.287 | - | 24.919 ± 0.559 | 40.365 ± 0.062 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 20.453 ± 2.760 | - | unsupported | 20.272 ± 0.077 | - | 25.247 ± 0.632 | 33.318 ± 9.797 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 17.207 ± 0.324 | - | unsupported | 16.524 ± 0.163 | - | 19.197 ± 0.468 | 28.180 ± 0.096 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 14.499 ± 0.268 | - | unsupported | 13.995 ± 0.118 | - | 19.707 ± 0.685 | 33.861 ± 0.334 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.598 ± 0.005 | unsupported | unsupported | 0.765 ± 0.006 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 9.202 ± 0.072 | unsupported | unsupported | 10.974 ± 0.203 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.461 ± 0.002 | unsupported | unsupported | 0.517 ± 0.002 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 5.262 ± 0.038 | unsupported | unsupported | 5.486 ± 0.083 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 22.651 ± 0.666 | - | unsupported | 22.406 ± 0.088 | - | 25.435 ± 0.195 | 40.388 ± 0.123 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 20.298 ± 0.767 | - | unsupported | 20.316 ± 0.117 | - | 24.757 ± 0.447 | 39.547 ± 5.013 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 16.777 ± 0.092 | - | 0.002 ± 0.000 | 15.096 ± 0.053 | 14.341 ± 0.450 | 20.629 ± 0.112 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 14.026 ± 0.190 | - | 0.002 ± 0.000 | 12.656 ± 0.320 | 13.677 ± 0.052 | 20.647 ± 0.556 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 14.769 ± 0.104 | - | 36.639 ± 0.533 | 14.856 ± 0.415 | 14.318 ± 0.056 | 14.288 ± 0.341 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 12.020 ± 0.330 | - | 29.360 ± 0.388 | 12.206 ± 0.267 | 12.100 ± 0.078 | 14.497 ± 0.559 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 61.777 ± 1.769 | - | 56.670 ± 1.317 | 26.231 ± 0.214 | 26.036 ± 0.173 | 26.685 ± 0.191 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 21.738 ± 0.220 | - | 22.029 ± 0.687 | 20.678 ± 0.292 | 20.214 ± 0.124 | 27.889 ± 0.195 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 12.166 ± 0.122 | - | 12.083 ± 0.086 | 16.819 ± 0.100 | 13.999 ± 0.146 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 9.663 ± 0.220 | - | 9.597 ± 0.162 | 12.047 ± 0.409 | 11.999 ± 0.135 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 23.645 ± 0.117 | - | 0.002 ± 0.000 | 19.804 ± 0.125 | 21.831 ± 0.103 | 25.196 ± 0.571 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 16.248 ± 0.451 | - | 0.002 ± 0.000 | 16.174 ± 0.150 | 17.429 ± 0.136 | 25.097 ± 0.708 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 38.126 ± 0.203 | - | 0.002 ± 0.000 | 27.695 ± 0.169 | 31.541 ± 0.379 | 242.617 ± 2.965 | 86.075 ± 0.670 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 15.487 ± 0.723 | - | 0.002 ± 0.000 | 27.581 ± 0.099 | 17.274 ± 0.498 | 252.278 ± 2.287 | 28.182 ± 3.711 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 10.622 ± 0.044 | - | 10.628 ± 0.087 | 13.019 ± 0.040 | 10.417 ± 0.140 | 16.348 ± 0.220 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 6.979 ± 0.319 | - | 7.298 ± 0.198 | 9.005 ± 0.051 | 9.104 ± 0.344 | 15.891 ± 0.594 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 10.778 ± 0.081 | - | 10.751 ± 0.073 | 14.541 ± 0.069 | 10.456 ± 0.165 | 16.689 ± 0.217 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 7.436 ± 0.110 | - | 7.507 ± 0.268 | 8.641 ± 0.246 | 8.866 ± 0.032 | 16.569 ± 0.690 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.001 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 4.700768 | 293798.00 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 2.807812 | 175488.25 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 3.262664 | 203916.50 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 2.949330 | 184333.12 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 4.500753 | 281297.06 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 6.301917 | 393869.81 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 14.211755 | 888234.69 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 14.041469 | 877591.81 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 5.688608 | 355538.00 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 5.138399 | 321149.94 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 9.364769 | 585298.06 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 7.899833 | 493739.56 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 6.485459 | 405341.19 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 6.649214 | 415575.88 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 118.755863 | 7422241.44 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 111.765142 | 6985321.38 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 103.114162 | 6444635.12 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 118.162013 | 7385125.81 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 22.429542 | 1401846.38 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 6.577045 | 411065.31 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 5.108853 | 319303.31 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 5.389965 | 336872.81 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 137.576580 | 8598536.25 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 73.778469 | 4611154.31 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 83.648768 | 5228048.00 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 84.986812 | 5311675.75 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 15.385158 | 961572.38 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 15.780681 | 986292.56 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 280.422264 | 17526391.50 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 282.093110 | 17630819.38 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 18.785754 | 1174109.62 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 19.062498 | 1191406.12 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 272.510840 | 17031927.50 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 273.602295 | 17100143.44 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 14.421823 | 901363.94 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 14.694321 | 918395.06 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 266.237153 | 16639822.06 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 268.560154 | 16785009.62 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 17.305657 | 1081603.56 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 17.560948 | 1097559.25 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 255.267296 | 15954206.00 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 257.221975 | 16076373.44 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.669668 | 104213.56 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.159343 | 127489.73 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.990086 | 77970.09 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.036446 | 78694.47 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 2.641677 | 330209.62 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 13.738784 | 1717348.00 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.900584 | 1237573.00 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.098400 | 1262300.00 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 1.374763 | 21480.67 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.079648 | 63744.50 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.516206 | 70565.72 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.770267 | 74535.42 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.366076 | 420759.50 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.665202 | 833150.25 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.568575 | 571071.88 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.587883 | 573485.38 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.841671 | 169401.11 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.813282 | 90832.53 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.404397 | 37568.70 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.422184 | 37846.62 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.422692 | 1302836.50 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.241195 | 905149.38 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.714232 | 589279.00 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.603958 | 700494.75 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.741629 | 120962.95 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.689893 | 57654.58 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.627839 | 25434.98 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.685355 | 26333.67 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.915880 | 739485.00 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.945503 | 493187.88 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.307176 | 288397.00 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.426745 | 303343.12 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.705230 | 136019.22 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 9.881126 | 154392.59 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.675055 | 104297.73 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.679120 | 104361.25 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.263542 | 782942.75 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 18.037214 | 2254651.75 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.229167 | 1653645.88 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 13.445322 | 1680665.25 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.746724 | 105417.56 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.490687 | 70166.98 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.579633 | 55931.77 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3.608446 | 56381.97 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.026261 | 503282.62 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.962065 | 995258.12 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.098868 | 762358.50 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 6.144125 | 768015.62 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.010125 | 156408.20 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.190219 | 127972.17 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.894463 | 45225.98 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.315244 | 67425.69 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.216621 | 902077.62 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 5.498700 | 687337.50 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.350958 | 543869.75 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.889796 | 611224.50 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.124039 | 142563.11 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.353708 | 52401.69 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 7.909770 | 123590.16 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 11.583354 | 180989.91 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.239772 | 654971.50 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.332677 | 291584.62 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.281737 | 410217.12 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.419111 | 552388.88 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.262208 | 191597.00 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.762185 | 90034.14 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.443075 | 69423.05 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.337588 | 83399.81 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.459071 | 1682383.88 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 5.980313 | 747539.12 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.316857 | 914607.12 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.037720 | 1379715.00 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.012650 | 187697.66 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.793081 | 43641.89 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 9.906282 | 154785.66 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 12.674398 | 198037.47 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.822324 | 1102790.50 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.705747 | 338218.38 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.591959 | 573994.88 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 6.732216 | 841527.00 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.247594 | 206993.66 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.768825 | 105762.89 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.373783 | 130840.36 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 16.321789 | 2040223.62 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 11.052662 | 1381582.75 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 14.078952 | 1759869.00 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 14.341306 | 224082.91 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 13.851167 | 216424.48 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 16.999038 | 265609.97 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.371865 | 1296483.12 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.196642 | 899580.25 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 9.848394 | 1231049.25 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.483513 | 38804.89 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.625669 | 9776.08 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.577196 | 9018.69 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 0.924357 | 14443.08 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 2.167196 | 270899.50 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.483385 | 60423.12 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.642213 | 80276.62 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.744457 | 93057.12 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.191124 | 34236.31 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.676064 | 10563.50 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.061340 | 16583.44 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.112775 | 33012.11 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.308083 | 163510.38 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.303013 | 37876.62 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.429483 | 53685.38 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 0.623634 | 77954.25 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.155086 | 127423.22 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.119932 | 79998.94 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.490456 | 23288.38 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.513561 | 39274.39 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.292748 | 661593.50 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 3.824568 | 478071.00 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.365900 | 295737.50 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 2.873551 | 359193.88 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.367650 | 115119.53 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.542445 | 86600.70 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.766708 | 58854.81 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6.398166 | 99971.34 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.354927 | 544365.88 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.039809 | 254976.12 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.701810 | 212726.25 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.351303 | 293912.88 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.176465 | 143382.27 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.772490 | 74570.16 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.662644 | 25978.81 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.661363 | 25958.80 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.128112 | 641014.00 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.134494 | 891811.75 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.195664 | 399458.00 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.584568 | 573071.00 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.278838 | 98106.84 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.720329 | 58130.14 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.318696 | 20604.62 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.333838 | 20841.22 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.396501 | 424562.62 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.645715 | 455714.38 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.600107 | 200013.38 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.858625 | 232328.12 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.149140 | 142955.31 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.676066 | 73063.53 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.711547 | 26742.92 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.765980 | 27593.44 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.585066 | 698133.25 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.167735 | 895966.88 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.320031 | 415003.88 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.742265 | 592783.12 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.301362 | 98458.78 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.769785 | 58902.89 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.337384 | 20896.62 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.443263 | 22550.98 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.242185 | 655273.12 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.752829 | 469103.62 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.660378 | 207547.25 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.928439 | 241054.88 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.761728 | 199402.00 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.168766 | 96386.97 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.142365 | 64724.45 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.416426 | 84631.66 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.976698 | 1622087.25 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 6.408846 | 801105.75 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.331219 | 916402.38 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 8.047093 | 1005886.62 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.829308 | 184832.94 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.718638 | 42478.72 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 9.615797 | 150246.83 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 12.631344 | 197364.75 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.886755 | 1110844.38 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.401442 | 300180.25 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.557808 | 569726.00 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.613490 | 701686.25 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 16.994365 | 265536.95 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 9.346473 | 146038.64 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.865390 | 91646.72 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.878610 | 91853.28 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 21.398413 | 2674801.62 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 16.072275 | 2009034.38 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 11.618511 | 1452313.88 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.890058 | 1486257.25 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.063375 | 157240.23 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 7.244498 | 113195.28 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.987005 | 46671.95 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3.020155 | 47189.92 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.598229 | 1449778.62 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.141911 | 892738.88 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.922575 | 615321.88 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.948754 | 618594.25 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.812599 | 153321.86 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 14.234037 | 222406.83 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 34.431607 | 537993.86 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 71.991647 | 1124869.48 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.087110 | 635888.75 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 27.568935 | 3446116.88 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 83.752843 | 10469105.38 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 113.020632 | 14127579.00 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.683489 | 104429.52 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 13.578884 | 212170.06 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 30.024683 | 469135.67 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 71.704352 | 1120380.50 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.552346 | 444043.25 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 14.607177 | 1825897.12 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 52.531893 | 6566486.62 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 85.584073 | 10698009.12 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 12.245173 | 765323.31 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 9.575914 | 598494.62 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 175.577546 | 10973596.62 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 147.227334 | 9201708.38 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 8.265375 | 516585.94 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 7.379016 | 461188.50 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 87.781596 | 5486349.75 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 84.186460 | 5261653.75 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.017776 | 1111.00 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002203 | 137.69 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.025428 | 1589.25 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002263 | 141.44 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000100 | 6.25 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.016815 | 1050.94 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.008042 | 502.63 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000100 | 6.25 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.023535 | 1470.94 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.007091 | 443.19 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000120 | 7.50 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.017576 | 1098.50 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.005818 | 363.62 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000181 | 11.31 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.025328 | 1583.00 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.005268 | 329.25 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000100 | 6.25 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.015864 | 991.50 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003595 | 224.69 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000090 | 5.62 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.022574 | 1410.88 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003395 | 212.19 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/norm_fro` (c64, threads=1, shape=`2048x1536`): `tenferro-direct` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 18.196 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 13.4x faster than the slowest successful cell (`tenferro-trace`, 12.599 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 26.742 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 25.9x faster than the slowest successful cell (`pytorch-cpu`, 26.742 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 18.1x faster than the slowest successful cell (`pytorch-cpu`, 26.742 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 18.0x faster than the slowest successful cell (`pytorch-cpu`, 26.742 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 12.8x faster than the slowest successful cell (`pytorch-cpu`, 9.156 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 12.8x faster than the slowest successful cell (`pytorch-cpu`, 9.156 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 26.225 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 27.0x faster than the slowest successful cell (`pytorch-cpu`, 26.225 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 18.2x faster than the slowest successful cell (`pytorch-cpu`, 26.225 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 18.0x faster than the slowest successful cell (`pytorch-cpu`, 26.225 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 9.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 9.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 768.1x faster than the slowest successful cell (`julia-base`, 2.808 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 1272.1x faster than the slowest successful cell (`julia-base`, 2.930 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 12.4x faster than the slowest successful cell (`jax-cpu`, 0.040 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 19.8x faster than the slowest successful cell (`jax-cpu`, 0.030 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 13.6x faster than the slowest successful cell (`jax-cpu`, 0.030 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.038 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x2x2,rhs=1`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.016 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 11.7x faster than the slowest successful cell (`jax-cpu`, 4.708 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 22.2x faster than the slowest successful cell (`tenferro-eager`, 14.128 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 10.7x faster than the slowest successful cell (`tenferro-eager`, 1.120 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 24.1x faster than the slowest successful cell (`tenferro-eager`, 10.698 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 9193.1x faster than the slowest successful cell (`julia-base`, 20.629 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 9725.3x faster than the slowest successful cell (`julia-base`, 20.647 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 10485.1x faster than the slowest successful cell (`julia-base`, 25.196 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 10897.6x faster than the slowest successful cell (`julia-base`, 25.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 114280.2x faster than the slowest successful cell (`julia-base`, 242.617 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `jax-cpu` is 14.6x faster than the slowest successful cell (`julia-base`, 252.278 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-direct` is 16.3x faster than the slowest successful cell (`julia-base`, 252.278 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 118830.8x faster than the slowest successful cell (`julia-base`, 252.278 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 175.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 245.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 137.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 143.9x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 165.3x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 235.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 4586.7 GB/s at 0.004 ms. The conservative physical lower bound is 0.168 ms, so the cell is 45.9x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 7281.4 GB/s at 0.002 ms. The conservative physical lower bound is 0.042 ms, so the cell is 18.2x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 119623.6 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1196.2x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 126441.6 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 316.1x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 111708.5 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1117.1x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 116559.0 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 291.4x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 63220.8 GB/s at 0.002 ms. The conservative physical lower bound is 1.342 ms, so the cell is 632.2x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 63220.8 GB/s at 0.002 ms. The conservative physical lower bound is 0.336 ms, so the cell is 158.1x faster than the bound.

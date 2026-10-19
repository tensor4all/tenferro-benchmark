# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`
- Run metadata: `data/results/amd-cpu/cpu/public_api/20261019_183932/run.yaml`
- Timestamp: `20261019_183932`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Sampling: `quick` profile, 7 measured runs, 3 warmups per row

This file is generated from sequential CPU public API runs under `data/results/amd-cpu/cpu/public_api/20261019_183932`.

- tenferro-rs commit: `14bc1ab933e3c7a9c966e08355670331ab9d993c`

## CPU Information

- Model: `AMD EPYC 9V45 96-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`

## Thread Environments

### Threads: 1

- Run metadata: `data/results/amd-cpu/cpu/public_api/20261019_183932/run_t1.yaml`
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

- Run metadata: `data/results/amd-cpu/cpu/public_api/20261019_183932/run_t4.yaml`
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

- CSV: `data/results/amd-cpu/cpu/public_api/20261019_183932/cpu_public_api_t1_20261019_183932.csv`
- CSV: `data/results/amd-cpu/cpu/public_api/20261019_183932/cpu_public_api_t4_20261019_183932.csv`
- Source table: `data/results/amd-cpu/cpu/public_api/20261019_183932/cpu_public_api_20261019_183932.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 2.991 ± 0.116 | - | 3.080 ± 0.065 | 3.475 ± 0.035 | 1.996 ± 0.101 | 2.000 ± 0.194 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 1.406 ± 0.012 | - | 1.585 ± 0.056 | 1.513 ± 0.013 | 1.570 ± 0.059 | 2.721 ± 0.574 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 19.572 ± 0.210 | - | 19.053 ± 0.241 | 23.315 ± 12.569 | 19.260 ± 0.299 | 20.278 ± 3.168 | 24.942 ± 0.681 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 13.965 ± 0.637 | - | 13.379 ± 0.315 | 14.395 ± 0.732 | 15.408 ± 0.512 | 20.659 ± 0.532 | 13.988 ± 0.172 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 33.615 ± 0.888 | - | 31.929 ± 0.365 | 26.811 ± 0.680 | 25.329 ± 0.250 | 39.828 ± 0.827 | 46.094 ± 0.635 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 16.300 ± 0.287 | - | 16.782 ± 3.015 | 11.938 ± 0.325 | 15.000 ± 0.283 | 36.169 ± 0.216 | 19.626 ± 0.330 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 35.572 ± 0.181 | - | 37.255 ± 0.534 | 35.085 ± 0.291 | 44.793 ± 2.709 | 21.826 ± 2.149 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 21.444 ± 0.074 | - | 22.079 ± 0.145 | 21.312 ± 0.171 | 19.145 ± 0.341 | 10.068 ± 0.142 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 35.975 ± 0.238 | - | unsupported | 36.142 ± 2.738 | 45.271 ± 0.850 | 21.433 ± 0.752 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 21.944 ± 0.139 | - | unsupported | 21.635 ± 0.222 | 19.615 ± 0.152 | 10.244 ± 0.358 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 6.868 ± 0.123 | - | 6.688 ± 0.099 | 7.232 ± 0.042 | 7.179 ± 0.027 | 8.804 ± 0.498 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 7.692 ± 0.070 | - | 7.721 ± 0.067 | 7.668 ± 0.057 | 9.345 ± 0.879 | 8.902 ± 0.295 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 46.877 ± 0.990 | - | 47.102 ± 1.526 | 53.258 ± 0.208 | 41.076 ± 0.120 | 33.865 ± 1.842 | 35.323 ± 1.814 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 18.897 ± 0.252 | - | 18.792 ± 0.333 | 21.785 ± 0.151 | 22.088 ± 0.238 | 31.718 ± 0.528 | 15.245 ± 1.008 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 60.892 ± 2.115 | - | 61.981 ± 1.929 | 73.598 ± 6.011 | 85.199 ± 0.320 | 63.233 ± 2.189 | 64.918 ± 1.834 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 28.909 ± 0.277 | - | 29.016 ± 0.394 | 31.938 ± 0.199 | 37.955 ± 0.331 | 60.305 ± 0.949 | 28.151 ± 1.688 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 13.438 ± 0.527 | - | 12.929 ± 0.276 | 13.437 ± 0.268 | 12.607 ± 0.136 | 13.404 ± 0.156 | 15.050 ± 0.378 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 10.054 ± 0.398 | - | 9.718 ± 0.064 | 9.920 ± 0.613 | 10.393 ± 0.264 | 13.162 ± 0.380 | 10.204 ± 0.168 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.356 ± 0.038 | - | 10.973 ± 0.093 | 11.163 ± 0.061 | 3.693 ± 0.311 | 2.979 ± 0.009 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 0.797 ± 0.099 | - | 9.473 ± 1.470 | 5.950 ± 0.028 | 2.097 ± 0.128 | 2.825 ± 0.011 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 5.375 ± 0.016 | - | 5.461 ± 0.050 | 6.366 ± 0.130 | 3.986 ± 0.057 | 5.076 ± 0.163 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 3.533 ± 0.051 | - | 3.569 ± 0.054 | 3.755 ± 0.032 | 3.776 ± 0.461 | 3.943 ± 0.137 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 3.733 ± 0.039 | - | 3.796 ± 0.097 | 3.751 ± 0.042 | 2.409 ± 0.012 | 2.439 ± 0.023 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 2.346 ± 1.278 | - | 2.288 ± 0.050 | 2.210 ± 0.214 | 1.961 ± 0.070 | 1.612 ± 0.029 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 3.983 ± 0.082 | - | 3.989 ± 0.057 | 4.989 ± 0.053 | 2.898 ± 0.020 | 3.014 ± 0.092 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 3.325 ± 0.049 | - | 3.427 ± 0.040 | 3.668 ± 0.106 | 3.207 ± 0.040 | 2.803 ± 0.017 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 35.532 ± 0.107 | - | 36.104 ± 0.356 | 35.727 ± 1.331 | 45.109 ± 1.788 | 21.090 ± 0.067 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 21.424 ± 0.025 | - | 22.109 ± 0.095 | 21.465 ± 0.176 | 19.149 ± 0.188 | 9.837 ± 0.127 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 35.156 ± 0.901 | - | unsupported | 41.560 ± 0.584 | 34.156 ± 1.586 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 19.758 ± 0.132 | - | unsupported | 21.535 ± 0.285 | 18.150 ± 0.046 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 19.192 ± 0.822 | - | 67.061 ± 0.247 | 19.153 ± 0.256 | 20.833 ± 0.658 | 21.008 ± 0.297 | 36.823 ± 0.652 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 14.397 ± 0.853 | - | 13.456 ± 1.086 | 13.660 ± 0.103 | 17.189 ± 0.305 | 21.389 ± 1.002 | 18.777 ± 0.731 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 23.651 ± 0.231 | - | 76.571 ± 5.057 | 24.442 ± 0.567 | 26.214 ± 0.241 | 25.033 ± 0.175 | 41.387 ± 0.724 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 19.491 ± 0.858 | - | 19.200 ± 0.663 | 20.052 ± 1.617 | 21.407 ± 0.947 | 25.530 ± 0.354 | 22.391 ± 0.457 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 8.456 ± 0.113 | - | 25.521 ± 0.083 | 18.894 ± 0.184 | 22.709 ± 1.197 | 35.333 ± 1.582 | 40.003 ± 1.105 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 5.763 ± 0.520 | - | 14.287 ± 0.079 | 10.301 ± 0.552 | 11.132 ± 0.056 | 36.685 ± 0.257 | 17.937 ± 3.224 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 5.978 ± 0.080 | - | 6.366 ± 0.082 | 8.469 ± 0.120 | 8.972 ± 0.505 | 8.894 ± 1.037 | 19.355 ± 1.937 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 5.619 ± 0.296 | - | 6.128 ± 0.351 | 6.913 ± 0.218 | 6.524 ± 0.103 | 7.866 ± 1.384 | 9.066 ± 3.033 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 13.313 ± 0.111 | - | 13.488 ± 0.105 | 17.299 ± 0.263 | 16.271 ± 0.630 | 13.359 ± 0.365 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 11.468 ± 0.712 | - | 11.642 ± 0.239 | 12.724 ± 0.525 | 14.878 ± 0.304 | 13.422 ± 0.264 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 8.157 ± 0.013 | - | 8.186 ± 0.045 | 21.943 ± 0.234 | 61.314 ± 0.579 | 29.068 ± 1.458 | 31.753 ± 1.404 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 3.568 ± 0.046 | - | 4.469 ± 0.892 | 10.239 ± 0.303 | 30.851 ± 0.042 | 29.381 ± 7.277 | 15.721 ± 2.149 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 24.151 ± 0.726 | - | 81.044 ± 1.180 | 23.903 ± 0.812 | 26.548 ± 1.152 | 25.411 ± 0.365 | 43.665 ± 1.917 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 20.226 ± 0.771 | - | 20.528 ± 0.612 | 20.091 ± 0.428 | 21.227 ± 0.348 | 25.751 ± 0.826 | 25.297 ± 20.678 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 4.156 ± 0.033 | - | 4.200 ± 0.019 | 13.928 ± 0.142 | 6.620 ± 0.123 | 23.057 ± 1.355 | 25.726 ± 1.182 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 2.952 ± 0.057 | - | 2.907 ± 0.119 | 6.483 ± 0.072 | 3.980 ± 0.860 | 24.335 ± 3.328 | 14.151 ± 1.056 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 6.150 ± 0.033 | - | 6.104 ± 0.013 | 7.230 ± 0.063 | 5.318 ± 0.113 | 15.149 ± 0.632 | 16.330 ± 1.381 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 2.581 ± 0.043 | - | 2.620 ± 0.065 | 3.171 ± 0.051 | 2.949 ± 0.554 | 15.391 ± 0.061 | 8.082 ± 2.194 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 7.912 ± 0.077 | - | 7.913 ± 0.062 | 18.880 ± 0.200 | 23.910 ± 0.335 | 29.406 ± 8.671 | 33.256 ± 1.152 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 3.868 ± 0.058 | - | 3.807 ± 0.037 | 8.035 ± 0.160 | 13.673 ± 0.382 | 30.734 ± 2.366 | 16.255 ± 2.090 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 4.141 ± 0.016 | - | 4.155 ± 0.008 | 7.403 ± 0.058 | 13.542 ± 0.683 | 17.983 ± 0.235 | 20.495 ± 1.267 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 2.019 ± 0.033 | - | 2.054 ± 0.024 | 3.185 ± 0.048 | 6.823 ± 0.485 | 18.808 ± 0.770 | 9.482 ± 2.484 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.110 ± 0.008 | 0.121 ± 0.005 | unsupported | unsupported | 0.188 ± 0.012 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.186 ± 0.025 | 0.225 ± 0.014 | unsupported | unsupported | 0.172 ± 0.008 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 25.121 ± 0.505 | - | 83.176 ± 0.503 | 24.468 ± 0.390 | 26.266 ± 0.888 | 26.527 ± 1.094 | 47.844 ± 1.934 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 19.503 ± 0.945 | - | 19.296 ± 0.519 | 21.301 ± 1.471 | 20.762 ± 1.458 | 25.598 ± 0.901 | 23.328 ± 0.879 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 24.800 ± 0.664 | - | 83.777 ± 1.830 | 24.526 ± 0.591 | 26.198 ± 0.650 | 26.541 ± 0.485 | 46.186 ± 0.596 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 19.815 ± 0.842 | - | 19.185 ± 0.372 | 19.342 ± 0.575 | 21.299 ± 0.538 | 26.123 ± 0.413 | 24.475 ± 1.293 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 23.985 ± 0.278 | - | 78.966 ± 2.515 | 24.465 ± 0.828 | 26.396 ± 0.511 | 25.708 ± 0.513 | 42.806 ± 0.751 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 20.274 ± 0.726 | - | 20.110 ± 0.215 | 19.695 ± 0.364 | 21.015 ± 1.507 | 25.778 ± 5.440 | 23.478 ± 1.367 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 19.469 ± 0.620 | - | 67.154 ± 0.608 | 19.252 ± 0.460 | 21.049 ± 0.676 | 20.726 ± 1.376 | 36.698 ± 0.273 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 14.461 ± 1.000 | - | 13.494 ± 0.192 | 13.844 ± 0.977 | 17.131 ± 1.090 | 20.677 ± 0.713 | 18.673 ± 1.014 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 32.574 ± 0.217 | - | 32.491 ± 0.077 | 16.060 ± 0.103 | 35.542 ± 1.816 | 81.482 ± 0.738 | 55.848 ± 1.135 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 15.369 ± 0.169 | - | 15.415 ± 0.044 | 6.758 ± 0.078 | 17.495 ± 0.498 | 84.715 ± 2.111 | 25.307 ± 2.813 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 10.388 ± 0.095 | - | 10.371 ± 0.046 | 9.278 ± 0.062 | 44.881 ± 0.084 | 10.058 ± 0.264 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 5.885 ± 0.228 | - | 5.826 ± 0.147 | 5.371 ± 0.058 | 22.573 ± 0.039 | 9.842 ± 0.244 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.346 ± 0.017 | - | 1.353 ± 0.005 | 28.264 ± 0.543 | 1.288 ± 0.027 | 1.215 ± 0.034 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.683 ± 0.078 | - | 0.615 ± 0.025 | 10.431 ± 0.130 | 0.743 ± 0.611 | 1.058 ± 0.128 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 1.143 ± 0.034 | - | 1.128 ± 0.030 | 2.047 ± 0.016 | 2.898 ± 0.010 | 0.881 ± 0.024 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 1.012 ± 0.153 | - | 1.030 ± 0.061 | 1.075 ± 0.014 | 2.206 ± 0.031 | 0.834 ± 0.052 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 9.666 ± 0.068 | - | 9.736 ± 0.082 | 9.121 ± 0.049 | 45.116 ± 0.567 | 10.658 ± 0.138 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 5.838 ± 0.239 | - | 6.133 ± 0.204 | 5.622 ± 0.158 | 22.583 ± 0.111 | 10.432 ± 0.336 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.225 ± 0.019 | - | 1.241 ± 0.055 | 23.491 ± 0.244 | 1.218 ± 0.012 | 1.169 ± 0.026 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.574 ± 0.029 | - | 0.581 ± 0.056 | 10.308 ± 0.198 | 0.966 ± 0.169 | 1.172 ± 0.039 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 6.747 ± 0.077 | - | 6.710 ± 0.180 | 8.244 ± 0.051 | 11.508 ± 0.061 | 5.383 ± 0.177 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 3.846 ± 0.208 | - | 3.855 ± 0.103 | 4.110 ± 0.131 | 8.722 ± 2.928 | 5.247 ± 0.179 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 6.770 ± 0.056 | - | 6.753 ± 0.164 | 6.202 ± 0.123 | 22.381 ± 0.226 | 6.497 ± 0.174 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 5.894 ± 0.093 | - | 5.894 ± 0.418 | 5.509 ± 0.125 | 11.298 ± 0.171 | 6.632 ± 0.307 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.712 ± 0.015 | - | 0.732 ± 0.078 | 2.034 ± 0.061 | 1.130 ± 0.045 | 0.727 ± 0.060 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.299 ± 0.047 | - | 0.344 ± 0.085 | 0.896 ± 0.012 | 1.357 ± 0.042 | 0.603 ± 0.052 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 0.809 ± 0.020 | - | 0.816 ± 0.012 | 0.573 ± 0.023 | 0.954 ± 0.017 | 0.807 ± 0.054 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.671 ± 0.043 | - | 0.691 ± 0.022 | 0.234 ± 0.031 | 0.741 ± 0.017 | 0.793 ± 0.045 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 6.679 ± 0.118 | - | 6.731 ± 0.132 | 6.181 ± 0.140 | 16.260 ± 0.092 | 6.247 ± 0.080 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 5.837 ± 0.030 | - | 5.882 ± 0.175 | 5.552 ± 0.135 | 8.248 ± 0.055 | 6.541 ± 0.251 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.762 ± 0.049 | - | 0.766 ± 0.074 | 1.993 ± 0.141 | 1.244 ± 0.065 | 0.682 ± 0.033 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.286 ± 0.042 | - | 0.294 ± 0.046 | 0.865 ± 0.054 | 0.737 ± 0.648 | 0.606 ± 0.072 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 0.812 ± 0.050 | - | 0.831 ± 0.033 | 0.566 ± 0.022 | 0.886 ± 0.012 | 1.235 ± 1.211 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.710 ± 0.036 | - | 0.733 ± 0.037 | 0.209 ± 0.022 | 0.679 ± 0.022 | 0.769 ± 0.006 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.774 ± 0.026 | 0.737 ± 0.015 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.252 ± 0.051 | 0.252 ± 0.023 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 22.911 ± 1.742 | - | 22.236 ± 0.776 | 19.734 ± 0.171 | 24.769 ± 0.440 | 28.598 ± 0.754 | 34.397 ± 1.803 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 9.643 ± 0.141 | - | 9.659 ± 0.208 | 9.321 ± 0.142 | 14.829 ± 6.550 | 29.619 ± 1.068 | 16.278 ± 1.672 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 87.335 ± 1.247 | - | 89.790 ± 1.517 | 33.861 ± 1.218 | 67.901 ± 1.755 | 102.880 ± 0.907 | 103.331 ± 2.298 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 16.140 ± 2.264 | - | 15.583 ± 1.255 | 15.965 ± 1.478 | 36.028 ± 0.172 | 99.389 ± 0.614 | 51.466 ± 2.565 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 25.722 ± 0.327 | - | 85.856 ± 0.853 | 28.457 ± 0.636 | 41.907 ± 1.654 | 28.928 ± 0.312 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 22.062 ± 0.989 | - | 22.206 ± 1.140 | 20.177 ± 0.346 | 28.249 ± 1.673 | 29.228 ± 1.197 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 21.208 ± 0.390 | - | 71.202 ± 0.727 | 21.069 ± 0.302 | 24.398 ± 15.609 | 21.777 ± 0.338 | 37.392 ± 0.841 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 13.686 ± 0.442 | - | 13.382 ± 0.299 | 13.825 ± 0.371 | 16.087 ± 0.942 | 21.414 ± 0.557 | 19.276 ± 1.568 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 6.176 ± 0.011 | - | 6.194 ± 0.026 | 25.643 ± 0.206 | 63.125 ± 6.215 | 27.906 ± 1.671 | 30.688 ± 1.218 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 3.067 ± 0.177 | - | 3.047 ± 0.232 | 10.698 ± 0.129 | 30.662 ± 0.195 | 28.979 ± 2.816 | 15.878 ± 3.719 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 79.218 ± 0.796 | - | 82.584 ± 0.362 | 57.174 ± 0.532 | 21.153 ± 0.313 | 70.356 ± 2.402 | 70.657 ± 0.820 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 13.983 ± 0.306 | - | 14.405 ± 0.092 | 27.492 ± 0.813 | 15.805 ± 1.148 | 69.888 ± 2.958 | 34.793 ± 1.171 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 23.687 ± 0.213 | - | 72.476 ± 2.282 | 24.505 ± 0.673 | 26.974 ± 0.817 | 25.564 ± 0.279 | 43.151 ± 0.711 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 19.346 ± 0.576 | - | 19.469 ± 0.555 | 19.378 ± 0.271 | 20.944 ± 0.544 | 26.731 ± 0.413 | 22.731 ± 1.108 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 14.498 ± 0.018 | - | 14.408 ± 0.052 | 33.310 ± 0.086 | 5.743 ± 0.572 | 33.661 ± 1.423 | 37.973 ± 8.582 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 6.786 ± 0.082 | - | 6.884 ± 0.092 | 15.046 ± 0.376 | 3.920 ± 0.059 | 33.961 ± 1.652 | 17.190 ± 1.428 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 84.125 ± 0.579 | 84.192 ± 1.094 | unsupported | 28.459 ± 0.888 | 42.044 ± 0.876 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 21.546 ± 1.229 | 32.118 ± 1.024 | unsupported | 20.752 ± 0.852 | 27.339 ± 0.424 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.561 ± 0.006 | - | 1.127 ± 0.016 | 0.409 ± 0.027 | 0.568 ± 0.023 | 1.651 ± 0.546 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.300 ± 0.050 | - | 0.865 ± 0.110 | 0.249 ± 0.012 | 0.548 ± 0.063 | 1.663 ± 0.653 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.551 ± 0.060 | - | 1.900 ± 0.067 | 0.456 ± 0.048 | 0.526 ± 0.151 | 1.938 ± 0.656 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.261 ± 0.122 | - | 1.700 ± 0.028 | 0.219 ± 0.016 | 0.407 ± 0.075 | 1.773 ± 0.578 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 0.841 ± 0.074 | - | unsupported | 0.740 ± 0.075 | 0.716 ± 0.046 | 2.121 ± 0.419 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.424 ± 0.114 | - | unsupported | 0.444 ± 0.095 | 0.607 ± 0.036 | 1.960 ± 0.972 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.188 ± 0.011 | - | 0.262 ± 0.030 | 0.356 ± 0.018 | 0.193 ± 0.040 | 0.783 ± 0.093 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.338 ± 0.042 | - | 0.330 ± 0.031 | 0.195 ± 0.011 | 0.167 ± 0.015 | 0.760 ± 0.491 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.477 ± 0.018 | - | 1.037 ± 0.027 | 0.661 ± 0.038 | 0.628 ± 0.022 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.233 ± 0.032 | - | 0.783 ± 0.096 | 0.361 ± 0.048 | 0.407 ± 0.011 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.543 ± 0.041 | - | 1.129 ± 0.018 | 0.433 ± 0.058 | 0.490 ± 0.037 | 0.848 ± 1.599 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.249 ± 0.095 | - | 0.774 ± 0.045 | 0.247 ± 0.035 | 0.263 ± 0.036 | 1.964 ± 1.470 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.429 ± 0.061 | - | 0.553 ± 0.011 | 0.206 ± 0.007 | 0.228 ± 0.023 | 0.922 ± 0.088 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.466 ± 0.031 | - | 0.611 ± 0.076 | 0.245 ± 0.021 | 0.274 ± 0.028 | 0.864 ± 0.152 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.135 ± 0.023 | - | 2.340 ± 0.050 | 1.114 ± 0.021 | 0.839 ± 0.029 | 2.601 ± 0.475 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.611 ± 0.046 | - | 1.924 ± 0.278 | 0.636 ± 0.058 | 0.706 ± 0.043 | 2.590 ± 0.876 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.027 ± 0.001 | - | 0.030 ± 0.000 | 0.043 ± 0.000 | 0.070 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.053 ± 0.000 | - | 0.059 ± 0.000 | 0.061 ± 0.000 | 0.099 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.134 ± 0.001 | - | 0.147 ± 0.001 | 0.155 ± 0.000 | 0.188 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.040 ± 0.001 | - | 0.045 ± 0.002 | 0.101 ± 0.002 | 0.081 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.001 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.019 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.325 ± 0.005 | - | 0.349 ± 0.012 | 0.812 ± 0.010 | 0.429 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.013 ± 0.000 | - | 0.016 ± 0.000 | 0.020 ± 0.000 | 0.040 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.020 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.026 ± 0.000 | - | 0.064 ± 0.002 | 0.047 ± 0.000 | 0.061 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.055 ± 0.012 | - | 0.092 ± 0.003 | 0.072 ± 0.000 | 0.086 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.140 ± 0.002 | - | 0.180 ± 0.009 | 0.179 ± 0.002 | 0.203 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.042 ± 0.002 | - | 0.076 ± 0.003 | 0.094 ± 0.001 | 0.074 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.002 | - | 0.018 ± 0.018 | 0.005 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.002 ± 0.000 | - | 0.018 ± 0.008 | 0.008 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.001 ± 0.001 | - | 0.017 ± 0.006 | 0.005 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.001 ± 0.001 | - | 0.020 ± 0.013 | 0.005 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.001 ± 0.000 | - | 0.017 ± 0.011 | 0.005 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.001 ± 0.000 | - | 0.018 ± 0.007 | 0.005 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.002 ± 0.002 | - | 0.012 ± 0.007 | 0.005 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.001 ± 0.000 | - | 0.010 ± 0.006 | 0.005 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.001 ± 0.000 | - | 0.017 ± 0.011 | 0.005 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.002 ± 0.002 | - | 0.017 ± 0.009 | 0.006 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.342 ± 0.022 | - | 0.391 ± 0.023 | 0.766 ± 0.011 | 1.804 ± 3.441 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.001 ± 0.000 | - | 0.016 ± 0.009 | 0.005 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.013 ± 0.000 | - | 0.048 ± 0.001 | 0.028 ± 0.000 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.002 ± 0.000 | - | 0.014 ± 0.003 | 0.005 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.002 ± 0.001 | - | 0.018 ± 0.020 | 0.006 ± 0.000 | 0.008 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.336 ± 0.006 | - | 0.353 ± 0.011 | 0.399 ± 0.006 | 0.492 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.350 ± 0.015 | - | 1.340 ± 0.006 | 1.343 ± 0.035 | 1.700 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 4.024 ± 0.065 | - | 3.991 ± 0.013 | 4.089 ± 0.008 | 5.707 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 2.196 ± 0.050 | - | 2.176 ± 0.023 | 2.062 ± 0.062 | 1.534 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.066 ± 0.004 | - | 0.067 ± 0.000 | 0.067 ± 0.001 | 0.090 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.013 ± 0.000 | - | 0.016 ± 0.000 | 0.016 ± 0.001 | 0.039 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.017 ± 0.000 | - | 0.020 ± 0.000 | 0.020 ± 0.000 | 0.042 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 17.916 ± 0.707 | - | 17.357 ± 0.051 | 15.671 ± 0.047 | 12.571 ± 0.124 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.507 ± 0.019 | - | 0.526 ± 0.010 | 0.518 ± 0.017 | 0.576 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.013 ± 0.001 | - | 0.018 ± 0.001 | 0.014 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.033 ± 0.000 | - | 0.036 ± 0.000 | 0.035 ± 0.009 | 0.064 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.546 ± 0.036 | - | 0.563 ± 0.016 | 0.583 ± 0.009 | 0.530 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 1.878 ± 0.113 | - | 1.874 ± 0.016 | 1.388 ± 0.008 | 1.463 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 5.517 ± 0.249 | - | 5.514 ± 0.021 | 5.430 ± 0.120 | 3.959 ± 0.219 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 2.061 ± 0.039 | - | 2.120 ± 0.062 | 1.382 ± 0.016 | 1.957 ± 3.093 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.020 ± 0.006 | 0.006 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.091 ± 0.003 | - | 0.113 ± 0.010 | 0.092 ± 0.003 | 0.096 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.018 ± 0.008 | 0.007 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.007 ± 0.000 | - | 0.018 ± 0.005 | 0.008 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.015 ± 0.004 | 0.008 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.007 ± 0.001 | - | 0.020 ± 0.007 | 0.011 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.018 ± 0.000 | - | 0.031 ± 0.006 | 0.016 ± 0.000 | 0.040 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.004 ± 0.003 | - | 0.020 ± 0.011 | 0.008 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.009 ± 0.003 | - | 0.028 ± 0.017 | 0.009 ± 0.004 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.023 ± 0.000 | - | 0.038 ± 0.005 | 0.020 ± 0.000 | 0.043 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 16.369 ± 0.096 | - | 16.756 ± 0.342 | 10.962 ± 0.458 | 14.949 ± 0.682 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.006 ± 0.003 | - | 0.017 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.708 ± 0.016 | - | 0.738 ± 0.014 | 0.697 ± 0.008 | 0.545 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.015 ± 0.000 | - | 0.031 ± 0.007 | 0.015 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.044 ± 0.000 | - | 0.063 ± 0.012 | 0.037 ± 0.000 | 0.067 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.352 ± 0.008 | - | 0.354 ± 0.003 | 0.219 ± 0.004 | 0.204 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.301 ± 0.014 | - | 1.307 ± 0.020 | 1.154 ± 0.005 | 1.179 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 3.538 ± 0.008 | - | 3.550 ± 0.025 | 3.505 ± 0.030 | 4.035 ± 0.132 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 0.829 ± 0.022 | - | 0.920 ± 0.185 | 0.846 ± 0.006 | 0.825 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.002 ± 0.001 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.042 ± 0.001 | - | 0.045 ± 0.001 | 0.048 ± 0.001 | 0.075 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.001 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.011 ± 0.000 | - | 0.015 ± 0.001 | 0.015 ± 0.001 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.015 ± 0.001 | - | 0.018 ± 0.000 | 0.018 ± 0.001 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 6.734 ± 0.044 | - | 6.881 ± 0.181 | 6.590 ± 0.111 | 6.321 ± 0.063 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.324 ± 0.011 | - | 0.331 ± 0.009 | 0.363 ± 0.016 | 0.450 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.011 ± 0.000 | - | 0.014 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.028 ± 0.000 | - | 0.032 ± 0.000 | 0.030 ± 0.000 | 0.050 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.580 ± 0.019 | - | 0.615 ± 0.019 | 0.383 ± 0.009 | 0.203 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.962 ± 0.054 | - | 1.940 ± 0.017 | 1.189 ± 0.003 | 1.047 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 5.531 ± 0.661 | - | 5.093 ± 0.035 | 4.838 ± 0.086 | 2.373 ± 0.061 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 0.739 ± 0.020 | - | 0.761 ± 0.024 | 0.725 ± 0.027 | 0.981 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.002 ± 0.002 | - | 0.020 ± 0.009 | 0.006 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.061 ± 0.003 | - | 0.085 ± 0.015 | 0.069 ± 0.002 | 0.071 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.018 ± 0.007 | 0.007 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.006 ± 0.001 | - | 0.021 ± 0.007 | 0.007 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.018 ± 0.003 | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.007 ± 0.000 | - | 0.020 ± 0.007 | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.016 ± 0.000 | - | 0.034 ± 0.006 | 0.014 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.021 ± 0.006 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.009 ± 0.000 | - | 0.030 ± 0.002 | 0.012 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.021 ± 0.000 | - | 0.034 ± 0.008 | 0.018 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 5.778 ± 0.038 | - | 5.815 ± 0.105 | 5.455 ± 0.042 | 8.095 ± 0.586 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.006 ± 0.002 | - | 0.024 ± 0.006 | 0.008 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.484 ± 0.014 | - | 0.519 ± 0.421 | 0.515 ± 0.027 | 0.269 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.016 ± 0.000 | - | 0.028 ± 0.007 | 0.018 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.040 ± 0.000 | - | 0.060 ± 0.007 | 0.031 ± 0.000 | 0.052 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.130 ± 0.006 | - | 0.138 ± 0.001 | 0.118 ± 0.002 | 0.205 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 0.677 ± 0.015 | - | 0.681 ± 0.008 | 0.586 ± 0.007 | 1.135 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 1.962 ± 0.008 | - | 1.980 ± 0.008 | 1.840 ± 0.005 | 3.914 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.323 ± 0.005 | - | 0.334 ± 0.012 | 0.360 ± 0.009 | 0.802 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.001 ± 0.001 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.019 ± 0.000 | - | 0.022 ± 0.000 | 0.024 ± 0.000 | 0.070 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 2.624 ± 0.016 | - | 2.698 ± 0.077 | 2.870 ± 0.016 | 6.093 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.144 ± 0.000 | - | 0.148 ± 0.003 | 0.151 ± 0.002 | 0.445 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.016 ± 0.000 | - | 0.019 ± 0.000 | 0.018 ± 0.000 | 0.050 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.234 ± 0.010 | - | 0.265 ± 0.020 | 0.208 ± 0.002 | 0.205 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 0.938 ± 0.023 | - | 0.968 ± 0.047 | 0.616 ± 0.122 | 1.063 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 2.518 ± 0.024 | - | 2.569 ± 0.025 | 2.237 ± 0.026 | 2.373 ± 0.126 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.437 ± 0.037 | - | 0.443 ± 0.014 | 0.440 ± 0.011 | 1.060 ± 0.066 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.004 ± 0.003 | - | 0.019 ± 0.013 | 0.007 ± 0.001 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.024 ± 0.001 | - | 0.042 ± 0.011 | 0.030 ± 0.002 | 0.070 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.014 ± 0.003 | 0.007 ± 0.001 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.019 ± 0.006 | 0.006 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.019 ± 0.007 | 0.007 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.022 ± 0.006 | 0.009 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.008 ± 0.001 | - | 0.027 ± 0.007 | 0.010 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.017 ± 0.006 | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | 0.023 ± 0.009 | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.011 ± 0.001 | - | 0.023 ± 0.019 | 0.012 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 3.306 ± 0.080 | - | 3.398 ± 0.090 | 3.297 ± 0.027 | 7.989 ± 0.178 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.004 ± 0.002 | - | 0.021 ± 0.007 | 0.008 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.178 ± 0.001 | - | 0.200 ± 0.010 | 0.188 ± 0.002 | 0.258 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.008 ± 0.001 | - | 0.020 ± 0.002 | 0.012 ± 0.000 | 0.014 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.020 ± 0.000 | - | 0.037 ± 0.009 | 0.019 ± 0.000 | 0.048 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.086 ± 0.001 | - | 0.089 ± 0.002 | 0.099 ± 0.009 | 0.076 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.096 ± 0.001 | - | 0.104 ± 0.001 | 0.138 ± 0.027 | 0.124 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.251 ± 0.008 | - | 0.280 ± 0.007 | 0.310 ± 0.006 | 0.344 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.077 ± 0.002 | - | 0.083 ± 0.002 | 0.101 ± 0.004 | 0.112 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.001 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.008 ± 0.000 | - | 0.012 ± 0.001 | 0.013 ± 0.001 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.002 ± 0.001 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.030 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.612 ± 0.035 | - | 0.632 ± 0.011 | 0.774 ± 0.009 | 0.774 ± 0.089 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.043 ± 0.000 | - | 0.047 ± 0.000 | 0.045 ± 0.000 | 0.060 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.101 ± 0.006 | - | 0.120 ± 0.006 | 0.090 ± 0.001 | 0.076 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.160 ± 0.005 | - | 0.183 ± 0.009 | 0.118 ± 0.004 | 0.122 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.398 ± 0.009 | - | 0.451 ± 0.012 | 0.204 ± 0.010 | 0.399 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.097 ± 0.001 | - | 0.130 ± 0.026 | 0.119 ± 0.015 | 0.113 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.020 ± 0.010 | 0.011 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.010 ± 0.000 | - | 0.029 ± 0.007 | 0.017 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.002 ± 0.002 | - | 0.021 ± 0.002 | 0.011 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.011 ± 0.000 | - | 0.031 ± 0.005 | 0.018 ± 0.001 | 0.023 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.014 ± 0.006 | 0.011 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.002 ± 0.002 | - | 0.014 ± 0.003 | 0.012 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.003 ± 0.002 | - | 0.015 ± 0.004 | 0.012 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.003 ± 0.000 | - | 0.016 ± 0.007 | 0.013 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.020 ± 0.003 | 0.012 ± 0.001 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.003 ± 0.001 | - | 0.021 ± 0.008 | 0.012 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.004 ± 0.000 | - | 0.016 ± 0.007 | 0.013 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.775 ± 0.009 | - | 0.847 ± 0.019 | 0.416 ± 0.042 | 0.560 ± 0.060 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.003 ± 0.002 | - | 0.015 ± 0.003 | 0.013 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.065 ± 0.017 | - | 0.082 ± 0.007 | 0.035 ± 0.001 | 0.085 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.004 ± 0.003 | - | 0.015 ± 0.006 | 0.013 ± 0.000 | 0.010 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.005 ± 0.000 | - | 0.024 ± 0.000 | 0.015 ± 0.000 | 0.016 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.120 ± 0.001 | - | 0.126 ± 0.001 | 0.109 ± 0.000 | 0.166 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.308 ± 0.007 | - | 0.320 ± 0.004 | 0.296 ± 0.006 | 0.436 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 0.746 ± 0.011 | - | 0.768 ± 0.012 | 0.771 ± 0.006 | 1.250 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.294 ± 0.013 | - | 0.302 ± 0.016 | 0.363 ± 0.007 | 0.362 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.013 ± 0.000 | 0.042 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.011 ± 0.000 | - | 0.015 ± 0.000 | 0.016 ± 0.000 | 0.032 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.003 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 2.387 ± 0.038 | - | 2.873 ± 0.023 | 2.861 ± 0.027 | 2.628 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.003 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.068 ± 0.000 | - | 0.072 ± 0.000 | 0.092 ± 0.001 | 0.128 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.121 ± 0.001 | - | 0.156 ± 0.012 | 0.161 ± 0.001 | 0.160 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.318 ± 0.017 | - | 0.339 ± 0.011 | 0.407 ± 0.009 | 0.435 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 0.749 ± 0.020 | - | 0.801 ± 0.036 | 1.080 ± 0.012 | 0.734 ± 0.188 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.399 ± 0.033 | - | 0.358 ± 0.013 | 0.341 ± 0.017 | 0.431 ± 0.062 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.017 ± 0.021 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.009 ± 0.000 | - | 0.040 ± 0.002 | 0.023 ± 0.001 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.001 ± 0.000 | - | 0.017 ± 0.008 | 0.005 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.016 ± 0.000 | - | 0.031 ± 0.007 | 0.024 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.014 ± 0.006 | 0.006 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.002 ± 0.001 | - | 0.018 ± 0.005 | 0.005 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.003 ± 0.002 | - | 0.018 ± 0.021 | 0.006 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.006 ± 0.002 | - | 0.018 ± 0.010 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.002 ± 0.002 | - | 0.015 ± 0.006 | 0.006 ± 0.000 | 0.007 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.003 ± 0.002 | - | 0.016 ± 0.003 | 0.007 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.007 ± 0.003 | - | 0.021 ± 0.010 | 0.009 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 2.729 ± 0.063 | - | 3.308 ± 0.019 | 2.449 ± 0.036 | 3.622 ± 0.806 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.003 ± 0.003 | - | 0.023 ± 0.007 | 0.006 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.068 ± 0.000 | - | 0.103 ± 0.003 | 0.142 ± 0.012 | 0.125 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.005 ± 0.002 | - | 0.022 ± 0.020 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.007 ± 0.000 | - | 0.024 ± 0.002 | 0.013 ± 0.185 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.105 ± 0.000 | - | 0.089 ± 0.001 | 0.049 ± 0.001 | 0.161 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.125 ± 0.002 | - | 0.107 ± 0.002 | 0.069 ± 0.001 | 0.223 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.288 ± 0.007 | - | 0.269 ± 0.007 | 0.246 ± 0.008 | 0.440 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.065 ± 0.001 | - | 0.072 ± 0.001 | 0.081 ± 0.001 | 0.091 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.005 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.006 ± 0.000 | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.001 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.006 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.526 ± 0.018 | - | 0.549 ± 0.012 | 0.620 ± 0.027 | 0.575 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.042 ± 0.000 | - | 0.046 ± 0.000 | 0.043 ± 0.000 | 0.075 ± 0.094 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.112 ± 0.000 | - | 0.123 ± 0.007 | 0.064 ± 0.000 | 0.164 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.134 ± 0.012 | - | 0.151 ± 0.010 | 0.086 ± 0.001 | 0.242 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.313 ± 0.020 | - | 0.327 ± 0.010 | 0.215 ± 0.006 | 0.511 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.068 ± 0.000 | - | 0.118 ± 0.022 | 0.100 ± 0.001 | 0.086 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.030 ± 0.017 | 0.012 ± 0.001 | 0.006 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.007 ± 0.000 | - | 0.040 ± 0.019 | 0.019 ± 0.001 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.022 ± 0.008 | 0.013 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.002 ± 0.002 | - | 0.033 ± 0.013 | 0.013 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.022 ± 0.011 | 0.013 ± 0.000 | 0.006 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.002 ± 0.001 | - | 0.025 ± 0.009 | 0.013 ± 0.001 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.025 ± 0.014 | 0.014 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.002 ± 0.001 | - | 0.030 ± 0.020 | 0.013 ± 0.002 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.002 ± 0.001 | - | 0.026 ± 0.007 | 0.013 ± 0.001 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.003 ± 0.001 | - | 0.025 ± 0.018 | 0.014 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.548 ± 0.017 | - | 0.617 ± 0.021 | 0.352 ± 0.029 | 0.453 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.024 ± 0.007 | 0.014 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.043 ± 0.000 | - | 0.090 ± 0.022 | 0.035 ± 0.001 | 0.085 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.003 ± 0.001 | - | 0.033 ± 0.014 | 0.014 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.004 ± 0.002 | - | 0.025 ± 0.014 | 0.016 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.038 ± 0.012 | - | 1.044 ± 0.011 | 0.878 ± 0.005 | 1.209 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 3.574 ± 0.016 | - | 3.577 ± 0.016 | 3.100 ± 0.009 | 3.180 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 10.750 ± 0.128 | - | 10.743 ± 0.051 | 9.990 ± 0.102 | 9.197 ± 0.068 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 3.006 ± 0.012 | - | 2.992 ± 0.017 | 3.689 ± 0.031 | 2.504 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.112 ± 0.002 | - | 0.116 ± 0.002 | 0.138 ± 0.006 | 0.133 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.103 ± 0.001 | - | 0.111 ± 0.004 | 0.118 ± 0.004 | 0.131 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.012 ± 0.000 | - | 0.016 ± 0.008 | 0.015 ± 0.001 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.012 ± 0.000 | - | 0.015 ± 0.000 | 0.014 ± 0.006 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.032 ± 0.000 | - | 0.036 ± 0.000 | 0.034 ± 0.001 | 0.036 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.015 ± 0.000 | - | 0.019 ± 0.001 | 0.016 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.043 ± 0.000 | - | 0.047 ± 0.000 | 0.043 ± 0.001 | 0.045 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 23.924 ± 0.073 | - | 24.653 ± 0.366 | 28.048 ± 0.384 | 19.938 ± 0.260 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 0.888 ± 0.009 | - | 0.907 ± 0.006 | 1.048 ± 0.020 | 0.942 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.029 ± 0.000 | - | 0.033 ± 0.000 | 0.028 ± 0.001 | 0.032 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.084 ± 0.000 | - | 0.088 ± 0.000 | 0.080 ± 0.000 | 0.105 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.596 ± 0.012 | - | 1.597 ± 0.006 | 0.892 ± 0.439 | 1.178 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 5.233 ± 0.075 | - | 5.418 ± 0.109 | 3.189 ± 0.024 | 1.529 ± 0.232 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 11.177 ± 0.107 | - | 11.262 ± 0.134 | 12.063 ± 0.225 | 4.806 ± 0.093 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 2.858 ± 0.034 | - | 2.917 ± 0.038 | 2.915 ± 0.025 | 2.745 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.016 ± 0.001 | 0.008 ± 0.000 | 0.008 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.164 ± 0.013 | - | 0.190 ± 0.009 | 0.174 ± 0.007 | 0.132 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.007 ± 0.000 | - | 0.026 ± 0.003 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.140 ± 0.001 | - | 0.169 ± 0.006 | 0.149 ± 0.004 | 0.121 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.017 ± 0.000 | - | 0.032 ± 0.012 | 0.015 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.007 ± 0.000 | - | 0.026 ± 0.011 | 0.011 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.017 ± 0.000 | - | 0.034 ± 0.011 | 0.014 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.046 ± 0.001 | - | 0.065 ± 0.002 | 0.034 ± 0.001 | 0.034 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.008 ± 0.000 | - | 0.022 ± 0.007 | 0.012 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.022 ± 0.000 | - | 0.037 ± 0.011 | 0.017 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.061 ± 0.000 | - | 0.081 ± 0.006 | 0.044 ± 0.000 | 0.043 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 22.760 ± 0.121 | - | 24.566 ± 0.182 | 23.112 ± 0.407 | 39.214 ± 24.638 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.014 ± 0.000 | - | 0.027 ± 0.006 | 0.017 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 1.257 ± 0.089 | - | 1.318 ± 0.036 | 1.323 ± 0.026 | 0.467 ± 0.098 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.042 ± 0.000 | - | 0.057 ± 0.013 | 0.029 ± 0.000 | 0.032 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.119 ± 0.005 | - | 0.140 ± 0.009 | 0.083 ± 0.000 | 0.090 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.790 ± 0.006 | - | 0.796 ± 0.006 | 0.633 ± 0.009 | 0.729 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 3.052 ± 0.012 | - | 3.076 ± 0.019 | 2.602 ± 0.032 | 2.353 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 9.220 ± 0.025 | - | 9.202 ± 0.032 | 8.543 ± 0.024 | 7.401 ± 0.062 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.400 ± 0.017 | - | 1.393 ± 0.010 | 1.516 ± 0.032 | 1.280 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.078 ± 0.008 | - | 0.080 ± 0.002 | 0.090 ± 0.002 | 0.111 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.063 ± 0.001 | - | 0.066 ± 0.001 | 0.071 ± 0.002 | 0.102 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.011 ± 0.007 | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.006 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.010 ± 0.000 | - | 0.014 ± 0.000 | 0.012 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.026 ± 0.000 | - | 0.030 ± 0.000 | 0.028 ± 0.001 | 0.027 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.013 ± 0.000 | - | 0.017 ± 0.000 | 0.014 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.035 ± 0.000 | - | 0.038 ± 0.000 | 0.035 ± 0.000 | 0.035 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 11.075 ± 0.045 | - | 11.169 ± 0.160 | 12.112 ± 0.347 | 10.682 ± 0.664 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 0.607 ± 0.008 | - | 0.613 ± 0.008 | 0.685 ± 0.009 | 0.701 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.025 ± 0.000 | - | 0.028 ± 0.000 | 0.024 ± 0.000 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.070 ± 0.002 | - | 0.076 ± 0.004 | 0.067 ± 0.017 | 0.080 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 1.214 ± 0.027 | - | 1.226 ± 0.012 | 0.869 ± 0.237 | 0.755 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 4.502 ± 0.029 | - | 4.546 ± 0.029 | 2.656 ± 0.029 | 1.217 ± 0.193 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 9.601 ± 3.384 | - | 9.704 ± 0.334 | 10.761 ± 0.046 | 4.023 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 1.543 ± 0.035 | - | 1.591 ± 0.011 | 1.769 ± 0.043 | 1.627 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.016 ± 0.011 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.113 ± 0.003 | - | 0.137 ± 0.007 | 0.127 ± 0.003 | 0.099 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | 0.018 ± 0.002 | 0.007 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.089 ± 0.003 | - | 0.113 ± 0.010 | 0.100 ± 0.004 | 0.109 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.014 ± 0.004 | - | 0.033 ± 0.005 | 0.013 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.008 ± 0.003 | - | 0.018 ± 0.003 | 0.009 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.015 ± 0.000 | - | 0.029 ± 0.006 | 0.012 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.037 ± 0.001 | - | 0.055 ± 0.009 | 0.028 ± 0.003 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.007 ± 0.003 | - | 0.021 ± 0.007 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.020 ± 0.000 | - | 0.033 ± 0.006 | 0.015 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.049 ± 0.001 | - | 0.067 ± 0.004 | 0.036 ± 0.001 | 0.037 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 12.200 ± 0.407 | - | 12.402 ± 0.061 | 13.239 ± 0.336 | 13.307 ± 0.925 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.011 ± 0.000 | - | 0.027 ± 0.015 | 0.013 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 0.873 ± 0.009 | - | 0.914 ± 0.014 | 0.959 ± 0.005 | 0.336 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.037 ± 0.000 | - | 0.049 ± 0.003 | 0.025 ± 0.000 | 0.027 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.099 ± 0.001 | - | 0.122 ± 0.018 | 0.069 ± 0.000 | 0.073 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.248 ± 0.005 | - | unsupported | 0.175 ± 0.002 | 0.393 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.263 ± 0.006 | - | unsupported | 1.128 ± 0.004 | 1.487 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 3.072 ± 0.005 | - | unsupported | 3.054 ± 0.025 | 3.818 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.568 ± 0.013 | - | unsupported | 0.688 ± 0.007 | 0.656 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.001 ± 0.002 | - | unsupported | 0.004 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.028 ± 0.002 | - | unsupported | 0.034 ± 0.003 | 0.039 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.023 ± 0.000 | - | unsupported | 0.030 ± 0.001 | 0.034 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.009 ± 0.000 | - | unsupported | 0.013 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.013 ± 0.000 | - | unsupported | 0.017 ± 0.002 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 4.490 ± 0.007 | - | unsupported | 5.500 ± 0.122 | 5.325 ± 0.065 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.003 ± 0.001 | - | unsupported | 0.005 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.220 ± 0.006 | - | unsupported | 0.243 ± 0.003 | 0.281 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.011 ± 0.000 | - | unsupported | 0.012 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.024 ± 0.000 | - | unsupported | 0.027 ± 0.000 | 0.034 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.420 ± 0.016 | - | unsupported | 0.290 ± 0.006 | 0.376 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 1.929 ± 0.004 | - | unsupported | 1.171 ± 0.012 | 0.884 ± 0.137 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 4.580 ± 0.017 | - | unsupported | 4.272 ± 0.081 | 2.049 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 0.691 ± 0.018 | - | unsupported | 0.824 ± 0.030 | 0.781 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.002 ± 0.003 | - | unsupported | 0.006 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.037 ± 0.006 | - | unsupported | 0.046 ± 0.002 | 0.039 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.031 ± 0.000 | - | unsupported | 0.039 ± 0.001 | 0.047 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.006 ± 0.000 | - | unsupported | 0.007 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.007 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.014 ± 0.000 | - | unsupported | 0.013 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.009 ± 0.000 | - | unsupported | 0.012 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.019 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 5.442 ± 0.032 | - | unsupported | 6.445 ± 0.167 | 6.767 ± 1.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.005 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.289 ± 0.008 | - | unsupported | 0.319 ± 0.058 | 0.174 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.016 ± 0.000 | - | unsupported | 0.018 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.036 ± 0.000 | - | unsupported | 0.028 ± 0.000 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.622 ± 0.011 | - | unsupported | 0.666 ± 0.010 | 1.603 ± 0.065 | 1.508 ± 0.037 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.120 ± 0.001 | - | unsupported | 0.031 ± 0.001 | 0.076 ± 0.001 | 0.071 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.079 ± 0.001 | - | unsupported | 0.045 ± 0.001 | 0.104 ± 0.002 | 0.116 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.175 ± 0.002 | - | unsupported | 0.147 ± 0.010 | 0.269 ± 0.017 | 0.431 ± 0.254 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.650 ± 0.015 | - | unsupported | 0.300 ± 0.010 | 0.671 ± 0.058 | 1.474 ± 0.613 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.120 ± 0.003 | - | unsupported | 0.032 ± 0.001 | 0.065 ± 0.009 | 0.069 ± 0.008 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.079 ± 0.001 | - | unsupported | 0.042 ± 0.001 | 0.113 ± 0.006 | 0.113 ± 0.007 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.177 ± 0.011 | - | unsupported | 0.083 ± 0.001 | 0.363 ± 0.050 | 0.427 ± 0.019 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.176 ± 0.001 | - | unsupported | 0.232 ± 0.007 | 0.461 ± 0.010 | 0.235 ± 0.006 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.017 ± 0.000 | - | unsupported | 0.022 ± 0.002 | 0.113 ± 0.001 | 0.049 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.036 ± 0.000 | - | unsupported | 0.028 ± 0.000 | 0.135 ± 0.002 | 0.072 ± 0.077 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.117 ± 0.002 | - | unsupported | 0.104 ± 0.000 | 0.213 ± 0.001 | 0.118 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.174 ± 0.010 | - | unsupported | 0.291 ± 0.009 | 0.540 ± 0.041 | 0.228 ± 0.007 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.017 ± 0.000 | - | unsupported | 0.036 ± 0.001 | 0.118 ± 0.005 | 0.049 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.036 ± 0.000 | - | unsupported | 0.048 ± 0.002 | 0.147 ± 0.005 | 0.069 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.116 ± 0.001 | - | unsupported | 0.129 ± 0.001 | 0.235 ± 0.009 | 0.118 ± 0.004 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.111 ± 0.002 | - | 0.161 ± 0.001 | 0.123 ± 0.002 | 0.210 ± 0.008 | 0.144 ± 0.016 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.041 ± 0.000 | - | 0.045 ± 0.000 | 0.040 ± 0.002 | 0.048 ± 0.000 | 0.066 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.050 ± 0.000 | - | 0.054 ± 0.000 | 0.049 ± 0.001 | 0.059 ± 0.000 | 0.079 ± 0.006 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.068 ± 0.000 | - | 0.083 ± 0.002 | 0.074 ± 0.001 | 0.099 ± 0.001 | 0.077 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.116 ± 0.002 | - | 0.184 ± 0.009 | 0.147 ± 0.002 | 0.258 ± 0.015 | 0.140 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.041 ± 0.000 | - | 0.082 ± 0.007 | 0.061 ± 0.002 | 0.051 ± 0.001 | 0.041 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.047 ± 0.000 | - | 0.090 ± 0.012 | 0.073 ± 0.001 | 0.064 ± 0.001 | 0.049 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.067 ± 0.001 | - | 0.112 ± 0.008 | 0.095 ± 0.002 | 0.115 ± 0.016 | 0.075 ± 0.001 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 28.685 ± 0.178 | - | 30.146 ± 0.737 | 46.560 ± 1.646 | 40.573 ± 0.385 | 25.625 ± 3.509 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 13.929 ± 0.349 | - | 15.221 ± 0.240 | 18.760 ± 0.140 | 33.180 ± 1.566 | 19.043 ± 8.761 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 17.543 ± 0.921 | - | 16.692 ± 0.360 | 16.546 ± 0.255 | 12.191 ± 0.066 | 10.323 ± 0.040 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 9.376 ± 0.066 | - | 8.487 ± 0.129 | 7.592 ± 0.211 | 7.192 ± 0.271 | 5.956 ± 0.458 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 4.521 ± 0.016 | - | 4.537 ± 0.039 | 5.026 ± 0.214 | 4.284 ± 0.032 | 5.238 ± 0.117 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 6.282 ± 0.094 | - | 6.365 ± 0.069 | 5.956 ± 0.077 | 6.320 ± 0.205 | 5.503 ± 0.215 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 5.276 ± 0.011 | - | 5.286 ± 0.041 | 5.449 ± 0.179 | 5.304 ± 0.025 | 6.967 ± 0.079 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 7.421 ± 0.120 | - | 7.518 ± 0.062 | 6.596 ± 0.075 | 7.849 ± 0.296 | 7.085 ± 0.069 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 10.814 ± 0.110 | - | 10.882 ± 0.101 | 11.546 ± 0.399 | 20.918 ± 0.582 | 7.821 ± 0.084 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 9.366 ± 0.227 | - | 9.590 ± 0.184 | 9.277 ± 0.035 | 20.111 ± 0.269 | 6.891 ± 0.230 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.298 ± 0.009 | 0.305 ± 0.002 | unsupported | 0.386 ± 0.019 | 0.417 ± 0.009 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 4.969 ± 0.613 | 4.927 ± 0.111 | unsupported | 5.602 ± 0.450 | 4.637 ± 0.032 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.213 ± 0.011 | 0.233 ± 0.001 | unsupported | 0.305 ± 0.048 | 0.667 ± 0.323 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 3.099 ± 0.076 | 3.213 ± 0.055 | unsupported | 3.533 ± 0.074 | 4.633 ± 0.187 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 25.731 ± 0.865 | - | 24.898 ± 0.134 | 29.425 ± 0.413 | 20.071 ± 0.112 | 14.157 ± 2.639 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 13.921 ± 0.450 | - | 13.687 ± 0.084 | 14.356 ± 0.099 | 12.013 ± 0.367 | 8.036 ± 0.132 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 10.956 ± 0.156 | 10.696 ± 0.110 | 5.734 ± 0.021 | 21.284 ± 0.379 | 3.642 ± 0.049 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 7.172 ± 0.239 | 7.067 ± 0.219 | 3.900 ± 0.196 | 23.023 ± 0.246 | 3.042 ± 0.577 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 17.361 ± 1.137 | - | 17.243 ± 0.216 | 26.763 ± 1.445 | 25.139 ± 0.306 | 16.896 ± 13.300 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 8.910 ± 0.342 | - | 9.425 ± 0.143 | 11.849 ± 0.168 | 18.177 ± 2.206 | 12.685 ± 13.626 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.781 ± 0.049 | - | 0.724 ± 0.070 | 0.760 ± 0.023 | 3.302 ± 0.048 | 1.970 ± 0.002 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.469 ± 0.038 | - | 0.503 ± 0.039 | 0.781 ± 0.031 | 3.211 ± 0.249 | 1.901 ± 0.008 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 12.078 ± 0.299 | - | 12.313 ± 0.073 | 13.588 ± 0.482 | 11.008 ± 0.075 | 11.491 ± 1.567 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 10.451 ± 0.174 | - | 10.091 ± 0.069 | 10.909 ± 0.119 | 11.744 ± 0.381 | 10.034 ± 0.603 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 12.335 ± 0.383 | - | 12.334 ± 0.061 | 13.594 ± 0.404 | 10.578 ± 0.418 | 10.078 ± 0.224 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 10.301 ± 0.152 | - | 10.232 ± 0.215 | 11.000 ± 0.104 | 11.738 ± 0.348 | 9.996 ± 0.297 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 0.695 ± 0.006 | 0.708 ± 0.005 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 13.832 ± 0.442 | 13.985 ± 0.409 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 0.773 ± 0.008 | 0.810 ± 0.018 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 12.234 ± 0.191 | 12.476 ± 0.194 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 0.653 ± 0.001 | 0.660 ± 0.004 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 13.588 ± 0.349 | 13.471 ± 0.243 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 0.722 ± 0.011 | 0.756 ± 0.010 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 11.664 ± 0.140 | 11.911 ± 0.056 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 16.912 ± 0.229 | - | 16.111 ± 0.202 | 16.465 ± 0.410 | 12.165 ± 0.189 | 10.467 ± 0.830 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 9.368 ± 0.217 | - | 8.408 ± 0.349 | 7.581 ± 0.115 | 7.024 ± 0.054 | 5.993 ± 1.621 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 36.817 ± 0.641 | 36.239 ± 0.444 | 38.630 ± 1.781 | 30.340 ± 0.559 | 30.860 ± 4.558 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 23.772 ± 0.158 | 23.856 ± 0.177 | 24.911 ± 0.299 | 29.438 ± 1.876 | 26.237 ± 0.806 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 32.352 ± 0.853 | - | 43.045 ± 1.100 | 21.221 ± 0.252 | 64.260 ± 0.875 | 14.751 ± 0.348 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 22.491 ± 2.315 | - | 30.875 ± 1.554 | 16.068 ± 0.312 | 49.832 ± 1.190 | 10.596 ± 1.038 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.040 ± 0.001 | 0.041 ± 0.000 | unsupported | 0.101 ± 0.020 | 0.027 ± 0.002 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 0.664 ± 0.009 | 0.646 ± 0.002 | unsupported | 1.292 ± 0.016 | 0.350 ± 0.232 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.040 ± 0.005 | 0.043 ± 0.004 | unsupported | 0.047 ± 0.121 | 0.026 ± 0.002 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.301 ± 0.147 | 0.302 ± 0.014 | unsupported | 0.611 ± 0.006 | 0.547 ± 0.063 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.185 ± 0.001 | 0.197 ± 0.003 | unsupported | 0.063 ± 0.008 | 0.137 ± 0.007 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 2.948 ± 0.043 | 2.955 ± 0.017 | unsupported | 0.561 ± 0.040 | 1.072 ± 0.018 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.214 ± 0.103 | 0.244 ± 0.042 | unsupported | 0.053 ± 0.002 | 0.093 ± 0.002 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 1.551 ± 0.016 | 1.541 ± 0.020 | unsupported | 0.307 ± 0.053 | 0.591 ± 0.006 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.576 ± 0.010 | 0.578 ± 0.002 | unsupported | 0.112 ± 0.001 | 0.110 ± 0.001 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 9.209 ± 0.013 | 9.431 ± 0.397 | unsupported | 1.476 ± 0.012 | 0.646 ± 0.005 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.448 ± 0.049 | 0.504 ± 0.082 | unsupported | 0.047 ± 0.001 | 0.084 ± 0.004 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 3.993 ± 0.065 | 3.987 ± 0.017 | unsupported | 0.580 ± 0.012 | 0.431 ± 0.007 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.028 ± 0.003 | 0.047 ± 0.001 | unsupported | 0.099 ± 0.001 | 0.114 ± 0.009 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.367 ± 0.021 | 0.381 ± 0.016 | unsupported | 0.910 ± 0.016 | 0.691 ± 0.012 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.097 ± 0.009 | 0.146 ± 0.012 | unsupported | 0.045 ± 0.002 | 0.109 ± 0.006 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.323 ± 0.036 | 0.335 ± 0.068 | unsupported | 0.247 ± 0.007 | 0.565 ± 0.095 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.043 ± 0.002 | 0.053 ± 0.001 | unsupported | 0.062 ± 0.001 | 0.165 ± 0.004 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 0.577 ± 0.024 | 0.594 ± 0.040 | unsupported | 0.674 ± 0.014 | 1.369 ± 0.012 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.108 ± 0.003 | 0.124 ± 0.006 | unsupported | 0.033 ± 0.002 | 0.155 ± 0.003 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.461 ± 0.171 | 0.430 ± 0.079 | unsupported | 0.191 ± 0.013 | 1.015 ± 0.076 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.065 ± 0.001 | 0.084 ± 0.002 | unsupported | unsupported | 0.179 ± 0.007 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 0.975 ± 0.023 | 1.319 ± 0.899 | unsupported | unsupported | 1.748 ± 0.030 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.131 ± 0.010 | 0.165 ± 0.010 | unsupported | unsupported | 0.179 ± 0.006 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 0.614 ± 0.007 | 0.670 ± 0.016 | unsupported | unsupported | 1.303 ± 0.146 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.006 ± 0.000 | 0.010 ± 0.000 | unsupported | 0.007 ± 0.000 | 0.029 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.052 ± 0.008 | 0.060 ± 0.010 | unsupported | 0.040 ± 0.004 | 0.176 ± 0.003 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.020 ± 0.003 | 0.028 ± 0.005 | unsupported | 0.008 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.065 ± 0.008 | 0.081 ± 0.018 | unsupported | 0.038 ± 0.008 | 0.150 ± 0.032 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.015 ± 0.003 | 0.027 ± 0.002 | unsupported | 0.057 ± 0.001 | 0.122 ± 0.009 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.194 ± 0.005 | 0.208 ± 0.007 | unsupported | 0.587 ± 0.027 | 0.665 ± 0.008 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.059 ± 0.002 | 0.075 ± 0.007 | unsupported | 0.056 ± 0.003 | 0.101 ± 0.001 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.198 ± 0.045 | 0.226 ± 0.021 | unsupported | 0.210 ± 0.011 | 0.494 ± 0.018 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.147 ± 0.004 | 0.147 ± 0.001 | unsupported | 0.053 ± 0.001 | 0.112 ± 0.005 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 2.419 ± 0.042 | 2.373 ± 0.064 | unsupported | 0.523 ± 0.008 | 0.482 ± 0.012 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.179 ± 0.052 | 0.237 ± 0.068 | unsupported | 0.042 ± 0.001 | 0.081 ± 0.009 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 1.194 ± 0.002 | 1.206 ± 0.017 | unsupported | 0.311 ± 0.038 | 0.358 ± 0.005 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.131 ± 0.003 | 0.131 ± 0.002 | unsupported | 0.053 ± 0.001 | 0.112 ± 0.007 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 2.103 ± 0.025 | 2.112 ± 0.009 | unsupported | 0.548 ± 0.012 | 0.517 ± 0.010 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.167 ± 0.015 | 0.179 ± 0.079 | unsupported | 0.052 ± 0.002 | 0.082 ± 0.001 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 1.114 ± 0.038 | 1.105 ± 0.026 | unsupported | 0.285 ± 0.005 | 0.379 ± 0.006 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.043 ± 0.003 | 0.055 ± 0.000 | unsupported | 0.072 ± 0.000 | 0.155 ± 0.004 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 0.547 ± 0.016 | 0.576 ± 0.006 | unsupported | 0.448 ± 0.007 | 1.350 ± 0.019 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.090 ± 0.012 | 0.123 ± 0.013 | unsupported | 0.034 ± 0.002 | 0.150 ± 0.004 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 0.385 ± 0.027 | 0.423 ± 0.013 | unsupported | 0.196 ± 0.024 | 0.968 ± 0.052 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.596 ± 0.024 | 0.596 ± 0.007 | unsupported | 0.103 ± 0.001 | 0.199 ± 0.001 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 9.558 ± 0.840 | 9.535 ± 0.030 | unsupported | 1.353 ± 0.008 | 2.231 ± 0.023 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.436 ± 0.045 | 0.452 ± 0.061 | unsupported | 0.089 ± 0.001 | 0.132 ± 0.012 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 4.105 ± 0.026 | 4.147 ± 0.119 | unsupported | 0.566 ± 0.005 | 1.237 ± 0.048 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.399 ± 0.012 | 0.530 ± 0.003 | unsupported | 0.168 ± 0.001 | 0.105 ± 0.002 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 6.861 ± 0.317 | 9.462 ± 0.398 | unsupported | 3.101 ± 0.065 | 0.531 ± 0.030 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.466 ± 0.026 | 0.717 ± 0.020 | unsupported | 0.185 ± 0.005 | 0.085 ± 0.006 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 5.210 ± 0.049 | 8.135 ± 0.158 | unsupported | 1.387 ± 0.036 | 0.387 ± 0.011 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 18.196 ± 0.100 | - | unsupported | 18.316 ± 0.040 | - | 18.375 ± 1.037 | 31.069 ± 1.367 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 18.653 ± 1.719 | - | unsupported | 16.973 ± 0.597 | - | 18.004 ± 0.103 | 18.562 ± 0.975 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 12.612 ± 0.065 | - | unsupported | 12.798 ± 0.150 | - | 13.129 ± 0.740 | 13.293 ± 0.198 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 12.969 ± 0.166 | - | unsupported | 12.435 ± 0.212 | - | 12.601 ± 0.171 | 11.894 ± 0.200 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 11.593 ± 0.133 | - | unsupported | 12.910 ± 0.278 | - | 11.905 ± 0.227 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 12.208 ± 0.276 | - | unsupported | 12.466 ± 0.379 | - | 11.677 ± 0.125 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 18.232 ± 0.067 | - | unsupported | 18.581 ± 0.137 | - | 18.587 ± 0.436 | 34.125 ± 0.319 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 17.897 ± 0.878 | - | unsupported | 17.537 ± 0.554 | - | 18.274 ± 0.245 | 18.271 ± 0.581 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 35.869 ± 0.850 | - | unsupported | 40.110 ± 1.500 | - | 20.372 ± 0.424 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 19.441 ± 0.576 | - | unsupported | 21.436 ± 0.142 | - | 9.978 ± 0.105 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 35.103 ± 0.297 | - | unsupported | 39.436 ± 0.630 | - | 20.207 ± 0.071 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 19.139 ± 0.221 | - | unsupported | 21.244 ± 0.206 | - | 9.847 ± 0.067 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 18.440 ± 0.487 | - | unsupported | 18.484 ± 0.046 | - | 18.648 ± 0.512 | 32.829 ± 0.967 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 17.709 ± 1.006 | - | unsupported | 17.452 ± 0.378 | - | 17.980 ± 0.227 | 20.166 ± 2.357 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 12.669 ± 0.086 | - | unsupported | 12.844 ± 0.075 | - | 13.000 ± 0.420 | 25.789 ± 0.562 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 12.185 ± 0.301 | - | unsupported | 12.090 ± 0.277 | - | 12.226 ± 0.064 | 12.380 ± 1.067 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.415 ± 0.006 | unsupported | unsupported | 0.522 ± 0.006 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 6.733 ± 0.341 | unsupported | unsupported | 8.668 ± 0.424 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.300 ± 0.003 | unsupported | unsupported | 0.352 ± 0.003 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 3.792 ± 0.024 | unsupported | unsupported | 4.138 ± 0.070 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 18.216 ± 0.152 | - | unsupported | 18.433 ± 0.070 | - | 18.987 ± 0.432 | 31.550 ± 0.727 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 17.905 ± 0.855 | - | unsupported | 16.919 ± 0.333 | - | 20.722 ± 0.054 | 18.553 ± 0.663 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 15.572 ± 0.482 | - | 15.428 ± 0.134 | 14.275 ± 0.273 | 14.458 ± 0.141 | 17.807 ± 0.544 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 12.310 ± 0.479 | - | 11.996 ± 0.212 | 9.986 ± 0.286 | 10.521 ± 0.086 | 17.761 ± 0.682 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 13.548 ± 0.191 | - | 41.132 ± 0.547 | 13.376 ± 0.168 | 14.162 ± 0.459 | 13.700 ± 0.414 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 10.163 ± 0.177 | - | 26.107 ± 0.718 | 9.873 ± 0.210 | 10.369 ± 0.290 | 13.106 ± 0.702 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 64.004 ± 0.702 | - | 63.900 ± 0.192 | 25.689 ± 0.239 | 26.312 ± 0.259 | 28.578 ± 0.241 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 18.389 ± 0.586 | - | 17.443 ± 1.609 | 16.426 ± 0.530 | 17.750 ± 0.192 | 28.330 ± 0.675 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 11.551 ± 0.100 | - | 112.262 ± 1.303 | 16.087 ± 0.312 | 12.690 ± 0.254 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 8.435 ± 0.352 | - | 23.898 ± 0.432 | 9.955 ± 0.071 | 9.780 ± 0.184 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 22.241 ± 0.350 | - | 19.247 ± 0.336 | 19.299 ± 0.227 | 20.066 ± 0.422 | 23.315 ± 1.062 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 13.903 ± 0.326 | - | 14.248 ± 0.517 | 13.601 ± 0.254 | 14.727 ± 1.121 | 22.801 ± 0.503 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 55.984 ± 0.552 | - | 55.457 ± 0.168 | 16.609 ± 0.275 | 34.007 ± 1.229 | 279.778 ± 9.796 | 87.926 ± 0.665 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 28.963 ± 2.465 | - | 28.250 ± 1.664 | 17.557 ± 0.375 | 17.551 ± 1.085 | 282.202 ± 9.259 | 25.964 ± 3.618 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 10.031 ± 0.192 | - | 28.232 ± 0.706 | 11.566 ± 0.677 | 13.000 ± 0.446 | 14.838 ± 0.202 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 6.077 ± 0.421 | - | 14.797 ± 0.487 | 6.795 ± 0.056 | 8.270 ± 0.252 | 14.810 ± 1.256 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 10.104 ± 0.256 | - | 28.052 ± 0.285 | 13.032 ± 0.060 | 12.696 ± 0.085 | 15.462 ± 1.013 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 6.172 ± 0.592 | - | 14.489 ± 0.917 | 6.418 ± 0.094 | 8.128 ± 0.104 | 14.381 ± 0.939 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.001 ± 0.000 | - | unsupported | 0.001 ± 0.001 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.001 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 3.013325 | 188332.81 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 1.762321 | 110145.06 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 1.932005 | 120750.31 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 2.751264 | 171954.00 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 2.975803 | 185987.69 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 3.605137 | 225321.06 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 12.382416 | 773901.00 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 11.793383 | 737086.44 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 4.038322 | 252395.12 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 4.024863 | 251553.94 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 6.678246 | 417390.38 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 6.177845 | 386115.31 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 4.767624 | 297976.50 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 4.886362 | 305397.62 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 74.198689 | 4637418.06 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 89.637267 | 5602329.19 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 79.496200 | 4968512.50 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 78.831323 | 4926957.69 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 10.679153 | 667447.06 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 4.884422 | 305276.38 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 3.404571 | 212785.69 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 3.732107 | 233256.69 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 74.133450 | 4633340.62 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 56.529724 | 3533107.75 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 49.578125 | 3098632.81 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 51.408369 | 3213023.06 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 11.112141 | 694508.81 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 11.324690 | 707793.12 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 221.311259 | 13831953.69 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 223.755632 | 13984727.00 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 12.366756 | 772922.25 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 12.965824 | 810364.00 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 195.743425 | 12233964.06 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 199.622329 | 12476395.56 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 10.452983 | 653311.44 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 10.562968 | 660185.50 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 217.414891 | 13588430.69 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 215.533792 | 13470862.00 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 11.550940 | 721933.75 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 12.091038 | 755689.88 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 186.626369 | 11664148.06 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 190.569565 | 11910597.81 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 1.756261 | 27441.58 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.441260 | 100644.69 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.576430 | 40256.72 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.599305 | 40614.14 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 2.801041 | 350130.12 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.332329 | 1291541.12 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.313401 | 664175.12 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.165309 | 645663.62 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 1.639290 | 25613.91 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.035663 | 47432.23 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.565145 | 40080.39 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.754047 | 43031.98 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.376723 | 547090.38 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.889209 | 611151.12 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.405693 | 300711.62 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.418704 | 302338.00 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.787821 | 137309.70 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.053993 | 63343.64 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 11.831948 | 184874.19 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 12.597164 | 196830.69 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 8.579413 | 1072426.62 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.488775 | 561096.88 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 23.586005 | 2948250.62 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 23.636851 | 2954606.38 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.971321 | 93301.89 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.408719 | 53261.23 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 13.725021 | 214453.45 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 15.592515 | 243633.05 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.728950 | 591118.75 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.455811 | 306976.38 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 12.405008 | 1550626.00 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 12.330345 | 1541293.12 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.019652 | 109682.06 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.187758 | 112308.72 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 36.873637 | 576150.58 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 36.963652 | 577557.06 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.168646 | 646080.75 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.807611 | 1475951.38 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 73.668271 | 9208533.88 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 75.448057 | 9431007.12 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.360662 | 83760.34 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.014730 | 47105.16 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 28.644011 | 447562.67 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 32.255932 | 503998.94 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.444312 | 430539.00 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.640551 | 580068.88 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 31.943398 | 3992924.75 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 31.892461 | 3986557.62 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.301188 | 114081.06 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.357093 | 99329.58 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.808299 | 28254.67 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.993305 | 46770.39 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.525388 | 690673.50 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.280958 | 910119.75 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.938109 | 367263.62 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.048074 | 381009.25 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.987999 | 109187.48 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.861378 | 44709.03 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.227777 | 97309.02 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 9.364096 | 146314.00 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.517831 | 564728.88 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.977785 | 247223.12 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.586542 | 323317.75 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.682643 | 335330.38 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.570455 | 165163.36 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.980293 | 62192.08 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.778573 | 43415.20 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.413705 | 53339.14 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.950146 | 1368768.25 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 5.390406 | 673800.75 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.617443 | 577180.38 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.754518 | 594314.75 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.909535 | 154836.48 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.083435 | 32553.67 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.924757 | 108199.33 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.935423 | 123990.98 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.123322 | 1015415.25 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.530465 | 191308.12 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.688341 | 461042.62 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.441292 | 430161.50 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.456518 | 179008.09 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.152101 | 64876.58 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.351247 | 83613.23 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.982259 | 1747782.38 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.803424 | 975428.00 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.553512 | 1319189.00 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.480582 | 179384.09 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.412965 | 131452.58 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 10.579894 | 165310.84 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.420859 | 1302607.38 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.910307 | 613788.38 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.359547 | 669943.38 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 1.838733 | 28730.20 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.441452 | 6897.69 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.373339 | 5833.42 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 0.610364 | 9536.94 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 1.410322 | 176290.25 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.323554 | 40444.25 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.416614 | 52076.75 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.481121 | 60140.12 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 1.434361 | 22411.89 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.502223 | 7847.23 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.270651 | 19853.92 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.809486 | 28273.22 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.199455 | 149931.88 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.304744 | 38093.00 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.517384 | 64673.00 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 0.644205 | 80525.62 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.827337 | 122302.14 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.674966 | 57421.34 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.982562 | 15352.53 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.716763 | 26824.42 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.316846 | 664605.75 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.698269 | 587283.62 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 1.551283 | 193910.38 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 1.666375 | 208296.88 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.465073 | 101016.77 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.579029 | 55922.33 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.790792 | 59231.12 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.770327 | 74536.36 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.953142 | 494142.75 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.678100 | 209762.50 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.587974 | 198496.75 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.808654 | 226081.75 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.173648 | 112088.25 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.383008 | 52859.50 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 9.405051 | 146953.92 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.403900 | 146935.94 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 3.853857 | 481732.12 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.185681 | 523210.12 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 19.355941 | 2419492.62 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 18.982872 | 2372859.00 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.154930 | 80545.78 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.687674 | 41994.91 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 11.462686 | 179104.47 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 15.178355 | 237161.80 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.863693 | 357961.62 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.491766 | 311470.75 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.548341 | 1193542.62 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 9.646139 | 1205767.38 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.194379 | 112412.17 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.419533 | 53430.20 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.356440 | 130569.37 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.357892 | 130592.06 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 4.133771 | 516721.38 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.385320 | 548165.00 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 16.826621 | 2103327.62 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 16.895184 | 2111898.00 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.237035 | 81828.67 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.354317 | 52411.20 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 10.675159 | 166799.36 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 11.439380 | 178740.31 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.028583 | 378572.88 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.278873 | 284859.12 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.912874 | 1114109.25 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 8.837124 | 1104640.50 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.947105 | 155423.52 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.633141 | 72392.83 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.740696 | 42823.38 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.493324 | 54583.19 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.798910 | 1349863.75 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 3.587545 | 448443.12 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.379948 | 547493.50 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.607649 | 575956.12 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.582615 | 149728.36 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.204780 | 34449.69 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.748934 | 89827.09 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.870004 | 122968.81 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.743261 | 967907.62 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.566270 | 195783.75 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.081607 | 385200.88 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.383354 | 422919.25 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.748597 | 199196.83 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.563102 | 102548.47 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 38.122689 | 595667.02 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 38.151101 | 596110.95 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 17.846739 | 2230842.38 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.822654 | 1352831.75 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 76.461151 | 9557643.88 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 76.282122 | 9535265.25 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.479771 | 132496.42 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.689656 | 88900.88 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 27.882132 | 435658.31 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 28.946976 | 452296.50 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.892259 | 1236532.38 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.526768 | 565846.00 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 32.837522 | 4104690.25 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 33.178013 | 4147251.62 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.706359 | 104786.86 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 10.763305 | 168176.64 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 25.553887 | 399279.48 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 33.900312 | 529692.38 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 4.247774 | 530971.75 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 24.810251 | 3101281.38 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 54.886129 | 6860766.12 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 75.693915 | 9461739.38 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.452911 | 85201.73 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 11.845639 | 185088.11 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 29.807069 | 465735.45 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 45.871753 | 716746.14 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.097397 | 387174.62 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 11.098994 | 1387374.25 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 41.677731 | 5209716.38 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 65.080364 | 8135045.50 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 8.353444 | 522090.25 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 6.638925 | 414932.81 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 138.680580 | 8667536.25 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 107.729561 | 6733097.56 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 5.637997 | 352374.81 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 4.796110 | 299756.88 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 66.204054 | 4137753.38 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 60.679857 | 3792491.06 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.013029 | 814.31 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.001292 | 80.75 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.020481 | 1280.06 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.001402 | 87.62 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000090 | 5.62 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.012559 | 784.94 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.008172 | 510.75 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000090 | 5.62 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.018649 | 1165.56 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.008843 | 552.69 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000231 | 14.44 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.012559 | 784.94 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.004897 | 306.06 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000230 | 14.38 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.020150 | 1259.38 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.005579 | 348.69 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000070 | 4.38 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.011818 | 738.62 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003695 | 230.94 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000060 | 3.75 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.018048 | 1128.00 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.004007 | 250.44 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 11.9x faster than the slowest successful cell (`tenferro-trace`, 9.473 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/pow` (f64, threads=4, shape=`4194304`): `pytorch-cpu` is 12.5x faster than the slowest successful cell (`julia-base`, 84.715 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 21.9x faster than the slowest successful cell (`pytorch-cpu`, 28.264 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 23.3x faster than the slowest successful cell (`pytorch-cpu`, 28.264 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 21.0x faster than the slowest successful cell (`pytorch-cpu`, 28.264 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 20.9x faster than the slowest successful cell (`pytorch-cpu`, 28.264 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 14.0x faster than the slowest successful cell (`pytorch-cpu`, 10.431 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 15.3x faster than the slowest successful cell (`pytorch-cpu`, 10.431 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 17.0x faster than the slowest successful cell (`pytorch-cpu`, 10.431 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 19.3x faster than the slowest successful cell (`pytorch-cpu`, 23.491 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 20.1x faster than the slowest successful cell (`pytorch-cpu`, 23.491 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 19.2x faster than the slowest successful cell (`pytorch-cpu`, 23.491 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 18.9x faster than the slowest successful cell (`pytorch-cpu`, 23.491 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 10.308 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 18.0x faster than the slowest successful cell (`pytorch-cpu`, 10.308 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 17.7x faster than the slowest successful cell (`pytorch-cpu`, 10.308 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sin` (f64, threads=1, shape=`8388608`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`jax-cpu`, 63.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sin` (f64, threads=1, shape=`8388608`): `tenferro-trace` is 10.2x faster than the slowest successful cell (`jax-cpu`, 63.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sin` (f64, threads=4, shape=`8388608`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`jax-cpu`, 30.662 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 11.9x faster than the slowest successful cell (`jax-cpu`, 0.026 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 17.3x faster than the slowest successful cell (`jax-cpu`, 0.019 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 12.5x faster than the slowest successful cell (`jax-cpu`, 0.020 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 22.5x faster than the slowest successful cell (`tenferro-trace`, 0.018 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 19.3x faster than the slowest successful cell (`tenferro-trace`, 0.017 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 16.6x faster than the slowest successful cell (`tenferro-trace`, 0.020 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 20.8x faster than the slowest successful cell (`tenferro-trace`, 0.017 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 17.4x faster than the slowest successful cell (`tenferro-trace`, 0.018 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`tenferro-trace`, 0.010 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 12.0x faster than the slowest successful cell (`tenferro-trace`, 0.017 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 13.1x faster than the slowest successful cell (`tenferro-trace`, 0.016 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=4, shape=`1x2x2`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`tenferro-trace`, 0.020 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 13.1x faster than the slowest successful cell (`jax-cpu`, 0.030 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.024 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 14.1x faster than the slowest successful cell (`tenferro-trace`, 0.017 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 11.6x faster than the slowest successful cell (`tenferro-trace`, 0.017 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x2x2,rhs=1`): `tenferro-direct` is 27.1x faster than the slowest successful cell (`tenferro-trace`, 0.030 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x4x4,rhs=1`): `tenferro-direct` is 14.5x faster than the slowest successful cell (`tenferro-trace`, 0.022 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x8x8,rhs=1`): `tenferro-direct` is 17.5x faster than the slowest successful cell (`tenferro-trace`, 0.033 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`3x2x2,rhs=1`): `tenferro-direct` is 16.7x faster than the slowest successful cell (`tenferro-trace`, 0.022 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`3x4x4,rhs=1`): `tenferro-direct` is 13.7x faster than the slowest successful cell (`tenferro-trace`, 0.025 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`4x2x2,rhs=1`): `tenferro-direct` is 14.7x faster than the slowest successful cell (`tenferro-trace`, 0.030 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`4x4x4,rhs=1`): `tenferro-direct` is 15.8x faster than the slowest successful cell (`tenferro-trace`, 0.026 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x2x2,rhs=1`): `tenferro-direct` is 10.8x faster than the slowest successful cell (`tenferro-trace`, 0.024 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x4x4,rhs=1`): `tenferro-direct` is 12.4x faster than the slowest successful cell (`tenferro-trace`, 0.033 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 14.6x faster than the slowest successful cell (`tenferro-eager`, 9.431 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 10.7x faster than the slowest successful cell (`tenferro-eager`, 0.504 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 17.8x faster than the slowest successful cell (`tenferro-eager`, 9.462 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 21.0x faster than the slowest successful cell (`tenferro-eager`, 8.135 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `pytorch-cpu` is 16.8x faster than the slowest successful cell (`julia-base`, 279.778 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `jax-cpu` is 16.1x faster than the slowest successful cell (`julia-base`, 282.202 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `pytorch-cpu` is 16.1x faster than the slowest successful cell (`julia-base`, 282.202 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `strided-jl` is 10.9x faster than the slowest successful cell (`julia-base`, 282.202 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 10.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 130.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 194.3x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 56.1x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 89.9x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 184.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 282.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

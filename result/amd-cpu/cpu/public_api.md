# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`
- Run metadata: `data/results/amd-cpu/cpu/public_api/20261023_183814/run.yaml`
- Timestamp: `20261023_183814`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Sampling: `quick` profile, 7 measured runs, 3 warmups per row

This file is generated from sequential CPU public API runs under `data/results/amd-cpu/cpu/public_api/20261023_183814`.

- tenferro-rs commit: `a24fccea2dc7cee09df442104201b7c52f9b943b`

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

- Run metadata: `data/results/amd-cpu/cpu/public_api/20261023_183814/run_t1.yaml`
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

- Run metadata: `data/results/amd-cpu/cpu/public_api/20261023_183814/run_t4.yaml`
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

- CSV: `data/results/amd-cpu/cpu/public_api/20261023_183814/cpu_public_api_t1_20261023_183814.csv`
- CSV: `data/results/amd-cpu/cpu/public_api/20261023_183814/cpu_public_api_t4_20261023_183814.csv`
- Source table: `data/results/amd-cpu/cpu/public_api/20261023_183814/cpu_public_api_20261023_183814.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 3.021 ± 0.012 | - | 3.022 ± 0.025 | 3.412 ± 0.184 | 1.976 ± 0.009 | 1.887 ± 0.924 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 1.407 ± 0.008 | - | 1.457 ± 0.017 | 1.406 ± 0.012 | 1.466 ± 0.060 | 1.331 ± 0.855 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 19.570 ± 0.248 | - | 18.950 ± 0.200 | 19.437 ± 0.246 | 18.550 ± 0.101 | 20.275 ± 0.364 | 24.812 ± 0.603 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 12.735 ± 0.255 | - | 12.221 ± 0.225 | 12.675 ± 0.204 | 14.503 ± 0.379 | 20.951 ± 0.503 | 14.274 ± 0.212 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 32.463 ± 0.037 | - | 32.532 ± 0.422 | 26.488 ± 0.103 | 26.394 ± 0.233 | 36.116 ± 0.419 | 41.199 ± 0.544 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 14.408 ± 0.594 | - | 16.123 ± 1.900 | 11.131 ± 0.074 | 14.707 ± 0.343 | 37.433 ± 1.069 | 20.070 ± 0.345 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 36.570 ± 0.258 | - | 36.469 ± 0.104 | 36.948 ± 1.305 | 42.048 ± 0.098 | 20.329 ± 0.165 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 19.840 ± 0.017 | - | 19.918 ± 0.157 | 19.786 ± 0.136 | 18.589 ± 0.185 | 10.048 ± 0.275 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 37.118 ± 0.231 | - | unsupported | 36.845 ± 1.261 | 44.133 ± 2.230 | 20.830 ± 2.677 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 20.212 ± 0.245 | - | unsupported | 19.842 ± 0.016 | 19.578 ± 1.437 | 10.449 ± 1.150 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 6.765 ± 0.076 | - | 6.772 ± 0.019 | 7.313 ± 0.073 | 7.142 ± 0.035 | 7.935 ± 0.040 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 7.128 ± 0.027 | - | 7.755 ± 0.117 | 7.055 ± 0.061 | 9.430 ± 0.871 | 8.745 ± 0.080 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 48.122 ± 0.170 | - | 48.182 ± 0.093 | 55.912 ± 1.110 | 41.364 ± 1.232 | 31.659 ± 0.280 | 32.173 ± 1.792 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 17.547 ± 0.055 | - | 18.086 ± 0.697 | 20.942 ± 0.092 | 22.081 ± 0.350 | 32.695 ± 1.557 | 15.933 ± 0.970 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 62.061 ± 1.036 | - | 62.001 ± 0.480 | 73.911 ± 3.264 | 84.856 ± 0.258 | 60.481 ± 0.481 | 62.783 ± 2.138 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 27.184 ± 0.261 | - | 27.362 ± 0.207 | 30.203 ± 0.153 | 37.771 ± 1.943 | 61.054 ± 0.867 | 28.986 ± 0.671 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 14.191 ± 0.139 | - | 13.581 ± 0.110 | 13.247 ± 0.070 | 12.830 ± 0.038 | 13.150 ± 0.059 | 14.878 ± 0.256 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 9.491 ± 0.536 | - | 9.520 ± 0.323 | 8.947 ± 0.155 | 9.979 ± 3.692 | 13.452 ± 0.103 | 10.152 ± 0.477 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.328 ± 0.042 | - | 11.117 ± 0.140 | 11.123 ± 0.235 | 3.376 ± 0.061 | 2.810 ± 0.010 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 0.786 ± 0.062 | - | 9.187 ± 0.013 | 5.549 ± 0.032 | 1.892 ± 0.153 | 2.870 ± 0.026 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 5.631 ± 0.018 | - | 5.647 ± 0.017 | 6.535 ± 0.203 | 4.038 ± 0.018 | 4.805 ± 0.113 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 3.266 ± 0.025 | - | 3.335 ± 0.024 | 3.430 ± 0.004 | 3.596 ± 0.009 | 4.639 ± 0.882 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 3.753 ± 0.040 | - | 3.811 ± 0.021 | 3.710 ± 0.072 | 2.480 ± 0.026 | 2.424 ± 0.007 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 2.243 ± 0.093 | - | 2.222 ± 0.056 | 1.944 ± 0.012 | 1.725 ± 0.121 | 1.624 ± 0.154 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 4.145 ± 0.022 | - | 4.142 ± 0.014 | 5.082 ± 0.105 | 2.896 ± 0.019 | 2.848 ± 0.040 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 3.081 ± 0.029 | - | 3.090 ± 0.010 | 3.276 ± 0.019 | 3.118 ± 0.064 | 3.141 ± 0.043 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 36.461 ± 0.237 | - | 36.424 ± 0.091 | 37.274 ± 0.380 | 41.996 ± 0.046 | 19.539 ± 0.103 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 19.798 ± 0.035 | - | 19.875 ± 0.538 | 19.743 ± 0.334 | 20.208 ± 2.014 | 9.719 ± 0.064 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 37.018 ± 0.177 | - | unsupported | 35.972 ± 1.272 | 34.044 ± 0.263 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 17.665 ± 0.316 | - | unsupported | 18.357 ± 0.675 | 16.652 ± 0.038 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 19.012 ± 0.163 | - | 52.796 ± 0.173 | 18.897 ± 0.112 | 20.614 ± 0.417 | 20.744 ± 0.110 | 37.656 ± 0.766 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 12.877 ± 0.157 | - | 12.512 ± 0.074 | 12.540 ± 0.227 | 16.148 ± 0.143 | 20.897 ± 0.193 | 17.197 ± 0.128 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 24.251 ± 0.103 | - | 54.932 ± 0.935 | 24.503 ± 0.224 | 26.191 ± 0.270 | 42.473 ± 56.184 | 83.785 ± 1.784 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 17.706 ± 0.420 | - | 17.648 ± 0.146 | 17.567 ± 0.030 | 19.675 ± 0.061 | 25.211 ± 0.283 | 21.356 ± 0.268 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 8.325 ± 0.096 | - | 27.916 ± 0.292 | 17.762 ± 0.315 | 23.141 ± 0.200 | 36.204 ± 1.671 | 39.565 ± 3.082 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 4.953 ± 0.135 | - | 14.069 ± 1.402 | 10.287 ± 0.281 | 10.991 ± 0.069 | 35.293 ± 0.653 | 17.876 ± 2.198 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 6.346 ± 0.482 | - | 6.598 ± 0.121 | 8.390 ± 0.066 | 9.674 ± 0.152 | 8.495 ± 0.502 | 16.494 ± 1.342 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 4.984 ± 0.013 | - | 5.058 ± 0.037 | 6.052 ± 0.086 | 6.530 ± 0.072 | 8.376 ± 0.586 | 8.528 ± 3.600 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 13.064 ± 0.121 | - | 13.165 ± 0.075 | 18.116 ± 0.241 | 17.306 ± 0.812 | 13.516 ± 0.570 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 11.002 ± 0.047 | - | 11.035 ± 0.078 | 12.231 ± 0.071 | 14.672 ± 0.134 | 13.024 ± 0.247 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 8.142 ± 0.202 | - | 8.261 ± 0.098 | 24.761 ± 0.217 | 64.636 ± 0.471 | 28.868 ± 1.334 | 32.340 ± 1.814 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 3.347 ± 0.040 | - | 3.330 ± 0.007 | 10.042 ± 0.137 | 30.886 ± 0.204 | 29.098 ± 7.203 | 14.070 ± 1.335 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 24.660 ± 0.107 | - | 83.281 ± 1.141 | 24.430 ± 0.356 | 26.058 ± 0.421 | 25.098 ± 0.061 | 45.537 ± 0.856 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 18.060 ± 0.260 | - | 18.127 ± 0.645 | 17.623 ± 0.069 | 20.093 ± 0.314 | 25.342 ± 0.150 | 21.330 ± 0.532 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 4.143 ± 0.033 | - | 4.144 ± 0.031 | 13.901 ± 0.200 | 6.841 ± 0.349 | 22.941 ± 1.353 | 25.627 ± 1.443 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 2.475 ± 0.133 | - | 2.445 ± 0.045 | 6.402 ± 0.097 | 4.042 ± 0.097 | 22.919 ± 1.591 | 12.742 ± 1.245 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 5.952 ± 0.213 | - | 6.176 ± 0.037 | 6.998 ± 0.041 | 5.572 ± 0.044 | 15.399 ± 1.414 | 16.987 ± 3.022 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 2.389 ± 0.012 | - | 2.449 ± 0.019 | 3.255 ± 0.113 | 2.906 ± 0.028 | 14.923 ± 0.653 | 8.098 ± 2.442 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 7.917 ± 0.049 | - | 7.893 ± 0.029 | 18.276 ± 0.160 | 26.936 ± 0.394 | 31.043 ± 7.053 | 34.708 ± 0.864 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 3.520 ± 0.030 | - | 3.554 ± 0.049 | 7.765 ± 0.062 | 13.754 ± 0.254 | 29.377 ± 1.464 | 15.577 ± 1.518 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 4.244 ± 0.017 | - | 4.187 ± 0.024 | 6.955 ± 0.111 | 13.363 ± 0.132 | 18.714 ± 1.178 | 20.141 ± 2.566 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 1.926 ± 0.007 | - | 1.939 ± 0.015 | 3.505 ± 0.065 | 6.980 ± 0.160 | 18.474 ± 0.580 | 9.764 ± 2.326 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.108 ± 0.004 | 0.119 ± 0.004 | unsupported | unsupported | 0.215 ± 0.003 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.159 ± 0.009 | 0.183 ± 0.004 | unsupported | unsupported | 0.158 ± 0.003 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 24.887 ± 0.102 | - | 86.518 ± 0.660 | 24.758 ± 0.407 | 26.359 ± 0.090 | 25.796 ± 0.271 | 46.174 ± 1.164 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 17.886 ± 0.235 | - | 18.124 ± 0.572 | 17.541 ± 0.263 | 19.785 ± 0.293 | 25.477 ± 0.084 | 21.503 ± 0.572 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 24.976 ± 0.189 | - | 85.340 ± 1.132 | 24.582 ± 0.315 | 26.022 ± 0.358 | 25.756 ± 0.145 | 46.171 ± 0.623 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 18.136 ± 0.329 | - | 17.944 ± 0.658 | 17.504 ± 0.263 | 19.975 ± 0.386 | 25.445 ± 0.145 | 21.646 ± 0.479 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 24.275 ± 0.140 | - | 59.597 ± 0.155 | 24.318 ± 0.263 | 26.362 ± 0.275 | 24.997 ± 0.112 | 44.617 ± 0.702 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 17.954 ± 0.557 | - | 17.637 ± 0.483 | 17.567 ± 0.147 | 20.081 ± 0.123 | 25.441 ± 0.348 | 21.086 ± 0.935 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 18.902 ± 0.085 | - | 52.651 ± 0.235 | 19.317 ± 11.613 | 20.483 ± 0.578 | 20.729 ± 0.153 | 37.122 ± 0.226 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 12.784 ± 0.050 | - | 12.345 ± 0.167 | 12.834 ± 0.277 | 16.133 ± 0.226 | 20.631 ± 0.904 | 17.057 ± 0.157 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 33.453 ± 0.688 | - | 33.328 ± 0.889 | 15.452 ± 0.201 | 36.468 ± 0.067 | 85.070 ± 4.253 | 58.485 ± 4.505 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 14.431 ± 0.240 | - | 14.718 ± 0.366 | 6.600 ± 0.029 | 17.228 ± 0.061 | 83.018 ± 4.566 | 24.213 ± 2.554 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 10.392 ± 0.112 | - | 10.396 ± 0.013 | 8.768 ± 0.035 | 47.217 ± 0.269 | 10.004 ± 0.080 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 5.676 ± 0.901 | - | 5.121 ± 0.231 | 4.992 ± 0.061 | 22.657 ± 0.248 | 9.889 ± 0.238 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.387 ± 0.021 | - | 1.375 ± 0.034 | 25.316 ± 0.799 | 1.293 ± 0.014 | 0.884 ± 0.105 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.541 ± 0.017 | - | 0.541 ± 0.012 | 9.708 ± 0.056 | 0.714 ± 0.010 | 0.908 ± 0.071 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 1.079 ± 0.070 | - | 1.088 ± 0.057 | 1.941 ± 0.034 | 3.048 ± 0.033 | 0.565 ± 0.082 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 0.683 ± 0.114 | - | 0.676 ± 0.029 | 1.079 ± 0.018 | 2.198 ± 0.554 | 0.795 ± 0.074 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 9.798 ± 0.045 | - | 9.848 ± 0.087 | 8.747 ± 0.052 | 47.381 ± 1.105 | 10.519 ± 0.054 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 5.316 ± 0.103 | - | 5.654 ± 0.107 | 5.011 ± 0.042 | 22.460 ± 0.088 | 10.191 ± 0.197 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.236 ± 0.018 | - | 1.234 ± 0.022 | 23.313 ± 0.389 | 1.315 ± 0.056 | 0.897 ± 0.021 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.527 ± 0.012 | - | 0.539 ± 0.015 | 9.400 ± 0.103 | 0.740 ± 0.018 | 1.109 ± 0.047 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 7.174 ± 0.110 | - | 7.297 ± 0.120 | 7.636 ± 0.041 | 12.226 ± 0.233 | 5.470 ± 0.054 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 3.561 ± 0.145 | - | 3.589 ± 0.059 | 3.862 ± 0.025 | 5.876 ± 2.874 | 5.673 ± 1.614 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 6.602 ± 0.097 | - | 6.562 ± 0.130 | 6.144 ± 0.088 | 23.358 ± 0.092 | 6.065 ± 0.054 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 5.140 ± 0.056 | - | 5.098 ± 0.038 | 4.962 ± 0.036 | 11.681 ± 0.232 | 6.138 ± 0.191 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.513 ± 0.038 | - | 0.537 ± 0.063 | 1.746 ± 0.023 | 1.331 ± 0.095 | 0.419 ± 0.024 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.217 ± 0.014 | - | 0.227 ± 0.074 | 0.748 ± 0.102 | 1.249 ± 0.526 | 0.517 ± 0.066 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 0.696 ± 0.119 | - | 0.674 ± 0.065 | 0.494 ± 0.028 | 0.976 ± 0.008 | 0.566 ± 0.079 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.405 ± 0.066 | - | 0.434 ± 0.027 | 0.180 ± 0.033 | 0.584 ± 0.232 | 0.585 ± 0.086 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 6.653 ± 0.076 | - | 6.562 ± 0.078 | 5.956 ± 0.072 | 16.869 ± 0.121 | 6.174 ± 0.061 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 5.140 ± 0.036 | - | 5.135 ± 0.032 | 5.029 ± 0.110 | 8.399 ± 0.048 | 6.301 ± 0.065 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.518 ± 0.085 | - | 0.593 ± 0.139 | 1.718 ± 0.029 | 1.326 ± 0.054 | 0.435 ± 0.035 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.205 ± 0.029 | - | 0.198 ± 0.021 | 0.708 ± 0.047 | 1.108 ± 0.190 | 0.491 ± 0.052 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 0.720 ± 0.199 | - | 0.678 ± 0.072 | 0.484 ± 0.036 | 0.934 ± 0.034 | 0.488 ± 0.041 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.385 ± 0.040 | - | 0.451 ± 0.034 | 0.178 ± 0.016 | 0.636 ± 0.019 | 0.656 ± 0.051 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.510 ± 0.050 | 0.569 ± 0.053 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.216 ± 0.004 | 0.211 ± 0.032 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 20.754 ± 0.778 | - | 25.437 ± 0.493 | 19.729 ± 0.402 | 27.174 ± 0.173 | 29.226 ± 3.392 | 34.251 ± 1.638 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 8.577 ± 0.100 | - | 9.149 ± 0.236 | 8.938 ± 0.231 | 14.798 ± 0.257 | 30.801 ± 0.930 | 15.409 ± 1.168 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 72.261 ± 1.317 | - | 94.629 ± 0.774 | 33.199 ± 0.301 | 67.935 ± 1.919 | 98.478 ± 1.215 | 98.880 ± 0.652 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 15.980 ± 1.646 | - | 15.063 ± 0.973 | 15.571 ± 2.613 | 35.869 ± 0.474 | 102.224 ± 3.935 | 49.233 ± 2.684 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 26.706 ± 0.174 | - | 86.015 ± 0.289 | 30.680 ± 0.338 | 41.863 ± 1.046 | 27.965 ± 0.100 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 19.183 ± 0.292 | - | 19.727 ± 0.245 | 18.400 ± 0.156 | 25.580 ± 0.183 | 28.303 ± 0.160 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 20.320 ± 0.177 | - | 54.052 ± 1.440 | 21.443 ± 0.376 | 21.366 ± 0.079 | 21.535 ± 0.179 | 37.014 ± 0.379 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 12.776 ± 0.159 | - | 12.392 ± 0.228 | 12.628 ± 0.103 | 14.979 ± 0.239 | 21.664 ± 0.398 | 17.198 ± 0.429 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 5.957 ± 0.034 | - | 6.322 ± 0.101 | 26.029 ± 0.945 | 63.908 ± 0.530 | 27.787 ± 0.579 | 30.497 ± 1.492 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 2.658 ± 0.131 | - | 2.694 ± 0.031 | 9.947 ± 0.108 | 31.608 ± 1.029 | 26.564 ± 1.561 | 13.437 ± 1.410 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 64.927 ± 0.349 | - | 85.303 ± 2.497 | 57.053 ± 0.907 | 22.109 ± 0.162 | 68.795 ± 0.933 | 68.725 ± 1.173 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 13.612 ± 0.150 | - | 13.788 ± 0.344 | 27.404 ± 0.545 | 14.579 ± 0.064 | 71.370 ± 6.636 | 34.021 ± 0.333 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 24.217 ± 0.104 | - | 59.648 ± 0.392 | 24.471 ± 0.304 | 26.415 ± 0.383 | 25.076 ± 0.073 | 48.737 ± 0.809 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 18.120 ± 0.491 | - | 17.637 ± 0.158 | 17.543 ± 0.142 | 19.871 ± 0.503 | 25.483 ± 0.238 | 21.655 ± 0.668 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 15.030 ± 0.146 | - | 14.513 ± 0.079 | 35.489 ± 0.659 | 6.597 ± 0.222 | 35.485 ± 2.488 | 52.792 ± 15.234 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 6.312 ± 0.071 | - | 8.351 ± 0.484 | 14.729 ± 0.309 | 3.957 ± 0.039 | 33.210 ± 1.012 | 16.120 ± 1.670 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 84.671 ± 0.516 | 86.333 ± 0.460 | unsupported | 30.064 ± 0.350 | 51.232 ± 5.066 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 19.587 ± 0.379 | 21.414 ± 2.320 | unsupported | 18.424 ± 0.346 | 25.753 ± 0.341 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.481 ± 0.041 | - | 0.452 ± 0.026 | 0.384 ± 0.063 | 0.670 ± 0.086 | 1.719 ± 0.607 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.186 ± 0.031 | - | 0.232 ± 0.020 | 0.201 ± 0.024 | 0.529 ± 0.010 | 1.650 ± 0.566 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.387 ± 0.051 | - | 0.384 ± 0.019 | 0.475 ± 0.063 | 0.569 ± 0.141 | 1.670 ± 0.583 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.164 ± 0.026 | - | 0.195 ± 0.012 | 0.182 ± 0.026 | 1.373 ± 0.190 | 1.714 ± 0.547 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 0.777 ± 0.025 | - | unsupported | 0.765 ± 0.032 | 0.840 ± 0.023 | 1.625 ± 1.098 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.395 ± 0.076 | - | unsupported | 0.369 ± 0.063 | 0.636 ± 0.041 | 2.017 ± 0.520 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.234 ± 0.008 | - | 0.268 ± 0.008 | 0.352 ± 0.008 | 0.215 ± 0.019 | 0.739 ± 0.067 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.150 ± 0.005 | - | 0.183 ± 0.001 | 0.170 ± 0.004 | 0.136 ± 0.007 | 0.824 ± 0.694 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.394 ± 0.031 | - | 0.391 ± 0.023 | 0.602 ± 0.062 | 0.661 ± 0.012 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.150 ± 0.042 | - | 0.158 ± 0.011 | 0.276 ± 0.051 | 0.575 ± 0.033 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.489 ± 0.076 | - | 0.476 ± 0.021 | 0.423 ± 0.036 | 0.533 ± 0.051 | 0.768 ± 1.444 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.183 ± 0.025 | - | 0.195 ± 0.038 | 0.185 ± 0.009 | 0.244 ± 0.023 | 1.912 ± 1.491 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.384 ± 0.004 | - | 0.418 ± 0.008 | 0.198 ± 0.005 | 0.210 ± 0.015 | 0.252 ± 0.031 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.462 ± 0.022 | - | 0.444 ± 0.023 | 0.221 ± 0.002 | 0.210 ± 0.004 | 0.885 ± 0.019 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.056 ± 0.062 | - | 0.002 ± 0.000 | 1.117 ± 0.077 | 0.857 ± 0.017 | 2.168 ± 0.906 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.629 ± 0.015 | - | 0.002 ± 0.000 | 0.637 ± 0.077 | 0.568 ± 0.042 | 2.582 ± 0.900 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.028 ± 0.000 | - | 0.032 ± 0.000 | 0.042 ± 0.000 | 0.061 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.058 ± 0.003 | - | 0.062 ± 0.000 | 0.060 ± 0.000 | 0.090 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.143 ± 0.000 | - | 0.152 ± 0.001 | 0.156 ± 0.002 | 0.184 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.044 ± 0.001 | - | 0.045 ± 0.001 | 0.100 ± 0.003 | 0.075 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.338 ± 0.007 | - | 0.367 ± 0.005 | 0.811 ± 0.023 | 0.427 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.013 ± 0.000 | - | 0.016 ± 0.000 | 0.020 ± 0.000 | 0.037 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.027 ± 0.000 | - | 0.029 ± 0.000 | 0.044 ± 0.000 | 0.053 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.054 ± 0.000 | - | 0.059 ± 0.001 | 0.067 ± 0.000 | 0.086 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.139 ± 0.015 | - | 0.180 ± 0.014 | 0.164 ± 0.001 | 0.189 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.041 ± 0.000 | - | 0.045 ± 0.002 | 0.087 ± 0.001 | 0.072 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.001 ± 0.001 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.332 ± 0.015 | - | 0.382 ± 0.018 | 0.714 ± 0.005 | 0.591 ± 0.256 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.027 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.329 ± 0.013 | - | 0.336 ± 0.011 | 0.414 ± 0.006 | 0.495 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.344 ± 0.015 | - | 1.345 ± 0.014 | 1.393 ± 0.011 | 1.670 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 4.099 ± 0.009 | - | 4.113 ± 0.010 | 4.217 ± 0.064 | 5.610 ± 0.167 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 2.264 ± 0.011 | - | 2.268 ± 0.014 | 1.987 ± 0.018 | 1.503 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.067 ± 0.000 | - | 0.070 ± 0.000 | 0.068 ± 0.001 | 0.097 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.016 ± 0.001 | 0.036 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.017 ± 0.000 | - | 0.019 ± 0.000 | 0.020 ± 0.000 | 0.046 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 17.946 ± 0.011 | - | 17.985 ± 0.115 | 15.571 ± 0.038 | 11.733 ± 0.178 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.529 ± 0.006 | - | 0.539 ± 0.010 | 0.519 ± 0.005 | 0.574 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.015 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.033 ± 0.003 | - | 0.035 ± 0.000 | 0.036 ± 0.000 | 0.062 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.495 ± 0.009 | - | 0.505 ± 0.009 | 0.581 ± 0.003 | 0.532 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 1.756 ± 0.475 | - | 1.780 ± 0.015 | 1.357 ± 0.004 | 1.391 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 5.089 ± 0.006 | - | 5.206 ± 0.142 | 5.149 ± 0.060 | 3.691 ± 0.066 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 1.959 ± 0.032 | - | 1.946 ± 0.022 | 1.267 ± 0.008 | 1.777 ± 0.070 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.087 ± 0.002 | - | 0.090 ± 0.000 | 0.086 ± 0.002 | 0.094 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.003 ± 0.002 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.016 ± 0.000 | - | 0.019 ± 0.000 | 0.016 ± 0.000 | 0.028 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.004 ± 0.002 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.021 ± 0.000 | - | 0.024 ± 0.000 | 0.020 ± 0.000 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 15.305 ± 0.058 | - | 15.652 ± 0.574 | 10.068 ± 0.089 | 14.647 ± 2.418 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.692 ± 0.011 | - | 0.701 ± 0.018 | 0.654 ± 0.012 | 0.883 ± 1.148 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.015 ± 0.000 | - | 0.017 ± 0.000 | 0.014 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.041 ± 0.000 | - | 0.044 ± 0.000 | 0.036 ± 0.000 | 0.063 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.359 ± 0.006 | - | 0.372 ± 0.008 | 0.230 ± 0.004 | 0.209 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.366 ± 0.012 | - | 1.365 ± 0.009 | 1.203 ± 0.013 | 1.176 ± 0.113 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 3.693 ± 0.010 | - | 3.709 ± 0.006 | 3.633 ± 0.017 | 4.078 ± 0.078 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 0.853 ± 0.017 | - | 0.853 ± 0.009 | 0.854 ± 0.012 | 0.813 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.002 ± 0.001 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.043 ± 0.001 | - | 0.045 ± 0.000 | 0.049 ± 0.001 | 0.077 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.015 ± 0.001 | 0.019 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.015 ± 0.000 | - | 0.017 ± 0.000 | 0.018 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 6.645 ± 0.029 | - | 6.658 ± 0.048 | 6.861 ± 0.145 | 6.220 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.339 ± 0.007 | - | 0.341 ± 0.006 | 0.360 ± 0.006 | 0.435 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.011 ± 0.000 | - | 0.014 ± 0.000 | 0.013 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.029 ± 0.001 | - | 0.032 ± 0.000 | 0.031 ± 0.000 | 0.055 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.540 ± 0.010 | - | 0.547 ± 0.010 | 0.360 ± 0.009 | 0.196 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.866 ± 0.010 | - | 1.861 ± 0.011 | 1.133 ± 0.003 | 1.044 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 4.746 ± 0.032 | - | 4.784 ± 0.042 | 4.854 ± 0.046 | 2.297 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 0.688 ± 0.010 | - | 0.688 ± 0.019 | 0.683 ± 0.028 | 0.962 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.002 ± 0.001 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.058 ± 0.001 | - | 0.061 ± 0.001 | 0.067 ± 0.001 | 0.063 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.001 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.007 ± 0.002 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.015 ± 0.000 | - | 0.018 ± 0.000 | 0.014 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.020 ± 0.000 | - | 0.023 ± 0.000 | 0.017 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 5.381 ± 0.066 | - | 5.561 ± 2.345 | 5.252 ± 0.066 | 7.687 ± 0.170 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.006 ± 0.002 | - | 0.008 ± 0.000 | 0.008 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.452 ± 0.013 | - | 0.461 ± 0.014 | 0.508 ± 0.007 | 0.243 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.015 ± 0.000 | - | 0.018 ± 0.000 | 0.017 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.038 ± 0.000 | - | 0.041 ± 0.000 | 0.030 ± 0.000 | 0.043 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.131 ± 0.001 | - | 0.136 ± 0.006 | 0.118 ± 0.001 | 0.213 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 0.722 ± 0.007 | - | 0.726 ± 0.010 | 0.586 ± 0.006 | 1.131 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 2.054 ± 0.005 | - | 2.067 ± 0.012 | 1.841 ± 0.003 | 3.898 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.340 ± 0.006 | - | 0.341 ± 0.007 | 0.358 ± 0.003 | 0.822 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.001 ± 0.001 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.004 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.020 ± 0.000 | - | 0.022 ± 0.000 | 0.024 ± 0.001 | 0.082 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.001 ± 0.001 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.012 ± 0.001 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 2.722 ± 0.011 | - | 2.732 ± 0.008 | 2.865 ± 0.012 | 6.077 ± 0.066 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.151 ± 0.000 | - | 0.154 ± 0.000 | 0.151 ± 0.006 | 0.466 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.017 ± 0.000 | - | 0.019 ± 0.000 | 0.018 ± 0.000 | 0.054 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.211 ± 0.001 | - | 0.219 ± 0.001 | 0.199 ± 0.007 | 0.196 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 0.894 ± 0.002 | - | 0.904 ± 0.006 | 0.586 ± 0.146 | 0.725 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 2.377 ± 0.011 | - | 2.435 ± 0.017 | 2.114 ± 0.004 | 2.261 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.388 ± 0.017 | - | 0.389 ± 0.012 | 0.392 ± 0.006 | 0.916 ± 0.077 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.001 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.023 ± 0.000 | - | 0.025 ± 0.000 | 0.028 ± 0.001 | 0.063 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.005 ± 0.001 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.011 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 3.070 ± 0.011 | - | 3.122 ± 0.024 | 3.060 ± 0.025 | 7.811 ± 0.850 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.170 ± 0.000 | - | 0.173 ± 0.009 | 0.175 ± 0.002 | 0.226 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.019 ± 0.000 | - | 0.022 ± 0.000 | 0.018 ± 0.000 | 0.040 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.093 ± 0.001 | - | 0.093 ± 0.001 | 0.093 ± 0.000 | 0.082 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.108 ± 0.003 | - | 0.116 ± 0.001 | 0.129 ± 0.001 | 0.118 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.280 ± 0.008 | - | 0.302 ± 0.009 | 0.288 ± 0.008 | 0.315 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.078 ± 0.003 | - | 0.082 ± 0.005 | 0.096 ± 0.003 | 0.120 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.024 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.034 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.004 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.649 ± 0.034 | - | 0.662 ± 0.014 | 0.736 ± 0.006 | 0.744 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.044 ± 0.000 | - | 0.048 ± 0.000 | 0.044 ± 0.000 | 0.059 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.115 ± 0.002 | - | 0.102 ± 0.001 | 0.082 ± 0.000 | 0.066 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.158 ± 0.002 | - | 0.156 ± 0.001 | 0.107 ± 0.000 | 0.110 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.395 ± 0.018 | - | 0.422 ± 0.008 | 0.177 ± 0.010 | 0.359 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.092 ± 0.002 | - | 0.098 ± 0.001 | 0.110 ± 0.002 | 0.101 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.011 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.016 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.011 ± 0.000 | - | 0.013 ± 0.001 | 0.017 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.003 ± 0.002 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.003 ± 0.002 | - | 0.005 ± 0.000 | 0.012 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.012 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.013 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.719 ± 0.012 | - | 0.771 ± 0.019 | 0.341 ± 0.007 | 0.539 ± 0.053 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.013 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.060 ± 0.006 | - | 0.064 ± 0.001 | 0.032 ± 0.000 | 0.075 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.013 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.005 ± 0.001 | - | 0.007 ± 0.000 | 0.014 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.120 ± 0.000 | - | 0.125 ± 0.001 | 0.108 ± 0.000 | 0.161 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.309 ± 0.007 | - | 0.316 ± 0.007 | 0.308 ± 0.069 | 0.432 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 0.772 ± 0.017 | - | 0.788 ± 0.009 | 0.775 ± 0.007 | 1.190 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.305 ± 0.008 | - | 0.309 ± 0.008 | 0.363 ± 0.009 | 0.341 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.013 ± 0.000 | 0.040 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.011 ± 0.000 | - | 0.014 ± 0.000 | 0.016 ± 0.000 | 0.030 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 2.445 ± 0.023 | - | 2.946 ± 0.017 | 2.946 ± 0.006 | 2.534 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.072 ± 0.000 | - | 0.074 ± 0.000 | 0.091 ± 0.000 | 0.123 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.003 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.115 ± 0.002 | - | 0.127 ± 0.001 | 0.157 ± 0.000 | 0.152 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.300 ± 0.013 | - | 0.307 ± 0.014 | 0.386 ± 0.008 | 0.431 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 0.739 ± 0.020 | - | 0.786 ± 0.019 | 1.017 ± 0.066 | 0.708 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.317 ± 0.008 | - | 0.323 ± 0.009 | 0.299 ± 0.019 | 0.423 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.001 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.010 ± 0.000 | - | 0.012 ± 0.006 | 0.021 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.015 ± 0.000 | - | 0.018 ± 0.000 | 0.023 ± 0.001 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.002 ± 0.001 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.003 ± 0.002 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.005 ± 0.003 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.002 ± 0.001 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.006 ± 0.002 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 2.480 ± 0.015 | - | 3.122 ± 0.026 | 2.322 ± 0.015 | 3.481 ± 0.760 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.069 ± 0.000 | - | 0.072 ± 0.000 | 0.140 ± 0.005 | 0.118 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.005 ± 0.005 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.013 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.108 ± 0.004 | - | 0.096 ± 0.001 | 0.047 ± 0.001 | 0.165 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.130 ± 0.001 | - | 0.109 ± 0.002 | 0.067 ± 0.004 | 0.217 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.303 ± 0.007 | - | 0.283 ± 0.015 | 0.242 ± 0.001 | 0.487 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.068 ± 0.001 | - | 0.073 ± 0.001 | 0.079 ± 0.001 | 0.094 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.006 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.013 ± 0.000 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.005 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.546 ± 0.009 | - | 0.572 ± 0.008 | 0.587 ± 0.019 | 0.568 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.043 ± 0.000 | - | 0.046 ± 0.000 | 0.042 ± 0.000 | 0.060 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.109 ± 0.016 | - | 0.087 ± 0.000 | 0.061 ± 0.000 | 0.153 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.128 ± 0.001 | - | 0.106 ± 0.003 | 0.081 ± 0.000 | 0.215 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.295 ± 0.047 | - | 0.337 ± 0.008 | 0.198 ± 0.001 | 0.497 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.066 ± 0.001 | - | 0.070 ± 0.001 | 0.094 ± 0.001 | 0.085 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.012 ± 0.000 | 0.006 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.018 ± 0.002 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.012 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.012 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.012 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.002 ± 0.001 | - | 0.004 ± 0.000 | 0.012 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.013 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.012 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.012 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.013 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.530 ± 0.017 | - | 0.576 ± 0.014 | 0.296 ± 0.010 | 0.412 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.014 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.042 ± 0.000 | - | 0.044 ± 0.000 | 0.032 ± 0.001 | 0.080 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.002 ± 0.001 | - | 0.005 ± 0.000 | 0.014 ± 0.000 | 0.008 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.015 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.095 ± 0.011 | - | 1.105 ± 0.009 | 0.865 ± 0.002 | 1.144 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 3.713 ± 0.015 | - | 3.718 ± 0.006 | 3.092 ± 0.025 | 3.170 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 11.076 ± 0.017 | - | 11.191 ± 0.055 | 9.656 ± 0.033 | 8.981 ± 0.138 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 3.138 ± 0.019 | - | 3.121 ± 0.007 | 3.478 ± 0.014 | 2.429 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.118 ± 0.003 | - | 0.120 ± 0.001 | 0.128 ± 0.002 | 0.137 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.107 ± 0.000 | - | 0.110 ± 0.001 | 0.117 ± 0.001 | 0.131 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.013 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.034 ± 0.000 | - | 0.036 ± 0.000 | 0.033 ± 0.001 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.016 ± 0.000 | - | 0.018 ± 0.000 | 0.016 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.045 ± 0.000 | - | 0.048 ± 0.000 | 0.043 ± 0.000 | 0.043 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 24.896 ± 0.083 | - | 25.924 ± 0.092 | 27.713 ± 0.478 | 18.911 ± 0.367 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 0.929 ± 0.003 | - | 0.940 ± 0.011 | 0.993 ± 0.004 | 0.908 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.030 ± 0.000 | - | 0.032 ± 0.000 | 0.029 ± 0.000 | 0.031 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.088 ± 0.000 | - | 0.090 ± 0.000 | 0.079 ± 0.000 | 0.099 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.513 ± 0.014 | - | 1.518 ± 0.012 | 0.862 ± 0.385 | 1.184 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 4.968 ± 0.018 | - | 4.973 ± 0.012 | 3.096 ± 0.008 | 1.513 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 10.534 ± 0.349 | - | 10.630 ± 0.015 | 11.186 ± 0.050 | 5.038 ± 1.081 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 2.643 ± 0.011 | - | 2.648 ± 0.014 | 2.698 ± 0.083 | 2.614 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.148 ± 0.004 | - | 0.151 ± 0.003 | 0.161 ± 0.004 | 0.121 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.132 ± 0.005 | - | 0.135 ± 0.001 | 0.140 ± 0.002 | 0.114 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.016 ± 0.000 | - | 0.018 ± 0.000 | 0.014 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.016 ± 0.000 | - | 0.019 ± 0.000 | 0.013 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.043 ± 0.000 | - | 0.046 ± 0.000 | 0.033 ± 0.000 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.021 ± 0.000 | - | 0.024 ± 0.000 | 0.016 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.058 ± 0.000 | - | 0.061 ± 0.001 | 0.042 ± 0.000 | 0.046 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 21.505 ± 0.708 | - | 22.273 ± 0.066 | 21.407 ± 0.125 | 37.634 ± 30.070 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.016 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 1.166 ± 0.012 | - | 1.186 ± 0.017 | 1.227 ± 0.006 | 0.404 ± 0.081 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.040 ± 0.000 | - | 0.043 ± 0.000 | 0.029 ± 0.000 | 0.032 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.112 ± 0.000 | - | 0.115 ± 0.001 | 0.079 ± 0.000 | 0.087 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.840 ± 0.010 | - | 0.843 ± 0.009 | 0.891 ± 0.209 | 0.711 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 3.168 ± 0.015 | - | 3.165 ± 0.017 | 2.609 ± 0.006 | 2.323 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 9.582 ± 0.019 | - | 9.607 ± 0.015 | 8.446 ± 0.009 | 7.267 ± 0.064 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.455 ± 0.015 | - | 1.464 ± 0.010 | 1.494 ± 0.010 | 1.281 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.080 ± 0.002 | - | 0.082 ± 0.000 | 0.089 ± 0.001 | 0.101 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.066 ± 0.002 | - | 0.068 ± 0.001 | 0.071 ± 0.001 | 0.102 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.013 ± 0.002 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.018 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.027 ± 0.001 | - | 0.029 ± 0.000 | 0.027 ± 0.000 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.020 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.036 ± 0.000 | - | 0.038 ± 0.000 | 0.035 ± 0.000 | 0.034 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 11.483 ± 0.041 | - | 11.516 ± 0.070 | 11.825 ± 0.063 | 9.767 ± 0.065 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.008 ± 0.005 | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 0.637 ± 0.005 | - | 0.640 ± 0.008 | 0.680 ± 0.007 | 0.684 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.026 ± 0.000 | - | 0.028 ± 0.000 | 0.024 ± 0.000 | 0.026 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.072 ± 0.000 | - | 0.074 ± 0.000 | 0.067 ± 0.000 | 0.076 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 1.139 ± 0.004 | - | 1.140 ± 0.006 | 0.860 ± 0.244 | 0.710 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 4.257 ± 0.026 | - | 4.267 ± 0.038 | 2.622 ± 0.018 | 1.111 ± 0.114 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 11.042 ± 2.850 | - | 9.468 ± 0.298 | 10.112 ± 0.073 | 3.709 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 1.427 ± 0.016 | - | 1.434 ± 0.018 | 1.529 ± 0.014 | 1.552 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.107 ± 0.003 | - | 0.110 ± 0.001 | 0.120 ± 0.001 | 0.100 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.083 ± 0.002 | - | 0.086 ± 0.002 | 0.095 ± 0.003 | 0.133 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.012 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.014 ± 0.000 | - | 0.017 ± 0.000 | 0.012 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.035 ± 0.000 | - | 0.037 ± 0.000 | 0.027 ± 0.000 | 0.027 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.015 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.046 ± 0.000 | - | 0.049 ± 0.001 | 0.035 ± 0.000 | 0.034 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 11.153 ± 0.025 | - | 11.259 ± 0.074 | 12.021 ± 0.452 | 12.955 ± 0.468 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.013 ± 0.000 | 0.013 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 0.836 ± 0.005 | - | 0.838 ± 0.006 | 0.909 ± 0.006 | 0.377 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.035 ± 0.000 | - | 0.038 ± 0.001 | 0.025 ± 0.000 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.093 ± 0.000 | - | 0.095 ± 0.001 | 0.067 ± 0.000 | 0.073 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.255 ± 0.007 | - | unsupported | 0.172 ± 0.003 | 0.367 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.343 ± 0.014 | - | unsupported | 1.130 ± 0.013 | 1.455 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 3.236 ± 0.006 | - | unsupported | 3.045 ± 0.025 | 3.791 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.602 ± 0.017 | - | unsupported | 0.656 ± 0.016 | 0.639 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.001 ± 0.001 | - | unsupported | 0.004 ± 0.000 | 0.005 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.030 ± 0.001 | - | unsupported | 0.033 ± 0.001 | 0.040 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.024 ± 0.000 | - | unsupported | 0.028 ± 0.000 | 0.033 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.010 ± 0.000 | - | unsupported | 0.013 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.013 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 4.733 ± 0.010 | - | unsupported | 5.163 ± 0.016 | 4.993 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.233 ± 0.005 | - | unsupported | 0.238 ± 0.005 | 0.278 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.011 ± 0.000 | - | unsupported | 0.012 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.026 ± 0.000 | - | unsupported | 0.027 ± 0.000 | 0.034 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.390 ± 0.012 | - | unsupported | 0.276 ± 0.008 | 0.610 ± 0.288 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 1.840 ± 0.006 | - | unsupported | 1.117 ± 0.002 | 0.751 ± 0.130 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 4.343 ± 0.007 | - | unsupported | 4.019 ± 0.006 | 1.934 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 0.651 ± 0.014 | - | unsupported | 0.769 ± 0.019 | 0.767 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.002 ± 0.001 | - | unsupported | 0.006 ± 0.000 | 0.005 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.035 ± 0.000 | - | unsupported | 0.042 ± 0.001 | 0.038 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.029 ± 0.000 | - | unsupported | 0.036 ± 0.002 | 0.043 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.006 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.013 ± 0.000 | - | unsupported | 0.013 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.008 ± 0.000 | - | unsupported | 0.012 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.018 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 5.108 ± 0.009 | - | unsupported | 5.898 ± 0.008 | 6.625 ± 1.937 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.005 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.274 ± 0.008 | - | unsupported | 0.302 ± 0.045 | 0.133 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.015 ± 0.000 | - | unsupported | 0.017 ± 0.002 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.034 ± 0.000 | - | unsupported | 0.027 ± 0.000 | 0.033 ± 0.001 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.674 ± 0.012 | - | unsupported | 0.637 ± 0.016 | 1.691 ± 0.032 | 1.482 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.124 ± 0.002 | - | unsupported | 0.031 ± 0.001 | 0.076 ± 0.008 | 0.065 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.083 ± 0.000 | - | unsupported | 0.043 ± 0.002 | 0.120 ± 0.029 | 0.106 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.180 ± 0.015 | - | unsupported | 0.146 ± 0.001 | 0.312 ± 0.013 | 0.426 ± 0.020 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.631 ± 0.008 | - | unsupported | 0.283 ± 0.011 | 0.613 ± 0.038 | 1.018 ± 0.623 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.119 ± 0.001 | - | unsupported | 0.030 ± 0.001 | 0.060 ± 0.002 | 0.071 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.080 ± 0.001 | - | unsupported | 0.039 ± 0.001 | 0.103 ± 0.001 | 0.112 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.166 ± 0.001 | - | unsupported | 0.078 ± 0.001 | 0.356 ± 0.020 | 0.438 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.181 ± 0.001 | - | unsupported | 0.228 ± 0.007 | 0.513 ± 0.087 | 0.256 ± 0.008 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.018 ± 0.000 | - | unsupported | 0.021 ± 0.000 | 0.132 ± 0.003 | 0.051 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.037 ± 0.000 | - | unsupported | 0.027 ± 0.000 | 0.153 ± 0.005 | 0.075 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.122 ± 0.001 | - | unsupported | 0.103 ± 0.000 | 0.247 ± 0.007 | 0.131 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.175 ± 0.002 | - | unsupported | 0.278 ± 0.008 | 0.448 ± 0.009 | 0.232 ± 0.008 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.017 ± 0.000 | - | unsupported | 0.034 ± 0.001 | 0.111 ± 0.001 | 0.051 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.036 ± 0.000 | - | unsupported | 0.047 ± 0.002 | 0.137 ± 0.003 | 0.073 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.118 ± 0.005 | - | unsupported | 0.122 ± 0.002 | 0.234 ± 0.026 | 0.118 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.121 ± 0.001 | - | 0.166 ± 0.001 | 0.122 ± 0.001 | 0.215 ± 0.004 | 0.179 ± 0.008 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.042 ± 0.001 | - | 0.046 ± 0.000 | 0.038 ± 0.000 | 0.050 ± 0.001 | 0.045 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.052 ± 0.000 | - | 0.056 ± 0.000 | 0.046 ± 0.000 | 0.062 ± 0.002 | 0.056 ± 0.008 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.070 ± 0.001 | - | 0.084 ± 0.000 | 0.072 ± 0.001 | 0.111 ± 0.003 | 0.091 ± 0.004 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.116 ± 0.001 | - | 0.167 ± 0.010 | 0.140 ± 0.001 | 0.245 ± 0.017 | 0.144 ± 0.003 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.042 ± 0.000 | - | 0.044 ± 0.000 | 0.058 ± 0.003 | 0.048 ± 0.001 | 0.043 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.051 ± 0.000 | - | 0.054 ± 0.000 | 0.068 ± 0.001 | 0.060 ± 0.000 | 0.050 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.068 ± 0.002 | - | 0.104 ± 0.001 | 0.092 ± 0.001 | 0.117 ± 0.027 | 0.080 ± 0.014 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 30.015 ± 0.155 | - | 31.546 ± 0.173 | 43.951 ± 0.168 | 40.697 ± 0.257 | 23.289 ± 12.282 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 12.798 ± 0.120 | - | 13.420 ± 0.367 | 17.466 ± 0.084 | 33.731 ± 3.530 | 18.984 ± 7.859 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 16.369 ± 0.369 | - | 15.576 ± 0.537 | 15.353 ± 0.045 | 12.824 ± 0.088 | 9.408 ± 0.067 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 8.154 ± 0.118 | - | 7.102 ± 0.061 | 6.886 ± 0.052 | 6.735 ± 0.239 | 5.977 ± 0.477 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 4.671 ± 0.046 | - | 4.627 ± 0.054 | 4.678 ± 0.058 | 4.516 ± 0.063 | 5.520 ± 0.161 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 5.875 ± 0.037 | - | 5.901 ± 0.050 | 5.477 ± 0.018 | 5.733 ± 0.346 | 5.449 ± 0.150 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 5.347 ± 0.113 | - | 5.401 ± 0.060 | 5.239 ± 0.063 | 5.691 ± 0.093 | 6.674 ± 0.331 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 6.866 ± 0.050 | - | 6.895 ± 0.023 | 6.045 ± 0.054 | 7.522 ± 0.415 | 6.991 ± 0.039 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 10.817 ± 0.122 | - | 11.267 ± 0.247 | 10.777 ± 0.019 | 21.905 ± 0.339 | 7.146 ± 0.139 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 8.579 ± 0.021 | - | 8.668 ± 0.085 | 8.464 ± 0.031 | 17.602 ± 0.625 | 6.997 ± 0.343 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.312 ± 0.002 | 0.318 ± 0.002 | unsupported | 0.390 ± 0.002 | 0.425 ± 0.035 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 4.899 ± 0.048 | 5.551 ± 0.023 | unsupported | 5.406 ± 0.047 | 4.681 ± 0.019 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.198 ± 0.004 | 0.215 ± 0.002 | unsupported | 0.290 ± 0.002 | 0.631 ± 0.557 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 3.582 ± 0.035 | 3.684 ± 0.039 | unsupported | 3.285 ± 0.082 | 4.308 ± 0.179 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 24.542 ± 0.789 | - | 24.522 ± 0.104 | 26.475 ± 0.443 | 20.729 ± 0.200 | 12.935 ± 0.052 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 12.499 ± 0.104 | - | 12.373 ± 0.090 | 12.982 ± 0.040 | 11.474 ± 0.434 | 8.169 ± 0.152 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 11.252 ± 0.041 | 11.076 ± 0.064 | 5.702 ± 0.083 | 22.809 ± 0.241 | 3.335 ± 0.008 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 6.935 ± 0.237 | 6.859 ± 0.222 | 3.516 ± 0.061 | 21.956 ± 0.688 | 3.978 ± 0.146 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 16.753 ± 0.437 | - | 18.018 ± 0.784 | 26.656 ± 2.956 | 27.100 ± 0.646 | 15.471 ± 11.440 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 8.268 ± 0.110 | - | 8.672 ± 0.153 | 10.356 ± 0.228 | 18.300 ± 0.497 | 11.680 ± 10.297 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.624 ± 0.057 | - | 0.540 ± 0.026 | 0.744 ± 0.038 | 3.567 ± 0.117 | 1.869 ± 0.007 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.211 ± 0.043 | - | 0.232 ± 0.022 | 0.605 ± 0.041 | 2.017 ± 0.177 | 1.895 ± 0.006 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 11.888 ± 0.201 | - | 12.215 ± 0.100 | 12.284 ± 0.227 | 11.219 ± 0.111 | 9.435 ± 2.417 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 9.036 ± 0.061 | - | 9.167 ± 0.075 | 9.795 ± 0.130 | 11.695 ± 0.895 | 10.007 ± 0.273 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 11.926 ± 0.112 | - | 12.224 ± 0.437 | 12.175 ± 0.517 | 11.233 ± 0.068 | 9.460 ± 1.446 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 9.101 ± 0.060 | - | 9.154 ± 0.046 | 9.766 ± 0.044 | 11.569 ± 0.272 | 9.835 ± 0.420 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 0.720 ± 0.006 | 0.729 ± 0.008 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 13.861 ± 0.571 | 14.104 ± 0.090 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 0.757 ± 0.002 | 0.770 ± 0.006 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 11.911 ± 0.170 | 11.807 ± 0.497 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 0.670 ± 0.005 | 0.681 ± 0.005 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 13.251 ± 0.099 | 13.457 ± 0.066 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 0.712 ± 0.020 | 0.704 ± 0.014 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 11.059 ± 0.281 | 11.242 ± 0.221 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 16.417 ± 0.633 | - | 15.359 ± 0.155 | 15.452 ± 0.231 | 12.870 ± 0.229 | 9.411 ± 0.340 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 8.314 ± 0.293 | - | 7.086 ± 0.165 | 6.869 ± 0.040 | 6.767 ± 0.040 | 6.142 ± 0.594 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 36.982 ± 0.161 | 36.852 ± 0.187 | 38.475 ± 0.590 | 32.432 ± 0.257 | 28.461 ± 0.300 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 22.555 ± 0.918 | 21.898 ± 0.528 | 22.438 ± 0.134 | 28.378 ± 0.669 | 30.857 ± 5.618 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 32.490 ± 0.485 | - | 41.291 ± 1.110 | 20.203 ± 0.647 | 67.524 ± 0.440 | 13.800 ± 0.345 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 19.717 ± 0.785 | - | 29.724 ± 2.146 | 14.914 ± 0.252 | 52.822 ± 20.829 | 11.479 ± 1.243 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.042 ± 0.001 | 0.042 ± 0.000 | unsupported | 0.098 ± 0.019 | 0.093 ± 0.008 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 0.668 ± 0.005 | 0.675 ± 0.010 | unsupported | 1.274 ± 0.046 | 0.334 ± 0.112 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.037 ± 0.002 | 0.038 ± 0.002 | unsupported | 0.044 ± 0.002 | 0.022 ± 0.005 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.274 ± 0.004 | 0.283 ± 0.004 | unsupported | 0.518 ± 0.009 | 0.368 ± 0.206 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.020 ± 0.000 | 0.020 ± 0.000 | unsupported | 0.064 ± 0.010 | 0.138 ± 0.004 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 0.315 ± 0.003 | 0.325 ± 0.007 | unsupported | 0.508 ± 0.006 | 1.074 ± 0.011 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.015 ± 0.000 | 0.015 ± 0.001 | unsupported | 0.041 ± 0.002 | 0.090 ± 0.002 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.156 ± 0.002 | 0.164 ± 0.003 | unsupported | 0.101 ± 0.007 | 0.575 ± 0.013 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.071 ± 0.000 | 0.071 ± 0.000 | unsupported | 0.113 ± 0.000 | 0.112 ± 0.005 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 1.135 ± 0.008 | 1.156 ± 0.016 | unsupported | 1.504 ± 0.007 | 0.625 ± 0.033 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.032 ± 0.000 | 0.033 ± 0.000 | unsupported | 0.029 ± 0.000 | 0.083 ± 0.002 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 0.449 ± 0.026 | 0.439 ± 0.048 | unsupported | 0.578 ± 0.012 | 0.427 ± 0.003 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.026 ± 0.004 | 0.041 ± 0.004 | unsupported | 0.098 ± 0.001 | 0.116 ± 0.001 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.351 ± 0.021 | 0.371 ± 0.019 | unsupported | 0.902 ± 0.016 | 0.645 ± 0.005 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.052 ± 0.010 | 0.101 ± 0.020 | unsupported | 0.025 ± 0.000 | 0.103 ± 0.007 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.243 ± 0.021 | 0.286 ± 0.007 | unsupported | 0.227 ± 0.002 | 0.506 ± 0.026 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.043 ± 0.003 | 0.054 ± 0.001 | unsupported | 0.061 ± 0.000 | 0.158 ± 0.004 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 0.556 ± 0.004 | 0.587 ± 0.013 | unsupported | 0.652 ± 0.005 | 1.342 ± 0.006 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.067 ± 0.009 | 0.087 ± 0.005 | unsupported | 0.012 ± 0.000 | 0.151 ± 0.002 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.339 ± 0.010 | 0.381 ± 0.007 | unsupported | 0.168 ± 0.002 | 0.947 ± 0.043 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.064 ± 0.003 | 0.082 ± 0.004 | unsupported | unsupported | 0.164 ± 0.014 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 0.932 ± 0.043 | 0.934 ± 0.017 | unsupported | unsupported | 1.634 ± 0.013 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.107 ± 0.012 | 0.142 ± 0.018 | unsupported | unsupported | 0.172 ± 0.001 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 0.575 ± 0.014 | 0.635 ± 0.028 | unsupported | unsupported | 1.140 ± 0.089 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.006 ± 0.000 | 0.010 ± 0.000 | unsupported | 0.007 ± 0.000 | 0.027 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.062 ± 0.004 | 0.061 ± 0.002 | unsupported | 0.038 ± 0.002 | 0.167 ± 0.003 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.011 ± 0.000 | 0.016 ± 0.001 | unsupported | 0.007 ± 0.000 | 0.026 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.040 ± 0.002 | 0.054 ± 0.002 | unsupported | 0.025 ± 0.005 | 0.111 ± 0.016 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.014 ± 0.003 | 0.023 ± 0.003 | unsupported | 0.059 ± 0.025 | 0.121 ± 0.009 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.200 ± 0.006 | 0.192 ± 0.004 | unsupported | 0.542 ± 0.004 | 0.650 ± 0.014 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.033 ± 0.005 | 0.043 ± 0.009 | unsupported | 0.025 ± 0.001 | 0.099 ± 0.001 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.125 ± 0.012 | 0.163 ± 0.004 | unsupported | 0.142 ± 0.012 | 0.485 ± 0.023 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.014 ± 0.000 | 0.014 ± 0.001 | unsupported | 0.052 ± 0.000 | 0.105 ± 0.002 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.221 ± 0.003 | 0.234 ± 0.004 | unsupported | 0.509 ± 0.006 | 0.497 ± 0.008 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.014 ± 0.000 | 0.014 ± 0.000 | unsupported | 0.038 ± 0.001 | 0.079 ± 0.002 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.107 ± 0.012 | 0.110 ± 0.003 | unsupported | 0.191 ± 0.007 | 0.442 ± 0.003 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.015 ± 0.000 | 0.015 ± 0.000 | unsupported | 0.052 ± 0.000 | 0.110 ± 0.003 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.234 ± 0.002 | 0.240 ± 0.001 | unsupported | 0.513 ± 0.013 | 0.533 ± 0.004 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.014 ± 0.001 | 0.013 ± 0.000 | unsupported | 0.052 ± 0.001 | 0.080 ± 0.002 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.112 ± 0.002 | 0.116 ± 0.002 | unsupported | 0.198 ± 0.008 | 0.469 ± 0.010 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.043 ± 0.001 | 0.056 ± 0.001 | unsupported | 0.072 ± 0.001 | 0.162 ± 0.003 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 0.568 ± 0.011 | 0.578 ± 0.007 | unsupported | 0.455 ± 0.002 | 1.299 ± 0.012 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.064 ± 0.002 | 0.087 ± 0.002 | unsupported | 0.014 ± 0.000 | 0.144 ± 0.005 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 0.343 ± 0.006 | 0.377 ± 0.007 | unsupported | 0.162 ± 0.002 | 0.851 ± 0.045 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.064 ± 0.000 | 0.064 ± 0.000 | unsupported | 0.103 ± 0.002 | 0.198 ± 0.001 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 1.023 ± 0.002 | 1.018 ± 0.009 | unsupported | 1.314 ± 0.036 | 2.125 ± 0.015 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.031 ± 0.000 | 0.031 ± 0.000 | unsupported | 0.083 ± 0.001 | 0.128 ± 0.020 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 0.394 ± 0.002 | 0.395 ± 0.029 | unsupported | 0.486 ± 0.020 | 1.208 ± 0.005 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.390 ± 0.003 | 0.827 ± 0.003 | unsupported | 0.194 ± 0.127 | 0.106 ± 0.001 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 6.882 ± 0.133 | 10.005 ± 0.152 | unsupported | 2.769 ± 0.014 | 0.531 ± 0.007 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.286 ± 0.003 | 0.761 ± 0.008 | unsupported | 0.167 ± 0.001 | 0.081 ± 0.002 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 4.152 ± 0.229 | 7.715 ± 0.198 | unsupported | 1.342 ± 0.013 | 0.494 ± 0.004 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 18.073 ± 0.072 | - | unsupported | 18.067 ± 0.076 | - | 18.010 ± 0.185 | 32.965 ± 0.850 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 16.254 ± 0.127 | - | unsupported | 16.175 ± 0.021 | - | 18.216 ± 0.165 | 18.947 ± 0.440 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 12.305 ± 0.078 | - | unsupported | 12.673 ± 0.353 | - | 12.184 ± 0.044 | 12.505 ± 0.125 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 11.111 ± 0.147 | - | unsupported | 11.309 ± 0.065 | - | 12.468 ± 0.110 | 11.357 ± 0.217 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 11.115 ± 0.440 | - | unsupported | 12.826 ± 0.144 | - | 11.378 ± 0.103 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 11.323 ± 0.037 | - | unsupported | 11.303 ± 0.029 | - | 11.190 ± 0.115 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 18.136 ± 0.089 | - | unsupported | 18.371 ± 0.320 | - | 17.933 ± 0.040 | 32.649 ± 0.704 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 16.172 ± 0.071 | - | unsupported | 16.326 ± 0.022 | - | 18.193 ± 0.077 | 23.202 ± 11.942 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 37.360 ± 1.188 | - | unsupported | 36.566 ± 0.566 | - | 18.686 ± 0.195 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 17.363 ± 0.729 | - | unsupported | 17.984 ± 0.298 | - | 9.996 ± 0.265 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 37.386 ± 0.422 | - | unsupported | 35.600 ± 0.070 | - | 19.466 ± 0.111 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 17.213 ± 0.018 | - | unsupported | 17.793 ± 0.031 | - | 9.804 ± 0.025 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 18.124 ± 0.070 | - | unsupported | 18.140 ± 0.193 | - | 17.881 ± 0.098 | 32.228 ± 0.535 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 16.275 ± 0.031 | - | unsupported | 16.319 ± 0.041 | - | 18.061 ± 0.161 | 18.426 ± 0.522 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 12.290 ± 0.071 | - | unsupported | 12.573 ± 0.084 | - | 12.210 ± 0.046 | 25.978 ± 0.587 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 11.290 ± 0.063 | - | unsupported | 11.256 ± 0.110 | - | 12.531 ± 0.119 | 12.977 ± 0.749 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.430 ± 0.003 | unsupported | unsupported | 0.524 ± 0.002 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 6.934 ± 0.011 | unsupported | unsupported | 7.813 ± 0.207 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.273 ± 0.002 | unsupported | unsupported | 0.329 ± 0.002 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 3.453 ± 0.072 | unsupported | unsupported | 3.756 ± 0.069 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 18.097 ± 0.040 | - | unsupported | 18.114 ± 0.057 | - | 17.918 ± 0.149 | 32.498 ± 0.235 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 16.232 ± 0.013 | - | unsupported | 16.136 ± 0.058 | - | 18.196 ± 0.192 | 18.953 ± 0.536 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 15.101 ± 0.195 | - | 0.002 ± 0.000 | 14.253 ± 0.188 | 15.759 ± 0.301 | 17.795 ± 0.069 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 11.258 ± 0.069 | - | 0.002 ± 0.000 | 9.271 ± 0.218 | 10.123 ± 0.125 | 17.678 ± 0.196 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 13.320 ± 0.094 | - | 34.272 ± 0.246 | 13.495 ± 0.169 | 14.920 ± 0.226 | 12.973 ± 0.109 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 9.231 ± 0.158 | - | 23.207 ± 0.308 | 9.002 ± 0.080 | 9.663 ± 0.134 | 13.115 ± 0.367 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 53.422 ± 0.195 | - | 54.006 ± 2.942 | 26.692 ± 0.622 | 27.973 ± 0.540 | 27.711 ± 0.260 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 14.805 ± 0.252 | - | 15.189 ± 0.556 | 14.839 ± 0.146 | 16.320 ± 0.766 | 28.310 ± 0.141 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 11.123 ± 0.079 | - | 11.034 ± 0.096 | 16.167 ± 0.183 | 13.238 ± 0.459 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 7.665 ± 0.166 | - | 7.544 ± 0.217 | 9.277 ± 0.323 | 9.578 ± 0.020 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 21.909 ± 0.361 | - | 0.002 ± 0.000 | 19.037 ± 0.246 | 22.129 ± 0.422 | 22.758 ± 0.371 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 12.882 ± 0.189 | - | 0.002 ± 0.000 | 12.613 ± 0.309 | 13.444 ± 0.444 | 23.666 ± 0.521 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 30.596 ± 1.130 | - | 0.002 ± 0.000 | 16.777 ± 0.130 | 38.942 ± 2.073 | 275.581 ± 7.282 | 85.090 ± 1.174 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 12.655 ± 0.225 | - | 0.001 ± 0.000 | 16.489 ± 0.136 | 17.300 ± 2.526 | 277.136 ± 8.269 | 28.096 ± 4.405 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 10.489 ± 0.320 | - | 10.263 ± 0.348 | 11.781 ± 0.156 | 13.229 ± 0.324 | 14.162 ± 0.185 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 5.418 ± 0.143 | - | 5.492 ± 0.091 | 6.665 ± 0.023 | 8.080 ± 0.302 | 14.659 ± 0.557 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 10.211 ± 0.058 | - | 10.348 ± 0.172 | 13.420 ± 0.097 | 13.218 ± 0.592 | 14.281 ± 0.080 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 5.677 ± 0.165 | - | 5.902 ± 0.277 | 6.168 ± 0.077 | 7.984 ± 0.093 | 14.429 ± 0.554 | - |
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
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 3.432822 | 214551.38 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 1.720598 | 107537.38 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 1.909001 | 119312.56 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 2.523023 | 157688.94 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 2.547289 | 159205.56 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 2.930556 | 183159.75 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 8.153963 | 509622.69 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 9.103030 | 568939.38 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 3.454064 | 215879.00 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 3.372130 | 210758.12 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 6.794888 | 424680.50 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 6.242277 | 390142.31 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 4.986218 | 311638.62 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 5.082553 | 317659.56 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 74.894439 | 4680902.44 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 86.492661 | 5405791.31 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 78.387085 | 4899192.81 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 88.810419 | 5550651.19 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 10.098263 | 631141.44 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 4.647478 | 290467.38 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 3.162856 | 197678.50 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 3.438531 | 214908.19 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 68.923497 | 4307718.56 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 52.562578 | 3285161.12 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 57.317817 | 3582363.56 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 58.945634 | 3684102.12 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 11.512863 | 719553.94 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 11.660565 | 728785.31 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 221.783498 | 13861468.62 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 225.661972 | 14103873.25 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 12.115768 | 757235.50 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 12.317632 | 769852.00 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 190.571185 | 11910699.06 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 188.906840 | 11806677.50 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 10.722084 | 670130.25 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 10.889666 | 680604.12 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 212.011570 | 13250723.12 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 215.309215 | 13456825.94 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 11.394364 | 712147.75 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 11.256116 | 703507.25 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 176.940724 | 11058795.25 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 179.866389 | 11241649.31 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.920983 | 92515.36 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.277150 | 98080.47 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.714702 | 42417.22 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.693550 | 42086.72 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 2.674381 | 334297.62 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.195593 | 1274449.12 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.344477 | 668059.62 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.398448 | 674806.00 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 1.398269 | 21847.95 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.814086 | 43970.09 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.384496 | 37257.75 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.400801 | 37512.52 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.943117 | 367889.62 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.143606 | 517950.75 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.193703 | 274212.88 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.262897 | 282862.12 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.863097 | 138485.89 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.108358 | 64193.09 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.265744 | 19777.25 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.293915 | 20217.42 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 8.594633 | 1074329.12 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.067187 | 508398.37 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.521392 | 315174.00 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 2.598658 | 324832.25 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.735448 | 89616.38 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.640893 | 41263.95 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.962554 | 15039.91 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 0.977296 | 15270.25 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.597527 | 574690.88 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.810301 | 101287.62 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.247473 | 155934.12 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.314846 | 164355.75 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.168840 | 112013.12 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.223072 | 112860.50 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.535170 | 70862.03 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.528380 | 70755.94 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 4.998967 | 624870.88 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 12.033146 | 1504143.25 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.081818 | 1135227.25 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 9.250992 | 1156374.00 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.308033 | 82938.02 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 1.861189 | 29081.08 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.057988 | 32156.06 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.096185 | 32752.89 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.415269 | 426908.62 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.623826 | 577978.25 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.588509 | 448563.62 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.515379 | 439422.38 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.405114 | 115704.91 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.287946 | 98249.16 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.666917 | 26045.58 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.649935 | 41405.23 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.162873 | 645359.12 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.216162 | 902020.25 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.806250 | 350781.25 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 2.968004 | 371000.50 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.623296 | 103489.00 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 1.583457 | 24741.52 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.320524 | 51883.19 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6.473035 | 101141.17 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.045950 | 505743.75 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.819395 | 227424.38 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.946019 | 243252.38 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.291770 | 286471.25 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.113768 | 158027.62 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.924221 | 61315.95 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.763806 | 43184.47 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.471660 | 54244.69 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.739648 | 1342456.00 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 5.215313 | 651914.12 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.449902 | 556237.75 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.696643 | 587080.38 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.671442 | 151116.28 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.798844 | 12481.94 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.303118 | 67236.22 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.570011 | 87031.42 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.576763 | 947095.38 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.347337 | 168417.12 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.713047 | 339130.88 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.048303 | 381037.88 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.491506 | 163929.78 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.099475 | 64054.30 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.274521 | 82414.39 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.071053 | 1633881.62 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.452146 | 931518.25 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.475472 | 934434.00 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.999379 | 171865.30 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.841307 | 106895.42 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 9.075874 | 141810.53 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.121893 | 1140236.62 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.596982 | 574622.75 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.077167 | 634645.88 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 1.702139 | 26595.92 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.457609 | 7150.14 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.366231 | 5722.36 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 0.615565 | 9618.20 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 1.334026 | 166753.25 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.302696 | 37837.00 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.493582 | 61697.75 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.489647 | 61205.88 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 1.643608 | 25681.38 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.465027 | 7266.05 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.722371 | 11287.05 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 0.998859 | 15607.17 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 0.885064 | 110633.00 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.199293 | 24911.62 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.322797 | 40349.62 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 0.432373 | 54046.62 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.712075 | 120501.17 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.768658 | 58885.28 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.893173 | 13955.83 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.491513 | 23304.89 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.202452 | 650306.50 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.339576 | 542447.00 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 1.600557 | 200069.62 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 1.536751 | 192093.88 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.312197 | 98628.08 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 1.577487 | 24648.23 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.093060 | 32704.06 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.773773 | 43340.20 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.880962 | 485120.25 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.133733 | 141716.62 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.999279 | 124909.88 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.305652 | 163206.50 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.712994 | 104890.53 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.306883 | 51670.05 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.897780 | 14027.81 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 0.902197 | 14096.83 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 3.976509 | 497063.62 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.075409 | 509426.12 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 1.769471 | 221183.88 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 1.872646 | 234080.75 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.026839 | 78544.36 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.416873 | 37763.64 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.881626 | 13775.41 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 0.891101 | 13923.45 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.536142 | 442017.75 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.527161 | 190895.12 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.855592 | 106949.00 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 0.881221 | 110152.62 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.014328 | 109598.88 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.342337 | 52224.02 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.948997 | 14828.08 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 0.966774 | 15105.84 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 4.262229 | 532778.62 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.100877 | 512609.62 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 1.869080 | 233635.00 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 1.920338 | 240042.25 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.101261 | 79707.20 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.342710 | 52229.84 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.889133 | 13892.70 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 0.862953 | 13483.64 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.752938 | 469117.25 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.585258 | 198157.25 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.897115 | 112139.38 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 0.929263 | 116157.88 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.343143 | 161611.61 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.583442 | 71616.28 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.753411 | 43022.05 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.580704 | 55948.50 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.388700 | 1298587.50 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 3.640825 | 455103.12 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.545716 | 568214.50 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.624404 | 578050.50 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.245890 | 144467.03 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.925266 | 14457.28 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.098549 | 64039.83 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.573296 | 87082.75 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.811341 | 851417.62 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.297652 | 162206.50 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.744144 | 343018.00 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.017686 | 377210.75 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.651104 | 197673.50 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.623170 | 103487.03 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.108859 | 64200.92 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.092654 | 63947.72 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 17.002835 | 2125354.38 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.513931 | 1314241.38 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 8.182886 | 1022860.75 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 8.143807 | 1017975.88 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.206213 | 128222.08 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.338302 | 83410.97 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.988813 | 31075.20 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.002615 | 31290.86 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.665323 | 1208165.38 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.891183 | 486397.88 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.151358 | 393919.75 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.160903 | 395112.88 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.752903 | 105514.11 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 12.433097 | 194267.14 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 24.930137 | 389533.39 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 52.911633 | 826744.27 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 4.249940 | 531242.50 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 22.150951 | 2768868.88 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 55.059440 | 6882430.00 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 80.039086 | 10004885.75 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.214892 | 81482.69 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 10.657635 | 166525.55 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 18.333645 | 286463.20 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 48.694751 | 760855.48 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.949565 | 493695.62 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 10.735463 | 1341932.88 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 33.220004 | 4152500.50 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 61.718704 | 7714838.00 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 8.382806 | 523925.38 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 6.879306 | 429956.62 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 125.006921 | 7812932.56 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 110.942958 | 6933934.88 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 5.265601 | 329100.06 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 4.368939 | 273058.69 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 60.100613 | 3756288.31 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 55.245711 | 3452856.94 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.013460 | 841.25 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.001222 | 76.38 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.019480 | 1217.50 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.001212 | 75.75 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000110 | 6.88 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.012418 | 776.12 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.007381 | 461.31 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000220 | 13.75 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.018749 | 1171.81 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.007041 | 440.06 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000070 | 4.38 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.012769 | 798.06 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.005028 | 314.25 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000180 | 11.25 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.019480 | 1217.50 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.005078 | 317.38 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000060 | 3.75 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.012139 | 758.69 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003646 | 227.88 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000070 | 4.38 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.017507 | 1094.19 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003585 | 224.06 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 11.7x faster than the slowest successful cell (`tenferro-trace`, 9.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/pow` (f64, threads=4, shape=`4194304`): `pytorch-cpu` is 12.6x faster than the slowest successful cell (`julia-base`, 83.018 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 19.6x faster than the slowest successful cell (`pytorch-cpu`, 25.316 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 28.6x faster than the slowest successful cell (`pytorch-cpu`, 25.316 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 18.2x faster than the slowest successful cell (`pytorch-cpu`, 25.316 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 18.4x faster than the slowest successful cell (`pytorch-cpu`, 25.316 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 13.6x faster than the slowest successful cell (`pytorch-cpu`, 9.708 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 9.708 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 17.9x faster than the slowest successful cell (`pytorch-cpu`, 9.708 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 17.9x faster than the slowest successful cell (`pytorch-cpu`, 9.708 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 17.7x faster than the slowest successful cell (`pytorch-cpu`, 23.313 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 26.0x faster than the slowest successful cell (`pytorch-cpu`, 23.313 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 18.9x faster than the slowest successful cell (`pytorch-cpu`, 23.313 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 18.9x faster than the slowest successful cell (`pytorch-cpu`, 23.313 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 9.400 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 17.8x faster than the slowest successful cell (`pytorch-cpu`, 9.400 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 17.4x faster than the slowest successful cell (`pytorch-cpu`, 9.400 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sin` (f64, threads=1, shape=`8388608`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`jax-cpu`, 63.908 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sin` (f64, threads=1, shape=`8388608`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`jax-cpu`, 63.908 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sin` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 11.9x faster than the slowest successful cell (`jax-cpu`, 31.608 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sin` (f64, threads=4, shape=`8388608`): `tenferro-trace` is 11.7x faster than the slowest successful cell (`jax-cpu`, 31.608 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `tenferro-direct` is 10.4x faster than the slowest successful cell (`julia-base`, 1.714 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 10.3x faster than the slowest successful cell (`julia-base`, 1.912 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`julia-base`, 1.912 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 1353.2x faster than the slowest successful cell (`julia-base`, 2.168 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 1552.5x faster than the slowest successful cell (`julia-base`, 2.582 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 18.5x faster than the slowest successful cell (`jax-cpu`, 0.023 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 13.2x faster than the slowest successful cell (`jax-cpu`, 0.023 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x2x2,rhs=1`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 12.1x faster than the slowest successful cell (`jax-cpu`, 0.151 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 18.8x faster than the slowest successful cell (`tenferro-eager`, 10.005 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 15.6x faster than the slowest successful cell (`tenferro-eager`, 7.715 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 10511.0x faster than the slowest successful cell (`julia-base`, 17.795 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 11464.3x faster than the slowest successful cell (`julia-base`, 17.678 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 13216.1x faster than the slowest successful cell (`julia-base`, 22.758 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 14671.8x faster than the slowest successful cell (`julia-base`, 23.666 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `pytorch-cpu` is 16.4x faster than the slowest successful cell (`julia-base`, 275.581 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 175194.4x faster than the slowest successful cell (`julia-base`, 275.581 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `jax-cpu` is 16.0x faster than the slowest successful cell (`julia-base`, 277.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `pytorch-cpu` is 16.8x faster than the slowest successful cell (`julia-base`, 277.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-direct` is 21.9x faster than the slowest successful cell (`julia-base`, 277.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 192055.2x faster than the slowest successful cell (`julia-base`, 277.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 16.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 110.9x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 83.7x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 199.5x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 110.6x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 189.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 273.5x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 10467.6 GB/s at 0.002 ms. The conservative physical lower bound is 0.168 ms, so the cell is 104.7x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 10083.6 GB/s at 0.002 ms. The conservative physical lower bound is 0.042 ms, so the cell is 25.2x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 158556.1 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1585.6x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 174082.7 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 435.2x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 155885.9 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1558.9x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 166420.0 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 416.0x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 85326.0 GB/s at 0.002 ms. The conservative physical lower bound is 1.342 ms, so the cell is 853.3x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 93013.0 GB/s at 0.001 ms. The conservative physical lower bound is 0.336 ms, so the cell is 232.5x faster than the bound.

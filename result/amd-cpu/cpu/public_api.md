# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/amd-cpu/cpu/public_api/20261107_191536/run.yaml`
- Timestamp: `2026-11-07T19:15:36.094222+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/amd-cpu/cpu/public_api/20261107_191536`.

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

- CSV: `data/results/amd-cpu/cpu/public_api/20261107_191536/cpu_public_api_t1_20261107_191536.csv`
- CSV: `data/results/amd-cpu/cpu/public_api/20261107_191536/cpu_public_api_t4_20261107_191536.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 4.223 ± 0.045 | - | 4.141 ± 0.051 | 4.517 ± 0.007 | 3.929 ± 0.005 | 3.678 ± 0.031 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 4.485 ± 0.038 | - | 4.269 ± 0.024 | 2.322 ± 0.029 | 3.947 ± 1.000 | 3.169 ± 0.534 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 122.495 ± 1.978 | - | 120.695 ± 3.046 | 121.324 ± 1.074 | 125.451 ± 0.591 | 141.513 ± 2.192 | 143.076 ± 1.867 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 40.351 ± 0.247 | - | 143.746 ± 2.723 | 39.819 ± 1.175 | 62.964 ± 0.932 | 140.688 ± 1.829 | 60.880 ± 0.445 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 106.753 ± 0.094 | - | 110.015 ± 1.358 | 91.466 ± 0.487 | 88.335 ± 0.539 | 124.320 ± 1.404 | 136.212 ± 2.097 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 42.580 ± 0.284 | - | 131.255 ± 0.585 | 30.313 ± 0.259 | 45.261 ± 0.351 | 124.245 ± 0.861 | 61.775 ± 1.467 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 43.361 ± 0.040 | - | 43.573 ± 0.375 | 42.159 ± 0.193 | 66.003 ± 0.616 | 46.449 ± 0.268 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 44.123 ± 0.098 | - | 44.232 ± 0.181 | 29.157 ± 1.060 | 29.379 ± 0.570 | 23.239 ± 0.347 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 44.466 ± 0.240 | - | unsupported | 42.462 ± 0.223 | 66.360 ± 0.245 | 44.596 ± 4.780 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 45.063 ± 0.218 | - | unsupported | 29.324 ± 0.018 | 30.016 ± 0.807 | 23.157 ± 4.411 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 10.883 ± 0.034 | - | 10.921 ± 0.049 | 11.515 ± 0.021 | 11.922 ± 0.065 | 11.833 ± 0.061 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 14.167 ± 0.040 | - | 14.220 ± 0.078 | 14.016 ± 0.070 | 17.511 ± 0.057 | 17.660 ± 0.053 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 71.903 ± 0.078 | - | 72.378 ± 0.343 | 106.637 ± 1.081 | 103.728 ± 1.523 | 82.303 ± 3.916 | 83.128 ± 9.160 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 30.922 ± 0.149 | - | 83.996 ± 23.997 | 42.978 ± 0.866 | 52.377 ± 0.770 | 82.862 ± 0.171 | 36.679 ± 5.311 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 120.166 ± 0.181 | - | 115.764 ± 0.147 | 152.651 ± 0.454 | 177.004 ± 2.366 | 147.063 ± 0.573 | 156.719 ± 0.530 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 49.576 ± 0.322 | - | 132.003 ± 12.002 | 63.007 ± 0.243 | 72.777 ± 0.973 | 150.426 ± 0.657 | 62.498 ± 0.353 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 70.383 ± 0.444 | - | 70.737 ± 1.498 | 67.188 ± 1.325 | 69.735 ± 1.429 | 75.007 ± 0.576 | 76.152 ± 0.156 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 24.243 ± 0.580 | - | 82.103 ± 10.850 | 22.775 ± 0.256 | 35.399 ± 0.553 | 74.998 ± 0.571 | 33.996 ± 0.816 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 2.212 ± 0.069 | - | 20.705 ± 0.096 | 24.158 ± 0.063 | 15.604 ± 0.330 | 4.196 ± 0.025 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 0.839 ± 0.030 | - | 59.985 ± 11.988 | 11.221 ± 0.021 | 2.198 ± 0.263 | 4.068 ± 0.031 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 8.563 ± 0.019 | - | 8.515 ± 0.084 | 9.888 ± 0.027 | 7.599 ± 0.081 | 8.959 ± 0.123 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 127.877 ± 12.085 | - | 119.030 ± 12.014 | 5.772 ± 0.027 | 7.004 ± 0.179 | 7.533 ± 0.196 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 5.494 ± 0.161 | - | 5.714 ± 0.286 | 5.788 ± 0.069 | 5.123 ± 0.017 | 4.823 ± 0.045 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 100.905 ± 10.097 | - | 133.975 ± 4.980 | 3.603 ± 0.080 | 4.604 ± 0.066 | 4.254 ± 0.045 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 6.520 ± 0.059 | - | 6.554 ± 0.114 | 7.602 ± 0.041 | 5.498 ± 0.018 | 5.272 ± 0.075 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 11.537 ± 0.187 | - | 11.578 ± 0.233 | 5.861 ± 0.036 | 6.103 ± 0.054 | 6.295 ± 0.071 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 43.398 ± 0.238 | - | 45.645 ± 2.291 | 42.155 ± 0.097 | 65.983 ± 0.288 | 44.090 ± 0.255 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 44.286 ± 0.328 | - | 46.658 ± 2.558 | 29.071 ± 0.072 | 29.627 ± 0.377 | 22.366 ± 0.183 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 49.212 ± 0.155 | - | unsupported | 49.735 ± 0.237 | 65.011 ± 4.153 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 52.789 ± 0.451 | - | unsupported | 25.149 ± 0.139 | 34.794 ± 0.317 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 120.669 ± 2.592 | - | 121.730 ± 3.854 | 119.741 ± 0.397 | 123.830 ± 0.352 | 136.308 ± 3.715 | 147.508 ± 3.847 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 40.259 ± 0.224 | - | 143.809 ± 9.188 | 40.137 ± 0.919 | 61.964 ± 1.055 | 136.264 ± 1.370 | 66.132 ± 1.246 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 130.301 ± 4.130 | - | 129.865 ± 4.528 | 129.587 ± 2.484 | 134.809 ± 0.659 | 141.334 ± 4.459 | 161.546 ± 1.771 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 44.257 ± 0.560 | - | 138.737 ± 3.216 | 44.331 ± 0.152 | 84.900 ± 2.691 | 140.955 ± 4.126 | 70.268 ± 1.117 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 27.128 ± 0.160 | - | 59.233 ± 0.228 | 74.483 ± 3.642 | 51.363 ± 0.501 | 80.211 ± 8.855 | 85.198 ± 0.144 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 12.509 ± 0.204 | - | 119.999 ± 0.011 | 32.487 ± 0.964 | 25.768 ± 0.059 | 79.584 ± 0.097 | 32.136 ± 11.517 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 9.293 ± 0.071 | - | 9.925 ± 0.131 | 34.689 ± 0.691 | 36.642 ± 0.771 | 42.977 ± 9.324 | 40.070 ± 12.527 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 7.115 ± 0.100 | - | 23.997 ± 0.003 | 12.228 ± 0.096 | 19.390 ± 0.219 | 35.071 ± 9.008 | 14.700 ± 11.688 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 21.484 ± 0.119 | - | 21.344 ± 0.464 | 42.334 ± 0.155 | 37.492 ± 0.168 | 22.286 ± 0.167 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 13.303 ± 0.079 | - | 36.001 ± 0.005 | 19.223 ± 0.116 | 22.834 ± 0.134 | 19.400 ± 0.372 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 28.249 ± 0.028 | - | 28.532 ± 0.045 | 55.424 ± 0.425 | 107.039 ± 0.709 | 70.187 ± 11.977 | 76.433 ± 12.120 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 11.928 ± 0.042 | - | 47.998 ± 0.010 | 21.967 ± 0.501 | 54.148 ± 0.353 | 67.082 ± 18.934 | 30.886 ± 11.068 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 130.404 ± 3.672 | - | 127.900 ± 4.345 | 128.188 ± 2.358 | 132.317 ± 0.529 | 146.981 ± 4.751 | 158.988 ± 4.829 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 44.227 ± 0.145 | - | 141.717 ± 0.719 | 44.437 ± 2.931 | 68.745 ± 1.250 | 146.081 ± 3.002 | 70.400 ± 0.528 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 20.427 ± 0.050 | - | 20.442 ± 0.048 | 43.758 ± 0.443 | 40.143 ± 0.555 | 68.213 ± 11.673 | 79.352 ± 12.344 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 8.397 ± 0.022 | - | 35.999 ± 0.007 | 16.565 ± 0.182 | 21.165 ± 0.493 | 61.908 ± 9.550 | 30.020 ± 11.389 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 20.368 ± 0.025 | - | 20.634 ± 0.389 | 31.824 ± 0.376 | 27.742 ± 0.150 | 37.275 ± 9.075 | 40.316 ± 0.186 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 8.098 ± 0.123 | - | 36.000 ± 0.013 | 11.409 ± 0.307 | 13.836 ± 0.153 | 37.482 ± 0.371 | 16.385 ± 10.132 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 25.382 ± 0.040 | - | 25.573 ± 0.041 | 49.579 ± 0.521 | 64.430 ± 0.086 | 82.551 ± 21.349 | 88.496 ± 8.009 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 10.942 ± 0.013 | - | 47.998 ± 0.020 | 19.265 ± 0.440 | 32.535 ± 0.125 | 83.155 ± 8.925 | 39.686 ± 16.857 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 13.242 ± 0.027 | - | 13.325 ± 0.042 | 32.099 ± 0.478 | 36.649 ± 0.421 | 45.784 ± 8.108 | 49.443 ± 0.500 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 5.639 ± 0.017 | - | 36.000 ± 0.006 | 11.794 ± 0.197 | 18.548 ± 0.166 | 45.639 ± 0.320 | 20.284 ± 10.114 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.443 ± 0.009 | 0.481 ± 0.009 | unsupported | unsupported | 0.477 ± 0.010 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.383 ± 0.003 | 52.000 ± 0.000 | unsupported | unsupported | 0.416 ± 0.051 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 135.389 ± 3.055 | - | 133.320 ± 4.421 | 128.427 ± 0.163 | 134.374 ± 8.630 | 150.882 ± 3.705 | 164.123 ± 3.981 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 44.612 ± 0.177 | - | 154.090 ± 11.854 | 43.905 ± 0.781 | 68.195 ± 0.722 | 148.975 ± 0.615 | 71.131 ± 0.826 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 134.945 ± 3.102 | - | 135.589 ± 3.477 | 127.933 ± 0.809 | 133.237 ± 0.439 | 151.307 ± 3.923 | 163.216 ± 4.520 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 44.678 ± 0.175 | - | 141.773 ± 0.713 | 43.895 ± 0.704 | 68.266 ± 0.458 | 146.584 ± 1.755 | 70.393 ± 3.081 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 127.607 ± 3.406 | - | 130.097 ± 4.345 | 129.546 ± 2.854 | 136.318 ± 2.030 | 144.676 ± 1.534 | 164.267 ± 4.558 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 44.659 ± 0.400 | - | 141.863 ± 8.763 | 43.923 ± 0.497 | 69.676 ± 0.309 | 144.404 ± 2.920 | 69.932 ± 0.640 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 119.898 ± 2.735 | - | 122.022 ± 3.148 | 118.587 ± 0.405 | 123.858 ± 0.136 | 135.790 ± 3.811 | 147.844 ± 3.917 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 40.201 ± 0.293 | - | 141.484 ± 11.714 | 39.878 ± 0.676 | 62.406 ± 0.449 | 136.336 ± 0.901 | 66.507 ± 0.554 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 58.116 ± 0.124 | - | 58.801 ± 0.136 | 62.523 ± 0.292 | 73.691 ± 0.108 | 125.015 ± 10.110 | 121.737 ± 0.587 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 23.596 ± 0.198 | - | 71.997 ± 0.008 | 22.060 ± 0.252 | 37.066 ± 0.133 | 125.034 ± 0.553 | 77.686 ± 7.581 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 15.943 ± 0.142 | - | 15.928 ± 0.120 | 11.944 ± 0.778 | 25.312 ± 0.150 | 12.851 ± 0.138 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 6.600 ± 0.240 | - | 35.997 ± 0.003 | 6.141 ± 0.199 | 13.505 ± 0.112 | 12.843 ± 0.187 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.827 ± 0.069 | - | 1.840 ± 0.125 | 23.751 ± 0.104 | 2.442 ± 0.034 | 1.193 ± 0.114 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.779 ± 0.024 | - | 23.998 ± 0.021 | 11.444 ± 0.282 | 1.524 ± 0.244 | 1.255 ± 0.074 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 2.412 ± 0.058 | - | 2.470 ± 0.042 | 4.032 ± 0.038 | 2.316 ± 0.150 | 1.185 ± 0.157 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 1.289 ± 0.117 | - | 23.994 ± 0.019 | 2.304 ± 0.045 | 1.370 ± 0.198 | 1.223 ± 0.133 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 15.862 ± 0.132 | - | 15.868 ± 0.172 | 11.677 ± 0.798 | 25.247 ± 1.279 | 12.725 ± 0.115 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 6.538 ± 0.104 | - | 35.999 ± 0.011 | 6.175 ± 0.270 | 13.491 ± 0.518 | 12.847 ± 0.305 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.809 ± 0.043 | - | 1.848 ± 0.080 | 22.681 ± 0.218 | 2.421 ± 0.023 | 1.279 ± 0.177 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.788 ± 0.039 | - | 23.999 ± 0.003 | 10.770 ± 0.161 | 1.340 ± 0.010 | 1.069 ± 0.076 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 10.162 ± 0.094 | - | 10.096 ± 0.107 | 16.156 ± 0.021 | 11.704 ± 0.300 | 6.252 ± 0.089 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 5.098 ± 0.192 | - | 36.000 ± 0.023 | 8.814 ± 0.062 | 6.423 ± 0.199 | 6.306 ± 0.022 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 10.441 ± 0.092 | - | 10.528 ± 0.155 | 10.394 ± 0.071 | 19.475 ± 0.425 | 10.381 ± 0.068 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 6.109 ± 0.159 | - | 23.999 ± 0.006 | 6.079 ± 0.116 | 10.512 ± 0.110 | 10.821 ± 0.109 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.966 ± 0.019 | - | 1.019 ± 0.114 | 1.968 ± 0.027 | 1.975 ± 0.015 | 0.980 ± 0.122 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.458 ± 0.072 | - | 23.998 ± 0.007 | 0.934 ± 0.039 | 1.117 ± 0.023 | 0.923 ± 0.146 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.145 ± 0.026 | - | 1.167 ± 0.072 | 0.764 ± 0.015 | 2.835 ± 0.271 | 1.125 ± 0.072 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.595 ± 0.052 | - | 23.998 ± 0.006 | 0.338 ± 0.049 | 1.686 ± 0.151 | 1.106 ± 0.037 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 10.514 ± 0.096 | - | 10.464 ± 0.286 | 10.560 ± 0.097 | 19.590 ± 0.056 | 10.341 ± 0.084 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 6.071 ± 0.085 | - | 23.999 ± 0.004 | 6.075 ± 0.054 | 10.647 ± 0.268 | 10.518 ± 0.236 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.993 ± 0.137 | - | 1.031 ± 0.074 | 1.940 ± 0.090 | 1.979 ± 0.024 | 0.909 ± 0.088 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.416 ± 0.054 | - | 23.998 ± 0.010 | 0.933 ± 0.043 | 1.121 ± 0.016 | 0.895 ± 0.096 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.070 ± 0.049 | - | 1.129 ± 0.132 | 0.767 ± 0.015 | 3.034 ± 0.337 | 0.970 ± 0.065 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.617 ± 0.065 | - | 24.001 ± 0.008 | 0.414 ± 0.066 | 1.576 ± 0.155 | 0.991 ± 0.287 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 1.051 ± 0.078 | 1.035 ± 0.062 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.361 ± 0.023 | 23.999 ± 0.001 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 36.631 ± 0.142 | - | 37.955 ± 0.110 | 54.326 ± 0.732 | 69.657 ± 0.737 | 73.293 ± 6.257 | 89.735 ± 9.127 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 17.685 ± 0.164 | - | 59.996 ± 0.007 | 20.492 ± 0.153 | 35.923 ± 4.714 | 69.273 ± 8.765 | 34.182 ± 11.081 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 140.293 ± 0.808 | - | 140.896 ± 3.669 | 144.399 ± 1.622 | 245.134 ± 2.717 | 261.304 ± 3.442 | 262.792 ± 2.552 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 46.559 ± 0.601 | - | 153.166 ± 2.877 | 46.080 ± 0.728 | 124.189 ± 0.613 | 259.882 ± 3.226 | 107.747 ± 2.805 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 132.097 ± 4.223 | - | 134.736 ± 4.613 | 136.407 ± 2.366 | 151.268 ± 1.394 | 150.786 ± 3.420 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 46.979 ± 0.378 | - | 153.924 ± 12.777 | 51.366 ± 1.250 | 78.610 ± 2.259 | 149.698 ± 1.549 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 123.314 ± 3.131 | - | 125.932 ± 3.347 | 124.014 ± 0.496 | 126.258 ± 1.087 | 138.866 ± 3.673 | 145.052 ± 3.718 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 40.399 ± 0.123 | - | 140.867 ± 2.617 | 40.270 ± 0.064 | 64.569 ± 1.011 | 138.751 ± 2.080 | 63.642 ± 1.890 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 24.403 ± 0.073 | - | 24.851 ± 0.054 | 56.806 ± 0.392 | 106.254 ± 0.282 | 69.171 ± 8.472 | 76.058 ± 13.202 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 10.226 ± 0.036 | - | 47.997 ± 0.003 | 22.002 ± 0.348 | 53.735 ± 0.271 | 66.250 ± 13.338 | 29.279 ± 15.157 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 128.832 ± 0.375 | - | 130.130 ± 2.466 | 187.462 ± 1.664 | 133.206 ± 3.311 | 212.508 ± 2.235 | 217.744 ± 3.667 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 43.072 ± 0.622 | - | 139.277 ± 4.847 | 67.666 ± 0.931 | 67.810 ± 1.394 | 211.696 ± 2.284 | 89.833 ± 0.746 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 127.290 ± 4.154 | - | 130.301 ± 4.271 | 130.092 ± 3.130 | 135.447 ± 0.229 | 142.604 ± 4.062 | 161.513 ± 1.702 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 44.785 ± 0.220 | - | 141.339 ± 2.926 | 43.976 ± 0.594 | 68.138 ± 0.721 | 146.158 ± 0.605 | 69.344 ± 0.795 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 51.333 ± 0.104 | - | 51.783 ± 0.080 | 75.351 ± 0.471 | 38.285 ± 0.360 | 84.121 ± 12.865 | 91.625 ± 17.771 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 20.482 ± 0.059 | - | 71.998 ± 0.008 | 28.296 ± 0.633 | 20.030 ± 0.468 | 84.750 ± 9.927 | 37.310 ± 11.084 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 132.613 ± 4.120 | 127.716 ± 4.215 | unsupported | 138.708 ± 2.119 | 148.984 ± 2.261 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 46.253 ± 0.139 | 142.275 ± 6.187 | unsupported | 51.843 ± 0.767 | 77.807 ± 2.120 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.753 ± 0.014 | - | 0.810 ± 0.091 | 0.803 ± 0.021 | 0.871 ± 0.073 | 7.277 ± 3.853 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.299 ± 0.067 | - | 11.999 ± 0.002 | 0.318 ± 0.034 | 0.847 ± 0.146 | 1.487 ± 5.839 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.840 ± 0.231 | - | 0.859 ± 0.158 | 0.780 ± 0.091 | 0.928 ± 0.273 | 7.458 ± 3.897 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.376 ± 0.171 | - | 12.004 ± 0.019 | 0.337 ± 0.035 | 0.746 ± 0.045 | 7.568 ± 3.892 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.144 ± 0.047 | - | unsupported | 1.272 ± 0.165 | 1.175 ± 0.096 | 4.416 ± 6.164 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.503 ± 0.043 | - | unsupported | 0.598 ± 0.061 | 0.867 ± 0.124 | 7.644 ± 3.723 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.397 ± 0.023 | - | 0.476 ± 0.044 | 0.689 ± 0.010 | 0.447 ± 0.025 | 1.174 ± 0.084 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.394 ± 0.066 | - | 47.998 ± 0.004 | 0.348 ± 0.013 | 0.421 ± 0.018 | 1.231 ± 0.077 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.913 ± 0.289 | - | 0.762 ± 0.038 | 1.141 ± 0.158 | 1.001 ± 0.075 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.399 ± 0.144 | - | 11.998 ± 0.005 | 0.497 ± 0.033 | 0.602 ± 0.032 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.807 ± 0.055 | - | 0.794 ± 0.060 | 0.756 ± 0.043 | 0.789 ± 0.100 | 1.433 ± 6.922 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.311 ± 0.072 | - | 11.999 ± 0.005 | 0.337 ± 0.117 | 0.461 ± 0.072 | 7.353 ± 6.938 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.784 ± 0.020 | - | 0.901 ± 0.010 | 0.470 ± 0.061 | 0.464 ± 0.024 | 0.505 ± 0.007 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 1.054 ± 0.019 | - | 35.994 ± 0.161 | 0.629 ± 0.009 | 0.491 ± 0.035 | 1.327 ± 0.084 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.629 ± 0.021 | - | 0.002 ± 0.000 | 1.660 ± 0.035 | 1.419 ± 0.108 | 8.726 ± 4.643 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.655 ± 0.051 | - | 0.002 ± 0.000 | 0.821 ± 0.158 | 1.070 ± 0.043 | 8.815 ± 4.763 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.050 ± 0.000 | - | 0.052 ± 0.000 | 0.076 ± 0.002 | 0.095 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.093 ± 0.000 | - | 0.095 ± 0.000 | 0.126 ± 0.001 | 0.149 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.236 ± 0.016 | - | 0.238 ± 0.010 | 0.286 ± 0.034 | 0.316 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.081 ± 0.003 | - | 0.086 ± 0.004 | 0.148 ± 0.013 | 0.137 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.042 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.014 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.031 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.673 ± 0.036 | - | 0.671 ± 0.005 | 1.155 ± 0.023 | 0.670 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.026 ± 0.001 | - | 0.028 ± 0.000 | 0.039 ± 0.009 | 0.062 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.032 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.050 ± 0.000 | - | 0.053 ± 0.000 | 0.085 ± 0.003 | 0.105 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.093 ± 0.002 | - | 0.095 ± 0.000 | 0.123 ± 0.002 | 0.164 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.235 ± 0.001 | - | 0.237 ± 0.002 | 0.339 ± 0.012 | 0.348 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.083 ± 0.003 | - | 0.085 ± 0.003 | 0.180 ± 0.010 | 0.120 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.010 ± 0.001 | 0.015 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.016 ± 0.001 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.011 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.686 ± 0.025 | - | 0.694 ± 4.458 | 1.369 ± 0.028 | 1.164 ± 0.889 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.026 ± 0.000 | - | 0.028 ± 0.000 | 0.059 ± 0.001 | 0.067 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.012 ± 0.000 | 0.016 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.616 ± 0.009 | - | 0.619 ± 0.011 | 0.721 ± 0.036 | 0.882 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 2.115 ± 0.005 | - | 2.118 ± 0.003 | 2.144 ± 0.005 | 2.693 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 6.466 ± 0.021 | - | 6.487 ± 0.014 | 6.508 ± 0.004 | 8.935 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 3.658 ± 0.048 | - | 3.660 ± 0.059 | 3.229 ± 0.019 | 2.681 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.001 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.109 ± 0.001 | - | 0.111 ± 0.002 | 0.121 ± 0.012 | 0.195 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.031 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.017 ± 0.001 | 0.044 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.012 ± 0.002 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.008 ± 0.001 | - | 0.010 ± 0.011 | 0.015 ± 0.001 | 0.039 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.020 ± 0.000 | - | 0.023 ± 0.004 | 0.029 ± 0.001 | 0.096 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.031 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.018 ± 0.001 | 0.034 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.027 ± 0.000 | - | 0.029 ± 0.000 | 0.035 ± 0.001 | 0.105 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 29.179 ± 0.166 | - | 29.261 ± 0.184 | 25.540 ± 0.087 | 22.201 ± 0.133 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.006 ± 0.001 | - | 0.008 ± 0.000 | 0.015 ± 0.000 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.863 ± 0.003 | - | 0.867 ± 0.003 | 0.874 ± 0.009 | 1.010 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.018 ± 0.000 | - | 0.020 ± 0.000 | 0.026 ± 0.001 | 0.055 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.052 ± 0.000 | - | 0.054 ± 0.001 | 0.061 ± 0.002 | 0.110 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.622 ± 0.030 | - | 0.627 ± 4.462 | 0.724 ± 0.424 | 0.916 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 2.152 ± 0.034 | - | 2.165 ± 0.027 | 2.132 ± 0.014 | 2.246 ± 0.194 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 6.531 ± 0.029 | - | 6.513 ± 0.086 | 9.295 ± 0.015 | 5.817 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 6.669 ± 0.028 | - | 6.677 ± 0.035 | 2.278 ± 0.024 | 3.386 ± 0.084 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.014 ± 0.004 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.110 ± 0.001 | - | 0.113 ± 0.002 | 0.177 ± 0.008 | 0.174 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.001 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.016 ± 0.002 | 0.032 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.016 ± 0.004 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.016 ± 0.001 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.021 ± 0.000 | - | 0.023 ± 0.000 | 0.029 ± 0.001 | 0.062 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.010 ± 0.000 | - | 0.013 ± 0.002 | 0.018 ± 0.001 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.027 ± 0.000 | - | 0.030 ± 0.000 | 0.035 ± 0.003 | 0.063 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 53.388 ± 0.185 | - | 53.483 ± 0.159 | 17.808 ± 0.174 | 28.133 ± 3.911 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.022 ± 0.001 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.867 ± 0.017 | - | 0.887 ± 4.427 | 1.251 ± 0.022 | 1.908 ± 0.389 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.026 ± 0.001 | 0.046 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.052 ± 0.000 | - | 0.055 ± 0.000 | 0.061 ± 0.002 | 0.117 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.671 ± 0.007 | - | 0.672 ± 0.005 | 0.522 ± 0.010 | 0.444 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 2.209 ± 0.008 | - | 2.211 ± 0.008 | 1.951 ± 0.005 | 1.949 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 5.801 ± 0.004 | - | 5.802 ± 0.025 | 5.702 ± 0.017 | 6.467 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.333 ± 0.062 | - | 1.372 ± 0.080 | 1.333 ± 0.010 | 1.454 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.071 ± 0.003 | - | 0.074 ± 0.007 | 0.089 ± 0.003 | 0.162 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.015 ± 0.001 | 0.028 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.015 ± 0.001 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.019 ± 0.001 | - | 0.021 ± 0.000 | 0.027 ± 0.002 | 0.041 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.009 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.018 ± 0.006 | 0.036 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.024 ± 0.000 | - | 0.027 ± 0.002 | 0.032 ± 0.001 | 0.059 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 10.373 ± 0.041 | - | 10.375 ± 0.068 | 10.448 ± 0.047 | 11.299 ± 0.096 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.552 ± 0.013 | - | 0.557 ± 0.012 | 0.626 ± 0.012 | 0.781 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.024 ± 0.001 | 0.046 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.046 ± 0.000 | - | 0.049 ± 0.002 | 0.054 ± 0.001 | 0.122 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.669 ± 0.022 | - | 0.672 ± 4.466 | 0.795 ± 0.089 | 0.445 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 2.211 ± 0.006 | - | 2.223 ± 0.027 | 1.948 ± 0.005 | 1.277 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 5.853 ± 0.012 | - | 5.847 ± 0.031 | 8.494 ± 0.012 | 3.658 ± 0.571 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 3.127 ± 0.029 | - | 3.127 ± 0.013 | 1.190 ± 0.035 | 1.910 ± 0.213 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.012 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.167 ± 0.036 | - | 0.155 ± 0.019 | 0.134 ± 0.004 | 0.150 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.015 ± 0.001 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.014 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.015 ± 0.000 | 0.026 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.019 ± 0.000 | - | 0.022 ± 0.000 | 0.027 ± 0.002 | 0.044 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.016 ± 0.004 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.011 ± 0.000 | - | 0.014 ± 0.000 | 0.017 ± 0.002 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.025 ± 0.000 | - | 0.028 ± 0.000 | 0.032 ± 0.000 | 0.051 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 25.087 ± 0.105 | - | 24.902 ± 0.323 | 9.186 ± 0.102 | 13.988 ± 0.123 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.001 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 1.102 ± 0.024 | - | 1.119 ± 0.030 | 0.926 ± 0.003 | 0.476 ± 0.078 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.019 ± 0.000 | - | 0.022 ± 0.000 | 0.024 ± 0.001 | 0.036 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.047 ± 0.000 | - | 0.050 ± 0.000 | 0.053 ± 0.002 | 0.106 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.273 ± 0.014 | - | 0.280 ± 0.011 | 0.240 ± 0.033 | 0.436 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 1.101 ± 0.005 | - | 1.106 ± 0.029 | 0.959 ± 0.003 | 1.950 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 3.104 ± 0.026 | - | 3.116 ± 0.015 | 2.880 ± 0.009 | 6.445 ± 0.062 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.512 ± 0.008 | - | 0.519 ± 0.011 | 0.546 ± 0.010 | 1.490 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.031 ± 0.000 | - | 0.033 ± 0.000 | 0.040 ± 0.001 | 0.125 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.011 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.001 | 0.026 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.020 ± 0.001 | 0.035 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.011 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.014 ± 0.001 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.022 ± 0.001 | 0.040 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 4.073 ± 0.026 | - | 4.077 ± 0.008 | 4.261 ± 0.017 | 11.135 ± 0.084 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.012 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.231 ± 0.025 | - | 0.234 ± 0.010 | 0.250 ± 0.019 | 0.734 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.010 ± 0.000 | - | 0.012 ± 0.001 | 0.018 ± 0.001 | 0.036 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.025 ± 0.001 | - | 0.027 ± 0.000 | 0.033 ± 0.005 | 0.088 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.279 ± 0.042 | - | 0.280 ± 0.012 | 0.396 ± 0.011 | 0.456 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 1.107 ± 0.023 | - | 1.110 ± 0.022 | 0.946 ± 0.003 | 1.274 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 3.130 ± 0.019 | - | 3.137 ± 0.011 | 3.650 ± 0.011 | 3.633 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 1.949 ± 0.005 | - | 1.974 ± 0.014 | 0.695 ± 0.015 | 1.748 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.014 ± 0.004 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.031 ± 0.000 | - | 0.034 ± 0.000 | 0.054 ± 0.002 | 0.142 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.015 ± 0.002 | 0.018 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.001 | 0.021 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.015 ± 0.001 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.019 ± 0.002 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.019 ± 0.001 | 0.034 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.001 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.019 ± 0.001 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.022 ± 0.003 | 0.042 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 15.806 ± 0.090 | - | 15.745 ± 0.017 | 5.218 ± 0.019 | 13.985 ± 1.435 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.017 ± 0.002 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.233 ± 0.001 | - | 0.238 ± 0.011 | 0.319 ± 0.014 | 0.451 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.024 ± 0.003 | 0.039 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.026 ± 0.000 | - | 0.028 ± 0.000 | 0.033 ± 0.001 | 0.089 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.152 ± 0.001 | - | 0.153 ± 0.001 | 0.189 ± 0.002 | 0.133 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.210 ± 0.011 | - | 0.214 ± 0.011 | 0.255 ± 0.015 | 0.209 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.503 ± 0.010 | - | 0.503 ± 0.012 | 0.556 ± 0.164 | 0.582 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.136 ± 0.004 | - | 0.141 ± 0.003 | 0.175 ± 0.004 | 0.207 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.014 ± 0.001 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.012 ± 0.001 | - | 0.014 ± 0.000 | 0.026 ± 0.002 | 0.053 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.014 ± 0.001 | 0.022 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.014 ± 0.000 | - | 0.016 ± 0.001 | 0.028 ± 0.001 | 0.056 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.014 ± 0.003 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.001 | 0.020 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.018 ± 0.000 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.018 ± 0.001 | 0.045 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.017 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.017 ± 0.001 | 0.024 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.019 ± 0.000 | 0.045 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 1.052 ± 0.008 | - | 1.059 ± 0.005 | 1.290 ± 0.020 | 1.243 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.017 ± 0.001 | 0.022 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.077 ± 0.008 | - | 0.080 ± 0.002 | 0.088 ± 0.012 | 0.121 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.018 ± 0.000 | 0.044 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.021 ± 0.019 | 0.053 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.153 ± 0.002 | - | 0.156 ± 0.000 | 0.171 ± 0.002 | 0.125 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.215 ± 0.015 | - | 0.220 ± 0.011 | 0.235 ± 0.015 | 0.212 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.524 ± 0.022 | - | 0.516 ± 0.022 | 0.379 ± 0.017 | 0.637 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.136 ± 0.003 | - | 0.151 ± 0.006 | 0.226 ± 0.019 | 0.201 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.024 ± 0.002 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.039 ± 0.001 | 0.076 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.024 ± 0.004 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.015 ± 0.000 | - | 0.018 ± 0.001 | 0.042 ± 0.001 | 0.078 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.024 ± 0.002 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.026 ± 0.003 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.026 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.028 ± 0.001 | 0.021 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.026 ± 0.000 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.027 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.029 ± 0.003 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 1.073 ± 0.025 | - | 1.077 ± 0.007 | 0.690 ± 0.008 | 1.005 ± 0.060 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.028 ± 0.016 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.079 ± 0.007 | - | 0.081 ± 0.000 | 0.078 ± 0.002 | 0.167 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.029 ± 0.000 | 0.019 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.031 ± 0.001 | 0.026 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.267 ± 0.008 | - | 0.275 ± 0.011 | 0.231 ± 0.013 | 0.321 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.602 ± 0.009 | - | 0.602 ± 0.011 | 0.528 ± 0.011 | 0.859 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.324 ± 0.009 | - | 1.368 ± 0.211 | 1.350 ± 0.010 | 2.316 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.504 ± 0.023 | - | 0.500 ± 0.017 | 0.563 ± 0.013 | 0.634 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.018 ± 0.001 | - | 0.020 ± 0.000 | 0.027 ± 0.001 | 0.070 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.023 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.030 ± 0.001 | 0.069 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 3.887 ± 0.090 | - | 4.642 ± 0.151 | 4.399 ± 0.020 | 4.698 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.129 ± 0.002 | - | 0.133 ± 0.015 | 0.175 ± 0.002 | 0.216 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.016 ± 0.000 | 0.039 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.255 ± 0.006 | - | 0.260 ± 0.015 | 0.321 ± 0.013 | 0.320 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.599 ± 0.028 | - | 0.610 ± 4.481 | 0.763 ± 0.009 | 0.860 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.343 ± 0.033 | - | 1.346 ± 0.038 | 2.027 ± 0.010 | 1.321 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 1.026 ± 0.019 | - | 1.038 ± 0.013 | 0.520 ± 0.021 | 0.995 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.018 ± 0.001 | - | 0.022 ± 0.000 | 0.045 ± 0.018 | 0.069 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.027 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.020 ± 0.003 | - | 0.023 ± 0.001 | 0.047 ± 0.001 | 0.061 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.001 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.002 | 0.012 ± 0.001 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.033 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.000 | 0.032 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 8.088 ± 0.172 | - | 8.931 ± 0.059 | 3.780 ± 0.036 | 7.992 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.001 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.136 ± 0.003 | - | 0.134 ± 0.004 | 0.289 ± 0.011 | 0.220 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.016 ± 0.000 | 0.032 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.012 ± 0.000 | - | 0.015 ± 0.000 | 0.026 ± 0.002 | 0.039 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.158 ± 0.001 | - | 0.151 ± 0.002 | 0.098 ± 0.003 | 0.280 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.203 ± 0.006 | - | 0.210 ± 0.045 | 0.141 ± 0.001 | 0.384 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.542 ± 0.011 | - | 0.561 ± 0.010 | 0.404 ± 0.011 | 0.820 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.120 ± 0.009 | - | 0.129 ± 0.042 | 0.147 ± 0.003 | 0.173 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.018 ± 0.001 | 0.025 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.011 ± 0.001 | - | 0.017 ± 0.003 | 0.031 ± 0.001 | 0.053 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.002 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.001 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.001 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.001 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.020 ± 0.001 | 0.025 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.003 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.021 ± 0.002 | 0.025 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.925 ± 0.038 | - | 0.993 ± 0.024 | 1.021 ± 0.003 | 1.002 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.000 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.071 ± 0.001 | - | 0.084 ± 0.021 | 0.081 ± 0.001 | 0.144 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.011 ± 0.000 | 0.021 ± 0.001 | 0.029 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.164 ± 0.015 | - | 0.152 ± 0.002 | 0.131 ± 0.003 | 0.264 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.211 ± 0.001 | - | 0.218 ± 0.043 | 0.172 ± 0.001 | 0.370 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.547 ± 0.033 | - | 5.999 ± 0.010 | 0.379 ± 0.018 | 0.843 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.119 ± 0.005 | - | 0.160 ± 0.041 | 0.201 ± 0.019 | 0.171 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.026 ± 0.004 | 0.016 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.011 ± 0.000 | - | 0.019 ± 0.002 | 0.042 ± 0.005 | 0.050 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.026 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.027 ± 0.004 | 0.024 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.027 ± 0.004 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.027 ± 0.007 | 0.025 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.029 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.027 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.027 ± 0.002 | 0.022 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.029 ± 0.004 | 0.022 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.928 ± 0.022 | - | 23.994 ± 0.118 | 0.584 ± 0.019 | 0.744 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.031 ± 0.004 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.072 ± 0.001 | - | 0.101 ± 0.020 | 0.072 ± 0.001 | 0.163 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.032 ± 0.007 | 0.027 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.013 ± 0.000 | 0.033 ± 0.003 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.805 ± 0.024 | - | 1.818 ± 0.009 | 1.551 ± 0.023 | 2.023 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 6.009 ± 0.010 | - | 6.058 ± 0.068 | 5.150 ± 0.024 | 5.103 ± 0.089 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 17.790 ± 0.099 | - | 17.808 ± 0.097 | 15.922 ± 0.063 | 14.713 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 4.938 ± 0.012 | - | 4.961 ± 0.044 | 5.558 ± 0.018 | 4.153 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.015 ± 0.001 | 0.034 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.206 ± 0.005 | - | 0.215 ± 0.017 | 0.230 ± 0.017 | 0.239 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.020 ± 0.002 | 0.034 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.178 ± 0.011 | - | 0.182 ± 0.005 | 0.207 ± 0.020 | 0.218 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.020 ± 0.000 | - | 0.023 ± 0.000 | 0.031 ± 0.002 | 0.045 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.007 | 0.033 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.020 ± 0.000 | - | 0.022 ± 0.000 | 0.030 ± 0.001 | 0.051 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.054 ± 0.000 | - | 0.058 ± 0.000 | 0.063 ± 0.001 | 0.092 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.009 ± 0.000 | - | 0.012 ± 0.006 | 0.019 ± 0.002 | 0.036 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.026 ± 0.000 | - | 0.028 ± 0.002 | 0.035 ± 0.004 | 0.048 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.073 ± 0.001 | - | 0.076 ± 0.004 | 0.079 ± 0.004 | 0.091 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 39.422 ± 0.056 | - | 40.645 ± 1.636 | 43.946 ± 0.151 | 32.940 ± 0.107 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.016 ± 0.000 | - | 0.019 ± 0.000 | 0.025 ± 0.001 | 0.053 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.597 ± 0.094 | - | 1.548 ± 0.014 | 1.637 ± 0.014 | 1.539 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.049 ± 0.000 | - | 0.052 ± 0.000 | 0.056 ± 0.001 | 0.076 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.140 ± 0.000 | - | 0.144 ± 0.002 | 0.141 ± 0.002 | 0.162 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.820 ± 0.041 | - | 1.823 ± 0.063 | 1.521 ± 0.013 | 2.022 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 6.089 ± 0.059 | - | 6.058 ± 0.036 | 5.065 ± 0.026 | 2.796 ± 0.420 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 17.721 ± 0.159 | - | 17.720 ± 0.027 | 18.544 ± 0.097 | 7.564 ± 0.281 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 9.090 ± 0.152 | - | 9.077 ± 0.050 | 4.739 ± 0.016 | 5.052 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.020 ± 0.002 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.304 ± 0.042 | - | 0.311 ± 0.028 | 0.330 ± 0.017 | 0.259 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.009 ± 0.000 | - | 0.013 ± 0.000 | 0.020 ± 0.001 | 0.035 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.223 ± 0.009 | - | 0.238 ± 0.017 | 0.288 ± 0.017 | 0.239 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.021 ± 0.000 | - | 0.024 ± 0.000 | 0.032 ± 0.013 | 0.044 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.024 ± 0.004 | 0.032 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.020 ± 0.000 | - | 0.023 ± 0.000 | 0.031 ± 0.008 | 0.042 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.055 ± 0.000 | - | 0.059 ± 0.000 | 0.064 ± 0.003 | 0.073 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.027 ± 0.002 | 0.035 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.026 ± 0.000 | - | 0.029 ± 0.000 | 0.035 ± 0.001 | 0.049 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.073 ± 0.000 | - | 0.077 ± 0.001 | 0.079 ± 0.003 | 0.090 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 73.011 ± 2.224 | - | 73.367 ± 2.062 | 36.944 ± 0.899 | 55.650 ± 51.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.017 ± 0.000 | - | 0.020 ± 0.000 | 0.040 ± 0.004 | 0.043 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 2.272 ± 0.034 | - | 2.293 ± 0.014 | 2.372 ± 0.017 | 0.774 ± 0.135 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.050 ± 0.001 | - | 0.053 ± 0.000 | 0.056 ± 0.009 | 0.070 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.142 ± 0.002 | - | 0.144 ± 0.002 | 0.142 ± 0.005 | 0.161 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 1.342 ± 0.006 | - | 1.348 ± 0.007 | 1.091 ± 0.002 | 1.359 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 5.122 ± 0.044 | - | 5.125 ± 0.043 | 4.348 ± 0.015 | 4.038 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 15.123 ± 0.086 | - | 15.146 ± 0.013 | 13.842 ± 0.049 | 12.267 ± 0.080 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 2.295 ± 0.024 | - | 2.322 ± 0.054 | 2.474 ± 1.088 | 2.381 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.001 | 0.026 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.139 ± 0.006 | - | 0.141 ± 0.005 | 0.165 ± 0.017 | 0.182 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.001 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.107 ± 0.006 | - | 0.111 ± 0.014 | 0.133 ± 0.004 | 0.159 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.017 ± 0.000 | - | 0.020 ± 0.000 | 0.029 ± 0.002 | 0.039 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.016 ± 0.001 | 0.033 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.017 ± 0.000 | - | 0.020 ± 0.000 | 0.028 ± 0.001 | 0.041 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.043 ± 0.002 | - | 0.046 ± 0.000 | 0.052 ± 0.001 | 0.062 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.008 ± 0.000 | - | 0.010 ± 0.002 | 0.017 ± 0.001 | 0.041 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.023 ± 0.000 | - | 0.025 ± 0.000 | 0.033 ± 0.005 | 0.042 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.057 ± 0.001 | - | 0.060 ± 0.002 | 0.065 ± 0.001 | 0.073 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 18.270 ± 0.230 | - | 18.289 ± 0.089 | 19.296 ± 0.103 | 18.528 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.021 ± 0.001 | 0.035 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 1.052 ± 0.001 | - | 1.063 ± 0.006 | 1.157 ± 0.008 | 1.127 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.042 ± 0.001 | - | 0.045 ± 0.000 | 0.048 ± 0.001 | 0.058 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.114 ± 0.005 | - | 0.120 ± 0.002 | 0.117 ± 0.002 | 0.136 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 1.310 ± 0.017 | - | 1.314 ± 0.019 | 1.061 ± 0.036 | 1.306 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 5.167 ± 0.027 | - | 5.161 ± 0.051 | 4.342 ± 0.026 | 2.092 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 15.125 ± 0.131 | - | 15.170 ± 0.249 | 16.684 ± 0.029 | 6.281 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 3.375 ± 0.009 | - | 3.406 ± 0.012 | 2.798 ± 0.020 | 3.081 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.002 | 0.032 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.236 ± 0.027 | - | 0.247 ± 0.018 | 0.254 ± 0.016 | 0.187 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.018 ± 0.001 | 0.030 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.150 ± 0.009 | - | 0.161 ± 0.007 | 0.196 ± 0.006 | 0.160 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.018 ± 0.000 | - | 0.021 ± 0.000 | 0.027 ± 0.001 | 0.039 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.021 ± 0.002 | 0.031 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.018 ± 0.000 | - | 0.021 ± 0.000 | 0.028 ± 0.011 | 0.038 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.043 ± 0.000 | - | 0.047 ± 0.001 | 0.052 ± 0.004 | 0.061 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.024 ± 0.004 | 0.031 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.023 ± 0.000 | - | 0.026 ± 0.000 | 0.031 ± 0.004 | 0.042 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.058 ± 0.002 | - | 0.061 ± 0.001 | 0.065 ± 0.002 | 0.073 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 27.078 ± 0.097 | - | 27.031 ± 0.061 | 21.852 ± 0.083 | 25.944 ± 4.485 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.013 ± 0.000 | - | 0.016 ± 0.000 | 0.030 ± 0.004 | 0.036 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 1.771 ± 0.026 | - | 1.783 ± 0.026 | 1.755 ± 0.007 | 0.667 ± 0.115 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.042 ± 0.000 | - | 0.046 ± 0.000 | 0.047 ± 0.001 | 0.058 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.115 ± 0.001 | - | 0.118 ± 0.007 | 0.118 ± 0.003 | 0.142 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.464 ± 0.012 | - | unsupported | 0.353 ± 0.012 | 0.756 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 2.272 ± 0.009 | - | unsupported | 1.930 ± 0.002 | 2.699 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 5.530 ± 0.024 | - | unsupported | 5.309 ± 0.012 | 6.798 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.981 ± 0.012 | - | unsupported | 1.135 ± 0.013 | 1.105 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.009 ± 0.001 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.048 ± 0.001 | - | unsupported | 0.062 ± 0.008 | 0.078 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | unsupported | 0.012 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.039 ± 0.002 | - | unsupported | 0.056 ± 0.006 | 0.070 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | unsupported | 0.015 ± 0.005 | 0.056 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.011 ± 0.002 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | unsupported | 0.015 ± 0.001 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.017 ± 0.000 | - | unsupported | 0.026 ± 0.001 | 0.037 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.011 ± 0.001 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | unsupported | 0.017 ± 0.001 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.023 ± 0.000 | - | unsupported | 0.032 ± 0.001 | 0.054 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 7.677 ± 0.075 | - | unsupported | 8.937 ± 0.071 | 8.448 ± 0.075 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | unsupported | 0.012 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.373 ± 0.008 | - | unsupported | 0.420 ± 0.011 | 0.477 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.019 ± 0.000 | - | unsupported | 0.025 ± 0.001 | 0.040 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.044 ± 0.000 | - | unsupported | 0.051 ± 0.001 | 0.072 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.474 ± 0.020 | - | unsupported | 0.580 ± 0.014 | 0.770 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 2.292 ± 0.030 | - | unsupported | 1.926 ± 0.013 | 1.409 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 5.593 ± 0.040 | - | unsupported | 7.676 ± 0.096 | 3.506 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.199 ± 0.019 | - | unsupported | 1.448 ± 0.015 | 1.445 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | unsupported | 0.013 ± 0.001 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.048 ± 0.002 | - | unsupported | 0.084 ± 0.028 | 0.074 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | unsupported | 0.016 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.039 ± 0.000 | - | unsupported | 0.080 ± 0.003 | 0.089 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.008 ± 0.000 | - | unsupported | 0.015 ± 0.000 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | unsupported | 0.015 ± 0.001 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.009 ± 0.000 | - | unsupported | 0.023 ± 0.003 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.018 ± 0.000 | - | unsupported | 0.025 ± 0.001 | 0.037 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | unsupported | 0.015 ± 0.001 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.011 ± 0.000 | - | unsupported | 0.017 ± 0.000 | 0.027 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.024 ± 0.000 | - | unsupported | 0.031 ± 0.001 | 0.043 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 9.363 ± 0.087 | - | unsupported | 11.227 ± 0.168 | 12.994 ± 0.740 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | unsupported | 0.018 ± 0.003 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.373 ± 0.021 | - | unsupported | 0.419 ± 0.012 | 0.302 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.019 ± 0.000 | - | unsupported | 0.025 ± 0.001 | 0.037 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.044 ± 0.000 | - | unsupported | 0.051 ± 0.001 | 0.088 ± 0.007 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 1.038 ± 0.003 | - | unsupported | 0.949 ± 0.004 | 3.550 ± 0.172 | 2.480 ± 0.200 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.131 ± 0.003 | - | unsupported | 0.059 ± 0.001 | 0.111 ± 0.008 | 0.097 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.138 ± 0.002 | - | unsupported | 0.084 ± 0.001 | 0.182 ± 0.013 | 0.175 ± 0.005 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.337 ± 0.008 | - | unsupported | 0.245 ± 0.013 | 0.516 ± 0.021 | 0.669 ± 0.018 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 1.042 ± 0.002 | - | unsupported | 0.482 ± 0.020 | 1.108 ± 0.051 | 2.445 ± 0.825 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.133 ± 0.003 | - | unsupported | 0.062 ± 0.007 | 0.114 ± 0.008 | 0.102 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.138 ± 0.002 | - | unsupported | 0.080 ± 0.002 | 0.183 ± 0.007 | 0.183 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.336 ± 0.010 | - | unsupported | 0.155 ± 0.001 | 0.536 ± 0.017 | 0.663 ± 0.043 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.302 ± 0.009 | - | unsupported | 0.375 ± 0.010 | 0.808 ± 0.010 | 0.423 ± 0.011 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.034 ± 0.000 | - | unsupported | 0.046 ± 0.001 | 0.209 ± 0.007 | 0.077 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.059 ± 0.000 | - | unsupported | 0.061 ± 0.009 | 0.245 ± 0.024 | 0.119 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.201 ± 0.001 | - | unsupported | 0.167 ± 0.003 | 0.394 ± 0.026 | 0.213 ± 0.007 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.304 ± 0.008 | - | unsupported | 0.551 ± 0.009 | 0.939 ± 0.013 | 0.414 ± 0.010 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.035 ± 0.000 | - | unsupported | 0.078 ± 0.001 | 0.207 ± 0.021 | 0.074 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.061 ± 0.003 | - | unsupported | 0.100 ± 0.001 | 0.246 ± 0.022 | 0.115 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.203 ± 0.001 | - | unsupported | 0.226 ± 0.006 | 0.378 ± 0.034 | 0.211 ± 0.008 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.193 ± 0.001 | - | 0.196 ± 0.003 | 0.193 ± 0.002 | 0.395 ± 0.014 | 0.253 ± 0.008 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.081 ± 0.000 | - | 0.087 ± 0.000 | 0.074 ± 0.007 | 0.077 ± 0.002 | 0.058 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.092 ± 0.000 | - | 0.096 ± 0.001 | 0.087 ± 0.004 | 0.100 ± 0.003 | 0.085 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.124 ± 0.000 | - | 0.127 ± 0.001 | 0.123 ± 0.003 | 0.172 ± 0.004 | 0.137 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.194 ± 0.001 | - | 0.198 ± 0.011 | 0.256 ± 0.015 | 0.461 ± 0.029 | 0.255 ± 0.009 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.082 ± 0.003 | - | 0.087 ± 0.000 | 0.125 ± 0.010 | 0.153 ± 0.006 | 0.060 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.093 ± 0.001 | - | 0.096 ± 0.000 | 0.141 ± 0.006 | 0.183 ± 0.027 | 0.087 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.124 ± 0.014 | - | 0.127 ± 0.001 | 0.177 ± 0.002 | 0.218 ± 0.036 | 0.140 ± 0.001 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 42.697 ± 0.130 | - | 49.988 ± 0.210 | 61.108 ± 0.178 | 69.357 ± 0.360 | 49.608 ± 5.844 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 45.036 ± 0.191 | - | 49.112 ± 1.057 | 25.197 ± 0.165 | 53.943 ± 11.253 | 45.353 ± 8.938 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 24.679 ± 0.366 | - | 24.302 ± 0.383 | 26.581 ± 0.053 | 21.428 ± 0.034 | 20.865 ± 2.618 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 253.044 ± 36.021 | - | 328.008 ± 15.043 | 11.273 ± 0.089 | 14.285 ± 0.344 | 12.497 ± 0.138 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 7.439 ± 0.037 | - | 7.505 ± 0.097 | 7.968 ± 0.036 | 8.043 ± 0.024 | 7.880 ± 0.236 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 16.300 ± 0.224 | - | 16.507 ± 0.057 | 10.506 ± 0.028 | 11.659 ± 0.148 | 12.127 ± 0.105 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 8.700 ± 0.012 | - | 8.744 ± 0.014 | 8.794 ± 0.015 | 9.548 ± 0.015 | 9.118 ± 0.034 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 12.886 ± 0.130 | - | 12.828 ± 0.095 | 11.816 ± 0.646 | 14.168 ± 0.095 | 13.800 ± 0.148 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 20.925 ± 0.052 | - | 21.638 ± 0.077 | 19.534 ± 0.130 | 38.818 ± 0.231 | 14.200 ± 0.256 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 38.459 ± 0.340 | - | 39.396 ± 0.260 | 15.177 ± 0.321 | 32.809 ± 0.510 | 13.365 ± 0.437 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.488 ± 0.002 | 0.504 ± 0.005 | unsupported | 0.605 ± 0.006 | 0.748 ± 0.023 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 7.803 ± 0.041 | 7.902 ± 0.016 | unsupported | 8.802 ± 0.024 | 9.555 ± 0.169 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.792 ± 0.027 | 0.811 ± 0.008 | unsupported | 0.511 ± 0.002 | 1.209 ± 0.961 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 2274.923 ± 3.857 | 2306.715 ± 1.817 | unsupported | 5.721 ± 0.018 | 9.718 ± 0.415 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 43.982 ± 3.252 | - | 41.133 ± 0.253 | 43.259 ± 0.187 | 31.328 ± 0.301 | 27.460 ± 0.292 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 233.245 ± 6.441 | - | 282.664 ± 7.987 | 21.714 ± 0.318 | 20.081 ± 0.277 | 16.673 ± 1.154 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 17.563 ± 0.781 | 17.308 ± 0.104 | 9.331 ± 0.071 | 46.026 ± 0.176 | 9.565 ± 0.054 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 2285.166 ± 6.886 | 2275.948 ± 16.054 | 6.535 ± 0.200 | 45.516 ± 5.164 | 8.798 ± 0.940 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 25.014 ± 0.134 | - | 25.350 ± 0.177 | 32.250 ± 0.148 | 32.442 ± 1.068 | 35.157 ± 8.495 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 241.183 ± 14.693 | - | 240.994 ± 9.834 | 17.141 ± 0.171 | 24.464 ± 1.757 | 24.104 ± 6.003 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 1.097 ± 0.040 | - | 1.079 ± 0.166 | 1.273 ± 0.087 | 4.084 ± 0.165 | 2.690 ± 0.027 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.458 ± 0.018 | - | 23.997 ± 0.009 | 1.266 ± 0.152 | 2.261 ± 0.109 | 2.697 ± 0.034 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 19.541 ± 0.066 | - | 19.509 ± 0.125 | 20.389 ± 0.041 | 19.757 ± 0.036 | 19.385 ± 0.196 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 219.073 ± 26.102 | - | 224.358 ± 14.862 | 16.663 ± 0.169 | 19.966 ± 0.856 | 20.294 ± 1.979 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 19.440 ± 0.071 | - | 19.465 ± 0.083 | 20.389 ± 0.107 | 19.928 ± 0.294 | 19.299 ± 0.225 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 215.797 ± 35.173 | - | 233.381 ± 32.994 | 16.739 ± 0.062 | 20.524 ± 1.324 | 19.854 ± 0.639 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.156 ± 0.010 | 1.167 ± 0.009 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 22.474 ± 0.023 | 22.555 ± 0.171 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.583 ± 0.009 | 1.598 ± 0.013 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 23.402 ± 0.135 | 23.397 ± 1.731 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.085 ± 0.011 | 1.088 ± 0.012 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 21.461 ± 0.059 | 21.800 ± 0.377 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.518 ± 0.006 | 1.535 ± 0.022 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 22.398 ± 0.091 | 22.401 ± 0.031 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 24.735 ± 0.224 | - | 24.404 ± 0.726 | 26.842 ± 0.209 | 21.418 ± 0.042 | 20.900 ± 0.578 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 164.126 ± 36.123 | - | 324.996 ± 9.053 | 11.535 ± 0.267 | 14.146 ± 0.849 | 12.546 ± 5.443 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 59.116 ± 1.002 | 59.140 ± 0.105 | 61.848 ± 0.246 | 65.766 ± 0.238 | 66.081 ± 0.373 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 2330.572 ± 7.164 | 2332.052 ± 8.196 | 39.059 ± 0.229 | 55.487 ± 0.747 | 56.914 ± 1.975 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 33.059 ± 0.191 | - | 33.216 ± 0.189 | 29.988 ± 0.229 | 125.409 ± 1.531 | 29.832 ± 0.536 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 53.302 ± 1.050 | - | 52.268 ± 0.359 | 23.992 ± 2.224 | 113.821 ± 6.135 | 19.955 ± 0.111 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.140 ± 0.001 | 0.140 ± 0.000 | unsupported | 0.155 ± 0.026 | 0.140 ± 0.012 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.264 ± 0.006 | 2.221 ± 0.009 | unsupported | 1.610 ± 0.014 | 0.633 ± 0.722 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.074 ± 0.000 | 5.984 ± 0.063 | unsupported | 0.076 ± 0.004 | 0.127 ± 0.014 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.897 ± 0.009 | 23.999 ± 0.002 | unsupported | 1.006 ± 0.009 | 0.225 ± 0.335 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.080 ± 0.000 | 0.079 ± 0.001 | unsupported | 0.123 ± 0.013 | 0.227 ± 0.004 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.253 ± 0.006 | 1.244 ± 0.012 | unsupported | 1.782 ± 0.024 | 2.609 ± 0.019 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.037 ± 0.000 | 12.000 ± 0.000 | unsupported | 0.096 ± 0.031 | 0.160 ± 0.001 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.563 ± 0.018 | 12.000 ± 0.000 | unsupported | 0.701 ± 0.010 | 1.159 ± 0.042 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.197 ± 0.001 | 0.198 ± 0.001 | unsupported | 0.281 ± 0.002 | 0.180 ± 0.003 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 3.139 ± 0.004 | 3.122 ± 0.008 | unsupported | 4.390 ± 0.014 | 1.789 ± 0.027 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.077 ± 0.002 | 12.000 ± 0.563 | unsupported | 0.107 ± 0.003 | 0.145 ± 0.002 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 1.177 ± 0.003 | 12.000 ± 0.001 | unsupported | 1.479 ± 0.007 | 0.967 ± 0.013 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.056 ± 0.003 | 0.088 ± 0.003 | unsupported | 0.156 ± 0.001 | 0.176 ± 0.002 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.639 ± 0.027 | 0.683 ± 0.032 | unsupported | 0.904 ± 0.006 | 1.822 ± 0.017 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.094 ± 0.003 | 36.000 ± 0.141 | unsupported | 0.067 ± 0.001 | 0.177 ± 0.002 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.391 ± 0.012 | 96.000 ± 0.001 | unsupported | 0.362 ± 0.004 | 1.111 ± 0.034 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.125 ± 0.002 | 0.152 ± 0.003 | unsupported | 0.183 ± 0.001 | 0.215 ± 0.004 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.679 ± 0.011 | 1.751 ± 0.052 | unsupported | 2.038 ± 0.021 | 2.235 ± 0.052 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.143 ± 0.001 | 43.875 ± 0.141 | unsupported | 0.062 ± 0.001 | 0.203 ± 0.005 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.958 ± 0.113 | 119.999 ± 0.002 | unsupported | 0.487 ± 0.006 | 1.336 ± 0.026 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.290 ± 0.003 | 0.333 ± 0.011 | unsupported | unsupported | 0.237 ± 0.005 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 4.294 ± 0.023 | 4.381 ± 0.060 | unsupported | unsupported | 3.100 ± 0.009 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.278 ± 0.004 | 51.875 ± 0.375 | unsupported | unsupported | 0.239 ± 0.008 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 1.999 ± 0.088 | 143.999 ± 0.001 | unsupported | unsupported | 2.207 ± 0.373 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.010 ± 0.000 | 0.019 ± 0.001 | unsupported | 0.014 ± 0.000 | 0.049 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.101 ± 0.000 | 0.120 ± 0.006 | unsupported | 0.089 ± 0.007 | 0.224 ± 0.006 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.016 ± 0.000 | 6.000 ± 0.047 | unsupported | 0.015 ± 0.000 | 0.038 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.113 ± 0.017 | 24.000 ± 0.002 | unsupported | 0.092 ± 0.010 | 0.149 ± 0.006 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.030 ± 0.004 | 0.050 ± 0.002 | unsupported | 0.097 ± 0.001 | 0.159 ± 0.002 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.348 ± 0.015 | 0.370 ± 0.043 | unsupported | 0.535 ± 0.029 | 1.626 ± 0.006 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.044 ± 0.003 | 18.000 ± 0.000 | unsupported | 0.092 ± 0.004 | 0.142 ± 0.003 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.198 ± 0.026 | 48.000 ± 0.001 | unsupported | 0.241 ± 0.058 | 1.007 ± 0.111 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.054 ± 0.000 | 0.055 ± 0.000 | unsupported | 0.114 ± 0.002 | 0.168 ± 0.015 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.856 ± 0.003 | 0.866 ± 0.001 | unsupported | 1.767 ± 0.019 | 1.705 ± 0.024 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.027 ± 0.007 | 12.000 ± 0.000 | unsupported | 0.083 ± 0.001 | 0.136 ± 0.010 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.389 ± 0.002 | 11.999 ± 0.001 | unsupported | 0.692 ± 0.016 | 0.911 ± 0.025 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.056 ± 0.001 | 0.057 ± 0.000 | unsupported | 0.117 ± 0.001 | 0.168 ± 0.003 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.888 ± 0.005 | 0.899 ± 0.008 | unsupported | 1.828 ± 0.014 | 1.717 ± 0.023 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.028 ± 0.000 | 12.000 ± 0.000 | unsupported | 0.087 ± 0.002 | 0.133 ± 0.010 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.406 ± 0.001 | 12.000 ± 0.001 | unsupported | 0.706 ± 0.007 | 0.926 ± 0.015 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.125 ± 0.001 | 0.153 ± 0.006 | unsupported | 0.136 ± 0.002 | 0.234 ± 0.018 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.663 ± 0.012 | 1.760 ± 0.066 | unsupported | 1.175 ± 0.010 | 2.132 ± 0.035 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.142 ± 0.002 | 43.765 ± 5.460 | unsupported | 0.061 ± 0.001 | 0.201 ± 0.002 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 0.962 ± 0.091 | 120.000 ± 0.375 | unsupported | 0.483 ± 0.004 | 1.269 ± 0.053 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.183 ± 0.000 | 0.183 ± 0.000 | unsupported | 0.270 ± 0.000 | 0.311 ± 0.002 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 2.915 ± 0.008 | 2.912 ± 0.013 | unsupported | 4.275 ± 0.037 | 3.945 ± 0.031 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.073 ± 0.001 | 12.000 ± 0.000 | unsupported | 0.168 ± 0.003 | 0.211 ± 0.013 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 1.123 ± 0.002 | 12.000 ± 0.000 | unsupported | 1.518 ± 0.015 | 2.048 ± 0.016 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.761 ± 0.027 | 1.431 ± 0.013 | unsupported | 0.262 ± 0.002 | 0.210 ± 0.004 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 18.513 ± 0.120 | 28.595 ± 0.279 | unsupported | 5.352 ± 0.043 | 2.308 ± 0.022 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.848 ± 0.030 | 92.812 ± 5.582 | unsupported | 0.227 ± 0.004 | 0.162 ± 0.025 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 8.529 ± 0.090 | 154.498 ± 1.248 | unsupported | 2.280 ± 0.032 | 1.232 ± 0.033 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 27.301 ± 0.258 | - | unsupported | 25.924 ± 0.228 | - | 31.391 ± 0.089 | 38.020 ± 0.159 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 20.485 ± 0.256 | - | unsupported | 20.308 ± 0.132 | - | 29.438 ± 0.104 | 20.291 ± 0.306 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 21.524 ± 0.358 | - | unsupported | 26.102 ± 0.278 | - | 21.176 ± 0.253 | 23.006 ± 0.250 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 14.476 ± 0.151 | - | unsupported | 14.387 ± 0.127 | - | 21.320 ± 0.300 | 14.248 ± 0.403 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 13.341 ± 0.252 | - | unsupported | 22.687 ± 0.467 | - | 13.381 ± 0.232 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 14.636 ± 0.253 | - | unsupported | 14.162 ± 0.187 | - | 13.291 ± 0.153 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 27.012 ± 0.190 | - | unsupported | 27.486 ± 0.225 | - | 31.272 ± 0.256 | 37.253 ± 0.413 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 20.462 ± 0.192 | - | unsupported | 20.458 ± 0.214 | - | 26.975 ± 0.282 | 20.368 ± 0.344 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 48.739 ± 0.182 | - | unsupported | 49.534 ± 0.432 | - | 48.167 ± 0.132 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 51.480 ± 0.129 | - | unsupported | 25.003 ± 0.126 | - | 23.417 ± 0.072 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 48.670 ± 0.176 | - | unsupported | 49.538 ± 0.092 | - | 47.917 ± 0.117 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 51.211 ± 0.475 | - | unsupported | 24.911 ± 0.080 | - | 23.294 ± 0.197 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 27.165 ± 0.285 | - | unsupported | 26.033 ± 0.321 | - | 31.379 ± 0.302 | 38.190 ± 0.336 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 20.312 ± 0.417 | - | unsupported | 20.495 ± 0.343 | - | 29.436 ± 0.380 | 20.521 ± 0.163 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 21.313 ± 0.247 | - | unsupported | 22.275 ± 0.324 | - | 25.404 ± 0.072 | 27.340 ± 0.117 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 14.109 ± 0.128 | - | unsupported | 14.209 ± 0.270 | - | 23.562 ± 0.081 | 14.393 ± 0.801 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.752 ± 0.003 | unsupported | unsupported | 0.899 ± 0.003 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 11.010 ± 0.027 | unsupported | unsupported | 12.738 ± 0.026 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 1.463 ± 0.008 | unsupported | unsupported | 0.582 ± 0.004 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 192.771 ± 5.875 | unsupported | unsupported | 6.894 ± 0.068 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 27.456 ± 0.406 | - | unsupported | 26.086 ± 0.493 | - | 31.512 ± 0.157 | 37.296 ± 0.236 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 20.316 ± 0.241 | - | unsupported | 20.459 ± 0.119 | - | 29.611 ± 0.176 | 20.693 ± 1.508 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 105.092 ± 3.589 | - | 0.002 ± 0.000 | 105.370 ± 2.303 | 106.012 ± 3.319 | 130.472 ± 3.418 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 42.648 ± 0.109 | - | 0.002 ± 0.000 | 35.008 ± 0.124 | 54.366 ± 0.248 | 130.182 ± 3.129 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 64.906 ± 2.353 | - | 203.969 ± 5.786 | 66.247 ± 1.169 | 67.547 ± 0.282 | 76.578 ± 1.444 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 22.516 ± 0.149 | - | 256.474 ± 11.007 | 22.370 ± 0.231 | 34.561 ± 0.366 | 76.257 ± 0.818 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 213.093 ± 0.962 | - | 214.160 ± 1.950 | 213.002 ± 1.116 | 214.854 ± 0.700 | 247.192 ± 3.338 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 70.564 ± 0.495 | - | 215.792 ± 10.324 | 69.871 ± 0.297 | 107.889 ± 0.762 | 246.267 ± 4.690 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 66.796 ± 1.221 | - | 67.031 ± 0.213 | 71.690 ± 2.389 | 66.118 ± 1.887 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 41.620 ± 0.213 | - | 67.806 ± 6.009 | 24.025 ± 0.273 | 34.614 ± 0.813 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 134.198 ± 0.325 | - | 0.002 ± 0.000 | 120.636 ± 1.756 | 117.559 ± 0.687 | 152.679 ± 2.088 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 40.286 ± 0.120 | - | 0.002 ± 0.000 | 39.856 ± 0.226 | 48.043 ± 6.229 | 151.862 ± 1.834 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 81.529 ± 2.367 | - | 0.002 ± 0.000 | 76.461 ± 0.343 | 80.944 ± 0.252 | 211.360 ± 9.848 | 127.128 ± 2.164 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 27.865 ± 1.861 | - | 0.002 ± 0.000 | 76.406 ± 0.617 | 41.929 ± 0.639 | 212.406 ± 1.659 | 51.394 ± 0.852 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 61.607 ± 2.799 | - | 61.363 ± 2.791 | 64.051 ± 0.523 | 63.514 ± 0.614 | 80.665 ± 1.617 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 20.913 ± 0.072 | - | 71.200 ± 1.259 | 23.363 ± 0.175 | 32.511 ± 0.617 | 79.069 ± 1.460 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 58.685 ± 2.722 | - | 57.863 ± 0.592 | 62.085 ± 0.448 | 62.281 ± 1.323 | 80.451 ± 0.158 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 20.181 ± 0.117 | - | 69.575 ± 2.436 | 22.803 ± 0.271 | 32.231 ± 0.523 | 78.567 ± 2.104 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | - | - |
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
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 7.631749 | 476984.31 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 7.084195 | 442762.19 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 7.703278 | 481454.88 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 6.658663 | 416166.44 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 6.125422 | 382838.88 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 831.993828 | 51999614.25 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 16.813299 | 1050831.19 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 16.558454 | 1034903.38 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 5.774109 | 360881.81 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 383.989230 | 23999326.88 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 11.973306 | 748331.62 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 9.687444 | 605465.25 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 7.800698 | 487543.62 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 8.062358 | 503897.38 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 152.882199 | 9555137.44 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 140.830886 | 8801930.38 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 124.847301 | 7802956.31 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 126.426045 | 7901627.81 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 19.343825 | 1208989.06 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 8.172810 | 510800.62 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 12.669804 | 791862.75 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 12.974804 | 810925.25 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 155.480817 | 9717551.06 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 91.530769 | 5720673.06 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 36398.772137 | 2274923258.56 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 36907.432595 | 2306714537.19 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 18.500973 | 1156310.81 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 18.678063 | 1167378.94 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 359.577531 | 22473595.69 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 360.875609 | 22554725.56 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 25.320092 | 1582505.75 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 25.570810 | 1598175.62 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 374.430201 | 23401887.56 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 374.347222 | 23396701.38 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 17.356675 | 1084792.19 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 17.406629 | 1087914.31 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 343.379394 | 21461212.12 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 348.792288 | 21799518.00 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 24.286051 | 1517878.19 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 24.567958 | 1535497.38 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 358.372295 | 22398268.44 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 358.409364 | 22400585.25 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.968516 | 140133.06 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 9.893859 | 154591.55 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.980394 | 140318.66 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.948354 | 139818.03 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.060701 | 632587.62 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 12.880612 | 1610076.50 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 18.112660 | 2264082.50 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 17.769961 | 2221245.12 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.113497 | 126773.39 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.872614 | 76134.59 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.715564 | 73680.69 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 382.983582 | 5984118.47 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.800064 | 225008.00 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.049040 | 1006130.00 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.179551 | 897443.88 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 191.995671 | 23999458.88 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.546849 | 227294.52 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.849321 | 122645.64 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.088612 | 79509.56 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.079645 | 79369.45 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 20.874338 | 2609292.25 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.253339 | 1781667.38 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.020792 | 1252599.00 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 9.952104 | 1244013.00 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.235871 | 159935.48 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.157536 | 96211.50 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.379443 | 37178.80 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.987523 | 11999805.05 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.275990 | 1159498.75 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.604833 | 700604.12 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.504822 | 563102.75 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.997020 | 11999627.50 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.537748 | 180277.31 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.985021 | 281015.95 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 12.616689 | 197135.77 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 12.645142 | 197580.34 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.311452 | 1788931.50 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 35.119237 | 4389904.62 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 25.115558 | 3139444.75 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 24.974826 | 3121853.25 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.271804 | 144871.94 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.823390 | 106615.47 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.901490 | 76585.78 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.999993 | 11999999.89 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.737986 | 967248.25 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 11.829764 | 1478720.50 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.417815 | 1177226.88 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.996468 | 11999558.50 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.293253 | 176457.08 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 9.998114 | 156220.53 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.607071 | 56360.48 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.647133 | 88236.45 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.572564 | 1821570.50 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.231561 | 903945.12 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.115592 | 639449.00 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.467998 | 683499.75 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.345129 | 177267.64 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.303332 | 67239.56 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.986666 | 93541.66 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2303.977583 | 35999649.73 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.889793 | 1111224.12 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.892898 | 361612.25 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.128247 | 391030.88 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 767.998081 | 95999760.12 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.753412 | 214897.06 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 11.743224 | 183487.88 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.026606 | 125415.72 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.701817 | 151590.89 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 17.876087 | 2234510.88 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 16.304300 | 2038037.50 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.434974 | 1679371.75 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 14.010956 | 1751369.50 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.023560 | 203493.12 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.999509 | 62492.33 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 9.157083 | 143079.42 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2807.979508 | 43874679.81 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.687951 | 1335993.88 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.895016 | 486877.00 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.664987 | 958123.38 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 959.994619 | 119999327.38 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 15.175560 | 237118.12 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 18.539565 | 289680.70 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 21.337739 | 333402.17 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 24.801597 | 3100199.62 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 34.350036 | 4293754.50 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 35.046943 | 4380867.88 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 15.293042 | 238953.78 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 17.794987 | 278046.67 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3319.978625 | 51874666.02 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 17.656927 | 2207115.88 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 15.995769 | 1999471.12 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1151.994266 | 143999283.25 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 3.167577 | 49493.39 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.874869 | 13669.83 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.657555 | 10274.30 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.203553 | 18805.52 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 1.795452 | 224431.50 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.708129 | 88516.12 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.808947 | 101118.38 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.963956 | 120494.50 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.458610 | 38415.78 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.928460 | 14507.19 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.992708 | 15511.06 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 383.975498 | 5999617.16 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.191840 | 148980.00 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.733389 | 91673.62 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.905226 | 113153.25 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 191.999088 | 23999886.00 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.191531 | 159242.67 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.213464 | 97085.38 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.888770 | 29512.03 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.175767 | 49621.36 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.004112 | 1625514.00 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.280563 | 535070.38 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.783576 | 347947.00 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 2.962370 | 370296.25 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.060911 | 141576.73 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.860155 | 91564.92 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.812504 | 43945.38 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1151.980641 | 17999697.52 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.058244 | 1007280.50 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.931589 | 241448.62 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.587696 | 198462.00 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 383.996769 | 47999596.12 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.754250 | 168035.16 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.318331 | 114348.92 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.479493 | 54367.08 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.514548 | 54914.81 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.643117 | 1705389.62 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.134428 | 1766803.50 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.844373 | 855546.62 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.927007 | 865875.88 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.678809 | 135606.39 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.313956 | 83030.56 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.714344 | 26786.62 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.981352 | 11999708.62 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.291835 | 911479.38 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.532749 | 691593.62 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.109762 | 388720.25 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.994805 | 11999350.62 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.732699 | 167698.42 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.489571 | 117024.55 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.602883 | 56295.05 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.636336 | 56817.75 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.735799 | 1716974.88 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.621265 | 1827658.12 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.103937 | 887992.12 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.191560 | 898945.00 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.486240 | 132597.50 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.538352 | 86536.75 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.765469 | 27585.45 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.985403 | 11999771.92 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.409304 | 926163.00 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.648384 | 706048.00 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.251066 | 406383.25 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.996058 | 11999507.25 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 15.004351 | 234442.98 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.681291 | 135645.17 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 7.994366 | 124911.97 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.796694 | 153073.34 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 17.058325 | 2132290.62 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.402404 | 1175300.50 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.301996 | 1662749.50 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 14.082010 | 1760251.25 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.834171 | 200533.92 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.884626 | 60697.28 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 9.112535 | 142383.36 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2800.979444 | 43765303.81 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.151844 | 1268980.50 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.866793 | 483349.12 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.699143 | 962392.88 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 959.996164 | 119999520.50 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 19.903722 | 310995.66 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.292651 | 270197.67 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 11.688540 | 182633.44 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 11.725178 | 183205.91 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 31.560130 | 3945016.25 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 34.202459 | 4275307.38 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 23.318861 | 2914857.62 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 23.294224 | 2911778.00 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.490284 | 210785.69 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 10.759771 | 168121.42 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.650594 | 72665.53 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.985479 | 11999773.11 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 16.385258 | 2048157.25 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.146751 | 1518343.88 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.980880 | 1122610.00 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.996078 | 11999509.75 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.420673 | 209698.02 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 16.770358 | 262036.84 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 48.725591 | 761337.36 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 91.586201 | 1431034.39 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.460556 | 2307569.50 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 42.812146 | 5351518.25 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 148.100786 | 18512598.25 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 228.757854 | 28594731.75 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.361184 | 161893.50 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 14.505721 | 226651.89 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 54.282167 | 848158.86 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5939.964150 | 92811939.84 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.855468 | 1231933.50 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 18.240920 | 2280115.00 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 68.232473 | 8529059.12 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1235.983525 | 154497940.62 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 14.380325 | 898770.31 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 12.039858 | 752491.12 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 203.802542 | 12737658.88 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 176.154888 | 11009680.50 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 9.310178 | 581886.12 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 23.406947 | 1462934.19 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 110.302849 | 6893928.06 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 3084.335697 | 192770981.06 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.024636 | 1539.75 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002475 | 154.69 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.023854 | 1490.88 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002535 | 158.44 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000130 | 8.12 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.023074 | 1442.12 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.006973 | 435.81 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000130 | 8.12 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.022151 | 1384.44 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.007333 | 458.31 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000120 | 7.50 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.022943 | 1433.94 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.004859 | 303.69 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000301 | 18.81 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.022662 | 1416.38 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.005610 | 350.62 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000110 | 6.88 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.021319 | 1332.44 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003256 | 203.50 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000101 | 6.31 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.021180 | 1323.75 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003336 | 208.50 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/norm_fro` (c64, threads=1, shape=`2048x1536`): `tenferro-direct` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 24.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `jax-cpu` is 27.3x faster than the slowest successful cell (`tenferro-trace`, 59.985 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `julia-base` is 14.7x faster than the slowest successful cell (`tenferro-trace`, 59.985 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 71.5x faster than the slowest successful cell (`tenferro-trace`, 59.985 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 18.3x faster than the slowest successful cell (`tenferro-direct`, 127.877 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `julia-base` is 17.0x faster than the slowest successful cell (`tenferro-direct`, 127.877 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 22.2x faster than the slowest successful cell (`tenferro-direct`, 127.877 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 29.1x faster than the slowest successful cell (`tenferro-trace`, 133.975 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `julia-base` is 31.5x faster than the slowest successful cell (`tenferro-trace`, 133.975 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 37.2x faster than the slowest successful cell (`tenferro-trace`, 133.975 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/masked_log_softmax_axis1` (f64, threads=4, shape=`1024x64`): `jax-cpu` is 124.9x faster than the slowest successful cell (`tenferro-eager`, 52.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/masked_log_softmax_axis1` (f64, threads=4, shape=`1024x64`): `tenferro-direct` is 135.8x faster than the slowest successful cell (`tenferro-eager`, 52.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 19.9x faster than the slowest successful cell (`pytorch-cpu`, 23.751 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 23.751 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 23.751 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 15.7x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 19.1x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 30.8x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 17.5x faster than the slowest successful cell (`tenferro-trace`, 23.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 19.6x faster than the slowest successful cell (`tenferro-trace`, 23.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`tenferro-trace`, 23.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 18.6x faster than the slowest successful cell (`tenferro-trace`, 23.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 17.7x faster than the slowest successful cell (`pytorch-cpu`, 22.681 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 12.5x faster than the slowest successful cell (`pytorch-cpu`, 22.681 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 12.3x faster than the slowest successful cell (`pytorch-cpu`, 22.681 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 17.9x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 22.4x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 30.5x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 21.5x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 26.0x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 25.7x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 52.3x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 14.2x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 21.7x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 71.0x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 40.3x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 21.4x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 26.8x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 25.7x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 57.6x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 15.2x faster than the slowest successful cell (`tenferro-trace`, 24.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 24.2x faster than the slowest successful cell (`tenferro-trace`, 24.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 58.0x faster than the slowest successful cell (`tenferro-trace`, 24.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 38.9x faster than the slowest successful cell (`tenferro-trace`, 24.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_squares_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 66.5x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `jax-cpu` is 14.2x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `pytorch-cpu` is 37.7x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `tenferro-direct` is 40.1x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `jax-cpu` is 16.1x faster than the slowest successful cell (`tenferro-trace`, 12.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `pytorch-cpu` is 35.6x faster than the slowest successful cell (`tenferro-trace`, 12.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `tenferro-direct` is 32.0x faster than the slowest successful cell (`tenferro-trace`, 12.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 12.8x faster than the slowest successful cell (`julia-base`, 7.644 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 15.2x faster than the slowest successful cell (`julia-base`, 7.644 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `jax-cpu` is 114.0x faster than the slowest successful cell (`tenferro-trace`, 47.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `julia-base` is 39.0x faster than the slowest successful cell (`tenferro-trace`, 47.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 138.1x faster than the slowest successful cell (`tenferro-trace`, 47.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `tenferro-direct` is 121.8x faster than the slowest successful cell (`tenferro-trace`, 47.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `jax-cpu` is 19.9x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 24.1x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 30.0x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `jax-cpu` is 26.0x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 35.6x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 38.5x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `jax-cpu` is 73.3x faster than the slowest successful cell (`tenferro-trace`, 35.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `julia-base` is 27.1x faster than the slowest successful cell (`tenferro-trace`, 35.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 57.2x faster than the slowest successful cell (`tenferro-trace`, 35.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `tenferro-direct` is 34.1x faster than the slowest successful cell (`tenferro-trace`, 35.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 4863.8x faster than the slowest successful cell (`julia-base`, 8.726 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `pytorch-cpu` is 10.7x faster than the slowest successful cell (`julia-base`, 8.815 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-direct` is 13.5x faster than the slowest successful cell (`julia-base`, 8.815 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 4859.6x faster than the slowest successful cell (`julia-base`, 8.815 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 10.0x faster than the slowest successful cell (`jax-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x8x8`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`3x2x2`): `tenferro-direct` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`3x4x4`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x2x2`): `tenferro-direct` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.013 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x4x4`): `tenferro-direct` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 16.0x faster than the slowest successful cell (`jax-cpu`, 0.031 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x2x2`): `tenferro-direct` is 10.0x faster than the slowest successful cell (`jax-cpu`, 0.013 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.032 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=1, shape=`1x2x2`): `tenferro-direct` is 12.6x faster than the slowest successful cell (`jax-cpu`, 0.028 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.020 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.045 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.045 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 11.3x faster than the slowest successful cell (`jax-cpu`, 0.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.020 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.023 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.025 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x2x2,rhs=1`): `tenferro-direct` is 13.9x faster than the slowest successful cell (`jax-cpu`, 0.025 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x4x4,rhs=1`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.018 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`3x4x4,rhs=1`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.022 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `pytorch-cpu` is 15.8x faster than the slowest successful cell (`tenferro-trace`, 5.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `tenferro-direct` is 11.0x faster than the slowest successful cell (`tenferro-trace`, 5.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x4x4,rhs=1`): `tenferro-direct` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.026 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`3x2x2,rhs=1`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.027 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `jax-cpu` is 32.2x faster than the slowest successful cell (`tenferro-trace`, 23.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `pytorch-cpu` is 41.1x faster than the slowest successful cell (`tenferro-trace`, 23.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `tenferro-direct` is 25.9x faster than the slowest successful cell (`tenferro-trace`, 23.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 23.0x faster than the slowest successful cell (`tenferro-trace`, 328.008 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `julia-base` is 26.2x faster than the slowest successful cell (`tenferro-trace`, 328.008 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 29.1x faster than the slowest successful cell (`tenferro-trace`, 328.008 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`768x384`): `jax-cpu` is 237.4x faster than the slowest successful cell (`tenferro-eager`, 2306.715 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 403.2x faster than the slowest successful cell (`tenferro-eager`, 2306.715 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 14.1x faster than the slowest successful cell (`tenferro-trace`, 282.664 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `julia-base` is 17.0x faster than the slowest successful cell (`tenferro-trace`, 282.664 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 13.0x faster than the slowest successful cell (`tenferro-trace`, 282.664 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `jax-cpu` is 50.2x faster than the slowest successful cell (`tenferro-eager`, 2285.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `julia-base` is 259.7x faster than the slowest successful cell (`tenferro-eager`, 2285.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 349.7x faster than the slowest successful cell (`tenferro-eager`, 2285.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `julia-base` is 10.0x faster than the slowest successful cell (`tenferro-direct`, 241.183 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 14.1x faster than the slowest successful cell (`tenferro-direct`, 241.183 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 10.6x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 19.0x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 52.4x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `jax-cpu` is 11.2x faster than the slowest successful cell (`tenferro-trace`, 224.358 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `julia-base` is 11.1x faster than the slowest successful cell (`tenferro-trace`, 224.358 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 13.5x faster than the slowest successful cell (`tenferro-trace`, 224.358 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `jax-cpu` is 11.4x faster than the slowest successful cell (`tenferro-trace`, 233.381 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `julia-base` is 11.8x faster than the slowest successful cell (`tenferro-trace`, 233.381 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 13.9x faster than the slowest successful cell (`tenferro-trace`, 233.381 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 23.0x faster than the slowest successful cell (`tenferro-trace`, 324.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `julia-base` is 25.9x faster than the slowest successful cell (`tenferro-trace`, 324.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 28.2x faster than the slowest successful cell (`tenferro-trace`, 324.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 42.0x faster than the slowest successful cell (`tenferro-trace`, 2332.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `julia-base` is 41.0x faster than the slowest successful cell (`tenferro-trace`, 2332.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 59.7x faster than the slowest successful cell (`tenferro-trace`, 2332.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 47.2x faster than the slowest successful cell (`tenferro-eager`, 5.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 78.6x faster than the slowest successful cell (`tenferro-eager`, 5.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 81.2x faster than the slowest successful cell (`tenferro-eager`, 5.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 106.7x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 23.9x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 26.7x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 75.0x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 124.7x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 322.8x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 10.3x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 17.1x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 21.3x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 82.8x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 112.6x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 156.7x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 12.4x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 203.1x faster than the slowest successful cell (`tenferro-eager`, 36.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 535.4x faster than the slowest successful cell (`tenferro-eager`, 36.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 384.9x faster than the slowest successful cell (`tenferro-eager`, 36.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 86.4x faster than the slowest successful cell (`tenferro-eager`, 96.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 265.5x faster than the slowest successful cell (`tenferro-eager`, 96.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 245.5x faster than the slowest successful cell (`tenferro-eager`, 96.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 215.6x faster than the slowest successful cell (`tenferro-eager`, 43.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 702.1x faster than the slowest successful cell (`tenferro-eager`, 43.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 306.6x faster than the slowest successful cell (`tenferro-eager`, 43.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 89.8x faster than the slowest successful cell (`tenferro-eager`, 119.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 246.5x faster than the slowest successful cell (`tenferro-eager`, 119.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 125.2x faster than the slowest successful cell (`tenferro-eager`, 119.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 217.1x faster than the slowest successful cell (`tenferro-eager`, 51.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 186.6x faster than the slowest successful cell (`tenferro-eager`, 51.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 65.2x faster than the slowest successful cell (`tenferro-eager`, 143.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 72.0x faster than the slowest successful cell (`tenferro-eager`, 143.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 156.2x faster than the slowest successful cell (`tenferro-eager`, 6.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 413.6x faster than the slowest successful cell (`tenferro-eager`, 6.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 386.8x faster than the slowest successful cell (`tenferro-eager`, 6.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 161.1x faster than the slowest successful cell (`tenferro-eager`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 261.8x faster than the slowest successful cell (`tenferro-eager`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 212.1x faster than the slowest successful cell (`tenferro-eager`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 127.1x faster than the slowest successful cell (`tenferro-eager`, 18.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 196.6x faster than the slowest successful cell (`tenferro-eager`, 18.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 409.6x faster than the slowest successful cell (`tenferro-eager`, 18.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 47.7x faster than the slowest successful cell (`tenferro-eager`, 48.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 198.8x faster than the slowest successful cell (`tenferro-eager`, 48.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 241.9x faster than the slowest successful cell (`tenferro-eager`, 48.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 88.5x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 144.5x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 448.0x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 13.2x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 17.4x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 30.9x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 90.5x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 138.7x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 435.0x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 13.0x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 17.0x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 29.5x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 218.2x faster than the slowest successful cell (`tenferro-eager`, 43.765 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 721.0x faster than the slowest successful cell (`tenferro-eager`, 43.765 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 307.4x faster than the slowest successful cell (`tenferro-eager`, 43.765 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 94.6x faster than the slowest successful cell (`tenferro-eager`, 120.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 248.3x faster than the slowest successful cell (`tenferro-eager`, 120.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 124.7x faster than the slowest successful cell (`tenferro-eager`, 120.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 56.9x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 71.4x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 165.1x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 12.4x faster than the slowest successful cell (`tenferro-eager`, 28.595 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 573.3x faster than the slowest successful cell (`tenferro-eager`, 92.812 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 409.5x faster than the slowest successful cell (`tenferro-eager`, 92.812 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 109.4x faster than the slowest successful cell (`tenferro-eager`, 92.812 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 125.4x faster than the slowest successful cell (`tenferro-eager`, 154.498 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 67.8x faster than the slowest successful cell (`tenferro-eager`, 154.498 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 18.1x faster than the slowest successful cell (`tenferro-eager`, 154.498 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/output_reuse/solve_read_into` (f64, threads=4, shape=`768x768,rhs=16`): `pytorch-cpu` is 28.0x faster than the slowest successful cell (`tenferro-direct`, 192.771 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 77065.3x faster than the slowest successful cell (`julia-base`, 130.472 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 72202.9x faster than the slowest successful cell (`julia-base`, 130.182 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/cast_f64_f32` (f64->f32, threads=4, shape=`33554432`): `pytorch-cpu` is 11.5x faster than the slowest successful cell (`tenferro-trace`, 256.474 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/cast_f64_f32` (f64->f32, threads=4, shape=`33554432`): `tenferro-direct` is 11.4x faster than the slowest successful cell (`tenferro-trace`, 256.474 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 85153.0x faster than the slowest successful cell (`julia-base`, 152.679 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 83762.9x faster than the slowest successful cell (`julia-base`, 151.862 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 127787.2x faster than the slowest successful cell (`julia-base`, 211.360 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 123276.8x faster than the slowest successful cell (`julia-base`, 212.406 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 180.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 173.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 179.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 74.5x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 190.3x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 220.7x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 9347.3 GB/s at 0.002 ms. The conservative physical lower bound is 0.168 ms, so the cell is 93.5x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 9244.2 GB/s at 0.002 ms. The conservative physical lower bound is 0.042 ms, so the cell is 23.1x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 158556.1 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1585.6x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 148882.7 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 372.2x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 149713.0 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1497.1x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 148061.5 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 370.2x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 81147.4 GB/s at 0.002 ms. The conservative physical lower bound is 1.342 ms, so the cell is 811.5x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 77897.7 GB/s at 0.002 ms. The conservative physical lower bound is 0.336 ms, so the cell is 194.7x faster than the bound.

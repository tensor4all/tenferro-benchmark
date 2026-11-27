# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/amd-cpu/cpu/public_api/20261127_185020/run.yaml`
- Timestamp: `2026-11-27T18:50:20.363807+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/amd-cpu/cpu/public_api/20261127_185020`.

- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

## CPU Information

- Model: `INTEL(R) XEON(R) PLATINUM 8573C`
- Vendor: `GenuineIntel`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1020-azure-x86_64-with-glibc2.39`
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

- CSV: `data/results/amd-cpu/cpu/public_api/20261127_185020/cpu_public_api_t1_20261127_185020.csv`
- CSV: `data/results/amd-cpu/cpu/public_api/20261127_185020/cpu_public_api_t4_20261127_185020.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 2.705 ± 0.069 | - | 2.687 ± 0.027 | 2.947 ± 0.045 | 3.429 ± 0.049 | 3.537 ± 0.919 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.025 ± 0.013 | - | 2.106 ± 0.029 | 1.574 ± 0.004 | 2.891 ± 0.017 | 5.214 ± 2.398 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 108.973 ± 0.271 | - | 111.103 ± 0.749 | 111.696 ± 1.037 | 105.216 ± 0.847 | 113.579 ± 2.998 | 132.591 ± 0.517 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 40.232 ± 0.434 | - | 39.757 ± 0.350 | 40.400 ± 0.891 | 59.001 ± 0.567 | 114.334 ± 0.664 | 53.278 ± 0.841 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 89.628 ± 0.204 | - | 88.964 ± 0.424 | 84.776 ± 0.892 | 84.614 ± 1.749 | 102.119 ± 0.386 | 105.704 ± 0.327 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 38.819 ± 0.528 | - | 38.616 ± 0.212 | 28.685 ± 0.318 | 44.218 ± 0.153 | 100.210 ± 0.274 | 50.868 ± 1.401 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 26.902 ± 0.160 | - | 26.877 ± 0.395 | 23.021 ± 0.118 | 70.245 ± 0.232 | 26.073 ± 6.244 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 13.703 ± 0.245 | - | 13.584 ± 0.071 | 11.847 ± 0.062 | 35.643 ± 0.456 | 13.462 ± 0.113 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 26.948 ± 0.211 | - | unsupported | 23.801 ± 0.233 | 77.671 ± 0.413 | 28.712 ± 3.798 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 13.746 ± 0.121 | - | unsupported | 12.327 ± 0.095 | 36.803 ± 0.775 | 16.794 ± 3.594 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 6.782 ± 0.033 | - | 7.004 ± 0.559 | 10.588 ± 1.494 | 11.203 ± 0.021 | 10.979 ± 0.065 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 9.503 ± 0.036 | - | 9.681 ± 0.178 | 10.325 ± 0.138 | 19.294 ± 1.186 | 18.307 ± 0.072 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 51.901 ± 1.043 | - | 51.846 ± 0.193 | 99.479 ± 0.380 | 93.299 ± 0.825 | 70.211 ± 10.659 | 68.391 ± 0.346 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 24.745 ± 0.094 | - | 24.771 ± 0.050 | 43.218 ± 0.118 | 47.548 ± 0.677 | 71.484 ± 5.514 | 33.170 ± 0.488 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 90.726 ± 0.526 | - | 90.635 ± 0.492 | 131.312 ± 1.221 | 193.402 ± 1.850 | 112.018 ± 0.198 | 114.375 ± 19.197 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 43.932 ± 0.119 | - | 42.281 ± 1.692 | 54.861 ± 0.423 | 76.696 ± 0.549 | 114.523 ± 3.284 | 55.517 ± 9.588 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 62.463 ± 0.167 | - | 64.104 ± 1.083 | 66.959 ± 0.709 | 64.165 ± 1.311 | 68.485 ± 2.789 | 69.647 ± 0.251 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 23.811 ± 0.289 | - | 23.763 ± 0.475 | 23.715 ± 0.175 | 35.563 ± 0.348 | 69.423 ± 11.220 | 32.010 ± 0.890 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 2.877 ± 0.202 | - | 16.989 ± 0.144 | 16.077 ± 0.083 | 13.990 ± 0.355 | 4.965 ± 0.365 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 0.934 ± 0.080 | - | 7.216 ± 0.175 | 7.979 ± 0.064 | 3.187 ± 0.355 | 6.808 ± 0.262 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 4.986 ± 0.077 | - | 4.888 ± 0.105 | 4.357 ± 0.077 | 5.145 ± 0.119 | 5.730 ± 0.115 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 3.203 ± 0.052 | - | 3.236 ± 0.080 | 2.952 ± 0.030 | 6.690 ± 0.126 | 5.906 ± 1.159 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 3.261 ± 0.046 | - | 3.389 ± 0.073 | 3.625 ± 0.412 | 4.015 ± 0.085 | 3.857 ± 0.073 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 2.671 ± 0.039 | - | 2.775 ± 0.042 | 1.979 ± 0.075 | 3.340 ± 0.087 | 3.244 ± 0.163 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 5.346 ± 0.116 | - | 5.338 ± 0.077 | 3.449 ± 0.090 | 4.471 ± 0.024 | 4.490 ± 0.076 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 5.339 ± 0.078 | - | 5.344 ± 0.021 | 3.439 ± 0.076 | 6.542 ± 0.803 | 7.488 ± 0.095 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 27.132 ± 0.291 | - | 27.077 ± 0.104 | 22.974 ± 0.197 | 70.397 ± 1.037 | 24.179 ± 0.151 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 13.565 ± 0.251 | - | 13.428 ± 0.119 | 11.877 ± 0.047 | 35.753 ± 0.412 | 13.627 ± 3.071 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 50.161 ± 1.013 | - | unsupported | 24.115 ± 0.103 | 64.970 ± 0.171 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 15.101 ± 0.069 | - | unsupported | 12.144 ± 0.166 | 32.860 ± 0.669 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 109.930 ± 2.130 | - | 106.169 ± 0.379 | 110.139 ± 0.589 | 111.820 ± 0.895 | 116.472 ± 0.685 | 139.531 ± 5.772 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 39.609 ± 0.521 | - | 39.246 ± 0.357 | 39.891 ± 0.855 | 61.423 ± 1.096 | 114.788 ± 15.895 | 57.779 ± 2.551 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 119.736 ± 0.151 | - | 118.650 ± 0.278 | 121.071 ± 0.776 | 139.554 ± 0.490 | 127.595 ± 9.425 | 144.104 ± 0.492 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 45.945 ± 0.842 | - | 44.939 ± 0.207 | 47.094 ± 0.519 | 75.191 ± 1.195 | 126.901 ± 7.471 | 62.478 ± 0.754 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 20.103 ± 0.670 | - | 56.130 ± 0.338 | 51.670 ± 0.356 | 55.078 ± 0.233 | 61.918 ± 3.989 | 64.379 ± 6.443 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 8.219 ± 0.050 | - | 22.543 ± 0.381 | 23.463 ± 0.380 | 27.663 ± 0.332 | 62.317 ± 6.105 | 29.448 ± 6.862 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 16.654 ± 0.148 | - | 16.488 ± 0.213 | 34.896 ± 0.382 | 40.286 ± 0.277 | 34.082 ± 4.801 | 39.324 ± 3.886 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 8.137 ± 0.137 | - | 8.193 ± 0.137 | 13.503 ± 0.215 | 21.205 ± 0.100 | 38.820 ± 4.998 | 13.995 ± 8.137 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 34.050 ± 0.325 | - | 33.876 ± 0.180 | 59.826 ± 0.253 | 45.980 ± 0.312 | 32.652 ± 0.569 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 16.652 ± 0.050 | - | 16.618 ± 0.067 | 22.365 ± 0.222 | 24.196 ± 0.134 | 32.543 ± 0.802 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 15.396 ± 0.183 | - | 15.451 ± 0.096 | 32.316 ± 0.234 | 74.082 ± 0.200 | 55.153 ± 4.193 | 61.980 ± 4.340 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 11.010 ± 5.216 | - | 6.569 ± 0.071 | 11.828 ± 0.326 | 38.959 ± 0.576 | 61.472 ± 5.225 | 24.831 ± 8.392 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 122.059 ± 0.609 | - | 119.430 ± 0.380 | 121.438 ± 0.328 | 166.616 ± 1.157 | 129.737 ± 1.717 | 144.867 ± 0.753 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 45.871 ± 1.137 | - | 45.245 ± 0.191 | 46.204 ± 0.196 | 89.205 ± 0.860 | 131.360 ± 1.674 | 60.673 ± 6.077 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 12.200 ± 0.072 | - | 12.090 ± 0.116 | 29.535 ± 0.222 | 33.763 ± 0.214 | 51.548 ± 7.515 | 54.519 ± 5.683 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 4.674 ± 0.114 | - | 4.479 ± 0.167 | 10.803 ± 0.144 | 18.511 ± 0.301 | 52.236 ± 7.159 | 23.949 ± 7.882 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 12.175 ± 0.047 | - | 12.185 ± 0.022 | 22.555 ± 0.167 | 20.746 ± 0.159 | 30.100 ± 4.508 | 33.531 ± 6.709 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 5.192 ± 0.067 | - | 5.199 ± 0.023 | 8.916 ± 0.087 | 10.973 ± 0.147 | 30.209 ± 5.386 | 19.032 ± 6.185 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 16.285 ± 0.101 | - | 16.320 ± 0.069 | 31.585 ± 0.316 | 68.167 ± 0.916 | 68.766 ± 5.775 | 72.846 ± 5.729 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 7.004 ± 0.040 | - | 7.023 ± 0.012 | 11.408 ± 0.171 | 34.913 ± 0.261 | 69.350 ± 8.162 | 30.516 ± 10.825 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 7.974 ± 0.517 | - | 7.795 ± 0.118 | 19.677 ± 0.079 | 44.142 ± 0.226 | 37.032 ± 6.037 | 41.124 ± 5.792 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 3.418 ± 0.026 | - | 3.431 ± 0.010 | 7.996 ± 0.102 | 22.866 ± 0.124 | 37.574 ± 6.698 | 16.464 ± 5.155 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.352 ± 0.031 | 0.366 ± 0.023 | unsupported | unsupported | 0.465 ± 0.017 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.246 ± 0.013 | 0.277 ± 0.018 | unsupported | unsupported | 0.357 ± 0.010 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 122.075 ± 0.614 | - | 119.694 ± 1.240 | 126.797 ± 0.748 | 144.015 ± 2.390 | 131.712 ± 16.228 | 145.494 ± 1.459 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 46.578 ± 0.465 | - | 45.629 ± 1.010 | 45.918 ± 0.993 | 77.355 ± 0.406 | 130.457 ± 3.898 | 62.536 ± 1.041 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 119.780 ± 0.244 | - | 119.385 ± 1.049 | 126.072 ± 0.665 | 144.796 ± 0.593 | 132.011 ± 1.773 | 146.842 ± 2.341 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 46.299 ± 0.654 | - | 45.839 ± 0.278 | 46.264 ± 5.141 | 77.377 ± 1.013 | 133.689 ± 3.733 | 60.867 ± 2.810 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 120.378 ± 0.238 | - | 119.495 ± 0.591 | 120.635 ± 0.682 | 141.282 ± 0.869 | 128.740 ± 1.281 | 144.414 ± 0.369 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 45.743 ± 0.602 | - | 44.845 ± 0.414 | 46.895 ± 0.373 | 74.970 ± 0.941 | 128.802 ± 6.408 | 61.116 ± 6.204 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 109.303 ± 0.491 | - | 106.980 ± 2.966 | 110.616 ± 1.684 | 112.716 ± 8.066 | 116.026 ± 1.397 | 134.424 ± 0.684 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 39.450 ± 0.641 | - | 39.488 ± 0.543 | 39.759 ± 0.547 | 62.051 ± 0.888 | 115.702 ± 3.790 | 54.904 ± 7.377 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 67.162 ± 0.190 | - | 67.159 ± 0.632 | 35.999 ± 0.847 | 77.195 ± 0.054 | 117.680 ± 0.597 | 88.010 ± 13.915 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 28.449 ± 1.045 | - | 29.182 ± 2.724 | 14.962 ± 5.047 | 39.539 ± 0.189 | 118.216 ± 4.191 | 39.361 ± 13.415 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 32.976 ± 0.130 | - | 32.943 ± 0.169 | 26.001 ± 0.345 | 88.078 ± 0.203 | 28.540 ± 0.180 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 10.780 ± 0.090 | - | 10.778 ± 0.055 | 9.761 ± 6.196 | 44.519 ± 0.333 | 28.829 ± 0.412 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 2.278 ± 0.097 | - | 2.277 ± 0.150 | 18.437 ± 0.396 | 1.809 ± 0.019 | 1.894 ± 0.146 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.912 ± 0.018 | - | 0.914 ± 0.007 | 12.296 ± 0.178 | 1.057 ± 0.038 | 2.265 ± 0.214 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 1.232 ± 0.053 | - | 1.236 ± 0.039 | 3.064 ± 0.162 | 5.353 ± 0.295 | 1.430 ± 0.120 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 0.620 ± 0.014 | - | 0.630 ± 0.014 | 1.927 ± 0.138 | 2.985 ± 1.267 | 1.515 ± 0.372 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 32.750 ± 0.199 | - | 33.111 ± 2.650 | 26.035 ± 0.297 | 88.139 ± 0.514 | 29.078 ± 0.189 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 10.730 ± 0.731 | - | 10.809 ± 0.134 | 9.889 ± 0.108 | 44.635 ± 0.200 | 28.284 ± 0.232 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 2.249 ± 0.302 | - | 2.227 ± 0.170 | 18.380 ± 0.226 | 1.804 ± 0.023 | 1.906 ± 0.352 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.905 ± 0.006 | - | 0.916 ± 0.021 | 11.986 ± 0.208 | 1.668 ± 0.076 | 2.310 ± 0.482 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 10.352 ± 0.081 | - | 10.377 ± 0.175 | 22.392 ± 2.940 | 23.163 ± 0.143 | 12.132 ± 0.286 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 4.119 ± 0.169 | - | 4.342 ± 0.389 | 9.666 ± 0.986 | 12.037 ± 4.918 | 12.200 ± 0.137 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 16.400 ± 0.169 | - | 16.373 ± 0.157 | 17.654 ± 0.887 | 42.186 ± 0.128 | 18.182 ± 0.220 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 7.978 ± 0.099 | - | 7.980 ± 0.171 | 8.497 ± 0.106 | 21.503 ± 0.052 | 18.198 ± 0.160 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 1.157 ± 0.011 | - | 1.168 ± 0.030 | 1.306 ± 0.036 | 1.325 ± 0.040 | 1.216 ± 0.016 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.614 ± 0.017 | - | 0.619 ± 0.016 | 0.865 ± 0.026 | 1.360 ± 0.105 | 1.239 ± 0.049 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.155 ± 0.029 | - | 1.175 ± 0.026 | 1.171 ± 0.005 | 1.672 ± 0.064 | 1.232 ± 0.037 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.599 ± 0.019 | - | 0.595 ± 0.019 | 0.620 ± 0.018 | 1.435 ± 0.409 | 1.244 ± 0.083 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 16.572 ± 0.214 | - | 16.327 ± 0.153 | 17.711 ± 0.047 | 26.585 ± 0.196 | 18.098 ± 0.290 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 7.969 ± 0.071 | - | 8.057 ± 0.087 | 8.574 ± 0.179 | 14.131 ± 7.430 | 18.040 ± 0.135 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 1.165 ± 0.017 | - | 1.168 ± 0.018 | 1.271 ± 0.012 | 1.323 ± 0.039 | 1.211 ± 0.019 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.611 ± 0.013 | - | 0.618 ± 0.014 | 0.821 ± 0.005 | 1.375 ± 0.081 | 1.220 ± 0.034 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.179 ± 0.021 | - | 1.167 ± 0.017 | 1.181 ± 0.007 | 1.504 ± 0.134 | 1.197 ± 0.020 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.584 ± 0.022 | - | 0.599 ± 0.010 | 0.617 ± 0.013 | 1.445 ± 0.327 | 1.219 ± 0.056 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 1.178 ± 0.019 | 1.174 ± 0.009 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.613 ± 0.025 | 0.612 ± 0.005 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 37.713 ± 1.198 | - | 37.619 ± 0.062 | 55.458 ± 0.557 | 66.012 ± 0.450 | 58.188 ± 12.323 | 67.575 ± 4.168 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 14.260 ± 0.087 | - | 14.558 ± 3.835 | 21.221 ± 0.120 | 38.562 ± 0.661 | 68.518 ± 8.914 | 33.608 ± 6.558 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 146.753 ± 0.666 | - | 144.959 ± 0.462 | 148.371 ± 1.284 | 207.122 ± 0.642 | 216.425 ± 0.484 | 215.329 ± 1.161 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 47.268 ± 1.308 | - | 47.204 ± 2.743 | 45.908 ± 0.369 | 109.200 ± 0.799 | 211.321 ± 0.730 | 87.736 ± 13.403 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 125.075 ± 0.549 | - | 124.816 ± 1.019 | 136.207 ± 5.788 | 134.135 ± 2.117 | 135.742 ± 2.294 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 49.887 ± 13.038 | - | 49.006 ± 0.278 | 49.878 ± 0.492 | 72.156 ± 4.931 | 129.055 ± 0.869 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 112.489 ± 0.948 | - | 110.107 ± 1.242 | 114.761 ± 1.270 | 118.329 ± 0.767 | 117.738 ± 1.090 | 140.932 ± 7.084 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 40.170 ± 0.736 | - | 39.824 ± 0.212 | 40.792 ± 0.386 | 64.486 ± 0.538 | 120.807 ± 3.998 | 56.499 ± 27.499 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 13.569 ± 0.083 | - | 13.667 ± 0.065 | 32.607 ± 0.110 | 74.740 ± 0.383 | 57.261 ± 7.489 | 61.511 ± 4.308 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 5.452 ± 0.033 | - | 5.790 ± 0.234 | 11.736 ± 0.128 | 39.379 ± 0.357 | 61.760 ± 5.705 | 31.959 ± 7.837 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 127.366 ± 2.487 | - | 124.236 ± 0.659 | 116.562 ± 0.434 | 125.582 ± 0.753 | 164.688 ± 2.078 | 171.051 ± 1.151 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 43.738 ± 4.894 | - | 42.051 ± 0.322 | 41.388 ± 3.024 | 68.706 ± 1.223 | 170.809 ± 0.963 | 71.187 ± 13.678 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 120.585 ± 0.591 | - | 118.982 ± 0.418 | 121.888 ± 0.336 | 139.303 ± 0.845 | 127.854 ± 1.092 | 143.439 ± 0.929 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 45.891 ± 0.987 | - | 44.879 ± 0.270 | 47.122 ± 0.353 | 74.057 ± 0.349 | 129.060 ± 17.355 | 61.193 ± 4.025 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 32.168 ± 0.151 | - | 32.214 ± 0.039 | 35.839 ± 0.123 | 32.587 ± 0.212 | 63.367 ± 6.158 | 69.928 ± 5.801 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 14.434 ± 0.268 | - | 14.220 ± 0.212 | 13.338 ± 0.114 | 18.095 ± 0.333 | 63.161 ± 7.853 | 29.361 ± 10.210 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 124.948 ± 0.564 | 124.224 ± 0.947 | unsupported | 135.616 ± 0.776 | 133.786 ± 0.603 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 49.204 ± 0.144 | 48.849 ± 0.582 | unsupported | 49.811 ± 0.285 | 71.912 ± 0.560 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 1.234 ± 0.034 | - | 1.271 ± 0.019 | 1.215 ± 0.012 | 1.233 ± 0.018 | 8.274 ± 4.957 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.635 ± 0.014 | - | 0.643 ± 0.019 | 0.650 ± 0.016 | 1.242 ± 0.022 | 8.828 ± 4.965 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 1.189 ± 0.011 | - | 1.204 ± 0.016 | 1.241 ± 0.078 | 1.255 ± 0.072 | 8.790 ± 5.289 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.713 ± 0.039 | - | 0.664 ± 0.027 | 0.638 ± 0.019 | 0.718 ± 0.041 | 2.201 ± 0.729 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.884 ± 0.036 | - | unsupported | 1.940 ± 0.058 | 1.826 ± 0.053 | 8.856 ± 6.282 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 1.003 ± 0.060 | - | unsupported | 1.014 ± 0.045 | 1.505 ± 0.114 | 9.277 ± 4.051 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.503 ± 0.031 | - | 0.687 ± 0.015 | 0.833 ± 0.006 | 0.413 ± 0.014 | 1.152 ± 0.121 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.416 ± 0.057 | - | 0.525 ± 0.016 | 0.480 ± 0.015 | 0.264 ± 0.016 | 1.164 ± 0.122 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 1.200 ± 0.011 | - | 1.211 ± 0.027 | 1.890 ± 0.025 | 1.299 ± 0.120 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.648 ± 0.014 | - | 0.645 ± 0.017 | 0.990 ± 0.013 | 0.725 ± 0.045 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 1.264 ± 0.023 | - | 1.237 ± 0.007 | 1.264 ± 0.049 | 1.290 ± 0.034 | 2.553 ± 12.772 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.633 ± 0.011 | - | 0.641 ± 0.021 | 0.644 ± 0.018 | 0.691 ± 0.017 | 8.227 ± 8.653 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.785 ± 0.010 | - | 1.036 ± 0.017 | 1.403 ± 0.014 | 0.767 ± 0.026 | 1.964 ± 0.218 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 1.007 ± 0.008 | - | 1.158 ± 0.075 | 1.206 ± 0.014 | 0.752 ± 0.070 | 1.913 ± 0.312 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 2.048 ± 0.181 | - | 0.001 ± 0.000 | 2.214 ± 0.383 | 1.945 ± 0.136 | 8.902 ± 7.013 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.954 ± 0.036 | - | 0.002 ± 0.000 | 0.952 ± 0.066 | 2.054 ± 1.275 | 10.032 ± 1.478 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.068 ± 0.002 | - | 0.069 ± 0.000 | 0.082 ± 0.001 | 0.095 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.122 ± 0.001 | - | 0.123 ± 0.002 | 0.097 ± 0.001 | 0.136 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.244 ± 0.007 | - | 0.246 ± 0.004 | 0.326 ± 0.011 | 0.268 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.042 ± 0.000 | - | 0.044 ± 0.000 | 0.141 ± 0.009 | 0.105 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.041 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.051 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.351 ± 0.014 | - | 0.348 ± 0.014 | 1.096 ± 0.002 | 0.664 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.017 ± 0.001 | - | 0.018 ± 0.000 | 0.032 ± 0.000 | 0.059 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.002 | 0.051 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.068 ± 0.002 | - | 0.097 ± 0.004 | 0.059 ± 0.002 | 0.088 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.121 ± 0.001 | - | 0.153 ± 0.004 | 0.102 ± 0.001 | 0.123 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.085 ± 0.001 | - | 0.088 ± 0.002 | 0.317 ± 0.009 | 0.297 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.061 ± 0.001 | - | 0.066 ± 0.018 | 0.107 ± 0.001 | 0.129 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.000 | 0.026 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.013 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.184 ± 0.011 | - | 0.188 ± 0.020 | 0.767 ± 0.010 | 2.922 ± 0.992 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.015 ± 0.000 | - | 0.020 ± 0.002 | 0.046 ± 0.001 | 0.049 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.018 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.491 ± 0.066 | - | 0.458 ± 0.103 | 0.718 ± 0.011 | 0.764 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 2.193 ± 0.063 | - | 2.112 ± 0.002 | 2.214 ± 0.021 | 2.583 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 7.322 ± 1.406 | - | 6.902 ± 0.527 | 6.948 ± 0.022 | 8.652 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 1.827 ± 0.020 | - | 1.828 ± 0.115 | 1.536 ± 0.010 | 2.415 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.004 ± 0.001 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.102 ± 0.003 | - | 0.102 ± 0.003 | 0.096 ± 0.011 | 0.142 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.009 ± 0.000 | - | 0.011 ± 0.003 | 0.015 ± 0.002 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.005 ± 0.001 | - | 0.006 ± 0.000 | 0.008 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.013 ± 0.003 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.022 ± 0.000 | - | 0.023 ± 0.000 | 0.028 ± 0.001 | 0.068 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.005 ± 0.004 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.011 ± 0.000 | - | 0.012 ± 0.018 | 0.015 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.028 ± 0.000 | - | 0.030 ± 0.000 | 0.035 ± 0.001 | 0.077 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 14.654 ± 0.063 | - | 14.692 ± 0.135 | 12.235 ± 0.126 | 20.446 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.003 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.801 ± 0.003 | - | 0.804 ± 0.007 | 0.702 ± 0.032 | 0.860 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.024 ± 0.000 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.056 ± 0.000 | - | 0.057 ± 0.000 | 0.061 ± 0.000 | 0.116 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.231 ± 0.009 | - | 0.235 ± 0.019 | 0.645 ± 0.081 | 0.818 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 0.835 ± 0.706 | - | 0.834 ± 0.941 | 2.237 ± 0.064 | 2.958 ± 0.234 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 2.508 ± 0.027 | - | 2.510 ± 0.020 | 8.206 ± 0.172 | 7.255 ± 0.184 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 1.860 ± 0.017 | - | 1.853 ± 0.037 | 1.340 ± 0.030 | 3.753 ± 0.137 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.019 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.102 ± 0.003 | - | 0.123 ± 0.004 | 0.120 ± 0.008 | 0.168 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.001 | 0.009 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.010 ± 0.000 | - | 0.014 ± 0.000 | 0.014 ± 0.001 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.009 ± 0.000 | - | 0.013 ± 0.000 | 0.013 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.022 ± 0.000 | - | 0.029 ± 0.001 | 0.027 ± 0.001 | 0.061 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.011 ± 0.000 | - | 0.015 ± 0.000 | 0.015 ± 0.002 | 0.025 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.029 ± 0.000 | - | 0.043 ± 0.006 | 0.034 ± 0.000 | 0.077 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 6.054 ± 0.071 | - | 6.071 ± 0.068 | 10.299 ± 0.100 | 32.233 ± 8.729 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | 0.010 ± 0.002 | 0.017 ± 0.002 | 0.019 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.308 ± 0.009 | - | 0.314 ± 0.010 | 0.934 ± 0.397 | 1.991 ± 0.252 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.019 ± 0.001 | - | 0.025 ± 0.000 | 0.024 ± 0.001 | 0.035 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.058 ± 0.002 | - | 0.070 ± 0.001 | 0.061 ± 0.004 | 0.111 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.408 ± 0.010 | - | 0.415 ± 0.009 | 0.453 ± 0.016 | 0.414 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.887 ± 0.009 | - | 1.881 ± 0.011 | 1.746 ± 0.016 | 1.843 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 6.199 ± 0.041 | - | 6.239 ± 0.046 | 5.117 ± 0.053 | 6.184 ± 0.143 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.582 ± 0.010 | - | 1.526 ± 0.004 | 0.860 ± 0.014 | 1.251 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.081 ± 0.001 | - | 0.083 ± 0.004 | 0.061 ± 0.008 | 0.115 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.012 ± 0.001 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.007 ± 0.000 | - | 0.010 ± 0.002 | 0.011 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.020 ± 0.000 | - | 0.021 ± 0.000 | 0.023 ± 0.002 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.002 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.026 ± 0.001 | - | 0.027 ± 0.000 | 0.027 ± 0.000 | 0.042 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 12.168 ± 0.092 | - | 12.138 ± 0.925 | 7.076 ± 0.209 | 9.358 ± 0.112 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.661 ± 0.006 | - | 0.663 ± 0.009 | 0.433 ± 0.010 | 0.683 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.016 ± 0.000 | - | 0.018 ± 0.000 | 0.019 ± 0.000 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.050 ± 0.001 | - | 0.052 ± 0.010 | 0.046 ± 0.000 | 0.094 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.408 ± 0.009 | - | 0.603 ± 0.015 | 0.603 ± 0.152 | 0.404 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 0.731 ± 0.008 | - | 0.732 ± 0.010 | 1.756 ± 0.184 | 1.298 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 2.224 ± 0.026 | - | 2.228 ± 0.019 | 6.512 ± 0.602 | 3.972 ± 0.103 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.427 ± 0.015 | - | 1.543 ± 0.065 | 0.886 ± 0.087 | 1.994 ± 0.313 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.083 ± 0.004 | - | 0.101 ± 0.004 | 0.082 ± 0.003 | 0.107 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.013 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.004 ± 0.002 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.008 ± 0.000 | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.020 ± 0.000 | - | 0.026 ± 0.000 | 0.023 ± 0.000 | 0.036 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.009 ± 0.000 | - | 0.014 ± 0.003 | 0.014 ± 0.001 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.026 ± 0.000 | - | 0.034 ± 0.000 | 0.028 ± 0.000 | 0.047 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 4.863 ± 0.498 | - | 4.393 ± 0.017 | 7.416 ± 0.755 | 13.103 ± 2.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.014 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.243 ± 0.003 | - | 0.242 ± 0.010 | 0.584 ± 0.014 | 0.400 ± 0.066 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.016 ± 0.000 | - | 0.023 ± 0.000 | 0.020 ± 0.000 | 0.027 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.051 ± 0.001 | - | 0.063 ± 0.001 | 0.051 ± 0.002 | 0.075 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.277 ± 0.008 | - | 0.283 ± 0.006 | 0.211 ± 0.012 | 0.408 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 1.555 ± 0.009 | - | 1.565 ± 0.011 | 0.946 ± 0.011 | 1.811 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 4.924 ± 0.047 | - | 4.888 ± 0.092 | 2.877 ± 0.018 | 6.093 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.906 ± 0.007 | - | 0.895 ± 0.002 | 0.448 ± 0.009 | 1.178 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.062 ± 0.001 | - | 0.064 ± 0.000 | 0.037 ± 0.001 | 0.127 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.016 ± 0.003 | - | 0.018 ± 0.000 | 0.018 ± 0.001 | 0.032 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.020 ± 0.000 | - | 0.023 ± 0.008 | 0.019 ± 0.000 | 0.039 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 7.176 ± 0.037 | - | 7.102 ± 0.022 | 3.627 ± 0.022 | 9.332 ± 0.070 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.494 ± 0.010 | - | 0.499 ± 0.013 | 0.224 ± 0.010 | 0.652 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.039 ± 0.000 | - | 0.041 ± 0.000 | 0.030 ± 0.000 | 0.117 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.281 ± 0.007 | - | 0.433 ± 0.013 | 0.295 ± 0.009 | 0.375 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 0.604 ± 0.017 | - | 0.598 ± 0.005 | 0.944 ± 0.003 | 1.739 ± 0.388 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 1.732 ± 0.076 | - | 1.743 ± 0.089 | 3.196 ± 0.020 | 3.962 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.905 ± 0.003 | - | 1.021 ± 0.122 | 0.602 ± 0.014 | 1.582 ± 0.094 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.007 ± 0.001 | 0.011 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.063 ± 0.000 | - | 0.074 ± 0.001 | 0.042 ± 0.002 | 0.104 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.006 ± 0.003 | - | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.018 ± 0.000 | - | 0.021 ± 0.000 | 0.016 ± 0.001 | 0.031 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.010 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.008 ± 0.002 | - | 0.011 ± 0.000 | 0.015 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.021 ± 0.000 | - | 0.027 ± 0.003 | 0.019 ± 0.000 | 0.039 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 2.415 ± 0.080 | - | 2.441 ± 0.082 | 4.705 ± 0.019 | 14.999 ± 4.021 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.180 ± 0.006 | - | 0.183 ± 0.011 | 0.255 ± 0.032 | 0.408 ± 0.060 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.013 ± 0.000 | - | 0.018 ± 0.002 | 0.019 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.040 ± 0.000 | - | 0.048 ± 0.001 | 0.030 ± 0.000 | 0.097 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.115 ± 0.004 | - | 0.119 ± 0.003 | 0.175 ± 0.008 | 0.126 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.236 ± 0.009 | - | 0.234 ± 0.005 | 0.224 ± 0.010 | 0.199 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.622 ± 0.013 | - | 0.634 ± 0.027 | 0.707 ± 0.013 | 0.551 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.121 ± 0.016 | - | 0.120 ± 0.007 | 0.129 ± 0.008 | 0.180 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.011 ± 0.000 | - | 0.012 ± 0.000 | 0.018 ± 0.001 | 0.051 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.011 ± 0.001 | - | 0.013 ± 0.001 | 0.020 ± 0.001 | 0.055 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.011 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.012 ± 0.001 | 0.021 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.015 ± 0.000 | 0.042 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.016 ± 0.001 | 0.044 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 1.003 ± 0.094 | - | 0.924 ± 0.080 | 1.124 ± 0.029 | 1.218 ± 0.091 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.055 ± 0.012 | - | 0.057 ± 0.001 | 0.067 ± 0.001 | 0.107 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.042 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.018 ± 0.001 | 0.046 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.117 ± 0.001 | - | 0.172 ± 0.011 | 0.143 ± 0.005 | 0.119 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.233 ± 0.011 | - | 0.301 ± 0.010 | 0.198 ± 0.017 | 0.199 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.262 ± 0.010 | - | 0.269 ± 0.016 | 0.407 ± 0.017 | 0.788 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.192 ± 0.006 | - | 0.195 ± 0.005 | 0.135 ± 0.005 | 0.180 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.016 ± 0.003 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.015 ± 0.000 | - | 0.019 ± 0.001 | 0.025 ± 0.001 | 0.047 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.017 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.013 ± 0.002 | - | 0.019 ± 0.001 | 0.027 ± 0.001 | 0.041 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.018 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.006 ± 0.002 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.022 ± 0.000 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.004 ± 0.002 | - | 0.006 ± 0.000 | 0.018 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.023 ± 0.001 | 0.023 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.428 ± 0.019 | - | 0.434 ± 0.013 | 0.550 ± 0.025 | 1.012 ± 0.194 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.020 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.043 ± 0.011 | - | 0.045 ± 0.010 | 0.061 ± 0.001 | 0.134 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.020 ± 0.001 | 0.032 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.025 ± 0.002 | 0.038 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.265 ± 0.007 | - | 0.268 ± 0.006 | 0.198 ± 0.006 | 0.276 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.709 ± 0.009 | - | 0.712 ± 0.009 | 0.651 ± 0.003 | 0.725 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 2.133 ± 0.023 | - | 2.136 ± 0.016 | 1.541 ± 0.019 | 1.920 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.411 ± 0.007 | - | 0.408 ± 0.008 | 0.343 ± 0.015 | 0.568 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.029 ± 0.001 | - | 0.031 ± 0.002 | 0.017 ± 0.003 | 0.055 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.031 ± 0.002 | - | 0.032 ± 0.000 | 0.016 ± 0.000 | 0.050 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.004 ± 0.002 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.010 ± 0.002 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 3.184 ± 0.021 | - | 3.191 ± 0.155 | 2.927 ± 0.022 | 4.418 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.216 ± 0.008 | - | 0.216 ± 0.003 | 0.087 ± 0.002 | 0.187 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.007 ± 0.015 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.019 ± 0.011 | - | 0.020 ± 0.000 | 0.019 ± 0.000 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.272 ± 0.013 | - | 0.349 ± 0.013 | 0.264 ± 0.003 | 0.276 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.280 ± 0.008 | - | 0.282 ± 0.013 | 0.809 ± 0.046 | 0.728 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 0.762 ± 0.005 | - | 0.766 ± 0.008 | 1.853 ± 0.013 | 1.291 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.421 ± 0.016 | - | 0.511 ± 0.011 | 0.323 ± 0.005 | 1.259 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.001 | 0.015 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.031 ± 0.001 | - | 0.040 ± 0.001 | 0.024 ± 0.002 | 0.045 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.033 ± 0.004 | - | 0.047 ± 0.003 | 0.026 ± 0.001 | 0.042 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.006 ± 0.000 | - | 0.009 ± 0.002 | 0.011 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.016 ± 0.001 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.011 ± 0.000 | - | 0.015 ± 0.000 | 0.017 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 1.412 ± 0.050 | - | 1.429 ± 0.041 | 2.456 ± 0.047 | 10.280 ± 0.074 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.107 ± 0.002 | - | 0.109 ± 0.004 | 0.123 ± 0.001 | 0.252 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.008 ± 0.000 | - | 0.012 ± 0.000 | 0.015 ± 0.001 | 0.019 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.019 ± 0.000 | - | 0.025 ± 0.000 | 0.024 ± 0.001 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.144 ± 0.001 | - | 0.122 ± 0.002 | 0.076 ± 0.003 | 0.257 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.221 ± 0.007 | - | 0.209 ± 0.040 | 0.102 ± 0.010 | 0.342 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.606 ± 0.018 | - | 0.579 ± 0.014 | 0.682 ± 0.030 | 0.730 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.078 ± 0.004 | - | 0.083 ± 0.004 | 0.096 ± 0.000 | 0.162 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.002 ± 0.001 | - | 0.007 ± 0.000 | 0.013 ± 0.001 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.008 ± 0.000 | - | 0.012 ± 0.000 | 0.021 ± 0.001 | 0.048 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.002 | 0.013 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.004 ± 0.002 | - | 0.008 ± 0.000 | 0.019 ± 0.005 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.020 ± 0.003 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.570 ± 0.018 | - | 0.663 ± 0.024 | 0.734 ± 0.008 | 0.951 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.043 ± 0.002 | - | 0.047 ± 0.006 | 0.064 ± 0.000 | 0.101 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.007 ± 0.001 | 0.014 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.019 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.147 ± 0.003 | - | 0.183 ± 0.013 | 0.116 ± 0.002 | 0.257 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.156 ± 0.009 | - | 0.160 ± 0.008 | 0.166 ± 0.019 | 0.337 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.316 ± 0.007 | - | 0.343 ± 0.011 | 0.485 ± 0.027 | 0.939 ± 0.093 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.133 ± 0.007 | - | 0.144 ± 0.006 | 0.109 ± 0.001 | 0.140 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.006 ± 0.002 | - | 0.010 ± 0.000 | 0.019 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.009 ± 0.000 | - | 0.016 ± 0.001 | 0.027 ± 0.001 | 0.036 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.009 ± 0.000 | 0.019 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.011 ± 0.001 | 0.022 ± 0.002 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.011 ± 0.000 | 0.019 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.010 ± 0.000 | 0.019 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.012 ± 0.000 | 0.026 ± 0.001 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.019 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.010 ± 0.000 | 0.019 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.012 ± 0.000 | 0.026 ± 0.004 | 0.016 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.274 ± 0.003 | - | 0.371 ± 0.019 | 0.388 ± 0.016 | 0.745 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.004 ± 0.003 | - | 0.013 ± 0.000 | 0.021 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.030 ± 0.001 | - | 0.048 ± 0.007 | 0.054 ± 0.001 | 0.127 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.011 ± 0.000 | 0.021 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.007 ± 0.000 | - | 0.015 ± 0.000 | 0.027 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.062 ± 0.008 | - | 1.065 ± 0.007 | 1.342 ± 0.008 | 1.793 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 3.814 ± 0.011 | - | 3.815 ± 0.005 | 4.981 ± 0.451 | 4.767 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 12.644 ± 0.055 | - | 12.606 ± 0.058 | 15.835 ± 0.081 | 13.865 ± 0.231 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 3.783 ± 0.062 | - | 3.777 ± 0.051 | 2.687 ± 0.065 | 3.583 ± 0.103 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.186 ± 0.004 | - | 0.188 ± 0.009 | 0.170 ± 0.010 | 0.204 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.007 ± 0.000 | - | 0.008 ± 0.013 | 0.013 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.161 ± 0.002 | - | 0.163 ± 0.002 | 0.126 ± 0.005 | 0.199 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.015 ± 0.000 | - | 0.017 ± 0.001 | 0.028 ± 0.001 | 0.032 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.008 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.014 ± 0.000 | - | 0.015 ± 0.000 | 0.022 ± 0.002 | 0.036 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.038 ± 0.000 | - | 0.040 ± 0.002 | 0.059 ± 0.003 | 0.055 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.023 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.017 ± 0.001 | - | 0.019 ± 0.003 | 0.027 ± 0.001 | 0.044 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.050 ± 0.000 | - | 0.052 ± 0.002 | 0.070 ± 0.009 | 0.070 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 30.384 ± 0.227 | - | 30.313 ± 0.275 | 21.467 ± 0.202 | 27.927 ± 0.185 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.011 ± 0.000 | - | 0.012 ± 0.000 | 0.018 ± 0.000 | 0.033 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.453 ± 0.080 | - | 1.465 ± 0.007 | 1.264 ± 0.021 | 1.408 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.032 ± 0.000 | - | 0.034 ± 0.000 | 0.048 ± 0.002 | 0.053 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.098 ± 0.002 | - | 0.099 ± 0.003 | 0.128 ± 0.007 | 0.146 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 0.464 ± 0.012 | - | 0.487 ± 0.222 | 1.295 ± 0.013 | 1.817 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 1.466 ± 0.057 | - | 1.446 ± 0.013 | 4.928 ± 0.046 | 2.713 ± 0.185 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 4.357 ± 0.020 | - | 4.386 ± 0.047 | 17.069 ± 0.061 | 7.293 ± 0.076 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 3.850 ± 0.077 | - | 3.820 ± 0.045 | 2.974 ± 0.052 | 5.474 ± 0.107 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.001 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.189 ± 0.003 | - | 0.218 ± 0.010 | 0.170 ± 0.010 | 0.212 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.009 ± 0.002 | - | 0.013 ± 0.002 | 0.013 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.172 ± 0.015 | - | 0.192 ± 0.004 | 0.179 ± 0.013 | 0.187 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.016 ± 0.000 | - | 0.020 ± 0.000 | 0.026 ± 0.001 | 0.033 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.017 ± 0.001 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.014 ± 0.000 | - | 0.022 ± 0.000 | 0.022 ± 0.001 | 0.028 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.039 ± 0.003 | - | 0.054 ± 0.007 | 0.055 ± 0.002 | 0.066 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.008 ± 0.000 | - | 0.012 ± 0.001 | 0.018 ± 0.001 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.018 ± 0.003 | - | 0.028 ± 0.000 | 0.027 ± 0.000 | 0.038 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.052 ± 0.001 | - | 0.063 ± 0.010 | 0.072 ± 0.009 | 0.083 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 12.971 ± 0.463 | - | 12.854 ± 0.064 | 23.469 ± 0.390 | 73.964 ± 26.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.026 ± 0.001 | 0.031 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 0.554 ± 0.010 | - | 0.554 ± 0.006 | 1.253 ± 0.015 | 0.798 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.033 ± 0.000 | - | 0.041 ± 0.001 | 0.047 ± 0.003 | 0.052 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.099 ± 0.003 | - | 0.117 ± 0.003 | 0.127 ± 0.002 | 0.137 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.889 ± 0.005 | - | 0.895 ± 0.003 | 1.050 ± 0.276 | 1.564 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 3.297 ± 0.021 | - | 3.314 ± 0.033 | 4.107 ± 0.078 | 3.657 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 10.910 ± 0.020 | - | 10.903 ± 0.075 | 11.867 ± 0.048 | 11.297 ± 0.082 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 2.194 ± 0.035 | - | 2.174 ± 0.013 | 1.645 ± 0.022 | 1.983 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.134 ± 0.004 | - | 0.133 ± 0.005 | 0.124 ± 0.014 | 0.160 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.001 | 0.022 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.101 ± 0.001 | - | 0.103 ± 0.006 | 0.080 ± 0.000 | 0.146 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.013 ± 0.000 | - | 0.014 ± 0.000 | 0.021 ± 0.001 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.013 ± 0.000 | - | 0.014 ± 0.000 | 0.021 ± 0.001 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.033 ± 0.000 | - | 0.034 ± 0.003 | 0.041 ± 0.001 | 0.046 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.016 ± 0.001 | - | 0.017 ± 0.000 | 0.024 ± 0.001 | 0.031 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.043 ± 0.000 | - | 0.045 ± 0.004 | 0.050 ± 0.001 | 0.059 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 17.555 ± 0.069 | - | 17.438 ± 0.118 | 13.143 ± 0.082 | 15.821 ± 0.083 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.016 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 1.042 ± 0.005 | - | 1.047 ± 0.006 | 0.839 ± 0.009 | 0.992 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.028 ± 0.000 | - | 0.030 ± 0.002 | 0.040 ± 0.002 | 0.043 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.086 ± 0.001 | - | 0.087 ± 0.004 | 0.096 ± 0.008 | 0.143 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.899 ± 0.007 | - | 0.945 ± 0.277 | 1.015 ± 0.017 | 1.143 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 1.238 ± 0.012 | - | 1.226 ± 0.024 | 4.050 ± 0.029 | 2.112 ± 0.283 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 3.731 ± 0.127 | - | 3.666 ± 0.159 | 13.171 ± 0.042 | 5.831 ± 1.054 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 2.186 ± 0.013 | - | 2.201 ± 0.033 | 1.798 ± 0.011 | 3.979 ± 1.908 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.001 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.136 ± 0.003 | - | 0.153 ± 0.005 | 0.112 ± 0.003 | 0.153 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.103 ± 0.003 | - | 0.126 ± 0.009 | 0.100 ± 0.003 | 0.139 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.014 ± 0.001 | - | 0.021 ± 0.001 | 0.020 ± 0.001 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.016 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.014 ± 0.000 | - | 0.017 ± 0.000 | 0.020 ± 0.001 | 0.031 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.034 ± 0.001 | - | 0.048 ± 0.001 | 0.040 ± 0.002 | 0.054 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.017 ± 0.000 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.016 ± 0.003 | - | 0.022 ± 0.000 | 0.023 ± 0.001 | 0.028 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.045 ± 0.003 | - | 0.053 ± 0.001 | 0.050 ± 0.000 | 0.056 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 7.027 ± 0.224 | - | 7.040 ± 0.070 | 14.181 ± 0.368 | 28.783 ± 9.502 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.009 ± 0.000 | - | 0.014 ± 0.001 | 0.023 ± 0.002 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 0.371 ± 0.010 | - | 0.377 ± 0.009 | 0.834 ± 0.024 | 0.513 ± 0.067 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.029 ± 0.000 | - | 0.036 ± 0.001 | 0.039 ± 0.000 | 0.048 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.087 ± 0.001 | - | 0.102 ± 0.001 | 0.093 ± 0.002 | 0.120 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.540 ± 0.007 | - | unsupported | 0.302 ± 0.008 | 0.670 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 2.195 ± 0.023 | - | unsupported | 1.688 ± 0.020 | 2.271 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 6.209 ± 0.037 | - | unsupported | 4.431 ± 0.304 | 5.756 ± 0.062 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 1.377 ± 0.010 | - | unsupported | 0.893 ± 0.008 | 1.091 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.067 ± 0.001 | - | unsupported | 0.048 ± 0.013 | 0.062 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.058 ± 0.001 | - | unsupported | 0.037 ± 0.001 | 0.051 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.008 ± 0.000 | - | unsupported | 0.011 ± 0.001 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.008 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | unsupported | 0.012 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.022 ± 0.001 | - | unsupported | 0.023 ± 0.001 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | unsupported | 0.013 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.029 ± 0.000 | - | unsupported | 0.025 ± 0.004 | 0.036 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 10.976 ± 0.033 | - | unsupported | 7.125 ± 0.057 | 8.523 ± 0.062 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.521 ± 0.017 | - | unsupported | 0.313 ± 0.009 | 0.435 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.018 ± 0.000 | - | unsupported | 0.020 ± 0.000 | 0.028 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.049 ± 0.000 | - | unsupported | 0.041 ± 0.000 | 0.063 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.543 ± 0.010 | - | unsupported | 0.443 ± 0.013 | 0.654 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 0.791 ± 0.007 | - | unsupported | 1.670 ± 0.023 | 1.399 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 2.182 ± 0.087 | - | unsupported | 5.875 ± 0.027 | 3.059 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.370 ± 0.103 | - | unsupported | 1.017 ± 0.005 | 1.745 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.009 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.076 ± 0.005 | - | unsupported | 0.045 ± 0.004 | 0.061 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | unsupported | 0.012 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.069 ± 0.004 | - | unsupported | 0.045 ± 0.002 | 0.072 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.010 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.010 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.008 ± 0.000 | - | unsupported | 0.012 ± 0.001 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.024 ± 0.001 | - | unsupported | 0.022 ± 0.002 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.010 ± 0.000 | - | unsupported | 0.013 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.031 ± 0.000 | - | unsupported | 0.027 ± 0.000 | 0.036 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 3.583 ± 0.016 | - | unsupported | 7.955 ± 0.024 | 19.711 ± 6.737 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | unsupported | 0.013 ± 0.005 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.188 ± 0.001 | - | unsupported | 0.312 ± 0.011 | 0.276 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.018 ± 0.000 | - | unsupported | 0.019 ± 0.000 | 0.026 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.058 ± 0.001 | - | unsupported | 0.040 ± 0.000 | 0.063 ± 0.002 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 1.215 ± 0.042 | - | unsupported | 1.469 ± 0.042 | 3.369 ± 0.170 | 2.539 ± 0.234 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.093 ± 0.000 | - | unsupported | 0.047 ± 0.001 | 0.111 ± 0.009 | 0.118 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.153 ± 0.003 | - | unsupported | 0.060 ± 0.002 | 0.168 ± 0.004 | 0.202 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.386 ± 0.018 | - | unsupported | 0.415 ± 0.039 | 0.438 ± 0.014 | 0.726 ± 0.381 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.641 ± 0.016 | - | unsupported | 0.553 ± 0.021 | 1.365 ± 0.152 | 2.858 ± 0.374 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.130 ± 0.015 | - | unsupported | 0.051 ± 0.002 | 0.111 ± 0.016 | 0.090 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.211 ± 0.014 | - | unsupported | 0.070 ± 0.003 | 0.160 ± 0.004 | 0.269 ± 0.162 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.192 ± 0.032 | - | unsupported | 0.179 ± 0.017 | 0.630 ± 0.037 | 0.795 ± 0.289 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.515 ± 0.009 | - | unsupported | 0.543 ± 0.011 | 0.869 ± 0.031 | 0.312 ± 0.020 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.045 ± 0.000 | - | unsupported | 0.032 ± 0.001 | 0.178 ± 0.012 | 0.056 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.058 ± 0.001 | - | unsupported | 0.042 ± 0.001 | 0.216 ± 0.024 | 0.082 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.184 ± 0.006 | - | unsupported | 0.260 ± 0.007 | 0.349 ± 0.024 | 0.143 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.275 ± 0.014 | - | unsupported | 0.617 ± 0.033 | 1.244 ± 0.083 | 0.316 ± 0.030 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.061 ± 0.001 | - | unsupported | 0.051 ± 0.001 | 0.170 ± 0.002 | 0.055 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.076 ± 0.000 | - | unsupported | 0.071 ± 0.001 | 0.211 ± 0.012 | 0.080 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.116 ± 0.006 | - | unsupported | 0.308 ± 0.009 | 0.352 ± 0.013 | 0.140 ± 0.040 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.272 ± 0.011 | - | 0.265 ± 0.017 | 0.270 ± 0.014 | 0.446 ± 0.019 | 0.183 ± 0.005 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.033 ± 0.000 | - | 0.035 ± 0.000 | 0.071 ± 0.004 | 0.075 ± 0.003 | 0.048 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.039 ± 0.000 | - | 0.042 ± 0.002 | 0.081 ± 0.005 | 0.092 ± 0.003 | 0.067 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.106 ± 0.010 | - | 0.104 ± 0.003 | 0.138 ± 0.004 | 0.159 ± 0.026 | 0.103 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.113 ± 0.001 | - | 0.115 ± 0.006 | 0.320 ± 0.015 | 0.630 ± 0.102 | 0.184 ± 0.007 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.046 ± 0.000 | - | 0.049 ± 0.000 | 0.087 ± 0.006 | 0.076 ± 0.002 | 0.045 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.040 ± 0.000 | - | 0.060 ± 0.001 | 0.111 ± 0.003 | 0.093 ± 0.005 | 0.063 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.055 ± 0.007 | - | 0.060 ± 0.008 | 0.172 ± 0.003 | 0.176 ± 0.010 | 0.101 ± 0.002 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 22.653 ± 0.398 | - | 23.564 ± 0.147 | 40.967 ± 0.743 | 59.085 ± 0.681 | 43.053 ± 16.000 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 19.529 ± 0.188 | - | 19.373 ± 0.122 | 19.635 ± 0.595 | 44.696 ± 1.008 | 35.888 ± 3.731 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 17.724 ± 0.165 | - | 14.454 ± 0.338 | 13.562 ± 0.077 | 17.740 ± 0.363 | 16.696 ± 0.553 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 15.014 ± 0.092 | - | 10.016 ± 0.292 | 6.464 ± 0.181 | 13.925 ± 2.721 | 10.779 ± 3.718 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 7.553 ± 0.053 | - | 7.485 ± 0.014 | 5.680 ± 0.058 | 6.613 ± 0.031 | 6.821 ± 0.112 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 10.265 ± 0.055 | - | 10.291 ± 0.052 | 8.055 ± 0.070 | 11.029 ± 1.026 | 10.811 ± 0.062 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 9.451 ± 0.118 | - | 9.472 ± 0.043 | 7.175 ± 0.065 | 8.383 ± 0.053 | 8.075 ± 0.038 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 13.148 ± 0.093 | - | 13.286 ± 0.040 | 9.404 ± 0.207 | 13.448 ± 0.079 | 13.189 ± 0.086 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 16.314 ± 0.079 | - | 16.350 ± 0.074 | 12.373 ± 0.076 | 26.440 ± 0.257 | 11.279 ± 0.731 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 16.813 ± 0.176 | - | 16.761 ± 0.281 | 11.777 ± 0.238 | 25.072 ± 0.658 | 10.979 ± 0.238 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.346 ± 0.014 | 0.371 ± 0.028 | unsupported | 0.276 ± 0.009 | 0.702 ± 0.013 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 5.001 ± 0.017 | 5.176 ± 0.020 | unsupported | 5.246 ± 0.160 | 7.659 ± 0.046 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.422 ± 0.007 | 0.453 ± 0.012 | unsupported | 0.213 ± 0.006 | 1.748 ± 0.103 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 4.646 ± 0.062 | 4.772 ± 0.085 | unsupported | 2.992 ± 0.016 | 10.593 ± 0.271 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 21.199 ± 0.219 | - | 20.773 ± 0.364 | 19.182 ± 0.127 | 37.608 ± 0.326 | 23.009 ± 2.219 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 14.344 ± 0.355 | - | 13.349 ± 0.309 | 10.110 ± 1.394 | 22.452 ± 0.140 | 14.774 ± 0.489 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 11.965 ± 0.169 | 18.516 ± 8.567 | 4.959 ± 0.032 | 42.030 ± 0.353 | 5.156 ± 0.118 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 8.624 ± 0.139 | 8.301 ± 0.056 | 3.252 ± 0.063 | 48.400 ± 2.026 | 4.865 ± 0.104 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 17.823 ± 0.170 | - | 17.530 ± 0.125 | 23.982 ± 0.119 | 28.359 ± 0.124 | 24.427 ± 11.995 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 14.974 ± 0.157 | - | 15.041 ± 0.251 | 12.475 ± 0.242 | 21.974 ± 0.799 | 33.555 ± 6.021 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 1.271 ± 0.106 | - | 1.178 ± 0.023 | 1.176 ± 0.043 | 17.553 ± 0.332 | 2.239 ± 0.227 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.582 ± 0.015 | - | 0.624 ± 0.023 | 1.225 ± 0.057 | 11.696 ± 1.484 | 3.663 ± 0.825 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 12.525 ± 0.246 | - | 11.732 ± 0.177 | 19.291 ± 0.135 | 13.873 ± 0.173 | 15.356 ± 0.295 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 13.508 ± 0.283 | - | 12.800 ± 0.164 | 51.864 ± 1.113 | 19.712 ± 3.061 | 18.986 ± 2.236 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 12.600 ± 0.202 | - | 11.918 ± 0.068 | 19.378 ± 0.101 | 13.771 ± 0.162 | 15.327 ± 0.470 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 13.336 ± 0.052 | - | 12.961 ± 0.459 | 51.998 ± 0.832 | 19.562 ± 0.762 | 19.261 ± 3.554 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.167 ± 0.011 | 1.146 ± 0.012 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 21.788 ± 0.205 | 21.677 ± 0.360 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.342 ± 0.007 | 1.312 ± 0.013 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 14.773 ± 0.808 | 15.124 ± 0.633 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.094 ± 0.009 | 1.065 ± 0.008 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 20.756 ± 0.083 | 20.981 ± 0.136 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.243 ± 0.011 | 1.264 ± 0.006 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 14.298 ± 0.196 | 13.859 ± 0.457 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 17.903 ± 0.367 | - | 14.828 ± 0.243 | 13.686 ± 0.289 | 17.769 ± 0.058 | 16.782 ± 0.390 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 15.069 ± 0.106 | - | 9.991 ± 0.068 | 6.650 ± 0.255 | 13.594 ± 0.310 | 10.685 ± 0.299 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 39.126 ± 0.410 | 38.936 ± 0.345 | 82.686 ± 0.360 | 44.901 ± 0.195 | 50.903 ± 8.688 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 29.308 ± 0.632 | 29.326 ± 1.116 | 190.751 ± 8.726 | 51.291 ± 1.419 | 54.207 ± 1.947 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 22.041 ± 0.324 | - | 22.429 ± 0.265 | 17.765 ± 0.116 | 140.623 ± 9.253 | 21.953 ± 0.128 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 14.259 ± 0.282 | - | 14.779 ± 0.252 | 9.114 ± 0.168 | 136.430 ± 8.565 | 16.812 ± 0.175 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.083 ± 0.001 | 0.078 ± 0.003 | unsupported | 0.086 ± 0.018 | 0.103 ± 0.022 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 1.266 ± 0.004 | 1.297 ± 0.004 | unsupported | 0.606 ± 0.006 | 0.583 ± 0.325 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.046 ± 0.002 | 0.046 ± 0.001 | unsupported | 0.146 ± 0.074 | 0.103 ± 0.016 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.598 ± 0.013 | 0.595 ± 0.003 | unsupported | 0.601 ± 0.012 | 0.266 ± 0.330 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.040 ± 0.003 | 0.039 ± 0.004 | unsupported | 0.088 ± 0.021 | 0.207 ± 0.005 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 0.626 ± 0.022 | 0.653 ± 0.009 | unsupported | 0.693 ± 0.292 | 2.316 ± 0.056 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.026 ± 0.001 | 0.026 ± 0.001 | unsupported | 0.022 ± 0.001 | 0.149 ± 0.014 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.309 ± 0.003 | 0.313 ± 0.002 | unsupported | 0.521 ± 0.016 | 1.029 ± 0.010 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.124 ± 0.001 | 0.130 ± 0.002 | unsupported | 0.179 ± 0.003 | 0.158 ± 0.006 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 2.182 ± 0.031 | 2.220 ± 0.054 | unsupported | 3.064 ± 0.117 | 1.640 ± 0.045 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.060 ± 0.002 | 0.061 ± 0.000 | unsupported | 0.082 ± 0.002 | 0.142 ± 0.026 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 0.896 ± 0.074 | 0.857 ± 0.004 | unsupported | 1.187 ± 0.141 | 0.916 ± 0.011 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.060 ± 0.015 | 0.076 ± 0.011 | unsupported | 0.134 ± 0.001 | 0.169 ± 0.003 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 1.193 ± 0.061 | 1.221 ± 0.078 | unsupported | 0.867 ± 0.011 | 2.224 ± 0.026 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.102 ± 0.008 | 0.117 ± 0.021 | unsupported | 0.062 ± 0.001 | 0.178 ± 0.008 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.628 ± 0.033 | 0.713 ± 0.153 | unsupported | 0.903 ± 0.030 | 1.431 ± 0.010 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.085 ± 0.013 | 0.085 ± 0.005 | unsupported | 0.088 ± 0.002 | 0.261 ± 0.015 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.562 ± 0.100 | 1.592 ± 0.114 | unsupported | 0.915 ± 0.016 | 3.582 ± 0.028 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.127 ± 0.008 | 0.130 ± 0.003 | unsupported | 0.051 ± 0.002 | 0.251 ± 0.012 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.823 ± 0.036 | 0.887 ± 0.032 | unsupported | 0.621 ± 0.004 | 2.248 ± 0.107 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.263 ± 0.003 | 0.274 ± 0.003 | unsupported | unsupported | 0.274 ± 0.010 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 4.477 ± 0.017 | 4.581 ± 0.128 | unsupported | unsupported | 4.695 ± 0.051 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.200 ± 0.004 | 0.231 ± 0.016 | unsupported | unsupported | 0.297 ± 0.016 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 1.818 ± 0.070 | 1.896 ± 0.211 | unsupported | unsupported | 2.642 ± 0.115 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.011 ± 0.000 | 0.015 ± 0.000 | unsupported | 0.011 ± 0.001 | 0.048 ± 0.005 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.159 ± 0.003 | 0.164 ± 0.006 | unsupported | 0.152 ± 0.002 | 0.285 ± 0.004 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.015 ± 0.000 | 0.019 ± 0.002 | unsupported | 0.012 ± 0.000 | 0.039 ± 0.006 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.072 ± 0.002 | 0.092 ± 0.002 | unsupported | 0.057 ± 0.004 | 0.179 ± 0.023 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.036 ± 0.017 | 0.043 ± 0.014 | unsupported | 0.076 ± 0.007 | 0.160 ± 0.011 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.634 ± 0.123 | 0.670 ± 0.105 | unsupported | 0.950 ± 0.028 | 2.119 ± 0.120 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.054 ± 0.003 | 0.065 ± 0.004 | unsupported | 0.085 ± 0.001 | 0.156 ± 0.009 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.339 ± 0.046 | 0.364 ± 0.046 | unsupported | 0.706 ± 0.008 | 1.409 ± 0.119 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.032 ± 0.002 | 0.029 ± 0.001 | unsupported | 0.070 ± 0.002 | 0.161 ± 0.005 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.504 ± 0.053 | 0.507 ± 0.034 | unsupported | 1.367 ± 0.012 | 1.567 ± 0.021 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.020 ± 0.002 | 0.022 ± 0.003 | unsupported | 0.139 ± 0.101 | 0.127 ± 0.004 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.235 ± 0.006 | 0.240 ± 0.004 | unsupported | 0.552 ± 0.012 | 1.149 ± 0.189 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.030 ± 0.002 | 0.030 ± 0.001 | unsupported | 0.072 ± 0.005 | 0.164 ± 0.007 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.500 ± 0.040 | 0.532 ± 0.026 | unsupported | 1.388 ± 0.019 | 1.613 ± 0.130 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.022 ± 0.001 | 0.022 ± 0.000 | unsupported | 0.064 ± 0.002 | 0.138 ± 0.010 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.248 ± 0.021 | 0.250 ± 0.003 | unsupported | 0.567 ± 0.029 | 1.168 ± 0.198 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.076 ± 0.006 | 0.093 ± 0.008 | unsupported | 0.097 ± 0.004 | 0.267 ± 0.013 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.446 ± 0.060 | 1.507 ± 0.053 | unsupported | 0.948 ± 0.063 | 3.400 ± 0.086 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.119 ± 0.004 | 0.128 ± 0.001 | unsupported | 0.049 ± 0.001 | 0.248 ± 0.007 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 0.816 ± 0.027 | 0.904 ± 0.036 | unsupported | 0.414 ± 0.005 | 2.194 ± 0.109 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.096 ± 0.009 | 0.092 ± 0.002 | unsupported | 0.155 ± 0.002 | 0.272 ± 0.008 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 1.497 ± 0.008 | 1.533 ± 0.011 | unsupported | 2.637 ± 0.023 | 3.503 ± 0.022 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.047 ± 0.002 | 0.049 ± 0.009 | unsupported | 0.111 ± 0.001 | 0.202 ± 0.007 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 0.660 ± 0.003 | 0.663 ± 0.181 | unsupported | 1.042 ± 0.026 | 1.852 ± 0.044 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.788 ± 0.006 | 1.292 ± 0.016 | unsupported | 0.332 ± 0.003 | 0.163 ± 0.002 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 16.029 ± 0.305 | 26.638 ± 1.964 | unsupported | 6.760 ± 0.027 | 1.655 ± 0.012 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.545 ± 0.006 | 1.187 ± 0.027 | unsupported | 0.205 ± 0.002 | 0.128 ± 0.016 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 8.098 ± 0.169 | 20.498 ± 0.943 | unsupported | 2.993 ± 0.049 | 0.947 ± 0.025 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 56.455 ± 1.029 | - | unsupported | 54.461 ± 1.158 | - | 53.951 ± 0.845 | 58.313 ± 0.152 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 24.827 ± 0.254 | - | unsupported | 24.994 ± 0.389 | - | 54.112 ± 0.229 | 26.685 ± 0.146 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 44.081 ± 0.378 | - | unsupported | 41.582 ± 0.617 | - | 38.506 ± 0.362 | 44.749 ± 0.417 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 17.837 ± 0.071 | - | unsupported | 18.044 ± 0.106 | - | 38.625 ± 0.342 | 19.257 ± 0.073 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 22.851 ± 0.268 | - | unsupported | 41.328 ± 0.717 | - | 22.559 ± 0.065 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 17.790 ± 0.036 | - | unsupported | 18.265 ± 0.186 | - | 22.281 ± 0.042 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 56.436 ± 0.389 | - | unsupported | 54.781 ± 1.235 | - | 54.712 ± 1.562 | 58.373 ± 0.319 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 24.584 ± 0.131 | - | unsupported | 24.961 ± 0.144 | - | 54.185 ± 0.362 | 27.038 ± 0.217 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 25.100 ± 1.494 | - | unsupported | 24.084 ± 0.033 | - | 25.825 ± 0.059 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 15.003 ± 0.199 | - | unsupported | 12.016 ± 0.016 | - | 13.131 ± 0.082 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 25.064 ± 0.459 | - | unsupported | 24.079 ± 0.103 | - | 25.523 ± 0.294 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 14.992 ± 0.101 | - | unsupported | 12.037 ± 0.062 | - | 12.782 ± 0.102 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 55.134 ± 0.173 | - | unsupported | 54.642 ± 0.674 | - | 54.370 ± 0.279 | 60.742 ± 1.556 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 24.740 ± 0.080 | - | unsupported | 24.829 ± 0.133 | - | 53.822 ± 0.298 | 26.671 ± 0.392 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 40.608 ± 0.273 | - | unsupported | 40.882 ± 0.287 | - | 40.145 ± 0.389 | 42.387 ± 0.316 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 18.114 ± 0.304 | - | unsupported | 18.048 ± 0.128 | - | 40.165 ± 0.370 | 18.409 ± 0.146 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.375 ± 0.004 | unsupported | unsupported | 0.431 ± 0.032 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 6.297 ± 0.060 | unsupported | unsupported | 7.078 ± 0.017 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.490 ± 0.014 | unsupported | unsupported | 0.410 ± 0.012 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 4.980 ± 0.077 | unsupported | unsupported | 3.803 ± 0.129 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 55.404 ± 0.659 | - | unsupported | 54.266 ± 0.228 | - | 54.174 ± 0.289 | 58.626 ± 0.249 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 24.874 ± 0.181 | - | unsupported | 24.909 ± 3.738 | - | 53.980 ± 0.725 | 26.821 ± 1.046 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 100.011 ± 0.578 | - | 0.001 ± 0.000 | 97.635 ± 0.524 | 101.238 ± 1.074 | 116.924 ± 1.452 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 42.705 ± 0.201 | - | 0.002 ± 0.000 | 34.221 ± 6.966 | 54.638 ± 0.558 | 116.851 ± 4.504 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 63.228 ± 0.258 | - | 62.202 ± 0.348 | 66.074 ± 0.539 | 66.416 ± 1.099 | 71.954 ± 4.703 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 24.419 ± 0.902 | - | 24.193 ± 0.694 | 24.116 ± 0.333 | 35.427 ± 0.915 | 69.826 ± 0.547 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 197.975 ± 2.455 | - | 197.790 ± 1.724 | 199.465 ± 7.171 | 202.665 ± 2.185 | 216.913 ± 0.747 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 69.903 ± 2.673 | - | 69.782 ± 0.895 | 68.337 ± 0.706 | 109.799 ± 1.435 | 214.186 ± 0.841 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 66.562 ± 0.532 | - | 65.327 ± 0.157 | 70.472 ± 0.407 | 61.717 ± 0.335 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 43.202 ± 0.294 | - | 42.795 ± 0.473 | 25.752 ± 1.209 | 32.434 ± 0.479 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 141.869 ± 0.699 | - | 0.001 ± 0.000 | 107.672 ± 0.499 | 122.799 ± 0.936 | 157.546 ± 20.943 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 40.431 ± 1.230 | - | 0.002 ± 0.000 | 38.914 ± 0.384 | 47.070 ± 7.197 | 156.024 ± 3.303 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 102.933 ± 4.480 | - | 0.001 ± 0.000 | 76.287 ± 0.198 | 83.301 ± 0.284 | 238.665 ± 14.041 | 90.864 ± 8.683 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 37.662 ± 0.900 | - | 0.002 ± 0.000 | 76.684 ± 0.411 | 44.257 ± 0.342 | 245.278 ± 4.121 | 40.699 ± 1.201 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 53.725 ± 0.145 | - | 54.357 ± 0.322 | 63.111 ± 0.663 | 57.345 ± 1.543 | 79.063 ± 0.821 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 21.511 ± 0.211 | - | 21.355 ± 0.106 | 21.787 ± 0.268 | 31.019 ± 0.933 | 78.897 ± 1.180 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 54.391 ± 0.353 | - | 54.411 ± 0.624 | 60.359 ± 0.640 | 57.292 ± 1.488 | 80.537 ± 8.872 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 21.371 ± 0.265 | - | 21.554 ± 0.039 | 20.544 ± 0.404 | 30.513 ± 0.886 | 81.881 ± 1.236 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.001 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 7.433108 | 464569.25 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 5.626229 | 351639.31 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 5.863281 | 366455.06 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 5.712535 | 357033.44 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 3.941616 | 246351.00 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 4.432146 | 277009.12 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 18.854633 | 1178414.56 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 18.789403 | 1174337.69 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 9.804576 | 612786.00 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 9.794798 | 612174.88 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 11.235849 | 702240.56 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 4.408044 | 275502.75 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 5.528531 | 345533.19 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 5.937859 | 371116.19 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 122.546559 | 7659159.94 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 83.929147 | 5245571.69 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 80.016683 | 5001042.69 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 82.821881 | 5176367.56 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 27.965106 | 1747819.12 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 3.400800 | 212550.00 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 6.749713 | 421857.06 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 7.245793 | 452862.06 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 169.480254 | 10592515.88 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 47.873364 | 2992085.25 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 74.333041 | 4645815.06 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 76.355239 | 4772202.44 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 18.672580 | 1167036.25 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 18.338971 | 1146185.69 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 348.613510 | 21788344.38 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 346.827299 | 21676706.19 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 21.478873 | 1342429.56 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 20.995936 | 1312246.00 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 236.369275 | 14773079.69 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 241.989002 | 15124312.62 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 17.511125 | 1094445.31 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 17.039067 | 1064941.69 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 332.098656 | 20756166.00 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 335.696053 | 20981003.31 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 19.880183 | 1242511.44 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 20.224732 | 1264045.75 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 228.762003 | 14297625.19 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 221.747472 | 13859217.00 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.603811 | 103184.55 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.501287 | 85957.61 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.287480 | 82616.88 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.007898 | 78248.41 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 4.664301 | 583037.62 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.845224 | 605653.00 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.125357 | 1265669.62 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.377728 | 1297216.00 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.560811 | 102512.67 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 9.364703 | 146323.48 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.934887 | 45857.61 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.963416 | 46303.38 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.131876 | 266484.50 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.810251 | 601281.38 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.787067 | 598383.38 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.763164 | 595395.50 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.258972 | 207171.44 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.652389 | 88318.58 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.561621 | 40025.33 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.513453 | 39272.70 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.529821 | 2316227.62 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 5.545712 | 693214.00 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.011245 | 626405.62 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.227612 | 653451.50 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.514277 | 148660.58 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 1.377878 | 21529.34 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.678626 | 26228.53 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.665367 | 26021.36 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.230156 | 1028769.50 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.167145 | 520893.12 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.469216 | 308652.00 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.505700 | 313212.50 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.083278 | 157551.22 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 11.465835 | 179153.67 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 7.946829 | 124169.20 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.347512 | 130429.88 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.123071 | 1640383.88 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 24.515406 | 3064425.75 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 17.458010 | 2182251.25 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 17.757053 | 2219631.62 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.115812 | 142434.56 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.257707 | 82151.67 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.810087 | 59532.61 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3.933943 | 61467.86 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.329911 | 916238.88 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 9.498265 | 1187283.12 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.168559 | 896069.88 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 6.859824 | 857478.00 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.806318 | 168848.72 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.603198 | 134424.97 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.858114 | 60283.03 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.844973 | 75702.70 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 17.791962 | 2223995.25 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 6.938927 | 867365.88 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.547348 | 1193418.50 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 9.766003 | 1220750.38 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.410013 | 178281.45 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.949105 | 61704.77 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.550923 | 102358.17 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.471677 | 116744.95 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.449566 | 1431195.75 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.223649 | 902956.12 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.027762 | 628470.25 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.700919 | 712614.88 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 16.717421 | 261209.70 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.625025 | 87891.02 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.451784 | 85184.12 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.439788 | 84996.69 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 28.654586 | 3581823.25 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.321927 | 915240.88 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 12.498220 | 1562277.50 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 12.738324 | 1592290.50 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 16.074160 | 251158.75 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.276753 | 51199.27 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.128170 | 127002.66 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 8.300780 | 129699.69 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 17.983972 | 2247996.50 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.971283 | 621410.38 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.585793 | 823224.12 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.099212 | 887401.50 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 17.563150 | 274424.22 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 16.863004 | 263484.44 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 17.514002 | 273656.28 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 37.559862 | 4694982.75 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 35.818785 | 4477348.12 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 36.645446 | 4580680.75 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 19.021965 | 297218.20 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 12.779708 | 199682.94 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 14.808832 | 231388.00 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 21.139298 | 2642412.25 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 14.544097 | 1818012.12 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 15.169559 | 1896194.88 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 3.074161 | 48033.77 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.732526 | 11445.72 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.722944 | 11296.00 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 0.934943 | 14608.48 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 2.280887 | 285110.88 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 1.215664 | 151958.00 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 1.274293 | 159286.62 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 1.310425 | 163803.12 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.490898 | 38920.28 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.797489 | 12460.77 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.983035 | 15359.92 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.223306 | 19114.16 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.431022 | 178877.75 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.458002 | 57250.25 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.578853 | 72356.62 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 0.736529 | 92066.12 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.213100 | 159579.69 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.835137 | 75549.02 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.297169 | 35893.27 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.721422 | 42522.22 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 16.948037 | 2118504.62 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.603406 | 950425.75 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.069593 | 633699.12 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.361515 | 670189.38 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.000466 | 156257.28 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.410388 | 84537.31 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.476284 | 54316.94 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.148220 | 64815.94 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.270413 | 1408801.62 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.645113 | 705639.12 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.713959 | 339244.88 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.912633 | 364079.12 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.298783 | 160918.48 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.451120 | 69548.75 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.043440 | 31928.75 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.876634 | 29322.41 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.532190 | 1566523.75 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.933633 | 1366704.12 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.028516 | 503564.50 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.058997 | 507374.62 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.151917 | 127373.70 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 8.903798 | 139121.84 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.306712 | 20417.38 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.390036 | 21719.31 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.190197 | 1148774.62 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.413314 | 551664.25 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.879726 | 234965.75 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.917485 | 239685.62 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.477853 | 163716.45 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.593676 | 71776.19 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.930924 | 30170.69 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.946936 | 30420.88 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.906635 | 1613329.38 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.107078 | 1388384.75 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.998123 | 499765.38 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.256537 | 532067.12 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.818218 | 137784.66 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.081132 | 63767.69 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.389763 | 21715.05 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.378426 | 21537.91 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.343477 | 1167934.62 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.536183 | 567022.88 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.980748 | 247593.50 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.000486 | 250060.75 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 17.076889 | 266826.39 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.197277 | 96832.45 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.893110 | 76454.84 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.927656 | 92619.62 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 27.196087 | 3399510.88 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.584675 | 948084.38 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 11.566780 | 1445847.50 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 12.052836 | 1506604.50 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 15.851897 | 247685.89 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.139232 | 49050.50 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 7.595151 | 118674.23 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 8.170984 | 127671.63 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 17.548793 | 2193599.12 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.309320 | 413665.00 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.528569 | 816071.12 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.233110 | 904138.75 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 17.430759 | 272355.61 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 9.926772 | 155105.81 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.165014 | 96328.34 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.919222 | 92487.84 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 28.020006 | 3502500.75 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 21.092612 | 2636576.50 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 11.979900 | 1497487.50 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 12.267721 | 1533465.12 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.915070 | 201797.97 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 7.111144 | 111111.62 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.979695 | 46557.73 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3.112282 | 48629.41 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 14.815471 | 1851933.88 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.337889 | 1042236.13 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.279668 | 659958.50 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.301038 | 662629.75 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.406933 | 162608.33 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 21.270974 | 332358.97 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 50.445841 | 788216.27 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 82.692550 | 1292071.09 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.240805 | 1655100.62 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 54.080636 | 6760079.50 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 128.232151 | 16029018.87 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 213.106727 | 26638340.88 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.194659 | 128041.55 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 13.118247 | 204972.61 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 34.896897 | 545264.02 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 75.983800 | 1187246.88 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.576001 | 947000.12 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 23.943504 | 2992938.00 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 64.786804 | 8098350.50 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 163.986742 | 20498342.75 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 6.901262 | 431328.88 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 6.001504 | 375094.00 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 113.255520 | 7078470.00 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 100.759071 | 6297441.94 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 6.556676 | 409792.25 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 7.832589 | 489536.81 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 60.844657 | 3802791.06 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 79.675228 | 4979701.75 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.018801 | 1175.06 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002072 | 129.50 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.018258 | 1141.12 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.001900 | 118.75 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000427 | 26.69 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.018612 | 1163.25 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.007261 | 453.81 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000196 | 12.25 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.017369 | 1085.56 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.006841 | 427.56 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000566 | 35.38 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.018687 | 1167.94 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.005749 | 359.31 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000407 | 25.44 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.017842 | 1115.12 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.005234 | 327.12 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000259 | 16.19 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.017161 | 1072.56 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003909 | 244.31 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000128 | 8.00 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.016132 | 1008.25 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003798 | 237.38 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/elementwise_reduction/exp` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 11.2x faster than the slowest successful cell (`julia-base`, 52.236 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/exp` (f64, threads=4, shape=`8388608`): `tenferro-trace` is 11.7x faster than the slowest successful cell (`julia-base`, 52.236 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log1p` (f64, threads=4, shape=`4194304`): `tenferro-direct` is 11.0x faster than the slowest successful cell (`julia-base`, 37.574 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log1p` (f64, threads=4, shape=`4194304`): `tenferro-trace` is 11.0x faster than the slowest successful cell (`julia-base`, 37.574 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 18.437 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 12.296 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 12.296 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 13.4x faster than the slowest successful cell (`pytorch-cpu`, 12.296 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 18.380 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 13.2x faster than the slowest successful cell (`pytorch-cpu`, 11.986 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 13.1x faster than the slowest successful cell (`pytorch-cpu`, 11.986 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sin` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 11.3x faster than the slowest successful cell (`julia-base`, 61.760 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sin` (f64, threads=4, shape=`8388608`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`julia-base`, 61.760 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `pytorch-cpu` is 13.6x faster than the slowest successful cell (`julia-base`, 8.828 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `tenferro-direct` is 13.9x faster than the slowest successful cell (`julia-base`, 8.828 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `tenferro-trace` is 13.7x faster than the slowest successful cell (`julia-base`, 8.828 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `jax-cpu` is 11.9x faster than the slowest successful cell (`julia-base`, 8.227 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 12.8x faster than the slowest successful cell (`julia-base`, 8.227 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 13.0x faster than the slowest successful cell (`julia-base`, 8.227 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-trace` is 12.8x faster than the slowest successful cell (`julia-base`, 8.227 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 5938.8x faster than the slowest successful cell (`julia-base`, 8.902 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `pytorch-cpu` is 10.5x faster than the slowest successful cell (`julia-base`, 10.032 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`julia-base`, 10.032 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 4988.6x faster than the slowest successful cell (`julia-base`, 10.032 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 14.0x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 26.5x faster than the slowest successful cell (`jax-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 15.0x faster than the slowest successful cell (`jax-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 18.3x faster than the slowest successful cell (`jax-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-trace` is 11.6x faster than the slowest successful cell (`jax-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x128x128`): `tenferro-direct` is 15.9x faster than the slowest successful cell (`jax-cpu`, 2.922 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x128x128`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`jax-cpu`, 2.922 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 14.9x faster than the slowest successful cell (`jax-cpu`, 48.400 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `pytorch-cpu` is 14.9x faster than the slowest successful cell (`jax-cpu`, 17.553 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 13.8x faster than the slowest successful cell (`jax-cpu`, 17.553 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`jax-cpu`, 17.553 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 20.1x faster than the slowest successful cell (`jax-cpu`, 11.696 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 18.7x faster than the slowest successful cell (`jax-cpu`, 11.696 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/triangular_solve` (f64, threads=4, shape=`4096x4096,rhs=64`): `pytorch-cpu` is 15.0x faster than the slowest successful cell (`jax-cpu`, 136.430 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 16.1x faster than the slowest successful cell (`tenferro-eager`, 26.638 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 21.6x faster than the slowest successful cell (`tenferro-eager`, 20.498 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 80692.8x faster than the slowest successful cell (`julia-base`, 116.924 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 65097.9x faster than the slowest successful cell (`julia-base`, 116.851 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 106955.7x faster than the slowest successful cell (`julia-base`, 157.546 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 84749.6x faster than the slowest successful cell (`julia-base`, 156.024 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 175876.9x faster than the slowest successful cell (`julia-base`, 238.665 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 141045.4x faster than the slowest successful cell (`julia-base`, 245.278 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 43.1x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 90.5x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 33.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 44.6x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 67.1x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 126.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 11186.8 GB/s at 0.001 ms. The conservative physical lower bound is 0.168 ms, so the cell is 111.9x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 8338.6 GB/s at 0.002 ms. The conservative physical lower bound is 0.042 ms, so the cell is 20.8x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 185255.7 GB/s at 0.001 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1852.6x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 149546.2 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 373.9x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 182237.2 GB/s at 0.001 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1822.4x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 145809.6 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 364.5x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 98907.7 GB/s at 0.001 ms. The conservative physical lower bound is 1.342 ms, so the cell is 989.1x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 77181.0 GB/s at 0.002 ms. The conservative physical lower bound is 0.336 ms, so the cell is 193.0x faster than the bound.

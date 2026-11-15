# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/amd-cpu/cpu/public_api/20261115_191824/run.yaml`
- Timestamp: `2026-11-15T19:18:24.150519+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/amd-cpu/cpu/public_api/20261115_191824`.

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

- CSV: `data/results/amd-cpu/cpu/public_api/20261115_191824/cpu_public_api_t1_20261115_191824.csv`
- CSV: `data/results/amd-cpu/cpu/public_api/20261115_191824/cpu_public_api_t4_20261115_191824.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 4.324 ± 0.070 | - | 4.224 ± 0.139 | 4.509 ± 0.012 | 3.845 ± 0.010 | 3.996 ± 2.332 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 4.539 ± 0.029 | - | 4.434 ± 0.893 | 2.281 ± 0.023 | 3.047 ± 0.983 | 3.684 ± 2.188 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 126.025 ± 4.019 | - | 126.146 ± 3.897 | 122.378 ± 0.943 | 125.494 ± 0.748 | 143.698 ± 2.033 | 147.340 ± 21.290 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 41.501 ± 0.284 | - | 142.677 ± 2.836 | 42.180 ± 1.192 | 65.533 ± 1.128 | 143.341 ± 2.868 | 63.226 ± 14.928 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 107.574 ± 1.290 | - | 112.838 ± 2.498 | 94.730 ± 1.042 | 92.586 ± 2.272 | 133.626 ± 4.780 | 136.507 ± 2.894 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 43.936 ± 0.421 | - | 130.624 ± 2.206 | 31.586 ± 1.319 | 47.770 ± 1.126 | 129.177 ± 14.777 | 60.920 ± 2.264 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 45.094 ± 0.482 | - | 44.708 ± 0.368 | 43.091 ± 0.246 | 68.243 ± 0.395 | 46.586 ± 0.256 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 45.339 ± 0.744 | - | 45.678 ± 0.697 | 29.051 ± 0.113 | 29.408 ± 0.697 | 23.732 ± 0.139 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 46.442 ± 0.271 | - | unsupported | 43.934 ± 0.594 | 68.531 ± 0.152 | 46.907 ± 4.826 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 46.690 ± 1.164 | - | unsupported | 29.465 ± 1.738 | 30.699 ± 1.234 | 27.071 ± 3.438 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 10.997 ± 0.097 | - | 11.052 ± 0.060 | 11.584 ± 0.122 | 11.913 ± 0.027 | 11.998 ± 0.180 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 14.169 ± 0.030 | - | 14.209 ± 0.056 | 14.065 ± 0.031 | 17.493 ± 0.585 | 17.647 ± 0.047 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 71.437 ± 0.275 | - | 72.722 ± 0.189 | 109.366 ± 4.131 | 103.993 ± 2.155 | 87.740 ± 12.945 | 82.766 ± 1.258 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 30.878 ± 0.703 | - | 83.999 ± 0.006 | 43.081 ± 0.646 | 52.253 ± 0.616 | 87.563 ± 5.386 | 36.974 ± 0.661 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 116.097 ± 1.695 | - | 115.679 ± 0.300 | 154.435 ± 0.388 | 176.644 ± 1.243 | 148.049 ± 0.369 | 160.694 ± 25.969 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 49.721 ± 0.202 | - | 131.996 ± 0.014 | 63.186 ± 0.566 | 72.370 ± 0.588 | 169.554 ± 1.149 | 63.759 ± 32.864 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 70.589 ± 0.383 | - | 70.826 ± 2.212 | 69.689 ± 1.284 | 71.576 ± 0.975 | 78.278 ± 1.045 | 78.084 ± 4.811 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 24.785 ± 0.630 | - | 94.999 ± 3.410 | 23.447 ± 0.123 | 36.831 ± 2.462 | 76.540 ± 0.762 | 33.862 ± 1.150 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 2.670 ± 0.124 | - | 20.981 ± 0.023 | 24.198 ± 0.066 | 16.191 ± 0.403 | 4.292 ± 0.020 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 0.990 ± 0.144 | - | 59.998 ± 0.013 | 11.293 ± 2.620 | 2.577 ± 0.501 | 4.096 ± 0.028 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 8.770 ± 0.043 | - | 8.961 ± 0.036 | 9.744 ± 0.090 | 7.579 ± 0.024 | 9.150 ± 0.229 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 130.983 ± 26.158 | - | 115.820 ± 11.942 | 5.772 ± 0.026 | 6.842 ± 0.075 | 7.254 ± 0.685 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 5.563 ± 0.243 | - | 5.769 ± 0.336 | 5.568 ± 0.034 | 5.123 ± 0.041 | 4.989 ± 0.068 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 100.930 ± 15.086 | - | 136.009 ± 13.007 | 3.623 ± 0.030 | 4.682 ± 1.127 | 4.318 ± 0.037 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 6.543 ± 0.110 | - | 6.572 ± 0.188 | 7.717 ± 0.041 | 5.475 ± 0.053 | 5.471 ± 0.270 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 11.605 ± 0.166 | - | 11.635 ± 0.195 | 5.876 ± 0.028 | 6.113 ± 0.023 | 5.818 ± 0.056 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 44.729 ± 0.625 | - | 58.740 ± 16.291 | 43.813 ± 0.722 | 68.099 ± 0.077 | 44.097 ± 0.192 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 45.641 ± 0.947 | - | 45.448 ± 10.798 | 29.084 ± 0.124 | 29.386 ± 0.428 | 22.289 ± 0.110 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 50.298 ± 0.504 | - | unsupported | 51.118 ± 0.267 | 61.188 ± 5.117 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 53.050 ± 0.336 | - | unsupported | 25.365 ± 0.156 | 34.768 ± 0.449 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 124.774 ± 5.020 | - | 124.772 ± 5.228 | 123.420 ± 2.358 | 126.175 ± 0.775 | 138.587 ± 30.898 | 156.973 ± 2.760 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 40.643 ± 0.191 | - | 143.419 ± 0.935 | 41.163 ± 0.526 | 63.997 ± 3.096 | 140.365 ± 8.925 | 66.500 ± 12.043 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 134.870 ± 4.397 | - | 133.978 ± 4.161 | 136.078 ± 2.765 | 139.064 ± 0.959 | 147.704 ± 10.072 | 163.723 ± 4.578 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 45.889 ± 1.347 | - | 150.165 ± 9.922 | 45.614 ± 0.890 | 71.053 ± 1.760 | 152.387 ± 2.782 | 69.636 ± 0.959 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 27.794 ± 0.163 | - | 59.443 ± 0.171 | 76.341 ± 0.742 | 50.991 ± 0.220 | 80.727 ± 8.891 | 86.447 ± 13.921 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 12.657 ± 0.266 | - | 119.996 ± 0.018 | 33.501 ± 16.214 | 25.923 ± 0.447 | 81.190 ± 0.355 | 33.526 ± 13.578 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 11.111 ± 0.341 | - | 10.633 ± 1.217 | 36.051 ± 1.313 | 38.423 ± 0.789 | 34.879 ± 9.534 | 48.793 ± 9.536 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 7.334 ± 0.122 | - | 23.997 ± 0.008 | 12.699 ± 0.370 | 23.090 ± 0.852 | 44.853 ± 9.233 | 14.647 ± 15.112 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 23.693 ± 1.157 | - | 21.839 ± 1.155 | 55.316 ± 0.179 | 45.104 ± 1.283 | 21.253 ± 0.294 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 14.979 ± 0.247 | - | 36.003 ± 12.002 | 23.284 ± 8.988 | 24.763 ± 0.249 | 25.056 ± 0.336 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 28.317 ± 0.133 | - | 28.563 ± 0.036 | 55.030 ± 1.079 | 108.020 ± 0.558 | 66.482 ± 7.836 | 71.541 ± 9.877 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 11.955 ± 0.021 | - | 47.998 ± 0.020 | 21.909 ± 0.392 | 54.289 ± 0.281 | 66.287 ± 9.378 | 29.674 ± 14.708 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 135.652 ± 2.870 | - | 136.543 ± 3.897 | 135.785 ± 2.603 | 138.676 ± 0.812 | 148.338 ± 1.737 | 161.986 ± 0.721 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 45.349 ± 0.392 | - | 153.408 ± 3.162 | 45.103 ± 0.192 | 76.209 ± 0.698 | 153.960 ± 6.805 | 94.846 ± 3.663 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 20.506 ± 0.200 | - | 20.530 ± 0.050 | 43.677 ± 0.131 | 41.381 ± 0.524 | 62.270 ± 13.358 | 68.541 ± 12.000 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 8.425 ± 0.098 | - | 36.000 ± 0.004 | 16.633 ± 0.190 | 25.487 ± 0.688 | 63.321 ± 17.215 | 30.189 ± 16.404 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 20.643 ± 0.040 | - | 20.421 ± 0.028 | 32.274 ± 0.443 | 28.051 ± 0.158 | 37.586 ± 8.875 | 40.216 ± 10.822 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 8.101 ± 0.062 | - | 35.996 ± 0.013 | 11.313 ± 0.569 | 13.948 ± 0.145 | 37.708 ± 0.720 | 16.415 ± 11.541 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 25.275 ± 0.049 | - | 25.650 ± 0.057 | 49.807 ± 0.481 | 65.327 ± 0.669 | 83.001 ± 12.803 | 89.165 ± 12.956 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 10.594 ± 0.017 | - | 47.997 ± 0.007 | 19.093 ± 0.167 | 50.096 ± 12.715 | 85.940 ± 14.761 | 36.660 ± 16.991 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 13.252 ± 0.068 | - | 13.239 ± 0.052 | 32.813 ± 0.448 | 37.201 ± 0.263 | 46.005 ± 8.936 | 49.788 ± 8.550 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 5.652 ± 0.027 | - | 35.999 ± 0.005 | 11.796 ± 0.236 | 18.807 ± 0.673 | 46.052 ± 0.118 | 27.974 ± 18.299 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.468 ± 0.011 | 0.501 ± 0.009 | unsupported | unsupported | 0.492 ± 0.004 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.410 ± 0.005 | 51.874 ± 0.563 | unsupported | unsupported | 0.223 ± 0.022 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 137.448 ± 2.941 | - | 136.050 ± 3.818 | 135.247 ± 1.255 | 138.868 ± 0.619 | 150.650 ± 3.649 | 163.469 ± 3.079 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 45.153 ± 11.351 | - | 150.994 ± 12.138 | 45.687 ± 0.891 | 72.259 ± 1.448 | 155.865 ± 2.992 | 71.828 ± 0.775 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 137.383 ± 2.099 | - | 136.884 ± 2.993 | 134.908 ± 1.422 | 138.404 ± 1.222 | 151.551 ± 2.682 | 162.450 ± 1.976 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 45.168 ± 1.218 | - | 152.995 ± 5.358 | 45.580 ± 0.170 | 71.355 ± 4.092 | 157.641 ± 8.344 | 71.039 ± 3.541 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 131.796 ± 4.479 | - | 134.207 ± 4.480 | 132.167 ± 3.066 | 139.054 ± 0.995 | 149.246 ± 4.387 | 162.347 ± 1.826 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 45.273 ± 0.405 | - | 150.421 ± 14.210 | 46.270 ± 1.832 | 71.048 ± 0.499 | 149.508 ± 4.198 | 70.925 ± 0.487 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 124.548 ± 7.805 | - | 125.989 ± 4.004 | 125.060 ± 2.188 | 125.683 ± 1.286 | 139.868 ± 24.249 | 152.701 ± 8.339 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 41.783 ± 0.497 | - | 142.541 ± 0.810 | 40.951 ± 0.334 | 64.981 ± 0.773 | 139.529 ± 25.102 | 65.990 ± 4.078 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 58.622 ± 0.120 | - | 58.529 ± 0.119 | 63.203 ± 0.784 | 74.251 ± 0.642 | 126.035 ± 0.472 | 122.855 ± 15.580 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 23.583 ± 0.953 | - | 72.004 ± 0.014 | 22.082 ± 0.178 | 36.872 ± 0.480 | 125.871 ± 21.187 | 49.734 ± 17.084 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 18.557 ± 0.730 | - | 18.074 ± 1.007 | 12.957 ± 0.373 | 26.869 ± 0.839 | 14.751 ± 0.262 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 7.157 ± 0.179 | - | 35.998 ± 0.009 | 6.727 ± 0.080 | 14.427 ± 0.240 | 13.767 ± 0.800 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.998 ± 0.086 | - | 2.000 ± 0.031 | 24.434 ± 0.144 | 2.438 ± 0.029 | 1.454 ± 0.166 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.787 ± 0.032 | - | 24.000 ± 0.007 | 11.310 ± 0.929 | 1.580 ± 0.084 | 1.223 ± 0.219 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 2.677 ± 0.047 | - | 2.674 ± 0.040 | 4.125 ± 0.100 | 2.669 ± 0.307 | 1.648 ± 0.056 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 1.290 ± 0.053 | - | 23.997 ± 0.010 | 2.312 ± 0.027 | 1.388 ± 0.088 | 1.380 ± 0.215 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 17.816 ± 0.289 | - | 17.340 ± 0.737 | 12.938 ± 0.428 | 27.308 ± 0.572 | 14.926 ± 0.495 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 6.989 ± 0.311 | - | 36.001 ± 0.016 | 6.744 ± 0.261 | 14.367 ± 0.493 | 13.719 ± 0.336 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 2.280 ± 0.100 | - | 2.145 ± 0.059 | 23.401 ± 0.124 | 2.494 ± 0.027 | 1.549 ± 0.085 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.796 ± 0.036 | - | 23.999 ± 0.003 | 10.640 ± 0.135 | 1.597 ± 0.109 | 1.170 ± 0.402 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 10.649 ± 0.149 | - | 10.710 ± 0.134 | 16.318 ± 0.101 | 12.462 ± 0.647 | 7.362 ± 0.243 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 5.327 ± 0.228 | - | 35.997 ± 0.010 | 8.808 ± 0.070 | 7.044 ± 0.658 | 7.149 ± 0.376 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 12.811 ± 0.299 | - | 12.677 ± 0.300 | 11.451 ± 0.148 | 20.703 ± 0.474 | 11.690 ± 0.167 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 6.674 ± 0.036 | - | 23.998 ± 0.007 | 6.629 ± 0.202 | 11.402 ± 0.064 | 11.865 ± 0.229 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 1.289 ± 0.081 | - | 1.360 ± 0.108 | 2.250 ± 0.144 | 2.073 ± 0.029 | 1.043 ± 0.102 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.628 ± 0.057 | - | 23.998 ± 0.005 | 0.959 ± 0.027 | 1.794 ± 0.049 | 1.037 ± 0.140 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.623 ± 0.026 | - | 1.600 ± 0.235 | 0.948 ± 0.016 | 3.405 ± 0.424 | 1.185 ± 0.085 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.682 ± 0.185 | - | 23.998 ± 0.007 | 0.356 ± 0.063 | 1.950 ± 0.308 | 1.160 ± 0.146 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 12.485 ± 0.346 | - | 12.273 ± 0.307 | 12.522 ± 0.994 | 20.958 ± 0.360 | 11.583 ± 0.391 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 6.690 ± 0.143 | - | 23.998 ± 0.008 | 6.991 ± 0.269 | 11.600 ± 0.324 | 11.555 ± 0.250 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 1.264 ± 0.174 | - | 1.360 ± 0.070 | 2.239 ± 0.190 | 2.055 ± 0.011 | 1.102 ± 0.127 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.571 ± 0.053 | - | 24.000 ± 0.004 | 0.921 ± 0.041 | 1.790 ± 0.706 | 0.944 ± 0.179 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.576 ± 0.075 | - | 1.650 ± 0.085 | 0.965 ± 0.071 | 3.187 ± 0.301 | 1.234 ± 0.068 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.664 ± 0.095 | - | 24.002 ± 0.014 | 0.357 ± 0.075 | 1.578 ± 0.056 | 1.060 ± 0.115 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 1.403 ± 0.063 | 1.458 ± 0.068 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.442 ± 0.037 | 21.536 ± 11.599 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 37.596 ± 0.125 | - | 36.867 ± 0.186 | 55.730 ± 0.722 | 70.716 ± 0.513 | 69.902 ± 13.263 | 78.022 ± 13.663 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 18.120 ± 0.269 | - | 59.994 ± 0.015 | 21.119 ± 0.861 | 37.810 ± 0.899 | 76.540 ± 11.786 | 43.765 ± 12.021 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 141.735 ± 2.378 | - | 144.041 ± 5.343 | 145.655 ± 1.237 | 248.350 ± 1.661 | 264.294 ± 2.487 | 264.579 ± 34.316 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 46.606 ± 0.595 | - | 155.077 ± 11.802 | 47.546 ± 0.975 | 124.979 ± 0.995 | 266.232 ± 6.059 | 108.565 ± 19.804 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 139.982 ± 2.688 | - | 138.450 ± 3.804 | 146.454 ± 4.580 | 157.463 ± 1.371 | 150.181 ± 4.978 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 47.796 ± 0.681 | - | 153.602 ± 12.545 | 53.294 ± 0.987 | 80.806 ± 1.033 | 155.570 ± 20.302 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 128.715 ± 5.083 | - | 129.598 ± 3.784 | 128.008 ± 1.794 | 133.615 ± 1.093 | 140.422 ± 3.688 | 148.288 ± 5.577 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 41.428 ± 0.480 | - | 143.452 ± 12.366 | 41.874 ± 0.817 | 65.105 ± 1.039 | 143.266 ± 31.676 | 63.676 ± 1.563 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 24.495 ± 0.056 | - | 25.024 ± 0.160 | 57.410 ± 1.049 | 107.311 ± 0.622 | 66.293 ± 13.629 | 70.957 ± 8.980 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 10.257 ± 0.032 | - | 47.996 ± 0.010 | 22.074 ± 0.264 | 53.937 ± 0.150 | 66.672 ± 9.303 | 42.026 ± 15.467 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 132.581 ± 1.874 | - | 133.202 ± 2.047 | 186.954 ± 0.771 | 136.335 ± 3.472 | 216.472 ± 6.493 | 218.945 ± 4.621 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 43.672 ± 0.409 | - | 155.357 ± 2.387 | 68.697 ± 1.801 | 69.645 ± 2.859 | 217.995 ± 3.431 | 92.813 ± 2.683 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 134.165 ± 3.028 | - | 134.574 ± 5.031 | 134.690 ± 3.777 | 138.524 ± 1.407 | 147.182 ± 4.481 | 164.579 ± 4.926 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 45.813 ± 0.810 | - | 150.337 ± 10.612 | 45.784 ± 1.747 | 71.412 ± 0.565 | 152.587 ± 1.723 | 68.766 ± 1.180 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 51.360 ± 0.033 | - | 51.779 ± 0.403 | 75.842 ± 1.092 | 38.901 ± 0.482 | 78.830 ± 13.635 | 89.706 ± 11.960 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 20.506 ± 4.902 | - | 71.995 ± 11.994 | 28.544 ± 0.163 | 20.096 ± 0.426 | 79.743 ± 14.547 | 34.827 ± 16.021 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 140.467 ± 4.453 | 133.095 ± 4.203 | unsupported | 148.371 ± 4.250 | 155.675 ± 1.200 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 47.318 ± 0.741 | 145.054 ± 9.743 | unsupported | 53.327 ± 0.761 | 80.428 ± 1.128 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.977 ± 0.095 | - | 1.134 ± 0.092 | 1.078 ± 0.076 | 0.990 ± 0.377 | 5.434 ± 6.111 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.417 ± 0.053 | - | 12.002 ± 0.033 | 0.347 ± 0.039 | 0.900 ± 0.111 | 7.556 ± 3.736 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 1.277 ± 0.181 | - | 1.318 ± 0.247 | 0.980 ± 0.140 | 1.066 ± 0.319 | 7.771 ± 3.685 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.496 ± 0.338 | - | 12.001 ± 0.008 | 0.349 ± 0.066 | 0.612 ± 0.278 | 7.607 ± 3.930 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.432 ± 0.195 | - | unsupported | 1.387 ± 0.053 | 1.392 ± 0.140 | 8.426 ± 10.694 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.638 ± 0.136 | - | unsupported | 0.716 ± 0.050 | 1.088 ± 0.113 | 7.040 ± 3.878 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.418 ± 0.053 | - | 0.521 ± 0.031 | 0.707 ± 0.036 | 0.429 ± 0.023 | 1.344 ± 0.088 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.488 ± 0.163 | - | 47.996 ± 0.021 | 0.350 ± 0.013 | 0.449 ± 0.052 | 1.198 ± 0.066 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 1.146 ± 0.299 | - | 1.074 ± 0.166 | 1.552 ± 0.276 | 1.063 ± 0.112 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.547 ± 0.291 | - | 12.000 ± 0.006 | 0.541 ± 0.069 | 1.046 ± 0.012 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 1.170 ± 0.029 | - | 1.052 ± 0.086 | 0.998 ± 0.190 | 0.979 ± 0.073 | 8.531 ± 11.208 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.423 ± 0.094 | - | 11.999 ± 0.005 | 0.346 ± 0.098 | 0.666 ± 0.100 | 8.625 ± 8.803 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.809 ± 0.008 | - | 0.942 ± 0.028 | 0.433 ± 0.029 | 0.496 ± 0.039 | 1.416 ± 0.047 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 1.226 ± 0.025 | - | 36.016 ± 0.149 | 0.679 ± 11.332 | 0.463 ± 0.022 | 1.504 ± 0.131 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 2.029 ± 0.042 | - | 0.002 ± 0.000 | 1.995 ± 0.092 | 1.502 ± 0.041 | 9.578 ± 2.739 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.834 ± 0.042 | - | 0.002 ± 0.000 | 0.909 ± 0.176 | 1.114 ± 0.174 | 9.110 ± 4.587 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.051 ± 0.001 | - | 0.052 ± 0.000 | 0.081 ± 0.002 | 0.114 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.093 ± 0.000 | - | 0.095 ± 0.002 | 0.130 ± 0.003 | 0.150 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.242 ± 0.001 | - | 0.242 ± 0.001 | 0.283 ± 0.009 | 0.320 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.082 ± 0.001 | - | 0.084 ± 0.002 | 0.148 ± 0.003 | 0.118 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.040 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.002 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.042 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.677 ± 0.009 | - | 0.691 ± 0.029 | 1.153 ± 0.024 | 0.679 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.026 ± 0.000 | - | 0.028 ± 0.000 | 0.039 ± 0.002 | 0.065 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.001 | 0.034 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.064 ± 0.000 | - | 0.073 ± 0.000 | 0.085 ± 0.001 | 0.100 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.094 ± 0.001 | - | 0.097 ± 0.001 | 0.124 ± 0.018 | 0.151 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.245 ± 0.025 | - | 0.247 ± 0.028 | 0.340 ± 0.011 | 0.361 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.085 ± 0.001 | - | 0.118 ± 0.033 | 0.178 ± 0.004 | 0.114 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.010 ± 0.002 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.016 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.011 ± 0.001 | 0.018 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.676 ± 0.009 | - | 0.699 ± 0.030 | 1.389 ± 0.041 | 1.045 ± 0.809 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.014 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.031 ± 0.000 | - | 0.033 ± 0.000 | 0.059 ± 0.004 | 0.049 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.658 ± 0.524 | - | 0.883 ± 0.467 | 0.723 ± 0.009 | 0.888 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 2.136 ± 0.009 | - | 2.145 ± 0.012 | 2.179 ± 0.048 | 2.713 ± 0.053 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 6.476 ± 0.056 | - | 6.490 ± 0.031 | 6.597 ± 0.063 | 8.935 ± 0.058 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 3.705 ± 0.020 | - | 3.686 ± 0.004 | 3.240 ± 0.020 | 2.644 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.109 ± 0.001 | - | 0.114 ± 0.004 | 0.123 ± 0.003 | 0.163 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.008 ± 0.005 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.028 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.017 ± 0.001 | 0.035 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.016 ± 0.003 | 0.032 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.020 ± 0.000 | - | 0.023 ± 0.000 | 0.030 ± 0.002 | 0.119 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.010 ± 0.001 | - | 0.012 ± 0.000 | 0.018 ± 0.001 | 0.043 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.027 ± 0.000 | - | 0.029 ± 0.002 | 0.037 ± 0.014 | 0.087 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 29.446 ± 0.549 | - | 29.503 ± 0.368 | 25.636 ± 0.226 | 20.420 ± 0.624 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.015 ± 0.001 | 0.028 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.865 ± 0.016 | - | 0.868 ± 0.006 | 0.874 ± 0.006 | 1.007 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.018 ± 0.000 | - | 0.020 ± 0.000 | 0.026 ± 0.001 | 0.044 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.052 ± 0.000 | - | 0.054 ± 0.000 | 0.060 ± 0.001 | 0.112 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.652 ± 0.036 | - | 0.668 ± 4.675 | 0.724 ± 0.425 | 0.920 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 2.132 ± 0.025 | - | 2.180 ± 1.420 | 2.132 ± 0.013 | 2.152 ± 0.104 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 6.535 ± 0.031 | - | 6.530 ± 0.060 | 9.320 ± 0.017 | 5.767 ± 0.206 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 6.776 ± 0.037 | - | 6.802 ± 0.035 | 2.238 ± 0.023 | 3.471 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.013 ± 0.002 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.111 ± 0.006 | - | 0.127 ± 0.011 | 0.177 ± 0.006 | 0.195 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.001 | 0.026 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.013 ± 0.000 | - | 0.011 ± 0.000 | 0.016 ± 0.001 | 0.039 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.016 ± 0.001 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.001 | 0.016 ± 0.001 | 0.039 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.022 ± 0.000 | - | 0.024 ± 0.000 | 0.029 ± 0.001 | 0.099 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.018 ± 0.001 | 0.040 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.027 ± 0.000 | - | 0.031 ± 0.001 | 0.035 ± 0.000 | 0.076 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 54.123 ± 0.470 | - | 54.155 ± 0.239 | 17.782 ± 0.769 | 29.986 ± 6.443 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.021 ± 0.001 | 0.028 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.872 ± 0.030 | - | 0.887 ± 0.034 | 1.247 ± 0.011 | 1.996 ± 4.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.018 ± 0.000 | - | 0.021 ± 0.000 | 0.026 ± 0.001 | 0.057 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.055 ± 0.001 | - | 0.055 ± 0.000 | 0.060 ± 0.001 | 0.114 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.677 ± 0.003 | - | 0.684 ± 0.010 | 0.522 ± 0.008 | 0.472 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 2.216 ± 0.012 | - | 2.218 ± 0.011 | 1.951 ± 0.010 | 1.981 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 5.821 ± 0.013 | - | 5.852 ± 0.053 | 5.726 ± 0.011 | 6.477 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.304 ± 0.015 | - | 1.332 ± 0.029 | 1.332 ± 0.017 | 1.517 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.070 ± 0.002 | - | 0.074 ± 0.001 | 0.090 ± 0.004 | 0.166 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.024 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.015 ± 0.001 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.014 ± 0.001 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.018 ± 0.000 | - | 0.021 ± 0.001 | 0.026 ± 0.001 | 0.041 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.028 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.017 ± 0.001 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.024 ± 0.000 | - | 0.027 ± 0.000 | 0.031 ± 0.001 | 0.046 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 10.429 ± 0.045 | - | 10.492 ± 0.110 | 10.598 ± 0.154 | 11.322 ± 2.233 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.551 ± 0.011 | - | 0.560 ± 0.014 | 0.630 ± 0.012 | 0.783 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.023 ± 0.000 | 0.036 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.046 ± 0.000 | - | 0.049 ± 0.003 | 0.053 ± 0.001 | 0.117 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.643 ± 0.024 | - | 0.659 ± 4.761 | 0.795 ± 0.096 | 0.441 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 2.219 ± 0.036 | - | 2.220 ± 0.023 | 1.933 ± 0.016 | 1.297 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 5.854 ± 0.009 | - | 5.860 ± 0.020 | 8.477 ± 0.013 | 3.686 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 3.210 ± 0.033 | - | 3.210 ± 0.049 | 1.179 ± 0.032 | 1.788 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.012 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.161 ± 0.041 | - | 0.163 ± 0.022 | 0.134 ± 0.005 | 0.129 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.001 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.015 ± 0.001 | 0.033 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.014 ± 0.001 | 0.025 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.014 ± 0.001 | 0.032 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.019 ± 0.000 | - | 0.022 ± 0.000 | 0.026 ± 0.001 | 0.043 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.015 ± 0.001 | 0.022 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.017 ± 0.000 | 0.033 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.025 ± 0.001 | - | 0.027 ± 0.000 | 0.032 ± 0.001 | 0.048 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 25.428 ± 0.211 | - | 25.304 ± 0.104 | 9.122 ± 0.100 | 14.946 ± 1.349 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.001 | 0.018 ± 0.002 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 1.105 ± 0.030 | - | 1.100 ± 0.035 | 0.923 ± 0.002 | 0.482 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.019 ± 0.000 | - | 0.022 ± 0.000 | 0.023 ± 0.001 | 0.036 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.047 ± 0.001 | - | 0.050 ± 0.000 | 0.053 ± 0.001 | 0.096 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.263 ± 0.012 | - | 0.268 ± 0.013 | 0.236 ± 0.014 | 0.697 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 1.110 ± 0.007 | - | 1.115 ± 0.009 | 0.952 ± 0.005 | 2.015 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 3.112 ± 0.009 | - | 3.127 ± 0.016 | 2.889 ± 0.015 | 6.623 ± 0.085 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.517 ± 0.009 | - | 0.519 ± 0.011 | 0.551 ± 0.013 | 1.494 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.010 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.031 ± 0.001 | - | 0.033 ± 0.000 | 0.041 ± 0.005 | 0.162 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.011 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.011 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.001 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.021 ± 0.002 | 0.037 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.011 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.001 | 0.014 ± 0.001 | 0.041 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.022 ± 0.001 | 0.039 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 4.123 ± 0.009 | - | 4.088 ± 0.086 | 4.273 ± 0.009 | 11.226 ± 0.129 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.012 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.231 ± 0.008 | - | 0.234 ± 0.004 | 0.243 ± 0.002 | 0.746 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.017 ± 0.001 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.025 ± 0.000 | - | 0.027 ± 0.000 | 0.033 ± 0.001 | 0.096 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.264 ± 0.010 | - | 0.267 ± 0.013 | 0.397 ± 0.013 | 0.442 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 1.115 ± 0.021 | - | 1.138 ± 0.032 | 0.945 ± 0.004 | 1.276 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 3.144 ± 0.042 | - | 3.147 ± 0.022 | 3.661 ± 0.009 | 3.663 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 2.035 ± 0.017 | - | 2.035 ± 0.006 | 0.703 ± 0.005 | 1.760 ± 0.120 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.014 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.031 ± 0.001 | - | 0.035 ± 0.000 | 0.056 ± 0.002 | 0.144 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.017 ± 0.004 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.001 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.015 ± 0.002 | 0.017 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.019 ± 0.001 | 0.024 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.019 ± 0.001 | 0.038 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.017 ± 0.005 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.019 ± 0.001 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.014 ± 0.000 | - | 0.017 ± 0.000 | 0.022 ± 0.001 | 0.041 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 16.154 ± 0.099 | - | 16.021 ± 0.163 | 5.325 ± 0.027 | 13.708 ± 1.622 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.017 ± 0.001 | 0.017 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.234 ± 0.025 | - | 0.240 ± 0.023 | 0.317 ± 0.016 | 0.433 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.024 ± 0.003 | 0.030 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.026 ± 0.000 | - | 0.028 ± 0.000 | 0.033 ± 0.001 | 0.086 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.151 ± 0.020 | - | 0.155 ± 0.001 | 0.189 ± 0.013 | 0.121 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.212 ± 0.011 | - | 0.218 ± 0.004 | 0.253 ± 0.013 | 0.252 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.508 ± 0.009 | - | 0.511 ± 0.009 | 0.537 ± 0.012 | 0.616 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.140 ± 0.007 | - | 0.139 ± 0.008 | 0.173 ± 0.016 | 0.207 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.014 ± 0.005 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.012 ± 0.001 | - | 0.015 ± 0.005 | 0.025 ± 0.002 | 0.075 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.013 ± 0.001 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.014 ± 0.001 | - | 0.016 ± 0.000 | 0.028 ± 0.001 | 0.053 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.005 | 0.013 ± 0.000 | 0.026 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.015 ± 0.000 | 0.026 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.001 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.017 ± 0.000 | 0.058 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.001 | 0.022 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.016 ± 0.001 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.018 ± 0.000 | 0.040 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 1.099 ± 0.036 | - | 1.073 ± 0.022 | 1.280 ± 0.009 | 1.282 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.000 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.078 ± 0.001 | - | 0.080 ± 0.001 | 0.087 ± 0.003 | 0.122 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.017 ± 0.001 | 0.034 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.001 | 0.020 ± 0.000 | 0.046 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.149 ± 0.009 | - | 0.173 ± 0.003 | 0.180 ± 0.011 | 0.146 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.215 ± 0.012 | - | 0.222 ± 0.009 | 0.233 ± 0.003 | 0.227 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.512 ± 0.027 | - | 0.523 ± 0.022 | 0.386 ± 0.017 | 0.638 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.142 ± 0.008 | - | 0.184 ± 0.051 | 0.226 ± 0.018 | 0.208 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.024 ± 0.005 | 0.025 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.013 ± 0.000 | - | 0.016 ± 0.000 | 0.039 ± 0.001 | 0.051 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.023 ± 0.004 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.015 ± 0.001 | - | 0.019 ± 0.001 | 0.042 ± 0.001 | 0.084 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.024 ± 0.002 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.025 ± 0.001 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.026 ± 0.001 | 0.024 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.028 ± 0.001 | 0.028 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.026 ± 0.000 | 0.027 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.026 ± 0.001 | 0.025 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.028 ± 0.000 | 0.034 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 1.072 ± 0.075 | - | 1.082 ± 0.045 | 0.691 ± 0.022 | 0.941 ± 0.148 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.027 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.087 ± 0.003 | - | 0.090 ± 0.001 | 0.078 ± 0.001 | 0.163 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.028 ± 0.006 | 0.033 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.031 ± 0.000 | 0.048 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.257 ± 0.011 | - | 0.258 ± 0.010 | 0.232 ± 0.011 | 0.324 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.605 ± 0.010 | - | 0.611 ± 0.012 | 0.528 ± 0.009 | 0.897 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.378 ± 0.015 | - | 1.372 ± 0.010 | 1.364 ± 0.020 | 2.354 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.483 ± 0.014 | - | 0.490 ± 0.007 | 0.566 ± 0.018 | 0.662 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.018 ± 0.001 | - | 0.020 ± 0.001 | 0.027 ± 0.001 | 0.092 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.025 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.029 ± 0.004 | 0.086 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.028 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.029 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.037 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 3.850 ± 0.029 | - | 4.654 ± 0.032 | 4.425 ± 0.095 | 4.815 ± 0.079 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.002 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.129 ± 0.001 | - | 0.132 ± 0.011 | 0.175 ± 0.003 | 0.241 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.016 ± 0.000 | 0.038 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.252 ± 0.002 | - | 0.252 ± 0.005 | 0.328 ± 0.015 | 0.340 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.605 ± 0.025 | - | 0.618 ± 4.668 | 0.764 ± 0.006 | 0.903 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.363 ± 0.020 | - | 1.351 ± 0.031 | 2.033 ± 0.020 | 1.966 ± 0.274 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 1.045 ± 0.024 | - | 1.057 ± 0.094 | 0.505 ± 0.020 | 2.471 ± 1.501 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.001 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.018 ± 0.000 | - | 0.025 ± 0.000 | 0.045 ± 0.002 | 0.078 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.002 | 0.024 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.020 ± 0.000 | - | 0.028 ± 0.004 | 0.048 ± 0.001 | 0.060 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.001 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.001 | 0.022 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.001 | 0.024 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.046 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.003 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.030 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.000 | 0.046 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 8.205 ± 0.062 | - | 9.125 ± 0.032 | 3.766 ± 0.051 | 7.357 ± 1.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.136 ± 0.001 | - | 0.139 ± 0.031 | 0.288 ± 0.011 | 0.280 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.016 ± 0.000 | 0.030 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.026 ± 0.000 | 0.061 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.163 ± 0.002 | - | 0.146 ± 0.002 | 0.101 ± 0.007 | 0.256 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.205 ± 0.011 | - | 0.196 ± 0.046 | 0.142 ± 0.003 | 0.390 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.535 ± 0.008 | - | 0.567 ± 0.006 | 0.411 ± 0.013 | 0.815 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.119 ± 0.004 | - | 0.129 ± 0.055 | 0.147 ± 0.005 | 0.208 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.021 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.011 ± 0.000 | - | 0.017 ± 0.003 | 0.029 ± 0.001 | 0.075 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.002 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.019 ± 0.000 | 0.025 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.004 | 0.017 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.019 ± 0.001 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.921 ± 0.018 | - | 1.004 ± 0.016 | 1.017 ± 0.026 | 1.006 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.070 ± 0.001 | - | 0.084 ± 0.019 | 0.081 ± 0.002 | 0.148 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.001 | 0.018 ± 0.000 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.011 ± 0.000 | 0.021 ± 0.001 | 0.027 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.162 ± 0.004 | - | 0.157 ± 0.004 | 0.137 ± 0.022 | 0.264 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.207 ± 0.001 | - | 0.204 ± 0.042 | 0.174 ± 0.001 | 0.372 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.542 ± 0.038 | - | 5.988 ± 0.048 | 0.382 ± 0.015 | 0.860 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.124 ± 0.008 | - | 0.176 ± 0.037 | 0.207 ± 0.009 | 0.198 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.026 ± 0.004 | 0.016 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.011 ± 0.000 | - | 0.020 ± 0.002 | 0.043 ± 0.004 | 0.036 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.025 ± 0.003 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.027 ± 0.002 | 0.017 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.026 ± 0.003 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.027 ± 0.002 | 0.018 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.010 ± 0.001 | 0.029 ± 0.002 | 0.021 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.026 ± 0.003 | 0.022 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.026 ± 0.002 | 0.023 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.011 ± 0.000 | 0.029 ± 0.004 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.957 ± 0.016 | - | 23.984 ± 0.180 | 0.583 ± 0.015 | 0.765 ± 0.072 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.010 ± 0.000 | 0.028 ± 0.001 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.075 ± 0.002 | - | 0.092 ± 0.026 | 0.073 ± 0.001 | 0.149 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.029 ± 0.005 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.007 ± 0.000 | - | 0.013 ± 0.001 | 0.033 ± 0.000 | 0.026 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.820 ± 0.012 | - | 1.811 ± 0.006 | 1.550 ± 0.027 | 2.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 6.101 ± 0.120 | - | 6.021 ± 0.025 | 5.100 ± 0.026 | 5.105 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 18.042 ± 0.071 | - | 17.978 ± 0.124 | 15.890 ± 0.028 | 14.762 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 5.011 ± 0.015 | - | 5.031 ± 0.060 | 5.572 ± 0.025 | 4.251 ± 0.244 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.004 ± 0.001 | - | 0.007 ± 0.000 | 0.015 ± 0.001 | 0.028 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.207 ± 0.011 | - | 0.206 ± 0.015 | 0.233 ± 0.011 | 0.272 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.020 ± 0.001 | 0.041 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.182 ± 0.009 | - | 0.187 ± 0.012 | 0.204 ± 0.009 | 0.222 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.021 ± 0.001 | - | 0.023 ± 0.001 | 0.032 ± 0.002 | 0.044 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.019 ± 0.001 | 0.040 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.020 ± 0.000 | - | 0.023 ± 0.000 | 0.030 ± 0.004 | 0.054 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.055 ± 0.002 | - | 0.057 ± 0.001 | 0.063 ± 0.001 | 0.074 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.020 ± 0.002 | 0.041 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.026 ± 0.000 | - | 0.029 ± 0.004 | 0.035 ± 0.001 | 0.057 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.098 ± 0.027 | - | 0.076 ± 0.000 | 0.081 ± 0.002 | 0.089 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 39.373 ± 0.191 | - | 41.153 ± 1.794 | 44.287 ± 0.243 | 33.132 ± 0.166 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.017 ± 0.000 | - | 0.019 ± 0.003 | 0.026 ± 0.000 | 0.052 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.546 ± 0.016 | - | 1.567 ± 0.032 | 1.635 ± 0.014 | 1.590 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.049 ± 0.000 | - | 0.052 ± 0.001 | 0.056 ± 0.001 | 0.071 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.142 ± 0.002 | - | 0.144 ± 0.008 | 0.141 ± 0.002 | 0.194 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.817 ± 0.014 | - | 1.840 ± 0.020 | 1.515 ± 0.018 | 2.099 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 6.057 ± 0.036 | - | 6.055 ± 0.049 | 5.072 ± 0.029 | 2.729 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 17.887 ± 0.129 | - | 17.831 ± 0.047 | 18.684 ± 0.145 | 8.605 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 9.112 ± 0.065 | - | 9.155 ± 0.089 | 4.668 ± 0.024 | 4.997 ± 0.114 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.005 ± 0.001 | - | 0.007 ± 0.000 | 0.020 ± 0.002 | 0.029 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.305 ± 0.014 | - | 0.316 ± 0.026 | 0.329 ± 0.016 | 0.234 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.019 ± 0.001 | 0.038 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.339 ± 0.021 | - | 0.362 ± 0.046 | 0.286 ± 0.016 | 0.286 ± 0.053 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.022 ± 0.000 | - | 0.025 ± 0.001 | 0.033 ± 0.006 | 0.046 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.008 ± 0.000 | - | 0.011 ± 0.001 | 0.025 ± 0.003 | 0.034 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.021 ± 0.000 | - | 0.024 ± 0.001 | 0.030 ± 0.001 | 0.041 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.056 ± 0.002 | - | 0.061 ± 0.001 | 0.063 ± 0.001 | 0.073 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.027 ± 0.004 | 0.041 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.027 ± 0.000 | - | 0.030 ± 0.000 | 0.035 ± 0.001 | 0.059 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.074 ± 0.000 | - | 0.081 ± 0.004 | 0.079 ± 0.001 | 0.091 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 73.682 ± 30.721 | - | 106.461 ± 0.914 | 36.497 ± 0.160 | 52.988 ± 17.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.017 ± 0.000 | - | 0.019 ± 0.000 | 0.038 ± 0.004 | 0.051 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 2.308 ± 0.033 | - | 2.349 ± 0.074 | 2.353 ± 0.041 | 0.767 ± 0.134 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.052 ± 0.023 | - | 0.054 ± 0.001 | 0.057 ± 0.006 | 0.068 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.142 ± 0.001 | - | 0.151 ± 0.012 | 0.141 ± 0.003 | 0.180 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 1.351 ± 0.045 | - | 1.354 ± 0.014 | 1.104 ± 0.010 | 1.361 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 5.174 ± 0.032 | - | 5.159 ± 0.026 | 4.355 ± 0.060 | 4.089 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 15.226 ± 0.101 | - | 15.244 ± 0.049 | 13.841 ± 0.050 | 12.261 ± 0.064 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 2.278 ± 0.019 | - | 2.327 ± 0.045 | 2.459 ± 0.014 | 2.440 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.001 | 0.025 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.138 ± 0.006 | - | 0.140 ± 0.011 | 0.167 ± 0.015 | 0.218 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.001 | 0.036 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.109 ± 0.005 | - | 0.113 ± 0.008 | 0.131 ± 0.009 | 0.195 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.017 ± 0.000 | - | 0.020 ± 0.001 | 0.029 ± 0.001 | 0.039 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.017 ± 0.002 | 0.037 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.018 ± 0.000 | - | 0.020 ± 0.000 | 0.028 ± 0.000 | 0.047 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.043 ± 0.000 | - | 0.046 ± 0.001 | 0.054 ± 0.003 | 0.062 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.001 | 0.036 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.023 ± 0.001 | - | 0.026 ± 0.000 | 0.031 ± 0.001 | 0.045 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.057 ± 0.000 | - | 0.060 ± 0.000 | 0.066 ± 0.002 | 0.073 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 18.431 ± 0.143 | - | 18.448 ± 0.103 | 19.214 ± 0.103 | 18.587 ± 0.082 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.022 ± 0.001 | 0.042 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 1.055 ± 0.002 | - | 1.065 ± 0.004 | 1.152 ± 0.008 | 1.179 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.043 ± 0.002 | - | 0.045 ± 0.000 | 0.048 ± 0.001 | 0.059 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.115 ± 0.001 | - | 0.118 ± 0.004 | 0.118 ± 0.007 | 0.135 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 1.371 ± 0.028 | - | 1.379 ± 0.024 | 1.071 ± 0.115 | 1.338 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 5.267 ± 0.078 | - | 5.250 ± 0.025 | 4.320 ± 0.021 | 2.110 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 15.143 ± 0.058 | - | 15.208 ± 0.100 | 16.851 ± 3.249 | 6.236 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 3.436 ± 0.035 | - | 3.442 ± 0.062 | 2.768 ± 0.013 | 3.072 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.002 | 0.031 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.248 ± 0.054 | - | 0.258 ± 0.023 | 0.254 ± 0.020 | 0.208 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.018 ± 0.002 | 0.038 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.154 ± 0.009 | - | 0.183 ± 0.031 | 0.197 ± 0.011 | 0.185 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.018 ± 0.000 | - | 0.022 ± 0.000 | 0.028 ± 0.006 | 0.038 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.022 ± 0.004 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.019 ± 0.000 | - | 0.022 ± 0.000 | 0.027 ± 0.001 | 0.040 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.049 ± 0.005 | - | 0.052 ± 0.000 | 0.052 ± 0.001 | 0.079 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.024 ± 0.004 | 0.028 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.024 ± 0.000 | - | 0.027 ± 0.000 | 0.031 ± 0.000 | 0.052 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.061 ± 0.000 | - | 0.064 ± 0.000 | 0.066 ± 0.001 | 0.099 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 27.358 ± 0.173 | - | 27.762 ± 0.165 | 21.755 ± 0.239 | 30.925 ± 9.996 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.013 ± 0.000 | - | 0.016 ± 0.000 | 0.030 ± 0.003 | 0.035 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 1.794 ± 0.018 | - | 1.805 ± 0.015 | 1.750 ± 0.013 | 0.662 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.044 ± 0.001 | - | 0.047 ± 0.001 | 0.048 ± 0.005 | 0.070 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.117 ± 0.002 | - | 0.125 ± 0.007 | 0.121 ± 0.003 | 0.137 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.465 ± 0.009 | - | unsupported | 0.360 ± 0.011 | 0.749 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 2.276 ± 0.010 | - | unsupported | 1.935 ± 0.007 | 2.740 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 5.537 ± 0.043 | - | unsupported | 5.313 ± 0.007 | 6.809 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.985 ± 0.023 | - | unsupported | 1.152 ± 0.008 | 1.102 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.009 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.048 ± 0.002 | - | unsupported | 0.062 ± 0.002 | 0.075 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | unsupported | 0.012 ± 0.001 | 0.025 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.039 ± 0.002 | - | unsupported | 0.053 ± 0.001 | 0.086 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | unsupported | 0.015 ± 0.001 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.011 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | unsupported | 0.016 ± 0.001 | 0.029 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.017 ± 0.001 | - | unsupported | 0.026 ± 0.001 | 0.037 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.011 ± 0.001 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | unsupported | 0.018 ± 0.001 | 0.027 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.023 ± 0.000 | - | unsupported | 0.032 ± 0.001 | 0.043 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 7.612 ± 0.065 | - | unsupported | 8.896 ± 0.037 | 8.498 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | unsupported | 0.012 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.375 ± 0.010 | - | unsupported | 0.422 ± 0.009 | 0.477 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.019 ± 0.000 | - | unsupported | 0.025 ± 0.001 | 0.047 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.044 ± 0.000 | - | unsupported | 0.052 ± 0.005 | 0.069 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.761 ± 0.008 | - | unsupported | 0.587 ± 0.023 | 0.773 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 3.558 ± 0.047 | - | unsupported | 1.934 ± 0.005 | 1.632 ± 0.255 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 5.586 ± 1.461 | - | unsupported | 7.689 ± 0.022 | 3.528 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.211 ± 0.023 | - | unsupported | 1.448 ± 0.017 | 1.434 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.005 ± 0.000 | - | unsupported | 0.014 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.048 ± 0.001 | - | unsupported | 0.091 ± 0.015 | 0.082 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.007 ± 0.000 | - | unsupported | 0.017 ± 0.003 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.040 ± 0.001 | - | unsupported | 0.079 ± 0.005 | 0.088 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.008 ± 0.001 | - | unsupported | 0.016 ± 0.001 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.006 ± 0.000 | - | unsupported | 0.016 ± 0.002 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.014 ± 0.000 | - | unsupported | 0.023 ± 0.001 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.028 ± 0.001 | - | unsupported | 0.026 ± 0.001 | 0.037 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.007 ± 0.000 | - | unsupported | 0.016 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.017 ± 0.001 | - | unsupported | 0.026 ± 0.005 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.037 ± 0.000 | - | unsupported | 0.032 ± 0.003 | 0.055 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 9.326 ± 0.121 | - | unsupported | 11.271 ± 0.050 | 11.059 ± 3.479 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.010 ± 0.000 | - | unsupported | 0.018 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.413 ± 0.044 | - | unsupported | 0.423 ± 0.015 | 0.254 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.031 ± 0.000 | - | unsupported | 0.025 ± 0.004 | 0.044 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.073 ± 0.008 | - | unsupported | 0.052 ± 0.001 | 0.071 ± 0.008 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 1.057 ± 0.022 | - | unsupported | 0.951 ± 0.014 | 3.488 ± 0.066 | 2.554 ± 0.044 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.133 ± 0.017 | - | unsupported | 0.059 ± 0.001 | 0.128 ± 0.003 | 0.125 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.140 ± 0.002 | - | unsupported | 0.084 ± 0.001 | 0.180 ± 0.009 | 0.214 ± 0.008 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.351 ± 0.021 | - | unsupported | 0.244 ± 0.011 | 0.498 ± 0.014 | 0.681 ± 0.188 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 1.054 ± 0.007 | - | unsupported | 0.489 ± 0.015 | 1.110 ± 0.081 | 2.460 ± 0.817 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.135 ± 0.003 | - | unsupported | 0.063 ± 0.001 | 0.108 ± 0.016 | 0.096 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.128 ± 0.003 | - | unsupported | 0.081 ± 0.000 | 0.184 ± 0.019 | 0.181 ± 0.011 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.348 ± 0.015 | - | unsupported | 0.161 ± 0.007 | 0.529 ± 0.032 | 0.673 ± 0.015 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.315 ± 0.015 | - | unsupported | 0.375 ± 0.015 | 0.802 ± 0.015 | 0.405 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.034 ± 0.000 | - | unsupported | 0.046 ± 0.001 | 0.185 ± 0.015 | 0.067 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.059 ± 0.003 | - | unsupported | 0.062 ± 0.001 | 0.222 ± 0.007 | 0.105 ± 0.005 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.201 ± 0.003 | - | unsupported | 0.168 ± 0.010 | 0.373 ± 0.013 | 0.200 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.314 ± 0.014 | - | unsupported | 0.557 ± 0.010 | 0.813 ± 0.024 | 0.417 ± 0.014 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.035 ± 0.000 | - | unsupported | 0.076 ± 0.003 | 0.204 ± 0.028 | 0.067 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.060 ± 0.000 | - | unsupported | 0.101 ± 0.002 | 0.242 ± 0.022 | 0.108 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.202 ± 0.009 | - | unsupported | 0.231 ± 0.002 | 0.416 ± 0.026 | 0.203 ± 0.011 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.191 ± 0.001 | - | 0.196 ± 0.009 | 0.194 ± 0.003 | 0.391 ± 0.014 | 0.258 ± 0.020 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.080 ± 0.000 | - | 0.083 ± 0.000 | 0.073 ± 0.001 | 0.092 ± 0.020 | 0.051 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.090 ± 0.005 | - | 0.094 ± 0.008 | 0.087 ± 0.006 | 0.102 ± 0.005 | 0.078 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.121 ± 0.001 | - | 0.124 ± 0.010 | 0.123 ± 0.006 | 0.174 ± 0.015 | 0.131 ± 0.007 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.199 ± 0.026 | - | 0.200 ± 0.010 | 0.261 ± 0.014 | 0.425 ± 0.099 | 0.259 ± 0.024 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.080 ± 0.005 | - | 0.104 ± 0.003 | 0.126 ± 0.005 | 0.082 ± 0.017 | 0.054 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.091 ± 0.001 | - | 0.110 ± 0.002 | 0.143 ± 0.015 | 0.105 ± 0.012 | 0.080 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.122 ± 0.008 | - | 0.126 ± 0.002 | 0.180 ± 0.011 | 0.215 ± 0.029 | 0.135 ± 0.005 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 46.362 ± 0.423 | - | 50.637 ± 0.197 | 60.033 ± 0.255 | 69.490 ± 0.269 | 52.119 ± 5.049 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 43.536 ± 0.233 | - | 51.318 ± 0.280 | 24.987 ± 0.588 | 70.806 ± 20.121 | 41.603 ± 4.097 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 26.583 ± 0.449 | - | 26.886 ± 0.927 | 24.690 ± 0.266 | 22.248 ± 0.112 | 22.074 ± 5.078 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 242.667 ± 44.305 | - | 327.006 ± 24.029 | 11.114 ± 0.127 | 14.743 ± 0.843 | 12.515 ± 4.086 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 7.504 ± 0.079 | - | 7.497 ± 0.053 | 7.920 ± 0.018 | 7.956 ± 0.023 | 8.126 ± 0.241 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 16.398 ± 0.081 | - | 16.460 ± 0.040 | 10.494 ± 0.038 | 11.676 ± 0.341 | 11.775 ± 0.268 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 8.772 ± 0.061 | - | 8.856 ± 0.123 | 8.799 ± 0.025 | 9.502 ± 0.025 | 9.153 ± 0.079 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 12.923 ± 0.072 | - | 12.880 ± 0.073 | 12.009 ± 0.099 | 14.169 ± 0.041 | 13.730 ± 0.048 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 21.282 ± 0.318 | - | 21.751 ± 0.112 | 19.277 ± 0.062 | 37.861 ± 0.466 | 14.198 ± 0.212 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 38.892 ± 0.051 | - | 39.438 ± 0.155 | 15.181 ± 5.870 | 32.989 ± 1.448 | 12.528 ± 0.782 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.498 ± 0.004 | 0.523 ± 0.006 | unsupported | 0.618 ± 0.007 | 0.749 ± 0.040 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 7.953 ± 0.020 | 9.087 ± 0.354 | unsupported | 8.874 ± 0.077 | 9.642 ± 0.097 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.782 ± 0.007 | 0.814 ± 0.015 | unsupported | 0.510 ± 0.016 | 1.260 ± 0.036 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 2275.688 ± 4.478 | 2310.707 ± 2.713 | unsupported | 5.640 ± 0.195 | 9.648 ± 1.075 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 45.565 ± 3.167 | - | 44.396 ± 0.964 | 42.572 ± 0.545 | 32.146 ± 0.482 | 27.843 ± 1.442 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 231.170 ± 9.103 | - | 280.682 ± 20.099 | 21.605 ± 0.229 | 19.909 ± 0.225 | 16.610 ± 0.968 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 18.090 ± 0.254 | 17.511 ± 0.103 | 9.105 ± 0.096 | 56.129 ± 0.233 | 9.802 ± 0.125 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 2288.912 ± 15.117 | 2284.107 ± 9.440 | 6.152 ± 0.071 | 53.515 ± 1.825 | 8.831 ± 4.157 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 26.939 ± 0.213 | - | 27.048 ± 0.512 | 33.485 ± 0.564 | 32.718 ± 0.255 | 29.173 ± 16.820 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 241.018 ± 11.823 | - | 250.029 ± 20.879 | 16.099 ± 0.553 | 24.982 ± 14.171 | 24.364 ± 11.185 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 1.765 ± 0.045 | - | 1.615 ± 0.084 | 1.412 ± 0.095 | 20.364 ± 0.780 | 2.737 ± 0.030 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.475 ± 0.050 | - | 24.002 ± 0.016 | 1.465 ± 0.079 | 2.991 ± 0.154 | 2.804 ± 0.045 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 20.715 ± 0.220 | - | 20.737 ± 0.261 | 20.111 ± 0.029 | 20.015 ± 2.392 | 19.676 ± 0.478 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 230.952 ± 24.740 | - | 223.952 ± 10.338 | 16.522 ± 0.593 | 20.194 ± 2.338 | 19.705 ± 6.444 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 20.652 ± 0.102 | - | 20.282 ± 0.166 | 20.222 ± 0.301 | 20.204 ± 0.108 | 19.786 ± 0.392 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 207.772 ± 66.069 | - | 184.144 ± 48.810 | 16.512 ± 1.152 | 20.350 ± 1.166 | 19.654 ± 4.531 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.184 ± 0.015 | 1.186 ± 0.006 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 21.894 ± 0.194 | 22.027 ± 0.183 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.572 ± 0.019 | 1.596 ± 0.016 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 22.653 ± 0.100 | 22.725 ± 0.060 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.102 ± 0.012 | 1.110 ± 0.007 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 20.829 ± 0.048 | 20.787 ± 0.277 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.501 ± 0.015 | 1.513 ± 0.008 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 21.613 ± 0.064 | 21.729 ± 0.182 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 26.683 ± 0.590 | - | 26.866 ± 0.302 | 24.821 ± 0.288 | 22.176 ± 0.105 | 21.723 ± 1.361 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 252.801 ± 41.853 | - | 327.021 ± 39.911 | 11.286 ± 0.124 | 14.047 ± 0.993 | 12.549 ± 0.489 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 59.621 ± 0.499 | 59.757 ± 0.387 | 61.084 ± 0.452 | 62.653 ± 0.466 | 64.183 ± 6.889 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 2331.991 ± 26.306 | 2336.912 ± 15.256 | 37.828 ± 1.236 | 52.498 ± 0.664 | 50.601 ± 2.774 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 34.456 ± 0.300 | - | 33.995 ± 0.350 | 30.549 ± 0.526 | 127.955 ± 2.125 | 29.870 ± 0.579 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 51.469 ± 0.441 | - | 51.295 ± 2.171 | 25.507 ± 5.636 | 113.596 ± 11.877 | 19.978 ± 5.395 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.140 ± 0.001 | 0.139 ± 0.001 | unsupported | 0.162 ± 0.031 | 0.038 ± 0.004 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.232 ± 0.005 | 2.221 ± 0.060 | unsupported | 1.657 ± 0.008 | 1.026 ± 0.428 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.073 ± 0.001 | 5.953 ± 0.110 | unsupported | 0.076 ± 0.003 | 0.124 ± 0.023 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.898 ± 0.005 | 23.999 ± 0.000 | unsupported | 1.028 ± 0.041 | 0.354 ± 0.234 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.078 ± 0.002 | 0.079 ± 0.000 | unsupported | 0.134 ± 0.009 | 0.252 ± 0.002 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.249 ± 0.008 | 1.244 ± 0.016 | unsupported | 1.803 ± 0.023 | 2.602 ± 0.010 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.037 ± 0.000 | 12.000 ± 0.547 | unsupported | 0.099 ± 0.032 | 0.156 ± 0.005 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.561 ± 0.005 | 11.999 ± 0.001 | unsupported | 0.647 ± 0.134 | 1.157 ± 0.161 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.197 ± 0.001 | 0.198 ± 0.001 | unsupported | 0.226 ± 0.004 | 0.205 ± 0.003 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 3.141 ± 0.009 | 3.122 ± 0.013 | unsupported | 4.413 ± 0.037 | 1.785 ± 0.063 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.076 ± 0.000 | 12.000 ± 0.187 | unsupported | 0.107 ± 0.001 | 0.143 ± 0.007 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 1.181 ± 0.009 | 11.999 ± 0.002 | unsupported | 4.456 ± 0.701 | 0.972 ± 0.015 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.067 ± 0.012 | 0.097 ± 0.012 | unsupported | 0.159 ± 0.001 | 0.200 ± 0.001 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.713 ± 0.018 | 0.787 ± 0.109 | unsupported | 0.898 ± 0.003 | 1.869 ± 0.029 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.094 ± 0.004 | 35.999 ± 0.156 | unsupported | 0.068 ± 0.002 | 0.179 ± 0.001 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.453 ± 0.059 | 95.999 ± 0.003 | unsupported | 1.084 ± 0.537 | 1.158 ± 0.038 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.132 ± 0.003 | 0.195 ± 0.121 | unsupported | 0.135 ± 0.004 | 0.241 ± 0.004 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.676 ± 0.015 | 1.744 ± 0.042 | unsupported | 1.239 ± 0.058 | 2.233 ± 0.016 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.149 ± 0.001 | 43.968 ± 7.249 | unsupported | 0.064 ± 0.001 | 0.204 ± 0.005 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 1.001 ± 0.156 | 120.000 ± 0.003 | unsupported | 1.557 ± 0.540 | 1.350 ± 0.058 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.301 ± 0.003 | 0.344 ± 0.010 | unsupported | unsupported | 0.244 ± 0.010 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 4.377 ± 0.043 | 4.510 ± 0.116 | unsupported | unsupported | 3.119 ± 0.085 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.287 ± 0.005 | 51.953 ± 0.156 | unsupported | unsupported | 0.252 ± 0.005 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 2.103 ± 0.132 | 143.998 ± 0.377 | unsupported | unsupported | 1.867 ± 0.049 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.012 ± 0.000 | 0.020 ± 0.001 | unsupported | 0.013 ± 0.001 | 0.065 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.121 ± 0.007 | 0.118 ± 0.005 | unsupported | 0.101 ± 0.018 | 0.229 ± 0.022 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.017 ± 0.001 | 5.984 ± 0.078 | unsupported | 0.015 ± 0.000 | 0.038 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.124 ± 0.025 | 23.999 ± 0.003 | unsupported | 0.273 ± 0.180 | 0.172 ± 0.012 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.033 ± 0.010 | 0.056 ± 0.003 | unsupported | 0.103 ± 0.003 | 0.186 ± 0.002 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.417 ± 0.031 | 0.468 ± 0.031 | unsupported | 0.548 ± 0.023 | 1.641 ± 0.055 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.053 ± 0.004 | 17.859 ± 3.609 | unsupported | 0.099 ± 0.004 | 0.149 ± 0.003 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.232 ± 0.025 | 47.998 ± 0.120 | unsupported | 1.343 ± 0.527 | 1.068 ± 0.065 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.054 ± 0.000 | 0.055 ± 0.001 | unsupported | 0.119 ± 0.001 | 0.193 ± 0.004 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.862 ± 0.006 | 0.868 ± 0.003 | unsupported | 1.781 ± 0.027 | 1.709 ± 0.023 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.027 ± 0.001 | 12.000 ± 0.000 | unsupported | 0.095 ± 0.000 | 0.135 ± 0.003 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.392 ± 0.004 | 11.999 ± 0.001 | unsupported | 0.688 ± 0.008 | 0.921 ± 0.048 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.056 ± 0.000 | 0.057 ± 0.001 | unsupported | 0.124 ± 0.001 | 0.183 ± 0.021 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.896 ± 0.002 | 0.901 ± 0.009 | unsupported | 1.856 ± 0.054 | 1.710 ± 0.013 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.028 ± 0.000 | 12.000 ± 0.000 | unsupported | 0.099 ± 0.003 | 0.137 ± 0.007 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.409 ± 0.000 | 11.999 ± 0.000 | unsupported | 0.713 ± 0.027 | 0.935 ± 0.016 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.132 ± 0.012 | 0.156 ± 0.006 | unsupported | 0.138 ± 0.001 | 0.214 ± 0.001 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.722 ± 0.116 | 1.771 ± 0.055 | unsupported | 1.156 ± 0.007 | 2.151 ± 0.028 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.145 ± 0.003 | 43.953 ± 0.126 | unsupported | 0.060 ± 0.002 | 0.203 ± 0.002 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 1.024 ± 0.275 | 99.312 ± 57.308 | unsupported | 1.550 ± 0.702 | 1.289 ± 0.141 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.183 ± 0.001 | 0.184 ± 0.001 | unsupported | 0.276 ± 0.005 | 0.337 ± 0.004 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 2.913 ± 0.013 | 2.911 ± 0.046 | unsupported | 4.324 ± 0.026 | 3.978 ± 0.046 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.073 ± 0.001 | 12.937 ± 2.625 | unsupported | 0.199 ± 0.003 | 0.204 ± 0.005 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 1.127 ± 0.001 | 11.999 ± 0.001 | unsupported | 1.589 ± 0.134 | 2.057 ± 0.028 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.752 ± 0.015 | 1.408 ± 0.040 | unsupported | 0.268 ± 0.004 | 0.225 ± 0.021 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 19.739 ± 0.259 | 30.797 ± 0.597 | unsupported | 5.800 ± 0.087 | 2.301 ± 0.049 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 3.077 ± 2.427 | 93.025 ± 0.135 | unsupported | 0.261 ± 0.003 | 0.158 ± 0.017 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 9.141 ± 0.091 | 154.750 ± 1.125 | unsupported | 2.274 ± 0.043 | 1.215 ± 0.019 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 34.104 ± 1.460 | - | unsupported | 33.406 ± 1.063 | - | 34.744 ± 0.722 | 40.563 ± 0.344 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 23.572 ± 1.110 | - | unsupported | 23.319 ± 0.578 | - | 30.387 ± 0.783 | 23.345 ± 0.826 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 23.770 ± 0.545 | - | unsupported | 28.860 ± 0.375 | - | 23.009 ± 0.195 | 24.745 ± 0.905 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 16.900 ± 0.917 | - | unsupported | 17.398 ± 0.987 | - | 20.662 ± 0.370 | 16.572 ± 0.163 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 15.966 ± 0.364 | - | unsupported | 25.341 ± 0.647 | - | 15.081 ± 0.491 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 16.722 ± 1.060 | - | unsupported | 17.151 ± 0.380 | - | 14.695 ± 0.320 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 31.126 ± 1.481 | - | unsupported | 31.563 ± 0.538 | - | 33.918 ± 0.968 | 38.833 ± 0.352 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 23.072 ± 0.644 | - | unsupported | 23.735 ± 5.461 | - | 28.495 ± 0.663 | 23.209 ± 0.383 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 50.392 ± 0.181 | - | unsupported | 51.233 ± 0.381 | - | 48.969 ± 0.097 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 54.710 ± 0.400 | - | unsupported | 25.268 ± 0.210 | - | 23.581 ± 0.233 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 50.082 ± 0.455 | - | unsupported | 51.246 ± 0.305 | - | 48.499 ± 0.192 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 54.185 ± 0.610 | - | unsupported | 25.115 ± 0.164 | - | 23.454 ± 8.804 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 32.608 ± 1.193 | - | unsupported | 31.191 ± 0.731 | - | 33.380 ± 0.161 | 39.851 ± 0.818 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 24.581 ± 0.629 | - | unsupported | 23.212 ± 0.287 | - | 28.814 ± 0.249 | 22.950 ± 2.282 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 25.516 ± 0.324 | - | unsupported | 25.351 ± 0.371 | - | 28.171 ± 0.607 | 29.372 ± 0.859 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 16.063 ± 0.521 | - | unsupported | 16.320 ± 0.202 | - | 23.432 ± 0.452 | 15.771 ± 0.268 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.744 ± 0.008 | unsupported | unsupported | 0.905 ± 0.007 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 11.384 ± 0.049 | unsupported | unsupported | 13.151 ± 0.194 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 1.478 ± 0.002 | unsupported | unsupported | 0.610 ± 0.004 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 192.749 ± 3.759 | unsupported | unsupported | 6.784 ± 0.111 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 34.161 ± 2.362 | - | unsupported | 32.588 ± 1.510 | - | 34.872 ± 0.449 | 40.806 ± 0.720 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 23.463 ± 0.523 | - | unsupported | 23.620 ± 1.175 | - | 29.891 ± 1.359 | 23.071 ± 0.414 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 109.807 ± 2.616 | - | 0.002 ± 0.000 | 109.269 ± 1.797 | 107.097 ± 0.330 | 132.578 ± 2.021 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 42.695 ± 0.217 | - | 0.002 ± 0.000 | 35.314 ± 0.376 | 54.857 ± 0.892 | 132.869 ± 3.763 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 67.381 ± 2.135 | - | 213.212 ± 8.202 | 69.961 ± 1.530 | 70.690 ± 1.734 | 77.952 ± 3.081 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 22.890 ± 0.261 | - | 257.453 ± 12.256 | 23.384 ± 0.727 | 35.558 ± 0.644 | 76.499 ± 2.711 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 213.574 ± 6.216 | - | 212.855 ± 5.479 | 216.727 ± 8.875 | 219.266 ± 5.112 | 246.861 ± 3.834 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 71.116 ± 0.459 | - | 226.702 ± 0.494 | 70.582 ± 2.357 | 110.000 ± 2.747 | 248.439 ± 3.003 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 68.505 ± 2.447 | - | 68.212 ± 1.961 | 73.313 ± 2.588 | 69.601 ± 1.985 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 41.704 ± 0.503 | - | 70.079 ± 1.524 | 25.159 ± 0.418 | 35.915 ± 0.601 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 137.297 ± 4.126 | - | 0.002 ± 0.000 | 126.627 ± 2.445 | 123.406 ± 2.203 | 155.954 ± 6.640 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 41.041 ± 0.339 | - | 0.002 ± 0.000 | 40.789 ± 0.541 | 52.186 ± 1.661 | 155.269 ± 6.398 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 86.520 ± 3.143 | - | 0.002 ± 0.000 | 82.677 ± 1.290 | 83.407 ± 0.968 | 215.280 ± 10.941 | 129.102 ± 1.591 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 28.761 ± 0.904 | - | 0.002 ± 0.000 | 80.028 ± 1.115 | 43.398 ± 0.565 | 213.190 ± 1.455 | 57.097 ± 10.696 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 61.443 ± 3.132 | - | 61.473 ± 2.156 | 66.116 ± 1.116 | 64.508 ± 2.610 | 81.490 ± 0.696 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 21.271 ± 0.304 | - | 71.317 ± 0.956 | 23.193 ± 0.357 | 33.113 ± 0.868 | 80.842 ± 3.121 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 60.204 ± 0.966 | - | 59.648 ± 1.589 | 66.303 ± 1.911 | 63.685 ± 1.408 | 82.172 ± 9.152 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 20.502 ± 1.567 | - | 71.209 ± 1.152 | 22.368 ± 0.532 | 33.086 ± 0.231 | 82.912 ± 1.057 | - |
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
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 7.864947 | 491559.19 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 7.489940 | 468121.25 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 8.017983 | 501123.94 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 3.560000 | 222500.00 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 6.567268 | 410454.25 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 829.982026 | 51873876.62 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 22.446747 | 1402921.69 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 23.331534 | 1458220.88 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 7.068532 | 441783.25 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 344.576826 | 21536051.62 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 11.976394 | 748524.62 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 9.888674 | 618042.12 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 7.961876 | 497617.25 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 8.375967 | 523497.94 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 154.272461 | 9642028.81 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 141.981484 | 8873842.75 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 127.251206 | 7953200.38 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 145.392495 | 9087030.94 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 20.164808 | 1260300.50 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 8.165403 | 510337.69 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 12.515387 | 782211.69 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 13.020148 | 813759.25 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 154.371557 | 9648222.31 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 90.238704 | 5639919.00 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 36411.010304 | 2275688144.00 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 36971.319013 | 2310707438.31 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 18.941141 | 1183821.31 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 18.976838 | 1186052.38 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 350.307742 | 21894233.88 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 352.430774 | 22026923.38 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 25.146961 | 1571685.06 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 25.529224 | 1595576.50 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 362.442280 | 22652642.50 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 363.600047 | 22725002.94 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 17.636431 | 1102276.94 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 17.757224 | 1109826.50 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 333.259629 | 20828726.81 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 332.599199 | 20787449.94 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 24.017456 | 1501091.00 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 24.210916 | 1513182.25 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 345.810328 | 21613145.50 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 347.664711 | 21729044.44 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.402928 | 37545.75 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 10.340967 | 161577.61 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.963569 | 140055.77 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.924014 | 139437.72 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 8.210959 | 1026369.88 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 13.259351 | 1657418.88 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 17.857306 | 2232163.25 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 17.764974 | 2220621.75 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.941939 | 124092.80 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.872211 | 76128.30 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.692138 | 73314.66 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 380.970843 | 5952669.42 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.832404 | 354050.50 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.224967 | 1028120.87 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.181364 | 897670.50 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 191.993177 | 23999147.12 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 16.107424 | 251678.50 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.582934 | 134108.34 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.017076 | 78391.81 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.063903 | 79123.48 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 20.817405 | 2602175.62 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.424843 | 1803105.38 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.993897 | 1249237.12 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 9.948213 | 1243526.62 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.012615 | 156447.11 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.347769 | 99183.89 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.389120 | 37330.00 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.982327 | 11999723.86 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.258791 | 1157348.88 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.173714 | 646714.25 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.490443 | 561305.38 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.993456 | 11999182.00 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.116966 | 204952.59 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 14.476639 | 226197.48 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 12.601395 | 196896.80 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 12.667598 | 197931.22 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.277095 | 1784636.88 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 35.305628 | 4413203.50 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 25.126107 | 3140763.38 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 24.979184 | 3122398.00 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.155038 | 143047.47 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.855304 | 107114.12 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.871673 | 76119.89 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.981691 | 11999713.92 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.774788 | 971848.50 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 35.644599 | 4455574.88 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.450149 | 1181268.62 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.993817 | 11999227.12 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.792161 | 199877.52 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 10.148528 | 158570.75 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.258073 | 66532.39 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.232451 | 97382.05 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.949387 | 1868673.38 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.182295 | 897786.88 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.705727 | 713215.88 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.294895 | 786861.88 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.435605 | 178681.33 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.359218 | 68112.78 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.984153 | 93502.39 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2303.957997 | 35999343.70 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.260454 | 1157556.75 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.671499 | 1083937.38 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.623357 | 452919.62 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 767.990536 | 95998817.00 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 15.454468 | 241476.06 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.626334 | 134786.47 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.449912 | 132029.88 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 12.463668 | 194744.81 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 17.865127 | 2233140.88 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.913151 | 1239143.88 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.405754 | 1675719.25 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 13.949176 | 1743647.00 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.044520 | 203820.62 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.117326 | 64333.22 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 9.529419 | 148897.17 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2813.965637 | 43968213.08 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.800561 | 1350070.12 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.452169 | 1556521.12 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.005239 | 1000654.88 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 959.997451 | 119999681.38 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 15.598462 | 243725.97 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 19.258998 | 300921.84 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 21.993322 | 343645.66 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 24.952252 | 3119031.50 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 35.014879 | 4376859.88 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 36.076396 | 4509549.50 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 16.101858 | 251591.53 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 18.367492 | 286992.06 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3324.975909 | 51952748.58 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 14.933408 | 1866676.00 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 16.823532 | 2102941.50 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1151.987334 | 143998416.75 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.148229 | 64816.08 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.828643 | 12947.55 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.740119 | 11564.36 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.270748 | 19855.44 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 1.831670 | 228958.75 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.804358 | 100544.75 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.967913 | 120989.12 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.945882 | 118235.25 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.435114 | 38048.66 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.931945 | 14561.64 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.089511 | 17023.61 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 382.972644 | 5983947.56 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.379179 | 172397.38 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.187213 | 273401.62 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.988711 | 123588.88 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 191.989167 | 23998645.88 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.882216 | 185659.62 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.582047 | 102844.48 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.138424 | 33412.88 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.561004 | 55640.69 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.129618 | 1641202.25 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.386118 | 548264.75 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.332757 | 416594.62 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.740567 | 467570.88 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.521931 | 148780.17 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.309179 | 98580.92 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.423555 | 53493.05 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1142.989127 | 17859205.11 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.545833 | 1068229.12 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 10.743227 | 1342903.38 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.853632 | 231704.00 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 383.986442 | 47998305.25 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.353680 | 193026.25 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.606655 | 118853.98 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.464614 | 54134.59 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.494501 | 54601.58 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.668222 | 1708527.75 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.249616 | 1781202.00 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.893482 | 861685.25 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.945860 | 868232.50 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.660847 | 135325.73 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.078096 | 94970.25 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.726816 | 26981.50 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.986055 | 11999782.11 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.364313 | 920539.12 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.504210 | 688026.25 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.138375 | 392296.88 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.993175 | 11999146.88 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.717755 | 183089.92 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.954844 | 124294.44 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.590949 | 56108.58 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.666690 | 57292.03 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.676677 | 1709584.62 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.850595 | 1856324.38 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.166450 | 895806.25 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.207687 | 900960.88 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.751637 | 136744.33 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.322141 | 98783.45 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.788030 | 27937.97 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.982589 | 11999727.95 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.476972 | 934621.50 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.704213 | 713026.62 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.272355 | 409044.38 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.994148 | 11999268.50 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.680014 | 213750.22 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.822820 | 137856.56 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.451825 | 132059.77 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.996072 | 156188.62 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 17.208324 | 2151040.50 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.248302 | 1156037.75 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.775883 | 1721985.38 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 14.166050 | 1770756.25 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.996381 | 203068.45 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.869224 | 60456.62 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 9.260987 | 144702.92 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2812.968512 | 43952633.00 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.313876 | 1289234.50 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.396716 | 1549589.50 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.194933 | 1024366.63 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 794.499663 | 99312457.88 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 21.573279 | 337082.48 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.659786 | 275934.16 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 11.727918 | 183248.72 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 11.769155 | 183893.05 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 31.824411 | 3978051.38 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 34.590536 | 4323817.00 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 23.302380 | 2912797.50 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 23.287101 | 2910887.62 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.050322 | 203911.28 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 12.733037 | 198953.70 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.667903 | 72935.98 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 827.972764 | 12937074.44 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 16.457981 | 2057247.62 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.713736 | 1589217.00 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.015359 | 1126919.88 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.993686 | 11999210.75 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.421390 | 225334.22 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.160657 | 268135.27 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 48.135832 | 752122.38 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 90.101129 | 1407830.14 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.409961 | 2301245.12 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 46.401330 | 5800166.25 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 157.914151 | 19739268.88 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 246.379987 | 30797498.38 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.121891 | 158154.55 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 16.731728 | 261433.25 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 196.908337 | 3076692.77 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5953.605918 | 93025092.47 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.718447 | 1214805.88 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 18.191807 | 2273975.88 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 73.124793 | 9140599.12 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1237.998106 | 154749763.25 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 14.472489 | 904530.56 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 11.897204 | 743575.25 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 210.409200 | 13150575.00 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 182.143508 | 11383969.25 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 9.762798 | 610174.88 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 23.647300 | 1477956.25 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 108.541212 | 6783825.75 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 3083.985978 | 192749123.62 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.023334 | 1458.38 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002725 | 170.31 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.023904 | 1494.00 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002665 | 166.56 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000121 | 7.56 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.022192 | 1387.00 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.007665 | 479.06 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000180 | 11.25 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.022502 | 1406.38 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.006913 | 432.06 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000120 | 7.50 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.022773 | 1423.31 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.006041 | 377.56 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000210 | 13.12 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.022572 | 1410.75 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.005360 | 335.00 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000110 | 6.88 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.020849 | 1303.06 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003466 | 216.62 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000100 | 6.25 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.021029 | 1314.31 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003226 | 201.62 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `jax-cpu` is 23.3x faster than the slowest successful cell (`tenferro-trace`, 59.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `julia-base` is 14.6x faster than the slowest successful cell (`tenferro-trace`, 59.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 60.6x faster than the slowest successful cell (`tenferro-trace`, 59.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 19.1x faster than the slowest successful cell (`tenferro-direct`, 130.983 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `julia-base` is 18.1x faster than the slowest successful cell (`tenferro-direct`, 130.983 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 22.7x faster than the slowest successful cell (`tenferro-direct`, 130.983 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 29.0x faster than the slowest successful cell (`tenferro-trace`, 136.009 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `julia-base` is 31.5x faster than the slowest successful cell (`tenferro-trace`, 136.009 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 37.5x faster than the slowest successful cell (`tenferro-trace`, 136.009 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/masked_log_softmax_axis1` (f64, threads=4, shape=`1024x64`): `jax-cpu` is 233.1x faster than the slowest successful cell (`tenferro-eager`, 51.874 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/masked_log_softmax_axis1` (f64, threads=4, shape=`1024x64`): `tenferro-direct` is 126.4x faster than the slowest successful cell (`tenferro-eager`, 51.874 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 10.0x faster than the slowest successful cell (`pytorch-cpu`, 24.434 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 16.8x faster than the slowest successful cell (`pytorch-cpu`, 24.434 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 12.2x faster than the slowest successful cell (`pytorch-cpu`, 24.434 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 12.2x faster than the slowest successful cell (`pytorch-cpu`, 24.434 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 15.2x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 19.6x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 30.5x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 17.3x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 17.4x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 18.6x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 15.1x faster than the slowest successful cell (`pytorch-cpu`, 23.401 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 23.401 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 23.401 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 15.0x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 20.5x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 30.1x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 13.4x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 23.1x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 25.0x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 38.2x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 12.3x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 20.7x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 67.4x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 35.2x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 13.4x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 25.4x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 26.1x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 42.1x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 15.2x faster than the slowest successful cell (`tenferro-trace`, 24.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 22.6x faster than the slowest successful cell (`tenferro-trace`, 24.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 67.3x faster than the slowest successful cell (`tenferro-trace`, 24.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 36.2x faster than the slowest successful cell (`tenferro-trace`, 24.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_squares_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 48.7x faster than the slowest successful cell (`tenferro-eager`, 21.536 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `jax-cpu` is 13.3x faster than the slowest successful cell (`tenferro-trace`, 12.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `pytorch-cpu` is 34.5x faster than the slowest successful cell (`tenferro-trace`, 12.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `tenferro-direct` is 28.8x faster than the slowest successful cell (`tenferro-trace`, 12.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `jax-cpu` is 19.6x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `pytorch-cpu` is 34.4x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `tenferro-direct` is 24.2x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 11.0x faster than the slowest successful cell (`julia-base`, 7.040 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `jax-cpu` is 106.9x faster than the slowest successful cell (`tenferro-trace`, 47.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `julia-base` is 40.1x faster than the slowest successful cell (`tenferro-trace`, 47.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 137.1x faster than the slowest successful cell (`tenferro-trace`, 47.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `tenferro-direct` is 98.4x faster than the slowest successful cell (`tenferro-trace`, 47.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `jax-cpu` is 11.5x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 22.2x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 21.9x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `jax-cpu` is 18.0x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 34.7x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 28.4x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `jax-cpu` is 77.7x faster than the slowest successful cell (`tenferro-trace`, 36.016 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `julia-base` is 23.9x faster than the slowest successful cell (`tenferro-trace`, 36.016 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 53.1x faster than the slowest successful cell (`tenferro-trace`, 36.016 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `tenferro-direct` is 29.4x faster than the slowest successful cell (`tenferro-trace`, 36.016 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 5251.4x faster than the slowest successful cell (`julia-base`, 9.578 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `pytorch-cpu` is 10.0x faster than the slowest successful cell (`julia-base`, 9.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-direct` is 10.9x faster than the slowest successful cell (`julia-base`, 9.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 4759.8x faster than the slowest successful cell (`julia-base`, 9.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.011 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 13.4x faster than the slowest successful cell (`jax-cpu`, 0.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x8x8`): `tenferro-direct` is 13.0x faster than the slowest successful cell (`jax-cpu`, 0.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`3x2x2`): `tenferro-direct` is 13.6x faster than the slowest successful cell (`jax-cpu`, 0.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`3x4x4`): `tenferro-direct` is 12.2x faster than the slowest successful cell (`jax-cpu`, 0.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x2x2`): `tenferro-direct` is 12.6x faster than the slowest successful cell (`jax-cpu`, 0.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 21.9x faster than the slowest successful cell (`jax-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 11.1x faster than the slowest successful cell (`jax-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x2x2`): `tenferro-direct` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 12.1x faster than the slowest successful cell (`jax-cpu`, 0.034 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 14.9x faster than the slowest successful cell (`jax-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 13.0x faster than the slowest successful cell (`jax-cpu`, 0.025 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 11.3x faster than the slowest successful cell (`jax-cpu`, 0.025 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x2x2,rhs=1`): `tenferro-direct` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.021 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x4x4,rhs=1`): `tenferro-direct` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.019 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `pytorch-cpu` is 15.7x faster than the slowest successful cell (`tenferro-trace`, 5.988 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `tenferro-direct` is 11.1x faster than the slowest successful cell (`tenferro-trace`, 5.988 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x4x4,rhs=1`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.025 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `jax-cpu` is 31.4x faster than the slowest successful cell (`tenferro-trace`, 23.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `pytorch-cpu` is 41.1x faster than the slowest successful cell (`tenferro-trace`, 23.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `tenferro-direct` is 25.1x faster than the slowest successful cell (`tenferro-trace`, 23.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 22.2x faster than the slowest successful cell (`tenferro-trace`, 327.006 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `julia-base` is 26.1x faster than the slowest successful cell (`tenferro-trace`, 327.006 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 29.4x faster than the slowest successful cell (`tenferro-trace`, 327.006 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`768x384`): `jax-cpu` is 239.5x faster than the slowest successful cell (`tenferro-eager`, 2310.707 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 409.7x faster than the slowest successful cell (`tenferro-eager`, 2310.707 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 14.1x faster than the slowest successful cell (`tenferro-trace`, 280.682 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `julia-base` is 16.9x faster than the slowest successful cell (`tenferro-trace`, 280.682 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 13.0x faster than the slowest successful cell (`tenferro-trace`, 280.682 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `jax-cpu` is 42.8x faster than the slowest successful cell (`tenferro-eager`, 2288.912 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `julia-base` is 259.2x faster than the slowest successful cell (`tenferro-eager`, 2288.912 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 372.1x faster than the slowest successful cell (`tenferro-eager`, 2288.912 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 10.0x faster than the slowest successful cell (`tenferro-trace`, 250.029 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `julia-base` is 10.3x faster than the slowest successful cell (`tenferro-trace`, 250.029 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 15.5x faster than the slowest successful cell (`tenferro-trace`, 250.029 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `pytorch-cpu` is 14.4x faster than the slowest successful cell (`jax-cpu`, 20.364 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 11.5x faster than the slowest successful cell (`jax-cpu`, 20.364 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 12.6x faster than the slowest successful cell (`jax-cpu`, 20.364 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 16.4x faster than the slowest successful cell (`tenferro-trace`, 24.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 50.5x faster than the slowest successful cell (`tenferro-trace`, 24.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `jax-cpu` is 11.4x faster than the slowest successful cell (`tenferro-direct`, 230.952 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `julia-base` is 11.7x faster than the slowest successful cell (`tenferro-direct`, 230.952 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 14.0x faster than the slowest successful cell (`tenferro-direct`, 230.952 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `jax-cpu` is 10.2x faster than the slowest successful cell (`tenferro-direct`, 207.772 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `julia-base` is 10.6x faster than the slowest successful cell (`tenferro-direct`, 207.772 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 12.6x faster than the slowest successful cell (`tenferro-direct`, 207.772 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 23.3x faster than the slowest successful cell (`tenferro-trace`, 327.021 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `julia-base` is 26.1x faster than the slowest successful cell (`tenferro-trace`, 327.021 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 29.0x faster than the slowest successful cell (`tenferro-trace`, 327.021 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 44.5x faster than the slowest successful cell (`tenferro-trace`, 2336.912 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `julia-base` is 46.2x faster than the slowest successful cell (`tenferro-trace`, 2336.912 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 61.8x faster than the slowest successful cell (`tenferro-trace`, 2336.912 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 48.0x faster than the slowest successful cell (`tenferro-eager`, 5.953 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 78.2x faster than the slowest successful cell (`tenferro-eager`, 5.953 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 81.2x faster than the slowest successful cell (`tenferro-eager`, 5.953 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 67.8x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 23.3x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 26.7x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 76.7x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 121.0x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 321.4x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 10.4x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 18.6x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 21.4x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 83.9x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 112.0x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 157.6x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 12.3x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 201.5x faster than the slowest successful cell (`tenferro-eager`, 35.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 528.5x faster than the slowest successful cell (`tenferro-eager`, 35.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 385.0x faster than the slowest successful cell (`tenferro-eager`, 35.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 82.9x faster than the slowest successful cell (`tenferro-eager`, 95.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 88.6x faster than the slowest successful cell (`tenferro-eager`, 95.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 212.0x faster than the slowest successful cell (`tenferro-eager`, 95.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 215.7x faster than the slowest successful cell (`tenferro-eager`, 43.968 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 683.4x faster than the slowest successful cell (`tenferro-eager`, 43.968 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 295.3x faster than the slowest successful cell (`tenferro-eager`, 43.968 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 88.9x faster than the slowest successful cell (`tenferro-eager`, 120.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 77.1x faster than the slowest successful cell (`tenferro-eager`, 120.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 119.9x faster than the slowest successful cell (`tenferro-eager`, 120.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 206.5x faster than the slowest successful cell (`tenferro-eager`, 51.953 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 181.0x faster than the slowest successful cell (`tenferro-eager`, 51.953 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 77.1x faster than the slowest successful cell (`tenferro-eager`, 143.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 68.5x faster than the slowest successful cell (`tenferro-eager`, 143.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 157.3x faster than the slowest successful cell (`tenferro-eager`, 5.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 410.9x faster than the slowest successful cell (`tenferro-eager`, 5.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 351.5x faster than the slowest successful cell (`tenferro-eager`, 5.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 139.2x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 87.8x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 194.2x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 120.0x faster than the slowest successful cell (`tenferro-eager`, 17.859 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 181.2x faster than the slowest successful cell (`tenferro-eager`, 17.859 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 333.9x faster than the slowest successful cell (`tenferro-eager`, 17.859 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 44.9x faster than the slowest successful cell (`tenferro-eager`, 47.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 35.7x faster than the slowest successful cell (`tenferro-eager`, 47.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 207.2x faster than the slowest successful cell (`tenferro-eager`, 47.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 88.7x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 126.4x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 444.7x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 13.0x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 17.4x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 30.6x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 87.8x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 121.5x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 429.5x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 12.8x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 16.8x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 29.3x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 216.4x faster than the slowest successful cell (`tenferro-eager`, 43.953 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 727.0x faster than the slowest successful cell (`tenferro-eager`, 43.953 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 303.7x faster than the slowest successful cell (`tenferro-eager`, 43.953 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 77.0x faster than the slowest successful cell (`tenferro-eager`, 99.312 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 64.1x faster than the slowest successful cell (`tenferro-eager`, 99.312 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 97.0x faster than the slowest successful cell (`tenferro-eager`, 99.312 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 63.4x faster than the slowest successful cell (`tenferro-eager`, 12.937 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 65.0x faster than the slowest successful cell (`tenferro-eager`, 12.937 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 177.4x faster than the slowest successful cell (`tenferro-eager`, 12.937 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 10.6x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 13.4x faster than the slowest successful cell (`tenferro-eager`, 30.797 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 588.2x faster than the slowest successful cell (`tenferro-eager`, 93.025 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 355.8x faster than the slowest successful cell (`tenferro-eager`, 93.025 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 30.2x faster than the slowest successful cell (`tenferro-eager`, 93.025 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 127.4x faster than the slowest successful cell (`tenferro-eager`, 154.750 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 68.1x faster than the slowest successful cell (`tenferro-eager`, 154.750 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 16.9x faster than the slowest successful cell (`tenferro-eager`, 154.750 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/output_reuse/solve_read_into` (f64, threads=4, shape=`768x768,rhs=16`): `pytorch-cpu` is 28.4x faster than the slowest successful cell (`tenferro-direct`, 192.749 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 72685.2x faster than the slowest successful cell (`julia-base`, 132.578 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 75365.4x faster than the slowest successful cell (`julia-base`, 132.869 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/cast_f64_f32` (f64->f32, threads=4, shape=`33554432`): `pytorch-cpu` is 11.0x faster than the slowest successful cell (`tenferro-trace`, 257.453 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/cast_f64_f32` (f64->f32, threads=4, shape=`33554432`): `tenferro-direct` is 11.2x faster than the slowest successful cell (`tenferro-trace`, 257.453 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 81908.6x faster than the slowest successful cell (`julia-base`, 155.954 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 76713.8x faster than the slowest successful cell (`julia-base`, 155.269 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 122806.6x faster than the slowest successful cell (`julia-base`, 215.280 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 128971.5x faster than the slowest successful cell (`julia-base`, 213.190 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 173.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 127.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 177.9x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 108.5x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 186.1x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 219.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 9193.5 GB/s at 0.002 ms. The conservative physical lower bound is 0.168 ms, so the cell is 91.9x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 8761.2 GB/s at 0.002 ms. The conservative physical lower bound is 0.042 ms, so the cell is 21.9x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 147168.6 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1471.7x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 152260.6 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 380.7x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 140985.0 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1409.9x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 132626.2 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 331.6x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 76564.6 GB/s at 0.002 ms. The conservative physical lower bound is 1.342 ms, so the cell is 765.6x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 81196.4 GB/s at 0.002 ms. The conservative physical lower bound is 0.336 ms, so the cell is 203.0x faster than the bound.

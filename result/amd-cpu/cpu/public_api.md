# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/amd-cpu/cpu/public_api/20261109_190549/run.yaml`
- Timestamp: `2026-11-09T19:05:49.422912+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/amd-cpu/cpu/public_api/20261109_190549`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

## CPU Information

- Model: `AMD EPYC 9V74 80-Core Processor`
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

- CSV: `data/results/amd-cpu/cpu/public_api/20261109_190549/cpu_public_api_t1_20261109_190549.csv`
- CSV: `data/results/amd-cpu/cpu/public_api/20261109_190549/cpu_public_api_t4_20261109_190549.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 3.528 ± 0.029 | - | 3.643 ± 0.025 | 3.820 ± 0.026 | 3.554 ± 0.025 | 3.652 ± 0.221 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 3.941 ± 0.075 | - | 3.859 ± 0.023 | 1.901 ± 0.018 | 2.993 ± 0.400 | 4.859 ± 2.351 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 105.794 ± 5.182 | - | 105.566 ± 6.539 | 101.191 ± 2.004 | 103.792 ± 3.474 | 117.162 ± 14.135 | 120.200 ± 4.972 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 38.675 ± 0.083 | - | 130.517 ± 3.280 | 35.841 ± 0.305 | 55.557 ± 6.187 | 118.771 ± 4.755 | 56.838 ± 3.981 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 88.648 ± 4.418 | - | 92.771 ± 4.681 | 83.802 ± 2.671 | 80.260 ± 4.856 | 100.609 ± 4.102 | 106.429 ± 3.307 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 33.414 ± 0.569 | - | 107.359 ± 4.101 | 26.435 ± 0.188 | 40.494 ± 3.295 | 106.938 ± 0.916 | 48.484 ± 2.077 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 40.702 ± 0.171 | - | 40.565 ± 0.120 | 40.833 ± 1.624 | 58.923 ± 0.611 | 43.173 ± 0.141 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 41.324 ± 0.471 | - | 42.762 ± 8.320 | 26.051 ± 0.066 | 25.933 ± 0.383 | 21.309 ± 0.316 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 41.549 ± 0.103 | - | unsupported | 40.776 ± 0.425 | 59.349 ± 0.336 | 41.779 ± 1.979 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 42.793 ± 0.233 | - | unsupported | 26.404 ± 0.181 | 27.010 ± 1.538 | 23.971 ± 4.070 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 9.068 ± 0.022 | - | 9.103 ± 0.036 | 9.654 ± 0.027 | 9.666 ± 0.099 | 9.618 ± 0.123 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 11.787 ± 0.105 | - | 11.932 ± 0.110 | 11.697 ± 0.042 | 14.826 ± 2.661 | 14.769 ± 0.045 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 60.226 ± 0.346 | - | 60.677 ± 0.173 | 94.357 ± 1.147 | 84.175 ± 2.157 | 65.794 ± 3.281 | 65.731 ± 4.041 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 25.742 ± 0.190 | - | 71.996 ± 0.002 | 37.375 ± 0.399 | 43.856 ± 0.990 | 63.159 ± 1.917 | 30.600 ± 10.320 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 87.897 ± 0.680 | - | 87.829 ± 0.203 | 126.129 ± 2.257 | 141.624 ± 2.876 | 114.574 ± 2.235 | 116.949 ± 0.753 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 40.331 ± 0.901 | - | 108.009 ± 12.001 | 52.189 ± 0.593 | 59.448 ± 0.777 | 114.709 ± 0.925 | 50.208 ± 2.122 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 61.199 ± 3.155 | - | 61.443 ± 3.501 | 60.116 ± 2.540 | 68.266 ± 3.873 | 64.751 ± 0.268 | 68.230 ± 0.723 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 20.906 ± 0.334 | - | 82.512 ± 1.371 | 22.113 ± 0.800 | 31.101 ± 1.424 | 65.138 ± 1.649 | 31.661 ± 1.022 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.302 ± 0.062 | - | 15.962 ± 0.063 | 18.273 ± 0.529 | 16.056 ± 0.447 | 3.463 ± 0.019 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 0.934 ± 0.072 | - | 60.000 ± 0.008 | 9.023 ± 0.021 | 2.903 ± 0.039 | 3.480 ± 0.012 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 6.942 ± 0.023 | - | 7.157 ± 0.057 | 8.162 ± 0.033 | 6.682 ± 0.705 | 8.105 ± 0.071 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 129.981 ± 61.089 | - | 115.031 ± 11.674 | 4.902 ± 0.053 | 6.988 ± 0.702 | 5.696 ± 0.096 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 4.611 ± 0.130 | - | 4.780 ± 0.134 | 4.849 ± 0.047 | 4.562 ± 0.029 | 4.246 ± 0.038 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 88.907 ± 18.077 | - | 121.007 ± 22.912 | 2.925 ± 0.050 | 3.308 ± 0.049 | 2.955 ± 0.020 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 5.195 ± 0.155 | - | 5.235 ± 0.083 | 6.496 ± 0.077 | 4.369 ± 0.052 | 4.316 ± 0.037 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 9.564 ± 0.112 | - | 9.580 ± 0.126 | 5.043 ± 0.019 | 5.430 ± 0.013 | 13.086 ± 7.123 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 40.589 ± 0.039 | - | 40.288 ± 0.069 | 40.592 ± 0.384 | 59.553 ± 0.868 | 41.055 ± 0.101 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 41.930 ± 0.395 | - | 41.740 ± 22.736 | 26.142 ± 0.080 | 26.337 ± 0.975 | 20.462 ± 0.074 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 44.460 ± 11.555 | - | unsupported | 45.345 ± 0.692 | 45.819 ± 0.252 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 48.043 ± 0.942 | - | unsupported | 22.877 ± 1.714 | 26.751 ± 0.721 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 104.753 ± 7.802 | - | 106.264 ± 10.431 | 103.486 ± 6.728 | 111.273 ± 11.555 | 118.086 ± 6.363 | 162.667 ± 28.161 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 36.975 ± 1.139 | - | 118.821 ± 0.446 | 35.336 ± 0.794 | 57.291 ± 5.831 | 113.835 ± 10.141 | 62.972 ± 1.061 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 117.840 ± 5.719 | - | 117.426 ± 7.355 | 114.874 ± 4.637 | 123.520 ± 7.322 | 123.786 ± 8.370 | 174.267 ± 1.986 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 40.077 ± 0.413 | - | 129.914 ± 4.975 | 40.539 ± 1.816 | 64.760 ± 2.959 | 123.655 ± 8.152 | 71.050 ± 2.734 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 14.417 ± 0.156 | - | 39.737 ± 0.108 | 54.227 ± 1.022 | 43.134 ± 0.664 | 63.217 ± 7.441 | 67.489 ± 10.464 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 7.973 ± 0.173 | - | 107.995 ± 1.000 | 25.357 ± 0.857 | 21.679 ± 0.376 | 61.533 ± 0.291 | 27.402 ± 11.440 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 8.333 ± 0.158 | - | 8.386 ± 0.250 | 29.954 ± 1.666 | 33.734 ± 0.916 | 37.715 ± 8.798 | 44.191 ± 10.183 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 6.989 ± 0.131 | - | 23.997 ± 12.007 | 11.737 ± 0.598 | 18.534 ± 1.554 | 39.803 ± 9.187 | 17.040 ± 19.461 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 18.559 ± 0.303 | - | 18.376 ± 0.858 | 42.068 ± 0.806 | 40.617 ± 0.768 | 19.448 ± 0.115 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 15.824 ± 0.138 | - | 35.992 ± 0.024 | 18.364 ± 0.628 | 22.668 ± 0.514 | 19.624 ± 0.399 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 13.492 ± 0.169 | - | 13.618 ± 0.061 | 50.094 ± 1.260 | 92.389 ± 0.764 | 51.403 ± 16.404 | 57.211 ± 8.460 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 5.925 ± 0.009 | - | 35.996 ± 0.008 | 19.560 ± 0.390 | 47.890 ± 1.166 | 53.608 ± 20.445 | 24.950 ± 13.228 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 120.028 ± 4.617 | - | 118.063 ± 8.126 | 112.250 ± 3.546 | 121.205 ± 9.894 | 130.990 ± 5.863 | 174.159 ± 4.331 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 40.099 ± 0.214 | - | 130.149 ± 1.946 | 39.449 ± 1.420 | 68.036 ± 4.212 | 128.706 ± 5.095 | 68.668 ± 1.453 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 7.976 ± 0.026 | - | 8.089 ± 0.034 | 39.549 ± 1.546 | 33.942 ± 1.198 | 48.163 ± 10.355 | 53.379 ± 10.061 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 3.862 ± 0.053 | - | 23.997 ± 0.004 | 14.331 ± 0.438 | 17.774 ± 5.147 | 58.311 ± 9.146 | 33.567 ± 11.633 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 10.997 ± 0.019 | - | 10.773 ± 0.031 | 21.676 ± 1.205 | 19.307 ± 0.379 | 30.531 ± 7.539 | 31.640 ± 9.199 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 4.634 ± 0.014 | - | 24.000 ± 0.010 | 7.628 ± 0.315 | 9.887 ± 0.682 | 31.264 ± 12.413 | 12.736 ± 10.605 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 13.803 ± 0.125 | - | 13.959 ± 0.021 | 41.515 ± 0.089 | 51.922 ± 0.307 | 61.785 ± 41.219 | 66.351 ± 8.270 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 6.887 ± 0.045 | - | 35.997 ± 0.009 | 15.671 ± 0.137 | 29.157 ± 0.167 | 61.466 ± 13.700 | 27.437 ± 13.798 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 6.878 ± 0.026 | - | 6.880 ± 0.029 | 18.435 ± 1.232 | 30.784 ± 0.368 | 36.594 ± 8.612 | 36.882 ± 7.982 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 3.413 ± 0.010 | - | 23.998 ± 0.006 | 7.119 ± 0.135 | 15.481 ± 0.348 | 35.290 ± 1.361 | 15.767 ± 10.844 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.302 ± 0.005 | 0.333 ± 0.002 | unsupported | unsupported | 0.398 ± 0.006 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.209 ± 0.017 | 112.741 ± 2.262 | unsupported | unsupported | 0.190 ± 0.014 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 120.936 ± 6.402 | - | 120.162 ± 8.281 | 114.601 ± 4.899 | 116.737 ± 1.723 | 135.052 ± 6.036 | 181.922 ± 4.567 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 39.525 ± 0.619 | - | 129.830 ± 0.716 | 40.052 ± 1.422 | 60.017 ± 3.139 | 131.233 ± 10.465 | 71.357 ± 2.445 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 114.233 ± 6.431 | - | 119.384 ± 7.491 | 112.909 ± 5.930 | 117.453 ± 2.283 | 134.617 ± 3.809 | 178.840 ± 5.566 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 39.635 ± 2.384 | - | 129.774 ± 3.449 | 40.763 ± 1.210 | 59.278 ± 0.581 | 128.850 ± 5.475 | 70.895 ± 1.353 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 115.888 ± 0.776 | - | 115.375 ± 1.070 | 112.719 ± 3.515 | 119.212 ± 5.785 | 131.180 ± 0.952 | 176.160 ± 6.001 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 40.274 ± 0.219 | - | 126.480 ± 4.836 | 39.737 ± 0.876 | 65.838 ± 1.785 | 130.425 ± 7.779 | 69.521 ± 2.257 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 103.248 ± 7.555 | - | 102.725 ± 1.507 | 101.453 ± 4.429 | 110.489 ± 7.985 | 119.015 ± 6.652 | 155.204 ± 4.112 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 36.667 ± 2.755 | - | 117.178 ± 2.886 | 35.395 ± 0.596 | 57.013 ± 0.623 | 113.879 ± 6.157 | 63.499 ± 2.316 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 43.661 ± 0.146 | - | 42.961 ± 0.072 | 33.864 ± 0.922 | 57.386 ± 0.139 | 109.219 ± 9.481 | 102.842 ± 13.252 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 20.558 ± 0.624 | - | 59.999 ± 0.009 | 12.496 ± 0.152 | 28.928 ± 0.782 | 107.469 ± 2.766 | 37.390 ± 8.824 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 11.572 ± 0.056 | - | 11.574 ± 0.058 | 8.217 ± 0.174 | 37.927 ± 1.367 | 8.202 ± 0.020 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 7.329 ± 0.233 | - | 23.998 ± 0.006 | 7.739 ± 0.225 | 18.944 ± 0.297 | 8.214 ± 0.080 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.500 ± 0.030 | - | 1.509 ± 0.029 | 22.276 ± 0.035 | 2.071 ± 0.012 | 0.936 ± 0.015 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.698 ± 0.035 | - | 23.998 ± 0.015 | 10.034 ± 0.376 | 1.423 ± 0.311 | 0.925 ± 0.011 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 1.796 ± 0.054 | - | 1.861 ± 0.041 | 4.653 ± 0.018 | 3.442 ± 0.025 | 1.027 ± 0.060 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 1.047 ± 0.054 | - | 24.000 ± 0.006 | 2.353 ± 0.015 | 1.801 ± 0.010 | 1.352 ± 0.164 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 11.095 ± 0.032 | - | 11.130 ± 0.053 | 8.449 ± 0.431 | 37.703 ± 0.117 | 8.422 ± 0.185 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 7.412 ± 0.122 | - | 23.999 ± 0.026 | 7.524 ± 0.193 | 19.019 ± 0.137 | 8.150 ± 0.084 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.447 ± 0.020 | - | 1.446 ± 0.016 | 22.389 ± 0.097 | 2.121 ± 0.011 | 0.952 ± 0.021 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.694 ± 0.033 | - | 23.999 ± 0.011 | 11.094 ± 6.722 | 1.160 ± 0.016 | 0.916 ± 0.030 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 8.625 ± 0.117 | - | 8.490 ± 0.175 | 13.943 ± 0.074 | 13.511 ± 0.019 | 7.498 ± 0.103 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 4.712 ± 0.050 | - | 35.993 ± 0.014 | 7.091 ± 0.751 | 6.935 ± 3.173 | 7.531 ± 0.175 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 8.372 ± 0.089 | - | 8.218 ± 0.515 | 8.106 ± 0.099 | 27.632 ± 0.153 | 7.882 ± 0.212 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 7.525 ± 0.189 | - | 23.998 ± 0.011 | 7.951 ± 0.169 | 14.914 ± 2.401 | 7.951 ± 0.163 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.686 ± 0.127 | - | 0.700 ± 0.143 | 2.388 ± 0.116 | 1.622 ± 0.053 | 0.666 ± 0.101 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.377 ± 0.129 | - | 23.997 ± 0.004 | 1.536 ± 0.407 | 0.924 ± 0.027 | 0.615 ± 0.054 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.293 ± 0.206 | - | 1.305 ± 0.149 | 0.673 ± 0.016 | 1.178 ± 0.124 | 0.843 ± 0.131 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.727 ± 0.083 | - | 24.002 ± 0.013 | 0.345 ± 0.030 | 0.968 ± 0.214 | 0.857 ± 0.033 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 9.351 ± 0.292 | - | 8.397 ± 0.386 | 8.372 ± 0.226 | 27.491 ± 0.106 | 8.042 ± 0.284 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 7.578 ± 0.223 | - | 23.998 ± 0.010 | 7.865 ± 0.150 | 13.931 ± 0.016 | 7.989 ± 0.392 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.693 ± 0.137 | - | 0.736 ± 0.119 | 2.396 ± 0.080 | 1.711 ± 0.312 | 0.624 ± 0.070 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.359 ± 0.093 | - | 23.998 ± 0.010 | 1.292 ± 0.202 | 0.945 ± 0.007 | 0.602 ± 0.058 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.346 ± 0.134 | - | 1.315 ± 0.165 | 0.668 ± 0.014 | 1.056 ± 0.034 | 0.944 ± 0.219 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.720 ± 0.144 | - | 23.999 ± 0.013 | 0.298 ± 0.028 | 0.661 ± 0.055 | 0.836 ± 0.111 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.691 ± 0.015 | 0.687 ± 0.012 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.280 ± 0.057 | 23.999 ± 0.001 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 32.171 ± 0.218 | - | 33.226 ± 0.625 | 50.628 ± 0.553 | 55.635 ± 0.518 | 61.032 ± 8.347 | 67.989 ± 7.566 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 15.005 ± 0.560 | - | 47.997 ± 0.004 | 18.231 ± 0.798 | 32.740 ± 3.335 | 61.548 ± 9.135 | 33.079 ± 9.461 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 107.016 ± 8.186 | - | 114.946 ± 7.901 | 114.888 ± 3.267 | 200.478 ± 7.796 | 226.776 ± 3.381 | 230.512 ± 3.542 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 37.684 ± 2.768 | - | 131.407 ± 3.334 | 38.196 ± 2.264 | 83.535 ± 1.430 | 219.885 ± 5.019 | 88.902 ± 4.580 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 114.842 ± 5.956 | - | 118.970 ± 7.364 | 133.406 ± 3.996 | 137.260 ± 4.058 | 135.010 ± 5.479 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 42.545 ± 2.243 | - | 135.924 ± 16.526 | 45.718 ± 0.476 | 75.561 ± 5.000 | 131.199 ± 2.349 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 102.294 ± 8.228 | - | 108.481 ± 7.242 | 106.714 ± 5.280 | 101.003 ± 3.212 | 121.209 ± 5.911 | 157.729 ± 7.428 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 37.618 ± 1.565 | - | 131.472 ± 10.614 | 37.446 ± 1.988 | 53.920 ± 7.112 | 107.810 ± 12.111 | 62.881 ± 0.684 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 10.819 ± 0.795 | - | 11.339 ± 0.011 | 50.492 ± 1.687 | 92.824 ± 0.904 | 50.484 ± 5.567 | 54.709 ± 10.512 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 5.012 ± 0.052 | - | 23.998 ± 0.010 | 19.159 ± 0.259 | 47.478 ± 1.401 | 56.091 ± 14.344 | 24.918 ± 13.637 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 106.374 ± 8.718 | - | 106.526 ± 8.638 | 148.813 ± 3.317 | 119.759 ± 13.829 | 180.215 ± 4.151 | 182.572 ± 6.374 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 36.502 ± 0.420 | - | 129.088 ± 12.477 | 55.520 ± 1.137 | 57.778 ± 4.185 | 173.652 ± 3.567 | 72.109 ± 1.187 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 114.864 ± 6.562 | - | 117.409 ± 7.042 | 114.901 ± 3.779 | 127.112 ± 7.053 | 124.862 ± 7.485 | 175.090 ± 2.971 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 40.389 ± 0.076 | - | 130.150 ± 5.281 | 40.433 ± 1.375 | 65.941 ± 1.053 | 125.823 ± 4.611 | 68.260 ± 2.351 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 27.108 ± 0.032 | - | 27.594 ± 0.023 | 67.055 ± 1.830 | 30.658 ± 0.737 | 59.724 ± 8.949 | 65.064 ± 22.056 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 11.880 ± 0.040 | - | 47.997 ± 0.007 | 24.742 ± 0.569 | 17.155 ± 1.427 | 66.444 ± 9.112 | 28.222 ± 12.613 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 112.442 ± 6.572 | 112.936 ± 7.197 | unsupported | 135.852 ± 4.031 | 139.224 ± 1.582 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 41.937 ± 2.268 | 141.680 ± 11.858 | unsupported | 45.157 ± 0.615 | 79.213 ± 5.431 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 1.152 ± 0.050 | - | 0.838 ± 0.103 | 0.673 ± 0.030 | 0.662 ± 0.058 | 6.452 ± 3.635 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.371 ± 0.069 | - | 12.001 ± 0.010 | 0.288 ± 0.034 | 0.739 ± 0.082 | 1.035 ± 5.386 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.818 ± 0.124 | - | 0.775 ± 0.123 | 0.702 ± 0.072 | 0.729 ± 0.163 | 6.588 ± 4.056 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.372 ± 0.074 | - | 11.996 ± 0.007 | 0.320 ± 0.043 | 0.609 ± 0.251 | 6.755 ± 4.126 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.186 ± 0.150 | - | unsupported | 1.056 ± 0.149 | 1.031 ± 0.121 | 5.839 ± 5.630 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.742 ± 0.101 | - | unsupported | 0.758 ± 0.100 | 0.953 ± 0.081 | 7.372 ± 3.245 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.324 ± 0.117 | - | 0.340 ± 0.034 | 0.685 ± 0.037 | 0.264 ± 0.033 | 1.070 ± 0.081 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.270 ± 0.012 | - | 47.998 ± 0.008 | 0.273 ± 0.013 | 0.173 ± 0.015 | 1.105 ± 0.199 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.754 ± 0.101 | - | 0.925 ± 0.172 | 0.985 ± 0.118 | 0.875 ± 0.024 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.468 ± 0.152 | - | 11.998 ± 0.012 | 0.482 ± 0.082 | 0.849 ± 0.019 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.814 ± 0.100 | - | 0.892 ± 0.097 | 0.975 ± 0.136 | 0.766 ± 0.049 | 1.046 ± 7.005 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.356 ± 0.183 | - | 11.997 ± 0.013 | 0.420 ± 0.050 | 0.694 ± 0.104 | 6.510 ± 6.302 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.609 ± 0.013 | - | 0.682 ± 0.057 | 0.312 ± 0.014 | 0.306 ± 0.014 | 0.538 ± 0.841 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.828 ± 0.025 | - | 36.046 ± 0.525 | 0.460 ± 0.022 | 0.310 ± 0.022 | 1.165 ± 0.028 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.424 ± 0.079 | - | 0.002 ± 0.000 | 1.512 ± 0.158 | 1.363 ± 0.335 | 8.712 ± 5.067 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.812 ± 0.069 | - | 0.002 ± 0.000 | 1.069 ± 0.268 | 1.031 ± 0.222 | 8.596 ± 4.707 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.042 ± 0.000 | - | 0.044 ± 0.026 | 0.058 ± 0.001 | 0.075 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.072 ± 0.001 | - | 0.075 ± 0.000 | 0.080 ± 0.001 | 0.124 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.189 ± 0.009 | - | 0.194 ± 0.004 | 0.234 ± 0.009 | 0.266 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.067 ± 0.001 | - | 0.069 ± 0.000 | 0.110 ± 0.002 | 0.101 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.032 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.001 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.031 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.546 ± 0.017 | - | 0.550 ± 0.015 | 0.939 ± 0.007 | 0.599 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.019 ± 0.000 | - | 0.022 ± 0.000 | 0.031 ± 0.000 | 0.055 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.042 ± 0.000 | - | 0.044 ± 0.001 | 0.070 ± 0.001 | 0.068 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.072 ± 0.001 | - | 0.074 ± 0.000 | 0.100 ± 0.000 | 0.117 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.190 ± 0.002 | - | 0.194 ± 0.001 | 0.246 ± 0.010 | 0.261 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.068 ± 0.001 | - | 0.069 ± 0.000 | 0.144 ± 0.003 | 0.102 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.014 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.543 ± 0.026 | - | 0.592 ± 3.780 | 1.119 ± 0.014 | 0.997 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.020 ± 0.000 | - | 0.022 ± 0.001 | 0.045 ± 0.000 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.412 ± 0.010 | - | 0.416 ± 0.007 | 0.537 ± 0.005 | 0.641 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.646 ± 0.009 | - | 1.650 ± 0.010 | 1.693 ± 0.005 | 2.080 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 4.994 ± 0.020 | - | 5.006 ± 0.013 | 5.217 ± 0.057 | 7.159 ± 0.083 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 2.993 ± 0.018 | - | 2.984 ± 0.012 | 2.671 ± 0.012 | 2.132 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.089 ± 0.002 | - | 0.090 ± 0.001 | 0.093 ± 0.003 | 0.116 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.012 ± 0.001 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.016 ± 0.000 | - | 0.018 ± 0.000 | 0.022 ± 0.001 | 0.046 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.021 ± 0.000 | - | 0.023 ± 0.001 | 0.026 ± 0.000 | 0.052 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 24.009 ± 0.209 | - | 23.950 ± 0.172 | 21.328 ± 0.111 | 17.124 ± 0.138 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.689 ± 0.013 | - | 0.694 ± 0.013 | 0.700 ± 0.015 | 0.757 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.014 ± 0.001 | - | 0.016 ± 0.000 | 0.019 ± 0.008 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.041 ± 0.000 | - | 0.043 ± 0.000 | 0.047 ± 0.005 | 0.081 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.415 ± 0.019 | - | 0.422 ± 0.030 | 0.788 ± 0.276 | 0.669 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 1.664 ± 0.032 | - | 1.666 ± 0.028 | 1.687 ± 0.014 | 1.980 ± 0.086 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 5.026 ± 0.013 | - | 5.062 ± 0.098 | 7.470 ± 0.078 | 5.000 ± 0.114 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 5.534 ± 0.049 | - | 5.647 ± 0.063 | 2.623 ± 0.918 | 2.995 ± 1.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.089 ± 0.001 | - | 0.092 ± 0.001 | 0.139 ± 0.009 | 0.132 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.001 | 0.012 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.001 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.017 ± 0.000 | - | 0.019 ± 0.000 | 0.021 ± 0.000 | 0.037 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.021 ± 0.000 | - | 0.024 ± 0.001 | 0.026 ± 0.000 | 0.045 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 45.052 ± 0.274 | - | 44.655 ± 0.040 | 15.540 ± 0.143 | 23.972 ± 7.822 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.014 ± 0.002 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.699 ± 0.018 | - | 0.711 ± 0.438 | 1.053 ± 0.055 | 1.315 ± 0.912 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.015 ± 0.000 | - | 0.017 ± 0.000 | 0.018 ± 0.000 | 0.027 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.041 ± 0.001 | - | 0.043 ± 0.000 | 0.046 ± 0.001 | 0.072 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.484 ± 0.015 | - | 0.490 ± 0.013 | 0.330 ± 0.008 | 0.295 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.730 ± 0.008 | - | 1.745 ± 0.015 | 1.475 ± 0.007 | 1.437 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 4.527 ± 0.014 | - | 4.529 ± 0.005 | 4.372 ± 0.035 | 5.018 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.096 ± 0.009 | - | 1.095 ± 0.012 | 1.105 ± 0.019 | 1.140 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.002 ± 0.001 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.057 ± 0.001 | - | 0.059 ± 0.001 | 0.067 ± 0.007 | 0.093 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.015 ± 0.000 | - | 0.017 ± 0.000 | 0.019 ± 0.002 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.023 ± 0.000 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 8.768 ± 0.153 | - | 8.754 ± 0.092 | 8.660 ± 0.019 | 8.972 ± 0.065 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.439 ± 0.007 | - | 0.441 ± 0.010 | 0.483 ± 0.011 | 0.579 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.015 ± 0.000 | - | 0.016 ± 0.000 | 0.016 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.037 ± 0.000 | - | 0.039 ± 0.000 | 0.040 ± 0.000 | 0.064 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.447 ± 0.024 | - | 0.454 ± 0.015 | 0.530 ± 0.012 | 0.282 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.692 ± 0.025 | - | 1.701 ± 0.020 | 1.457 ± 0.007 | 0.958 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 4.523 ± 0.004 | - | 4.534 ± 0.028 | 6.262 ± 0.014 | 2.964 ± 0.462 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 2.662 ± 0.031 | - | 2.672 ± 0.028 | 1.008 ± 0.007 | 1.340 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.007 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.124 ± 0.033 | - | 0.121 ± 0.016 | 0.098 ± 0.004 | 0.084 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.001 | 0.016 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.013 ± 0.000 | 0.015 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.017 ± 0.002 | - | 0.017 ± 0.000 | 0.018 ± 0.001 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.016 ± 0.002 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.020 ± 0.001 | - | 0.021 ± 0.000 | 0.022 ± 0.000 | 0.030 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 21.282 ± 0.033 | - | 21.213 ± 0.055 | 8.025 ± 0.230 | 10.772 ± 0.646 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.891 ± 0.011 | - | 0.908 ± 0.022 | 0.720 ± 0.011 | 0.375 ± 0.097 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.015 ± 0.000 | - | 0.017 ± 0.000 | 0.023 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.037 ± 0.000 | - | 0.039 ± 0.001 | 0.040 ± 0.000 | 0.055 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.180 ± 0.001 | - | 0.183 ± 0.004 | 0.174 ± 0.002 | 0.291 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 0.834 ± 0.009 | - | 0.828 ± 0.005 | 0.720 ± 0.033 | 1.438 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 2.409 ± 0.013 | - | 2.412 ± 0.005 | 2.281 ± 0.019 | 5.326 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.420 ± 0.009 | - | 0.430 ± 0.012 | 0.443 ± 0.007 | 1.158 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.025 ± 0.000 | - | 0.027 ± 0.000 | 0.031 ± 0.001 | 0.092 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.015 ± 0.001 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.011 ± 0.000 | - | 0.012 ± 0.000 | 0.016 ± 0.000 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 3.391 ± 0.057 | - | 3.386 ± 0.012 | 3.512 ± 0.041 | 8.859 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.186 ± 0.008 | - | 0.190 ± 0.008 | 0.196 ± 0.001 | 0.602 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.020 ± 0.000 | - | 0.022 ± 0.000 | 0.024 ± 0.000 | 0.065 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.181 ± 0.011 | - | 0.183 ± 0.014 | 0.283 ± 0.009 | 0.280 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 0.826 ± 0.018 | - | 0.842 ± 3.746 | 0.723 ± 0.011 | 0.950 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 2.426 ± 0.013 | - | 2.433 ± 0.016 | 2.768 ± 0.041 | 2.950 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 1.667 ± 0.011 | - | 1.698 ± 0.023 | 0.590 ± 0.008 | 1.329 ± 0.681 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.026 ± 0.000 | - | 0.027 ± 0.000 | 0.040 ± 0.003 | 0.084 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.013 ± 0.001 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.014 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.015 ± 0.000 | 0.028 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 13.371 ± 0.104 | - | 13.528 ± 0.053 | 4.551 ± 0.066 | 10.355 ± 0.716 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.188 ± 0.005 | - | 0.190 ± 0.008 | 0.246 ± 0.009 | 0.340 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.015 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.020 ± 0.000 | - | 0.022 ± 0.000 | 0.024 ± 0.001 | 0.054 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.145 ± 0.002 | - | 0.143 ± 0.002 | 0.146 ± 0.002 | 0.090 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.186 ± 0.005 | - | 0.181 ± 0.004 | 0.202 ± 0.007 | 0.178 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.406 ± 0.010 | - | 0.408 ± 0.011 | 0.417 ± 0.012 | 0.449 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.106 ± 0.001 | - | 0.111 ± 0.004 | 0.132 ± 0.002 | 0.154 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.016 ± 0.000 | 0.045 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.018 ± 0.001 | 0.035 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.840 ± 0.021 | - | 0.847 ± 0.010 | 1.020 ± 0.031 | 1.127 ± 0.072 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.057 ± 0.001 | - | 0.059 ± 0.001 | 0.061 ± 0.002 | 0.095 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.031 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.150 ± 0.002 | - | 0.145 ± 0.001 | 0.124 ± 0.001 | 0.099 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.191 ± 0.014 | - | 0.186 ± 0.009 | 0.166 ± 0.011 | 0.172 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.423 ± 0.022 | - | 0.411 ± 0.022 | 0.289 ± 0.014 | 0.490 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.123 ± 0.027 | - | 0.110 ± 0.002 | 0.167 ± 0.001 | 0.145 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.015 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.010 ± 0.000 | - | 0.012 ± 0.001 | 0.024 ± 0.001 | 0.023 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.015 ± 0.001 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.012 ± 0.000 | - | 0.014 ± 0.001 | 0.026 ± 0.002 | 0.028 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.015 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.019 ± 0.002 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.018 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.017 ± 0.001 | 0.011 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.017 ± 0.001 | 0.011 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.018 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.858 ± 0.019 | - | 0.866 ± 0.017 | 0.489 ± 0.021 | 0.756 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.018 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.059 ± 0.001 | - | 0.060 ± 0.001 | 0.052 ± 0.001 | 0.102 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.018 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.020 ± 0.000 | 0.029 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.213 ± 0.003 | - | 0.206 ± 0.005 | 0.171 ± 0.013 | 0.247 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.442 ± 0.017 | - | 0.446 ± 0.010 | 0.411 ± 0.010 | 0.608 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.037 ± 0.003 | - | 1.060 ± 0.012 | 1.040 ± 0.002 | 1.739 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.394 ± 0.014 | - | 0.391 ± 0.012 | 0.445 ± 0.011 | 0.521 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.014 ± 0.001 | - | 0.015 ± 0.000 | 0.019 ± 0.001 | 0.054 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.015 ± 0.000 | - | 0.017 ± 0.000 | 0.022 ± 0.000 | 0.054 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.002 ± 0.001 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 3.121 ± 0.013 | - | 3.837 ± 0.041 | 3.495 ± 0.017 | 3.863 ± 0.066 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.093 ± 0.001 | - | 0.095 ± 0.001 | 0.129 ± 0.001 | 0.175 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.010 ± 0.000 | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.212 ± 0.003 | - | 0.212 ± 0.012 | 0.245 ± 0.002 | 0.233 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.437 ± 0.019 | - | 0.441 ± 0.020 | 0.581 ± 0.012 | 0.599 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.029 ± 0.016 | - | 1.037 ± 0.009 | 1.530 ± 0.006 | 1.281 ± 0.080 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.824 ± 0.014 | - | 0.815 ± 0.016 | 0.416 ± 0.011 | 0.989 ± 0.074 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.001 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.014 ± 0.000 | - | 0.016 ± 0.001 | 0.032 ± 0.001 | 0.034 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.016 ± 0.000 | - | 0.018 ± 0.002 | 0.034 ± 0.001 | 0.032 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.018 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 6.624 ± 0.030 | - | 7.476 ± 0.031 | 3.142 ± 0.139 | 8.984 ± 0.650 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.094 ± 0.001 | - | 0.097 ± 0.002 | 0.211 ± 0.010 | 0.147 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.019 ± 0.000 | 0.026 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.137 ± 0.002 | - | 0.127 ± 0.005 | 0.075 ± 0.000 | 0.197 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.166 ± 0.006 | - | 0.131 ± 0.005 | 0.116 ± 0.001 | 0.284 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.425 ± 0.008 | - | 0.377 ± 0.010 | 0.336 ± 0.009 | 0.585 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.092 ± 0.002 | - | 0.099 ± 0.001 | 0.110 ± 0.001 | 0.152 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.008 ± 0.000 | - | 0.013 ± 0.001 | 0.018 ± 0.001 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.726 ± 0.011 | - | 0.766 ± 0.011 | 0.820 ± 0.029 | 0.862 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.053 ± 0.002 | - | 0.059 ± 0.001 | 0.056 ± 0.001 | 0.096 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.014 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.146 ± 0.016 | - | 0.128 ± 0.003 | 0.098 ± 0.001 | 0.183 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.172 ± 0.001 | - | 6.003 ± 0.009 | 0.134 ± 0.003 | 0.266 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.448 ± 0.010 | - | 23.999 ± 0.004 | 0.301 ± 0.009 | 0.647 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.091 ± 0.004 | - | 5.992 ± 0.014 | 0.148 ± 0.002 | 0.124 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.016 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.009 ± 0.001 | - | 0.015 ± 0.001 | 0.026 ± 0.002 | 0.023 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.002 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.018 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.018 ± 0.002 | 0.012 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.730 ± 0.032 | - | 24.016 ± 0.591 | 0.441 ± 0.013 | 0.581 ± 0.079 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.019 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.053 ± 0.000 | - | 0.061 ± 0.001 | 0.048 ± 0.000 | 0.102 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.019 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.011 ± 0.000 | 0.021 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.374 ± 0.009 | - | 1.377 ± 0.012 | 1.192 ± 0.011 | 1.612 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 4.541 ± 0.022 | - | 4.551 ± 0.045 | 3.984 ± 0.016 | 4.085 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 13.558 ± 0.062 | - | 13.562 ± 0.029 | 12.095 ± 0.036 | 11.943 ± 0.063 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 4.077 ± 0.015 | - | 4.066 ± 0.047 | 4.702 ± 0.037 | 3.490 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.158 ± 0.003 | - | 0.159 ± 0.004 | 0.182 ± 0.012 | 0.204 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.140 ± 0.010 | - | 0.144 ± 0.006 | 0.161 ± 0.012 | 0.190 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.017 ± 0.001 | - | 0.018 ± 0.001 | 0.021 ± 0.003 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.015 ± 0.000 | - | 0.017 ± 0.000 | 0.019 ± 0.001 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.043 ± 0.000 | - | 0.045 ± 0.000 | 0.044 ± 0.001 | 0.049 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.020 ± 0.000 | - | 0.022 ± 0.000 | 0.022 ± 0.001 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.057 ± 0.001 | - | 0.059 ± 0.001 | 0.056 ± 0.003 | 0.061 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 32.422 ± 0.119 | - | 33.720 ± 1.412 | 37.229 ± 0.253 | 27.609 ± 0.095 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.013 ± 0.000 | - | 0.014 ± 0.000 | 0.016 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.222 ± 0.013 | - | 1.222 ± 0.007 | 1.336 ± 0.016 | 1.250 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.038 ± 0.000 | - | 0.040 ± 0.002 | 0.040 ± 0.001 | 0.045 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.109 ± 0.001 | - | 0.112 ± 0.001 | 0.103 ± 0.004 | 0.139 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.357 ± 0.024 | - | 1.365 ± 0.011 | 1.150 ± 0.021 | 1.582 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 4.559 ± 0.021 | - | 4.556 ± 0.013 | 3.943 ± 0.020 | 2.007 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 13.558 ± 0.094 | - | 13.543 ± 0.030 | 14.546 ± 0.026 | 6.024 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 7.640 ± 0.015 | - | 7.603 ± 0.043 | 4.098 ± 0.250 | 4.724 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.241 ± 0.010 | - | 0.246 ± 0.025 | 0.277 ± 0.015 | 0.163 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.175 ± 0.004 | - | 0.182 ± 0.008 | 0.235 ± 0.012 | 0.155 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.017 ± 0.001 | - | 0.019 ± 0.000 | 0.020 ± 0.002 | 0.026 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.014 ± 0.002 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.016 ± 0.000 | - | 0.018 ± 0.000 | 0.018 ± 0.000 | 0.024 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.043 ± 0.001 | - | 0.046 ± 0.002 | 0.045 ± 0.001 | 0.049 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.015 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.020 ± 0.000 | - | 0.022 ± 0.000 | 0.022 ± 0.000 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.058 ± 0.000 | - | 0.060 ± 0.000 | 0.056 ± 0.001 | 0.061 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 61.358 ± 0.367 | - | 62.007 ± 2.652 | 31.644 ± 0.314 | 53.916 ± 43.050 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.022 ± 0.002 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 1.842 ± 0.023 | - | 1.851 ± 0.032 | 2.039 ± 0.010 | 0.605 ± 0.237 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.039 ± 0.001 | - | 0.040 ± 0.000 | 0.039 ± 0.001 | 0.044 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.110 ± 0.001 | - | 0.145 ± 0.020 | 0.103 ± 0.001 | 0.110 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 1.025 ± 0.007 | - | 1.029 ± 0.007 | 0.835 ± 0.001 | 0.995 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 3.867 ± 0.011 | - | 3.871 ± 0.010 | 3.352 ± 0.013 | 3.185 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 11.561 ± 0.015 | - | 11.606 ± 0.032 | 10.477 ± 0.130 | 9.830 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.894 ± 0.026 | - | 1.893 ± 0.015 | 1.987 ± 0.008 | 1.850 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.001 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.107 ± 0.003 | - | 0.109 ± 0.003 | 0.126 ± 0.005 | 0.138 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.001 | 0.016 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.086 ± 0.002 | - | 0.087 ± 0.001 | 0.098 ± 0.003 | 0.139 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.019 ± 0.003 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.018 ± 0.001 | 0.021 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.034 ± 0.000 | - | 0.036 ± 0.001 | 0.037 ± 0.001 | 0.040 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.017 ± 0.000 | - | 0.019 ± 0.000 | 0.020 ± 0.000 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.045 ± 0.000 | - | 0.048 ± 0.000 | 0.048 ± 0.002 | 0.048 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 15.106 ± 0.056 | - | 15.190 ± 0.067 | 15.986 ± 0.046 | 14.413 ± 0.089 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.013 ± 0.000 | 0.019 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 0.825 ± 0.007 | - | 0.831 ± 0.011 | 0.911 ± 0.005 | 0.900 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.032 ± 0.000 | - | 0.035 ± 0.000 | 0.033 ± 0.001 | 0.036 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.089 ± 0.000 | - | 0.091 ± 0.000 | 0.087 ± 0.001 | 0.100 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 1.007 ± 0.032 | - | 1.008 ± 0.018 | 0.822 ± 0.392 | 0.965 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 3.876 ± 0.020 | - | 3.897 ± 0.031 | 3.323 ± 0.012 | 1.543 ± 0.274 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 11.704 ± 0.072 | - | 11.623 ± 0.055 | 13.034 ± 0.007 | 4.917 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 2.882 ± 0.091 | - | 2.809 ± 0.018 | 2.396 ± 0.005 | 2.596 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.001 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.185 ± 0.023 | - | 0.186 ± 0.019 | 0.189 ± 0.016 | 0.149 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.118 ± 0.006 | - | 0.125 ± 0.011 | 0.147 ± 0.006 | 0.111 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.017 ± 0.002 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.014 ± 0.000 | - | 0.016 ± 0.001 | 0.017 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.034 ± 0.000 | - | 0.036 ± 0.000 | 0.037 ± 0.001 | 0.039 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.014 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.018 ± 0.000 | - | 0.020 ± 0.001 | 0.020 ± 0.001 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.046 ± 0.001 | - | 0.048 ± 0.000 | 0.046 ± 0.001 | 0.049 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 22.542 ± 0.081 | - | 22.614 ± 0.124 | 18.586 ± 0.488 | 19.925 ± 2.846 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.010 ± 0.000 | - | 0.012 ± 0.001 | 0.018 ± 0.003 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 1.430 ± 0.033 | - | 1.426 ± 0.017 | 1.400 ± 0.010 | 0.440 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.033 ± 0.000 | - | 0.035 ± 0.000 | 0.033 ± 0.001 | 0.036 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.089 ± 0.001 | - | 0.092 ± 0.001 | 0.086 ± 0.000 | 0.090 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.364 ± 0.010 | - | unsupported | 0.259 ± 0.013 | 0.535 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.665 ± 0.010 | - | unsupported | 1.424 ± 0.022 | 1.974 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 4.115 ± 0.004 | - | unsupported | 3.945 ± 0.022 | 5.131 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.817 ± 0.024 | - | unsupported | 0.923 ± 0.008 | 0.948 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.038 ± 0.002 | - | unsupported | 0.049 ± 0.003 | 0.055 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.031 ± 0.001 | - | unsupported | 0.040 ± 0.001 | 0.047 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.006 ± 0.000 | - | unsupported | 0.010 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.007 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.013 ± 0.000 | - | unsupported | 0.018 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.010 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.007 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.017 ± 0.000 | - | unsupported | 0.022 ± 0.001 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 6.413 ± 0.039 | - | unsupported | 7.289 ± 0.035 | 7.349 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.299 ± 0.007 | - | unsupported | 0.338 ± 0.015 | 0.375 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.014 ± 0.000 | - | unsupported | 0.017 ± 0.001 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.033 ± 0.000 | - | unsupported | 0.037 ± 0.001 | 0.049 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.372 ± 0.020 | - | unsupported | 0.416 ± 0.009 | 0.525 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 1.692 ± 0.027 | - | unsupported | 1.440 ± 0.009 | 1.027 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 4.172 ± 0.027 | - | unsupported | 5.669 ± 0.030 | 2.659 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 0.998 ± 0.019 | - | unsupported | 1.187 ± 0.016 | 1.386 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.039 ± 0.001 | - | unsupported | 0.065 ± 0.002 | 0.054 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.032 ± 0.001 | - | unsupported | 0.055 ± 0.005 | 0.059 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.006 ± 0.001 | - | unsupported | 0.010 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.010 ± 0.004 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.006 ± 0.000 | - | unsupported | 0.015 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.014 ± 0.000 | - | unsupported | 0.018 ± 0.000 | 0.022 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.008 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.018 ± 0.000 | - | unsupported | 0.022 ± 0.000 | 0.028 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 7.891 ± 0.046 | - | unsupported | 9.254 ± 0.081 | 11.734 ± 1.506 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.005 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.300 ± 0.006 | - | unsupported | 0.330 ± 0.016 | 0.241 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.014 ± 0.000 | - | unsupported | 0.025 ± 0.002 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.034 ± 0.000 | - | unsupported | 0.037 ± 0.001 | 0.048 ± 0.001 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.821 ± 0.006 | - | unsupported | 0.780 ± 0.006 | 3.126 ± 0.093 | 2.173 ± 0.020 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.105 ± 0.001 | - | unsupported | 0.046 ± 0.003 | 0.090 ± 0.002 | 0.115 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.140 ± 0.003 | - | unsupported | 0.070 ± 0.001 | 0.150 ± 0.007 | 0.183 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.288 ± 0.010 | - | unsupported | 0.202 ± 0.000 | 0.395 ± 0.019 | 0.662 ± 0.011 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.833 ± 0.030 | - | unsupported | 0.390 ± 0.015 | 0.834 ± 0.038 | 2.110 ± 0.986 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.116 ± 0.001 | - | unsupported | 0.047 ± 0.002 | 0.086 ± 0.003 | 0.088 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.140 ± 0.002 | - | unsupported | 0.058 ± 0.001 | 0.127 ± 0.002 | 0.155 ± 0.006 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.292 ± 0.009 | - | unsupported | 0.125 ± 0.008 | 0.414 ± 0.012 | 0.612 ± 0.082 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.247 ± 0.009 | - | unsupported | 0.303 ± 0.017 | 0.634 ± 0.019 | 0.298 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.028 ± 0.000 | - | unsupported | 0.033 ± 0.001 | 0.143 ± 0.005 | 0.061 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.051 ± 0.000 | - | unsupported | 0.050 ± 0.000 | 0.161 ± 0.004 | 0.090 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.163 ± 0.001 | - | unsupported | 0.138 ± 0.001 | 0.278 ± 0.010 | 0.152 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.248 ± 0.009 | - | unsupported | 0.431 ± 0.008 | 0.734 ± 0.083 | 0.298 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.029 ± 0.000 | - | unsupported | 0.056 ± 0.002 | 0.139 ± 0.007 | 0.062 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.051 ± 0.001 | - | unsupported | 0.076 ± 0.002 | 0.170 ± 0.006 | 0.092 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.164 ± 0.009 | - | unsupported | 0.183 ± 0.012 | 0.276 ± 0.015 | 0.154 ± 0.017 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.164 ± 0.016 | - | 0.160 ± 0.003 | 0.161 ± 0.006 | 0.295 ± 0.016 | 0.172 ± 0.008 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.062 ± 0.002 | - | 0.064 ± 0.001 | 0.056 ± 0.001 | 0.056 ± 0.006 | 0.046 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.073 ± 0.001 | - | 0.076 ± 0.001 | 0.068 ± 0.001 | 0.068 ± 0.001 | 0.062 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.099 ± 0.001 | - | 0.102 ± 0.001 | 0.097 ± 0.001 | 0.128 ± 0.011 | 0.101 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.158 ± 0.001 | - | 0.160 ± 0.004 | 0.209 ± 0.002 | 0.330 ± 0.019 | 0.171 ± 0.004 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.071 ± 0.000 | - | 0.064 ± 0.001 | 0.096 ± 0.013 | 0.059 ± 0.003 | 0.047 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.076 ± 0.001 | - | 0.077 ± 0.000 | 0.109 ± 0.001 | 0.071 ± 0.002 | 0.064 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.099 ± 0.002 | - | 0.103 ± 0.002 | 0.141 ± 0.002 | 0.154 ± 0.011 | 0.102 ± 0.003 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 37.921 ± 0.534 | - | 42.830 ± 10.928 | 56.250 ± 0.424 | 53.217 ± 0.660 | 51.129 ± 7.808 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 39.694 ± 0.285 | - | 39.689 ± 0.663 | 20.056 ± 0.180 | 48.899 ± 1.640 | 43.730 ± 14.063 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 22.249 ± 1.207 | - | 20.752 ± 0.659 | 23.823 ± 0.270 | 21.913 ± 0.206 | 18.324 ± 0.305 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 240.037 ± 11.958 | - | 313.002 ± 12.024 | 9.571 ± 0.256 | 12.982 ± 0.415 | 11.195 ± 3.394 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 6.137 ± 0.044 | - | 6.197 ± 0.180 | 6.579 ± 0.042 | 6.092 ± 0.063 | 5.947 ± 0.206 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 13.904 ± 0.210 | - | 13.832 ± 0.264 | 9.118 ± 0.121 | 8.898 ± 0.110 | 8.594 ± 0.259 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 7.077 ± 0.012 | - | 7.119 ± 0.005 | 7.214 ± 0.056 | 7.314 ± 0.034 | 6.989 ± 0.031 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 10.542 ± 0.042 | - | 10.600 ± 0.035 | 9.769 ± 0.093 | 11.008 ± 0.063 | 10.383 ± 0.030 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 17.371 ± 0.185 | - | 17.354 ± 0.077 | 15.724 ± 0.161 | 32.163 ± 0.054 | 11.029 ± 0.411 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 31.207 ± 0.103 | - | 31.218 ± 0.262 | 12.812 ± 0.427 | 27.241 ± 0.820 | 9.253 ± 0.161 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.413 ± 0.004 | 0.425 ± 0.001 | unsupported | 0.494 ± 0.004 | 0.610 ± 0.011 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 6.469 ± 0.038 | 6.546 ± 0.019 | unsupported | 7.281 ± 0.117 | 7.822 ± 0.051 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.646 ± 0.014 | 29.123 ± 0.437 | unsupported | 0.416 ± 0.004 | 1.401 ± 0.880 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 2270.243 ± 5.078 | 2299.821 ± 1.696 | unsupported | 4.625 ± 0.016 | 8.901 ± 0.176 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 35.316 ± 0.669 | - | 35.119 ± 0.329 | 37.646 ± 0.661 | 36.198 ± 0.481 | 24.299 ± 1.923 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 216.876 ± 9.322 | - | 268.280 ± 9.201 | 19.405 ± 0.859 | 20.300 ± 0.181 | 14.808 ± 0.823 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 14.759 ± 0.214 | 14.334 ± 0.724 | 7.775 ± 0.121 | 37.206 ± 0.229 | 6.395 ± 0.051 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 2292.054 ± 8.435 | 2278.168 ± 6.943 | 5.186 ± 0.065 | 38.978 ± 7.899 | 4.835 ± 0.117 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 22.107 ± 0.146 | - | 22.102 ± 0.303 | 29.220 ± 0.516 | 28.052 ± 0.701 | 28.874 ± 14.056 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 245.055 ± 15.133 | - | 245.875 ± 10.159 | 14.007 ± 0.381 | 19.813 ± 0.787 | 21.156 ± 18.521 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.729 ± 0.016 | - | 0.722 ± 0.069 | 0.959 ± 0.021 | 19.624 ± 0.653 | 2.314 ± 0.012 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.393 ± 0.071 | - | 23.995 ± 0.038 | 0.944 ± 0.025 | 11.699 ± 2.751 | 2.307 ± 0.014 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 16.829 ± 0.398 | - | 16.637 ± 0.069 | 17.621 ± 0.298 | 16.497 ± 0.127 | 17.830 ± 0.338 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 201.962 ± 22.507 | - | 221.933 ± 27.894 | 14.631 ± 0.180 | 18.961 ± 0.354 | 16.928 ± 2.156 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 17.056 ± 0.126 | - | 16.852 ± 0.204 | 17.635 ± 0.142 | 16.704 ± 0.082 | 16.051 ± 1.725 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 207.100 ± 13.634 | - | 215.668 ± 32.897 | 14.628 ± 0.165 | 18.706 ± 0.358 | 16.941 ± 0.260 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 0.978 ± 0.011 | 0.990 ± 0.006 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 17.823 ± 0.077 | 17.766 ± 0.114 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.319 ± 0.007 | 1.336 ± 0.012 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 18.858 ± 0.168 | 18.857 ± 0.373 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 0.913 ± 0.004 | 0.926 ± 0.009 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 17.022 ± 0.087 | 17.076 ± 0.080 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.255 ± 0.006 | 1.268 ± 0.007 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 18.096 ± 0.090 | 18.078 ± 0.278 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 21.646 ± 0.680 | - | 20.932 ± 0.285 | 23.807 ± 0.147 | 21.686 ± 0.248 | 18.305 ± 1.659 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 240.154 ± 13.952 | - | 325.004 ± 25.960 | 9.771 ± 0.087 | 13.087 ± 0.351 | 10.906 ± 3.446 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 49.138 ± 5.815 | 48.773 ± 0.227 | 52.243 ± 0.950 | 51.031 ± 0.700 | 51.786 ± 2.005 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 2316.601 ± 6.695 | 2313.236 ± 10.657 | 33.553 ± 0.155 | 44.146 ± 0.701 | 42.595 ± 7.853 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 30.894 ± 0.560 | - | 32.945 ± 0.571 | 26.114 ± 0.271 | 133.715 ± 2.935 | 25.809 ± 0.289 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 52.731 ± 1.612 | - | 52.603 ± 2.681 | 21.457 ± 5.920 | 127.917 ± 9.737 | 18.591 ± 0.218 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.079 ± 0.000 | 0.079 ± 0.001 | unsupported | 0.135 ± 0.001 | 0.130 ± 0.017 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 1.260 ± 0.005 | 1.252 ± 0.004 | unsupported | 1.351 ± 0.006 | 0.338 ± 0.576 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.040 ± 0.007 | 23.984 ± 0.328 | unsupported | 0.066 ± 0.006 | 0.119 ± 0.009 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.570 ± 0.019 | 23.999 ± 0.375 | unsupported | 0.646 ± 0.002 | 0.795 ± 0.255 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.038 ± 0.000 | 0.038 ± 0.000 | unsupported | 0.099 ± 0.009 | 0.176 ± 0.009 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 0.591 ± 0.001 | 0.615 ± 0.023 | unsupported | 1.376 ± 0.025 | 2.039 ± 0.118 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.020 ± 0.001 | 12.000 ± 0.031 | unsupported | 0.065 ± 0.005 | 0.119 ± 0.010 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.282 ± 0.001 | 11.999 ± 0.000 | unsupported | 0.496 ± 0.017 | 0.882 ± 0.119 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.106 ± 0.000 | 0.106 ± 0.000 | unsupported | 0.164 ± 0.003 | 0.145 ± 0.003 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 1.673 ± 0.003 | 1.682 ± 0.005 | unsupported | 2.619 ± 0.058 | 1.552 ± 0.038 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.050 ± 0.001 | 12.000 ± 0.156 | unsupported | 0.072 ± 0.001 | 0.107 ± 0.009 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 0.762 ± 0.001 | 11.998 ± 0.005 | unsupported | 1.013 ± 0.022 | 1.050 ± 0.033 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.048 ± 0.008 | 0.070 ± 0.005 | unsupported | 0.135 ± 0.001 | 0.168 ± 0.005 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.630 ± 0.044 | 0.633 ± 0.020 | unsupported | 0.688 ± 0.007 | 1.619 ± 0.019 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.073 ± 0.001 | 64.762 ± 4.772 | unsupported | 0.054 ± 0.001 | 0.146 ± 0.007 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.366 ± 0.011 | 95.999 ± 0.125 | unsupported | 0.296 ± 0.002 | 1.008 ± 0.062 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.072 ± 0.006 | 0.082 ± 0.002 | unsupported | 0.102 ± 0.001 | 0.213 ± 0.009 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 0.895 ± 0.010 | 0.967 ± 0.063 | unsupported | 0.738 ± 0.023 | 2.408 ± 0.017 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.096 ± 0.001 | 89.048 ± 0.551 | unsupported | 0.045 ± 0.001 | 0.195 ± 0.006 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.497 ± 0.012 | 119.875 ± 0.125 | unsupported | 0.329 ± 0.058 | 1.437 ± 0.068 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.225 ± 0.001 | 0.247 ± 0.005 | unsupported | unsupported | 0.219 ± 0.020 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 3.301 ± 0.013 | 3.354 ± 0.036 | unsupported | unsupported | 3.247 ± 0.071 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.172 ± 0.005 | 112.676 ± 3.489 | unsupported | unsupported | 0.226 ± 0.002 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 1.375 ± 0.006 | 143.873 ± 0.501 | unsupported | unsupported | 1.837 ± 0.074 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.008 ± 0.000 | 0.014 ± 0.000 | unsupported | 0.010 ± 0.000 | 0.035 ± 0.000 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.093 ± 0.001 | 0.098 ± 0.001 | unsupported | 0.060 ± 0.002 | 0.268 ± 0.002 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.011 ± 0.001 | 23.984 ± 0.219 | unsupported | 0.011 ± 0.000 | 0.067 ± 0.045 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.051 ± 0.003 | 23.999 ± 0.379 | unsupported | 0.038 ± 0.001 | 0.158 ± 0.005 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.026 ± 0.005 | 0.038 ± 0.005 | unsupported | 0.097 ± 0.001 | 0.153 ± 0.003 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.335 ± 0.009 | 0.368 ± 0.177 | unsupported | 0.445 ± 0.006 | 1.515 ± 0.021 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.040 ± 0.003 | 32.687 ± 0.368 | unsupported | 0.092 ± 0.006 | 0.149 ± 0.111 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.198 ± 0.017 | 47.999 ± 0.127 | unsupported | 0.250 ± 0.008 | 0.861 ± 0.062 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.026 ± 0.001 | 0.027 ± 0.000 | unsupported | 0.080 ± 0.003 | 0.147 ± 0.006 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.403 ± 0.002 | 0.425 ± 0.003 | unsupported | 1.289 ± 0.063 | 1.385 ± 0.035 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.014 ± 0.000 | 12.000 ± 0.359 | unsupported | 0.062 ± 0.005 | 0.104 ± 0.006 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.194 ± 0.010 | 11.999 ± 0.002 | unsupported | 0.472 ± 0.014 | 0.712 ± 0.022 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.027 ± 0.000 | 0.027 ± 0.000 | unsupported | 0.080 ± 0.001 | 0.134 ± 0.003 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.417 ± 0.001 | 0.449 ± 0.005 | unsupported | 1.279 ± 0.090 | 1.395 ± 0.035 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.015 ± 0.000 | 11.984 ± 9.235 | unsupported | 0.063 ± 0.001 | 0.107 ± 0.002 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.200 ± 0.001 | 11.999 ± 0.002 | unsupported | 0.473 ± 0.006 | 0.780 ± 0.060 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.065 ± 0.001 | 0.082 ± 0.001 | unsupported | 0.104 ± 0.001 | 0.200 ± 0.005 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 0.932 ± 0.008 | 0.923 ± 0.053 | unsupported | 0.812 ± 0.005 | 2.366 ± 0.045 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.089 ± 0.003 | 88.948 ± 0.719 | unsupported | 0.044 ± 0.002 | 0.190 ± 0.003 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 0.534 ± 0.009 | 120.000 ± 1.623 | unsupported | 0.290 ± 0.006 | 1.422 ± 0.094 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.092 ± 0.001 | 0.092 ± 0.000 | unsupported | 0.152 ± 0.002 | 0.257 ± 0.004 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 1.456 ± 0.002 | 1.469 ± 0.004 | unsupported | 2.408 ± 0.073 | 3.427 ± 0.042 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.040 ± 0.000 | 12.000 ± 0.000 | unsupported | 0.118 ± 0.002 | 0.168 ± 0.003 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 0.609 ± 0.003 | 11.999 ± 0.001 | unsupported | 0.914 ± 0.028 | 1.749 ± 0.096 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.522 ± 0.001 | 1.101 ± 0.006 | unsupported | 0.231 ± 0.004 | 0.150 ± 0.016 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 11.108 ± 1.248 | 20.184 ± 0.562 | unsupported | 4.814 ± 0.148 | 1.444 ± 0.183 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.388 ± 0.015 | 104.187 ± 0.094 | unsupported | 0.239 ± 0.004 | 0.102 ± 0.004 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 7.195 ± 0.271 | 143.249 ± 2.125 | unsupported | 2.045 ± 0.062 | 0.864 ± 0.223 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 29.814 ± 2.603 | - | unsupported | 23.231 ± 1.230 | - | 27.410 ± 1.294 | 71.365 ± 1.950 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 21.011 ± 0.513 | - | unsupported | 20.937 ± 0.424 | - | 26.907 ± 1.936 | 26.777 ± 0.582 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 18.874 ± 0.534 | - | unsupported | 16.813 ± 0.147 | - | 20.435 ± 0.341 | 16.863 ± 0.374 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 13.979 ± 0.119 | - | unsupported | 14.489 ± 0.256 | - | 15.949 ± 0.280 | 14.593 ± 9.292 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 18.990 ± 0.200 | - | unsupported | 17.516 ± 0.336 | - | 19.464 ± 0.352 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 14.473 ± 0.131 | - | unsupported | 14.677 ± 1.081 | - | 18.607 ± 0.232 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 22.503 ± 0.620 | - | unsupported | 25.854 ± 0.334 | - | 24.416 ± 0.552 | 74.782 ± 1.554 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 20.863 ± 0.178 | - | unsupported | 21.104 ± 0.397 | - | 24.403 ± 0.575 | 26.932 ± 0.423 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 43.949 ± 0.536 | - | unsupported | 46.020 ± 0.436 | - | 40.316 ± 0.102 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 47.706 ± 0.711 | - | unsupported | 22.351 ± 0.356 | - | 19.500 ± 0.099 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 43.355 ± 0.450 | - | unsupported | 45.610 ± 0.231 | - | 40.115 ± 0.072 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 46.621 ± 0.332 | - | unsupported | 22.426 ± 0.301 | - | 19.320 ± 0.147 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 24.528 ± 1.536 | - | unsupported | 23.901 ± 1.272 | - | 29.639 ± 1.370 | 71.250 ± 1.399 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 20.979 ± 0.336 | - | unsupported | 21.034 ± 0.606 | - | 23.266 ± 0.922 | 26.884 ± 0.259 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 18.048 ± 0.501 | - | unsupported | 17.756 ± 0.153 | - | 20.033 ± 0.386 | 57.316 ± 1.004 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 14.264 ± 0.112 | - | unsupported | 14.605 ± 0.264 | - | 20.961 ± 12.709 | 20.426 ± 0.310 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.603 ± 0.007 | unsupported | unsupported | 0.771 ± 0.008 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 9.471 ± 0.041 | unsupported | unsupported | 11.569 ± 0.107 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 1.264 ± 0.004 | unsupported | unsupported | 0.518 ± 0.003 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 181.623 ± 6.741 | unsupported | unsupported | 5.716 ± 0.179 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 25.183 ± 0.763 | - | unsupported | 22.755 ± 1.122 | - | 29.065 ± 0.927 | 71.424 ± 0.458 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 21.237 ± 0.219 | - | unsupported | 21.087 ± 0.331 | - | 23.147 ± 0.480 | 27.129 ± 0.557 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 87.071 ± 6.253 | - | 0.002 ± 0.000 | 88.500 ± 1.280 | 85.485 ± 2.936 | 110.952 ± 10.913 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 38.023 ± 0.270 | - | 0.001 ± 0.000 | 29.132 ± 1.436 | 50.807 ± 5.563 | 110.350 ± 5.394 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 59.285 ± 6.745 | - | 177.813 ± 4.429 | 61.130 ± 1.183 | 65.859 ± 4.477 | 69.914 ± 1.503 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 20.858 ± 0.160 | - | 227.577 ± 21.142 | 20.806 ± 0.540 | 35.665 ± 4.662 | 70.369 ± 1.788 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 178.464 ± 9.424 | - | 179.675 ± 10.954 | 174.111 ± 2.758 | 174.037 ± 2.152 | 202.549 ± 8.504 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 57.896 ± 0.550 | - | 178.939 ± 1.209 | 57.669 ± 1.339 | 90.267 ± 8.593 | 205.349 ± 8.683 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 53.681 ± 4.089 | - | 57.597 ± 3.825 | 58.756 ± 6.085 | 62.111 ± 3.945 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 36.097 ± 0.392 | - | 59.265 ± 1.644 | 22.086 ± 0.497 | 35.921 ± 2.120 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 114.889 ± 8.122 | - | 0.002 ± 0.000 | 100.812 ± 1.936 | 102.360 ± 4.305 | 129.572 ± 10.215 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 37.124 ± 0.592 | - | 0.002 ± 0.000 | 36.107 ± 1.835 | 41.441 ± 3.470 | 134.218 ± 8.950 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 81.433 ± 0.783 | - | 0.001 ± 0.000 | 71.467 ± 0.462 | 67.617 ± 0.547 | 176.900 ± 9.904 | 114.846 ± 2.671 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 33.295 ± 5.719 | - | 0.001 ± 0.000 | 72.396 ± 1.014 | 34.394 ± 1.092 | 176.644 ± 5.444 | 43.794 ± 0.873 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 50.497 ± 1.121 | - | 50.164 ± 2.336 | 57.437 ± 2.263 | 49.290 ± 3.956 | 68.389 ± 6.279 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 20.146 ± 0.199 | - | 69.736 ± 9.878 | 19.640 ± 0.391 | 29.201 ± 3.410 | 70.314 ± 4.757 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 49.727 ± 0.694 | - | 49.359 ± 2.318 | 52.706 ± 3.203 | 49.108 ± 3.158 | 68.478 ± 3.578 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 18.828 ± 0.229 | - | 66.782 ± 10.438 | 18.298 ± 0.296 | 27.157 ± 0.382 | 70.372 ± 7.364 | - |
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
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 6.366213 | 397888.31 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 4.835720 | 302232.50 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 5.331593 | 333224.56 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 3.040421 | 190026.31 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 3.351186 | 209449.12 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 1803.850228 | 112740639.25 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 11.056798 | 691049.88 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 10.991552 | 686972.00 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 4.482069 | 280129.31 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 383.985465 | 23999091.56 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 9.766571 | 610410.69 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 7.910266 | 494391.62 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 6.603525 | 412720.31 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 6.801276 | 425079.75 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 125.150769 | 7821923.06 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 116.498340 | 7281146.25 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 103.510257 | 6469391.06 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 104.733036 | 6545814.75 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 22.412546 | 1400784.12 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 6.653388 | 415836.75 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 10.339253 | 646203.31 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 465.975534 | 29123470.88 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 142.423786 | 8901486.62 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 73.996384 | 4624774.00 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 36323.891921 | 2270243245.06 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 36797.131239 | 2299820702.44 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 15.648008 | 978000.50 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 15.841334 | 990083.38 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 285.161439 | 17822589.94 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 284.257552 | 17766097.00 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 21.102063 | 1318878.94 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 21.381856 | 1336366.00 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 301.735049 | 18858440.56 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 301.714188 | 18857136.75 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 14.600469 | 912529.31 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 14.820474 | 926279.62 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 272.358243 | 17022390.19 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 273.211375 | 17075710.94 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 20.085019 | 1255313.69 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 20.288609 | 1268038.06 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 289.538294 | 18096143.38 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 289.241827 | 18077614.19 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.327694 | 130120.22 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.628999 | 134828.11 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.084398 | 79443.72 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.037779 | 78715.30 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 2.702156 | 337769.50 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.806569 | 1350821.12 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.078242 | 1259780.25 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.019686 | 1252460.75 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.610475 | 118913.67 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.245037 | 66328.70 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.535519 | 39617.48 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1534.977361 | 23984021.27 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.363381 | 795422.62 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.167942 | 645992.75 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.561084 | 570135.50 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 191.991174 | 23998896.75 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.283255 | 176300.86 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.321688 | 98776.38 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.401691 | 37526.42 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.439918 | 38123.72 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 16.312691 | 2039086.38 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.006996 | 1375874.50 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.729393 | 591174.12 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.919534 | 614941.75 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.644951 | 119452.36 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.174082 | 65220.03 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.263098 | 19735.91 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.980572 | 11999696.44 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.054102 | 881762.75 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.966227 | 495778.38 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.255444 | 281930.50 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.992638 | 11999079.75 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.258921 | 144670.64 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 10.502159 | 164096.23 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.753696 | 105526.50 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.775799 | 105871.86 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.412353 | 1551544.12 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 20.948715 | 2618589.38 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.384681 | 1673085.12 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 13.459452 | 1682431.50 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.877561 | 107461.89 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.576357 | 71505.58 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.182366 | 49724.47 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.977672 | 11999651.12 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.396236 | 1049529.50 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.102827 | 1012853.38 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.096354 | 762044.25 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.987210 | 11998401.25 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.742184 | 167846.62 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.638352 | 134974.25 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.071843 | 47997.55 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.471181 | 69862.20 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.950680 | 1618835.00 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 5.505000 | 688125.00 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.038180 | 629772.50 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.061083 | 632635.38 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.342366 | 145974.47 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.487767 | 54496.36 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.683286 | 73176.34 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4144.778740 | 64762167.81 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.064176 | 1008022.00 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.364468 | 295558.50 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.924334 | 365541.75 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 767.991557 | 95998944.62 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.651175 | 213299.61 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.559909 | 102498.58 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.582996 | 71609.31 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.239797 | 81871.83 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 19.267541 | 2408442.62 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 5.906333 | 738291.62 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.158845 | 894855.62 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.736540 | 967067.50 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.455039 | 194609.98 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.894700 | 45229.69 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.116432 | 95569.25 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5699.098756 | 89048418.06 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.495490 | 1436936.25 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.630211 | 328776.38 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.972614 | 496576.75 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 958.998631 | 119874828.88 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.987928 | 218561.38 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 14.401225 | 225019.14 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 15.807563 | 246993.17 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 25.977262 | 3247157.75 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 26.405975 | 3300746.88 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 26.835019 | 3354377.38 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 14.449129 | 225767.64 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 11.036109 | 172439.20 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7211.233626 | 112675525.41 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 14.696885 | 1837110.62 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 11.002258 | 1375282.25 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1150.982725 | 143872840.62 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.270367 | 35474.48 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.620798 | 9699.97 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.519979 | 8124.67 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 0.876446 | 13694.47 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 2.147716 | 268464.50 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.483735 | 60466.88 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.740575 | 92571.88 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.787765 | 98470.62 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.297565 | 67149.45 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.679374 | 10615.22 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.680426 | 10631.66 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1534.979851 | 23984060.17 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.262667 | 157833.38 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.307175 | 38396.88 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.404368 | 50546.00 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 191.991914 | 23998989.25 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.812750 | 153324.22 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.237013 | 97453.33 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.665723 | 26026.92 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.412056 | 37688.38 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.118729 | 1514841.12 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 3.557080 | 444635.00 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.678210 | 334776.25 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 2.947168 | 368396.00 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.519693 | 148745.20 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.878322 | 91848.78 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.541909 | 39717.33 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2091.970754 | 32687043.03 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.884231 | 860528.88 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.000579 | 250072.38 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.580346 | 197543.25 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 383.990255 | 47998781.88 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.397766 | 146840.09 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.100772 | 79699.56 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.652743 | 25824.11 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.700995 | 26578.05 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.082397 | 1385299.62 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.315904 | 1289488.00 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.226692 | 403336.50 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.402661 | 425332.62 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.667330 | 104177.03 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.965203 | 61956.30 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.903607 | 14118.86 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 768.000780 | 12000012.19 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.692548 | 711568.50 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.773663 | 471707.88 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.553948 | 194243.50 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.992708 | 11999088.50 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.584754 | 134136.78 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.145518 | 80398.72 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.719593 | 26868.64 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.732101 | 27064.08 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.157458 | 1394682.25 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.234193 | 1279274.12 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.332948 | 416618.50 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.591961 | 448995.12 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.876129 | 107439.52 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.042017 | 63156.52 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.930026 | 14531.66 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 766.968519 | 11983883.11 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.239217 | 779902.12 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.784789 | 473098.62 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.598654 | 199831.75 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.991145 | 11998893.12 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.775071 | 199610.48 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.658665 | 104041.64 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.148174 | 64815.22 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.265215 | 82268.98 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.927948 | 2365993.50 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 6.495945 | 811993.12 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.454413 | 931801.62 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.383497 | 922937.12 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.172622 | 190197.22 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.839228 | 44362.94 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.664657 | 88510.27 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5692.685978 | 88948218.41 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.377606 | 1422200.75 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.317419 | 289677.38 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.268802 | 533600.25 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 959.998427 | 119999803.38 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 16.418758 | 256543.09 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 9.721184 | 151893.50 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.868217 | 91690.89 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.892833 | 92075.52 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 27.417682 | 3427210.25 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 19.263464 | 2407933.00 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 11.644087 | 1455510.88 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.753128 | 1469141.00 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.774955 | 168358.67 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 7.582892 | 118482.69 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.570030 | 40156.72 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.978078 | 11999657.47 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 13.994996 | 1749374.50 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.312118 | 914014.75 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.868629 | 608578.62 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.992407 | 11999050.88 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.619579 | 150305.92 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 14.767065 | 230735.39 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 33.417242 | 522144.41 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 70.487756 | 1101371.19 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.550630 | 1443828.75 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 38.510874 | 4813859.25 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 88.864368 | 11108046.00 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 161.469273 | 20183659.12 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.525511 | 101961.11 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 15.295827 | 238997.30 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 24.845841 | 388216.27 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6667.967920 | 104186998.75 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.911081 | 863885.12 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 16.359794 | 2044974.25 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 57.558178 | 7194772.25 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1145.993811 | 143249226.38 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 12.332215 | 770763.44 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 9.654405 | 603400.31 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 185.111274 | 11569454.62 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 151.536991 | 9471061.94 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 8.282467 | 517654.19 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 20.227330 | 1264208.12 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 91.462049 | 5716378.06 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 2905.973401 | 181623337.56 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.017626 | 1101.62 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002063 | 128.94 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.017686 | 1105.38 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002033 | 127.06 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000120 | 7.50 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.016614 | 1038.38 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.007050 | 440.62 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000300 | 18.75 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.016415 | 1025.94 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.007491 | 468.19 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000120 | 7.50 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.017066 | 1066.62 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.005038 | 314.88 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000481 | 30.06 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.017016 | 1063.50 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.005528 | 345.50 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000090 | 5.62 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.015904 | 994.00 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003184 | 199.00 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000110 | 6.88 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.015413 | 963.31 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003335 | 208.44 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/norm_fro` (c64, threads=1, shape=`2048x1536`): `tenferro-direct` is 14.0x faster than the slowest successful cell (`pytorch-cpu`, 18.273 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `jax-cpu` is 20.7x faster than the slowest successful cell (`tenferro-trace`, 60.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `julia-base` is 17.2x faster than the slowest successful cell (`tenferro-trace`, 60.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 64.2x faster than the slowest successful cell (`tenferro-trace`, 60.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 18.6x faster than the slowest successful cell (`tenferro-direct`, 129.981 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `julia-base` is 22.8x faster than the slowest successful cell (`tenferro-direct`, 129.981 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 26.5x faster than the slowest successful cell (`tenferro-direct`, 129.981 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 36.6x faster than the slowest successful cell (`tenferro-trace`, 121.007 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `julia-base` is 40.9x faster than the slowest successful cell (`tenferro-trace`, 121.007 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 41.4x faster than the slowest successful cell (`tenferro-trace`, 121.007 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/chain_log1p_exp_mul` (f64, threads=4, shape=`4194304`): `tenferro-direct` is 13.5x faster than the slowest successful cell (`tenferro-trace`, 107.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/exp` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 15.1x faster than the slowest successful cell (`julia-base`, 58.311 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log1p` (f64, threads=4, shape=`4194304`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`julia-base`, 35.290 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/masked_log_softmax_axis1` (f64, threads=4, shape=`1024x64`): `jax-cpu` is 593.3x faster than the slowest successful cell (`tenferro-eager`, 112.741 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/masked_log_softmax_axis1` (f64, threads=4, shape=`1024x64`): `tenferro-direct` is 538.3x faster than the slowest successful cell (`tenferro-eager`, 112.741 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 22.276 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 23.8x faster than the slowest successful cell (`pytorch-cpu`, 22.276 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 22.276 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 14.8x faster than the slowest successful cell (`pytorch-cpu`, 22.276 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 16.9x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 26.0x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 34.4x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 13.3x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 17.7x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 10.2x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 22.9x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 22.389 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 23.5x faster than the slowest successful cell (`pytorch-cpu`, 22.389 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 15.5x faster than the slowest successful cell (`pytorch-cpu`, 22.389 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 15.5x faster than the slowest successful cell (`pytorch-cpu`, 22.389 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 20.7x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 26.2x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 34.6x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 26.0x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 39.0x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 15.6x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 63.6x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 24.8x faster than the slowest successful cell (`tenferro-trace`, 24.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 28.0x faster than the slowest successful cell (`tenferro-trace`, 24.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 69.5x faster than the slowest successful cell (`tenferro-trace`, 24.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 33.0x faster than the slowest successful cell (`tenferro-trace`, 24.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 25.4x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 39.9x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 18.6x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 66.9x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 36.3x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 28.7x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 80.5x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 33.3x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_squares_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 85.7x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sin` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 11.2x faster than the slowest successful cell (`julia-base`, 56.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `jax-cpu` is 16.2x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `julia-base` is 11.6x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `pytorch-cpu` is 41.7x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `tenferro-direct` is 32.4x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `jax-cpu` is 19.7x faster than the slowest successful cell (`tenferro-trace`, 11.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `pytorch-cpu` is 37.5x faster than the slowest successful cell (`tenferro-trace`, 11.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `tenferro-direct` is 32.2x faster than the slowest successful cell (`tenferro-trace`, 11.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `jax-cpu` is 277.9x faster than the slowest successful cell (`tenferro-trace`, 47.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `julia-base` is 43.4x faster than the slowest successful cell (`tenferro-trace`, 47.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 175.7x faster than the slowest successful cell (`tenferro-trace`, 47.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `tenferro-direct` is 177.7x faster than the slowest successful cell (`tenferro-trace`, 47.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `jax-cpu` is 14.1x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 24.9x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 25.6x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `jax-cpu` is 17.3x faster than the slowest successful cell (`tenferro-trace`, 11.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 28.6x faster than the slowest successful cell (`tenferro-trace`, 11.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 33.7x faster than the slowest successful cell (`tenferro-trace`, 11.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `jax-cpu` is 116.1x faster than the slowest successful cell (`tenferro-trace`, 36.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `julia-base` is 30.9x faster than the slowest successful cell (`tenferro-trace`, 36.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 78.3x faster than the slowest successful cell (`tenferro-trace`, 36.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `tenferro-direct` is 43.5x faster than the slowest successful cell (`tenferro-trace`, 36.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 5541.8x faster than the slowest successful cell (`julia-base`, 8.712 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-direct` is 10.6x faster than the slowest successful cell (`julia-base`, 8.596 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 5232.0x faster than the slowest successful cell (`julia-base`, 8.596 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 20.4x faster than the slowest successful cell (`jax-cpu`, 0.031 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x4x4,rhs=1`): `jax-cpu` is 22.5x faster than the slowest successful cell (`tenferro-trace`, 6.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x4x4,rhs=1`): `pytorch-cpu` is 44.7x faster than the slowest successful cell (`tenferro-trace`, 6.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x4x4,rhs=1`): `tenferro-direct` is 34.8x faster than the slowest successful cell (`tenferro-trace`, 6.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `jax-cpu` is 37.1x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `pytorch-cpu` is 79.6x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `tenferro-direct` is 53.6x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x128x128,rhs=1`): `jax-cpu` is 48.4x faster than the slowest successful cell (`tenferro-trace`, 5.992 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x128x128,rhs=1`): `pytorch-cpu` is 40.4x faster than the slowest successful cell (`tenferro-trace`, 5.992 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x128x128,rhs=1`): `tenferro-direct` is 65.6x faster than the slowest successful cell (`tenferro-trace`, 5.992 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `jax-cpu` is 41.4x faster than the slowest successful cell (`tenferro-trace`, 24.016 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `pytorch-cpu` is 54.5x faster than the slowest successful cell (`tenferro-trace`, 24.016 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `tenferro-direct` is 32.9x faster than the slowest successful cell (`tenferro-trace`, 24.016 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 24.1x faster than the slowest successful cell (`tenferro-trace`, 313.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `julia-base` is 28.0x faster than the slowest successful cell (`tenferro-trace`, 313.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 32.7x faster than the slowest successful cell (`tenferro-trace`, 313.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`256x128`): `jax-cpu` is 20.8x faster than the slowest successful cell (`tenferro-eager`, 29.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`256x128`): `pytorch-cpu` is 70.0x faster than the slowest successful cell (`tenferro-eager`, 29.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`256x128`): `tenferro-direct` is 45.1x faster than the slowest successful cell (`tenferro-eager`, 29.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`768x384`): `jax-cpu` is 258.4x faster than the slowest successful cell (`tenferro-eager`, 2299.821 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 497.3x faster than the slowest successful cell (`tenferro-eager`, 2299.821 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 13.2x faster than the slowest successful cell (`tenferro-trace`, 268.280 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `julia-base` is 18.1x faster than the slowest successful cell (`tenferro-trace`, 268.280 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 13.8x faster than the slowest successful cell (`tenferro-trace`, 268.280 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `jax-cpu` is 58.8x faster than the slowest successful cell (`tenferro-eager`, 2292.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `julia-base` is 474.1x faster than the slowest successful cell (`tenferro-eager`, 2292.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 442.0x faster than the slowest successful cell (`tenferro-eager`, 2292.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 12.4x faster than the slowest successful cell (`tenferro-trace`, 245.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `julia-base` is 11.6x faster than the slowest successful cell (`tenferro-trace`, 245.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 17.6x faster than the slowest successful cell (`tenferro-trace`, 245.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `pytorch-cpu` is 20.5x faster than the slowest successful cell (`jax-cpu`, 19.624 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 26.9x faster than the slowest successful cell (`jax-cpu`, 19.624 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 27.2x faster than the slowest successful cell (`jax-cpu`, 19.624 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `julia-base` is 10.4x faster than the slowest successful cell (`tenferro-trace`, 23.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 25.4x faster than the slowest successful cell (`tenferro-trace`, 23.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 61.1x faster than the slowest successful cell (`tenferro-trace`, 23.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `jax-cpu` is 11.7x faster than the slowest successful cell (`tenferro-trace`, 221.933 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `julia-base` is 13.1x faster than the slowest successful cell (`tenferro-trace`, 221.933 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 15.2x faster than the slowest successful cell (`tenferro-trace`, 221.933 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `jax-cpu` is 11.5x faster than the slowest successful cell (`tenferro-trace`, 215.668 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `julia-base` is 12.7x faster than the slowest successful cell (`tenferro-trace`, 215.668 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 14.7x faster than the slowest successful cell (`tenferro-trace`, 215.668 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 24.8x faster than the slowest successful cell (`tenferro-trace`, 325.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `julia-base` is 29.8x faster than the slowest successful cell (`tenferro-trace`, 325.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 33.3x faster than the slowest successful cell (`tenferro-trace`, 325.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 52.5x faster than the slowest successful cell (`tenferro-eager`, 2316.601 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `julia-base` is 54.4x faster than the slowest successful cell (`tenferro-eager`, 2316.601 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 69.0x faster than the slowest successful cell (`tenferro-eager`, 2316.601 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 201.7x faster than the slowest successful cell (`tenferro-eager`, 23.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 361.6x faster than the slowest successful cell (`tenferro-eager`, 23.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 605.4x faster than the slowest successful cell (`tenferro-eager`, 23.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 30.2x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 37.2x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 42.1x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 100.5x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 184.0x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 608.0x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 13.6x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 24.2x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 42.6x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 111.7x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 167.8x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 241.3x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 11.4x faster than the slowest successful cell (`tenferro-eager`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 11.8x faster than the slowest successful cell (`tenferro-eager`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 15.7x faster than the slowest successful cell (`tenferro-eager`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 443.7x faster than the slowest successful cell (`tenferro-eager`, 64.762 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 1188.4x faster than the slowest successful cell (`tenferro-eager`, 64.762 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 885.0x faster than the slowest successful cell (`tenferro-eager`, 64.762 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 95.2x faster than the slowest successful cell (`tenferro-eager`, 95.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 324.8x faster than the slowest successful cell (`tenferro-eager`, 95.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 262.6x faster than the slowest successful cell (`tenferro-eager`, 95.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 457.6x faster than the slowest successful cell (`tenferro-eager`, 89.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 1968.8x faster than the slowest successful cell (`tenferro-eager`, 89.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 931.8x faster than the slowest successful cell (`tenferro-eager`, 89.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 83.4x faster than the slowest successful cell (`tenferro-eager`, 119.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 364.6x faster than the slowest successful cell (`tenferro-eager`, 119.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 241.4x faster than the slowest successful cell (`tenferro-eager`, 119.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 499.1x faster than the slowest successful cell (`tenferro-eager`, 112.676 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 653.4x faster than the slowest successful cell (`tenferro-eager`, 112.676 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 78.3x faster than the slowest successful cell (`tenferro-eager`, 143.873 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 104.6x faster than the slowest successful cell (`tenferro-eager`, 143.873 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 357.2x faster than the slowest successful cell (`tenferro-eager`, 23.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 2259.4x faster than the slowest successful cell (`tenferro-eager`, 23.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 2255.8x faster than the slowest successful cell (`tenferro-eager`, 23.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 152.1x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 625.0x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 474.8x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 219.8x faster than the slowest successful cell (`tenferro-eager`, 32.687 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 355.9x faster than the slowest successful cell (`tenferro-eager`, 32.687 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 823.0x faster than the slowest successful cell (`tenferro-eager`, 32.687 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 55.8x faster than the slowest successful cell (`tenferro-eager`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 191.9x faster than the slowest successful cell (`tenferro-eager`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 243.0x faster than the slowest successful cell (`tenferro-eager`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 115.2x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 193.7x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 849.9x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 16.9x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 25.4x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 61.8x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 111.5x faster than the slowest successful cell (`tenferro-eager`, 11.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 189.7x faster than the slowest successful cell (`tenferro-eager`, 11.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 824.7x faster than the slowest successful cell (`tenferro-eager`, 11.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 15.4x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 25.4x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 60.0x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 467.7x faster than the slowest successful cell (`tenferro-eager`, 88.948 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 2005.0x faster than the slowest successful cell (`tenferro-eager`, 88.948 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 1005.0x faster than the slowest successful cell (`tenferro-eager`, 88.948 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 84.4x faster than the slowest successful cell (`tenferro-eager`, 120.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 414.3x faster than the slowest successful cell (`tenferro-eager`, 120.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 224.9x faster than the slowest successful cell (`tenferro-eager`, 120.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 71.3x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 101.3x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 298.8x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 13.1x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 19.7x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 14.0x faster than the slowest successful cell (`tenferro-eager`, 20.184 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 1021.8x faster than the slowest successful cell (`tenferro-eager`, 104.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 435.9x faster than the slowest successful cell (`tenferro-eager`, 104.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 268.4x faster than the slowest successful cell (`tenferro-eager`, 104.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 165.8x faster than the slowest successful cell (`tenferro-eager`, 143.249 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 70.0x faster than the slowest successful cell (`tenferro-eager`, 143.249 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 19.9x faster than the slowest successful cell (`tenferro-eager`, 143.249 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/output_reuse/solve_read_into` (f64, threads=4, shape=`768x768,rhs=16`): `pytorch-cpu` is 31.8x faster than the slowest successful cell (`tenferro-direct`, 181.623 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 73380.9x faster than the slowest successful cell (`julia-base`, 110.952 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 78652.9x faster than the slowest successful cell (`julia-base`, 110.350 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/cast_f64_f32` (f64->f32, threads=4, shape=`33554432`): `pytorch-cpu` is 10.9x faster than the slowest successful cell (`tenferro-trace`, 227.577 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/cast_f64_f32` (f64->f32, threads=4, shape=`33554432`): `tenferro-direct` is 10.9x faster than the slowest successful cell (`tenferro-trace`, 227.577 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 81338.5x faster than the slowest successful cell (`julia-base`, 129.572 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 85927.1x faster than the slowest successful cell (`julia-base`, 134.218 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 128002.8x faster than the slowest successful cell (`julia-base`, 176.900 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 124221.9x faster than the slowest successful cell (`julia-base`, 176.644 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 129.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 54.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 133.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 35.5x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 165.7x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 137.6x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 10667.3 GB/s at 0.002 ms. The conservative physical lower bound is 0.168 ms, so the cell is 106.7x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 10206.3 GB/s at 0.002 ms. The conservative physical lower bound is 0.042 ms, so the cell is 25.5x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 177536.7 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1775.4x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 191329.6 GB/s at 0.001 ms. The conservative physical lower bound is 0.671 ms, so the cell is 478.3x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 168509.4 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1685.1x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 171853.7 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 429.6x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 97118.5 GB/s at 0.001 ms. The conservative physical lower bound is 1.342 ms, so the cell is 971.2x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 94386.6 GB/s at 0.001 ms. The conservative physical lower bound is 0.336 ms, so the cell is 236.0x faster than the bound.

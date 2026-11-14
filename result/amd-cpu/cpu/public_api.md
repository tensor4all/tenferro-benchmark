# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/amd-cpu/cpu/public_api/20261114_191755/run.yaml`
- Timestamp: `2026-11-14T19:17:55.104593+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/amd-cpu/cpu/public_api/20261114_191755`.

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

- CSV: `data/results/amd-cpu/cpu/public_api/20261114_191755/cpu_public_api_t1_20261114_191755.csv`
- CSV: `data/results/amd-cpu/cpu/public_api/20261114_191755/cpu_public_api_t4_20261114_191755.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 4.264 ± 0.042 | - | 4.201 ± 0.057 | 4.526 ± 0.082 | 3.835 ± 0.019 | 3.697 ± 1.007 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 4.550 ± 0.067 | - | 4.403 ± 0.056 | 2.322 ± 0.094 | 3.491 ± 1.738 | 4.000 ± 1.933 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 124.628 ± 3.710 | - | 125.840 ± 4.247 | 121.944 ± 2.994 | 128.318 ± 3.139 | 145.314 ± 3.868 | 146.367 ± 13.063 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 48.040 ± 33.596 | - | 143.180 ± 0.538 | 41.156 ± 0.761 | 64.608 ± 1.561 | 145.176 ± 3.309 | 64.805 ± 18.489 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 107.133 ± 0.199 | - | 112.345 ± 2.755 | 93.830 ± 1.229 | 94.062 ± 1.811 | 133.306 ± 4.870 | 142.687 ± 1.600 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 44.308 ± 1.278 | - | 129.852 ± 3.093 | 31.046 ± 1.559 | 48.398 ± 0.788 | 129.386 ± 5.127 | 63.347 ± 3.156 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 44.516 ± 0.931 | - | 45.323 ± 1.259 | 42.661 ± 0.414 | 68.770 ± 0.922 | 46.786 ± 0.297 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 45.354 ± 1.125 | - | 46.625 ± 2.365 | 29.219 ± 0.827 | 29.615 ± 0.734 | 23.397 ± 1.389 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 46.732 ± 2.346 | - | unsupported | 43.893 ± 0.947 | 68.834 ± 1.241 | 47.383 ± 5.513 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 45.654 ± 0.514 | - | unsupported | 29.442 ± 0.132 | 30.333 ± 0.817 | 24.273 ± 4.552 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 10.939 ± 0.108 | - | 10.972 ± 0.063 | 11.613 ± 0.130 | 11.881 ± 0.094 | 11.959 ± 0.212 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 14.201 ± 0.047 | - | 14.266 ± 0.112 | 14.050 ± 0.102 | 17.522 ± 0.149 | 17.710 ± 0.056 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 72.559 ± 0.557 | - | 72.461 ± 0.238 | 107.205 ± 1.154 | 103.690 ± 2.014 | 85.373 ± 18.482 | 82.850 ± 1.520 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 31.062 ± 0.464 | - | 95.997 ± 0.008 | 69.065 ± 11.527 | 72.764 ± 15.764 | 82.688 ± 11.150 | 37.311 ± 5.798 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 116.587 ± 0.460 | - | 116.754 ± 0.708 | 153.535 ± 1.070 | 176.725 ± 0.630 | 149.872 ± 1.963 | 160.295 ± 4.068 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 49.601 ± 9.539 | - | 131.995 ± 0.012 | 86.432 ± 33.328 | 102.769 ± 30.995 | 149.146 ± 5.673 | 66.559 ± 16.898 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 72.084 ± 1.687 | - | 72.046 ± 2.109 | 68.788 ± 1.513 | 71.719 ± 2.548 | 79.052 ± 4.404 | 79.436 ± 2.004 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 25.052 ± 0.825 | - | 93.656 ± 0.490 | 23.187 ± 0.424 | 37.334 ± 0.950 | 77.903 ± 0.806 | 34.530 ± 1.332 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 2.538 ± 0.085 | - | 20.943 ± 0.097 | 24.222 ± 0.099 | 16.244 ± 0.323 | 4.293 ± 0.083 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 1.025 ± 0.089 | - | 59.998 ± 0.016 | 11.247 ± 0.034 | 2.980 ± 0.218 | 4.319 ± 0.085 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 8.644 ± 0.041 | - | 8.806 ± 0.061 | 9.605 ± 0.034 | 7.600 ± 0.065 | 8.935 ± 0.251 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 107.920 ± 23.205 | - | 127.951 ± 12.098 | 5.726 ± 0.054 | 6.908 ± 0.096 | 7.580 ± 0.167 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 5.486 ± 0.115 | - | 5.711 ± 0.294 | 5.420 ± 0.068 | 5.111 ± 0.095 | 4.868 ± 0.071 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 101.021 ± 12.817 | - | 140.069 ± 14.734 | 3.637 ± 0.085 | 4.792 ± 0.172 | 4.638 ± 0.992 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 6.486 ± 0.025 | - | 6.663 ± 0.222 | 7.614 ± 0.020 | 5.403 ± 0.042 | 5.369 ± 0.126 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 11.611 ± 0.151 | - | 11.690 ± 0.242 | 6.317 ± 0.453 | 6.154 ± 0.199 | 5.954 ± 0.839 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 44.872 ± 0.856 | - | 47.609 ± 4.702 | 43.197 ± 0.308 | 68.681 ± 0.376 | 44.086 ± 0.209 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 45.258 ± 3.162 | - | 47.387 ± 12.799 | 29.365 ± 0.610 | 29.501 ± 0.592 | 22.329 ± 0.193 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 50.190 ± 0.634 | - | unsupported | 50.156 ± 1.344 | 60.377 ± 0.440 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 54.706 ± 0.548 | - | unsupported | 25.284 ± 0.451 | 34.884 ± 0.436 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 123.552 ± 4.146 | - | 126.278 ± 2.728 | 122.800 ± 0.958 | 125.344 ± 2.381 | 139.100 ± 40.275 | 155.059 ± 3.538 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 42.142 ± 1.514 | - | 143.101 ± 3.263 | 40.857 ± 0.441 | 64.591 ± 0.632 | 139.019 ± 4.139 | 65.453 ± 0.860 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 136.638 ± 4.182 | - | 134.323 ± 4.321 | 137.701 ± 2.703 | 139.181 ± 1.503 | 151.297 ± 10.411 | 164.530 ± 4.286 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 47.376 ± 2.491 | - | 150.528 ± 9.396 | 45.737 ± 1.637 | 72.209 ± 3.106 | 149.658 ± 6.502 | 68.703 ± 4.054 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 27.513 ± 0.235 | - | 58.806 ± 0.835 | 78.527 ± 2.610 | 51.379 ± 0.807 | 80.244 ± 9.006 | 85.064 ± 9.399 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 12.816 ± 0.217 | - | 119.997 ± 0.005 | 33.414 ± 0.744 | 25.946 ± 0.649 | 84.858 ± 10.416 | 33.612 ± 10.270 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 10.956 ± 0.343 | - | 10.016 ± 0.455 | 36.511 ± 0.851 | 38.961 ± 0.882 | 36.708 ± 8.494 | 46.691 ± 10.006 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 7.720 ± 0.633 | - | 23.994 ± 0.009 | 12.723 ± 0.620 | 19.869 ± 0.364 | 35.404 ± 8.879 | 14.488 ± 17.397 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 21.448 ± 0.840 | - | 22.050 ± 0.808 | 56.063 ± 2.512 | 42.893 ± 0.763 | 22.434 ± 1.691 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 16.222 ± 0.960 | - | 36.001 ± 12.001 | 20.374 ± 2.694 | 24.825 ± 0.726 | 23.981 ± 0.921 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 28.360 ± 0.188 | - | 28.428 ± 0.054 | 58.687 ± 0.759 | 107.565 ± 0.285 | 70.272 ± 9.542 | 70.674 ± 16.393 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 11.904 ± 0.064 | - | 47.998 ± 0.007 | 21.766 ± 0.281 | 54.345 ± 0.471 | 66.467 ± 14.481 | 30.281 ± 16.580 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 137.379 ± 3.148 | - | 135.537 ± 4.275 | 131.183 ± 1.652 | 148.382 ± 6.907 | 152.783 ± 1.258 | 160.427 ± 4.954 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 45.714 ± 0.326 | - | 152.578 ± 11.936 | 45.011 ± 0.593 | 75.767 ± 3.923 | 152.971 ± 35.110 | 68.031 ± 3.265 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 20.428 ± 0.070 | - | 20.797 ± 0.044 | 44.506 ± 0.928 | 41.074 ± 0.439 | 77.991 ± 18.365 | 69.931 ± 15.036 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 8.412 ± 0.038 | - | 35.997 ± 0.008 | 16.658 ± 0.376 | 21.079 ± 0.329 | 62.669 ± 13.606 | 32.175 ± 18.096 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 20.381 ± 0.019 | - | 20.652 ± 0.066 | 32.551 ± 1.344 | 28.163 ± 0.187 | 38.011 ± 8.598 | 40.328 ± 10.792 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 8.100 ± 0.030 | - | 36.001 ± 0.013 | 11.386 ± 0.202 | 13.867 ± 0.102 | 37.625 ± 9.655 | 16.274 ± 9.986 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 25.937 ± 0.125 | - | 25.374 ± 0.054 | 50.493 ± 0.832 | 66.320 ± 5.377 | 85.303 ± 12.898 | 89.062 ± 12.462 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 10.970 ± 0.031 | - | 48.003 ± 2.999 | 18.855 ± 1.036 | 32.488 ± 0.145 | 83.681 ± 14.774 | 38.439 ± 19.311 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 13.230 ± 0.040 | - | 13.333 ± 0.061 | 33.425 ± 0.679 | 36.885 ± 0.262 | 46.296 ± 8.823 | 50.152 ± 8.929 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 5.826 ± 0.022 | - | 35.995 ± 0.010 | 11.825 ± 0.143 | 18.860 ± 7.010 | 46.429 ± 10.285 | 20.905 ± 10.310 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.461 ± 0.019 | 0.496 ± 0.022 | unsupported | unsupported | 0.517 ± 0.012 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.402 ± 0.010 | 52.000 ± 0.002 | unsupported | unsupported | 0.388 ± 0.009 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 137.572 ± 6.465 | - | 138.848 ± 3.889 | 136.139 ± 2.302 | 139.480 ± 2.790 | 156.515 ± 31.608 | 168.741 ± 2.083 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 46.295 ± 0.674 | - | 153.794 ± 3.482 | 45.452 ± 5.275 | 71.850 ± 1.013 | 154.518 ± 8.029 | 70.409 ± 1.114 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 137.774 ± 2.924 | - | 137.413 ± 4.639 | 137.139 ± 3.274 | 140.073 ± 2.068 | 156.552 ± 42.398 | 165.258 ± 4.682 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 46.629 ± 0.590 | - | 153.748 ± 0.489 | 44.903 ± 0.694 | 71.919 ± 2.082 | 155.900 ± 3.633 | 69.653 ± 2.230 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 133.614 ± 2.657 | - | 134.005 ± 5.252 | 132.434 ± 2.362 | 137.798 ± 1.341 | 148.991 ± 5.541 | 166.943 ± 3.441 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 48.294 ± 1.400 | - | 150.573 ± 12.356 | 45.705 ± 1.038 | 71.260 ± 1.250 | 152.015 ± 7.215 | 68.846 ± 3.163 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 124.551 ± 3.805 | - | 126.837 ± 3.653 | 121.696 ± 1.025 | 124.687 ± 0.631 | 141.723 ± 12.124 | 153.541 ± 3.871 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 41.460 ± 0.284 | - | 143.153 ± 0.265 | 40.821 ± 0.665 | 64.407 ± 0.874 | 138.427 ± 3.661 | 64.450 ± 1.798 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 58.661 ± 0.221 | - | 58.920 ± 0.127 | 62.780 ± 0.417 | 74.011 ± 0.281 | 125.432 ± 1.219 | 122.760 ± 20.399 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 23.984 ± 0.321 | - | 71.998 ± 0.006 | 22.153 ± 4.344 | 37.482 ± 0.991 | 125.492 ± 10.189 | 51.061 ± 14.613 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 17.554 ± 0.194 | - | 17.864 ± 0.584 | 13.328 ± 1.260 | 26.828 ± 0.561 | 15.461 ± 0.603 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 7.009 ± 0.298 | - | 35.998 ± 0.011 | 6.796 ± 0.490 | 14.494 ± 0.408 | 14.253 ± 0.849 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 2.100 ± 0.068 | - | 2.125 ± 0.151 | 24.406 ± 0.403 | 2.466 ± 0.040 | 1.312 ± 0.453 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.806 ± 0.006 | - | 23.998 ± 0.009 | 11.192 ± 0.524 | 1.583 ± 0.049 | 1.372 ± 0.184 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 2.524 ± 0.053 | - | 2.565 ± 0.048 | 4.082 ± 0.031 | 2.855 ± 0.108 | 1.417 ± 0.263 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 1.300 ± 0.024 | - | 23.995 ± 0.011 | 2.307 ± 0.019 | 1.376 ± 0.213 | 1.930 ± 0.108 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 18.955 ± 0.840 | - | 17.764 ± 0.797 | 13.001 ± 1.044 | 26.927 ± 0.448 | 14.931 ± 0.900 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 6.746 ± 0.250 | - | 35.996 ± 0.014 | 6.654 ± 0.463 | 14.209 ± 0.519 | 13.756 ± 0.504 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.988 ± 0.119 | - | 2.017 ± 0.149 | 23.725 ± 0.504 | 2.569 ± 0.041 | 1.674 ± 0.175 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.803 ± 0.031 | - | 24.000 ± 0.001 | 10.666 ± 0.706 | 1.338 ± 0.039 | 1.527 ± 0.070 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 10.709 ± 0.243 | - | 11.035 ± 0.578 | 16.393 ± 0.502 | 12.651 ± 0.071 | 7.287 ± 0.107 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 5.249 ± 0.118 | - | 35.997 ± 0.014 | 8.823 ± 0.049 | 7.003 ± 0.480 | 7.745 ± 0.593 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 12.075 ± 0.589 | - | 12.631 ± 0.811 | 11.052 ± 0.179 | 20.579 ± 0.738 | 12.146 ± 0.452 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 6.536 ± 0.319 | - | 23.999 ± 0.009 | 7.002 ± 0.123 | 11.167 ± 0.242 | 11.965 ± 0.440 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 1.371 ± 0.071 | - | 1.243 ± 0.160 | 2.123 ± 0.099 | 2.027 ± 0.043 | 1.211 ± 0.066 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.636 ± 0.094 | - | 23.998 ± 0.012 | 0.973 ± 0.081 | 1.129 ± 0.012 | 1.273 ± 0.138 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.402 ± 0.073 | - | 1.329 ± 0.223 | 0.881 ± 0.133 | 3.315 ± 0.466 | 1.288 ± 0.101 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.920 ± 0.179 | - | 23.999 ± 0.009 | 0.362 ± 0.054 | 1.788 ± 0.181 | 1.476 ± 0.153 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 11.786 ± 0.313 | - | 12.223 ± 0.449 | 11.102 ± 0.234 | 20.726 ± 0.575 | 12.893 ± 0.331 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 6.391 ± 0.197 | - | 35.994 ± 11.998 | 7.392 ± 0.254 | 11.349 ± 0.172 | 11.658 ± 0.297 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 1.313 ± 0.114 | - | 1.205 ± 0.128 | 2.042 ± 0.089 | 2.260 ± 0.119 | 1.035 ± 0.114 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.593 ± 0.042 | - | 23.998 ± 0.009 | 0.927 ± 0.049 | 1.140 ± 0.017 | 1.314 ± 0.218 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.431 ± 0.153 | - | 1.495 ± 0.222 | 0.870 ± 0.031 | 4.077 ± 0.231 | 1.103 ± 0.128 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.736 ± 0.062 | - | 23.993 ± 0.012 | 0.399 ± 0.066 | 1.819 ± 0.062 | 1.405 ± 0.215 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 1.198 ± 0.057 | 1.128 ± 0.134 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.535 ± 0.235 | 24.000 ± 0.001 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 40.006 ± 0.239 | - | 40.424 ± 1.386 | 54.974 ± 0.962 | 72.371 ± 1.353 | 70.341 ± 14.958 | 77.938 ± 8.849 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 18.247 ± 0.367 | - | 59.999 ± 0.013 | 20.532 ± 0.407 | 36.863 ± 0.309 | 71.086 ± 11.526 | 33.538 ± 10.567 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 143.959 ± 4.027 | - | 143.454 ± 2.714 | 146.983 ± 2.185 | 248.715 ± 4.122 | 266.539 ± 1.549 | 267.055 ± 1.332 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 47.093 ± 0.359 | - | 164.286 ± 12.228 | 46.837 ± 0.594 | 124.855 ± 1.247 | 262.435 ± 2.609 | 108.010 ± 4.337 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 139.675 ± 4.177 | - | 137.863 ± 3.224 | 148.529 ± 4.673 | 153.965 ± 2.892 | 150.565 ± 5.576 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 48.817 ± 2.223 | - | 156.513 ± 0.559 | 51.480 ± 0.711 | 80.864 ± 2.653 | 154.221 ± 4.445 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 131.547 ± 6.012 | - | 131.322 ± 3.769 | 127.021 ± 2.078 | 126.853 ± 2.744 | 145.357 ± 30.163 | 149.091 ± 2.511 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 42.408 ± 1.042 | - | 143.090 ± 1.065 | 42.283 ± 1.052 | 65.060 ± 1.213 | 143.838 ± 3.333 | 63.060 ± 0.618 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 24.501 ± 0.047 | - | 24.540 ± 0.138 | 59.464 ± 0.908 | 108.339 ± 1.707 | 66.493 ± 13.733 | 70.148 ± 8.963 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 10.226 ± 0.051 | - | 47.999 ± 0.010 | 22.141 ± 0.355 | 54.349 ± 0.404 | 76.597 ± 9.877 | 43.287 ± 12.049 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 133.205 ± 4.755 | - | 136.279 ± 3.527 | 186.675 ± 1.203 | 134.736 ± 2.845 | 218.909 ± 2.610 | 218.733 ± 3.231 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 44.030 ± 0.329 | - | 155.054 ± 10.951 | 69.121 ± 1.404 | 68.706 ± 0.734 | 215.870 ± 1.100 | 90.828 ± 38.477 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 134.885 ± 4.248 | - | 133.700 ± 5.258 | 133.551 ± 5.406 | 138.369 ± 1.235 | 151.400 ± 3.883 | 165.597 ± 5.232 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 47.627 ± 1.008 | - | 153.285 ± 12.718 | 45.491 ± 0.311 | 71.913 ± 0.317 | 150.905 ± 29.218 | 68.146 ± 1.450 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 51.361 ± 0.070 | - | 51.426 ± 0.128 | 75.617 ± 0.976 | 39.388 ± 0.667 | 81.282 ± 13.008 | 87.670 ± 12.820 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 20.493 ± 0.239 | - | 71.999 ± 0.014 | 28.470 ± 0.294 | 20.161 ± 0.407 | 79.148 ± 14.503 | 34.603 ± 15.839 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 137.706 ± 4.888 | 135.634 ± 4.715 | unsupported | 150.968 ± 8.527 | 156.774 ± 1.851 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 47.337 ± 0.675 | 156.779 ± 14.203 | unsupported | 51.069 ± 0.714 | 80.766 ± 2.838 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 1.059 ± 0.123 | - | 0.954 ± 0.213 | 0.804 ± 0.024 | 1.004 ± 0.410 | 7.539 ± 4.108 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.468 ± 0.040 | - | 11.996 ± 0.006 | 0.499 ± 0.059 | 1.008 ± 0.154 | 7.517 ± 3.987 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.989 ± 0.348 | - | 1.084 ± 0.073 | 0.963 ± 0.084 | 1.092 ± 0.192 | 1.542 ± 1.286 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.508 ± 0.132 | - | 11.999 ± 0.007 | 0.445 ± 0.077 | 0.816 ± 0.123 | 8.057 ± 4.147 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.420 ± 0.072 | - | unsupported | 1.378 ± 0.055 | 1.358 ± 0.260 | 8.237 ± 3.093 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.770 ± 0.066 | - | unsupported | 0.964 ± 0.288 | 1.048 ± 0.099 | 8.279 ± 8.514 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.405 ± 0.030 | - | 0.493 ± 0.026 | 0.695 ± 0.023 | 0.439 ± 0.013 | 1.261 ± 0.087 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.428 ± 0.015 | - | 47.998 ± 0.019 | 0.350 ± 0.016 | 0.397 ± 0.030 | 1.157 ± 0.070 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.991 ± 0.179 | - | 1.029 ± 0.265 | 1.169 ± 0.139 | 1.115 ± 0.131 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.451 ± 0.171 | - | 12.001 ± 0.014 | 0.587 ± 0.104 | 1.073 ± 0.036 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.879 ± 0.108 | - | 1.018 ± 0.061 | 0.817 ± 0.041 | 1.030 ± 0.131 | 1.683 ± 11.009 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.485 ± 0.023 | - | 11.998 ± 0.009 | 0.610 ± 0.267 | 0.734 ± 0.172 | 8.465 ± 8.436 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.827 ± 0.023 | - | 0.941 ± 0.014 | 0.525 ± 0.079 | 0.505 ± 0.022 | 1.380 ± 0.262 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 1.236 ± 0.035 | - | 35.975 ± 0.141 | 0.657 ± 0.027 | 0.544 ± 0.042 | 1.476 ± 0.088 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.812 ± 0.057 | - | 0.002 ± 0.000 | 1.936 ± 0.060 | 1.635 ± 0.164 | 9.346 ± 5.507 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.760 ± 0.065 | - | 0.002 ± 0.000 | 1.433 ± 0.280 | 1.178 ± 0.054 | 9.069 ± 4.372 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.051 ± 0.000 | - | 0.054 ± 0.014 | 0.076 ± 0.002 | 0.116 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.093 ± 0.000 | - | 0.096 ± 0.001 | 0.121 ± 0.001 | 0.157 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.245 ± 0.022 | - | 0.245 ± 0.003 | 0.274 ± 0.005 | 0.324 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.082 ± 0.002 | - | 0.083 ± 0.001 | 0.148 ± 0.002 | 0.120 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.049 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.015 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.037 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.687 ± 0.029 | - | 0.699 ± 0.012 | 1.157 ± 0.011 | 0.674 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.026 ± 0.000 | - | 0.028 ± 0.001 | 0.038 ± 0.001 | 0.061 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.042 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.064 ± 0.000 | - | 0.073 ± 0.007 | 0.085 ± 0.001 | 0.105 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.094 ± 0.001 | - | 0.098 ± 0.026 | 0.124 ± 0.001 | 0.158 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.244 ± 0.021 | - | 0.247 ± 0.004 | 0.339 ± 0.014 | 0.361 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.083 ± 0.005 | - | 0.107 ± 0.015 | 0.179 ± 0.005 | 0.127 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.013 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.004 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.002 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.001 | 0.014 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.011 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.001 | 0.015 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.691 ± 0.029 | - | 0.700 ± 4.652 | 1.405 ± 0.030 | 1.054 ± 2.874 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.031 ± 0.000 | - | 0.033 ± 0.000 | 0.059 ± 0.004 | 0.056 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.618 ± 0.015 | - | 0.617 ± 0.007 | 0.719 ± 0.017 | 0.894 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 2.110 ± 0.019 | - | 2.124 ± 0.056 | 2.144 ± 0.008 | 2.756 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 6.463 ± 0.008 | - | 6.482 ± 0.031 | 6.528 ± 0.027 | 8.996 ± 0.074 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 3.702 ± 0.031 | - | 3.722 ± 0.029 | 3.227 ± 0.067 | 2.613 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.001 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.114 ± 0.028 | - | 0.112 ± 0.001 | 0.122 ± 0.005 | 0.188 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.031 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.008 ± 0.001 | - | 0.010 ± 0.000 | 0.018 ± 0.002 | 0.032 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.001 | 0.030 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.016 ± 0.001 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.020 ± 0.000 | - | 0.023 ± 0.001 | 0.029 ± 0.001 | 0.069 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.034 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | 0.012 ± 0.001 | 0.018 ± 0.000 | 0.035 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.027 ± 0.000 | - | 0.029 ± 0.000 | 0.036 ± 0.002 | 0.106 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 29.321 ± 0.216 | - | 29.345 ± 0.159 | 25.560 ± 0.041 | 20.674 ± 0.441 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.014 ± 0.002 | 0.033 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.866 ± 0.004 | - | 0.871 ± 0.003 | 0.874 ± 0.005 | 0.966 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.018 ± 0.000 | - | 0.020 ± 0.000 | 0.026 ± 0.004 | 0.054 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.052 ± 0.000 | - | 0.054 ± 0.003 | 0.061 ± 0.001 | 0.116 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.619 ± 0.041 | - | 0.638 ± 4.636 | 0.727 ± 0.426 | 0.922 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 2.126 ± 0.015 | - | 2.137 ± 0.038 | 2.142 ± 0.021 | 2.189 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 6.519 ± 0.025 | - | 6.573 ± 0.055 | 9.381 ± 0.035 | 7.374 ± 1.874 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 6.657 ± 0.024 | - | 6.722 ± 0.019 | 2.317 ± 0.046 | 3.470 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.014 ± 0.002 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.110 ± 0.001 | - | 0.125 ± 0.007 | 0.180 ± 0.008 | 0.178 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.001 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.017 ± 0.002 | 0.039 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.017 ± 0.002 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.018 ± 0.005 | 0.032 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.021 ± 0.000 | - | 0.024 ± 0.000 | 0.030 ± 0.001 | 0.089 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.021 ± 0.005 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.018 ± 0.001 | 0.041 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.028 ± 0.000 | - | 0.030 ± 0.000 | 0.036 ± 0.000 | 0.096 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 53.868 ± 0.347 | - | 53.546 ± 0.202 | 18.060 ± 0.256 | 28.981 ± 7.506 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.021 ± 0.001 | 0.028 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.876 ± 0.032 | - | 0.883 ± 0.014 | 1.270 ± 0.012 | 0.992 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.018 ± 0.000 | - | 0.021 ± 0.000 | 0.026 ± 0.001 | 0.045 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.054 ± 0.013 | - | 0.055 ± 0.000 | 0.061 ± 0.001 | 0.132 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.682 ± 0.009 | - | 0.685 ± 0.008 | 0.521 ± 0.010 | 0.488 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 2.232 ± 0.007 | - | 2.232 ± 0.018 | 1.976 ± 0.005 | 2.007 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 5.814 ± 0.029 | - | 5.799 ± 0.019 | 5.752 ± 0.046 | 6.492 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.333 ± 0.023 | - | 1.340 ± 0.054 | 1.322 ± 0.010 | 1.498 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.021 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.071 ± 0.002 | - | 0.074 ± 0.005 | 0.089 ± 0.007 | 0.130 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.028 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.015 ± 0.000 | 0.035 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.015 ± 0.006 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.026 ± 0.001 | 0.042 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.011 ± 0.000 | - | 0.013 ± 0.001 | 0.017 ± 0.001 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.024 ± 0.000 | - | 0.027 ± 0.000 | 0.032 ± 0.003 | 0.048 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 10.671 ± 0.093 | - | 10.648 ± 0.145 | 10.433 ± 0.038 | 11.412 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.001 | 0.012 ± 0.000 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.549 ± 0.011 | - | 0.564 ± 0.040 | 0.629 ± 0.015 | 0.737 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.024 ± 0.001 | 0.036 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.074 ± 0.032 | - | 0.052 ± 0.026 | 0.053 ± 0.001 | 0.123 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.666 ± 0.028 | - | 0.661 ± 4.692 | 0.796 ± 0.099 | 0.448 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 2.196 ± 0.014 | - | 2.216 ± 0.017 | 1.940 ± 0.043 | 1.275 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 5.858 ± 0.033 | - | 5.834 ± 0.051 | 8.529 ± 0.129 | 3.705 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 3.156 ± 0.020 | - | 3.159 ± 0.031 | 1.232 ± 7.059 | 1.794 ± 0.210 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.012 ± 0.000 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.140 ± 0.040 | - | 0.149 ± 0.040 | 0.136 ± 0.005 | 0.154 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.001 | 0.025 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.015 ± 0.001 | 0.033 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.015 ± 0.005 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.015 ± 0.001 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.021 ± 0.000 | - | 0.023 ± 0.000 | 0.026 ± 0.001 | 0.050 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.016 ± 0.002 | 0.029 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.011 ± 0.000 | - | 0.014 ± 0.000 | 0.017 ± 0.001 | 0.031 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.026 ± 0.000 | - | 0.029 ± 0.000 | 0.031 ± 0.001 | 0.046 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 25.098 ± 0.138 | - | 25.203 ± 0.173 | 9.293 ± 0.075 | 13.974 ± 0.980 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.001 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 1.113 ± 0.022 | - | 1.119 ± 0.021 | 0.927 ± 0.011 | 0.462 ± 0.059 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.019 ± 0.000 | - | 0.023 ± 0.000 | 0.024 ± 0.001 | 0.037 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.049 ± 0.002 | - | 0.052 ± 0.000 | 0.053 ± 0.001 | 0.078 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.266 ± 0.010 | - | 0.267 ± 0.014 | 0.237 ± 0.084 | 0.525 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 1.115 ± 0.012 | - | 1.118 ± 0.003 | 0.943 ± 0.020 | 1.946 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 3.099 ± 0.012 | - | 3.110 ± 0.007 | 2.886 ± 0.075 | 6.407 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.516 ± 0.007 | - | 0.514 ± 0.008 | 0.550 ± 0.011 | 1.443 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.009 ± 0.001 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.031 ± 0.000 | - | 0.033 ± 0.000 | 0.041 ± 0.002 | 0.128 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.001 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.002 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.001 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.019 ± 0.002 | 0.041 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.001 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.021 ± 0.001 | 0.043 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 4.119 ± 0.060 | - | 4.065 ± 0.008 | 4.276 ± 0.015 | 11.168 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.002 | - | 0.005 ± 0.000 | 0.011 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.231 ± 0.012 | - | 0.235 ± 0.004 | 0.244 ± 0.013 | 0.734 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.010 ± 0.001 | - | 0.012 ± 0.000 | 0.017 ± 0.001 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.025 ± 0.000 | - | 0.027 ± 0.000 | 0.033 ± 0.004 | 0.092 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.269 ± 0.022 | - | 0.265 ± 0.004 | 0.398 ± 0.012 | 0.468 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 1.119 ± 0.029 | - | 1.132 ± 0.015 | 0.943 ± 0.006 | 1.869 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 3.126 ± 0.020 | - | 3.148 ± 0.021 | 3.659 ± 0.013 | 3.838 ± 0.337 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 2.013 ± 0.121 | - | 2.023 ± 0.017 | 0.703 ± 0.017 | 1.779 ± 0.234 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.014 ± 0.002 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.032 ± 0.001 | - | 0.036 ± 0.000 | 0.056 ± 0.002 | 0.113 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.002 | 0.019 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.001 | 0.029 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.004 | 0.021 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.019 ± 0.004 | 0.023 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.019 ± 0.001 | 0.043 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.001 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.020 ± 0.001 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.014 ± 0.000 | - | 0.017 ± 0.000 | 0.022 ± 0.001 | 0.044 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 16.161 ± 0.281 | - | 16.230 ± 0.144 | 5.327 ± 0.047 | 14.740 ± 5.920 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.019 ± 0.004 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.236 ± 0.022 | - | 0.239 ± 0.025 | 0.320 ± 0.012 | 0.417 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.024 ± 0.001 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.027 ± 0.000 | - | 0.028 ± 0.000 | 0.033 ± 0.001 | 0.111 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.148 ± 0.002 | - | 0.158 ± 0.001 | 0.189 ± 0.016 | 0.147 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.224 ± 0.002 | - | 0.221 ± 0.010 | 0.253 ± 0.003 | 0.235 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.509 ± 0.009 | - | 0.519 ± 0.012 | 0.537 ± 0.010 | 0.613 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.140 ± 0.005 | - | 0.139 ± 0.002 | 0.172 ± 0.016 | 0.223 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.013 ± 0.001 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.012 ± 0.000 | - | 0.015 ± 0.001 | 0.024 ± 0.001 | 0.077 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.013 ± 0.001 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.014 ± 0.001 | - | 0.017 ± 0.001 | 0.027 ± 0.001 | 0.053 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.013 ± 0.001 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.015 ± 0.001 | 0.021 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.017 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.017 ± 0.001 | 0.064 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.015 ± 0.001 | 0.022 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.015 ± 0.001 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.017 ± 0.001 | 0.068 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 1.082 ± 0.035 | - | 1.072 ± 0.019 | 1.292 ± 0.019 | 1.297 ± 0.053 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.015 ± 0.001 | 0.022 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.078 ± 0.001 | - | 0.081 ± 0.001 | 0.087 ± 0.000 | 0.150 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.016 ± 0.001 | 0.042 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.019 ± 0.001 | 0.071 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.150 ± 0.007 | - | 0.170 ± 0.014 | 0.175 ± 0.002 | 0.126 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.216 ± 0.008 | - | 0.220 ± 0.010 | 0.233 ± 0.017 | 0.226 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.507 ± 0.037 | - | 0.514 ± 0.020 | 0.378 ± 0.015 | 0.639 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.138 ± 0.003 | - | 0.171 ± 0.030 | 0.227 ± 0.017 | 0.238 ± 0.075 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.024 ± 0.003 | 0.019 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.013 ± 0.000 | - | 0.016 ± 0.001 | 0.038 ± 0.001 | 0.073 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.024 ± 0.003 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.015 ± 0.000 | - | 0.020 ± 0.001 | 0.042 ± 0.001 | 0.053 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.024 ± 0.004 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.026 ± 0.001 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.026 ± 0.000 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.028 ± 0.001 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.002 | 0.026 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.026 ± 0.006 | 0.025 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.029 ± 0.001 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 1.092 ± 0.024 | - | 1.081 ± 0.022 | 0.684 ± 0.032 | 1.006 ± 0.152 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.028 ± 0.001 | 0.027 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.086 ± 0.001 | - | 0.085 ± 0.013 | 0.087 ± 0.009 | 0.193 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.028 ± 0.000 | 0.019 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.031 ± 0.004 | 0.028 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.250 ± 0.016 | - | 0.247 ± 0.010 | 0.222 ± 0.005 | 0.329 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.600 ± 0.010 | - | 0.604 ± 0.009 | 0.527 ± 0.011 | 0.898 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.365 ± 0.018 | - | 1.354 ± 0.008 | 1.364 ± 0.003 | 2.351 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.496 ± 0.026 | - | 0.501 ± 0.016 | 0.569 ± 0.016 | 0.673 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.018 ± 0.000 | - | 0.020 ± 0.001 | 0.026 ± 0.001 | 0.089 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.021 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.019 ± 0.000 | - | 0.022 ± 0.000 | 0.029 ± 0.001 | 0.061 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.002 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.027 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.025 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.011 | 0.011 ± 0.001 | 0.036 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 3.927 ± 0.048 | - | 4.900 ± 0.081 | 4.405 ± 0.024 | 4.709 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.025 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.128 ± 0.001 | - | 0.133 ± 0.002 | 0.174 ± 0.003 | 0.243 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.012 ± 0.001 | - | 0.014 ± 0.000 | 0.016 ± 0.001 | 0.047 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.255 ± 0.013 | - | 0.253 ± 0.005 | 0.321 ± 0.012 | 0.328 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.617 ± 0.029 | - | 0.609 ± 4.676 | 0.765 ± 0.009 | 0.867 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.367 ± 0.049 | - | 1.346 ± 0.052 | 2.030 ± 0.040 | 1.335 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 1.050 ± 0.039 | - | 1.053 ± 0.026 | 0.514 ± 0.017 | 0.952 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.003 ± 0.003 | - | 0.005 ± 0.000 | 0.011 ± 0.001 | 0.025 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.019 ± 0.002 | - | 0.025 ± 0.000 | 0.044 ± 0.004 | 0.087 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.020 ± 0.000 | - | 0.031 ± 0.001 | 0.046 ± 0.002 | 0.067 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.028 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.001 | 0.027 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.016 ± 0.000 | 0.030 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.003 | 0.029 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 8.224 ± 0.069 | - | 9.020 ± 0.057 | 3.840 ± 0.052 | 8.931 ± 1.703 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.027 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.155 ± 0.036 | - | 0.153 ± 0.006 | 0.290 ± 0.010 | 0.303 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.015 ± 0.001 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.026 ± 0.001 | 0.038 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.164 ± 0.002 | - | 0.164 ± 0.002 | 0.098 ± 0.001 | 0.278 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.210 ± 0.001 | - | 0.219 ± 0.045 | 0.139 ± 0.002 | 0.359 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.557 ± 0.010 | - | 0.569 ± 0.010 | 0.411 ± 0.010 | 0.779 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.122 ± 0.008 | - | 0.151 ± 0.039 | 0.144 ± 0.004 | 0.206 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.001 | 0.016 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.011 ± 0.004 | - | 0.018 ± 0.002 | 0.030 ± 0.010 | 0.089 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.018 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.001 | 0.020 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.017 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.008 | 0.018 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.019 ± 0.001 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.019 ± 0.001 | 0.024 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.922 ± 0.006 | - | 1.003 ± 0.018 | 1.010 ± 0.012 | 1.037 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.070 ± 0.001 | - | 0.100 ± 0.020 | 0.081 ± 0.004 | 0.180 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.024 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.011 ± 0.000 | 0.021 ± 0.001 | 0.026 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.160 ± 0.021 | - | 0.150 ± 0.004 | 0.131 ± 0.002 | 0.276 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.199 ± 0.002 | - | 0.215 ± 0.040 | 0.172 ± 0.002 | 0.382 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.544 ± 0.023 | - | 5.993 ± 0.013 | 0.377 ± 0.014 | 0.850 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.119 ± 0.002 | - | 0.174 ± 0.067 | 0.203 ± 0.006 | 0.180 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.027 ± 0.004 | 0.016 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.011 ± 0.001 | - | 0.020 ± 0.002 | 0.044 ± 0.004 | 0.067 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.009 ± 0.000 | 0.026 ± 0.004 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.027 ± 0.001 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.027 ± 0.003 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.027 ± 0.004 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.030 ± 0.005 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.027 ± 0.003 | 0.020 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.027 ± 0.003 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.011 ± 0.000 | 0.030 ± 0.004 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.934 ± 0.016 | - | 23.998 ± 0.163 | 0.579 ± 0.018 | 0.793 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.030 ± 0.005 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.076 ± 0.001 | - | 0.094 ± 0.020 | 0.072 ± 0.002 | 0.213 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.030 ± 0.003 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.007 ± 0.000 | - | 0.013 ± 0.000 | 0.035 ± 0.004 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.830 ± 0.008 | - | 1.821 ± 0.008 | 1.552 ± 0.007 | 2.021 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 6.067 ± 0.017 | - | 6.141 ± 0.067 | 5.108 ± 0.057 | 5.079 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 17.928 ± 0.093 | - | 18.621 ± 0.213 | 15.929 ± 0.049 | 14.722 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 4.979 ± 0.017 | - | 5.130 ± 0.163 | 5.561 ± 0.039 | 4.224 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.014 ± 0.001 | 0.034 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.206 ± 0.012 | - | 0.210 ± 0.011 | 0.240 ± 0.033 | 0.276 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.020 ± 0.001 | 0.033 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.185 ± 0.012 | - | 0.190 ± 0.014 | 0.208 ± 0.020 | 0.225 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.021 ± 0.000 | - | 0.025 ± 0.005 | 0.033 ± 0.002 | 0.044 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.019 ± 0.002 | 0.032 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.020 ± 0.000 | - | 0.023 ± 0.001 | 0.030 ± 0.002 | 0.044 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.055 ± 0.000 | - | 0.060 ± 0.001 | 0.064 ± 0.008 | 0.073 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.019 ± 0.001 | 0.035 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.026 ± 0.000 | - | 0.029 ± 0.000 | 0.035 ± 0.001 | 0.058 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.073 ± 0.001 | - | 0.080 ± 0.005 | 0.079 ± 0.002 | 0.091 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 39.520 ± 0.052 | - | 41.139 ± 1.885 | 44.414 ± 0.204 | 33.269 ± 0.260 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.017 ± 0.001 | - | 0.019 ± 0.000 | 0.026 ± 0.003 | 0.050 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.555 ± 0.016 | - | 1.582 ± 0.015 | 1.659 ± 0.011 | 1.604 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.050 ± 0.000 | - | 0.053 ± 0.000 | 0.057 ± 0.003 | 0.069 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.142 ± 0.002 | - | 0.150 ± 0.003 | 0.142 ± 0.004 | 0.194 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.769 ± 0.028 | - | 1.780 ± 0.047 | 1.512 ± 0.012 | 2.053 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 6.052 ± 0.024 | - | 6.047 ± 0.026 | 5.057 ± 0.022 | 2.758 ± 0.062 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 17.806 ± 0.076 | - | 17.758 ± 0.108 | 18.641 ± 0.055 | 8.589 ± 1.499 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 9.086 ± 0.117 | - | 9.126 ± 0.263 | 4.735 ± 0.026 | 4.959 ± 0.116 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.005 ± 0.003 | - | 0.008 ± 0.000 | 0.019 ± 0.002 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.298 ± 0.027 | - | 0.314 ± 0.023 | 0.337 ± 0.016 | 0.262 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.009 ± 0.000 | - | 0.013 ± 0.000 | 0.020 ± 0.001 | 0.039 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.232 ± 0.028 | - | 0.240 ± 0.030 | 0.294 ± 0.019 | 0.224 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.021 ± 0.000 | - | 0.025 ± 0.000 | 0.032 ± 0.001 | 0.042 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.026 ± 0.004 | 0.032 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.021 ± 0.000 | - | 0.024 ± 0.000 | 0.030 ± 0.001 | 0.042 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.057 ± 0.003 | - | 0.065 ± 0.001 | 0.063 ± 0.001 | 0.090 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.028 ± 0.006 | 0.035 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.027 ± 0.000 | - | 0.031 ± 0.000 | 0.035 ± 0.000 | 0.049 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.081 ± 0.008 | - | 0.079 ± 0.008 | 0.079 ± 0.001 | 0.090 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 72.848 ± 0.273 | - | 74.573 ± 0.872 | 37.283 ± 0.358 | 97.966 ± 60.751 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.017 ± 0.000 | - | 0.020 ± 0.000 | 0.037 ± 0.004 | 0.041 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 2.282 ± 0.036 | - | 2.310 ± 0.022 | 2.337 ± 0.025 | 0.853 ± 0.111 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.053 ± 0.001 | - | 0.054 ± 0.001 | 0.056 ± 0.004 | 0.068 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.142 ± 0.003 | - | 0.149 ± 0.005 | 0.141 ± 0.002 | 0.167 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 1.350 ± 0.008 | - | 1.356 ± 0.011 | 1.101 ± 0.004 | 1.310 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 5.196 ± 0.071 | - | 5.206 ± 0.024 | 4.426 ± 0.278 | 4.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 15.149 ± 0.075 | - | 16.325 ± 0.048 | 13.851 ± 0.070 | 12.248 ± 0.127 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 2.300 ± 0.015 | - | 2.347 ± 0.032 | 2.456 ± 0.029 | 2.394 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.014 ± 0.002 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.135 ± 0.005 | - | 0.141 ± 0.003 | 0.164 ± 0.017 | 0.185 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.007 ± 0.000 | - | 0.010 ± 0.001 | 0.017 ± 0.001 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.114 ± 0.010 | - | 0.112 ± 0.005 | 0.135 ± 0.005 | 0.159 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.017 ± 0.000 | - | 0.021 ± 0.005 | 0.027 ± 0.002 | 0.039 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.016 ± 0.001 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.017 ± 0.000 | - | 0.020 ± 0.000 | 0.029 ± 0.010 | 0.039 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.043 ± 0.000 | - | 0.049 ± 0.000 | 0.053 ± 0.007 | 0.061 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.017 ± 0.001 | 0.029 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.023 ± 0.001 | - | 0.026 ± 0.000 | 0.030 ± 0.001 | 0.042 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.057 ± 0.001 | - | 0.064 ± 0.001 | 0.066 ± 0.004 | 0.075 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 18.480 ± 0.208 | - | 18.500 ± 0.055 | 19.422 ± 0.026 | 18.636 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.021 ± 0.002 | 0.033 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 1.052 ± 0.003 | - | 1.082 ± 0.001 | 1.158 ± 0.011 | 1.168 ± 0.107 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.042 ± 0.000 | - | 0.046 ± 0.001 | 0.048 ± 0.002 | 0.057 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.114 ± 0.001 | - | 0.125 ± 0.002 | 0.119 ± 0.008 | 0.139 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 1.331 ± 0.009 | - | 1.334 ± 0.025 | 1.053 ± 0.024 | 1.311 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 5.149 ± 0.023 | - | 5.153 ± 0.026 | 4.294 ± 0.021 | 2.237 ± 0.267 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 15.241 ± 0.239 | - | 15.160 ± 0.026 | 16.721 ± 0.106 | 8.836 ± 2.811 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 3.396 ± 0.030 | - | 3.433 ± 0.054 | 2.810 ± 0.016 | 3.089 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.019 ± 0.004 | 0.032 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.229 ± 0.013 | - | 0.240 ± 0.023 | 0.254 ± 0.017 | 0.189 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.019 ± 0.001 | 0.030 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.149 ± 0.011 | - | 0.172 ± 0.020 | 0.198 ± 0.008 | 0.166 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.018 ± 0.000 | - | 0.021 ± 0.000 | 0.028 ± 0.002 | 0.039 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.023 ± 0.005 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.018 ± 0.000 | - | 0.022 ± 0.000 | 0.027 ± 0.001 | 0.045 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.046 ± 0.000 | - | 0.049 ± 0.001 | 0.052 ± 0.004 | 0.061 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.025 ± 0.006 | 0.036 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.025 ± 0.000 | - | 0.026 ± 0.000 | 0.032 ± 0.004 | 0.043 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.061 ± 0.003 | - | 0.064 ± 0.000 | 0.066 ± 0.005 | 0.073 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 27.211 ± 0.121 | - | 27.345 ± 0.409 | 22.162 ± 0.040 | 28.767 ± 7.955 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.013 ± 0.000 | - | 0.016 ± 0.000 | 0.031 ± 0.004 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 1.780 ± 0.034 | - | 1.808 ± 0.012 | 1.755 ± 0.016 | 0.583 ± 0.118 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.043 ± 0.002 | - | 0.046 ± 0.001 | 0.047 ± 0.001 | 0.058 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.115 ± 0.007 | - | 0.133 ± 0.002 | 0.118 ± 0.003 | 0.155 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.467 ± 0.010 | - | unsupported | 0.355 ± 0.012 | 0.749 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 2.265 ± 0.006 | - | unsupported | 1.942 ± 0.003 | 2.692 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 5.524 ± 0.022 | - | unsupported | 5.305 ± 0.012 | 6.788 ± 0.077 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.986 ± 0.007 | - | unsupported | 1.150 ± 0.034 | 1.119 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.009 ± 0.001 | 0.016 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.048 ± 0.001 | - | unsupported | 0.061 ± 0.003 | 0.082 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | unsupported | 0.012 ± 0.001 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.039 ± 0.000 | - | unsupported | 0.053 ± 0.003 | 0.077 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | unsupported | 0.015 ± 0.001 | 0.028 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.011 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.008 ± 0.008 | - | unsupported | 0.016 ± 0.000 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.017 ± 0.000 | - | unsupported | 0.026 ± 0.001 | 0.045 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.010 ± 0.001 | 0.021 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.010 ± 0.008 | - | unsupported | 0.017 ± 0.001 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.023 ± 0.000 | - | unsupported | 0.031 ± 0.001 | 0.044 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 7.640 ± 0.028 | - | unsupported | 8.910 ± 0.085 | 8.564 ± 0.101 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | unsupported | 0.012 ± 0.001 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.385 ± 0.028 | - | unsupported | 0.432 ± 0.042 | 0.477 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.019 ± 0.000 | - | unsupported | 0.025 ± 0.006 | 0.085 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.044 ± 0.000 | - | unsupported | 0.051 ± 0.001 | 0.071 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.467 ± 0.022 | - | unsupported | 0.583 ± 0.013 | 0.749 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 2.303 ± 0.056 | - | unsupported | 1.937 ± 0.022 | 1.671 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 5.611 ± 0.043 | - | unsupported | 7.790 ± 0.081 | 3.528 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.205 ± 0.045 | - | unsupported | 1.452 ± 0.010 | 1.436 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | unsupported | 0.014 ± 0.001 | 0.022 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.048 ± 0.003 | - | unsupported | 0.064 ± 0.030 | 0.080 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | unsupported | 0.017 ± 0.001 | 0.024 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.040 ± 0.000 | - | unsupported | 0.079 ± 0.003 | 0.093 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.008 ± 0.000 | - | unsupported | 0.015 ± 0.001 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | unsupported | 0.016 ± 0.006 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.009 ± 0.000 | - | unsupported | 0.024 ± 0.004 | 0.024 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.018 ± 0.000 | - | unsupported | 0.026 ± 0.001 | 0.037 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.011 ± 0.000 | - | unsupported | 0.026 ± 0.009 | 0.028 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.024 ± 0.002 | - | unsupported | 0.031 ± 0.001 | 0.045 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 9.419 ± 0.094 | - | unsupported | 11.401 ± 0.089 | 11.123 ± 0.574 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | unsupported | 0.018 ± 0.004 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.373 ± 0.024 | - | unsupported | 0.421 ± 0.013 | 0.315 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.019 ± 0.000 | - | unsupported | 0.025 ± 0.000 | 0.037 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.044 ± 0.000 | - | unsupported | 0.051 ± 0.002 | 0.069 ± 0.003 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 1.062 ± 0.009 | - | unsupported | 0.948 ± 0.019 | 3.430 ± 0.038 | 2.569 ± 0.162 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.130 ± 0.002 | - | unsupported | 0.059 ± 0.003 | 0.108 ± 0.025 | 0.124 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.139 ± 0.003 | - | unsupported | 0.083 ± 0.003 | 0.182 ± 0.039 | 0.212 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.341 ± 0.010 | - | unsupported | 0.243 ± 0.012 | 0.501 ± 0.011 | 0.644 ± 0.190 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 1.056 ± 0.010 | - | unsupported | 0.493 ± 0.015 | 1.152 ± 0.052 | 2.581 ± 0.314 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.147 ± 0.002 | - | unsupported | 0.062 ± 0.002 | 0.112 ± 0.006 | 0.093 ± 0.015 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.142 ± 0.002 | - | unsupported | 0.079 ± 0.001 | 0.176 ± 0.018 | 0.177 ± 0.006 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.350 ± 0.012 | - | unsupported | 0.157 ± 0.019 | 0.544 ± 0.016 | 0.697 ± 0.021 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.326 ± 0.030 | - | unsupported | 0.371 ± 0.010 | 0.806 ± 0.008 | 0.413 ± 0.013 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.034 ± 0.000 | - | unsupported | 0.045 ± 0.001 | 0.188 ± 0.012 | 0.067 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.059 ± 0.001 | - | unsupported | 0.060 ± 0.003 | 0.226 ± 0.013 | 0.106 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.201 ± 0.005 | - | unsupported | 0.166 ± 0.004 | 0.379 ± 0.012 | 0.205 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.309 ± 0.010 | - | unsupported | 0.556 ± 0.013 | 0.857 ± 0.040 | 0.405 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.035 ± 0.000 | - | unsupported | 0.074 ± 0.002 | 0.204 ± 0.020 | 0.066 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.059 ± 0.000 | - | unsupported | 0.098 ± 0.001 | 0.240 ± 0.007 | 0.106 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.201 ± 0.002 | - | unsupported | 0.227 ± 0.012 | 0.428 ± 0.030 | 0.201 ± 0.007 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.194 ± 0.000 | - | 0.198 ± 0.011 | 0.193 ± 0.012 | 0.394 ± 0.013 | 0.249 ± 0.010 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.080 ± 0.001 | - | 0.082 ± 0.001 | 0.073 ± 0.010 | 0.077 ± 0.003 | 0.053 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.090 ± 0.001 | - | 0.093 ± 0.000 | 0.087 ± 0.001 | 0.101 ± 0.007 | 0.077 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.122 ± 0.000 | - | 0.124 ± 0.001 | 0.121 ± 0.004 | 0.173 ± 0.038 | 0.131 ± 0.005 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.193 ± 0.002 | - | 0.200 ± 0.014 | 0.255 ± 0.005 | 0.462 ± 0.064 | 0.256 ± 0.017 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.080 ± 0.000 | - | 0.116 ± 0.014 | 0.125 ± 0.003 | 0.081 ± 0.009 | 0.052 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.091 ± 0.000 | - | 0.120 ± 0.030 | 0.141 ± 0.011 | 0.106 ± 0.005 | 0.078 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.121 ± 0.000 | - | 0.126 ± 0.006 | 0.177 ± 0.002 | 0.195 ± 0.016 | 0.132 ± 0.002 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 44.564 ± 0.696 | - | 51.428 ± 0.446 | 60.414 ± 1.383 | 60.957 ± 1.114 | 49.352 ± 7.995 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 43.910 ± 0.238 | - | 53.359 ± 1.902 | 25.324 ± 0.181 | 49.979 ± 5.643 | 41.414 ± 5.584 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 25.622 ± 0.312 | - | 26.010 ± 0.761 | 25.944 ± 0.325 | 22.673 ± 0.134 | 22.602 ± 7.052 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 232.273 ± 57.207 | - | 323.011 ± 85.057 | 11.033 ± 0.105 | 14.147 ± 0.244 | 37.860 ± 19.804 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 7.504 ± 0.071 | - | 7.495 ± 0.084 | 7.960 ± 0.047 | 7.926 ± 0.050 | 8.076 ± 0.178 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 16.524 ± 0.075 | - | 16.479 ± 0.139 | 10.712 ± 0.085 | 11.988 ± 1.097 | 11.718 ± 0.116 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 8.722 ± 0.126 | - | 8.769 ± 0.080 | 8.797 ± 0.037 | 9.594 ± 0.068 | 9.158 ± 0.021 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 12.907 ± 0.102 | - | 12.943 ± 0.081 | 12.045 ± 0.117 | 14.225 ± 1.170 | 13.779 ± 0.030 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 21.190 ± 0.087 | - | 21.908 ± 0.269 | 19.093 ± 0.292 | 38.419 ± 0.505 | 14.530 ± 0.196 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 38.902 ± 0.201 | - | 39.165 ± 0.290 | 15.504 ± 0.142 | 32.983 ± 3.152 | 12.771 ± 0.864 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.495 ± 0.003 | 0.516 ± 0.006 | unsupported | 0.610 ± 0.005 | 0.739 ± 0.024 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 7.932 ± 0.091 | 8.783 ± 0.045 | unsupported | 8.875 ± 0.113 | 9.673 ± 0.108 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.805 ± 2.188 | 0.844 ± 0.016 | unsupported | 0.519 ± 0.057 | 1.226 ± 1.882 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 2277.531 ± 5.653 | 2309.016 ± 4.044 | unsupported | 5.597 ± 0.059 | 9.516 ± 0.377 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 44.829 ± 4.656 | - | 42.705 ± 1.014 | 43.250 ± 0.697 | 32.459 ± 0.257 | 28.103 ± 0.294 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 224.912 ± 14.659 | - | 278.682 ± 11.977 | 21.576 ± 0.313 | 20.346 ± 0.343 | 17.266 ± 0.553 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 17.891 ± 0.459 | 17.469 ± 0.356 | 9.101 ± 0.087 | 56.205 ± 0.141 | 9.678 ± 0.060 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 2286.157 ± 6.607 | 2287.986 ± 11.862 | 6.111 ± 0.131 | 53.751 ± 1.134 | 8.811 ± 0.164 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 25.523 ± 0.896 | - | 26.948 ± 1.216 | 32.711 ± 0.264 | 33.491 ± 0.332 | 33.444 ± 17.107 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 252.621 ± 23.982 | - | 250.162 ± 11.412 | 16.872 ± 0.616 | 24.746 ± 1.176 | 24.709 ± 5.689 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 1.489 ± 0.145 | - | 1.300 ± 0.195 | 1.387 ± 0.146 | 21.539 ± 1.130 | 2.874 ± 0.042 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 2.018 ± 0.077 | - | 10.989 ± 1.920 | 1.572 ± 0.111 | 3.445 ± 0.595 | 2.732 ± 0.026 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 20.426 ± 0.348 | - | 20.356 ± 0.444 | 21.488 ± 0.322 | 20.050 ± 0.311 | 19.677 ± 2.656 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 218.882 ± 15.156 | - | 212.680 ± 14.879 | 16.282 ± 0.254 | 21.171 ± 1.728 | 19.631 ± 0.278 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 20.205 ± 1.133 | - | 19.841 ± 0.399 | 20.843 ± 0.420 | 19.820 ± 0.061 | 19.375 ± 1.807 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 227.920 ± 14.503 | - | 215.019 ± 19.108 | 16.939 ± 0.584 | 21.187 ± 0.183 | 19.655 ± 2.800 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.177 ± 0.013 | 1.189 ± 0.022 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 22.916 ± 0.086 | 22.926 ± 0.372 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.613 ± 0.037 | 1.593 ± 0.015 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 23.093 ± 5.019 | 22.780 ± 0.279 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.084 ± 0.012 | 1.098 ± 0.014 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 21.908 ± 0.255 | 21.916 ± 0.193 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.503 ± 0.025 | 1.564 ± 0.042 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 21.645 ± 0.352 | 21.650 ± 0.195 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 25.596 ± 0.454 | - | 25.074 ± 0.324 | 25.571 ± 0.260 | 22.640 ± 0.318 | 21.196 ± 1.456 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 238.001 ± 24.789 | - | 321.018 ± 40.987 | 11.071 ± 0.106 | 14.094 ± 0.197 | 42.149 ± 17.113 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 61.623 ± 0.953 | 61.323 ± 0.679 | 61.617 ± 0.812 | 64.221 ± 1.253 | 66.253 ± 2.830 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 2339.546 ± 11.469 | 2330.272 ± 22.504 | 38.599 ± 0.311 | 52.511 ± 0.663 | 56.572 ± 9.057 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 35.177 ± 0.459 | - | 34.507 ± 0.454 | 30.427 ± 0.407 | 130.692 ± 4.898 | 30.198 ± 0.623 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 51.556 ± 4.737 | - | 53.295 ± 7.275 | 24.321 ± 1.259 | 122.930 ± 23.398 | 46.535 ± 15.188 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.141 ± 0.002 | 0.140 ± 0.003 | unsupported | 0.158 ± 0.031 | 0.038 ± 0.004 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.225 ± 0.005 | 2.219 ± 0.009 | unsupported | 2.446 ± 0.008 | 1.039 ± 0.613 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.074 ± 0.001 | 6.000 ± 0.016 | unsupported | 0.079 ± 0.004 | 0.124 ± 0.014 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.897 ± 0.011 | 23.999 ± 0.001 | unsupported | 1.032 ± 0.074 | 0.674 ± 0.338 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.078 ± 0.000 | 0.079 ± 0.000 | unsupported | 0.142 ± 0.019 | 0.231 ± 0.002 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.252 ± 0.008 | 1.241 ± 0.010 | unsupported | 1.747 ± 0.023 | 2.623 ± 0.030 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.037 ± 0.000 | 12.000 ± 0.609 | unsupported | 0.089 ± 0.007 | 0.158 ± 0.011 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.565 ± 0.003 | 11.999 ± 0.001 | unsupported | 0.731 ± 0.008 | 1.116 ± 0.052 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.197 ± 0.000 | 0.199 ± 0.001 | unsupported | 0.283 ± 0.008 | 0.184 ± 0.002 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 3.137 ± 0.004 | 3.139 ± 0.006 | unsupported | 4.383 ± 0.033 | 1.827 ± 0.050 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.076 ± 0.000 | 12.000 ± 0.000 | unsupported | 0.110 ± 0.004 | 0.147 ± 0.002 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 1.180 ± 0.003 | 11.999 ± 0.001 | unsupported | 1.511 ± 0.024 | 0.975 ± 0.021 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.063 ± 0.003 | 0.094 ± 0.004 | unsupported | 0.162 ± 0.001 | 0.182 ± 0.003 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.697 ± 0.024 | 0.694 ± 0.055 | unsupported | 1.735 ± 0.021 | 1.860 ± 0.023 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.089 ± 0.005 | 36.000 ± 0.000 | unsupported | 0.070 ± 0.001 | 0.183 ± 0.007 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.482 ± 0.082 | 95.999 ± 1.125 | unsupported | 0.363 ± 0.007 | 1.182 ± 0.127 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.130 ± 0.005 | 0.151 ± 0.011 | unsupported | 0.128 ± 0.004 | 0.220 ± 0.003 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.663 ± 0.014 | 1.744 ± 0.042 | unsupported | 2.042 ± 0.037 | 2.324 ± 0.089 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.146 ± 0.002 | 43.875 ± 0.203 | unsupported | 0.069 ± 0.005 | 0.205 ± 0.002 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 1.030 ± 0.134 | 120.000 ± 0.003 | unsupported | 0.531 ± 0.013 | 1.392 ± 0.039 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.293 ± 0.004 | 0.333 ± 0.002 | unsupported | unsupported | 0.242 ± 0.004 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 4.352 ± 0.023 | 4.457 ± 0.094 | unsupported | unsupported | 3.183 ± 0.051 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.278 ± 0.008 | 51.640 ± 8.138 | unsupported | unsupported | 0.244 ± 0.015 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 2.080 ± 0.071 | 144.000 ± 0.001 | unsupported | unsupported | 1.855 ± 0.027 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.012 ± 0.001 | 0.020 ± 0.001 | unsupported | 0.013 ± 0.000 | 0.050 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.108 ± 0.003 | 0.116 ± 0.001 | unsupported | 0.095 ± 0.002 | 0.223 ± 0.002 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.018 ± 0.000 | 2.898 ± 3.284 | unsupported | 0.015 ± 0.001 | 0.038 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.132 ± 0.021 | 23.997 ± 1.498 | unsupported | 0.098 ± 0.008 | 0.154 ± 0.009 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.037 ± 0.006 | 0.054 ± 0.004 | unsupported | 0.106 ± 0.003 | 0.168 ± 0.003 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.349 ± 0.008 | 0.409 ± 0.038 | unsupported | 1.380 ± 0.038 | 1.654 ± 0.024 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.052 ± 0.003 | 18.000 ± 0.000 | unsupported | 0.103 ± 0.006 | 0.153 ± 0.001 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.245 ± 0.025 | 48.000 ± 0.003 | unsupported | 0.291 ± 0.083 | 1.052 ± 0.139 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.054 ± 0.000 | 0.055 ± 0.001 | unsupported | 0.117 ± 0.003 | 0.197 ± 0.003 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.859 ± 0.004 | 0.865 ± 0.008 | unsupported | 1.801 ± 0.028 | 1.733 ± 0.032 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.027 ± 0.000 | 12.000 ± 0.000 | unsupported | 0.098 ± 0.003 | 0.138 ± 0.002 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.393 ± 0.003 | 11.999 ± 0.002 | unsupported | 0.691 ± 0.007 | 0.932 ± 0.028 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.056 ± 0.000 | 0.057 ± 0.001 | unsupported | 0.123 ± 0.003 | 0.196 ± 0.003 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.895 ± 0.007 | 0.898 ± 0.006 | unsupported | 1.859 ± 0.013 | 1.743 ± 0.015 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.028 ± 0.000 | 13.265 ± 1.640 | unsupported | 0.100 ± 0.002 | 0.137 ± 0.015 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.410 ± 0.002 | 11.999 ± 0.000 | unsupported | 0.736 ± 0.012 | 0.944 ± 0.039 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.129 ± 0.003 | 0.156 ± 0.006 | unsupported | 0.137 ± 0.001 | 0.224 ± 0.009 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.657 ± 0.020 | 1.745 ± 0.077 | unsupported | 1.231 ± 0.005 | 2.204 ± 0.017 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.156 ± 0.003 | 43.874 ± 0.281 | unsupported | 0.063 ± 0.002 | 0.203 ± 0.003 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 0.991 ± 0.158 | 120.000 ± 0.376 | unsupported | 0.486 ± 0.005 | 1.322 ± 0.086 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.183 ± 0.001 | 0.184 ± 0.001 | unsupported | 0.274 ± 0.000 | 0.317 ± 0.004 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 2.915 ± 0.007 | 2.912 ± 0.013 | unsupported | 4.296 ± 0.041 | 4.023 ± 0.223 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.073 ± 0.000 | 12.187 ± 2.812 | unsupported | 0.202 ± 0.005 | 0.203 ± 0.013 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 1.125 ± 0.005 | 11.999 ± 0.001 | unsupported | 1.559 ± 0.048 | 2.062 ± 0.013 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.743 ± 0.019 | 1.353 ± 0.015 | unsupported | 0.268 ± 0.003 | 0.223 ± 0.070 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 19.010 ± 1.044 | 29.488 ± 1.148 | unsupported | 6.120 ± 0.256 | 2.293 ± 0.308 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.851 ± 0.035 | 92.827 ± 0.243 | unsupported | 0.258 ± 0.001 | 0.178 ± 0.006 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 8.794 ± 0.196 | 156.250 ± 3.000 | unsupported | 2.360 ± 0.038 | 1.215 ± 0.016 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 30.453 ± 1.714 | - | unsupported | 30.023 ± 0.629 | - | 32.586 ± 1.250 | 41.610 ± 0.589 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 24.909 ± 4.563 | - | unsupported | 22.319 ± 2.814 | - | 34.046 ± 3.201 | 22.725 ± 0.596 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 22.943 ± 0.213 | - | unsupported | 27.296 ± 1.045 | - | 21.811 ± 2.012 | 23.629 ± 1.294 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 16.970 ± 0.761 | - | unsupported | 15.705 ± 0.991 | - | 23.618 ± 1.550 | 15.984 ± 0.538 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 14.856 ± 0.655 | - | unsupported | 24.569 ± 1.561 | - | 13.601 ± 0.216 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 16.559 ± 0.426 | - | unsupported | 15.504 ± 0.345 | - | 14.676 ± 0.373 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 29.080 ± 1.567 | - | unsupported | 30.551 ± 0.626 | - | 32.562 ± 0.472 | 40.477 ± 0.754 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 23.310 ± 1.638 | - | unsupported | 22.286 ± 0.824 | - | 32.070 ± 0.937 | 24.064 ± 2.712 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 50.101 ± 0.407 | - | unsupported | 49.947 ± 0.758 | - | 48.575 ± 0.086 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 54.262 ± 0.496 | - | unsupported | 25.218 ± 0.570 | - | 23.506 ± 0.221 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 49.377 ± 0.244 | - | unsupported | 49.676 ± 0.408 | - | 48.264 ± 0.312 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 54.162 ± 0.555 | - | unsupported | 25.369 ± 0.761 | - | 23.414 ± 0.187 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 30.403 ± 0.517 | - | unsupported | 30.341 ± 0.839 | - | 32.022 ± 3.252 | 39.949 ± 0.853 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 23.395 ± 1.496 | - | unsupported | 22.545 ± 1.022 | - | 34.447 ± 0.576 | 22.648 ± 0.952 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 23.681 ± 0.730 | - | unsupported | 24.436 ± 0.572 | - | 24.957 ± 0.983 | 30.139 ± 1.005 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 15.957 ± 0.841 | - | unsupported | 15.492 ± 0.261 | - | 27.667 ± 0.346 | 16.523 ± 1.031 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.746 ± 0.003 | unsupported | unsupported | 0.894 ± 0.003 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 11.241 ± 0.118 | unsupported | unsupported | 12.876 ± 0.189 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 1.474 ± 0.006 | unsupported | unsupported | 0.622 ± 0.006 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 195.651 ± 3.232 | unsupported | unsupported | 6.652 ± 0.084 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 29.846 ± 1.129 | - | unsupported | 30.470 ± 1.817 | - | 33.208 ± 4.655 | 40.932 ± 1.264 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 23.721 ± 0.943 | - | unsupported | 25.733 ± 3.780 | - | 34.060 ± 0.743 | 25.413 ± 4.580 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 107.266 ± 3.947 | - | 0.002 ± 0.000 | 109.169 ± 2.138 | 106.180 ± 1.087 | 133.701 ± 4.882 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 42.994 ± 1.035 | - | 0.002 ± 0.000 | 35.364 ± 1.221 | 55.401 ± 1.724 | 132.728 ± 2.531 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 67.399 ± 2.122 | - | 209.217 ± 6.356 | 69.599 ± 2.029 | 70.163 ± 1.108 | 77.156 ± 1.930 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 22.934 ± 0.244 | - | 259.537 ± 1.782 | 23.163 ± 0.298 | 36.238 ± 0.437 | 77.933 ± 3.311 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 215.254 ± 4.571 | - | 215.579 ± 3.699 | 215.740 ± 4.243 | 214.121 ± 1.685 | 250.214 ± 6.572 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 71.095 ± 0.995 | - | 226.642 ± 2.973 | 70.770 ± 1.289 | 109.875 ± 0.779 | 249.877 ± 3.742 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 67.898 ± 0.825 | - | 68.017 ± 1.445 | 74.861 ± 4.690 | 68.806 ± 2.929 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 41.813 ± 0.339 | - | 70.945 ± 1.881 | 24.701 ± 0.274 | 36.495 ± 1.358 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 135.720 ± 3.669 | - | 0.002 ± 0.000 | 124.930 ± 3.299 | 120.488 ± 2.827 | 154.581 ± 2.966 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 41.094 ± 0.601 | - | 0.002 ± 0.000 | 40.199 ± 0.536 | 50.733 ± 8.024 | 157.049 ± 7.268 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 85.330 ± 3.085 | - | 0.002 ± 0.000 | 77.965 ± 0.448 | 84.270 ± 1.285 | 216.912 ± 11.578 | 129.819 ± 1.802 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 29.160 ± 0.525 | - | 0.002 ± 0.000 | 79.874 ± 2.169 | 43.256 ± 0.816 | 213.076 ± 2.562 | 52.752 ± 1.539 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 59.660 ± 1.256 | - | 60.048 ± 1.492 | 65.302 ± 0.458 | 64.452 ± 0.820 | 82.476 ± 0.732 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 21.487 ± 0.321 | - | 71.096 ± 0.523 | 24.198 ± 0.637 | 32.880 ± 1.682 | 81.006 ± 2.040 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 58.853 ± 1.163 | - | 58.863 ± 0.804 | 65.041 ± 1.110 | 65.008 ± 1.045 | 81.739 ± 3.681 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 20.980 ± 0.848 | - | 71.041 ± 0.669 | 22.584 ± 0.505 | 33.035 ± 1.103 | 81.553 ± 2.893 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.001 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.001 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 8.277738 | 517358.62 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 7.382798 | 461424.88 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 7.938152 | 496134.50 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 6.203376 | 387711.00 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 6.436521 | 402282.56 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 831.994616 | 51999663.50 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 19.162108 | 1197631.75 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 18.050058 | 1128128.62 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 8.557810 | 534863.12 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 383.994668 | 23999666.75 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 11.821699 | 738856.19 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 9.763008 | 610188.00 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 7.917500 | 494843.75 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 8.251598 | 515724.88 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 154.761124 | 9672570.25 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 142.001091 | 8875068.19 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 126.909839 | 7931864.94 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 140.531841 | 8783240.06 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 19.622462 | 1226403.88 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 8.297516 | 518594.75 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 12.885227 | 805326.69 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 13.501545 | 843846.56 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 152.263499 | 9516468.69 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 89.544632 | 5596539.50 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 36440.501050 | 2277531315.62 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 36944.254094 | 2309015880.88 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 18.828633 | 1176789.56 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 19.020003 | 1188750.19 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 366.660501 | 22916281.31 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 366.811374 | 22925710.88 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 25.806540 | 1612908.75 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 25.493321 | 1593332.56 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 369.489593 | 23093099.56 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 364.481168 | 22780073.00 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 17.338014 | 1083625.88 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 17.560211 | 1097513.19 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 350.524311 | 21907769.44 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 350.663647 | 21916477.94 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 24.050630 | 1503164.38 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 25.022207 | 1563887.94 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 346.320213 | 21645013.31 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 346.397694 | 21649855.88 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.420605 | 37821.95 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 10.121382 | 158146.59 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 9.012087 | 140813.86 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.967722 | 140120.66 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 8.313402 | 1039175.25 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 19.571775 | 2446471.88 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 17.802094 | 2225261.75 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 17.752340 | 2219042.50 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.961015 | 124390.86 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.075219 | 79300.30 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.718820 | 73731.56 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 383.976397 | 5999631.20 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.393960 | 674245.00 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.255196 | 1031899.50 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.176859 | 897107.38 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 191.993464 | 23999183.00 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.803269 | 231301.08 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 9.086659 | 141979.05 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.007414 | 78240.84 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.045195 | 78831.17 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 20.985589 | 2623198.62 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 13.977085 | 1747135.62 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.016492 | 1252061.50 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 9.924139 | 1240517.38 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.143308 | 158489.19 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.687139 | 88861.55 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.396040 | 37438.12 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 768.000519 | 12000008.11 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.929936 | 1116242.00 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.850205 | 731275.62 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.517440 | 564680.00 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.991021 | 11998877.62 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.759274 | 183738.66 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 18.136798 | 283387.47 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 12.590235 | 196722.42 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 12.749665 | 199213.52 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.612230 | 1826528.75 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 35.067089 | 4383386.12 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 25.096894 | 3137111.75 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 25.114858 | 3139357.25 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.377100 | 146517.19 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 7.067373 | 110427.70 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.886035 | 76344.30 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.979259 | 11999675.92 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.798160 | 974770.00 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.087066 | 1510883.25 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.440824 | 1180103.00 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.992433 | 11999054.12 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.650789 | 182043.58 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 10.383173 | 162237.08 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.019369 | 62802.64 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.993857 | 93654.02 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.877048 | 1859631.00 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 13.879491 | 1734936.38 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.574539 | 696817.38 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.549012 | 693626.50 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.707499 | 182929.67 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.454472 | 69601.12 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.678124 | 88720.69 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2303.976969 | 35999640.14 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.456626 | 1182078.25 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.901202 | 362650.25 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.854345 | 481793.12 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 767.991264 | 95998908.00 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.097864 | 220279.12 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.181929 | 127842.64 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.294829 | 129606.70 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.671363 | 151115.05 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.595480 | 2324435.00 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 16.337725 | 2042215.62 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.301992 | 1662749.00 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 13.953396 | 1744174.50 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.088785 | 204512.27 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.411161 | 68924.39 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 9.333347 | 145833.55 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2808.008446 | 43875131.97 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.136355 | 1392044.38 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.250719 | 531339.88 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.239504 | 1029938.00 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 959.996231 | 119999528.88 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 15.505147 | 242267.92 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 18.720208 | 292503.25 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 21.315372 | 333052.69 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 25.466383 | 3183297.88 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 34.813322 | 4351665.25 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 35.657227 | 4457153.38 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 15.604104 | 243814.12 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 17.804453 | 278194.58 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3304.954035 | 51639906.80 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 14.836532 | 1854566.50 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 16.639161 | 2079895.13 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1151.998802 | 143999850.25 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 3.220728 | 50323.88 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.853303 | 13332.86 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.777830 | 12153.59 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.281487 | 20023.23 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 1.785083 | 223135.38 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.756289 | 94536.12 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.860326 | 107540.75 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.925759 | 115719.88 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.459924 | 38436.31 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.966496 | 15101.50 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.132276 | 17691.81 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 185.482128 | 2898158.25 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.232877 | 154109.62 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.783212 | 97901.50 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.059081 | 132385.12 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 191.977583 | 23997197.88 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.773692 | 168338.94 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.760379 | 105630.92 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.337651 | 36525.80 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.443747 | 53808.55 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.234703 | 1654337.88 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.038368 | 1379796.00 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.793166 | 349145.75 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.274239 | 409279.88 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.794592 | 153040.50 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.583784 | 102871.62 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.358567 | 52477.61 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1151.969528 | 17999523.88 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.415911 | 1051988.88 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.327264 | 290908.00 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.960385 | 245048.12 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 383.998208 | 47999776.00 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.580425 | 196569.14 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.472636 | 116759.94 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.471410 | 54240.78 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.495184 | 54612.25 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.860067 | 1732508.38 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.409445 | 1801180.62 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.869782 | 858722.75 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.922741 | 865342.62 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.845271 | 138207.36 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.280023 | 98125.36 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.716413 | 26818.95 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.981386 | 11999709.16 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.458641 | 932330.12 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.527910 | 690988.75 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.147100 | 393387.50 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.995099 | 11999387.38 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.550949 | 196108.58 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.891222 | 123300.34 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.594350 | 56161.72 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.642982 | 56921.59 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.945488 | 1743186.00 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.871082 | 1858885.25 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.157482 | 894685.25 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.182409 | 897801.12 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.763835 | 136934.92 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.415508 | 100242.31 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.789330 | 27958.28 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 848.978683 | 13265291.92 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.549352 | 943669.00 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.889940 | 736242.50 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.283046 | 410380.75 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.993546 | 11999193.25 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.338115 | 224033.05 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.783979 | 137249.67 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.282164 | 129408.81 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.966177 | 155721.52 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 17.630078 | 2203759.75 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.850699 | 1231337.38 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.253921 | 1656740.12 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 13.961230 | 1745153.75 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.014258 | 203347.78 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.038150 | 63096.09 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 9.991012 | 156109.56 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2807.966117 | 43874470.58 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.574939 | 1321867.38 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.886494 | 485811.75 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.931868 | 991483.50 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 959.996531 | 119999566.38 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 20.262300 | 316598.44 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.557490 | 274335.78 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 11.693442 | 182710.03 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 11.756720 | 183698.75 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 32.184218 | 4023027.25 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 34.369830 | 4296228.75 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 23.319185 | 2914898.12 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 23.294910 | 2911863.75 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.994838 | 203044.34 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 12.939821 | 202184.70 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.677483 | 73085.67 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 779.996535 | 12187445.86 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 16.492082 | 2061510.25 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.471900 | 1558987.50 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.996417 | 1124552.12 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.991922 | 11998990.25 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.265160 | 222893.12 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.134981 | 267734.08 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 47.569238 | 743269.34 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 86.611623 | 1353306.61 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.345982 | 2293247.75 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 48.963402 | 6120425.25 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 152.078064 | 19009758.00 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 235.901446 | 29487680.75 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.412127 | 178314.48 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 16.511915 | 257998.67 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 54.471179 | 851112.17 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5940.914817 | 92826794.02 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.719010 | 1214876.25 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 18.881916 | 2360239.50 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 70.353819 | 8794227.38 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1250.001078 | 156250134.75 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 14.305241 | 894077.56 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 11.937675 | 746104.69 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 206.023650 | 12876478.12 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 179.853632 | 11240852.00 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 9.948067 | 621754.19 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 23.589061 | 1474316.31 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 106.427043 | 6651690.19 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 3130.421837 | 195651364.81 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.024666 | 1541.62 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002545 | 159.06 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.024477 | 1529.81 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002515 | 157.19 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000180 | 11.25 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.039584 | 2474.00 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.006933 | 433.31 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000130 | 8.12 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.021981 | 1373.81 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.007324 | 457.75 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000190 | 11.88 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.022852 | 1428.25 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.005761 | 360.06 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000150 | 9.38 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.022783 | 1423.94 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.005751 | 359.44 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000100 | 6.25 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.021069 | 1316.81 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003196 | 199.75 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000101 | 6.31 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.020709 | 1294.31 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003747 | 234.19 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `jax-cpu` is 20.1x faster than the slowest successful cell (`tenferro-trace`, 59.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `julia-base` is 13.9x faster than the slowest successful cell (`tenferro-trace`, 59.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 58.5x faster than the slowest successful cell (`tenferro-trace`, 59.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 18.5x faster than the slowest successful cell (`tenferro-trace`, 127.951 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `julia-base` is 16.9x faster than the slowest successful cell (`tenferro-trace`, 127.951 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 22.3x faster than the slowest successful cell (`tenferro-trace`, 127.951 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 29.2x faster than the slowest successful cell (`tenferro-trace`, 140.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `julia-base` is 30.2x faster than the slowest successful cell (`tenferro-trace`, 140.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 38.5x faster than the slowest successful cell (`tenferro-trace`, 140.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/masked_log_softmax_axis1` (f64, threads=4, shape=`1024x64`): `jax-cpu` is 134.1x faster than the slowest successful cell (`tenferro-eager`, 52.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/masked_log_softmax_axis1` (f64, threads=4, shape=`1024x64`): `tenferro-direct` is 129.3x faster than the slowest successful cell (`tenferro-eager`, 52.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 18.6x faster than the slowest successful cell (`pytorch-cpu`, 24.406 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 24.406 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 24.406 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 15.2x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 17.5x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 29.8x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 17.4x faster than the slowest successful cell (`tenferro-trace`, 23.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 12.4x faster than the slowest successful cell (`tenferro-trace`, 23.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`tenferro-trace`, 23.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 18.5x faster than the slowest successful cell (`tenferro-trace`, 23.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 23.725 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 23.725 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 23.725 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 17.9x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 15.7x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 29.9x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 21.3x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 18.9x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 24.7x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 37.7x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 13.4x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 16.3x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 66.3x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 26.1x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 21.1x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 18.3x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 25.9x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 40.5x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 13.2x faster than the slowest successful cell (`tenferro-trace`, 23.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 17.1x faster than the slowest successful cell (`tenferro-trace`, 23.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 60.1x faster than the slowest successful cell (`tenferro-trace`, 23.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 32.6x faster than the slowest successful cell (`tenferro-trace`, 23.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_squares_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 44.9x faster than the slowest successful cell (`tenferro-eager`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `jax-cpu` is 11.9x faster than the slowest successful cell (`tenferro-trace`, 11.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `pytorch-cpu` is 24.0x faster than the slowest successful cell (`tenferro-trace`, 11.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `tenferro-direct` is 25.6x faster than the slowest successful cell (`tenferro-trace`, 11.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `jax-cpu` is 14.7x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `pytorch-cpu` is 26.9x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `tenferro-direct` is 23.6x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`julia-base`, 8.279 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `jax-cpu` is 120.8x faster than the slowest successful cell (`tenferro-trace`, 47.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `julia-base` is 41.5x faster than the slowest successful cell (`tenferro-trace`, 47.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 137.0x faster than the slowest successful cell (`tenferro-trace`, 47.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `tenferro-direct` is 112.1x faster than the slowest successful cell (`tenferro-trace`, 47.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `jax-cpu` is 11.2x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 20.4x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 26.6x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `jax-cpu` is 16.3x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 19.7x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 24.7x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `jax-cpu` is 66.1x faster than the slowest successful cell (`tenferro-trace`, 35.975 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `julia-base` is 24.4x faster than the slowest successful cell (`tenferro-trace`, 35.975 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 54.7x faster than the slowest successful cell (`tenferro-trace`, 35.975 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `tenferro-direct` is 29.1x faster than the slowest successful cell (`tenferro-trace`, 35.975 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 4882.7x faster than the slowest successful cell (`julia-base`, 9.346 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-direct` is 11.9x faster than the slowest successful cell (`julia-base`, 9.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 4269.8x faster than the slowest successful cell (`julia-base`, 9.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 12.4x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 13.7x faster than the slowest successful cell (`jax-cpu`, 0.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x8x8`): `tenferro-direct` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`3x2x2`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.011 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x2x2`): `tenferro-direct` is 12.3x faster than the slowest successful cell (`jax-cpu`, 0.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x4x4`): `tenferro-direct` is 12.1x faster than the slowest successful cell (`jax-cpu`, 0.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 19.5x faster than the slowest successful cell (`jax-cpu`, 0.037 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 14.7x faster than the slowest successful cell (`jax-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=1, shape=`1x2x2`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.022 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 16.4x faster than the slowest successful cell (`jax-cpu`, 0.064 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.064 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 15.9x faster than the slowest successful cell (`jax-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.020 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `pytorch-cpu` is 15.9x faster than the slowest successful cell (`tenferro-trace`, 5.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `tenferro-direct` is 11.0x faster than the slowest successful cell (`tenferro-trace`, 5.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x4x4,rhs=1`): `tenferro-direct` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.026 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`3x2x2,rhs=1`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.027 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `jax-cpu` is 30.3x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `pytorch-cpu` is 41.4x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `tenferro-direct` is 25.7x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 22.8x faster than the slowest successful cell (`tenferro-trace`, 323.011 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 29.3x faster than the slowest successful cell (`tenferro-trace`, 323.011 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`768x384`): `jax-cpu` is 242.6x faster than the slowest successful cell (`tenferro-eager`, 2309.016 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 412.6x faster than the slowest successful cell (`tenferro-eager`, 2309.016 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 13.7x faster than the slowest successful cell (`tenferro-trace`, 278.682 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `julia-base` is 16.1x faster than the slowest successful cell (`tenferro-trace`, 278.682 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 12.9x faster than the slowest successful cell (`tenferro-trace`, 278.682 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `jax-cpu` is 42.6x faster than the slowest successful cell (`tenferro-trace`, 2287.986 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `julia-base` is 259.7x faster than the slowest successful cell (`tenferro-trace`, 2287.986 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 374.4x faster than the slowest successful cell (`tenferro-trace`, 2287.986 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 10.2x faster than the slowest successful cell (`tenferro-direct`, 252.621 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `julia-base` is 10.2x faster than the slowest successful cell (`tenferro-direct`, 252.621 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 15.0x faster than the slowest successful cell (`tenferro-direct`, 252.621 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `pytorch-cpu` is 15.5x faster than the slowest successful cell (`jax-cpu`, 21.539 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 14.5x faster than the slowest successful cell (`jax-cpu`, 21.539 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 16.6x faster than the slowest successful cell (`jax-cpu`, 21.539 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `jax-cpu` is 10.3x faster than the slowest successful cell (`tenferro-direct`, 218.882 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `julia-base` is 11.1x faster than the slowest successful cell (`tenferro-direct`, 218.882 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 13.4x faster than the slowest successful cell (`tenferro-direct`, 218.882 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `jax-cpu` is 10.8x faster than the slowest successful cell (`tenferro-direct`, 227.920 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `julia-base` is 11.6x faster than the slowest successful cell (`tenferro-direct`, 227.920 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 13.5x faster than the slowest successful cell (`tenferro-direct`, 227.920 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 22.8x faster than the slowest successful cell (`tenferro-trace`, 321.018 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 29.0x faster than the slowest successful cell (`tenferro-trace`, 321.018 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 44.6x faster than the slowest successful cell (`tenferro-eager`, 2339.546 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `julia-base` is 41.4x faster than the slowest successful cell (`tenferro-eager`, 2339.546 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 60.6x faster than the slowest successful cell (`tenferro-eager`, 2339.546 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 48.2x faster than the slowest successful cell (`tenferro-eager`, 6.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 75.7x faster than the slowest successful cell (`tenferro-eager`, 6.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 81.4x faster than the slowest successful cell (`tenferro-eager`, 6.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 35.6x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 23.3x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 26.8x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 75.7x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 135.0x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 320.5x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 10.7x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 16.4x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 21.2x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 81.9x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 108.7x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 157.2x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 12.3x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 196.8x faster than the slowest successful cell (`tenferro-eager`, 36.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 517.2x faster than the slowest successful cell (`tenferro-eager`, 36.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 405.8x faster than the slowest successful cell (`tenferro-eager`, 36.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 81.2x faster than the slowest successful cell (`tenferro-eager`, 95.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 264.7x faster than the slowest successful cell (`tenferro-eager`, 95.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 199.3x faster than the slowest successful cell (`tenferro-eager`, 95.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 214.5x faster than the slowest successful cell (`tenferro-eager`, 43.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 636.6x faster than the slowest successful cell (`tenferro-eager`, 43.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 300.9x faster than the slowest successful cell (`tenferro-eager`, 43.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 86.2x faster than the slowest successful cell (`tenferro-eager`, 120.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 225.8x faster than the slowest successful cell (`tenferro-eager`, 120.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 116.5x faster than the slowest successful cell (`tenferro-eager`, 120.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 211.8x faster than the slowest successful cell (`tenferro-eager`, 51.640 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 185.6x faster than the slowest successful cell (`tenferro-eager`, 51.640 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 77.6x faster than the slowest successful cell (`tenferro-eager`, 144.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 69.2x faster than the slowest successful cell (`tenferro-eager`, 144.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 75.4x faster than the slowest successful cell (`tenferro-eager`, 2.898 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 191.9x faster than the slowest successful cell (`tenferro-eager`, 2.898 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 163.8x faster than the slowest successful cell (`tenferro-eager`, 2.898 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 155.7x faster than the slowest successful cell (`tenferro-eager`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 245.1x faster than the slowest successful cell (`tenferro-eager`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 181.3x faster than the slowest successful cell (`tenferro-eager`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 117.6x faster than the slowest successful cell (`tenferro-eager`, 18.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 175.0x faster than the slowest successful cell (`tenferro-eager`, 18.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 343.0x faster than the slowest successful cell (`tenferro-eager`, 18.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 45.6x faster than the slowest successful cell (`tenferro-eager`, 48.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 165.0x faster than the slowest successful cell (`tenferro-eager`, 48.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 195.9x faster than the slowest successful cell (`tenferro-eager`, 48.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 86.8x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 122.3x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 447.4x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 12.9x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 17.4x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 30.5x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 96.9x faster than the slowest successful cell (`tenferro-eager`, 13.265 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 132.3x faster than the slowest successful cell (`tenferro-eager`, 13.265 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 474.5x faster than the slowest successful cell (`tenferro-eager`, 13.265 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 12.7x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 16.3x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 29.2x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 215.8x faster than the slowest successful cell (`tenferro-eager`, 43.874 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 695.4x faster than the slowest successful cell (`tenferro-eager`, 43.874 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 281.0x faster than the slowest successful cell (`tenferro-eager`, 43.874 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 90.8x faster than the slowest successful cell (`tenferro-eager`, 120.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 247.0x faster than the slowest successful cell (`tenferro-eager`, 120.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 121.0x faster than the slowest successful cell (`tenferro-eager`, 120.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 60.0x faster than the slowest successful cell (`tenferro-eager`, 12.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 60.3x faster than the slowest successful cell (`tenferro-eager`, 12.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 166.8x faster than the slowest successful cell (`tenferro-eager`, 12.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 12.9x faster than the slowest successful cell (`tenferro-eager`, 29.488 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 520.6x faster than the slowest successful cell (`tenferro-eager`, 92.827 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 359.8x faster than the slowest successful cell (`tenferro-eager`, 92.827 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 109.1x faster than the slowest successful cell (`tenferro-eager`, 92.827 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 128.6x faster than the slowest successful cell (`tenferro-eager`, 156.250 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 66.2x faster than the slowest successful cell (`tenferro-eager`, 156.250 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 17.8x faster than the slowest successful cell (`tenferro-eager`, 156.250 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/output_reuse/solve_read_into` (f64, threads=4, shape=`768x768,rhs=16`): `pytorch-cpu` is 29.4x faster than the slowest successful cell (`tenferro-direct`, 195.651 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 73300.8x faster than the slowest successful cell (`julia-base`, 133.701 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 74399.0x faster than the slowest successful cell (`julia-base`, 132.728 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/cast_f64_f32` (f64->f32, threads=4, shape=`33554432`): `pytorch-cpu` is 11.2x faster than the slowest successful cell (`tenferro-trace`, 259.537 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/cast_f64_f32` (f64->f32, threads=4, shape=`33554432`): `tenferro-direct` is 11.3x faster than the slowest successful cell (`tenferro-trace`, 259.537 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 81187.6x faster than the slowest successful cell (`julia-base`, 154.581 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 77593.6x faster than the slowest successful cell (`julia-base`, 157.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 123667.1x faster than the slowest successful cell (`julia-base`, 216.912 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 124387.7x faster than the slowest successful cell (`julia-base`, 213.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 224.9x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 171.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 119.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 158.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 219.5x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 215.7x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 8761.2 GB/s at 0.002 ms. The conservative physical lower bound is 0.168 ms, so the cell is 87.6x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 7895.0 GB/s at 0.002 ms. The conservative physical lower bound is 0.042 ms, so the cell is 19.7x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 147168.6 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1471.7x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 150468.3 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 376.2x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 140985.0 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1409.9x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 132626.2 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 331.6x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 76520.9 GB/s at 0.002 ms. The conservative physical lower bound is 1.342 ms, so the cell is 765.2x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 78352.4 GB/s at 0.002 ms. The conservative physical lower bound is 0.336 ms, so the cell is 195.9x faster than the bound.

# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/amd-cpu/cpu/public_api/20261113_191625/run.yaml`
- Timestamp: `2026-11-13T19:16:25.069730+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/amd-cpu/cpu/public_api/20261113_191625`.

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

- CSV: `data/results/amd-cpu/cpu/public_api/20261113_191625/cpu_public_api_t1_20261113_191625.csv`
- CSV: `data/results/amd-cpu/cpu/public_api/20261113_191625/cpu_public_api_t4_20261113_191625.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 4.193 ± 0.026 | - | 4.078 ± 0.084 | 4.461 ± 0.122 | 3.810 ± 0.057 | 3.757 ± 2.246 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 4.530 ± 0.045 | - | 4.364 ± 0.048 | 2.266 ± 0.038 | 3.217 ± 0.805 | 3.378 ± 0.445 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 122.083 ± 2.689 | - | 119.942 ± 1.976 | 120.466 ± 4.127 | 124.676 ± 0.507 | 142.481 ± 2.220 | 145.904 ± 1.597 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 41.011 ± 0.338 | - | 142.845 ± 3.959 | 40.199 ± 0.841 | 64.367 ± 1.214 | 141.169 ± 3.018 | 63.096 ± 0.672 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 106.020 ± 1.253 | - | 111.853 ± 4.301 | 92.653 ± 0.654 | 91.695 ± 1.652 | 128.274 ± 1.346 | 174.777 ± 1.651 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 43.984 ± 0.840 | - | 118.866 ± 0.535 | 30.973 ± 0.535 | 45.501 ± 0.598 | 125.079 ± 1.413 | 62.924 ± 1.897 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 44.390 ± 0.756 | - | 43.347 ± 0.127 | 42.288 ± 0.207 | 66.319 ± 0.487 | 46.616 ± 0.149 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 45.942 ± 0.363 | - | 45.726 ± 0.564 | 28.935 ± 0.774 | 29.772 ± 0.508 | 23.329 ± 0.879 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 44.959 ± 0.797 | - | unsupported | 42.966 ± 0.447 | 66.598 ± 0.259 | 44.607 ± 0.366 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 46.667 ± 2.448 | - | unsupported | 29.377 ± 0.573 | 29.883 ± 1.075 | 22.957 ± 4.085 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 10.901 ± 0.055 | - | 10.891 ± 0.060 | 11.554 ± 0.073 | 11.897 ± 0.016 | 11.898 ± 0.088 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 14.181 ± 0.109 | - | 14.167 ± 0.108 | 13.972 ± 0.012 | 17.381 ± 0.055 | 17.522 ± 0.015 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 71.185 ± 0.267 | - | 71.366 ± 0.712 | 107.536 ± 0.971 | 104.959 ± 1.341 | 83.387 ± 4.198 | 83.124 ± 23.810 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 57.518 ± 36.293 | - | 128.185 ± 40.019 | 42.711 ± 0.111 | 52.483 ± 0.906 | 82.534 ± 0.607 | 37.534 ± 5.984 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 115.587 ± 0.531 | - | 115.732 ± 0.394 | 152.115 ± 1.150 | 176.971 ± 1.659 | 226.470 ± 0.467 | 151.211 ± 0.724 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 49.754 ± 0.815 | - | 131.995 ± 23.996 | 63.045 ± 1.600 | 72.441 ± 0.908 | 148.721 ± 1.075 | 65.810 ± 7.065 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 70.070 ± 1.293 | - | 69.220 ± 1.220 | 67.534 ± 0.911 | 72.337 ± 1.327 | 76.933 ± 0.262 | 77.125 ± 0.195 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 24.415 ± 0.802 | - | 82.243 ± 3.250 | 22.815 ± 0.217 | 35.921 ± 0.459 | 75.305 ± 0.126 | 33.813 ± 0.990 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 2.346 ± 0.102 | - | 20.768 ± 0.114 | 24.233 ± 0.219 | 16.049 ± 0.301 | 4.073 ± 0.035 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 0.883 ± 0.070 | - | 59.995 ± 0.028 | 11.254 ± 0.109 | 2.597 ± 0.366 | 4.063 ± 0.014 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 8.641 ± 0.072 | - | 8.555 ± 0.071 | 9.605 ± 0.052 | 7.583 ± 0.015 | 8.897 ± 0.125 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 134.154 ± 14.891 | - | 107.110 ± 12.002 | 5.765 ± 0.069 | 7.090 ± 1.099 | 7.480 ± 0.957 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 5.452 ± 0.112 | - | 5.698 ± 0.291 | 5.419 ± 0.051 | 5.018 ± 0.050 | 4.858 ± 0.023 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 100.967 ± 10.881 | - | 145.977 ± 12.094 | 3.610 ± 0.019 | 4.661 ± 8.337 | 4.131 ± 0.216 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 6.465 ± 0.043 | - | 6.500 ± 0.149 | 7.653 ± 0.061 | 5.425 ± 0.036 | 5.318 ± 0.169 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 11.584 ± 0.144 | - | 11.541 ± 0.140 | 5.812 ± 0.009 | 6.029 ± 0.016 | 6.245 ± 0.587 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 44.706 ± 0.807 | - | 46.864 ± 10.203 | 42.510 ± 1.585 | 66.441 ± 0.389 | 43.957 ± 0.232 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 46.452 ± 0.691 | - | 46.551 ± 11.785 | 28.978 ± 0.740 | 29.439 ± 0.315 | 22.175 ± 0.180 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 48.132 ± 0.250 | - | unsupported | 50.982 ± 0.854 | 61.010 ± 0.451 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 52.476 ± 0.158 | - | unsupported | 25.222 ± 0.094 | 34.757 ± 0.084 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 121.615 ± 2.444 | - | 120.519 ± 0.639 | 119.498 ± 2.525 | 123.599 ± 0.931 | 137.352 ± 4.474 | 150.780 ± 4.011 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 40.857 ± 0.682 | - | 142.507 ± 1.030 | 40.075 ± 0.558 | 63.122 ± 1.356 | 137.440 ± 3.444 | 62.712 ± 1.889 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 130.910 ± 2.832 | - | 128.118 ± 3.604 | 130.806 ± 2.446 | 135.913 ± 1.079 | 145.340 ± 7.741 | 163.830 ± 2.131 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 45.461 ± 1.352 | - | 141.352 ± 13.977 | 44.509 ± 1.122 | 70.329 ± 2.197 | 145.682 ± 4.228 | 66.880 ± 2.580 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 27.375 ± 0.410 | - | 58.231 ± 0.291 | 74.217 ± 2.295 | 50.712 ± 0.321 | 80.153 ± 9.354 | 85.566 ± 9.153 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 12.389 ± 0.268 | - | 119.997 ± 0.006 | 33.294 ± 0.716 | 25.886 ± 0.328 | 81.293 ± 1.746 | 33.544 ± 10.224 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 9.873 ± 0.344 | - | 10.094 ± 0.492 | 35.388 ± 0.397 | 37.242 ± 1.301 | 43.773 ± 9.715 | 40.335 ± 12.534 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 6.777 ± 0.136 | - | 23.998 ± 0.005 | 12.278 ± 0.378 | 20.659 ± 0.406 | 42.666 ± 8.155 | 16.658 ± 21.500 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 22.167 ± 0.173 | - | 20.448 ± 0.395 | 52.395 ± 2.009 | 41.306 ± 1.148 | 23.164 ± 0.177 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 13.490 ± 0.157 | - | 36.002 ± 0.015 | 19.206 ± 0.080 | 22.696 ± 0.174 | 22.957 ± 0.500 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 28.309 ± 0.059 | - | 28.342 ± 0.092 | 55.585 ± 0.903 | 107.935 ± 0.298 | 66.831 ± 14.119 | 70.947 ± 14.188 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 11.911 ± 0.097 | - | 47.997 ± 0.016 | 21.904 ± 0.071 | 53.886 ± 0.378 | 66.293 ± 18.448 | 42.923 ± 14.384 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 132.131 ± 1.099 | - | 131.828 ± 4.088 | 131.794 ± 1.909 | 135.699 ± 1.354 | 147.274 ± 4.572 | 160.406 ± 4.448 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 44.967 ± 0.693 | - | 152.976 ± 10.473 | 44.050 ± 0.593 | 73.079 ± 1.248 | 143.240 ± 1.164 | 66.802 ± 0.323 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 20.419 ± 0.126 | - | 20.782 ± 0.110 | 43.222 ± 0.337 | 41.729 ± 0.540 | 61.939 ± 13.263 | 67.183 ± 12.788 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 8.406 ± 0.083 | - | 36.001 ± 23.990 | 16.602 ± 0.258 | 21.185 ± 0.559 | 70.793 ± 10.005 | 41.609 ± 14.856 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 20.452 ± 0.026 | - | 20.370 ± 0.014 | 31.480 ± 1.051 | 27.765 ± 0.102 | 37.601 ± 8.870 | 41.191 ± 9.807 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 8.090 ± 0.035 | - | 35.998 ± 0.006 | 11.439 ± 0.282 | 13.835 ± 0.277 | 38.465 ± 1.321 | 15.994 ± 11.500 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 25.612 ± 0.074 | - | 25.240 ± 0.132 | 49.133 ± 0.367 | 64.183 ± 0.591 | 83.758 ± 55.423 | 96.891 ± 8.382 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 10.993 ± 0.064 | - | 47.999 ± 0.004 | 18.928 ± 0.956 | 33.114 ± 1.238 | 91.107 ± 15.480 | 37.618 ± 14.749 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 13.204 ± 0.045 | - | 13.200 ± 0.018 | 32.289 ± 0.468 | 37.108 ± 0.180 | 46.027 ± 9.467 | 49.797 ± 10.882 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 5.639 ± 0.009 | - | 36.000 ± 0.005 | 11.766 ± 0.081 | 18.489 ± 0.160 | 45.740 ± 0.748 | 21.885 ± 20.018 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.449 ± 0.013 | 0.486 ± 0.011 | unsupported | unsupported | 0.480 ± 0.006 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.387 ± 0.006 | 51.999 ± 0.812 | unsupported | unsupported | 0.257 ± 0.017 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 135.517 ± 2.856 | - | 134.386 ± 2.030 | 131.402 ± 3.018 | 135.584 ± 2.441 | 152.277 ± 4.541 | 164.315 ± 4.619 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 45.656 ± 1.291 | - | 144.086 ± 14.557 | 44.074 ± 0.747 | 70.750 ± 1.111 | 151.703 ± 3.878 | 68.705 ± 1.375 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 134.836 ± 2.402 | - | 135.625 ± 2.879 | 131.845 ± 2.707 | 133.997 ± 1.153 | 151.109 ± 4.245 | 164.689 ± 1.728 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 45.455 ± 1.081 | - | 150.377 ± 11.847 | 44.312 ± 0.537 | 70.435 ± 0.477 | 151.441 ± 2.833 | 68.005 ± 1.096 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 131.452 ± 2.843 | - | 129.382 ± 3.744 | 131.765 ± 2.419 | 136.549 ± 1.225 | 147.479 ± 2.247 | 163.346 ± 4.786 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 46.267 ± 2.339 | - | 141.670 ± 12.090 | 44.416 ± 0.201 | 86.033 ± 0.819 | 146.271 ± 4.480 | 67.253 ± 7.694 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 119.211 ± 3.223 | - | 122.596 ± 2.990 | 120.864 ± 2.415 | 123.055 ± 1.041 | 137.165 ± 4.136 | 151.405 ± 5.280 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 40.487 ± 0.736 | - | 142.423 ± 11.267 | 40.541 ± 0.809 | 63.347 ± 1.544 | 137.126 ± 2.776 | 63.278 ± 1.004 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 58.100 ± 0.084 | - | 57.894 ± 0.221 | 62.417 ± 0.429 | 73.648 ± 0.265 | 125.382 ± 13.654 | 125.533 ± 11.718 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 23.571 ± 0.356 | - | 71.999 ± 0.015 | 22.126 ± 0.181 | 36.924 ± 0.486 | 125.066 ± 0.606 | 81.658 ± 20.585 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 16.499 ± 0.399 | - | 16.446 ± 0.541 | 12.274 ± 0.550 | 26.095 ± 0.173 | 13.136 ± 0.252 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 6.600 ± 0.180 | - | 35.998 ± 0.011 | 6.493 ± 0.320 | 13.511 ± 0.199 | 13.012 ± 0.241 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.879 ± 0.111 | - | 1.946 ± 0.057 | 24.257 ± 0.342 | 2.461 ± 0.037 | 1.543 ± 0.101 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.804 ± 0.097 | - | 24.000 ± 0.006 | 11.162 ± 0.598 | 1.500 ± 0.072 | 1.240 ± 0.182 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 2.516 ± 0.147 | - | 2.551 ± 0.056 | 4.055 ± 0.024 | 2.517 ± 0.262 | 1.255 ± 0.114 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 1.267 ± 0.049 | - | 23.995 ± 0.008 | 2.314 ± 0.099 | 1.449 ± 0.248 | 1.356 ± 0.117 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 16.588 ± 0.403 | - | 16.378 ± 0.270 | 12.073 ± 0.427 | 26.072 ± 0.328 | 13.519 ± 0.290 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 6.887 ± 0.078 | - | 35.999 ± 0.009 | 6.119 ± 0.082 | 13.776 ± 0.482 | 13.066 ± 0.118 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.939 ± 0.188 | - | 1.918 ± 0.095 | 23.026 ± 0.247 | 2.412 ± 0.059 | 1.550 ± 0.194 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.808 ± 0.017 | - | 23.999 ± 0.008 | 10.505 ± 0.644 | 1.334 ± 0.044 | 1.226 ± 0.131 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 10.301 ± 0.148 | - | 10.579 ± 0.329 | 16.288 ± 0.174 | 11.672 ± 0.472 | 6.463 ± 0.165 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 5.263 ± 0.357 | - | 35.994 ± 0.013 | 8.798 ± 0.071 | 6.406 ± 0.330 | 6.480 ± 0.328 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 11.281 ± 0.470 | - | 11.268 ± 0.324 | 10.951 ± 0.509 | 19.703 ± 0.168 | 11.442 ± 0.274 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 6.231 ± 0.081 | - | 23.999 ± 0.004 | 6.331 ± 0.264 | 10.644 ± 0.140 | 10.974 ± 0.567 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 1.278 ± 0.049 | - | 1.277 ± 0.108 | 1.958 ± 0.050 | 1.978 ± 0.016 | 1.065 ± 0.226 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.492 ± 0.049 | - | 23.997 ± 0.009 | 0.944 ± 0.018 | 1.113 ± 0.072 | 0.942 ± 0.166 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.440 ± 0.096 | - | 1.388 ± 0.109 | 0.789 ± 0.014 | 3.313 ± 0.252 | 1.342 ± 0.239 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.658 ± 0.071 | - | 24.000 ± 0.006 | 5.992 ± 5.559 | 1.630 ± 0.242 | 1.170 ± 0.131 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 10.815 ± 0.124 | - | 11.305 ± 0.425 | 10.662 ± 0.524 | 19.779 ± 0.146 | 11.481 ± 0.308 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 6.459 ± 0.202 | - | 23.999 ± 0.011 | 6.112 ± 0.024 | 10.675 ± 0.150 | 10.870 ± 0.140 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 1.181 ± 0.143 | - | 1.444 ± 0.133 | 1.982 ± 0.055 | 1.983 ± 0.023 | 0.949 ± 0.174 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.429 ± 0.062 | - | 23.997 ± 0.004 | 0.908 ± 0.026 | 1.138 ± 0.050 | 0.997 ± 0.163 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.355 ± 0.189 | - | 1.450 ± 0.111 | 0.836 ± 0.044 | 3.054 ± 0.369 | 0.988 ± 0.079 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.595 ± 0.049 | - | 23.996 ± 0.016 | 0.405 ± 0.092 | 1.597 ± 0.246 | 1.125 ± 0.090 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 1.157 ± 0.116 | 1.169 ± 0.122 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.369 ± 0.039 | 23.999 ± 0.002 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 38.558 ± 3.405 | - | 38.325 ± 0.206 | 54.239 ± 0.502 | 72.620 ± 1.292 | 68.820 ± 7.511 | 83.231 ± 8.872 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 17.989 ± 0.139 | - | 59.996 ± 0.006 | 20.472 ± 0.162 | 37.893 ± 1.712 | 68.511 ± 6.923 | 41.972 ± 9.790 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 141.135 ± 1.032 | - | 140.535 ± 1.389 | 142.766 ± 2.244 | 245.757 ± 0.806 | 261.779 ± 4.732 | 261.141 ± 2.269 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 46.216 ± 0.695 | - | 177.247 ± 8.290 | 46.425 ± 0.711 | 123.610 ± 0.647 | 259.435 ± 2.015 | 102.410 ± 5.644 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 137.489 ± 2.563 | - | 136.533 ± 5.473 | 143.175 ± 2.690 | 150.629 ± 1.069 | 150.252 ± 1.057 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 47.578 ± 0.739 | - | 155.804 ± 12.101 | 52.100 ± 0.949 | 78.056 ± 1.046 | 149.273 ± 2.378 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 125.210 ± 3.272 | - | 125.732 ± 3.616 | 125.451 ± 3.037 | 132.837 ± 1.062 | 138.441 ± 4.026 | 145.705 ± 4.290 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 40.733 ± 0.867 | - | 143.075 ± 1.717 | 40.353 ± 0.649 | 64.326 ± 0.934 | 140.978 ± 3.504 | 61.396 ± 1.137 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 24.496 ± 0.411 | - | 24.512 ± 0.084 | 57.300 ± 0.139 | 107.458 ± 0.245 | 73.946 ± 12.051 | 69.568 ± 15.794 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 10.250 ± 0.152 | - | 47.998 ± 0.011 | 23.103 ± 0.298 | 54.070 ± 0.266 | 66.058 ± 12.990 | 30.795 ± 16.145 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 130.297 ± 2.433 | - | 130.047 ± 3.840 | 184.974 ± 0.767 | 133.444 ± 0.581 | 211.562 ± 4.070 | 217.771 ± 4.177 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 43.528 ± 0.491 | - | 142.530 ± 4.068 | 68.021 ± 1.873 | 67.315 ± 0.877 | 211.000 ± 1.154 | 89.635 ± 1.295 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 128.812 ± 4.045 | - | 131.103 ± 4.130 | 130.219 ± 2.555 | 136.803 ± 1.099 | 144.120 ± 3.756 | 164.298 ± 1.352 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 45.855 ± 0.438 | - | 143.741 ± 0.788 | 44.535 ± 1.704 | 69.620 ± 0.360 | 146.593 ± 4.653 | 67.429 ± 0.721 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 51.315 ± 0.052 | - | 51.374 ± 0.056 | 75.690 ± 0.568 | 39.741 ± 0.352 | 78.841 ± 13.790 | 86.487 ± 20.632 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 20.493 ± 0.137 | - | 71.997 ± 0.008 | 28.555 ± 0.360 | 19.694 ± 0.489 | 84.686 ± 12.171 | 33.771 ± 14.378 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 136.985 ± 3.323 | 130.350 ± 3.453 | unsupported | 144.933 ± 1.556 | 151.241 ± 1.187 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 46.104 ± 0.343 | 144.715 ± 14.240 | unsupported | 51.928 ± 0.758 | 77.933 ± 0.544 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.851 ± 0.051 | - | 1.075 ± 0.061 | 0.729 ± 0.028 | 0.844 ± 0.154 | 7.509 ± 3.950 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.332 ± 0.043 | - | 11.998 ± 0.008 | 0.329 ± 0.045 | 0.855 ± 0.121 | 7.251 ± 3.590 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 1.171 ± 0.288 | - | 0.864 ± 0.092 | 0.857 ± 0.093 | 0.898 ± 0.166 | 7.475 ± 3.761 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.434 ± 0.118 | - | 11.998 ± 0.007 | 0.353 ± 0.029 | 0.794 ± 0.093 | 7.517 ± 3.846 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.210 ± 0.029 | - | unsupported | 1.446 ± 0.084 | 1.594 ± 0.277 | 5.760 ± 5.837 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.608 ± 0.089 | - | unsupported | 0.638 ± 0.117 | 0.964 ± 0.114 | 8.299 ± 3.946 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.388 ± 0.012 | - | 0.488 ± 0.039 | 0.686 ± 0.026 | 0.440 ± 0.037 | 1.287 ± 0.102 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.432 ± 0.085 | - | 48.000 ± 0.021 | 0.347 ± 0.022 | 0.363 ± 0.024 | 1.331 ± 0.100 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.915 ± 0.117 | - | 0.876 ± 0.253 | 1.394 ± 0.412 | 1.045 ± 0.160 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.354 ± 0.099 | - | 12.002 ± 0.011 | 0.497 ± 0.031 | 0.643 ± 0.030 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.875 ± 0.147 | - | 1.023 ± 0.091 | 0.791 ± 0.083 | 0.857 ± 0.068 | 1.552 ± 7.050 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.409 ± 0.162 | - | 11.997 ± 0.007 | 0.336 ± 0.133 | 0.486 ± 0.070 | 7.912 ± 7.125 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.834 ± 0.026 | - | 0.929 ± 0.006 | 0.476 ± 0.020 | 0.471 ± 0.032 | 0.528 ± 0.048 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 1.221 ± 0.019 | - | 36.004 ± 0.271 | 0.643 ± 0.015 | 0.468 ± 0.045 | 1.352 ± 0.057 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.667 ± 0.042 | - | 0.002 ± 0.000 | 1.801 ± 0.432 | 1.384 ± 0.025 | 8.807 ± 4.900 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.730 ± 0.072 | - | 0.002 ± 0.000 | 0.815 ± 0.255 | 1.148 ± 0.168 | 9.083 ± 5.038 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.051 ± 0.001 | - | 0.052 ± 0.002 | 0.075 ± 0.001 | 0.117 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.093 ± 0.006 | - | 0.095 ± 0.002 | 0.124 ± 0.004 | 0.150 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.238 ± 0.010 | - | 0.242 ± 0.009 | 0.277 ± 0.011 | 0.322 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.084 ± 0.006 | - | 0.083 ± 0.009 | 0.147 ± 0.004 | 0.122 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.035 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.031 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.660 ± 0.009 | - | 0.676 ± 0.010 | 1.154 ± 0.006 | 0.686 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.026 ± 0.001 | - | 0.027 ± 0.000 | 0.038 ± 0.001 | 0.063 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.030 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.055 ± 0.000 | - | 0.056 ± 0.000 | 0.085 ± 0.003 | 0.105 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.094 ± 0.001 | - | 0.096 ± 0.000 | 0.123 ± 0.001 | 0.163 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.240 ± 0.000 | - | 0.242 ± 0.001 | 0.342 ± 0.014 | 0.365 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.087 ± 0.006 | - | 0.090 ± 0.002 | 0.178 ± 0.002 | 0.127 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.001 | 0.013 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.005 ± 0.005 | - | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.673 ± 0.022 | - | 0.669 ± 4.613 | 1.397 ± 0.018 | 1.003 ± 0.968 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.027 ± 0.000 | - | 0.029 ± 0.000 | 0.059 ± 0.001 | 0.070 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.012 ± 0.000 | 0.017 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.602 ± 0.009 | - | 0.604 ± 0.008 | 0.702 ± 0.010 | 0.847 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 2.074 ± 0.003 | - | 2.080 ± 0.004 | 2.141 ± 0.016 | 2.682 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 6.439 ± 0.077 | - | 6.429 ± 0.041 | 6.529 ± 0.092 | 8.864 ± 0.062 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 3.656 ± 0.015 | - | 3.660 ± 0.020 | 3.234 ± 0.015 | 2.653 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.109 ± 0.001 | - | 0.111 ± 0.006 | 0.129 ± 0.025 | 0.159 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.008 ± 0.002 | - | 0.010 ± 0.000 | 0.017 ± 0.001 | 0.031 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.001 | 0.028 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.016 ± 0.009 | 0.028 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.020 ± 0.000 | - | 0.023 ± 0.000 | 0.030 ± 0.001 | 0.074 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.004 ± 0.006 | - | 0.006 ± 0.000 | 0.012 ± 0.001 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.018 ± 0.001 | 0.035 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.026 ± 0.000 | - | 0.029 ± 0.000 | 0.035 ± 0.001 | 0.089 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 29.184 ± 0.291 | - | 29.191 ± 0.371 | 25.763 ± 0.085 | 20.910 ± 0.120 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.014 ± 0.000 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.861 ± 0.004 | - | 0.868 ± 0.026 | 0.875 ± 0.001 | 0.963 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.018 ± 0.000 | - | 0.020 ± 0.000 | 0.026 ± 0.001 | 0.046 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.051 ± 0.003 | - | 0.054 ± 0.000 | 0.060 ± 0.003 | 0.112 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.619 ± 0.036 | - | 0.646 ± 2.320 | 0.744 ± 0.399 | 0.889 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 2.133 ± 0.019 | - | 2.142 ± 0.005 | 2.140 ± 0.022 | 2.144 ± 0.081 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 6.531 ± 0.033 | - | 6.535 ± 0.039 | 9.294 ± 0.203 | 5.873 ± 0.418 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 6.613 ± 0.088 | - | 6.606 ± 0.095 | 2.270 ± 0.047 | 3.489 ± 0.153 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.012 ± 0.001 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.111 ± 0.001 | - | 0.115 ± 0.002 | 0.178 ± 0.004 | 0.190 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.001 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.016 ± 0.001 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.016 ± 0.001 | 0.025 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.003 | 0.015 ± 0.000 | 0.037 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.021 ± 0.000 | - | 0.024 ± 0.000 | 0.029 ± 0.001 | 0.071 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.002 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.018 ± 0.000 | 0.033 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.027 ± 0.000 | - | 0.030 ± 0.000 | 0.035 ± 0.001 | 0.080 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 52.965 ± 0.386 | - | 53.003 ± 0.441 | 17.558 ± 0.159 | 28.256 ± 4.935 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.021 ± 0.001 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.869 ± 0.017 | - | 0.876 ± 4.576 | 1.241 ± 0.011 | 1.293 ± 0.560 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.018 ± 0.000 | - | 0.021 ± 0.000 | 0.026 ± 0.001 | 0.046 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.052 ± 0.001 | - | 0.055 ± 0.000 | 0.061 ± 0.002 | 0.114 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.661 ± 0.012 | - | 0.660 ± 0.011 | 0.521 ± 0.014 | 0.441 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 2.205 ± 0.033 | - | 2.202 ± 0.017 | 1.936 ± 0.008 | 1.933 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 5.781 ± 0.007 | - | 5.783 ± 0.004 | 5.696 ± 0.043 | 6.430 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.312 ± 0.028 | - | 1.321 ± 0.022 | 1.340 ± 0.016 | 1.830 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.071 ± 0.003 | - | 0.074 ± 0.006 | 0.090 ± 0.004 | 0.129 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.015 ± 0.001 | 0.028 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.001 | 0.024 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.014 ± 0.001 | 0.031 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.018 ± 0.000 | - | 0.022 ± 0.007 | 0.027 ± 0.002 | 0.049 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.026 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.017 ± 0.004 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.024 ± 0.000 | - | 0.027 ± 0.000 | 0.031 ± 0.001 | 0.047 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 10.369 ± 0.084 | - | 10.430 ± 0.048 | 10.589 ± 0.073 | 11.214 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.001 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.545 ± 0.008 | - | 0.557 ± 0.025 | 0.627 ± 0.015 | 0.751 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.018 ± 0.001 | - | 0.021 ± 0.000 | 0.023 ± 0.001 | 0.037 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.046 ± 0.001 | - | 0.049 ± 0.000 | 0.053 ± 0.002 | 0.091 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.648 ± 0.024 | - | 0.650 ± 4.637 | 0.803 ± 0.099 | 0.451 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 2.200 ± 0.022 | - | 2.176 ± 0.024 | 1.895 ± 0.003 | 1.275 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 5.806 ± 0.018 | - | 5.820 ± 0.022 | 8.400 ± 3.941 | 3.687 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 3.105 ± 0.030 | - | 3.113 ± 0.055 | 1.199 ± 0.110 | 1.771 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.011 ± 0.003 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.145 ± 0.040 | - | 0.149 ± 0.027 | 0.133 ± 0.007 | 0.151 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.002 | 0.023 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.014 ± 0.001 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.014 ± 0.001 | 0.025 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.014 ± 0.001 | 0.026 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.019 ± 0.000 | - | 0.022 ± 0.000 | 0.027 ± 0.001 | 0.040 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.015 ± 0.002 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.017 ± 0.000 | 0.030 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.025 ± 0.000 | - | 0.028 ± 0.000 | 0.031 ± 0.000 | 0.057 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 24.798 ± 0.302 | - | 24.700 ± 0.152 | 9.035 ± 0.040 | 13.891 ± 1.175 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.017 ± 0.002 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 1.086 ± 0.020 | - | 1.079 ± 0.025 | 0.925 ± 0.003 | 0.468 ± 0.146 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.024 ± 0.004 | 0.035 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.047 ± 0.001 | - | 0.050 ± 0.000 | 0.053 ± 0.002 | 0.091 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.264 ± 0.008 | - | 0.268 ± 0.009 | 0.235 ± 0.013 | 0.443 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 1.082 ± 0.003 | - | 1.085 ± 0.005 | 0.951 ± 0.004 | 1.914 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 3.081 ± 0.008 | - | 3.120 ± 0.028 | 2.905 ± 0.027 | 6.460 ± 0.085 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.514 ± 0.006 | - | 0.518 ± 0.010 | 0.557 ± 0.014 | 1.486 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.010 ± 0.001 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.031 ± 0.000 | - | 0.033 ± 0.000 | 0.042 ± 0.004 | 0.151 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.011 ± 0.002 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.018 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.019 ± 0.002 | 0.040 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.014 ± 0.001 | 0.026 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.022 ± 0.001 | 0.040 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 4.074 ± 0.032 | - | 4.055 ± 0.007 | 4.312 ± 0.017 | 11.260 ± 0.281 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.231 ± 0.009 | - | 0.234 ± 0.002 | 0.244 ± 0.006 | 0.745 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.017 ± 0.001 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.025 ± 0.001 | - | 0.027 ± 0.000 | 0.032 ± 0.000 | 0.103 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.264 ± 0.004 | - | 0.269 ± 0.016 | 0.396 ± 0.011 | 0.746 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 1.110 ± 0.006 | - | 1.115 ± 0.020 | 0.947 ± 0.001 | 1.260 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 3.136 ± 0.045 | - | 3.137 ± 0.017 | 3.657 ± 0.017 | 3.655 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 1.946 ± 0.013 | - | 1.976 ± 0.017 | 0.698 ± 0.010 | 1.885 ± 0.261 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.014 ± 0.001 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.031 ± 0.002 | - | 0.034 ± 0.001 | 0.055 ± 0.003 | 0.130 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.001 | 0.022 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.015 ± 0.004 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.019 ± 0.000 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.015 ± 0.003 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.019 ± 0.006 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.022 ± 0.001 | 0.041 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 15.524 ± 0.072 | - | 15.623 ± 0.067 | 5.199 ± 0.036 | 14.994 ± 1.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.017 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.232 ± 0.023 | - | 0.239 ± 0.017 | 0.322 ± 0.015 | 0.450 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.024 ± 0.003 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.025 ± 0.000 | - | 0.028 ± 0.000 | 0.033 ± 0.001 | 0.089 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.148 ± 0.003 | - | 0.152 ± 0.005 | 0.190 ± 0.016 | 0.122 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.213 ± 0.010 | - | 0.213 ± 0.014 | 0.255 ± 0.013 | 0.209 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.506 ± 0.020 | - | 0.517 ± 0.008 | 0.537 ± 0.013 | 0.578 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.138 ± 0.019 | - | 0.138 ± 0.002 | 0.174 ± 0.013 | 0.238 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.013 ± 0.001 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.012 ± 0.001 | - | 0.015 ± 0.000 | 0.026 ± 0.033 | 0.064 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.013 ± 0.001 | 0.019 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.014 ± 0.000 | - | 0.017 ± 0.000 | 0.029 ± 0.001 | 0.054 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.013 ± 0.001 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.016 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.005 | - | 0.005 ± 0.000 | 0.017 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.004 | 0.017 ± 0.000 | 0.032 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.000 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.005 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.018 ± 0.001 | 0.038 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 1.051 ± 0.009 | - | 1.052 ± 0.007 | 1.306 ± 0.015 | 1.264 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.005 | 0.017 ± 0.002 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.076 ± 0.000 | - | 0.080 ± 0.001 | 0.086 ± 0.003 | 0.157 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.016 ± 0.000 | 0.043 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.020 ± 0.001 | 0.051 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.153 ± 0.010 | - | 0.152 ± 0.007 | 0.171 ± 0.020 | 0.128 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.216 ± 0.016 | - | 0.215 ± 0.017 | 0.233 ± 0.017 | 0.213 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.524 ± 0.012 | - | 0.518 ± 0.020 | 0.383 ± 0.019 | 0.640 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.138 ± 0.005 | - | 0.157 ± 0.004 | 0.225 ± 0.017 | 0.223 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.023 ± 0.005 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.013 ± 0.001 | - | 0.015 ± 0.000 | 0.038 ± 0.001 | 0.059 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.023 ± 0.005 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.015 ± 0.007 | - | 0.018 ± 0.000 | 0.042 ± 0.004 | 0.067 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.022 ± 0.004 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.025 ± 0.001 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.025 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.027 ± 0.001 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.001 | 0.025 ± 0.001 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.025 ± 0.000 | 0.025 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.027 ± 0.000 | 0.019 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 1.080 ± 0.057 | - | 1.084 ± 0.046 | 0.695 ± 0.023 | 1.012 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.027 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.081 ± 0.001 | - | 0.083 ± 0.002 | 0.077 ± 0.001 | 0.189 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.027 ± 0.000 | 0.021 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.030 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.249 ± 0.005 | - | 0.254 ± 0.010 | 0.226 ± 0.011 | 0.332 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.597 ± 0.009 | - | 0.604 ± 0.009 | 0.528 ± 0.009 | 0.860 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.337 ± 0.036 | - | 1.360 ± 0.130 | 1.368 ± 0.013 | 2.360 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.492 ± 0.016 | - | 0.496 ± 0.014 | 0.572 ± 0.010 | 0.630 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.018 ± 0.000 | - | 0.020 ± 0.004 | 0.027 ± 0.001 | 0.064 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.003 | 0.006 ± 0.000 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.019 ± 0.000 | - | 0.021 ± 0.001 | 0.030 ± 0.001 | 0.090 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.002 | 0.030 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.035 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 3.848 ± 0.004 | - | 4.744 ± 0.113 | 4.476 ± 0.038 | 4.733 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.128 ± 0.001 | - | 0.131 ± 0.002 | 0.175 ± 0.028 | 0.215 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.016 ± 0.001 | 0.038 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.261 ± 0.002 | - | 0.251 ± 0.009 | 0.324 ± 0.012 | 0.324 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.605 ± 0.026 | - | 0.600 ± 4.642 | 0.762 ± 0.012 | 0.859 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.358 ± 0.017 | - | 1.329 ± 0.038 | 2.032 ± 0.018 | 1.328 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 1.015 ± 0.038 | - | 1.025 ± 0.011 | 0.506 ± 0.023 | 0.929 ± 0.091 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.018 ± 0.000 | - | 0.023 ± 0.000 | 0.044 ± 0.001 | 0.069 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.020 ± 0.000 | - | 0.026 ± 0.001 | 0.047 ± 0.008 | 0.092 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.001 | 0.026 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.002 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.029 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.000 | 0.029 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 7.970 ± 0.089 | - | 8.792 ± 0.079 | 3.768 ± 0.372 | 7.858 ± 1.124 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.132 ± 0.022 | - | 0.142 ± 0.010 | 0.293 ± 0.013 | 0.216 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.015 ± 0.000 | 0.032 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.026 ± 0.000 | 0.037 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.155 ± 0.002 | - | 0.151 ± 0.005 | 0.099 ± 0.002 | 0.259 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.265 ± 0.023 | - | 0.215 ± 0.029 | 0.140 ± 0.002 | 0.382 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.545 ± 0.019 | - | 0.559 ± 0.013 | 0.404 ± 0.014 | 0.783 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.117 ± 0.003 | - | 0.157 ± 0.043 | 0.145 ± 0.016 | 0.177 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.001 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.011 ± 0.000 | - | 0.017 ± 0.003 | 0.031 ± 0.002 | 0.056 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.000 | 0.018 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.010 | 0.018 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.001 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.020 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.001 | 0.019 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.001 | 0.020 ± 0.003 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.913 ± 0.012 | - | 0.974 ± 0.012 | 1.023 ± 0.023 | 1.042 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.001 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.071 ± 0.002 | - | 0.078 ± 0.019 | 0.080 ± 0.004 | 0.124 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.011 ± 0.000 | 0.021 ± 0.001 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.235 ± 0.016 | - | 0.148 ± 0.013 | 0.131 ± 0.002 | 0.278 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.207 ± 0.003 | - | 0.234 ± 0.041 | 0.174 ± 0.019 | 0.372 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.543 ± 0.035 | - | 5.998 ± 0.025 | 0.383 ± 0.014 | 0.831 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.121 ± 0.004 | - | 0.173 ± 0.037 | 0.205 ± 0.019 | 0.186 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.026 ± 0.002 | 0.019 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.011 ± 0.000 | - | 0.019 ± 0.003 | 0.042 ± 0.004 | 0.057 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.025 ± 0.002 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.026 ± 0.005 | 0.016 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.026 ± 0.004 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.025 ± 0.001 | 0.022 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.029 ± 0.003 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.025 ± 0.002 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.026 ± 0.004 | 0.025 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.011 ± 0.000 | 0.029 ± 0.004 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.930 ± 0.029 | - | 23.996 ± 0.046 | 0.576 ± 0.016 | 0.737 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.017 ± 0.002 | 0.028 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.071 ± 0.001 | - | 0.083 ± 0.001 | 0.072 ± 0.000 | 0.165 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.028 ± 0.002 | 0.021 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.013 ± 0.001 | 0.032 ± 0.001 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.769 ± 0.010 | - | 1.775 ± 0.008 | 1.522 ± 0.011 | 1.965 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 5.948 ± 0.003 | - | 5.949 ± 0.088 | 5.091 ± 0.034 | 5.032 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 17.685 ± 0.042 | - | 17.684 ± 0.074 | 15.955 ± 0.141 | 14.609 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 4.936 ± 0.020 | - | 4.947 ± 0.050 | 5.618 ± 0.021 | 4.176 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.015 ± 0.001 | 0.028 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.199 ± 0.005 | - | 0.205 ± 0.018 | 0.236 ± 0.031 | 0.245 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.019 ± 0.000 | 0.033 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.182 ± 0.016 | - | 0.184 ± 0.011 | 0.207 ± 0.017 | 0.219 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.020 ± 0.000 | - | 0.023 ± 0.001 | 0.033 ± 0.001 | 0.051 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.008 ± 0.000 | - | 0.010 ± 0.005 | 0.018 ± 0.001 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.019 ± 0.000 | - | 0.023 ± 0.003 | 0.029 ± 0.002 | 0.043 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.054 ± 0.001 | - | 0.057 ± 0.004 | 0.063 ± 0.001 | 0.076 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.019 ± 0.001 | 0.033 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.025 ± 0.000 | - | 0.028 ± 0.000 | 0.037 ± 0.009 | 0.049 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.072 ± 0.002 | - | 0.075 ± 0.001 | 0.079 ± 0.003 | 0.090 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 39.197 ± 0.308 | - | 40.955 ± 1.880 | 44.319 ± 0.255 | 33.221 ± 0.535 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.016 ± 0.000 | - | 0.019 ± 0.004 | 0.025 ± 0.005 | 0.042 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.540 ± 0.011 | - | 1.545 ± 0.012 | 1.642 ± 0.027 | 1.532 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.049 ± 0.000 | - | 0.052 ± 0.003 | 0.055 ± 0.001 | 0.101 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.140 ± 0.001 | - | 0.143 ± 0.001 | 0.140 ± 0.004 | 0.160 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.748 ± 0.018 | - | 1.761 ± 0.025 | 1.508 ± 0.025 | 1.984 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 5.947 ± 0.009 | - | 5.952 ± 0.047 | 5.047 ± 0.057 | 2.748 ± 0.092 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 17.684 ± 0.079 | - | 17.661 ± 0.122 | 18.552 ± 0.311 | 7.542 ± 0.225 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 8.927 ± 0.083 | - | 8.940 ± 0.091 | 4.651 ± 0.029 | 4.925 ± 0.189 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.019 ± 0.002 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.297 ± 0.027 | - | 0.310 ± 0.023 | 0.326 ± 0.013 | 0.261 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.009 ± 0.000 | - | 0.013 ± 0.000 | 0.020 ± 0.001 | 0.031 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.220 ± 0.014 | - | 0.237 ± 0.025 | 0.289 ± 0.018 | 0.222 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.021 ± 0.001 | - | 0.025 ± 0.000 | 0.032 ± 0.001 | 0.044 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.025 ± 0.005 | 0.030 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.020 ± 0.000 | - | 0.024 ± 0.000 | 0.029 ± 0.001 | 0.042 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.056 ± 0.001 | - | 0.060 ± 0.001 | 0.063 ± 0.001 | 0.072 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.028 ± 0.005 | 0.032 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.026 ± 0.000 | - | 0.030 ± 0.000 | 0.035 ± 0.001 | 0.046 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.073 ± 0.001 | - | 0.080 ± 0.001 | 0.079 ± 0.001 | 0.107 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 71.043 ± 0.386 | - | 72.779 ± 0.215 | 36.352 ± 0.036 | 62.989 ± 30.051 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.016 ± 0.000 | - | 0.019 ± 0.000 | 0.037 ± 0.005 | 0.040 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 2.268 ± 0.051 | - | 2.283 ± 0.029 | 2.330 ± 0.013 | 0.782 ± 0.162 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.050 ± 0.001 | - | 0.053 ± 0.001 | 0.055 ± 0.003 | 0.070 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.144 ± 0.026 | - | 0.146 ± 0.008 | 0.140 ± 0.002 | 0.159 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 1.350 ± 0.030 | - | 1.324 ± 0.009 | 1.064 ± 0.009 | 1.271 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 5.102 ± 0.021 | - | 5.101 ± 0.014 | 4.332 ± 0.045 | 4.023 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 15.068 ± 0.046 | - | 15.086 ± 0.049 | 13.846 ± 0.036 | 12.240 ± 2.402 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 2.283 ± 0.011 | - | 2.307 ± 0.017 | 2.467 ± 0.023 | 2.458 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.001 | 0.030 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.134 ± 0.007 | - | 0.138 ± 0.009 | 0.166 ± 0.011 | 0.210 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.000 | 0.030 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.109 ± 0.012 | - | 0.110 ± 0.003 | 0.136 ± 0.010 | 0.165 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.017 ± 0.000 | - | 0.020 ± 0.000 | 0.029 ± 0.006 | 0.045 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.016 ± 0.001 | 0.034 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.017 ± 0.000 | - | 0.020 ± 0.000 | 0.028 ± 0.001 | 0.037 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.043 ± 0.009 | - | 0.045 ± 0.000 | 0.052 ± 0.001 | 0.060 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.017 ± 0.001 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.022 ± 0.000 | - | 0.025 ± 0.001 | 0.031 ± 0.001 | 0.051 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.057 ± 0.001 | - | 0.061 ± 0.027 | 0.065 ± 0.000 | 0.072 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 18.237 ± 0.186 | - | 18.324 ± 0.239 | 19.614 ± 0.443 | 18.765 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.013 ± 0.002 | - | 0.015 ± 0.000 | 0.021 ± 0.001 | 0.033 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 1.046 ± 0.002 | - | 1.064 ± 0.030 | 1.162 ± 0.027 | 1.195 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.042 ± 0.000 | - | 0.045 ± 0.000 | 0.047 ± 0.000 | 0.058 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.114 ± 0.003 | - | 0.118 ± 0.005 | 0.117 ± 0.001 | 0.137 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 1.277 ± 0.032 | - | 1.285 ± 0.033 | 1.036 ± 0.035 | 1.319 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 5.090 ± 0.043 | - | 5.087 ± 0.020 | 4.259 ± 0.013 | 2.088 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 15.117 ± 0.161 | - | 15.030 ± 0.139 | 16.633 ± 0.044 | 6.292 ± 0.298 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 3.376 ± 0.112 | - | 3.392 ± 0.030 | 2.760 ± 0.011 | 3.093 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.004 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.231 ± 0.015 | - | 0.239 ± 0.016 | 0.254 ± 0.013 | 0.186 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.018 ± 0.002 | 0.035 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.150 ± 0.009 | - | 0.166 ± 0.013 | 0.198 ± 0.019 | 0.171 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.018 ± 0.000 | - | 0.021 ± 0.001 | 0.027 ± 0.001 | 0.047 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.021 ± 0.001 | 0.034 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.018 ± 0.000 | - | 0.021 ± 0.000 | 0.026 ± 0.001 | 0.037 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.044 ± 0.001 | - | 0.048 ± 0.001 | 0.052 ± 0.000 | 0.063 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.025 ± 0.004 | 0.034 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.023 ± 0.000 | - | 0.026 ± 0.001 | 0.031 ± 0.001 | 0.042 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.058 ± 0.002 | - | 0.062 ± 0.000 | 0.064 ± 0.003 | 0.072 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 26.978 ± 0.204 | - | 26.988 ± 0.262 | 21.615 ± 0.632 | 26.995 ± 8.828 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.030 ± 0.004 | 0.040 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 1.739 ± 0.027 | - | 1.774 ± 0.022 | 1.745 ± 0.004 | 0.675 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.043 ± 0.000 | - | 0.046 ± 0.000 | 0.047 ± 0.001 | 0.057 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.114 ± 0.001 | - | 0.119 ± 0.003 | 0.116 ± 0.002 | 0.158 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.456 ± 0.009 | - | unsupported | 0.350 ± 0.012 | 0.751 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 2.230 ± 0.009 | - | unsupported | 1.908 ± 0.005 | 2.671 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 5.493 ± 0.018 | - | unsupported | 5.284 ± 0.030 | 6.745 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.978 ± 0.011 | - | unsupported | 1.161 ± 0.023 | 1.104 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.048 ± 0.000 | - | unsupported | 0.064 ± 0.005 | 0.073 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | unsupported | 0.012 ± 0.001 | 0.019 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.039 ± 0.000 | - | unsupported | 0.057 ± 0.001 | 0.070 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | unsupported | 0.015 ± 0.002 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.011 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.017 ± 0.000 | - | unsupported | 0.026 ± 0.001 | 0.037 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | unsupported | 0.017 ± 0.001 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.023 ± 0.000 | - | unsupported | 0.031 ± 0.001 | 0.057 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 7.594 ± 0.040 | - | unsupported | 9.071 ± 0.111 | 8.578 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | unsupported | 0.012 ± 0.001 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.379 ± 0.009 | - | unsupported | 0.422 ± 0.013 | 0.477 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.019 ± 0.000 | - | unsupported | 0.025 ± 0.003 | 0.037 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.043 ± 0.000 | - | unsupported | 0.051 ± 0.001 | 0.072 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.461 ± 0.010 | - | unsupported | 0.574 ± 0.013 | 1.197 ± 0.058 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 2.275 ± 0.020 | - | unsupported | 1.897 ± 0.004 | 1.433 ± 0.266 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 5.580 ± 0.021 | - | unsupported | 7.590 ± 0.012 | 3.515 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.181 ± 0.021 | - | unsupported | 1.450 ± 0.019 | 1.427 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | unsupported | 0.013 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.048 ± 0.002 | - | unsupported | 0.090 ± 0.016 | 0.074 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | unsupported | 0.016 ± 0.002 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.040 ± 0.000 | - | unsupported | 0.079 ± 0.005 | 0.068 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.008 ± 0.000 | - | unsupported | 0.015 ± 0.001 | 0.024 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | unsupported | 0.015 ± 0.002 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.009 ± 0.000 | - | unsupported | 0.022 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.018 ± 0.000 | - | unsupported | 0.026 ± 0.001 | 0.037 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | unsupported | 0.015 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.011 ± 0.000 | - | unsupported | 0.025 ± 0.001 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.023 ± 0.000 | - | unsupported | 0.031 ± 0.002 | 0.046 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 9.191 ± 0.042 | - | unsupported | 11.228 ± 0.037 | 11.427 ± 0.550 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | unsupported | 0.017 ± 0.001 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.375 ± 0.013 | - | unsupported | 0.424 ± 0.014 | 0.311 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.019 ± 0.002 | - | unsupported | 0.024 ± 0.001 | 0.037 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.044 ± 0.000 | - | unsupported | 0.051 ± 0.000 | 0.068 ± 0.003 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 1.053 ± 0.007 | - | unsupported | 0.956 ± 0.054 | 3.439 ± 0.044 | 2.528 ± 0.057 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.133 ± 0.001 | - | unsupported | 0.060 ± 0.002 | 0.131 ± 0.002 | 0.131 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.139 ± 0.001 | - | unsupported | 0.083 ± 0.001 | 0.202 ± 0.006 | 0.217 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.344 ± 0.021 | - | unsupported | 0.242 ± 0.004 | 0.527 ± 0.032 | 0.717 ± 0.013 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 1.063 ± 0.011 | - | unsupported | 0.486 ± 0.015 | 1.099 ± 0.075 | 2.457 ± 0.849 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.135 ± 0.004 | - | unsupported | 0.062 ± 0.002 | 0.119 ± 0.018 | 0.100 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.142 ± 0.007 | - | unsupported | 0.080 ± 0.001 | 0.193 ± 0.003 | 0.177 ± 0.013 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.349 ± 0.019 | - | unsupported | 0.160 ± 0.018 | 0.538 ± 0.022 | 0.663 ± 0.011 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.309 ± 0.010 | - | unsupported | 0.373 ± 0.013 | 0.804 ± 0.010 | 0.474 ± 0.010 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.034 ± 0.000 | - | unsupported | 0.046 ± 0.005 | 0.210 ± 0.003 | 0.078 ± 0.006 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.059 ± 0.000 | - | unsupported | 0.060 ± 0.004 | 0.227 ± 0.015 | 0.126 ± 0.007 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.201 ± 0.010 | - | unsupported | 0.166 ± 0.002 | 0.377 ± 0.014 | 0.238 ± 0.013 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.312 ± 0.011 | - | unsupported | 0.558 ± 0.011 | 0.822 ± 0.069 | 0.417 ± 0.024 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.035 ± 0.000 | - | unsupported | 0.076 ± 0.004 | 0.194 ± 0.020 | 0.074 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.059 ± 0.001 | - | unsupported | 0.101 ± 0.001 | 0.234 ± 0.015 | 0.115 ± 0.031 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.201 ± 0.001 | - | unsupported | 0.229 ± 0.016 | 0.409 ± 0.022 | 0.211 ± 0.003 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.192 ± 0.009 | - | 0.195 ± 0.018 | 0.194 ± 0.012 | 0.398 ± 0.044 | 0.329 ± 0.025 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.080 ± 0.001 | - | 0.083 ± 0.016 | 0.074 ± 0.004 | 0.076 ± 0.002 | 0.063 ± 0.009 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.091 ± 0.001 | - | 0.093 ± 0.000 | 0.088 ± 0.009 | 0.101 ± 0.006 | 0.096 ± 0.003 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.121 ± 0.000 | - | 0.125 ± 0.005 | 0.121 ± 0.004 | 0.176 ± 0.005 | 0.192 ± 0.024 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.199 ± 0.020 | - | 0.198 ± 0.010 | 0.258 ± 0.011 | 0.470 ± 0.050 | 0.275 ± 0.009 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.081 ± 0.016 | - | 0.090 ± 0.007 | 0.126 ± 0.005 | 0.081 ± 0.003 | 0.073 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.091 ± 0.002 | - | 0.094 ± 0.000 | 0.144 ± 0.006 | 0.103 ± 0.004 | 0.101 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.121 ± 0.001 | - | 0.125 ± 0.002 | 0.177 ± 0.004 | 0.216 ± 0.014 | 0.154 ± 0.002 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 43.160 ± 0.215 | - | 49.841 ± 0.825 | 60.598 ± 0.971 | 69.722 ± 0.263 | 45.989 ± 18.028 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 43.432 ± 0.216 | - | 50.964 ± 0.447 | 25.156 ± 0.353 | 51.733 ± 0.618 | 37.797 ± 17.314 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 24.878 ± 0.302 | - | 24.181 ± 0.474 | 24.631 ± 0.364 | 21.507 ± 0.124 | 20.973 ± 0.293 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 248.755 ± 12.093 | - | 311.026 ± 61.013 | 11.204 ± 0.190 | 14.168 ± 0.121 | 12.697 ± 1.694 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 7.496 ± 0.097 | - | 7.460 ± 0.037 | 7.934 ± 0.071 | 7.860 ± 0.015 | 7.972 ± 0.101 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 16.155 ± 0.062 | - | 16.138 ± 0.153 | 10.374 ± 0.170 | 12.990 ± 1.343 | 11.549 ± 0.073 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 8.816 ± 0.111 | - | 8.722 ± 0.029 | 8.799 ± 0.020 | 9.496 ± 0.037 | 9.123 ± 0.050 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 12.902 ± 0.083 | - | 12.835 ± 0.089 | 11.753 ± 0.014 | 14.161 ± 0.095 | 13.964 ± 0.230 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 20.835 ± 0.106 | - | 21.285 ± 0.061 | 19.147 ± 0.253 | 37.672 ± 0.648 | 14.748 ± 0.475 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 38.659 ± 0.134 | - | 39.392 ± 0.178 | 15.199 ± 0.178 | 32.646 ± 0.279 | 13.539 ± 1.341 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.488 ± 0.002 | 0.512 ± 0.002 | unsupported | 0.607 ± 0.006 | 0.741 ± 0.029 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 7.767 ± 0.018 | 8.662 ± 0.014 | unsupported | 8.671 ± 0.083 | 9.667 ± 0.055 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.780 ± 0.006 | 0.809 ± 0.006 | unsupported | 0.511 ± 0.008 | 1.189 ± 0.885 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 2274.921 ± 1.640 | 2304.562 ± 1.752 | unsupported | 5.636 ± 0.161 | 9.479 ± 0.388 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 43.573 ± 3.175 | - | 41.004 ± 0.833 | 42.541 ± 0.572 | 31.805 ± 0.583 | 27.918 ± 1.652 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 232.588 ± 3.254 | - | 274.097 ± 12.947 | 21.756 ± 0.070 | 19.932 ± 6.290 | 16.677 ± 0.476 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 17.531 ± 0.118 | 17.233 ± 0.151 | 9.078 ± 0.046 | 56.135 ± 0.720 | 9.636 ± 0.173 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 2288.136 ± 14.214 | 2274.119 ± 12.085 | 6.298 ± 0.084 | 53.788 ± 0.640 | 8.773 ± 0.046 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 25.608 ± 0.171 | - | 25.755 ± 0.482 | 33.169 ± 0.298 | 33.251 ± 0.073 | 31.066 ± 7.541 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 240.264 ± 29.810 | - | 249.832 ± 14.610 | 15.665 ± 0.240 | 23.990 ± 3.686 | 26.186 ± 10.110 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 1.084 ± 0.055 | - | 1.119 ± 0.141 | 1.361 ± 0.056 | 3.955 ± 0.254 | 2.723 ± 0.017 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.511 ± 0.060 | - | 23.998 ± 0.017 | 1.291 ± 0.083 | 12.317 ± 0.524 | 2.674 ± 0.020 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 19.489 ± 0.249 | - | 19.338 ± 0.067 | 20.546 ± 0.269 | 19.618 ± 0.054 | 19.639 ± 3.450 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 228.407 ± 5.920 | - | 210.981 ± 11.437 | 16.820 ± 0.360 | 20.994 ± 1.812 | 21.427 ± 1.611 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 19.790 ± 0.381 | - | 20.171 ± 0.292 | 20.538 ± 0.093 | 19.707 ± 0.129 | 19.747 ± 1.430 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 227.789 ± 14.865 | - | 210.529 ± 12.357 | 16.842 ± 0.213 | 19.711 ± 0.702 | 19.724 ± 2.134 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.164 ± 0.007 | 1.178 ± 0.028 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 21.297 ± 0.126 | 21.346 ± 0.054 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.563 ± 0.006 | 1.599 ± 0.027 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 23.520 ± 0.293 | 23.647 ± 0.299 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.093 ± 0.017 | 1.096 ± 0.011 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 20.347 ± 0.116 | 20.319 ± 0.030 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.488 ± 0.015 | 1.500 ± 0.025 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 22.594 ± 0.671 | 22.616 ± 0.143 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 24.880 ± 0.646 | - | 24.033 ± 0.160 | 24.443 ± 0.268 | 21.582 ± 0.169 | 21.089 ± 0.605 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 137.127 ± 77.925 | - | 316.992 ± 41.018 | 11.268 ± 0.057 | 14.047 ± 0.166 | 12.648 ± 3.644 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 58.248 ± 0.506 | 58.473 ± 0.179 | 61.447 ± 0.597 | 63.486 ± 0.266 | 65.829 ± 7.181 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 2333.574 ± 12.968 | 2343.540 ± 12.046 | 38.359 ± 0.576 | 52.370 ± 0.419 | 57.340 ± 4.257 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 32.858 ± 0.193 | - | 32.803 ± 0.108 | 30.577 ± 0.255 | 126.842 ± 2.637 | 30.592 ± 0.286 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 49.969 ± 2.019 | - | 49.417 ± 0.342 | 24.408 ± 1.059 | 114.993 ± 19.138 | 20.845 ± 0.110 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.140 ± 0.001 | 0.139 ± 0.001 | unsupported | 0.153 ± 0.029 | 0.038 ± 0.005 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.219 ± 0.011 | 2.216 ± 0.007 | unsupported | 2.407 ± 0.009 | 0.630 ± 0.622 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.073 ± 0.000 | 6.000 ± 0.000 | unsupported | 0.077 ± 0.003 | 0.118 ± 0.018 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.898 ± 0.003 | 23.999 ± 0.001 | unsupported | 1.009 ± 0.123 | 0.684 ± 0.021 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.078 ± 0.000 | 0.079 ± 0.000 | unsupported | 0.133 ± 0.013 | 0.250 ± 0.005 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.246 ± 0.004 | 1.238 ± 0.007 | unsupported | 1.738 ± 0.017 | 2.613 ± 0.041 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.037 ± 0.000 | 12.000 ± 0.000 | unsupported | 0.043 ± 0.001 | 0.159 ± 0.003 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.562 ± 0.001 | 11.999 ± 0.000 | unsupported | 0.723 ± 0.010 | 1.117 ± 0.096 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.196 ± 0.000 | 0.199 ± 0.000 | unsupported | 0.275 ± 0.003 | 0.206 ± 0.001 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 3.129 ± 0.005 | 3.121 ± 0.049 | unsupported | 4.374 ± 0.009 | 1.795 ± 0.014 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.076 ± 0.000 | 12.000 ± 0.328 | unsupported | 0.084 ± 0.001 | 0.137 ± 0.014 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 1.186 ± 0.121 | 11.999 ± 0.245 | unsupported | 1.514 ± 0.041 | 0.976 ± 0.100 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.055 ± 0.003 | 0.088 ± 0.005 | unsupported | 0.158 ± 0.001 | 0.181 ± 0.003 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.628 ± 0.007 | 0.670 ± 0.074 | unsupported | 1.714 ± 0.020 | 1.848 ± 0.040 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.089 ± 0.004 | 35.843 ± 3.519 | unsupported | 0.042 ± 0.001 | 0.181 ± 0.001 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.424 ± 0.037 | 96.000 ± 0.002 | unsupported | 0.362 ± 0.002 | 1.159 ± 0.022 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.124 ± 0.007 | 0.151 ± 0.001 | unsupported | 0.129 ± 0.003 | 0.221 ± 0.007 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.636 ± 0.025 | 1.701 ± 0.073 | unsupported | 1.993 ± 0.019 | 2.233 ± 0.036 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.144 ± 0.004 | 44.000 ± 0.187 | unsupported | 0.037 ± 0.000 | 0.211 ± 0.010 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 1.043 ± 0.161 | 120.000 ± 0.002 | unsupported | 0.483 ± 0.015 | 1.331 ± 0.023 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.292 ± 0.004 | 0.325 ± 0.002 | unsupported | unsupported | 0.249 ± 0.022 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 4.294 ± 0.041 | 4.357 ± 0.054 | unsupported | unsupported | 3.152 ± 0.079 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.287 ± 0.007 | 51.859 ± 0.234 | unsupported | unsupported | 0.240 ± 0.004 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 2.022 ± 0.116 | 144.000 ± 1.751 | unsupported | unsupported | 1.804 ± 0.067 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.011 ± 0.000 | 0.019 ± 0.000 | unsupported | 0.013 ± 0.000 | 0.066 ± 0.012 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.105 ± 0.009 | 0.111 ± 0.001 | unsupported | 0.093 ± 0.004 | 0.247 ± 0.002 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.017 ± 0.001 | 6.000 ± 0.078 | unsupported | 0.015 ± 0.000 | 0.039 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.117 ± 0.027 | 24.000 ± 0.002 | unsupported | 0.101 ± 0.004 | 0.170 ± 0.006 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.029 ± 0.004 | 0.048 ± 0.003 | unsupported | 0.100 ± 0.004 | 0.164 ± 0.003 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.379 ± 0.037 | 0.367 ± 0.046 | unsupported | 1.393 ± 0.046 | 1.661 ± 0.064 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.046 ± 0.001 | 18.000 ± 0.141 | unsupported | 0.057 ± 0.001 | 0.154 ± 0.006 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.215 ± 0.010 | 47.999 ± 0.001 | unsupported | 0.275 ± 0.023 | 0.986 ± 0.098 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.054 ± 0.000 | 0.054 ± 0.000 | unsupported | 0.111 ± 0.001 | 0.196 ± 0.002 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.858 ± 0.003 | 0.863 ± 0.011 | unsupported | 1.759 ± 0.023 | 1.709 ± 0.076 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.027 ± 0.000 | 12.000 ± 0.406 | unsupported | 0.094 ± 0.009 | 0.133 ± 0.005 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.392 ± 0.004 | 11.999 ± 0.001 | unsupported | 0.690 ± 0.042 | 0.944 ± 0.359 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.056 ± 0.000 | 0.057 ± 0.000 | unsupported | 0.118 ± 0.002 | 0.193 ± 0.015 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.893 ± 0.005 | 0.894 ± 0.002 | unsupported | 1.798 ± 0.028 | 1.741 ± 0.032 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.028 ± 0.000 | 12.187 ± 0.937 | unsupported | 0.098 ± 0.002 | 0.144 ± 0.009 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.411 ± 0.007 | 11.999 ± 0.002 | unsupported | 0.731 ± 0.051 | 0.943 ± 0.024 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.122 ± 0.001 | 0.152 ± 0.005 | unsupported | 0.132 ± 0.001 | 0.218 ± 0.001 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.655 ± 0.043 | 1.720 ± 0.075 | unsupported | 1.233 ± 0.006 | 2.173 ± 0.046 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.142 ± 0.002 | 43.875 ± 12.963 | unsupported | 0.036 ± 0.001 | 0.205 ± 0.004 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 0.983 ± 0.090 | 119.999 ± 0.126 | unsupported | 0.484 ± 0.004 | 1.277 ± 0.035 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.183 ± 0.001 | 0.183 ± 0.000 | unsupported | 0.268 ± 0.001 | 0.335 ± 0.012 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 2.912 ± 0.012 | 2.903 ± 0.033 | unsupported | 4.332 ± 0.021 | 3.939 ± 0.032 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.073 ± 0.000 | 12.172 ± 1.125 | unsupported | 0.165 ± 0.001 | 0.201 ± 0.003 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 1.124 ± 0.003 | 11.999 ± 0.001 | unsupported | 1.537 ± 0.008 | 2.052 ± 0.011 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.739 ± 0.028 | 1.343 ± 0.008 | unsupported | 0.262 ± 0.003 | 0.213 ± 0.003 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 18.195 ± 0.245 | 29.289 ± 0.370 | unsupported | 5.881 ± 0.080 | 2.268 ± 0.047 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.877 ± 0.031 | 92.902 ± 0.461 | unsupported | 0.229 ± 0.009 | 0.168 ± 0.002 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 8.989 ± 0.173 | 154.250 ± 2.625 | unsupported | 2.288 ± 0.058 | 1.212 ± 0.014 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 27.464 ± 0.261 | - | unsupported | 27.678 ± 0.312 | - | 30.443 ± 0.266 | 38.417 ± 0.698 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 20.883 ± 0.449 | - | unsupported | 20.544 ± 0.094 | - | 30.562 ± 0.446 | 21.103 ± 0.294 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 20.648 ± 0.193 | - | unsupported | 26.073 ± 0.186 | - | 20.592 ± 0.548 | 21.592 ± 0.397 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 14.868 ± 0.706 | - | unsupported | 15.089 ± 0.141 | - | 20.722 ± 0.371 | 14.799 ± 0.738 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 13.490 ± 0.232 | - | unsupported | 22.920 ± 0.328 | - | 13.862 ± 0.409 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 14.408 ± 0.092 | - | unsupported | 14.948 ± 0.242 | - | 13.607 ± 0.147 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 25.909 ± 0.478 | - | unsupported | 27.308 ± 0.373 | - | 30.240 ± 0.112 | 36.506 ± 0.194 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 20.802 ± 0.293 | - | unsupported | 20.548 ± 0.104 | - | 29.718 ± 0.255 | 20.633 ± 0.302 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 48.087 ± 0.257 | - | unsupported | 49.606 ± 0.666 | - | 47.729 ± 0.086 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 52.779 ± 0.738 | - | unsupported | 25.317 ± 0.344 | - | 23.631 ± 0.095 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 47.859 ± 0.252 | - | unsupported | 50.078 ± 1.478 | - | 47.522 ± 0.289 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 51.942 ± 1.139 | - | unsupported | 25.000 ± 0.086 | - | 23.456 ± 0.181 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 27.327 ± 0.326 | - | unsupported | 27.229 ± 0.415 | - | 31.242 ± 0.807 | 38.453 ± 0.397 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 20.604 ± 0.104 | - | unsupported | 20.630 ± 0.161 | - | 30.136 ± 0.392 | 20.952 ± 0.463 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 21.779 ± 0.108 | - | unsupported | 23.679 ± 1.838 | - | 24.207 ± 0.292 | 25.515 ± 0.217 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 15.484 ± 0.042 | - | unsupported | 14.408 ± 6.145 | - | 25.872 ± 0.161 | 15.051 ± 0.566 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.740 ± 0.004 | unsupported | unsupported | 0.899 ± 0.006 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 10.941 ± 0.016 | unsupported | unsupported | 12.624 ± 0.026 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 1.459 ± 0.008 | unsupported | unsupported | 0.614 ± 0.004 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 193.628 ± 2.876 | unsupported | unsupported | 6.713 ± 0.041 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 27.582 ± 0.408 | - | unsupported | 27.622 ± 0.250 | - | 30.929 ± 0.513 | 37.845 ± 0.386 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 20.732 ± 0.417 | - | unsupported | 20.864 ± 0.690 | - | 30.124 ± 0.094 | 20.880 ± 0.199 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 107.314 ± 4.035 | - | 0.002 ± 0.000 | 105.531 ± 1.651 | 104.509 ± 2.353 | 132.817 ± 3.058 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 42.922 ± 0.340 | - | 0.002 ± 0.000 | 34.817 ± 1.513 | 54.906 ± 1.411 | 131.399 ± 2.812 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 66.228 ± 1.743 | - | 204.974 ± 5.932 | 65.499 ± 1.591 | 70.141 ± 0.927 | 76.678 ± 1.875 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 22.686 ± 0.197 | - | 254.923 ± 16.448 | 22.270 ± 0.767 | 34.795 ± 0.711 | 75.992 ± 1.099 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 213.149 ± 1.961 | - | 213.670 ± 3.047 | 210.843 ± 1.757 | 216.267 ± 4.520 | 249.718 ± 5.395 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 70.920 ± 1.013 | - | 216.047 ± 2.218 | 69.120 ± 1.199 | 109.799 ± 1.051 | 249.096 ± 4.149 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 66.341 ± 1.751 | - | 66.125 ± 2.090 | 69.448 ± 1.960 | 66.970 ± 1.986 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 42.035 ± 0.194 | - | 67.623 ± 3.614 | 23.907 ± 1.101 | 34.868 ± 0.175 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 133.088 ± 4.912 | - | 0.002 ± 0.000 | 123.495 ± 2.374 | 117.826 ± 1.258 | 156.688 ± 2.636 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 40.885 ± 0.358 | - | 0.002 ± 0.000 | 39.490 ± 0.503 | 43.701 ± 1.128 | 151.860 ± 0.408 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 82.479 ± 0.875 | - | 0.002 ± 0.000 | 77.584 ± 1.442 | 81.105 ± 1.172 | 213.604 ± 9.384 | 129.025 ± 1.803 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 30.452 ± 0.796 | - | 0.002 ± 0.000 | 76.973 ± 0.769 | 42.251 ± 0.699 | 210.975 ± 4.387 | 52.776 ± 1.559 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 57.520 ± 1.206 | - | 58.481 ± 1.157 | 63.534 ± 0.724 | 64.837 ± 0.730 | 80.826 ± 0.875 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 20.859 ± 0.107 | - | 71.290 ± 0.636 | 23.263 ± 0.258 | 31.590 ± 0.348 | 80.023 ± 0.942 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 57.945 ± 0.738 | - | 57.399 ± 0.651 | 61.952 ± 0.298 | 62.971 ± 1.973 | 80.468 ± 1.736 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 20.381 ± 0.167 | - | 71.237 ± 1.127 | 22.596 ± 0.419 | 31.732 ± 0.854 | 80.382 ± 2.124 | - |
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
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 7.683661 | 480228.81 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 7.190229 | 449389.31 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 7.774051 | 485878.19 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 4.111694 | 256980.88 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 6.188938 | 386808.62 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 831.991510 | 51999469.38 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 18.513655 | 1157103.44 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 18.698831 | 1168676.94 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 5.906310 | 369144.38 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 383.980072 | 23998754.50 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 11.853856 | 740866.00 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 9.704193 | 606512.06 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 7.803279 | 487704.94 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 8.198227 | 512389.19 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 154.677892 | 9667368.25 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 138.741069 | 8671316.81 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 124.271043 | 7766940.19 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 138.586049 | 8661628.06 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 19.027493 | 1189218.31 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 8.170391 | 510649.44 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 12.472050 | 779503.12 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 12.946206 | 809137.88 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 151.663954 | 9478997.12 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 90.183537 | 5636471.06 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 36398.742777 | 2274921423.56 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 36872.999766 | 2304562485.38 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 18.627263 | 1164203.94 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 18.850129 | 1178133.06 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 340.754793 | 21297174.56 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 341.528463 | 21345528.94 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 25.011556 | 1563222.25 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 25.582473 | 1598904.56 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 376.319131 | 23519945.69 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 378.355777 | 23647236.06 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 17.481882 | 1092617.62 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 17.543548 | 1096471.75 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 325.546386 | 20346649.12 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 325.099996 | 20318749.75 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 23.805471 | 1487841.94 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 24.006563 | 1500410.19 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 361.499545 | 22593721.56 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 361.849068 | 22615566.75 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.414513 | 37726.77 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 9.768063 | 152625.98 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.971446 | 140178.84 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.897527 | 139023.86 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.037856 | 629732.00 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 19.258726 | 2407340.75 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 17.754378 | 2219297.25 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 17.724733 | 2215591.62 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.555401 | 118053.14 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.899237 | 76550.58 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.698502 | 73414.09 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 383.981167 | 5999705.73 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.469632 | 683704.00 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.075804 | 1009475.50 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.185290 | 898161.25 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 191.995505 | 23999438.12 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 15.979927 | 249686.36 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.496124 | 132751.94 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.012367 | 78318.23 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.031472 | 78616.75 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 20.900575 | 2612571.88 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 13.905116 | 1738139.50 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.967226 | 1245903.25 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 9.905270 | 1238158.75 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.177675 | 159026.17 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.733469 | 42710.45 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.379878 | 37185.59 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.979084 | 11999673.19 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.938907 | 1117363.38 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.782368 | 722796.00 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.495421 | 561927.62 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.994680 | 11999335.00 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.154156 | 205533.69 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.578395 | 274662.42 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 12.570352 | 196411.75 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 12.732694 | 198948.34 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.360059 | 1795007.38 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 34.995438 | 4374429.75 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 25.029593 | 3128699.12 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 24.969791 | 3121223.88 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.760784 | 136887.25 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.382340 | 84099.06 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.878709 | 76229.83 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.991848 | 11999872.62 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.809806 | 976225.75 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.113451 | 1514181.38 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.484974 | 1185621.75 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.994611 | 11999326.38 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.581156 | 180955.56 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 10.085806 | 157590.72 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.540065 | 55313.52 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.602350 | 87536.72 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.787017 | 1848377.12 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 13.712878 | 1714109.75 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.023467 | 627933.38 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.363863 | 670482.88 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.604299 | 181317.17 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.668698 | 41698.41 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.665178 | 88518.41 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2293.966372 | 35843224.56 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.271968 | 1158996.00 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.897626 | 362203.25 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.392620 | 424077.50 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 767.997724 | 95999715.50 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.114371 | 220537.05 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.239906 | 128748.53 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 7.941090 | 124079.53 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.654391 | 150849.86 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 17.860011 | 2232501.38 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 15.940613 | 1992576.62 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.087407 | 1635925.88 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 13.605775 | 1700721.88 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.495865 | 210872.89 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.357257 | 36832.14 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 9.206207 | 143846.98 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2815.989113 | 43999829.89 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.644313 | 1330539.12 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.864112 | 483014.00 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.342052 | 1042756.50 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 959.998485 | 119999810.62 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 15.962335 | 249411.48 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 18.711438 | 292366.22 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 20.800812 | 325012.69 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 25.217573 | 3152196.62 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 34.355911 | 4294488.88 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 34.853111 | 4356638.88 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 15.358014 | 239968.97 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 18.397236 | 287456.81 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3318.996595 | 51859321.80 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 14.428063 | 1803507.88 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 16.178398 | 2022299.75 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1151.996670 | 143999583.75 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.205009 | 65703.27 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.836374 | 13068.34 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.734653 | 11478.95 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.213187 | 18956.05 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 1.974961 | 246870.12 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.747387 | 93423.38 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.836964 | 104620.50 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.888881 | 111110.12 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.481198 | 38768.72 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.945618 | 14775.28 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.105656 | 17275.88 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 383.976673 | 5999635.52 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.358820 | 169852.50 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.809714 | 101214.25 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.935218 | 116902.25 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 191.996063 | 23999507.88 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.487512 | 163867.38 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.404764 | 100074.44 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.831703 | 28620.36 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.085556 | 48211.81 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.286043 | 1660755.38 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.140310 | 1392538.75 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.032767 | 379095.88 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 2.935014 | 366876.75 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.854829 | 153981.70 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.643691 | 56932.67 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.929095 | 45767.11 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1151.970814 | 17999543.97 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.886769 | 985846.12 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.198249 | 274781.12 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.722970 | 215371.25 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 383.993722 | 47999215.25 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.520601 | 195634.39 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.103821 | 110997.20 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.453814 | 53965.84 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.475976 | 54312.12 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.668377 | 1708547.12 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.073962 | 1759245.25 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.860019 | 857502.38 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.902620 | 862827.50 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.543187 | 133487.30 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.040510 | 94382.97 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.700889 | 26576.39 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.988248 | 11999816.38 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.553598 | 944199.75 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.522733 | 690341.62 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.136192 | 392024.00 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.995949 | 11999493.62 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.338001 | 192781.27 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.546368 | 117912.00 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.584138 | 56002.16 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.637447 | 56835.11 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.928713 | 1741089.12 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.382780 | 1797847.50 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.142818 | 892852.25 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.153508 | 894188.50 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.222756 | 144105.56 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.268987 | 97952.92 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.775619 | 27744.05 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 779.973796 | 12187090.56 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.545252 | 943156.50 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.846618 | 730827.25 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.286993 | 410874.12 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.995538 | 11999442.25 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.971474 | 218304.28 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.463203 | 132237.55 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 7.820205 | 122190.70 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.704325 | 151630.08 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 17.380736 | 2172592.00 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.860566 | 1232570.75 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.241515 | 1655189.38 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 13.759071 | 1719883.88 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.114491 | 204913.92 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.302633 | 35978.64 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 9.086613 | 141978.33 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2807.976649 | 43874635.14 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.216124 | 1277015.50 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.871506 | 483938.25 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.863264 | 982908.00 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 959.993043 | 119999130.38 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 21.433000 | 334890.62 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.131909 | 267686.08 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 11.682011 | 182531.42 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 11.734129 | 183345.77 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 31.515675 | 3939459.38 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 34.656463 | 4332057.88 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 23.294141 | 2911767.62 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 23.221084 | 2902635.50 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.844156 | 200689.94 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 10.528849 | 164513.27 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.640263 | 72504.11 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 778.987303 | 12171676.61 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 16.416292 | 2052036.50 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.298626 | 1537328.25 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.990563 | 1123820.38 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.993177 | 11999147.12 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.650253 | 213285.20 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 16.775624 | 262119.12 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 47.299012 | 739047.06 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 85.976402 | 1343381.28 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.145094 | 2268136.75 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 47.048720 | 5881090.00 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 145.560903 | 18195112.88 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 234.311173 | 29288896.62 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.729955 | 167655.55 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 14.683253 | 229425.83 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 56.103651 | 876619.55 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5945.747353 | 92902302.39 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.694688 | 1211836.00 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 18.304132 | 2288016.50 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 71.913850 | 8989231.25 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1233.997846 | 154249730.75 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 14.385976 | 899123.50 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 11.839920 | 739995.00 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 201.988187 | 12624261.69 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 175.052748 | 10940796.75 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 9.828850 | 614303.12 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 23.337509 | 1458594.31 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 107.405254 | 6712828.38 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 3098.046965 | 193627935.31 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.023564 | 1472.75 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002545 | 159.06 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.022833 | 1427.06 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002736 | 171.00 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000120 | 7.50 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.021240 | 1327.50 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.006953 | 434.56 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000300 | 18.75 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.022382 | 1398.88 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.007053 | 440.81 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000191 | 11.94 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.022182 | 1386.38 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.005410 | 338.12 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000171 | 10.69 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.022172 | 1385.75 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.005430 | 339.38 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000120 | 7.50 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.020478 | 1279.88 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003136 | 196.00 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000101 | 6.31 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.019917 | 1244.81 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003246 | 202.88 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/norm_fro` (c64, threads=1, shape=`2048x1536`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 24.233 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `jax-cpu` is 23.1x faster than the slowest successful cell (`tenferro-trace`, 59.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `julia-base` is 14.8x faster than the slowest successful cell (`tenferro-trace`, 59.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 68.0x faster than the slowest successful cell (`tenferro-trace`, 59.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 18.9x faster than the slowest successful cell (`tenferro-direct`, 134.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `julia-base` is 17.9x faster than the slowest successful cell (`tenferro-direct`, 134.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 23.3x faster than the slowest successful cell (`tenferro-direct`, 134.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 31.3x faster than the slowest successful cell (`tenferro-trace`, 145.977 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `julia-base` is 35.3x faster than the slowest successful cell (`tenferro-trace`, 145.977 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 40.4x faster than the slowest successful cell (`tenferro-trace`, 145.977 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/masked_log_softmax_axis1` (f64, threads=4, shape=`1024x64`): `jax-cpu` is 202.3x faster than the slowest successful cell (`tenferro-eager`, 51.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/masked_log_softmax_axis1` (f64, threads=4, shape=`1024x64`): `tenferro-direct` is 134.4x faster than the slowest successful cell (`tenferro-eager`, 51.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 15.7x faster than the slowest successful cell (`pytorch-cpu`, 24.257 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 24.257 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 12.5x faster than the slowest successful cell (`pytorch-cpu`, 24.257 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 16.0x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 19.4x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 29.8x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 16.6x faster than the slowest successful cell (`tenferro-trace`, 23.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 17.7x faster than the slowest successful cell (`tenferro-trace`, 23.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`tenferro-trace`, 23.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 18.9x faster than the slowest successful cell (`tenferro-trace`, 23.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 23.026 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 23.026 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 23.026 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 18.0x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 19.6x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 29.7x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 21.6x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 25.5x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 25.4x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 48.8x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 14.7x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 20.5x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 36.5x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 21.1x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 24.1x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 26.4x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 56.0x faster than the slowest successful cell (`tenferro-trace`, 23.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 15.0x faster than the slowest successful cell (`tenferro-trace`, 23.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 21.3x faster than the slowest successful cell (`tenferro-trace`, 23.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 59.3x faster than the slowest successful cell (`tenferro-trace`, 23.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 40.3x faster than the slowest successful cell (`tenferro-trace`, 23.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_squares_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 65.0x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=1, shape=`1048576+1048576`): `pytorch-cpu` is 10.3x faster than the slowest successful cell (`julia-base`, 7.509 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `jax-cpu` is 14.0x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `pytorch-cpu` is 36.5x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `tenferro-direct` is 36.1x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `jax-cpu` is 15.1x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `pytorch-cpu` is 34.0x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `tenferro-direct` is 27.7x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 13.0x faster than the slowest successful cell (`julia-base`, 8.299 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 13.7x faster than the slowest successful cell (`julia-base`, 8.299 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `jax-cpu` is 132.1x faster than the slowest successful cell (`tenferro-trace`, 48.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `julia-base` is 36.1x faster than the slowest successful cell (`tenferro-trace`, 48.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 138.2x faster than the slowest successful cell (`tenferro-trace`, 48.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `tenferro-direct` is 111.1x faster than the slowest successful cell (`tenferro-trace`, 48.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `jax-cpu` is 18.7x faster than the slowest successful cell (`tenferro-trace`, 12.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 24.2x faster than the slowest successful cell (`tenferro-trace`, 12.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 33.9x faster than the slowest successful cell (`tenferro-trace`, 12.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `jax-cpu` is 24.7x faster than the slowest successful cell (`tenferro-trace`, 11.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 35.7x faster than the slowest successful cell (`tenferro-trace`, 11.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 29.3x faster than the slowest successful cell (`tenferro-trace`, 11.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `jax-cpu` is 76.9x faster than the slowest successful cell (`tenferro-trace`, 36.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `julia-base` is 26.6x faster than the slowest successful cell (`tenferro-trace`, 36.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 56.0x faster than the slowest successful cell (`tenferro-trace`, 36.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `tenferro-direct` is 29.5x faster than the slowest successful cell (`tenferro-trace`, 36.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 4857.5x faster than the slowest successful cell (`julia-base`, 8.807 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `pytorch-cpu` is 11.1x faster than the slowest successful cell (`julia-base`, 9.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-direct` is 12.4x faster than the slowest successful cell (`julia-base`, 9.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 4979.9x faster than the slowest successful cell (`julia-base`, 9.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.011 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 13.8x faster than the slowest successful cell (`jax-cpu`, 0.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x8x8`): `tenferro-direct` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`3x2x2`): `tenferro-direct` is 11.0x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x2x2`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 16.1x faster than the slowest successful cell (`jax-cpu`, 0.031 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x2x2`): `tenferro-direct` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.030 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=1, shape=`1x2x2`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.021 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.043 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.020 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.024 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x4x4,rhs=1`): `tenferro-direct` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.019 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`3x4x4,rhs=1`): `tenferro-direct` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.023 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `pytorch-cpu` is 15.7x faster than the slowest successful cell (`tenferro-trace`, 5.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `tenferro-direct` is 11.0x faster than the slowest successful cell (`tenferro-trace`, 5.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x4x4,rhs=1`): `tenferro-direct` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.025 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x8x8,rhs=1`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.026 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`3x2x2,rhs=1`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.026 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `jax-cpu` is 32.6x faster than the slowest successful cell (`tenferro-trace`, 23.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `pytorch-cpu` is 41.7x faster than the slowest successful cell (`tenferro-trace`, 23.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `tenferro-direct` is 25.8x faster than the slowest successful cell (`tenferro-trace`, 23.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 22.0x faster than the slowest successful cell (`tenferro-trace`, 311.026 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `julia-base` is 24.5x faster than the slowest successful cell (`tenferro-trace`, 311.026 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 27.8x faster than the slowest successful cell (`tenferro-trace`, 311.026 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`768x384`): `jax-cpu` is 243.1x faster than the slowest successful cell (`tenferro-eager`, 2304.562 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 408.9x faster than the slowest successful cell (`tenferro-eager`, 2304.562 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 13.8x faster than the slowest successful cell (`tenferro-trace`, 274.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `julia-base` is 16.4x faster than the slowest successful cell (`tenferro-trace`, 274.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 12.6x faster than the slowest successful cell (`tenferro-trace`, 274.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `jax-cpu` is 42.5x faster than the slowest successful cell (`tenferro-eager`, 2288.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `julia-base` is 260.8x faster than the slowest successful cell (`tenferro-eager`, 2288.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 363.3x faster than the slowest successful cell (`tenferro-eager`, 2288.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 10.4x faster than the slowest successful cell (`tenferro-trace`, 249.832 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 15.9x faster than the slowest successful cell (`tenferro-trace`, 249.832 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 18.6x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 47.0x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `jax-cpu` is 10.9x faster than the slowest successful cell (`tenferro-direct`, 228.407 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `julia-base` is 10.7x faster than the slowest successful cell (`tenferro-direct`, 228.407 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 13.6x faster than the slowest successful cell (`tenferro-direct`, 228.407 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `jax-cpu` is 11.6x faster than the slowest successful cell (`tenferro-direct`, 227.789 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `julia-base` is 11.5x faster than the slowest successful cell (`tenferro-direct`, 227.789 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 13.5x faster than the slowest successful cell (`tenferro-direct`, 227.789 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 22.6x faster than the slowest successful cell (`tenferro-trace`, 316.992 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `julia-base` is 25.1x faster than the slowest successful cell (`tenferro-trace`, 316.992 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 28.1x faster than the slowest successful cell (`tenferro-trace`, 316.992 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 44.7x faster than the slowest successful cell (`tenferro-trace`, 2343.540 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `julia-base` is 40.9x faster than the slowest successful cell (`tenferro-trace`, 2343.540 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 61.1x faster than the slowest successful cell (`tenferro-trace`, 2343.540 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 50.8x faster than the slowest successful cell (`tenferro-eager`, 6.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 78.4x faster than the slowest successful cell (`tenferro-eager`, 6.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 81.7x faster than the slowest successful cell (`tenferro-eager`, 6.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 35.1x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 23.8x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 26.7x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 75.5x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 281.0x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 322.7x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 10.7x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 16.6x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 21.4x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 87.7x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 142.7x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 157.4x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 12.3x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 197.7x faster than the slowest successful cell (`tenferro-eager`, 35.843 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 859.6x faster than the slowest successful cell (`tenferro-eager`, 35.843 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 404.9x faster than the slowest successful cell (`tenferro-eager`, 35.843 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 82.8x faster than the slowest successful cell (`tenferro-eager`, 96.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 265.0x faster than the slowest successful cell (`tenferro-eager`, 96.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 226.4x faster than the slowest successful cell (`tenferro-eager`, 96.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 208.7x faster than the slowest successful cell (`tenferro-eager`, 44.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 1194.6x faster than the slowest successful cell (`tenferro-eager`, 44.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 305.9x faster than the slowest successful cell (`tenferro-eager`, 44.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 90.2x faster than the slowest successful cell (`tenferro-eager`, 120.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 248.4x faster than the slowest successful cell (`tenferro-eager`, 120.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 115.1x faster than the slowest successful cell (`tenferro-eager`, 120.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 216.1x faster than the slowest successful cell (`tenferro-eager`, 51.859 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 180.4x faster than the slowest successful cell (`tenferro-eager`, 51.859 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 79.8x faster than the slowest successful cell (`tenferro-eager`, 144.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 71.2x faster than the slowest successful cell (`tenferro-eager`, 144.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 154.8x faster than the slowest successful cell (`tenferro-eager`, 6.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 406.1x faster than the slowest successful cell (`tenferro-eager`, 6.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 347.3x faster than the slowest successful cell (`tenferro-eager`, 6.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 141.3x faster than the slowest successful cell (`tenferro-eager`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 237.1x faster than the slowest successful cell (`tenferro-eager`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 205.3x faster than the slowest successful cell (`tenferro-eager`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 116.9x faster than the slowest successful cell (`tenferro-eager`, 18.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 316.2x faster than the slowest successful cell (`tenferro-eager`, 18.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 393.3x faster than the slowest successful cell (`tenferro-eager`, 18.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 48.7x faster than the slowest successful cell (`tenferro-eager`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 174.7x faster than the slowest successful cell (`tenferro-eager`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 222.9x faster than the slowest successful cell (`tenferro-eager`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 89.9x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 127.1x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 451.5x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 12.7x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 17.4x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 30.6x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 84.6x faster than the slowest successful cell (`tenferro-eager`, 12.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 124.4x faster than the slowest successful cell (`tenferro-eager`, 12.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 439.3x faster than the slowest successful cell (`tenferro-eager`, 12.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 12.7x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 16.4x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 29.2x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 214.1x faster than the slowest successful cell (`tenferro-eager`, 43.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 1219.5x faster than the slowest successful cell (`tenferro-eager`, 43.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 309.0x faster than the slowest successful cell (`tenferro-eager`, 43.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 94.0x faster than the slowest successful cell (`tenferro-eager`, 119.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 248.0x faster than the slowest successful cell (`tenferro-eager`, 119.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 122.1x faster than the slowest successful cell (`tenferro-eager`, 119.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 60.6x faster than the slowest successful cell (`tenferro-eager`, 12.172 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 74.0x faster than the slowest successful cell (`tenferro-eager`, 12.172 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 167.9x faster than the slowest successful cell (`tenferro-eager`, 12.172 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 12.9x faster than the slowest successful cell (`tenferro-eager`, 29.289 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 554.1x faster than the slowest successful cell (`tenferro-eager`, 92.902 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 404.9x faster than the slowest successful cell (`tenferro-eager`, 92.902 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 106.0x faster than the slowest successful cell (`tenferro-eager`, 92.902 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 127.3x faster than the slowest successful cell (`tenferro-eager`, 154.250 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 67.4x faster than the slowest successful cell (`tenferro-eager`, 154.250 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 17.2x faster than the slowest successful cell (`tenferro-eager`, 154.250 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/output_reuse/solve_read_into` (f64, threads=4, shape=`768x768,rhs=16`): `pytorch-cpu` is 28.8x faster than the slowest successful cell (`tenferro-direct`, 193.628 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 76200.3x faster than the slowest successful cell (`julia-base`, 132.817 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 74914.1x faster than the slowest successful cell (`julia-base`, 131.399 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/cast_f64_f32` (f64->f32, threads=4, shape=`33554432`): `pytorch-cpu` is 11.4x faster than the slowest successful cell (`tenferro-trace`, 254.923 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/cast_f64_f32` (f64->f32, threads=4, shape=`33554432`): `tenferro-direct` is 11.2x faster than the slowest successful cell (`tenferro-trace`, 254.923 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 88324.7x faster than the slowest successful cell (`julia-base`, 156.688 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 86579.5x faster than the slowest successful cell (`julia-base`, 151.860 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 129929.7x faster than the slowest successful cell (`julia-base`, 213.604 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 129194.4x faster than the slowest successful cell (`julia-base`, 210.975 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 166.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 73.6x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 115.5x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 126.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 160.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 207.5x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 9249.3 GB/s at 0.002 ms. The conservative physical lower bound is 0.168 ms, so the cell is 92.5x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 9193.5 GB/s at 0.002 ms. The conservative physical lower bound is 0.042 ms, so the cell is 23.0x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 154007.7 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1540.1x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 153041.9 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 382.6x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 151316.5 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1513.2x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 153041.9 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 382.6x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 81641.0 GB/s at 0.002 ms. The conservative physical lower bound is 1.342 ms, so the cell is 816.4x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 82190.9 GB/s at 0.002 ms. The conservative physical lower bound is 0.336 ms, so the cell is 205.5x faster than the bound.

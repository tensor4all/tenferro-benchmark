# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/amd-cpu/cpu/public_api/20261106_190329/run.yaml`
- Timestamp: `2026-11-06T19:03:29.179388+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/amd-cpu/cpu/public_api/20261106_190329`.

- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

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

- CSV: `data/results/amd-cpu/cpu/public_api/20261106_190329/cpu_public_api_t1_20261106_190329.csv`
- CSV: `data/results/amd-cpu/cpu/public_api/20261106_190329/cpu_public_api_t4_20261106_190329.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 2.180 ± 0.098 | - | 2.329 ± 0.108 | 3.011 ± 0.149 | 3.388 ± 0.081 | 5.499 ± 1.889 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 1.397 ± 0.025 | - | 2.999 ± 0.000 | 1.612 ± 0.030 | 2.993 ± 1.319 | 2.493 ± 0.151 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 109.077 ± 2.423 | - | 106.272 ± 0.395 | 108.785 ± 1.817 | 104.041 ± 2.880 | 111.842 ± 0.401 | 127.740 ± 0.618 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 39.432 ± 0.669 | - | 39.333 ± 0.455 | 39.035 ± 0.495 | 56.783 ± 0.509 | 111.660 ± 0.384 | 53.792 ± 1.533 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 90.028 ± 0.340 | - | 90.054 ± 0.304 | 82.108 ± 0.789 | 83.690 ± 1.612 | 101.375 ± 1.134 | 105.590 ± 0.244 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 38.059 ± 0.561 | - | 37.925 ± 0.430 | 27.947 ± 0.258 | 43.743 ± 0.914 | 101.221 ± 0.840 | 47.766 ± 0.720 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 23.376 ± 0.610 | - | 23.125 ± 0.482 | 22.876 ± 0.403 | 70.987 ± 0.583 | 26.335 ± 0.536 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 12.435 ± 0.261 | - | 12.612 ± 0.439 | 11.998 ± 0.271 | 35.555 ± 0.784 | 13.667 ± 0.092 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 24.361 ± 0.285 | - | unsupported | 24.375 ± 0.243 | 78.908 ± 0.616 | 25.499 ± 2.666 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 13.555 ± 0.202 | - | unsupported | 12.401 ± 0.406 | 37.461 ± 1.106 | 13.401 ± 1.392 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 10.631 ± 0.323 | - | 10.702 ± 0.259 | 10.115 ± 0.094 | 11.326 ± 0.053 | 11.137 ± 0.288 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 12.698 ± 0.098 | - | 19.275 ± 0.298 | 10.388 ± 0.223 | 18.117 ± 0.376 | 18.179 ± 0.430 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 52.122 ± 0.345 | - | 51.958 ± 0.257 | 99.062 ± 0.358 | 93.411 ± 0.466 | 67.724 ± 0.555 | 68.730 ± 5.650 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 25.084 ± 0.807 | - | 24.759 ± 0.503 | 42.876 ± 0.495 | 47.197 ± 0.214 | 69.398 ± 2.047 | 31.677 ± 7.425 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 90.589 ± 0.886 | - | 90.342 ± 0.147 | 129.986 ± 0.320 | 192.939 ± 1.314 | 110.631 ± 0.555 | 112.608 ± 0.469 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 40.650 ± 0.190 | - | 40.761 ± 0.535 | 55.039 ± 1.087 | 76.710 ± 0.715 | 112.653 ± 0.632 | 51.481 ± 0.918 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 60.571 ± 0.863 | - | 61.655 ± 0.365 | 63.781 ± 0.700 | 63.222 ± 1.822 | 66.619 ± 0.634 | 69.154 ± 0.241 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 22.798 ± 0.410 | - | 23.421 ± 0.314 | 22.539 ± 0.297 | 34.041 ± 0.903 | 64.925 ± 0.386 | 29.956 ± 0.259 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 2.219 ± 0.280 | - | 16.732 ± 0.081 | 15.643 ± 0.124 | 14.186 ± 0.293 | 3.225 ± 0.379 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 5.999 ± 0.002 | - | 7.032 ± 0.070 | 7.846 ± 0.025 | 4.207 ± 1.245 | 3.079 ± 0.457 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 3.851 ± 0.093 | - | 4.041 ± 0.042 | 4.507 ± 0.044 | 5.198 ± 0.166 | 5.714 ± 0.300 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 2.973 ± 0.077 | - | 9.421 ± 0.264 | 2.950 ± 0.046 | 6.484 ± 0.191 | 5.347 ± 0.133 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 2.554 ± 0.053 | - | 2.795 ± 0.071 | 2.950 ± 0.075 | 3.939 ± 0.089 | 3.924 ± 0.280 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 1.994 ± 0.064 | - | 9.027 ± 0.157 | 2.003 ± 0.079 | 3.404 ± 2.644 | 2.927 ± 0.068 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 3.318 ± 0.083 | - | 3.373 ± 0.037 | 3.432 ± 0.055 | 4.683 ± 0.163 | 4.649 ± 0.207 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 3.578 ± 0.156 | - | 11.142 ± 0.446 | 3.504 ± 0.154 | 6.579 ± 0.209 | 6.452 ± 0.123 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 23.268 ± 0.617 | - | 23.194 ± 1.004 | 23.107 ± 0.923 | 70.715 ± 0.388 | 24.249 ± 0.437 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 12.468 ± 0.289 | - | 12.484 ± 0.179 | 12.630 ± 0.437 | 35.781 ± 0.612 | 12.218 ± 0.522 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 24.694 ± 0.461 | - | unsupported | 24.288 ± 0.366 | 65.445 ± 1.003 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 12.395 ± 0.296 | - | unsupported | 12.276 ± 0.232 | 32.728 ± 1.055 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 104.506 ± 0.539 | - | 102.768 ± 0.294 | 106.863 ± 1.258 | 112.375 ± 1.916 | 110.787 ± 0.450 | 130.022 ± 0.262 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 38.659 ± 0.216 | - | 38.866 ± 0.239 | 38.540 ± 0.197 | 59.685 ± 1.985 | 118.155 ± 0.504 | 57.848 ± 0.696 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 116.939 ± 0.391 | - | 116.605 ± 0.505 | 122.184 ± 1.764 | 138.743 ± 1.490 | 123.629 ± 1.188 | 141.775 ± 0.653 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 45.173 ± 0.618 | - | 44.157 ± 0.581 | 45.643 ± 24.807 | 72.919 ± 0.938 | 122.591 ± 0.433 | 60.840 ± 2.408 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 19.080 ± 0.459 | - | 55.213 ± 1.311 | 48.616 ± 0.261 | 54.836 ± 0.361 | 61.947 ± 4.225 | 63.915 ± 6.075 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 8.079 ± 0.140 | - | 22.726 ± 0.644 | 22.776 ± 0.402 | 27.834 ± 0.608 | 62.588 ± 6.220 | 27.481 ± 10.697 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 16.600 ± 0.189 | - | 16.375 ± 0.212 | 34.886 ± 0.328 | 40.587 ± 0.170 | 36.701 ± 3.840 | 38.171 ± 6.141 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 7.780 ± 0.129 | - | 7.890 ± 0.142 | 13.147 ± 0.214 | 21.132 ± 0.654 | 33.832 ± 5.452 | 13.646 ± 6.978 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 32.482 ± 0.167 | - | 32.190 ± 0.184 | 59.601 ± 0.237 | 46.307 ± 0.230 | 32.204 ± 0.400 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 16.561 ± 0.203 | - | 16.505 ± 0.182 | 22.119 ± 0.528 | 23.684 ± 0.170 | 31.635 ± 0.397 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 15.216 ± 0.133 | - | 15.609 ± 0.256 | 31.925 ± 0.490 | 74.566 ± 0.344 | 55.360 ± 6.017 | 57.841 ± 5.770 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 6.598 ± 0.159 | - | 6.574 ± 0.076 | 11.602 ± 0.195 | 38.558 ± 0.645 | 55.106 ± 15.664 | 32.308 ± 7.856 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 117.991 ± 1.939 | - | 116.564 ± 0.815 | 120.889 ± 2.106 | 166.409 ± 4.126 | 124.260 ± 0.684 | 142.283 ± 0.123 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 44.790 ± 0.271 | - | 44.356 ± 0.524 | 44.647 ± 0.706 | 86.834 ± 1.124 | 129.209 ± 1.094 | 60.905 ± 3.999 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 11.409 ± 0.281 | - | 11.548 ± 0.893 | 29.101 ± 0.309 | 33.874 ± 0.237 | 50.863 ± 6.550 | 53.938 ± 5.886 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 4.461 ± 0.262 | - | 4.464 ± 0.144 | 10.772 ± 0.229 | 18.138 ± 0.443 | 57.325 ± 5.611 | 28.894 ± 6.815 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 12.201 ± 0.137 | - | 12.271 ± 0.239 | 22.212 ± 0.229 | 20.646 ± 0.391 | 29.968 ± 4.034 | 32.022 ± 6.182 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 5.278 ± 0.120 | - | 5.264 ± 0.118 | 9.352 ± 0.301 | 10.884 ± 0.443 | 30.347 ± 5.855 | 13.762 ± 5.029 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 15.717 ± 0.155 | - | 16.091 ± 0.125 | 31.565 ± 0.240 | 68.072 ± 0.637 | 69.321 ± 13.938 | 75.947 ± 3.742 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 7.022 ± 0.118 | - | 7.099 ± 0.137 | 11.558 ± 0.332 | 34.620 ± 0.474 | 69.946 ± 7.812 | 31.609 ± 11.414 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 7.498 ± 0.108 | - | 7.383 ± 0.151 | 19.279 ± 0.317 | 44.230 ± 0.510 | 36.765 ± 3.934 | 39.583 ± 5.458 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 3.408 ± 0.115 | - | 3.431 ± 0.072 | 8.051 ± 0.404 | 22.658 ± 0.291 | 36.945 ± 5.992 | 16.463 ± 7.173 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.331 ± 0.044 | 0.351 ± 0.032 | unsupported | unsupported | 0.475 ± 0.037 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.219 ± 0.014 | 0.263 ± 0.007 | unsupported | unsupported | 0.375 ± 0.016 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 116.796 ± 0.791 | - | 115.355 ± 0.138 | 122.575 ± 1.742 | 145.886 ± 1.113 | 126.409 ± 0.751 | 142.744 ± 0.334 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 45.875 ± 1.369 | - | 45.261 ± 0.110 | 45.787 ± 0.614 | 77.231 ± 0.235 | 128.641 ± 1.323 | 63.664 ± 0.850 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 115.544 ± 1.275 | - | 115.547 ± 0.643 | 122.712 ± 1.611 | 146.102 ± 1.247 | 128.164 ± 0.941 | 142.356 ± 0.601 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 45.892 ± 0.547 | - | 44.471 ± 0.420 | 45.641 ± 0.428 | 77.277 ± 0.243 | 133.316 ± 1.215 | 63.287 ± 1.726 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 117.205 ± 0.519 | - | 116.063 ± 0.293 | 120.827 ± 2.434 | 140.268 ± 1.206 | 124.350 ± 0.158 | 140.346 ± 0.397 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 44.253 ± 1.262 | - | 44.354 ± 1.337 | 45.023 ± 0.998 | 72.970 ± 0.668 | 130.857 ± 1.278 | 62.339 ± 2.749 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 104.798 ± 0.792 | - | 103.168 ± 0.346 | 106.811 ± 0.789 | 112.571 ± 3.041 | 111.764 ± 0.386 | 128.850 ± 0.682 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 38.710 ± 0.324 | - | 38.510 ± 0.436 | 38.508 ± 0.707 | 58.334 ± 0.718 | 116.511 ± 0.931 | 57.867 ± 0.819 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 67.584 ± 0.485 | - | 67.333 ± 0.394 | 35.547 ± 0.535 | 77.398 ± 0.400 | 117.936 ± 6.835 | 87.455 ± 7.405 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 28.594 ± 0.327 | - | 28.495 ± 0.159 | 13.560 ± 0.388 | 39.557 ± 0.502 | 118.050 ± 1.122 | 37.638 ± 5.909 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 34.241 ± 0.051 | - | 34.045 ± 0.252 | 26.417 ± 0.288 | 87.667 ± 0.301 | 29.082 ± 0.181 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 11.068 ± 0.037 | - | 11.070 ± 0.115 | 9.615 ± 0.125 | 44.289 ± 0.122 | 28.979 ± 0.231 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 2.049 ± 0.021 | - | 2.074 ± 0.033 | 17.420 ± 0.195 | 1.827 ± 0.035 | 1.534 ± 0.130 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.908 ± 0.021 | - | 0.904 ± 0.015 | 12.568 ± 0.134 | 1.627 ± 0.026 | 1.425 ± 0.081 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 1.149 ± 0.030 | - | 1.143 ± 0.016 | 2.914 ± 0.062 | 5.069 ± 0.017 | 1.300 ± 0.047 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 0.575 ± 0.015 | - | 0.623 ± 0.019 | 2.056 ± 0.374 | 2.642 ± 0.049 | 1.229 ± 0.052 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 34.152 ± 0.273 | - | 33.670 ± 0.286 | 26.546 ± 0.155 | 87.805 ± 0.237 | 29.748 ± 0.285 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 10.943 ± 0.151 | - | 10.926 ± 0.109 | 9.313 ± 0.073 | 44.288 ± 0.316 | 29.411 ± 0.463 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 2.040 ± 0.018 | - | 2.041 ± 0.030 | 17.183 ± 0.173 | 1.775 ± 0.040 | 1.597 ± 0.076 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.889 ± 0.006 | - | 0.897 ± 0.004 | 11.853 ± 0.059 | 0.994 ± 0.005 | 1.455 ± 0.026 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 9.246 ± 0.174 | - | 9.232 ± 0.463 | 21.429 ± 0.852 | 22.819 ± 0.204 | 12.144 ± 0.311 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 3.539 ± 0.132 | - | 3.741 ± 0.216 | 9.735 ± 0.496 | 11.363 ± 0.281 | 11.213 ± 0.602 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 15.887 ± 0.248 | - | 16.010 ± 0.218 | 17.127 ± 0.224 | 42.343 ± 0.268 | 17.290 ± 0.293 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 7.708 ± 0.232 | - | 7.839 ± 0.088 | 8.244 ± 0.151 | 21.377 ± 0.135 | 17.109 ± 0.165 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 1.146 ± 0.028 | - | 1.162 ± 0.018 | 1.244 ± 0.038 | 1.231 ± 0.019 | 1.181 ± 0.030 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.584 ± 0.011 | - | 0.609 ± 0.011 | 0.859 ± 0.019 | 0.736 ± 0.031 | 1.121 ± 0.013 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.170 ± 0.033 | - | 1.184 ± 0.016 | 1.136 ± 0.001 | 1.417 ± 0.083 | 1.210 ± 0.048 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.583 ± 0.017 | - | 0.615 ± 0.019 | 0.592 ± 0.010 | 0.884 ± 0.024 | 1.124 ± 0.030 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 16.212 ± 0.118 | - | 16.147 ± 0.259 | 17.453 ± 0.164 | 25.949 ± 0.205 | 17.562 ± 0.229 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 7.823 ± 0.181 | - | 7.942 ± 0.163 | 8.350 ± 0.109 | 13.361 ± 0.112 | 17.152 ± 0.123 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 1.147 ± 0.014 | - | 1.169 ± 0.023 | 1.228 ± 0.011 | 1.252 ± 0.009 | 1.166 ± 0.016 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.559 ± 0.028 | - | 0.593 ± 0.016 | 0.810 ± 0.015 | 0.738 ± 0.021 | 1.120 ± 0.012 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.139 ± 0.011 | - | 1.146 ± 0.008 | 1.188 ± 0.031 | 1.312 ± 0.034 | 1.164 ± 0.037 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.572 ± 0.020 | - | 0.624 ± 0.017 | 0.595 ± 0.010 | 0.740 ± 0.029 | 1.127 ± 0.010 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 1.160 ± 0.016 | 1.184 ± 0.019 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.596 ± 0.010 | 0.594 ± 0.005 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 37.411 ± 0.170 | - | 37.546 ± 0.504 | 55.132 ± 0.518 | 66.426 ± 1.048 | 61.527 ± 6.635 | 70.303 ± 4.026 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 14.615 ± 0.487 | - | 14.716 ± 0.159 | 21.226 ± 0.212 | 39.132 ± 5.646 | 60.995 ± 3.693 | 29.646 ± 5.844 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 144.967 ± 0.388 | - | 144.505 ± 0.669 | 146.600 ± 0.671 | 210.962 ± 1.440 | 208.199 ± 0.491 | 208.053 ± 0.367 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 47.438 ± 0.414 | - | 46.695 ± 0.278 | 45.233 ± 0.247 | 107.047 ± 1.338 | 212.738 ± 0.755 | 86.444 ± 3.746 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 120.591 ± 0.723 | - | 119.679 ± 0.738 | 132.956 ± 0.313 | 132.232 ± 2.886 | 127.307 ± 0.637 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 48.826 ± 0.937 | - | 48.369 ± 1.008 | 48.919 ± 0.579 | 71.030 ± 0.368 | 132.967 ± 1.805 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 107.271 ± 0.590 | - | 105.867 ± 0.560 | 110.918 ± 0.518 | 117.848 ± 0.512 | 114.996 ± 0.438 | 130.433 ± 0.418 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 39.294 ± 1.280 | - | 38.856 ± 0.631 | 40.329 ± 0.664 | 62.960 ± 0.594 | 119.855 ± 0.342 | 58.067 ± 3.546 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 13.048 ± 0.156 | - | 14.435 ± 1.626 | 33.076 ± 0.452 | 75.497 ± 0.592 | 57.160 ± 6.872 | 56.395 ± 5.799 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 5.521 ± 0.155 | - | 5.622 ± 0.034 | 11.788 ± 0.274 | 38.586 ± 0.430 | 56.866 ± 8.594 | 24.233 ± 10.334 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 123.076 ± 1.458 | - | 121.641 ± 1.067 | 115.297 ± 0.540 | 129.239 ± 1.489 | 163.090 ± 0.368 | 163.628 ± 0.720 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 42.320 ± 0.320 | - | 42.009 ± 0.678 | 40.864 ± 0.700 | 68.580 ± 0.893 | 167.335 ± 1.297 | 70.459 ± 1.640 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 116.863 ± 0.658 | - | 115.905 ± 0.299 | 122.067 ± 1.914 | 139.027 ± 1.139 | 123.227 ± 1.649 | 141.317 ± 0.454 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 45.165 ± 0.602 | - | 43.866 ± 0.629 | 45.394 ± 0.759 | 73.073 ± 2.110 | 126.671 ± 0.504 | 62.095 ± 2.018 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 32.146 ± 0.072 | - | 32.206 ± 0.123 | 35.318 ± 0.283 | 32.189 ± 0.386 | 61.768 ± 6.020 | 67.616 ± 17.526 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 14.166 ± 0.177 | - | 14.514 ± 0.279 | 13.852 ± 0.304 | 17.607 ± 0.663 | 68.755 ± 6.332 | 27.746 ± 7.685 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 120.545 ± 0.380 | 121.076 ± 1.865 | unsupported | 132.890 ± 0.702 | 133.331 ± 2.239 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 48.173 ± 0.612 | 48.434 ± 0.406 | unsupported | 48.938 ± 0.653 | 71.077 ± 1.931 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 1.169 ± 0.013 | - | 1.178 ± 0.013 | 1.184 ± 0.016 | 1.212 ± 0.014 | 7.438 ± 5.007 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.613 ± 0.005 | - | 0.620 ± 0.013 | 0.633 ± 0.009 | 1.193 ± 0.011 | 7.870 ± 4.843 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 1.190 ± 0.055 | - | 1.174 ± 0.006 | 1.177 ± 0.010 | 1.200 ± 0.037 | 6.993 ± 5.079 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.602 ± 0.013 | - | 0.616 ± 0.014 | 0.618 ± 0.013 | 0.658 ± 0.023 | 1.574 ± 1.121 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.784 ± 0.012 | - | unsupported | 1.822 ± 0.044 | 1.918 ± 0.039 | 3.902 ± 5.740 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.923 ± 0.018 | - | unsupported | 1.006 ± 0.034 | 1.547 ± 0.225 | 7.857 ± 4.336 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.490 ± 0.012 | - | 0.652 ± 0.013 | 0.808 ± 0.028 | 0.430 ± 0.030 | 1.143 ± 0.035 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.361 ± 0.035 | - | 0.465 ± 0.020 | 0.466 ± 0.018 | 0.391 ± 0.055 | 1.148 ± 0.009 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 1.179 ± 0.013 | - | 1.179 ± 0.015 | 1.813 ± 0.011 | 1.217 ± 0.026 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.622 ± 0.014 | - | 0.621 ± 0.012 | 0.963 ± 0.007 | 0.660 ± 0.020 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 1.168 ± 0.008 | - | 1.170 ± 0.013 | 1.180 ± 0.025 | 1.212 ± 0.022 | 2.382 ± 4.911 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.606 ± 0.013 | - | 0.619 ± 0.013 | 0.626 ± 0.011 | 0.654 ± 0.017 | 6.709 ± 5.377 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.770 ± 0.010 | - | 1.023 ± 0.005 | 1.391 ± 0.015 | 0.778 ± 0.028 | 1.858 ± 0.630 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.968 ± 0.007 | - | 1.096 ± 0.015 | 1.198 ± 0.022 | 1.151 ± 0.454 | 1.900 ± 0.240 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.849 ± 0.066 | - | 0.002 ± 0.000 | 1.848 ± 0.078 | 1.876 ± 0.055 | 8.414 ± 1.464 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.909 ± 0.013 | - | 0.003 ± 0.000 | 0.914 ± 0.030 | 0.959 ± 0.019 | 8.465 ± 0.996 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.048 ± 0.000 | - | 0.053 ± 0.005 | 0.080 ± 0.012 | 0.108 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.072 ± 0.000 | - | 0.079 ± 0.001 | 0.096 ± 0.003 | 0.152 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.272 ± 0.008 | - | 0.301 ± 0.012 | 0.325 ± 0.011 | 0.291 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.050 ± 0.001 | - | 0.061 ± 0.002 | 0.138 ± 0.004 | 0.109 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.046 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.044 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.438 ± 0.019 | - | 0.526 ± 0.012 | 1.096 ± 0.004 | 0.664 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.017 ± 0.000 | - | 0.023 ± 0.000 | 0.032 ± 0.000 | 0.073 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.002 | 0.007 ± 0.000 | 0.050 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.048 ± 0.002 | - | 0.062 ± 0.002 | 0.061 ± 0.001 | 0.088 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.071 ± 0.001 | - | 0.112 ± 0.002 | 0.106 ± 0.001 | 0.126 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.282 ± 0.005 | - | 0.365 ± 0.010 | 0.333 ± 0.012 | 0.326 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.053 ± 0.001 | - | 0.074 ± 0.000 | 0.109 ± 0.003 | 0.150 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.004 ± 0.002 | - | 0.010 ± 0.000 | 0.017 ± 0.000 | 0.029 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.001 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.001 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.001 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.017 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.001 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.001 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.412 ± 0.011 | - | 0.566 ± 0.005 | 0.784 ± 0.016 | 1.948 ± 1.813 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.018 ± 0.000 | - | 0.034 ± 0.000 | 0.047 ± 0.003 | 0.048 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.004 ± 0.000 | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.026 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.475 ± 0.011 | - | 0.483 ± 0.011 | 0.654 ± 0.017 | 0.754 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 2.176 ± 0.023 | - | 2.146 ± 0.073 | 2.211 ± 0.033 | 2.629 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 6.637 ± 0.105 | - | 7.177 ± 0.185 | 6.943 ± 0.135 | 8.775 ± 0.126 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 1.938 ± 0.055 | - | 1.972 ± 0.009 | 1.566 ± 0.048 | 2.474 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.001 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.093 ± 0.002 | - | 0.098 ± 0.003 | 0.100 ± 0.010 | 0.172 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.012 ± 0.000 | - | 0.015 ± 0.000 | 0.014 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.004 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.012 ± 0.001 | 0.013 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.026 ± 0.000 | - | 0.029 ± 0.000 | 0.029 ± 0.002 | 0.061 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.011 ± 0.000 | - | 0.014 ± 0.000 | 0.015 ± 0.003 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.030 ± 0.001 | - | 0.033 ± 0.004 | 0.035 ± 0.001 | 0.072 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 15.735 ± 0.274 | - | 16.243 ± 0.221 | 12.290 ± 0.405 | 20.976 ± 0.484 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.647 ± 0.043 | - | 0.701 ± 0.031 | 0.708 ± 0.017 | 0.879 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.019 ± 0.000 | - | 0.023 ± 0.000 | 0.025 ± 0.000 | 0.032 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.056 ± 0.001 | - | 0.059 ± 0.000 | 0.061 ± 0.002 | 0.130 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.912 ± 0.022 | - | 0.872 ± 0.008 | 0.659 ± 0.153 | 0.773 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 3.527 ± 0.060 | - | 3.264 ± 0.119 | 2.228 ± 0.048 | 2.590 ± 1.229 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 9.867 ± 3.159 | - | 9.096 ± 0.086 | 8.192 ± 0.185 | 6.723 ± 0.748 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 2.444 ± 0.092 | - | 9.675 ± 0.257 | 1.337 ± 0.035 | 4.987 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.098 ± 0.004 | - | 0.116 ± 0.004 | 0.136 ± 0.019 | 0.156 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.007 ± 0.001 | - | 0.013 ± 0.001 | 0.009 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.014 ± 0.001 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.005 ± 0.001 | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.015 ± 0.000 | - | 0.017 ± 0.000 | 0.013 ± 0.001 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.033 ± 0.000 | - | 0.035 ± 0.000 | 0.027 ± 0.001 | 0.055 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.005 ± 0.001 | - | 0.013 ± 0.001 | 0.013 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.017 ± 0.001 | - | 0.020 ± 0.000 | 0.015 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.043 ± 0.001 | - | 0.044 ± 0.000 | 0.035 ± 0.000 | 0.067 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 18.481 ± 0.215 | - | 28.560 ± 2.676 | 10.304 ± 0.238 | 28.993 ± 3.686 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.008 ± 0.000 | - | 0.017 ± 0.000 | 0.017 ± 0.002 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.751 ± 0.034 | - | 0.875 ± 0.075 | 0.890 ± 0.022 | 1.994 ± 0.550 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.031 ± 0.001 | - | 0.032 ± 0.001 | 0.024 ± 0.000 | 0.034 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.082 ± 0.002 | - | 0.082 ± 0.003 | 0.063 ± 0.002 | 0.095 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.527 ± 0.018 | - | 0.548 ± 0.007 | 0.446 ± 0.011 | 0.397 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 2.011 ± 0.014 | - | 2.002 ± 0.007 | 1.745 ± 0.003 | 1.852 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 6.150 ± 0.124 | - | 6.166 ± 0.077 | 5.171 ± 0.113 | 6.186 ± 0.078 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 0.933 ± 0.021 | - | 0.943 ± 0.028 | 0.894 ± 0.009 | 1.284 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.002 ± 0.001 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.065 ± 0.006 | - | 0.067 ± 0.005 | 0.066 ± 0.002 | 0.140 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.012 ± 0.001 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.023 ± 0.001 | - | 0.026 ± 0.002 | 0.022 ± 0.001 | 0.035 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.015 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.026 ± 0.000 | - | 0.032 ± 0.001 | 0.027 ± 0.000 | 0.038 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 7.446 ± 0.135 | - | 7.285 ± 0.104 | 7.142 ± 0.115 | 9.585 ± 0.287 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.509 ± 0.006 | - | 0.507 ± 0.009 | 0.455 ± 0.013 | 0.677 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.017 ± 0.000 | - | 0.020 ± 0.000 | 0.022 ± 0.000 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.050 ± 0.001 | - | 0.053 ± 0.000 | 0.046 ± 0.001 | 0.086 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.912 ± 0.011 | - | 0.908 ± 0.006 | 0.613 ± 0.172 | 0.418 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 3.293 ± 0.018 | - | 3.399 ± 0.076 | 1.748 ± 0.017 | 1.664 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 6.302 ± 1.194 | - | 8.030 ± 0.096 | 6.275 ± 0.042 | 3.963 ± 0.062 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 0.971 ± 0.028 | - | 7.924 ± 0.135 | 0.890 ± 0.016 | 1.653 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.004 ± 0.001 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.090 ± 0.008 | - | 0.178 ± 0.011 | 0.078 ± 0.002 | 0.112 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.014 ± 0.002 | - | 0.017 ± 0.001 | 0.012 ± 0.001 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.013 ± 0.000 | - | 0.017 ± 0.000 | 0.012 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.031 ± 0.000 | - | 0.034 ± 0.000 | 0.022 ± 0.001 | 0.035 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.016 ± 0.000 | - | 0.020 ± 0.000 | 0.014 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.040 ± 0.003 | - | 0.042 ± 0.001 | 0.027 ± 0.000 | 0.035 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 7.849 ± 0.129 | - | 17.974 ± 6.335 | 6.963 ± 0.181 | 13.712 ± 2.065 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.010 ± 0.000 | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.677 ± 0.668 | - | 0.705 ± 0.009 | 0.570 ± 0.017 | 0.448 ± 0.084 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.029 ± 0.000 | - | 0.033 ± 0.001 | 0.020 ± 0.000 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.072 ± 0.002 | - | 0.071 ± 0.005 | 0.046 ± 0.001 | 0.074 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.231 ± 0.011 | - | 0.240 ± 0.008 | 0.220 ± 0.006 | 0.406 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 1.046 ± 0.003 | - | 1.056 ± 0.004 | 0.945 ± 0.029 | 1.815 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 3.071 ± 0.004 | - | 3.110 ± 0.066 | 2.878 ± 0.153 | 6.049 ± 0.113 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.437 ± 0.020 | - | 0.446 ± 0.016 | 0.462 ± 0.026 | 1.205 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.002 ± 0.003 | - | 0.005 ± 0.000 | 0.007 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.032 ± 0.012 | - | 0.034 ± 0.001 | 0.037 ± 0.002 | 0.128 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.011 ± 0.000 | - | 0.014 ± 0.000 | 0.018 ± 0.001 | 0.031 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.005 ± 0.000 | - | 0.009 ± 0.001 | 0.011 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.013 ± 0.000 | - | 0.017 ± 0.000 | 0.021 ± 0.001 | 0.033 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 3.408 ± 0.050 | - | 3.456 ± 0.032 | 3.623 ± 0.105 | 9.357 ± 0.264 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.224 ± 0.003 | - | 0.234 ± 0.009 | 0.237 ± 0.024 | 0.685 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.014 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.025 ± 0.000 | - | 0.028 ± 0.002 | 0.032 ± 0.000 | 0.087 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.417 ± 0.013 | - | 0.360 ± 0.013 | 0.294 ± 0.010 | 0.396 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 1.515 ± 0.061 | - | 1.503 ± 0.054 | 0.938 ± 0.006 | 1.262 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 4.339 ± 0.149 | - | 4.089 ± 0.143 | 3.253 ± 0.112 | 3.926 ± 0.493 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.606 ± 0.010 | - | 3.004 ± 0.159 | 0.607 ± 0.012 | 1.756 ± 0.349 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.004 ± 0.003 | - | 0.009 ± 0.001 | 0.011 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.031 ± 0.001 | - | 0.044 ± 0.001 | 0.042 ± 0.002 | 0.107 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.007 ± 0.000 | - | 0.012 ± 0.001 | 0.010 ± 0.003 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.009 ± 0.000 | - | 0.013 ± 0.000 | 0.014 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.021 ± 0.000 | - | 0.023 ± 0.001 | 0.016 ± 0.001 | 0.032 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.011 ± 0.000 | - | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.021 ± 0.000 | - | 0.029 ± 0.002 | 0.019 ± 0.000 | 0.034 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 5.019 ± 0.086 | - | 19.884 ± 6.672 | 4.754 ± 0.148 | 13.944 ± 5.736 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.013 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.251 ± 0.014 | - | 0.285 ± 0.014 | 0.258 ± 0.043 | 0.462 ± 0.115 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.017 ± 0.000 | - | 0.020 ± 0.000 | 0.019 ± 0.003 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.040 ± 0.000 | - | 0.041 ± 0.002 | 0.030 ± 0.000 | 0.088 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.136 ± 0.002 | - | 0.128 ± 0.002 | 0.176 ± 0.012 | 0.124 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.173 ± 0.009 | - | 0.172 ± 0.009 | 0.221 ± 0.004 | 0.197 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.597 ± 0.010 | - | 0.625 ± 0.010 | 0.730 ± 0.038 | 0.573 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.091 ± 0.003 | - | 0.099 ± 0.003 | 0.125 ± 0.000 | 0.183 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.017 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.012 ± 0.000 | - | 0.015 ± 0.000 | 0.020 ± 0.001 | 0.048 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.013 ± 0.000 | - | 0.016 ± 0.000 | 0.020 ± 0.000 | 0.055 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.004 ± 0.001 | - | 0.008 ± 0.001 | 0.017 ± 0.015 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.015 ± 0.000 | 0.036 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.003 | 0.011 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.037 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.807 ± 0.027 | - | 0.890 ± 0.024 | 1.127 ± 0.119 | 1.256 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.069 ± 0.000 | - | 0.074 ± 0.002 | 0.068 ± 0.010 | 0.105 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.035 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.018 ± 0.011 | 0.045 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.181 ± 0.008 | - | 0.182 ± 0.004 | 0.141 ± 0.006 | 0.113 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.254 ± 0.002 | - | 0.296 ± 0.025 | 0.195 ± 0.018 | 0.193 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.806 ± 0.016 | - | 0.774 ± 0.014 | 0.414 ± 0.021 | 0.747 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.115 ± 0.009 | - | 0.126 ± 0.003 | 0.134 ± 0.003 | 0.173 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.016 ± 0.001 | 0.015 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.019 ± 0.001 | - | 0.019 ± 0.000 | 0.025 ± 0.001 | 0.038 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.019 ± 0.001 | - | 0.022 ± 0.000 | 0.027 ± 0.001 | 0.041 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.000 | 0.013 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.022 ± 0.001 | 0.029 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.018 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.021 ± 0.000 | 0.045 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.944 ± 0.013 | - | 1.028 ± 0.013 | 0.533 ± 0.025 | 0.948 ± 0.197 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.020 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.095 ± 0.003 | - | 0.097 ± 0.002 | 0.064 ± 0.001 | 0.127 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.020 ± 0.001 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.010 ± 0.000 | - | 0.014 ± 0.000 | 0.026 ± 0.000 | 0.045 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.199 ± 0.009 | - | 0.207 ± 0.009 | 0.198 ± 0.005 | 0.278 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.477 ± 0.008 | - | 0.484 ± 0.006 | 0.650 ± 0.009 | 0.729 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.202 ± 0.030 | - | 1.237 ± 0.004 | 1.542 ± 0.013 | 1.909 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.253 ± 0.011 | - | 0.270 ± 0.010 | 0.352 ± 0.008 | 0.596 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.013 ± 0.001 | - | 0.016 ± 0.000 | 0.017 ± 0.000 | 0.069 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.011 ± 0.000 | - | 0.014 ± 0.000 | 0.017 ± 0.004 | 0.054 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.006 ± 0.000 | - | 0.013 ± 0.002 | 0.011 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 2.137 ± 0.025 | - | 2.615 ± 0.017 | 2.795 ± 0.018 | 4.524 ± 0.130 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.076 ± 0.002 | - | 0.084 ± 0.004 | 0.088 ± 0.000 | 0.193 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.005 ± 0.000 | - | 0.010 ± 0.000 | 0.009 ± 0.002 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.011 ± 0.000 | - | 0.018 ± 0.000 | 0.017 ± 0.000 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.200 ± 0.007 | - | 0.215 ± 0.004 | 0.265 ± 0.010 | 0.263 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.481 ± 0.012 | - | 0.721 ± 0.030 | 0.839 ± 0.016 | 0.723 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.191 ± 0.029 | - | 1.748 ± 0.010 | 1.824 ± 0.004 | 1.288 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.322 ± 0.013 | - | 3.007 ± 0.032 | 0.327 ± 0.014 | 1.281 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.013 ± 0.000 | - | 0.023 ± 0.000 | 0.024 ± 0.000 | 0.044 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.019 ± 0.000 | - | 0.023 ± 0.000 | 0.029 ± 0.003 | 0.049 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.007 ± 0.000 | - | 0.012 ± 0.000 | 0.015 ± 0.001 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.007 ± 0.000 | - | 0.014 ± 0.003 | 0.019 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 2.314 ± 0.025 | - | 10.477 ± 0.611 | 2.563 ± 0.236 | 10.495 ± 4.480 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.077 ± 0.004 | - | 0.083 ± 0.001 | 0.124 ± 0.004 | 0.193 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.006 ± 0.000 | - | 0.012 ± 0.000 | 0.015 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.011 ± 0.000 | - | 0.021 ± 0.000 | 0.027 ± 0.002 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.140 ± 0.004 | - | 0.123 ± 0.002 | 0.076 ± 0.001 | 0.263 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.158 ± 0.003 | - | 0.156 ± 0.004 | 0.099 ± 0.001 | 0.346 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.575 ± 0.006 | - | 0.570 ± 0.006 | 0.678 ± 0.010 | 0.731 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.066 ± 0.009 | - | 0.077 ± 0.005 | 0.096 ± 0.001 | 0.149 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.001 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.010 ± 0.000 | - | 0.015 ± 0.000 | 0.021 ± 0.001 | 0.043 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.009 ± 0.000 | 0.014 ± 0.001 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.001 | 0.015 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.015 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.532 ± 0.029 | - | 0.645 ± 0.015 | 0.726 ± 0.019 | 0.980 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.061 ± 0.000 | - | 0.069 ± 0.002 | 0.063 ± 0.000 | 0.102 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.011 ± 0.000 | 0.018 ± 0.001 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.146 ± 0.007 | - | 0.156 ± 0.003 | 0.113 ± 0.007 | 0.242 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.173 ± 0.006 | - | 0.194 ± 0.003 | 0.177 ± 0.012 | 0.335 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.575 ± 0.007 | - | 0.721 ± 0.017 | 0.495 ± 0.008 | 0.929 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.066 ± 0.005 | - | 0.105 ± 0.003 | 0.121 ± 0.015 | 0.140 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.010 ± 0.000 | 0.019 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.010 ± 0.000 | - | 0.024 ± 0.001 | 0.029 ± 0.001 | 0.035 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.010 ± 0.000 | 0.019 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.011 ± 0.000 | 0.021 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.010 ± 0.000 | 0.019 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.011 ± 0.000 | 0.019 ± 0.000 | 0.016 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.004 ± 0.002 | - | 0.016 ± 0.000 | 0.025 ± 0.002 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.010 ± 0.000 | 0.019 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.011 ± 0.000 | 0.020 ± 0.002 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.013 ± 0.000 | 0.025 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.523 ± 0.018 | - | 0.781 ± 0.055 | 0.396 ± 0.034 | 0.755 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.011 ± 0.000 | 0.021 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.061 ± 0.002 | - | 0.093 ± 0.006 | 0.056 ± 0.001 | 0.126 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.012 ± 0.000 | 0.021 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.016 ± 0.000 | 0.029 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.592 ± 0.011 | - | 1.554 ± 0.006 | 1.331 ± 0.002 | 1.915 ± 0.288 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 5.670 ± 0.064 | - | 5.822 ± 0.166 | 5.035 ± 0.117 | 4.868 ± 0.151 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 16.577 ± 0.207 | - | 16.655 ± 0.392 | 15.690 ± 0.184 | 14.574 ± 0.339 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 2.600 ± 0.039 | - | 2.714 ± 0.073 | 2.748 ± 0.032 | 3.702 ± 0.053 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.157 ± 0.014 | - | 0.156 ± 0.004 | 0.172 ± 0.012 | 0.205 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.009 ± 0.001 | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.118 ± 0.010 | - | 0.121 ± 0.004 | 0.140 ± 0.004 | 0.198 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.023 ± 0.003 | - | 0.023 ± 0.006 | 0.025 ± 0.001 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.007 ± 0.001 | - | 0.012 ± 0.000 | 0.011 ± 0.001 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.020 ± 0.000 | - | 0.023 ± 0.000 | 0.022 ± 0.001 | 0.030 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.051 ± 0.001 | - | 0.055 ± 0.006 | 0.057 ± 0.003 | 0.060 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.013 ± 0.001 | 0.034 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.025 ± 0.000 | - | 0.029 ± 0.002 | 0.027 ± 0.000 | 0.042 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.069 ± 0.002 | - | 0.069 ± 0.001 | 0.069 ± 0.002 | 0.077 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 21.295 ± 0.399 | - | 21.478 ± 0.837 | 21.369 ± 0.417 | 28.559 ± 0.440 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.015 ± 0.000 | - | 0.018 ± 0.000 | 0.018 ± 0.000 | 0.032 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.127 ± 0.043 | - | 1.161 ± 0.034 | 1.289 ± 0.026 | 1.348 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.048 ± 0.000 | - | 0.051 ± 0.003 | 0.050 ± 0.004 | 0.053 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.133 ± 0.002 | - | 0.140 ± 0.002 | 0.128 ± 0.003 | 0.176 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 2.614 ± 0.049 | - | 2.665 ± 0.074 | 1.308 ± 0.039 | 1.794 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 9.074 ± 0.165 | - | 5.877 ± 0.277 | 4.862 ± 0.022 | 2.700 ± 0.163 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 16.650 ± 0.127 | - | 19.027 ± 0.358 | 17.412 ± 0.240 | 9.125 ± 2.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 3.291 ± 0.053 | - | 9.987 ± 0.613 | 3.134 ± 0.263 | 5.779 ± 0.197 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.013 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.165 ± 0.014 | - | 0.192 ± 0.004 | 0.178 ± 0.014 | 0.245 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.014 ± 0.000 | - | 0.016 ± 0.001 | 0.013 ± 0.000 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.170 ± 0.008 | - | 0.198 ± 0.026 | 0.176 ± 0.002 | 0.200 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.023 ± 0.000 | - | 0.033 ± 0.000 | 0.025 ± 0.000 | 0.033 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.012 ± 0.000 | - | 0.017 ± 0.000 | 0.016 ± 0.001 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.031 ± 0.002 | - | 0.032 ± 0.002 | 0.023 ± 0.000 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.052 ± 0.000 | - | 0.072 ± 0.001 | 0.056 ± 0.001 | 0.057 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.017 ± 0.002 | - | 0.020 ± 0.000 | 0.018 ± 0.001 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.040 ± 0.001 | - | 0.039 ± 0.001 | 0.028 ± 0.000 | 0.032 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.069 ± 0.001 | - | 0.095 ± 0.001 | 0.072 ± 0.001 | 0.072 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 26.084 ± 0.286 | - | 38.973 ± 0.640 | 24.215 ± 0.816 | 70.234 ± 58.666 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.025 ± 0.000 | - | 0.027 ± 0.002 | 0.026 ± 0.000 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 1.168 ± 0.024 | - | 1.233 ± 0.189 | 1.335 ± 0.068 | 0.885 ± 0.285 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.074 ± 0.004 | - | 0.074 ± 0.006 | 0.048 ± 0.000 | 0.051 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.129 ± 0.002 | - | 0.183 ± 0.053 | 0.134 ± 0.002 | 0.136 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 1.037 ± 0.015 | - | 1.065 ± 0.037 | 1.041 ± 0.001 | 1.126 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 4.762 ± 0.146 | - | 4.737 ± 0.028 | 4.153 ± 0.096 | 3.631 ± 0.074 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 13.995 ± 0.218 | - | 14.216 ± 0.178 | 11.890 ± 0.204 | 11.290 ± 0.217 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.608 ± 0.066 | - | 1.712 ± 0.025 | 1.681 ± 0.021 | 2.260 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.004 ± 0.001 | - | 0.008 ± 0.000 | 0.009 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.126 ± 0.012 | - | 0.119 ± 0.005 | 0.122 ± 0.011 | 0.181 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.008 ± 0.000 | - | 0.012 ± 0.000 | 0.012 ± 0.001 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.099 ± 0.014 | - | 0.091 ± 0.006 | 0.079 ± 0.004 | 0.143 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.017 ± 0.001 | - | 0.020 ± 0.000 | 0.020 ± 0.001 | 0.030 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.011 ± 0.002 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.019 ± 0.001 | - | 0.022 ± 0.000 | 0.020 ± 0.001 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.040 ± 0.001 | - | 0.043 ± 0.001 | 0.040 ± 0.002 | 0.053 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.007 ± 0.000 | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.022 ± 0.000 | - | 0.027 ± 0.001 | 0.027 ± 0.004 | 0.034 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.053 ± 0.001 | - | 0.057 ± 0.001 | 0.050 ± 0.000 | 0.060 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 12.801 ± 0.237 | - | 13.262 ± 0.303 | 13.337 ± 0.305 | 15.899 ± 0.545 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.011 ± 0.000 | - | 0.016 ± 0.000 | 0.016 ± 0.000 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 0.878 ± 0.025 | - | 0.884 ± 0.012 | 0.864 ± 0.014 | 0.991 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.041 ± 0.001 | - | 0.044 ± 0.006 | 0.042 ± 0.000 | 0.045 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.108 ± 0.005 | - | 0.110 ± 0.004 | 0.093 ± 0.002 | 0.137 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 1.789 ± 0.019 | - | 1.769 ± 0.121 | 1.046 ± 0.012 | 1.157 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 7.540 ± 0.117 | - | 6.737 ± 1.782 | 4.135 ± 0.113 | 2.066 ± 0.053 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 14.534 ± 0.147 | - | 16.673 ± 0.212 | 13.303 ± 0.202 | 5.825 ± 0.114 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 1.954 ± 0.037 | - | 8.916 ± 0.096 | 1.767 ± 0.014 | 3.224 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.006 ± 0.001 | - | 0.009 ± 0.000 | 0.012 ± 0.002 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.133 ± 0.036 | - | 0.145 ± 0.004 | 0.114 ± 0.016 | 0.154 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.012 ± 0.001 | - | 0.014 ± 0.000 | 0.014 ± 0.001 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.115 ± 0.007 | - | 0.108 ± 0.008 | 0.105 ± 0.004 | 0.132 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.018 ± 0.001 | - | 0.027 ± 0.000 | 0.020 ± 0.001 | 0.025 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.010 ± 0.001 | - | 0.013 ± 0.000 | 0.016 ± 0.002 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.027 ± 0.001 | - | 0.027 ± 0.000 | 0.022 ± 0.001 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.041 ± 0.003 | - | 0.060 ± 0.001 | 0.040 ± 0.002 | 0.045 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.017 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.035 ± 0.001 | - | 0.039 ± 0.001 | 0.026 ± 0.004 | 0.033 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.054 ± 0.001 | - | 0.079 ± 0.002 | 0.051 ± 0.001 | 0.057 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 15.436 ± 0.350 | - | 22.204 ± 0.238 | 14.453 ± 0.317 | 26.122 ± 2.877 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.023 ± 0.001 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 0.883 ± 0.022 | - | 1.003 ± 0.163 | 0.835 ± 0.002 | 0.572 ± 0.093 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.063 ± 0.003 | - | 0.060 ± 0.001 | 0.040 ± 0.001 | 0.044 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.110 ± 0.001 | - | 0.155 ± 0.004 | 0.099 ± 0.001 | 0.116 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.360 ± 0.006 | - | unsupported | 0.303 ± 0.009 | 0.660 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.866 ± 0.008 | - | unsupported | 1.723 ± 0.055 | 2.260 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 4.781 ± 0.283 | - | unsupported | 4.349 ± 0.133 | 5.759 ± 0.237 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.850 ± 0.015 | - | unsupported | 0.897 ± 0.008 | 1.114 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.002 ± 0.001 | - | unsupported | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.043 ± 0.003 | - | unsupported | 0.046 ± 0.004 | 0.066 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.030 ± 0.000 | - | unsupported | 0.036 ± 0.001 | 0.052 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.007 ± 0.000 | - | unsupported | 0.012 ± 0.000 | 0.015 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.015 ± 0.003 | - | unsupported | 0.020 ± 0.003 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.008 ± 0.000 | - | unsupported | 0.013 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.020 ± 0.002 | - | unsupported | 0.024 ± 0.000 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 6.781 ± 0.147 | - | unsupported | 7.104 ± 0.040 | 8.691 ± 0.106 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.013 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.318 ± 0.018 | - | unsupported | 0.314 ± 0.009 | 0.457 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.015 ± 0.000 | - | unsupported | 0.020 ± 0.005 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.038 ± 0.001 | - | unsupported | 0.046 ± 0.018 | 0.054 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.609 ± 0.010 | - | unsupported | 0.449 ± 0.010 | 0.665 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 3.081 ± 0.006 | - | unsupported | 1.671 ± 0.007 | 1.330 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 7.821 ± 0.073 | - | unsupported | 6.049 ± 0.186 | 3.016 ± 0.088 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.035 ± 0.033 | - | unsupported | 1.032 ± 0.021 | 1.764 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.004 ± 0.003 | - | unsupported | 0.010 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.057 ± 0.002 | - | unsupported | 0.047 ± 0.001 | 0.063 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | unsupported | 0.012 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.046 ± 0.001 | - | unsupported | 0.046 ± 0.002 | 0.075 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.011 ± 0.001 | - | unsupported | 0.012 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.005 ± 0.000 | - | unsupported | 0.012 ± 0.002 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.012 ± 0.000 | - | unsupported | 0.012 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.026 ± 0.000 | - | unsupported | 0.022 ± 0.000 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.006 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.015 ± 0.000 | - | unsupported | 0.013 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.033 ± 0.000 | - | unsupported | 0.024 ± 0.000 | 0.033 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 8.153 ± 0.064 | - | unsupported | 8.037 ± 0.163 | 13.502 ± 2.623 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.008 ± 0.001 | - | unsupported | 0.013 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.434 ± 0.011 | - | unsupported | 0.323 ± 0.011 | 0.264 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.026 ± 0.001 | - | unsupported | 0.020 ± 0.000 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.062 ± 0.001 | - | unsupported | 0.040 ± 0.000 | 0.058 ± 0.013 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 1.310 ± 0.045 | - | unsupported | 1.453 ± 0.016 | 3.221 ± 0.080 | 2.507 ± 0.038 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.104 ± 0.001 | - | unsupported | 0.048 ± 0.005 | 0.115 ± 0.011 | 0.137 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.119 ± 0.001 | - | unsupported | 0.061 ± 0.002 | 0.179 ± 0.010 | 0.205 ± 0.005 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.445 ± 0.009 | - | unsupported | 0.412 ± 0.015 | 0.450 ± 0.022 | 0.728 ± 0.036 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 1.317 ± 0.042 | - | unsupported | 0.549 ± 0.018 | 1.208 ± 0.106 | 2.452 ± 0.100 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.106 ± 0.001 | - | unsupported | 0.050 ± 0.002 | 0.096 ± 0.011 | 0.113 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.119 ± 0.001 | - | unsupported | 0.070 ± 0.001 | 0.158 ± 0.015 | 0.173 ± 0.015 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.418 ± 0.009 | - | unsupported | 0.170 ± 0.007 | 0.605 ± 0.073 | 0.649 ± 0.011 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.269 ± 0.008 | - | unsupported | 0.541 ± 0.011 | 0.897 ± 0.008 | 0.364 ± 0.012 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.027 ± 0.000 | - | unsupported | 0.032 ± 0.001 | 0.171 ± 0.021 | 0.078 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.051 ± 0.001 | - | unsupported | 0.043 ± 0.001 | 0.214 ± 0.007 | 0.107 ± 0.006 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.152 ± 0.001 | - | unsupported | 0.263 ± 0.007 | 0.341 ± 0.006 | 0.183 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.272 ± 0.009 | - | unsupported | 0.617 ± 0.029 | 1.205 ± 0.075 | 0.327 ± 0.011 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.029 ± 0.000 | - | unsupported | 0.051 ± 0.004 | 0.178 ± 0.012 | 0.074 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.051 ± 0.000 | - | unsupported | 0.071 ± 0.004 | 0.209 ± 0.009 | 0.096 ± 0.007 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.152 ± 0.001 | - | unsupported | 0.311 ± 0.011 | 0.338 ± 0.016 | 0.163 ± 0.005 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.164 ± 0.002 | - | 0.326 ± 0.015 | 0.270 ± 0.013 | 0.436 ± 0.018 | 0.251 ± 0.033 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.078 ± 0.002 | - | 0.089 ± 0.006 | 0.068 ± 0.001 | 0.075 ± 0.003 | 0.069 ± 0.003 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.080 ± 0.004 | - | 0.089 ± 0.005 | 0.083 ± 0.000 | 0.091 ± 0.009 | 0.083 ± 0.003 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.095 ± 0.000 | - | 0.120 ± 0.005 | 0.143 ± 0.005 | 0.156 ± 0.008 | 0.128 ± 0.014 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.165 ± 0.006 | - | 0.330 ± 0.011 | 0.314 ± 0.014 | 0.560 ± 0.035 | 0.191 ± 0.010 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.077 ± 0.001 | - | 0.107 ± 0.002 | 0.088 ± 0.002 | 0.075 ± 0.001 | 0.065 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.078 ± 0.001 | - | 0.123 ± 0.003 | 0.116 ± 0.005 | 0.092 ± 0.007 | 0.073 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.095 ± 0.001 | - | 0.165 ± 0.011 | 0.178 ± 0.007 | 0.171 ± 0.011 | 0.107 ± 0.002 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 18.647 ± 0.271 | - | 20.941 ± 0.244 | 37.800 ± 0.770 | 57.005 ± 0.346 | 47.367 ± 6.130 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 11.175 ± 0.316 | - | 21.240 ± 2.095 | 17.498 ± 0.403 | 54.840 ± 9.011 | 36.107 ± 11.421 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 13.089 ± 0.613 | - | 12.741 ± 1.026 | 13.283 ± 0.351 | 17.623 ± 0.376 | 16.757 ± 0.197 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 9.503 ± 0.169 | - | 28.997 ± 0.340 | 6.593 ± 0.274 | 12.775 ± 0.852 | 10.184 ± 3.146 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 6.040 ± 0.094 | - | 6.025 ± 0.038 | 5.623 ± 0.055 | 6.839 ± 0.219 | 6.851 ± 0.288 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 10.588 ± 0.113 | - | 17.622 ± 0.198 | 8.162 ± 0.579 | 11.017 ± 0.359 | 11.073 ± 0.307 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 7.623 ± 0.118 | - | 7.710 ± 0.154 | 7.406 ± 0.399 | 8.481 ± 0.234 | 8.120 ± 0.242 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 13.666 ± 0.179 | - | 20.666 ± 0.168 | 9.373 ± 0.610 | 13.650 ± 0.206 | 14.484 ± 0.893 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 12.164 ± 0.392 | - | 12.085 ± 0.275 | 11.646 ± 0.270 | 26.513 ± 0.418 | 11.412 ± 0.806 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 11.824 ± 0.228 | - | 18.839 ± 0.332 | 11.641 ± 0.241 | 24.940 ± 0.993 | 11.047 ± 0.183 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.264 ± 0.011 | 0.275 ± 0.009 | unsupported | 0.283 ± 0.010 | 0.714 ± 0.036 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 5.142 ± 0.077 | 5.334 ± 0.081 | unsupported | 5.222 ± 0.089 | 7.682 ± 0.077 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.199 ± 0.002 | 6.561 ± 1.813 | unsupported | 0.217 ± 0.004 | 1.707 ± 0.058 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 3.987 ± 0.061 | 6.762 ± 2.303 | unsupported | 3.087 ± 0.284 | 10.284 ± 0.201 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 17.787 ± 0.525 | - | 17.574 ± 0.365 | 18.565 ± 0.262 | 37.170 ± 0.786 | 23.804 ± 2.832 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 20.706 ± 4.413 | - | 25.715 ± 5.287 | 9.754 ± 0.113 | 22.138 ± 0.665 | 14.233 ± 0.433 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | failed | failed | failed | 32.627 ± 0.535 | 5.231 ± 0.657 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | failed | 37.854 ± 2.594 | failed | 38.590 ± 1.452 | 4.895 ± 0.428 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 13.588 ± 0.174 | - | 14.539 ± 0.243 | 22.686 ± 0.602 | 27.826 ± 0.147 | 30.734 ± 10.257 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 9.594 ± 0.709 | - | 19.536 ± 0.728 | 11.355 ± 0.608 | 20.972 ± 1.696 | 26.131 ± 6.213 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 1.094 ± 0.047 | - | 1.133 ± 0.021 | 1.132 ± 0.025 | 8.467 ± 0.543 | 1.723 ± 0.061 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 5.999 ± 0.005 | - | 6.000 ± 0.004 | 1.141 ± 0.033 | 4.844 ± 0.350 | 1.814 ± 0.034 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 19.375 ± 0.246 | - | 18.782 ± 0.927 | 18.953 ± 0.458 | 14.000 ± 0.413 | 15.678 ± 1.049 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 66.991 ± 0.821 | - | 86.525 ± 2.015 | 52.103 ± 0.676 | 19.142 ± 0.864 | 18.373 ± 4.653 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 19.548 ± 1.263 | - | 18.490 ± 0.669 | 19.052 ± 0.369 | 13.756 ± 0.205 | 15.546 ± 0.298 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 65.954 ± 7.303 | - | 77.218 ± 11.005 | 52.151 ± 0.497 | 20.311 ± 0.825 | 18.369 ± 1.626 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 0.756 ± 0.039 | 0.751 ± 0.056 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 15.449 ± 0.163 | 15.868 ± 0.129 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.089 ± 0.051 | 11.592 ± 2.329 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 11.558 ± 0.432 | 18.919 ± 1.745 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 0.697 ± 0.025 | 0.719 ± 0.013 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 14.901 ± 0.521 | 15.069 ± 0.271 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 0.978 ± 0.041 | 10.004 ± 0.277 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 10.755 ± 0.224 | 17.846 ± 0.844 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 12.837 ± 0.911 | - | 12.941 ± 0.830 | 13.468 ± 0.304 | 17.362 ± 0.613 | 16.920 ± 0.547 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 10.103 ± 0.361 | - | 19.463 ± 0.397 | 6.567 ± 0.188 | 13.210 ± 2.238 | 10.052 ± 1.888 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 88.447 ± 2.986 | 87.291 ± 1.274 | 82.958 ± 1.710 | 44.955 ± 0.192 | 48.039 ± 0.278 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 197.784 ± 7.370 | 212.897 ± 17.702 | 191.272 ± 1.746 | 51.607 ± 1.016 | 53.730 ± 2.563 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 17.609 ± 0.771 | - | 79.464 ± 1.341 | 16.478 ± 0.970 | 137.628 ± 9.186 | 21.970 ± 0.939 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 9.697 ± 0.104 | - | 48.855 ± 18.338 | 8.772 ± 0.436 | 135.324 ± 14.452 | 34.521 ± 11.279 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.080 ± 0.002 | 0.079 ± 0.003 | unsupported | 0.080 ± 0.022 | 0.100 ± 0.020 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 1.273 ± 0.035 | 1.297 ± 0.025 | unsupported | 0.608 ± 0.016 | 0.493 ± 0.483 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.043 ± 0.005 | 0.045 ± 0.002 | unsupported | 0.046 ± 0.003 | 0.039 ± 0.009 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.607 ± 0.060 | 0.606 ± 0.008 | unsupported | 0.346 ± 0.142 | 0.764 ± 0.292 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.038 ± 0.001 | 0.039 ± 0.001 | unsupported | 0.098 ± 0.033 | 0.218 ± 0.001 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 0.613 ± 0.020 | 0.660 ± 0.029 | unsupported | 0.540 ± 0.052 | 2.349 ± 0.076 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.023 ± 0.002 | 0.023 ± 0.002 | unsupported | 0.062 ± 0.002 | 0.139 ± 0.003 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.315 ± 0.007 | 0.319 ± 0.010 | unsupported | 0.536 ± 0.021 | 1.032 ± 0.058 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.125 ± 0.004 | 0.133 ± 0.004 | unsupported | 0.171 ± 0.005 | 0.171 ± 0.002 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 2.179 ± 0.029 | 2.222 ± 0.059 | unsupported | 3.003 ± 0.096 | 1.621 ± 0.025 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.059 ± 0.004 | 0.058 ± 0.003 | unsupported | 0.082 ± 0.003 | 0.118 ± 0.004 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 0.907 ± 0.026 | 0.900 ± 0.025 | unsupported | 1.219 ± 0.028 | 0.966 ± 0.248 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.053 ± 0.017 | 0.080 ± 0.014 | unsupported | 0.129 ± 0.004 | 0.167 ± 0.009 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 1.029 ± 0.174 | 1.104 ± 0.170 | unsupported | 0.846 ± 0.005 | 2.194 ± 0.077 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.087 ± 0.004 | 0.116 ± 0.010 | unsupported | 0.061 ± 0.001 | 0.158 ± 0.002 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.602 ± 0.049 | 0.703 ± 0.072 | unsupported | 0.817 ± 0.030 | 1.390 ± 0.052 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.073 ± 0.016 | 0.089 ± 0.005 | unsupported | 0.081 ± 0.004 | 0.263 ± 0.007 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.445 ± 0.150 | 1.490 ± 0.173 | unsupported | 0.774 ± 0.038 | 3.567 ± 0.022 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.103 ± 0.006 | 0.123 ± 0.007 | unsupported | 0.049 ± 0.002 | 0.248 ± 0.011 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.792 ± 0.059 | 0.859 ± 0.042 | unsupported | 0.387 ± 0.030 | 2.193 ± 0.044 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.257 ± 0.005 | 0.286 ± 0.009 | unsupported | unsupported | 0.271 ± 0.011 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 4.238 ± 0.074 | 4.245 ± 0.212 | unsupported | unsupported | 4.609 ± 0.071 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.174 ± 0.004 | 0.214 ± 0.007 | unsupported | unsupported | 0.280 ± 0.010 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 1.748 ± 0.043 | 2.090 ± 0.258 | unsupported | unsupported | 2.683 ± 0.463 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.009 ± 0.001 | 0.017 ± 0.001 | unsupported | 0.011 ± 0.001 | 0.042 ± 0.008 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.152 ± 0.002 | 0.163 ± 0.005 | unsupported | 0.158 ± 0.012 | 0.246 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.012 ± 0.001 | 0.020 ± 0.001 | unsupported | 0.013 ± 0.001 | 0.039 ± 0.000 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.078 ± 0.003 | 0.088 ± 0.005 | unsupported | 0.052 ± 0.004 | 0.169 ± 0.010 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.033 ± 0.018 | 0.049 ± 0.010 | unsupported | 0.071 ± 0.002 | 0.161 ± 0.003 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.547 ± 0.185 | 0.566 ± 0.182 | unsupported | 0.805 ± 0.022 | 2.130 ± 0.036 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.046 ± 0.003 | 0.066 ± 0.006 | unsupported | 0.084 ± 0.002 | 0.137 ± 0.008 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.307 ± 0.060 | 0.357 ± 0.053 | unsupported | 0.430 ± 0.035 | 1.443 ± 0.084 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.029 ± 0.003 | 0.029 ± 0.002 | unsupported | 0.064 ± 0.001 | 0.164 ± 0.007 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.468 ± 0.042 | 0.504 ± 0.028 | unsupported | 1.230 ± 0.101 | 1.572 ± 0.004 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.017 ± 0.000 | 0.018 ± 0.000 | unsupported | 0.065 ± 0.002 | 0.128 ± 0.014 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.232 ± 0.005 | 0.244 ± 0.010 | unsupported | 0.531 ± 0.044 | 0.875 ± 0.059 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.032 ± 0.002 | 0.032 ± 0.002 | unsupported | 0.066 ± 0.004 | 0.161 ± 0.013 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.505 ± 0.038 | 0.530 ± 0.037 | unsupported | 1.295 ± 0.074 | 1.615 ± 0.048 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.018 ± 0.002 | 0.019 ± 0.004 | unsupported | 0.067 ± 0.003 | 0.111 ± 0.003 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.245 ± 0.004 | 0.254 ± 0.010 | unsupported | 0.542 ± 0.013 | 0.914 ± 0.009 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.073 ± 0.007 | 0.094 ± 0.007 | unsupported | 0.091 ± 0.004 | 0.261 ± 0.004 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.358 ± 0.028 | 1.420 ± 0.065 | unsupported | 0.840 ± 0.008 | 3.352 ± 0.034 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.100 ± 0.003 | 0.130 ± 0.003 | unsupported | 0.049 ± 0.001 | 0.247 ± 0.002 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 0.816 ± 0.021 | 0.859 ± 0.027 | unsupported | 0.329 ± 0.003 | 2.152 ± 0.150 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.090 ± 0.001 | 0.093 ± 0.002 | unsupported | 0.148 ± 0.002 | 0.294 ± 0.005 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 1.495 ± 0.034 | 1.539 ± 0.072 | unsupported | 2.576 ± 0.028 | 3.531 ± 0.078 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.045 ± 0.001 | 0.045 ± 0.002 | unsupported | 0.128 ± 0.002 | 0.187 ± 0.006 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 0.682 ± 0.019 | 0.686 ± 0.023 | unsupported | 1.031 ± 0.011 | 1.871 ± 0.046 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.776 ± 0.020 | 1.347 ± 0.102 | unsupported | 0.376 ± 0.004 | 0.185 ± 0.024 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 14.207 ± 0.920 | 24.657 ± 0.502 | unsupported | 6.554 ± 0.149 | 1.648 ± 0.062 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.542 ± 0.011 | 1.229 ± 0.017 | unsupported | 0.320 ± 0.002 | 0.137 ± 0.013 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 7.576 ± 0.127 | 18.887 ± 0.498 | unsupported | 2.881 ± 0.032 | 0.936 ± 0.016 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 53.840 ± 0.375 | - | unsupported | 52.284 ± 0.662 | - | 52.676 ± 0.163 | 57.453 ± 0.343 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 24.155 ± 0.136 | - | unsupported | 24.277 ± 0.254 | - | 51.653 ± 0.127 | 25.345 ± 0.256 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 42.591 ± 0.093 | - | unsupported | 39.325 ± 0.239 | - | 37.992 ± 0.658 | 45.070 ± 1.084 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 17.938 ± 0.523 | - | unsupported | 17.608 ± 0.126 | - | 36.313 ± 0.305 | 18.508 ± 0.350 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 21.774 ± 0.234 | - | unsupported | 39.022 ± 0.392 | - | 21.865 ± 0.223 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 17.658 ± 0.124 | - | unsupported | 17.805 ± 0.199 | - | 22.007 ± 0.289 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 54.767 ± 0.269 | - | unsupported | 52.275 ± 0.285 | - | 52.819 ± 0.597 | 57.595 ± 0.119 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 24.156 ± 0.275 | - | unsupported | 24.564 ± 2.623 | - | 51.924 ± 0.416 | 25.551 ± 0.092 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 24.557 ± 0.526 | - | unsupported | 24.271 ± 1.655 | - | 25.315 ± 0.227 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 12.417 ± 0.373 | - | unsupported | 12.224 ± 0.351 | - | 13.184 ± 0.129 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 25.198 ± 0.912 | - | unsupported | 24.112 ± 0.825 | - | 25.902 ± 0.768 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 12.390 ± 0.138 | - | unsupported | 12.052 ± 0.291 | - | 12.621 ± 0.284 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 53.899 ± 0.426 | - | unsupported | 52.086 ± 0.225 | - | 52.441 ± 0.377 | 57.327 ± 0.362 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 24.503 ± 0.043 | - | unsupported | 24.184 ± 0.154 | - | 51.498 ± 0.520 | 25.393 ± 0.918 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 38.868 ± 0.380 | - | unsupported | 38.901 ± 0.181 | - | 38.864 ± 0.195 | 42.030 ± 0.106 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 17.660 ± 0.194 | - | unsupported | 17.811 ± 0.221 | - | 38.382 ± 0.294 | 17.872 ± 0.083 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.322 ± 0.007 | unsupported | unsupported | 0.420 ± 0.005 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 5.457 ± 0.041 | unsupported | unsupported | 7.008 ± 0.060 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.374 ± 0.006 | unsupported | unsupported | 0.416 ± 0.012 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 3.977 ± 0.059 | unsupported | unsupported | 3.814 ± 0.047 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 53.907 ± 0.957 | - | unsupported | 51.835 ± 0.253 | - | 53.098 ± 0.426 | 57.370 ± 0.656 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 24.424 ± 0.243 | - | unsupported | 24.537 ± 0.235 | - | 51.772 ± 0.672 | 25.399 ± 0.342 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 95.840 ± 1.039 | - | 0.002 ± 0.000 | 99.240 ± 0.824 | 98.173 ± 1.191 | 109.027 ± 0.624 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 41.199 ± 0.537 | - | 0.002 ± 0.000 | 33.398 ± 0.768 | 52.102 ± 0.671 | 110.225 ± 1.350 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 61.033 ± 2.576 | - | 172.928 ± 1.150 | 64.807 ± 0.791 | 64.573 ± 1.773 | 68.561 ± 0.386 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 23.975 ± 0.160 | - | 75.893 ± 1.667 | 24.110 ± 0.832 | 34.161 ± 0.239 | 70.166 ± 0.491 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 191.981 ± 0.877 | - | 191.767 ± 0.825 | 193.791 ± 0.693 | 193.503 ± 5.891 | 207.361 ± 0.400 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 68.148 ± 0.434 | - | 68.096 ± 0.695 | 65.141 ± 0.425 | 105.528 ± 3.173 | 212.427 ± 1.943 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 65.190 ± 0.471 | - | 64.084 ± 0.203 | 70.224 ± 0.414 | 57.428 ± 0.381 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 34.597 ± 0.290 | - | 34.347 ± 0.180 | 25.792 ± 0.485 | 31.484 ± 0.368 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 139.669 ± 0.528 | - | 0.002 ± 0.000 | 108.713 ± 0.618 | 119.556 ± 3.694 | 151.062 ± 1.186 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 38.811 ± 0.591 | - | 0.003 ± 0.000 | 38.989 ± 1.034 | 44.329 ± 0.668 | 153.659 ± 0.890 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 125.508 ± 1.286 | - | 0.002 ± 0.000 | 73.510 ± 0.323 | 82.654 ± 0.836 | 243.083 ± 7.185 | 86.365 ± 0.984 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 31.628 ± 0.602 | - | 0.003 ± 0.000 | 73.766 ± 0.276 | 43.160 ± 0.598 | 230.230 ± 4.088 | 37.107 ± 0.485 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 51.599 ± 0.271 | - | 52.203 ± 0.273 | 60.522 ± 0.400 | 55.278 ± 0.588 | 75.417 ± 0.579 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 20.783 ± 0.098 | - | 20.799 ± 0.089 | 21.995 ± 0.113 | 29.000 ± 0.165 | 76.857 ± 1.456 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 52.664 ± 0.745 | - | 52.260 ± 0.396 | 57.930 ± 0.690 | 55.380 ± 1.266 | 76.546 ± 0.853 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 20.912 ± 0.128 | - | 20.750 ± 0.186 | 20.410 ± 0.128 | 29.034 ± 0.461 | 77.439 ± 0.549 | - |
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
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 7.601692 | 475105.75 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 5.296751 | 331046.94 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 5.616281 | 351017.56 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 6.004361 | 375272.56 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 3.508132 | 219258.25 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 4.200197 | 262512.31 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 18.567042 | 1160440.12 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 18.938060 | 1183628.75 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 9.541001 | 596312.56 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 9.505924 | 594120.25 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 11.428471 | 714279.44 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 4.522026 | 282626.62 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 4.218065 | 263629.06 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 4.407264 | 275454.00 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 122.914449 | 7682153.06 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 83.558141 | 5222383.81 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 82.264092 | 5141505.75 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 85.342414 | 5333900.88 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 27.306676 | 1706667.25 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 3.475509 | 217219.31 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 3.181084 | 198817.75 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 104.971030 | 6560689.38 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 164.547799 | 10284237.44 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 49.399022 | 3087438.88 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 63.785110 | 3986569.38 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 108.196277 | 6762267.31 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 12.100810 | 756300.62 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 12.016877 | 751054.81 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 247.176989 | 15448561.81 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 253.893899 | 15868368.69 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 17.426451 | 1089153.19 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 185.472579 | 11592036.19 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 184.930085 | 11558130.31 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 302.700912 | 18918807.00 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 11.146175 | 696635.94 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 11.507021 | 719188.81 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 238.411138 | 14900696.12 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 241.099044 | 15068690.25 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 15.648766 | 978047.88 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 160.060498 | 10003781.12 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 172.077272 | 10754829.50 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 285.537746 | 17846109.12 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.379780 | 99684.06 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.097316 | 79645.56 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.141460 | 80335.31 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.071324 | 79239.44 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 3.946763 | 493345.38 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.865486 | 608185.75 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.183168 | 1272896.00 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.377146 | 1297143.25 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.526712 | 39479.88 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.966704 | 46354.75 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.732891 | 42701.42 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.848194 | 44503.03 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.108022 | 763502.75 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.766078 | 345759.75 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.853365 | 606670.62 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.850202 | 606275.25 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.965756 | 218214.94 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.293480 | 98335.62 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.436320 | 38067.50 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.480549 | 38758.58 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.795574 | 2349446.75 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.319431 | 539928.88 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.901879 | 612734.88 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.283927 | 660490.88 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.914397 | 139287.45 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.982823 | 62231.61 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.493682 | 23338.78 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.489020 | 23265.94 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.252531 | 1031566.37 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.288009 | 536001.12 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.517838 | 314729.75 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.549006 | 318625.75 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.948707 | 171073.55 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 10.971596 | 171431.19 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.023961 | 125374.39 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.507653 | 132932.08 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.970486 | 1621310.75 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 24.025208 | 3003151.00 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 17.432217 | 2179027.12 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 17.772272 | 2221534.00 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.575020 | 118359.69 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.244672 | 81948.00 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.748610 | 58572.03 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3.740612 | 58447.06 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.724487 | 965560.88 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 9.751685 | 1218960.62 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.259792 | 907474.00 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.201303 | 900162.88 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.705707 | 167276.67 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.250040 | 128906.88 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.365067 | 52579.17 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.097173 | 79643.33 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 17.551312 | 2193914.00 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 6.767273 | 845909.12 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 8.234173 | 1029271.62 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 8.832782 | 1104097.75 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.118187 | 158096.67 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.929242 | 61394.41 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.540989 | 86577.95 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.392596 | 115509.31 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.118440 | 1389805.00 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.539437 | 817429.62 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.818141 | 602267.62 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.622032 | 702754.00 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 16.821510 | 262836.09 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.214198 | 81471.84 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.703559 | 73493.11 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.683365 | 88802.58 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 28.536916 | 3567114.50 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 6.190141 | 773767.62 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 11.561762 | 1445220.25 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.920322 | 1490040.25 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 15.862414 | 247850.22 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.159470 | 49366.72 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.569232 | 102644.25 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.892345 | 123317.89 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 17.547119 | 2193389.88 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.092024 | 386503.00 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.337012 | 792126.50 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 6.869192 | 858649.00 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 17.366793 | 271356.14 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 16.475083 | 257423.17 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 18.310493 | 286101.45 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 36.875004 | 4609375.50 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 33.905732 | 4238216.50 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 33.958186 | 4244773.25 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 17.934453 | 280225.83 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 11.153294 | 174270.22 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 13.672639 | 213634.98 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 21.462291 | 2682786.38 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 13.985501 | 1748187.62 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 16.718227 | 2089778.37 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.670887 | 41732.61 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.731680 | 11432.50 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.606042 | 9469.41 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.088635 | 17009.92 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 1.965815 | 245726.88 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 1.266027 | 158253.38 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 1.212828 | 151603.50 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 1.306388 | 163298.50 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.494971 | 38983.92 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.823970 | 12874.53 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.780616 | 12197.12 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.259446 | 19678.84 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.352304 | 169038.00 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.414516 | 51814.50 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.622393 | 77799.12 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 0.701247 | 87655.88 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.325997 | 161343.70 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.574039 | 71469.36 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.082446 | 32538.22 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.142520 | 49101.88 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 17.038432 | 2129804.00 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 6.440091 | 805011.38 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.376985 | 547123.12 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.531600 | 566450.00 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.780822 | 137200.34 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.367062 | 83860.34 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.953137 | 46142.77 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.192233 | 65503.64 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.541486 | 1442685.75 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.441575 | 430196.88 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.452075 | 306509.38 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.858726 | 357340.75 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.527588 | 164493.56 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.125524 | 64461.31 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.842440 | 28788.12 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.839527 | 28742.61 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.574847 | 1571855.88 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.840909 | 1230113.62 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.747557 | 468444.62 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.034686 | 504335.75 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.186052 | 127907.06 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.191885 | 65498.20 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.117823 | 17465.98 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.134589 | 17727.95 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.997598 | 874699.75 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.250918 | 531364.75 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.859073 | 232384.12 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.953743 | 244217.88 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.296979 | 160890.30 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.237121 | 66205.02 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.024741 | 31636.58 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.030762 | 31730.66 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.920964 | 1615120.50 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.357130 | 1294641.25 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.038685 | 504835.62 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.242021 | 530252.62 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.095001 | 110859.39 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.316400 | 67443.75 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.152024 | 18000.38 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.217974 | 19030.84 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.310780 | 913847.50 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.333760 | 541720.00 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.963802 | 245475.25 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.030507 | 253813.38 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 16.729973 | 261405.83 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.829940 | 91092.81 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.675755 | 73058.67 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.002380 | 93787.19 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 26.817725 | 3352215.62 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 6.722444 | 840305.50 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.866910 | 1358363.75 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.360968 | 1420121.00 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 15.823226 | 247237.91 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.129439 | 48897.48 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.378941 | 99670.95 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 8.292232 | 129566.12 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 17.216525 | 2152065.62 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.630077 | 328759.62 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.530079 | 816259.88 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 6.871292 | 858911.50 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 18.830055 | 294219.61 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 9.450423 | 147662.86 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.769787 | 90152.92 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.943853 | 92872.70 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 28.251532 | 3531441.50 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 20.605568 | 2575696.00 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 11.957708 | 1494713.50 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 12.314563 | 1539320.38 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.950838 | 186731.84 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 8.173704 | 127714.13 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.902373 | 45349.58 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.859124 | 44673.81 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 14.964444 | 1870555.50 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.249388 | 1031173.50 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.456334 | 682041.75 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.487341 | 685917.62 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.813844 | 184591.31 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 24.044675 | 375698.05 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 49.637777 | 775590.27 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 86.220338 | 1347192.78 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.183711 | 1647963.88 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 52.428659 | 6553582.38 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 113.655127 | 14206890.88 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 197.254022 | 24656752.75 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.749180 | 136705.94 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 20.454078 | 319594.97 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 34.687095 | 541985.86 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 78.679211 | 1229362.67 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.490195 | 936274.38 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 23.047164 | 2880895.50 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 60.607442 | 7575930.25 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 151.094905 | 18886863.12 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 6.721170 | 420073.12 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 5.145831 | 321614.44 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 112.123908 | 7007744.25 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 87.312536 | 5457033.50 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 6.658463 | 416153.94 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 5.979870 | 373741.88 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 61.028438 | 3814277.38 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 63.632911 | 3977056.94 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.017539 | 1096.19 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.001993 | 124.56 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.018701 | 1168.81 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.001883 | 117.69 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000295 | 18.44 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.017540 | 1096.25 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.006996 | 437.25 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000403 | 25.19 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.018702 | 1168.88 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.006621 | 413.81 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000252 | 15.75 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.017497 | 1093.56 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.005179 | 323.69 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.001208 | 75.50 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.017872 | 1117.00 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.005010 | 313.12 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000330 | 20.62 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.016901 | 1056.31 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003889 | 243.06 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000239 | 14.94 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.016589 | 1036.81 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003687 | 230.44 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/elementwise_reduction/exp` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 12.8x faster than the slowest successful cell (`julia-base`, 57.325 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/exp` (f64, threads=4, shape=`8388608`): `tenferro-trace` is 12.8x faster than the slowest successful cell (`julia-base`, 57.325 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log1p` (f64, threads=4, shape=`4194304`): `tenferro-direct` is 10.8x faster than the slowest successful cell (`julia-base`, 36.945 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log1p` (f64, threads=4, shape=`4194304`): `tenferro-trace` is 10.8x faster than the slowest successful cell (`julia-base`, 36.945 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 17.420 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 13.8x faster than the slowest successful cell (`pytorch-cpu`, 12.568 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 13.9x faster than the slowest successful cell (`pytorch-cpu`, 12.568 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 17.183 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 11.853 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 13.3x faster than the slowest successful cell (`pytorch-cpu`, 11.853 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 13.2x faster than the slowest successful cell (`pytorch-cpu`, 11.853 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sin` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`julia-base`, 56.866 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sin` (f64, threads=4, shape=`8388608`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`julia-base`, 56.866 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `pytorch-cpu` is 12.4x faster than the slowest successful cell (`julia-base`, 7.870 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `tenferro-direct` is 12.8x faster than the slowest successful cell (`julia-base`, 7.870 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `tenferro-trace` is 12.7x faster than the slowest successful cell (`julia-base`, 7.870 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `jax-cpu` is 10.3x faster than the slowest successful cell (`julia-base`, 6.709 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 10.7x faster than the slowest successful cell (`julia-base`, 6.709 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 11.1x faster than the slowest successful cell (`julia-base`, 6.709 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-trace` is 10.8x faster than the slowest successful cell (`julia-base`, 6.709 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 3533.8x faster than the slowest successful cell (`julia-base`, 8.414 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 3066.0x faster than the slowest successful cell (`julia-base`, 8.465 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.008 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 16.5x faster than the slowest successful cell (`jax-cpu`, 0.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.008 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`3x4x4`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.009 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 22.2x faster than the slowest successful cell (`jax-cpu`, 0.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 16.0x faster than the slowest successful cell (`jax-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 10.0x faster than the slowest successful cell (`jax-cpu`, 0.035 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`256x128`): `pytorch-cpu` is 30.2x faster than the slowest successful cell (`tenferro-eager`, 6.561 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`256x128`): `tenferro-direct` is 33.0x faster than the slowest successful cell (`tenferro-eager`, 6.561 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/rank_revealing_qr` (f64, threads=4, shape=`256x128`): `tenferro-direct` is 10.6x faster than the slowest successful cell (`tenferro-eager`, 11.592 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/rank_revealing_qr_deficient` (f64, threads=4, shape=`256x128`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`tenferro-eager`, 10.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/triangular_solve` (f64, threads=4, shape=`4096x4096,rhs=64`): `pytorch-cpu` is 15.4x faster than the slowest successful cell (`jax-cpu`, 135.324 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/triangular_solve` (f64, threads=4, shape=`4096x4096,rhs=64`): `tenferro-direct` is 14.0x faster than the slowest successful cell (`jax-cpu`, 135.324 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 15.0x faster than the slowest successful cell (`tenferro-eager`, 24.657 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 20.2x faster than the slowest successful cell (`tenferro-eager`, 18.887 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 46374.9x faster than the slowest successful cell (`julia-base`, 109.027 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 48407.9x faster than the slowest successful cell (`julia-base`, 110.225 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 63418.2x faster than the slowest successful cell (`julia-base`, 151.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 54624.6x faster than the slowest successful cell (`julia-base`, 153.659 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 109793.4x faster than the slowest successful cell (`julia-base`, 243.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 90357.0x faster than the slowest successful cell (`julia-base`, 230.230 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 60.9x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 46.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 68.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 50.3x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 69.1x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 7042.8 GB/s at 0.002 ms. The conservative physical lower bound is 0.168 ms, so the cell is 70.4x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 6073.5 GB/s at 0.003 ms. The conservative physical lower bound is 0.042 ms, so the cell is 15.2x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 114179.3 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1141.8x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 117890.0 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 294.7x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 112693.3 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1126.9x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 95426.8 GB/s at 0.003 ms. The conservative physical lower bound is 0.671 ms, so the cell is 238.6x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 60622.3 GB/s at 0.002 ms. The conservative physical lower bound is 1.342 ms, so the cell is 606.2x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 52675.7 GB/s at 0.003 ms. The conservative physical lower bound is 0.336 ms, so the cell is 131.7x faster than the bound.

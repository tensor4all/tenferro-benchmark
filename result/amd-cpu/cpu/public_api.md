# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/amd-cpu/cpu/public_api/20261104_220530/run.yaml`
- Timestamp: `2026-11-04T22:05:30.608151+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/amd-cpu/cpu/public_api/20261104_220530`.

- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

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
- Python platform: `Linux-7.0.0-1017-azure-x86_64-with-glibc2.39`
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

- CSV: `data/results/amd-cpu/cpu/public_api/20261104_220530/cpu_public_api_t1_20261104_220530.csv`
- CSV: `data/results/amd-cpu/cpu/public_api/20261104_220530/cpu_public_api_t4_20261104_220530.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 4.242 ± 0.087 | - | 4.327 ± 0.059 | 4.625 ± 0.086 | 3.910 ± 0.046 | 3.822 ± 0.070 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.406 ± 0.046 | - | 7.004 ± 0.022 | 2.294 ± 0.022 | 3.275 ± 0.077 | 3.970 ± 2.938 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 137.198 ± 7.263 | - | 136.687 ± 8.957 | 133.948 ± 4.504 | 138.215 ± 3.309 | 153.185 ± 2.620 | 155.968 ± 3.060 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 43.405 ± 1.379 | - | 43.003 ± 2.669 | 43.888 ± 0.617 | 69.094 ± 2.752 | 155.169 ± 8.471 | 67.740 ± 0.365 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 109.488 ± 4.138 | - | 116.320 ± 4.844 | 97.365 ± 3.250 | 95.367 ± 1.711 | 133.137 ± 4.957 | 136.844 ± 0.743 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 44.287 ± 0.206 | - | 45.543 ± 1.229 | 33.432 ± 0.966 | 50.572 ± 0.386 | 135.713 ± 2.494 | 65.179 ± 1.314 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 45.413 ± 0.589 | - | 45.200 ± 1.308 | 42.755 ± 0.393 | 67.157 ± 0.383 | 46.849 ± 0.238 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 29.605 ± 0.071 | - | 29.553 ± 0.340 | 29.380 ± 0.083 | 29.728 ± 0.693 | 23.694 ± 0.506 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 45.888 ± 1.871 | - | unsupported | 43.559 ± 0.695 | 67.807 ± 0.350 | 44.925 ± 3.249 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 30.643 ± 0.627 | - | unsupported | 29.718 ± 0.090 | 31.395 ± 1.473 | 22.941 ± 4.480 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 11.020 ± 0.051 | - | 10.996 ± 0.023 | 11.767 ± 0.136 | 11.975 ± 0.080 | 11.950 ± 0.070 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 14.618 ± 0.055 | - | 14.658 ± 0.089 | 14.145 ± 0.042 | 17.694 ± 0.066 | 17.871 ± 0.044 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 71.828 ± 0.196 | - | 72.460 ± 0.247 | 109.190 ± 1.691 | 106.269 ± 1.248 | 87.078 ± 2.109 | 85.604 ± 10.149 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 31.315 ± 0.278 | - | 31.512 ± 0.298 | 43.828 ± 0.535 | 53.965 ± 1.068 | 86.515 ± 1.478 | 41.785 ± 11.871 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 116.078 ± 0.255 | - | 116.119 ± 0.217 | 157.675 ± 1.515 | 179.185 ± 1.213 | 152.682 ± 1.012 | 158.280 ± 0.920 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 49.879 ± 0.172 | - | 50.058 ± 0.268 | 64.079 ± 0.498 | 73.448 ± 0.861 | 228.492 ± 1.278 | 65.020 ± 0.702 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 77.092 ± 2.500 | - | 78.382 ± 3.789 | 73.522 ± 3.506 | 78.135 ± 2.405 | 83.930 ± 0.713 | 83.532 ± 0.738 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 25.327 ± 0.312 | - | 24.726 ± 0.266 | 25.596 ± 1.564 | 39.340 ± 1.043 | 82.014 ± 0.850 | 36.784 ± 0.535 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 2.437 ± 0.032 | - | 21.067 ± 0.116 | 24.291 ± 0.147 | 18.200 ± 0.404 | 4.106 ± 0.061 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 1.101 ± 0.056 | - | 9.965 ± 9.571 | 11.336 ± 0.278 | 2.833 ± 0.582 | 4.246 ± 0.049 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 8.723 ± 0.073 | - | 8.827 ± 0.027 | 9.367 ± 0.032 | 7.825 ± 0.137 | 9.087 ± 0.107 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 5.739 ± 0.068 | - | 13.583 ± 2.717 | 5.735 ± 0.042 | 7.032 ± 0.278 | 6.971 ± 0.923 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 5.886 ± 0.473 | - | 5.925 ± 0.445 | 5.509 ± 0.057 | 5.263 ± 0.190 | 4.936 ± 0.047 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.884 ± 0.155 | - | 14.014 ± 5.306 | 3.670 ± 0.054 | 4.762 ± 0.337 | 4.596 ± 0.189 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 6.615 ± 0.151 | - | 6.720 ± 0.120 | 7.770 ± 0.045 | 5.663 ± 0.053 | 5.428 ± 0.164 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 5.760 ± 0.203 | - | 5.831 ± 0.185 | 5.966 ± 0.089 | 6.421 ± 0.352 | 5.984 ± 1.004 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 45.272 ± 0.861 | - | 43.352 ± 0.180 | 43.067 ± 0.284 | 66.973 ± 0.314 | 44.320 ± 0.296 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 29.707 ± 0.703 | - | 29.032 ± 0.124 | 29.300 ± 0.074 | 30.088 ± 0.869 | 22.443 ± 0.168 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 49.860 ± 0.561 | - | unsupported | 53.076 ± 0.463 | 61.911 ± 0.665 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 24.879 ± 0.128 | - | unsupported | 25.744 ± 0.381 | 35.161 ± 0.140 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 135.493 ± 6.125 | - | 136.774 ± 8.968 | 132.464 ± 1.460 | 138.222 ± 1.328 | 154.644 ± 9.999 | 169.413 ± 10.750 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 42.717 ± 0.868 | - | 42.571 ± 1.057 | 43.107 ± 0.735 | 72.688 ± 5.211 | 149.629 ± 8.142 | 69.266 ± 1.107 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 148.576 ± 10.483 | - | 144.332 ± 9.119 | 141.827 ± 3.627 | 146.447 ± 0.964 | 155.620 ± 10.579 | 182.392 ± 1.764 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 49.980 ± 2.729 | - | 47.227 ± 0.746 | 50.353 ± 4.873 | 75.684 ± 2.457 | 158.526 ± 11.942 | 75.072 ± 1.897 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 27.971 ± 0.292 | - | 60.856 ± 0.187 | 79.604 ± 1.681 | 53.480 ± 0.424 | 82.570 ± 1.342 | 87.690 ± 8.838 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 12.919 ± 0.601 | - | 26.861 ± 0.499 | 34.748 ± 1.468 | 26.700 ± 0.129 | 82.793 ± 9.591 | 32.602 ± 15.201 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 11.055 ± 0.165 | - | 10.954 ± 0.510 | 37.174 ± 2.644 | 40.107 ± 0.233 | 46.205 ± 9.508 | 44.582 ± 12.504 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 7.468 ± 0.397 | - | 7.921 ± 0.123 | 13.514 ± 0.192 | 22.291 ± 0.822 | 47.696 ± 9.721 | 15.047 ± 20.271 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 25.563 ± 0.941 | - | 22.758 ± 0.726 | 55.384 ± 1.261 | 49.670 ± 1.636 | 25.487 ± 0.855 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 15.638 ± 0.646 | - | 16.060 ± 0.654 | 20.488 ± 0.335 | 24.369 ± 0.792 | 24.699 ± 1.307 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 28.610 ± 0.066 | - | 28.782 ± 0.826 | 56.758 ± 2.576 | 110.550 ± 0.817 | 69.602 ± 12.964 | 73.641 ± 12.782 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 12.003 ± 0.129 | - | 12.146 ± 0.053 | 22.942 ± 0.424 | 55.553 ± 0.663 | 70.216 ± 23.192 | 30.724 ± 14.761 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 144.491 ± 4.643 | - | 145.888 ± 9.157 | 143.484 ± 5.278 | 157.716 ± 1.929 | 160.448 ± 5.227 | 180.576 ± 7.051 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 47.476 ± 0.452 | - | 47.219 ± 0.867 | 46.995 ± 2.877 | 81.540 ± 3.279 | 158.239 ± 8.088 | 72.637 ± 3.714 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 20.631 ± 0.049 | - | 21.022 ± 0.103 | 47.478 ± 1.003 | 44.204 ± 1.751 | 65.916 ± 12.894 | 70.389 ± 12.841 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 8.486 ± 0.006 | - | 8.516 ± 0.035 | 17.338 ± 0.370 | 23.179 ± 0.565 | 74.055 ± 12.657 | 38.817 ± 11.037 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 20.517 ± 0.084 | - | 20.722 ± 0.046 | 33.443 ± 1.040 | 29.574 ± 0.062 | 39.532 ± 1.446 | 42.820 ± 10.721 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 8.132 ± 0.046 | - | 8.242 ± 0.114 | 11.666 ± 0.210 | 14.843 ± 0.406 | 38.657 ± 9.700 | 16.381 ± 11.963 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 25.777 ± 0.016 | - | 25.531 ± 0.172 | 53.822 ± 0.373 | 67.813 ± 0.317 | 86.094 ± 19.515 | 93.180 ± 8.326 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 10.507 ± 0.023 | - | 10.722 ± 0.047 | 19.958 ± 0.398 | 34.548 ± 0.289 | 87.730 ± 13.095 | 37.549 ± 14.387 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 13.367 ± 0.053 | - | 13.385 ± 0.078 | 33.781 ± 0.521 | 38.196 ± 0.849 | 47.664 ± 1.133 | 50.996 ± 10.403 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 5.660 ± 0.021 | - | 5.775 ± 0.032 | 11.939 ± 0.297 | 19.632 ± 0.252 | 47.458 ± 9.760 | 20.721 ± 12.287 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.468 ± 0.015 | 0.516 ± 0.009 | unsupported | unsupported | 0.539 ± 0.013 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.355 ± 0.007 | 0.499 ± 0.007 | unsupported | unsupported | 0.434 ± 0.028 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 150.072 ± 4.703 | - | 147.677 ± 10.764 | 139.694 ± 6.052 | 151.115 ± 2.082 | 165.204 ± 3.704 | 181.997 ± 11.158 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 49.071 ± 1.489 | - | 47.102 ± 0.690 | 47.246 ± 0.435 | 76.751 ± 1.886 | 163.047 ± 8.478 | 73.905 ± 2.948 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 145.775 ± 5.134 | - | 147.342 ± 11.738 | 140.727 ± 5.355 | 156.101 ± 5.161 | 171.160 ± 13.900 | 183.679 ± 7.111 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 47.720 ± 0.563 | - | 47.066 ± 0.528 | 47.534 ± 1.033 | 75.678 ± 1.454 | 163.484 ± 3.630 | 79.747 ± 1.225 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 147.122 ± 9.532 | - | 144.903 ± 8.713 | 143.973 ± 4.499 | 149.893 ± 6.628 | 166.174 ± 2.722 | 180.468 ± 5.006 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 48.409 ± 0.788 | - | 47.305 ± 0.587 | 47.940 ± 0.275 | 78.243 ± 2.637 | 161.871 ± 9.509 | 73.734 ± 2.461 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 133.068 ± 4.514 | - | 136.041 ± 8.984 | 130.643 ± 3.982 | 138.015 ± 2.356 | 157.419 ± 7.745 | 168.596 ± 12.416 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 43.145 ± 0.555 | - | 42.899 ± 0.392 | 42.597 ± 0.651 | 69.270 ± 0.734 | 158.101 ± 1.770 | 69.814 ± 1.926 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 58.695 ± 0.166 | - | 59.314 ± 0.129 | 63.839 ± 1.063 | 75.689 ± 0.730 | 127.535 ± 1.085 | 124.340 ± 8.716 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 23.673 ± 0.321 | - | 23.844 ± 0.147 | 22.377 ± 0.718 | 40.260 ± 0.144 | 127.661 ± 1.637 | 50.226 ± 11.616 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 17.449 ± 0.188 | - | 17.282 ± 0.796 | 14.600 ± 0.645 | 27.971 ± 0.769 | 15.791 ± 0.879 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 7.719 ± 0.208 | - | 7.544 ± 0.023 | 7.160 ± 0.238 | 14.496 ± 0.330 | 14.510 ± 0.646 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 2.141 ± 0.041 | - | 2.148 ± 0.077 | 24.718 ± 0.159 | 2.507 ± 0.041 | 1.603 ± 0.085 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.826 ± 0.023 | - | 0.819 ± 0.018 | 11.113 ± 0.794 | 1.633 ± 0.256 | 1.830 ± 0.177 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 2.639 ± 0.077 | - | 2.655 ± 0.052 | 4.095 ± 0.090 | 2.967 ± 0.272 | 1.552 ± 0.188 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 1.366 ± 0.079 | - | 1.384 ± 0.058 | 2.309 ± 0.013 | 1.524 ± 0.108 | 1.914 ± 0.058 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 17.686 ± 0.673 | - | 17.041 ± 0.428 | 13.290 ± 0.782 | 28.258 ± 0.566 | 15.534 ± 1.876 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 7.582 ± 0.457 | - | 7.752 ± 0.196 | 7.058 ± 0.301 | 14.705 ± 0.505 | 14.635 ± 0.340 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 2.273 ± 0.147 | - | 2.216 ± 0.077 | 23.809 ± 0.142 | 2.483 ± 0.034 | 1.546 ± 0.208 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.826 ± 0.021 | - | 0.827 ± 0.026 | 10.720 ± 0.145 | 1.694 ± 0.358 | 1.932 ± 0.206 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 10.510 ± 0.311 | - | 11.402 ± 0.742 | 16.503 ± 4.793 | 12.354 ± 0.583 | 7.870 ± 0.189 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 5.557 ± 0.209 | - | 5.752 ± 0.160 | 8.857 ± 0.163 | 7.121 ± 0.230 | 7.497 ± 0.176 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 12.204 ± 0.100 | - | 12.116 ± 0.177 | 12.934 ± 0.779 | 21.289 ± 0.371 | 12.171 ± 0.996 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 7.201 ± 0.391 | - | 7.569 ± 0.323 | 7.436 ± 0.139 | 11.923 ± 0.410 | 12.513 ± 0.410 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 1.403 ± 0.140 | - | 1.379 ± 0.118 | 2.226 ± 0.190 | 2.084 ± 0.013 | 1.177 ± 0.060 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.486 ± 0.035 | - | 0.469 ± 0.187 | 1.062 ± 0.014 | 1.910 ± 0.702 | 1.136 ± 0.271 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.408 ± 0.057 | - | 1.516 ± 0.016 | 0.927 ± 0.084 | 3.359 ± 0.282 | 1.395 ± 0.122 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.661 ± 0.054 | - | 0.706 ± 0.094 | 0.424 ± 0.036 | 1.567 ± 0.107 | 1.293 ± 0.073 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 12.133 ± 0.173 | - | 12.255 ± 0.128 | 12.345 ± 1.236 | 20.969 ± 0.233 | 13.704 ± 1.559 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 6.962 ± 0.153 | - | 7.221 ± 0.167 | 7.126 ± 0.152 | 11.828 ± 0.394 | 12.291 ± 0.185 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 1.403 ± 0.044 | - | 1.392 ± 0.045 | 2.334 ± 0.123 | 2.095 ± 0.046 | 1.326 ± 0.199 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.500 ± 0.082 | - | 0.564 ± 0.197 | 1.057 ± 0.045 | 1.353 ± 0.742 | 1.214 ± 0.153 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.494 ± 0.088 | - | 1.405 ± 0.118 | 1.007 ± 0.043 | 3.539 ± 0.195 | 1.244 ± 0.167 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.703 ± 0.045 | - | 0.716 ± 0.071 | 0.427 ± 0.041 | 1.845 ± 0.068 | 1.343 ± 0.105 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 1.428 ± 0.079 | 1.443 ± 0.132 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.426 ± 0.018 | 0.437 ± 0.024 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 39.856 ± 0.311 | - | 38.494 ± 0.457 | 58.280 ± 0.331 | 72.147 ± 0.283 | 78.673 ± 7.125 | 85.907 ± 8.124 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 18.343 ± 0.733 | - | 18.382 ± 0.082 | 21.107 ± 0.223 | 39.982 ± 0.835 | 81.084 ± 11.547 | 44.930 ± 12.346 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 150.603 ± 6.961 | - | 156.825 ± 10.926 | 153.978 ± 1.168 | 259.252 ± 6.426 | 279.129 ± 13.604 | 277.568 ± 10.619 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 48.392 ± 1.020 | - | 48.993 ± 0.916 | 49.208 ± 0.504 | 131.497 ± 0.681 | 272.133 ± 5.572 | 109.910 ± 2.355 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 144.900 ± 9.216 | - | 150.624 ± 10.556 | 154.236 ± 10.224 | 166.185 ± 2.357 | 169.488 ± 8.870 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 50.494 ± 0.844 | - | 49.624 ± 0.816 | 54.330 ± 1.063 | 84.498 ± 1.054 | 163.787 ± 5.979 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 138.665 ± 3.888 | - | 141.929 ± 8.861 | 132.038 ± 5.789 | 138.765 ± 0.409 | 156.945 ± 9.273 | 164.452 ± 7.973 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 42.981 ± 1.090 | - | 43.405 ± 1.604 | 43.451 ± 0.803 | 68.868 ± 0.219 | 154.981 ± 3.346 | 68.237 ± 1.093 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 24.726 ± 0.135 | - | 24.911 ± 0.123 | 58.745 ± 2.118 | 110.398 ± 0.866 | 69.893 ± 10.024 | 75.564 ± 13.065 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 10.277 ± 0.034 | - | 10.476 ± 0.087 | 23.348 ± 0.250 | 55.539 ± 0.573 | 69.442 ± 14.676 | 30.077 ± 15.635 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 144.682 ± 8.505 | - | 144.678 ± 10.556 | 194.269 ± 0.842 | 141.065 ± 4.292 | 227.702 ± 9.108 | 235.792 ± 8.325 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 45.742 ± 1.133 | - | 46.235 ± 0.834 | 72.136 ± 0.995 | 74.336 ± 0.993 | 224.218 ± 8.263 | 92.097 ± 2.237 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 139.033 ± 5.596 | - | 141.322 ± 7.005 | 139.459 ± 1.916 | 149.396 ± 2.326 | 155.266 ± 9.944 | 187.451 ± 6.506 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 47.712 ± 0.879 | - | 47.882 ± 0.709 | 48.432 ± 1.337 | 76.460 ± 2.686 | 162.516 ± 1.108 | 73.151 ± 3.328 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 51.324 ± 0.117 | - | 51.308 ± 0.101 | 76.970 ± 2.391 | 42.182 ± 0.920 | 82.237 ± 13.494 | 89.213 ± 27.336 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 20.607 ± 0.269 | - | 20.746 ± 0.089 | 29.526 ± 0.215 | 21.582 ± 0.149 | 81.988 ± 10.555 | 34.918 ± 16.969 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 144.536 ± 11.945 | 140.143 ± 9.060 | unsupported | 162.184 ± 10.420 | 166.393 ± 1.066 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 50.423 ± 17.770 | 48.616 ± 0.895 | unsupported | 54.514 ± 1.634 | 85.677 ± 1.960 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 1.014 ± 0.113 | - | 1.060 ± 0.138 | 0.922 ± 0.034 | 0.987 ± 0.166 | 7.999 ± 4.469 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.529 ± 0.073 | - | 0.546 ± 0.151 | 0.486 ± 0.022 | 1.003 ± 0.167 | 8.255 ± 4.415 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 1.479 ± 0.240 | - | 1.064 ± 0.134 | 0.990 ± 0.119 | 1.213 ± 0.278 | 1.619 ± 1.348 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.639 ± 0.107 | - | 0.493 ± 0.201 | 0.385 ± 0.090 | 1.062 ± 0.241 | 1.785 ± 1.376 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.577 ± 0.313 | - | unsupported | 1.575 ± 0.047 | 1.503 ± 0.195 | 6.434 ± 6.881 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.830 ± 1.264 | - | unsupported | 0.725 ± 0.067 | 1.216 ± 0.145 | 8.720 ± 6.909 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.439 ± 0.051 | - | 0.495 ± 0.039 | 0.685 ± 0.014 | 0.537 ± 0.080 | 1.380 ± 0.063 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.518 ± 3.617 | - | 0.405 ± 0.033 | 0.357 ± 0.015 | 0.316 ± 0.032 | 1.327 ± 0.103 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 1.155 ± 0.112 | - | 1.225 ± 0.114 | 1.303 ± 0.227 | 1.305 ± 0.071 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.591 ± 0.170 | - | 0.691 ± 0.178 | 0.653 ± 0.074 | 0.730 ± 0.119 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 1.098 ± 0.216 | - | 1.111 ± 0.212 | 0.946 ± 0.104 | 0.992 ± 0.243 | 1.574 ± 8.134 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.531 ± 0.099 | - | 0.552 ± 0.066 | 0.525 ± 0.183 | 0.725 ± 0.136 | 9.320 ± 8.038 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.818 ± 0.012 | - | 0.930 ± 0.029 | 0.471 ± 0.009 | 0.545 ± 0.111 | 0.588 ± 0.821 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 1.196 ± 0.015 | - | 1.331 ± 0.026 | 0.629 ± 0.013 | 0.546 ± 0.315 | 1.618 ± 0.192 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 2.370 ± 0.120 | - | 0.003 ± 0.000 | 1.907 ± 0.077 | 1.621 ± 0.050 | 10.076 ± 7.167 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.823 ± 0.049 | - | 0.004 ± 0.000 | 0.957 ± 0.096 | 1.480 ± 0.238 | 10.588 ± 1.475 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.050 ± 0.000 | - | 0.054 ± 0.000 | 0.075 ± 0.001 | 0.098 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.093 ± 0.000 | - | 0.102 ± 0.003 | 0.126 ± 0.003 | 0.154 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.234 ± 0.004 | - | 0.270 ± 0.009 | 0.279 ± 0.015 | 0.338 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.080 ± 0.003 | - | 0.094 ± 0.005 | 0.147 ± 0.006 | 0.120 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.001 | 0.033 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.002 | 0.013 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.015 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.031 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.683 ± 0.013 | - | 0.745 ± 0.020 | 1.162 ± 0.006 | 0.694 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.026 ± 0.000 | - | 0.032 ± 0.000 | 0.039 ± 0.002 | 0.072 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.001 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.039 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.051 ± 0.000 | - | 0.099 ± 0.002 | 0.086 ± 0.003 | 0.088 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.094 ± 0.001 | - | 0.155 ± 0.002 | 0.129 ± 0.020 | 0.141 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.237 ± 0.001 | - | 0.407 ± 0.017 | 0.341 ± 0.018 | 0.367 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.083 ± 0.003 | - | 0.125 ± 0.003 | 0.179 ± 0.018 | 0.133 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.005 ± 0.001 | - | 0.014 ± 0.001 | 0.016 ± 0.000 | 0.012 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.009 ± 0.001 | 0.010 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.009 ± 0.001 | 0.009 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.009 ± 0.001 | 0.010 ± 0.000 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.025 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.009 ± 0.001 | 0.010 ± 0.001 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.009 ± 0.001 | 0.010 ± 0.001 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.003 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.689 ± 0.025 | - | 0.940 ± 0.014 | 1.373 ± 0.020 | 1.996 ± 0.216 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.010 ± 0.004 | 0.010 ± 0.005 | 0.014 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.027 ± 0.000 | - | 0.054 ± 0.004 | 0.059 ± 0.002 | 0.048 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.004 ± 0.000 | - | 0.010 ± 0.001 | 0.010 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.004 ± 0.000 | - | 0.012 ± 0.001 | 0.016 ± 0.042 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.629 ± 0.014 | - | 0.639 ± 0.014 | 0.741 ± 0.016 | 0.896 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 2.116 ± 0.005 | - | 2.145 ± 0.011 | 2.176 ± 0.009 | 2.725 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 6.476 ± 0.014 | - | 6.552 ± 0.034 | 6.598 ± 0.098 | 9.017 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 3.676 ± 0.012 | - | 3.720 ± 0.081 | 3.253 ± 0.033 | 2.724 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.001 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.110 ± 0.001 | - | 0.115 ± 0.005 | 0.123 ± 0.006 | 0.200 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.001 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.008 ± 0.000 | - | 0.013 ± 0.000 | 0.017 ± 0.001 | 0.040 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.001 | 0.024 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.013 ± 0.000 | 0.016 ± 0.004 | 0.031 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.021 ± 0.000 | - | 0.026 ± 0.001 | 0.030 ± 0.001 | 0.069 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.001 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.010 ± 0.008 | - | 0.015 ± 0.000 | 0.018 ± 0.000 | 0.037 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.028 ± 0.009 | - | 0.032 ± 0.000 | 0.036 ± 0.001 | 0.082 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 29.467 ± 0.235 | - | 29.693 ± 0.137 | 25.936 ± 0.151 | 20.869 ± 0.152 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.007 ± 0.000 | - | 0.011 ± 0.001 | 0.014 ± 0.001 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.867 ± 0.004 | - | 0.882 ± 0.003 | 0.880 ± 0.005 | 1.033 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.018 ± 0.000 | - | 0.023 ± 0.004 | 0.027 ± 0.001 | 0.047 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.052 ± 0.001 | - | 0.058 ± 0.001 | 0.061 ± 0.001 | 0.145 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 1.038 ± 0.008 | - | 1.070 ± 0.009 | 0.717 ± 0.439 | 0.942 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 3.446 ± 0.020 | - | 3.474 ± 0.023 | 2.135 ± 0.017 | 2.377 ± 0.186 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 9.994 ± 0.475 | - | 6.626 ± 0.037 | 9.342 ± 0.020 | 5.945 ± 0.143 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 3.581 ± 0.108 | - | 3.600 ± 0.021 | 2.238 ± 0.042 | 3.503 ± 0.069 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.005 ± 0.000 | - | 0.012 ± 0.001 | 0.014 ± 0.004 | 0.026 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.112 ± 0.008 | - | 0.179 ± 0.005 | 0.174 ± 0.011 | 0.178 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.008 ± 0.001 | - | 0.015 ± 0.000 | 0.011 ± 0.001 | 0.031 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.014 ± 0.000 | - | 0.022 ± 0.004 | 0.016 ± 0.002 | 0.039 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.008 ± 0.001 | - | 0.014 ± 0.000 | 0.016 ± 0.002 | 0.028 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.015 ± 0.001 | - | 0.022 ± 0.003 | 0.016 ± 0.003 | 0.039 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.034 ± 0.000 | - | 0.041 ± 0.005 | 0.029 ± 0.001 | 0.069 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.008 ± 0.001 | - | 0.015 ± 0.000 | 0.017 ± 0.003 | 0.033 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.018 ± 0.000 | - | 0.025 ± 0.001 | 0.017 ± 0.001 | 0.034 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.044 ± 0.001 | - | 0.051 ± 0.003 | 0.035 ± 0.000 | 0.087 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 28.716 ± 1.525 | - | 39.023 ± 0.626 | 17.804 ± 0.144 | 26.508 ± 5.864 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.013 ± 0.000 | - | 0.019 ± 0.001 | 0.022 ± 0.001 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.876 ± 0.005 | - | 1.327 ± 0.412 | 1.256 ± 0.019 | 1.281 ± 1.753 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.032 ± 0.001 | - | 0.039 ± 0.004 | 0.026 ± 0.001 | 0.048 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.082 ± 0.002 | - | 0.092 ± 0.001 | 0.060 ± 0.004 | 0.101 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.673 ± 0.011 | - | 0.677 ± 0.012 | 0.526 ± 0.022 | 0.446 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 2.197 ± 0.010 | - | 2.212 ± 0.004 | 1.954 ± 0.016 | 1.966 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 5.813 ± 0.041 | - | 5.855 ± 0.009 | 5.707 ± 0.065 | 6.485 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.318 ± 0.017 | - | 1.360 ± 0.041 | 1.354 ± 0.017 | 1.484 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.003 ± 0.001 | - | 0.006 ± 0.000 | 0.008 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.069 ± 0.002 | - | 0.077 ± 0.001 | 0.091 ± 0.005 | 0.132 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.001 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | 0.012 ± 0.000 | 0.016 ± 0.001 | 0.029 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.013 ± 0.000 | 0.015 ± 0.001 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.019 ± 0.000 | - | 0.023 ± 0.001 | 0.026 ± 0.001 | 0.041 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.022 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.010 ± 0.001 | - | 0.015 ± 0.000 | 0.017 ± 0.001 | 0.036 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.024 ± 0.000 | - | 0.029 ± 0.000 | 0.032 ± 0.005 | 0.048 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 10.484 ± 0.094 | - | 10.598 ± 0.083 | 10.576 ± 0.130 | 11.517 ± 0.088 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.023 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.554 ± 0.010 | - | 0.568 ± 0.014 | 0.637 ± 0.069 | 0.765 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.018 ± 0.000 | - | 0.023 ± 0.003 | 0.024 ± 0.001 | 0.036 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.046 ± 0.000 | - | 0.052 ± 0.005 | 0.053 ± 0.000 | 0.093 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 1.109 ± 0.024 | - | 1.119 ± 0.009 | 0.801 ± 0.100 | 0.455 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 3.566 ± 0.019 | - | 3.575 ± 0.020 | 1.947 ± 0.016 | 1.903 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 5.815 ± 3.180 | - | 5.893 ± 0.045 | 8.542 ± 0.024 | 3.722 ± 0.085 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.262 ± 0.016 | - | 1.270 ± 0.024 | 1.193 ± 0.041 | 1.798 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.005 ± 0.001 | - | 0.012 ± 0.000 | 0.012 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.118 ± 0.017 | - | 0.137 ± 0.011 | 0.133 ± 0.007 | 0.152 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.008 ± 0.000 | - | 0.015 ± 0.001 | 0.012 ± 0.007 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.013 ± 0.000 | - | 0.020 ± 0.001 | 0.015 ± 0.001 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.007 ± 0.000 | - | 0.014 ± 0.000 | 0.015 ± 0.004 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.015 ± 0.001 | - | 0.021 ± 0.001 | 0.015 ± 0.001 | 0.026 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.031 ± 0.002 | - | 0.037 ± 0.003 | 0.026 ± 0.001 | 0.041 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.008 ± 0.000 | - | 0.015 ± 0.000 | 0.015 ± 0.001 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.019 ± 0.000 | - | 0.026 ± 0.001 | 0.017 ± 0.001 | 0.037 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.039 ± 0.002 | - | 0.046 ± 0.004 | 0.032 ± 0.003 | 0.048 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 10.166 ± 0.414 | - | 20.575 ± 0.781 | 9.274 ± 0.199 | 14.987 ± 1.732 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.013 ± 0.000 | - | 0.019 ± 0.000 | 0.018 ± 0.001 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.880 ± 0.007 | - | 0.902 ± 0.007 | 0.921 ± 0.001 | 0.464 ± 0.067 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.031 ± 0.000 | - | 0.039 ± 0.000 | 0.024 ± 0.008 | 0.039 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.074 ± 0.002 | - | 0.082 ± 0.002 | 0.053 ± 0.001 | 0.106 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.276 ± 0.021 | - | 0.288 ± 0.023 | 0.237 ± 0.019 | 0.468 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 1.099 ± 0.013 | - | 1.111 ± 0.005 | 0.953 ± 0.053 | 1.969 ± 0.073 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 3.093 ± 0.005 | - | 3.131 ± 0.012 | 2.903 ± 0.009 | 6.501 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.518 ± 0.011 | - | 0.530 ± 0.012 | 0.558 ± 0.014 | 1.485 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.002 ± 0.002 | - | 0.005 ± 0.000 | 0.010 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.031 ± 0.000 | - | 0.035 ± 0.000 | 0.043 ± 0.002 | 0.171 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.001 | 0.026 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.002 | 0.022 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.010 ± 0.000 | - | 0.014 ± 0.000 | 0.020 ± 0.002 | 0.042 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.002 | 0.010 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.013 ± 0.001 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.013 ± 0.000 | - | 0.017 ± 0.000 | 0.022 ± 0.001 | 0.041 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 4.110 ± 0.015 | - | 4.152 ± 0.018 | 4.303 ± 0.019 | 11.269 ± 0.053 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.232 ± 0.011 | - | 0.238 ± 0.013 | 0.244 ± 0.007 | 0.783 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.010 ± 0.000 | - | 0.014 ± 0.000 | 0.017 ± 0.001 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.025 ± 0.000 | - | 0.029 ± 0.008 | 0.033 ± 0.009 | 0.119 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.461 ± 0.017 | - | 0.477 ± 0.014 | 0.401 ± 0.022 | 0.460 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 1.600 ± 0.014 | - | 1.616 ± 0.021 | 0.946 ± 0.002 | 1.289 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 4.186 ± 0.011 | - | 4.222 ± 0.013 | 3.671 ± 0.013 | 3.692 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.685 ± 0.016 | - | 0.708 ± 0.041 | 0.696 ± 0.021 | 1.774 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.005 ± 0.002 | - | 0.010 ± 0.001 | 0.015 ± 0.001 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.041 ± 0.001 | - | 0.048 ± 0.002 | 0.056 ± 0.003 | 0.120 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | 0.011 ± 0.000 | 0.016 ± 0.001 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.008 ± 0.000 | - | 0.014 ± 0.000 | 0.012 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.005 ± 0.001 | - | 0.011 ± 0.001 | 0.016 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.008 ± 0.000 | - | 0.015 ± 0.001 | 0.019 ± 0.005 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.016 ± 0.000 | - | 0.022 ± 0.001 | 0.019 ± 0.001 | 0.043 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.006 ± 0.000 | - | 0.011 ± 0.000 | 0.016 ± 0.002 | 0.018 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.010 ± 0.001 | - | 0.016 ± 0.000 | 0.020 ± 0.002 | 0.021 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.020 ± 0.000 | - | 0.026 ± 0.003 | 0.022 ± 0.002 | 0.040 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 5.446 ± 0.078 | - | 14.764 ± 1.391 | 5.215 ± 0.065 | 14.989 ± 3.119 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.007 ± 0.001 | - | 0.013 ± 0.001 | 0.018 ± 0.002 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.296 ± 0.014 | - | 0.312 ± 0.019 | 0.319 ± 0.019 | 0.420 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.016 ± 0.000 | - | 0.022 ± 0.001 | 0.024 ± 0.004 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.036 ± 0.000 | - | 0.042 ± 0.003 | 0.033 ± 0.001 | 0.080 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.150 ± 0.001 | - | 0.156 ± 0.003 | 0.192 ± 0.021 | 0.150 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.212 ± 0.002 | - | 0.221 ± 0.013 | 0.256 ± 0.017 | 0.239 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.499 ± 0.011 | - | 0.537 ± 0.012 | 0.541 ± 0.019 | 0.586 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.140 ± 0.006 | - | 0.152 ± 0.004 | 0.175 ± 0.002 | 0.220 ± 0.066 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.014 ± 0.002 | 0.020 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.012 ± 0.000 | - | 0.017 ± 0.005 | 0.026 ± 0.002 | 0.059 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.014 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.014 ± 0.002 | - | 0.019 ± 0.001 | 0.029 ± 0.002 | 0.085 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.014 ± 0.000 | 0.025 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.016 ± 0.001 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.072 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.001 | 0.016 ± 0.001 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.016 ± 0.001 | 0.025 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.021 ± 0.002 | 0.034 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 1.084 ± 0.013 | - | 1.149 ± 0.051 | 1.348 ± 0.031 | 1.274 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.016 ± 0.001 | 0.020 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.076 ± 0.000 | - | 0.086 ± 0.013 | 0.086 ± 0.002 | 0.137 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.001 | 0.031 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.022 ± 0.001 | 0.046 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.201 ± 0.014 | - | 0.215 ± 0.001 | 0.179 ± 0.009 | 0.170 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.293 ± 0.017 | - | 0.317 ± 0.019 | 0.238 ± 0.019 | 0.234 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.771 ± 0.017 | - | 0.829 ± 0.058 | 0.387 ± 0.021 | 0.645 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.195 ± 0.002 | - | 0.221 ± 0.019 | 0.229 ± 0.022 | 0.230 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.006 ± 0.000 | - | 0.011 ± 0.000 | 0.023 ± 0.002 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.022 ± 0.001 | - | 0.027 ± 0.004 | 0.038 ± 0.000 | 0.053 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.006 ± 0.001 | - | 0.011 ± 0.001 | 0.024 ± 0.011 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.024 ± 0.000 | - | 0.032 ± 0.004 | 0.042 ± 0.002 | 0.043 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.006 ± 0.001 | - | 0.011 ± 0.001 | 0.023 ± 0.005 | 0.021 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.006 ± 0.001 | - | 0.011 ± 0.002 | 0.026 ± 0.000 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.006 ± 0.001 | - | 0.011 ± 0.000 | 0.026 ± 0.000 | 0.021 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.008 ± 0.001 | - | 0.013 ± 0.000 | 0.028 ± 0.004 | 0.029 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.006 ± 0.001 | - | 0.011 ± 0.000 | 0.026 ± 0.006 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.006 ± 0.001 | - | 0.012 ± 0.000 | 0.026 ± 0.000 | 0.026 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.008 ± 0.001 | - | 0.014 ± 0.001 | 0.029 ± 0.000 | 0.037 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 1.543 ± 0.034 | - | 1.568 ± 0.036 | 0.712 ± 0.014 | 0.902 ± 0.129 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.012 ± 0.000 | 0.027 ± 0.000 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.125 ± 0.002 | - | 0.135 ± 0.003 | 0.078 ± 0.001 | 0.176 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.008 ± 0.001 | - | 0.013 ± 0.000 | 0.028 ± 0.000 | 0.033 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.011 ± 0.000 | - | 0.017 ± 0.000 | 0.031 ± 0.004 | 0.024 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.247 ± 0.010 | - | 0.251 ± 0.004 | 0.231 ± 0.015 | 0.350 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.612 ± 0.008 | - | 0.623 ± 0.015 | 0.525 ± 0.013 | 0.872 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.346 ± 0.012 | - | 1.391 ± 0.019 | 1.355 ± 0.017 | 2.345 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.493 ± 0.016 | - | 0.509 ± 0.007 | 0.584 ± 0.024 | 0.647 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.018 ± 0.000 | - | 0.022 ± 0.000 | 0.027 ± 0.004 | 0.069 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.019 ± 0.002 | - | 0.024 ± 0.000 | 0.030 ± 0.004 | 0.063 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.001 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.005 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.036 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.007 ± 0.000 | - | 0.011 ± 0.003 | 0.011 ± 0.000 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 3.903 ± 0.018 | - | 4.835 ± 0.021 | 4.499 ± 0.030 | 4.781 ± 0.062 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.132 ± 0.001 | - | 0.138 ± 0.003 | 0.181 ± 0.035 | 0.222 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.006 ± 0.001 | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.033 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.012 ± 0.000 | - | 0.016 ± 0.000 | 0.018 ± 0.014 | 0.039 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.246 ± 0.034 | - | 0.417 ± 0.017 | 0.321 ± 0.015 | 0.338 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.609 ± 0.011 | - | 0.903 ± 0.008 | 0.766 ± 0.023 | 0.897 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.338 ± 0.021 | - | 1.369 ± 0.387 | 2.037 ± 0.004 | 1.332 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.588 ± 0.032 | - | 0.604 ± 0.019 | 0.504 ± 0.018 | 0.995 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.001 | 0.024 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.019 ± 0.001 | - | 0.037 ± 0.003 | 0.045 ± 0.003 | 0.050 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.033 ± 0.003 | - | 0.041 ± 0.001 | 0.048 ± 0.004 | 0.065 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.027 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.007 ± 0.000 | - | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.007 ± 0.000 | - | 0.016 ± 0.001 | 0.016 ± 0.001 | 0.027 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.012 ± 0.001 | 0.011 ± 0.000 | 0.033 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | 0.014 ± 0.000 | 0.013 ± 0.000 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.008 ± 0.000 | - | 0.018 ± 0.001 | 0.018 ± 0.000 | 0.030 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 4.417 ± 0.046 | - | 16.820 ± 1.769 | 3.852 ± 0.048 | 7.984 ± 0.793 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.005 ± 0.000 | - | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.134 ± 0.001 | - | 0.230 ± 0.004 | 0.291 ± 0.014 | 0.325 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.012 ± 0.005 | - | 0.017 ± 0.000 | 0.015 ± 0.000 | 0.032 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.013 ± 0.000 | - | 0.026 ± 0.001 | 0.026 ± 0.001 | 0.038 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.169 ± 0.012 | - | 0.146 ± 0.002 | 0.098 ± 0.001 | 0.274 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.220 ± 0.002 | - | 0.218 ± 0.036 | 0.139 ± 0.002 | 0.390 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.545 ± 0.010 | - | 0.570 ± 0.017 | 0.403 ± 0.014 | 0.808 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.118 ± 0.004 | - | 0.154 ± 0.040 | 0.149 ± 0.013 | 0.190 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.001 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.012 ± 0.000 | - | 0.019 ± 0.003 | 0.030 ± 0.002 | 0.056 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.018 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.021 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.020 ± 0.001 | 0.027 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.020 ± 0.001 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.918 ± 0.008 | - | 0.989 ± 0.006 | 1.029 ± 0.006 | 1.036 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.073 ± 0.002 | - | 0.099 ± 0.024 | 0.123 ± 0.053 | 0.152 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.002 | - | 0.008 ± 0.000 | 0.018 ± 0.001 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.012 ± 0.001 | 0.021 ± 0.001 | 0.028 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.156 ± 0.008 | - | 0.179 ± 0.020 | 0.133 ± 0.005 | 0.263 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.220 ± 0.003 | - | 0.262 ± 0.075 | 0.173 ± 0.002 | 0.374 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.572 ± 0.023 | - | 0.750 ± 0.046 | 0.381 ± 0.018 | 0.846 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.121 ± 0.003 | - | 0.262 ± 0.068 | 0.205 ± 0.023 | 0.176 ± 0.069 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.015 ± 0.000 | 0.027 ± 0.004 | 0.019 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.012 ± 0.001 | - | 0.022 ± 0.002 | 0.043 ± 0.003 | 0.048 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.015 ± 0.000 | 0.027 ± 0.004 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.016 ± 0.000 | 0.027 ± 0.003 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.016 ± 0.000 | 0.026 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.016 ± 0.001 | 0.027 ± 0.003 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.017 ± 0.001 | 0.030 ± 0.003 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.016 ± 0.000 | 0.027 ± 0.001 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.016 ± 0.000 | 0.028 ± 0.005 | 0.020 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.008 ± 0.000 | - | 0.018 ± 0.000 | 0.030 ± 0.001 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.920 ± 0.008 | - | 1.411 ± 0.021 | 0.591 ± 0.561 | 0.832 ± 0.090 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.016 ± 0.000 | 0.029 ± 0.000 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.072 ± 0.006 | - | 0.166 ± 0.043 | 0.072 ± 0.001 | 0.151 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.017 ± 0.000 | 0.030 ± 0.000 | 0.025 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.011 ± 0.000 | - | 0.023 ± 0.003 | 0.033 ± 0.008 | 0.025 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.816 ± 0.008 | - | 1.829 ± 0.009 | 1.558 ± 0.019 | 2.035 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 6.010 ± 0.008 | - | 6.030 ± 0.012 | 5.144 ± 0.017 | 5.139 ± 0.061 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 17.818 ± 0.044 | - | 17.954 ± 0.181 | 16.042 ± 0.063 | 14.876 ± 0.097 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 4.936 ± 0.010 | - | 4.968 ± 0.043 | 5.643 ± 0.049 | 4.219 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.004 ± 0.001 | - | 0.009 ± 0.000 | 0.014 ± 0.001 | 0.029 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.207 ± 0.021 | - | 0.213 ± 0.020 | 0.237 ± 0.016 | 0.247 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.009 ± 0.000 | - | 0.015 ± 0.000 | 0.020 ± 0.001 | 0.034 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.190 ± 0.016 | - | 0.190 ± 0.012 | 0.211 ± 0.019 | 0.224 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.021 ± 0.000 | - | 0.027 ± 0.000 | 0.032 ± 0.002 | 0.045 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.008 ± 0.000 | - | 0.013 ± 0.000 | 0.020 ± 0.009 | 0.033 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.020 ± 0.000 | - | 0.026 ± 0.000 | 0.031 ± 0.001 | 0.043 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.054 ± 0.000 | - | 0.061 ± 0.002 | 0.064 ± 0.003 | 0.073 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.009 ± 0.000 | - | 0.015 ± 0.000 | 0.020 ± 0.001 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.026 ± 0.000 | - | 0.032 ± 0.000 | 0.035 ± 0.002 | 0.050 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.072 ± 0.002 | - | 0.080 ± 0.008 | 0.080 ± 0.002 | 0.089 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 40.048 ± 0.196 | - | 42.248 ± 0.284 | 45.188 ± 0.235 | 33.646 ± 0.336 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.016 ± 0.000 | - | 0.022 ± 0.005 | 0.026 ± 0.011 | 0.042 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.583 ± 0.018 | - | 1.590 ± 0.050 | 1.648 ± 0.013 | 1.539 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.049 ± 0.000 | - | 0.055 ± 0.000 | 0.057 ± 0.007 | 0.069 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.140 ± 0.001 | - | 0.148 ± 0.003 | 0.140 ± 0.005 | 0.164 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 3.065 ± 0.009 | - | 3.081 ± 0.014 | 1.562 ± 0.011 | 2.023 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 9.789 ± 0.023 | - | 6.033 ± 0.049 | 5.075 ± 0.075 | 2.788 ± 0.108 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 17.858 ± 0.073 | - | 18.019 ± 0.277 | 18.685 ± 0.058 | 7.784 ± 0.086 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 4.643 ± 0.057 | - | 4.790 ± 0.026 | 4.724 ± 0.125 | 5.386 ± 0.193 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.009 ± 0.001 | - | 0.016 ± 0.001 | 0.019 ± 0.002 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.320 ± 0.017 | - | 0.346 ± 0.033 | 0.340 ± 0.021 | 0.246 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.016 ± 0.000 | - | 0.026 ± 0.005 | 0.020 ± 0.001 | 0.033 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.276 ± 0.021 | - | 0.314 ± 2.526 | 0.292 ± 0.023 | 0.244 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.022 ± 0.007 | - | 0.044 ± 0.004 | 0.032 ± 0.001 | 0.045 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.015 ± 0.001 | - | 0.022 ± 0.001 | 0.025 ± 0.005 | 0.032 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.035 ± 0.000 | - | 0.045 ± 0.005 | 0.030 ± 0.001 | 0.041 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.056 ± 0.000 | - | 0.100 ± 0.003 | 0.064 ± 0.002 | 0.073 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.018 ± 0.000 | - | 0.025 ± 0.000 | 0.028 ± 0.003 | 0.036 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.045 ± 0.003 | - | 0.053 ± 0.003 | 0.035 ± 0.001 | 0.051 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.074 ± 0.000 | - | 0.130 ± 0.007 | 0.079 ± 0.004 | 0.090 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 37.324 ± 0.664 | - | 48.808 ± 0.436 | 36.617 ± 0.731 | 49.947 ± 22.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.031 ± 0.001 | - | 0.038 ± 0.003 | 0.037 ± 0.004 | 0.044 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 2.406 ± 0.027 | - | 2.394 ± 0.027 | 2.352 ± 0.009 | 0.825 ± 0.137 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.082 ± 0.004 | - | 0.092 ± 0.001 | 0.057 ± 0.003 | 0.068 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.143 ± 0.005 | - | 0.238 ± 0.005 | 0.142 ± 0.006 | 0.183 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 1.390 ± 0.009 | - | 1.416 ± 0.023 | 1.101 ± 0.004 | 1.314 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 5.193 ± 0.017 | - | 5.190 ± 0.016 | 4.367 ± 0.025 | 4.041 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 15.366 ± 0.114 | - | 15.304 ± 0.071 | 13.888 ± 0.128 | 12.294 ± 0.063 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 2.320 ± 0.014 | - | 2.341 ± 0.031 | 2.506 ± 0.015 | 2.426 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.001 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.134 ± 0.007 | - | 0.145 ± 0.008 | 0.165 ± 0.014 | 0.192 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.007 ± 0.000 | - | 0.012 ± 0.000 | 0.018 ± 0.001 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.111 ± 0.005 | - | 0.117 ± 0.007 | 0.139 ± 0.018 | 0.163 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.017 ± 0.000 | - | 0.022 ± 0.008 | 0.030 ± 0.002 | 0.039 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.006 ± 0.000 | - | 0.011 ± 0.000 | 0.017 ± 0.001 | 0.028 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.018 ± 0.005 | - | 0.022 ± 0.000 | 0.028 ± 0.001 | 0.038 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.043 ± 0.000 | - | 0.048 ± 0.003 | 0.054 ± 0.008 | 0.064 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.008 ± 0.000 | - | 0.012 ± 0.005 | 0.017 ± 0.001 | 0.037 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.023 ± 0.000 | - | 0.028 ± 0.000 | 0.031 ± 0.000 | 0.043 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.058 ± 0.000 | - | 0.063 ± 0.006 | 0.065 ± 0.001 | 0.076 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 18.562 ± 0.105 | - | 18.701 ± 0.127 | 19.629 ± 0.205 | 18.681 ± 0.098 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.013 ± 0.000 | - | 0.017 ± 0.000 | 0.021 ± 0.001 | 0.066 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 1.065 ± 0.007 | - | 1.072 ± 0.006 | 1.161 ± 0.008 | 1.155 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.042 ± 0.000 | - | 0.047 ± 0.000 | 0.048 ± 0.001 | 0.071 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.115 ± 0.001 | - | 0.120 ± 0.016 | 0.120 ± 0.037 | 0.137 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 2.270 ± 0.024 | - | 2.279 ± 0.012 | 1.049 ± 0.013 | 1.313 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 8.371 ± 0.030 | - | 5.244 ± 3.266 | 4.308 ± 0.009 | 2.139 ± 0.357 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 15.198 ± 0.113 | - | 15.394 ± 0.341 | 16.791 ± 0.049 | 6.305 ± 1.149 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 2.658 ± 0.029 | - | 2.657 ± 0.030 | 2.773 ± 0.027 | 3.095 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.008 ± 0.001 | - | 0.014 ± 0.006 | 0.018 ± 0.003 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.212 ± 0.011 | - | 0.234 ± 0.031 | 0.253 ± 0.018 | 0.190 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.014 ± 0.004 | - | 0.021 ± 0.002 | 0.018 ± 0.001 | 0.031 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.169 ± 0.008 | - | 0.192 ± 0.022 | 0.200 ± 0.022 | 0.223 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.018 ± 0.000 | - | 0.037 ± 0.003 | 0.029 ± 0.002 | 0.042 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.013 ± 0.001 | - | 0.019 ± 0.010 | 0.022 ± 0.004 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.030 ± 0.000 | - | 0.038 ± 0.003 | 0.027 ± 0.001 | 0.039 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.044 ± 0.004 | - | 0.081 ± 0.008 | 0.053 ± 0.004 | 0.068 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.014 ± 0.000 | - | 0.021 ± 0.001 | 0.024 ± 0.002 | 0.031 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.039 ± 0.003 | - | 0.046 ± 0.003 | 0.031 ± 0.001 | 0.043 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.058 ± 0.000 | - | 0.104 ± 0.006 | 0.065 ± 0.001 | 0.073 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 21.080 ± 0.167 | - | 30.910 ± 0.972 | 21.933 ± 0.070 | 24.751 ± 3.076 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.023 ± 0.000 | - | 0.029 ± 0.003 | 0.030 ± 0.003 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 1.641 ± 0.027 | - | 1.675 ± 0.027 | 1.756 ± 0.007 | 0.669 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.072 ± 0.003 | - | 0.080 ± 0.003 | 0.047 ± 0.001 | 0.059 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.116 ± 0.003 | - | 0.195 ± 0.005 | 0.119 ± 0.004 | 0.143 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.530 ± 0.010 | - | unsupported | 0.367 ± 0.027 | 0.755 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 2.318 ± 0.110 | - | unsupported | 1.940 ± 0.005 | 2.740 ± 0.169 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 5.573 ± 0.011 | - | unsupported | 5.357 ± 0.051 | 6.890 ± 0.064 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.991 ± 0.012 | - | unsupported | 1.166 ± 0.015 | 1.121 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.002 ± 0.001 | - | unsupported | 0.009 ± 0.001 | 0.016 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.047 ± 0.003 | - | unsupported | 0.065 ± 0.009 | 0.077 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | unsupported | 0.012 ± 0.001 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.039 ± 0.000 | - | unsupported | 0.055 ± 0.002 | 0.087 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | unsupported | 0.015 ± 0.001 | 0.023 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.011 ± 0.001 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | unsupported | 0.016 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.017 ± 0.000 | - | unsupported | 0.027 ± 0.002 | 0.039 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | unsupported | 0.011 ± 0.005 | 0.020 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | unsupported | 0.017 ± 0.001 | 0.033 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.023 ± 0.000 | - | unsupported | 0.032 ± 0.004 | 0.044 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 7.701 ± 0.118 | - | unsupported | 9.137 ± 0.082 | 8.697 ± 0.176 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | unsupported | 0.012 ± 0.000 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.375 ± 0.011 | - | unsupported | 0.430 ± 0.018 | 0.496 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.019 ± 0.000 | - | unsupported | 0.025 ± 0.000 | 0.037 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.044 ± 0.000 | - | unsupported | 0.053 ± 0.002 | 0.071 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.828 ± 0.002 | - | unsupported | 0.586 ± 0.023 | 0.752 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 3.618 ± 0.019 | - | unsupported | 1.932 ± 0.004 | 1.420 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 8.432 ± 0.015 | - | unsupported | 7.682 ± 0.007 | 3.533 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.236 ± 0.014 | - | unsupported | 1.454 ± 0.020 | 1.468 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.006 ± 0.004 | - | unsupported | 0.014 ± 0.001 | 0.016 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.066 ± 0.001 | - | unsupported | 0.088 ± 0.017 | 0.076 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.007 ± 0.000 | - | unsupported | 0.017 ± 0.001 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.058 ± 0.008 | - | unsupported | 0.077 ± 0.002 | 0.066 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.012 ± 0.001 | - | unsupported | 0.015 ± 0.001 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.007 ± 0.001 | - | unsupported | 0.016 ± 0.004 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.014 ± 0.000 | - | unsupported | 0.023 ± 0.001 | 0.026 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.028 ± 0.000 | - | unsupported | 0.027 ± 0.001 | 0.036 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.008 ± 0.001 | - | unsupported | 0.016 ± 0.001 | 0.017 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.018 ± 0.000 | - | unsupported | 0.030 ± 0.012 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.037 ± 0.001 | - | unsupported | 0.032 ± 0.001 | 0.044 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 9.745 ± 0.118 | - | unsupported | 11.303 ± 0.075 | 11.559 ± 1.533 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.012 ± 0.002 | - | unsupported | 0.018 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.508 ± 0.015 | - | unsupported | 0.428 ± 0.025 | 0.321 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.032 ± 0.001 | - | unsupported | 0.025 ± 0.001 | 0.045 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.068 ± 0.000 | - | unsupported | 0.053 ± 0.005 | 0.072 ± 0.011 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 1.057 ± 0.011 | - | unsupported | 0.958 ± 0.004 | 3.732 ± 0.166 | 2.591 ± 0.099 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.137 ± 0.004 | - | unsupported | 0.066 ± 0.003 | 0.107 ± 0.010 | 0.132 ± 0.005 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.141 ± 0.003 | - | unsupported | 0.085 ± 0.012 | 0.183 ± 0.007 | 0.218 ± 0.016 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.337 ± 0.010 | - | unsupported | 0.244 ± 0.003 | 0.544 ± 0.030 | 0.492 ± 0.199 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 1.081 ± 0.024 | - | unsupported | 0.495 ± 0.018 | 1.238 ± 0.106 | 2.748 ± 0.114 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.138 ± 0.001 | - | unsupported | 0.061 ± 0.002 | 0.122 ± 0.004 | 0.109 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.143 ± 0.023 | - | unsupported | 0.080 ± 0.002 | 0.181 ± 0.021 | 0.192 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.344 ± 0.012 | - | unsupported | 0.161 ± 0.013 | 0.542 ± 0.044 | 0.725 ± 0.027 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.313 ± 0.008 | - | unsupported | 0.377 ± 0.014 | 0.829 ± 0.002 | 0.431 ± 0.029 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.034 ± 0.000 | - | unsupported | 0.046 ± 0.005 | 0.216 ± 0.017 | 0.074 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.059 ± 0.000 | - | unsupported | 0.061 ± 0.001 | 0.238 ± 0.018 | 0.115 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.199 ± 0.011 | - | unsupported | 0.171 ± 0.016 | 0.385 ± 0.014 | 0.211 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.319 ± 0.011 | - | unsupported | 0.563 ± 0.014 | 0.829 ± 0.078 | 0.424 ± 0.014 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.035 ± 0.000 | - | unsupported | 0.075 ± 0.006 | 0.192 ± 0.023 | 0.075 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.060 ± 0.001 | - | unsupported | 0.100 ± 0.004 | 0.231 ± 0.024 | 0.116 ± 0.006 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.204 ± 0.002 | - | unsupported | 0.226 ± 0.017 | 0.398 ± 0.031 | 0.212 ± 0.013 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.193 ± 0.001 | - | 0.364 ± 0.071 | 0.195 ± 0.014 | 0.395 ± 0.024 | 0.278 ± 0.021 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.080 ± 0.000 | - | 0.086 ± 0.000 | 0.074 ± 0.005 | 0.078 ± 0.003 | 0.058 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.092 ± 0.005 | - | 0.103 ± 0.003 | 0.089 ± 0.003 | 0.102 ± 0.007 | 0.085 ± 0.007 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.121 ± 0.000 | - | 0.155 ± 0.002 | 0.124 ± 0.003 | 0.174 ± 0.005 | 0.138 ± 0.005 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.195 ± 0.001 | - | 0.388 ± 0.016 | 0.256 ± 0.004 | 0.421 ± 0.116 | 0.306 ± 0.062 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.081 ± 0.000 | - | 0.159 ± 0.003 | 0.125 ± 0.005 | 0.080 ± 0.003 | 0.061 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.093 ± 0.000 | - | 0.186 ± 0.016 | 0.142 ± 0.007 | 0.109 ± 0.007 | 0.088 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.122 ± 0.001 | - | 0.231 ± 0.004 | 0.178 ± 0.003 | 0.219 ± 0.012 | 0.142 ± 0.009 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 45.181 ± 1.167 | - | 48.176 ± 0.371 | 63.975 ± 0.439 | 71.417 ± 0.319 | 53.844 ± 7.402 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 21.749 ± 0.516 | - | 43.866 ± 0.231 | 25.112 ± 0.385 | 52.981 ± 20.411 | 40.398 ± 3.590 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 25.994 ± 0.572 | - | 25.573 ± 0.056 | 26.846 ± 0.696 | 22.809 ± 0.438 | 21.436 ± 0.465 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 13.303 ± 0.243 | - | 24.999 ± 10.795 | 11.125 ± 0.111 | 14.521 ± 0.766 | 14.806 ± 6.790 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 7.503 ± 0.056 | - | 7.545 ± 0.096 | 8.203 ± 0.094 | 8.082 ± 0.100 | 8.096 ± 0.156 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 11.271 ± 0.059 | - | 11.390 ± 0.082 | 10.480 ± 0.204 | 11.981 ± 4.426 | 12.366 ± 0.117 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 8.737 ± 0.018 | - | 8.899 ± 0.099 | 9.001 ± 0.146 | 9.635 ± 0.065 | 9.180 ± 0.056 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 13.229 ± 0.066 | - | 13.387 ± 0.665 | 11.931 ± 0.042 | 14.432 ± 0.103 | 13.994 ± 0.062 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 21.428 ± 0.135 | - | 22.211 ± 0.479 | 19.789 ± 0.247 | 39.761 ± 0.185 | 14.480 ± 0.583 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 15.442 ± 0.092 | - | 26.594 ± 0.819 | 15.496 ± 0.318 | 32.925 ± 0.098 | 13.284 ± 0.916 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.512 ± 0.009 | 0.527 ± 0.018 | unsupported | 0.630 ± 0.007 | 0.756 ± 0.032 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 7.992 ± 0.088 | 8.172 ± 0.084 | unsupported | 8.912 ± 0.075 | 10.371 ± 0.176 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.394 ± 0.003 | 0.418 ± 0.006 | unsupported | 0.525 ± 0.002 | 1.327 ± 0.132 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 6.578 ± 0.077 | 16.690 ± 1.177 | unsupported | 5.668 ± 0.022 | 9.620 ± 0.327 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 42.983 ± 1.510 | - | 44.188 ± 1.485 | 45.073 ± 0.533 | 32.968 ± 0.359 | 27.991 ± 0.223 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 32.660 ± 2.282 | - | 32.653 ± 6.541 | 22.208 ± 0.230 | 20.508 ± 0.243 | 16.965 ± 0.285 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 18.510 ± 0.188 | 18.181 ± 0.175 | 9.385 ± 0.100 | 46.384 ± 0.330 | 9.858 ± 0.093 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 30.743 ± 8.250 | 24.602 ± 1.152 | 6.150 ± 0.054 | 42.926 ± 1.363 | 9.202 ± 0.371 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 27.138 ± 0.586 | - | 27.090 ± 0.613 | 34.315 ± 0.737 | 33.111 ± 0.308 | 32.928 ± 8.332 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 13.999 ± 0.542 | - | 25.714 ± 2.624 | 16.373 ± 0.510 | 25.981 ± 0.652 | 22.019 ± 14.235 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 1.903 ± 0.193 | - | 1.665 ± 0.113 | 1.478 ± 0.034 | 22.714 ± 0.549 | 2.743 ± 0.014 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 5.998 ± 0.007 | - | 5.998 ± 5.321 | 1.487 ± 0.039 | 13.418 ± 2.430 | 2.972 ± 0.061 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 20.692 ± 0.298 | - | 20.443 ± 0.208 | 21.246 ± 0.180 | 20.693 ± 0.355 | 19.876 ± 2.152 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 26.008 ± 1.035 | - | 28.999 ± 0.103 | 16.876 ± 0.307 | 20.686 ± 1.228 | 20.178 ± 0.359 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 20.837 ± 0.219 | - | 20.408 ± 0.355 | 21.140 ± 0.255 | 20.491 ± 0.259 | 19.654 ± 1.700 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 26.021 ± 2.097 | - | 28.984 ± 1.980 | 16.710 ± 0.557 | 20.966 ± 0.612 | 20.096 ± 4.327 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.196 ± 0.013 | 1.226 ± 0.012 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 23.198 ± 0.153 | 23.488 ± 0.209 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.349 ± 0.039 | 1.394 ± 0.018 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 19.672 ± 0.975 | 23.710 ± 0.548 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.112 ± 0.010 | 1.124 ± 0.008 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 22.295 ± 0.256 | 22.431 ± 0.065 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.249 ± 0.019 | 1.284 ± 0.055 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 18.105 ± 0.214 | 21.996 ± 0.253 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 26.251 ± 0.787 | - | 25.821 ± 0.969 | 26.789 ± 0.397 | 22.716 ± 0.445 | 21.383 ± 0.307 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 13.624 ± 0.314 | - | 25.003 ± 9.128 | 11.336 ± 0.098 | 14.272 ± 0.146 | 12.878 ± 0.753 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 63.260 ± 0.756 | 65.092 ± 0.798 | 62.099 ± 0.869 | 63.778 ± 0.666 | 66.826 ± 6.929 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 49.750 ± 0.925 | 47.211 ± 0.265 | 38.867 ± 0.571 | 55.911 ± 2.030 | 51.902 ± 5.190 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 33.880 ± 0.261 | - | 110.557 ± 2.519 | 30.940 ± 0.345 | 139.919 ± 3.341 | 30.083 ± 0.378 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 24.561 ± 0.805 | - | 69.863 ± 20.377 | 24.048 ± 2.679 | 125.401 ± 15.896 | 21.419 ± 1.241 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.140 ± 0.001 | 0.140 ± 0.002 | unsupported | 0.167 ± 0.001 | 0.157 ± 0.009 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.223 ± 0.010 | 2.218 ± 0.013 | unsupported | 2.505 ± 0.028 | 0.656 ± 0.838 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.074 ± 0.002 | 0.076 ± 0.003 | unsupported | 0.078 ± 0.003 | 0.127 ± 0.012 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.895 ± 0.044 | 0.897 ± 0.006 | unsupported | 0.751 ± 0.018 | 0.580 ± 0.473 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.078 ± 0.000 | 0.079 ± 0.000 | unsupported | 0.146 ± 0.021 | 0.242 ± 0.007 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.239 ± 0.007 | 1.244 ± 0.003 | unsupported | 1.838 ± 0.026 | 2.783 ± 0.043 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.038 ± 0.000 | 0.039 ± 0.002 | unsupported | 0.096 ± 0.031 | 0.163 ± 0.003 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.562 ± 0.003 | 0.562 ± 0.003 | unsupported | 0.774 ± 0.027 | 1.131 ± 0.094 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.197 ± 0.001 | 0.198 ± 0.028 | unsupported | 0.293 ± 0.004 | 0.194 ± 0.002 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 3.135 ± 0.008 | 3.128 ± 0.015 | unsupported | 4.517 ± 0.039 | 1.971 ± 0.025 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.077 ± 0.001 | 0.078 ± 0.002 | unsupported | 0.108 ± 0.005 | 0.148 ± 0.007 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 1.185 ± 0.001 | 1.184 ± 0.050 | unsupported | 1.542 ± 0.023 | 1.004 ± 0.012 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.059 ± 0.003 | 0.098 ± 0.003 | unsupported | 0.167 ± 0.004 | 0.205 ± 0.022 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.713 ± 0.038 | 0.785 ± 0.041 | unsupported | 1.834 ± 0.040 | 2.262 ± 0.179 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.099 ± 0.005 | 0.160 ± 0.005 | unsupported | 0.070 ± 0.001 | 0.185 ± 0.003 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.432 ± 0.011 | 0.538 ± 0.045 | unsupported | 0.372 ± 0.003 | 1.270 ± 0.136 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.133 ± 0.001 | 0.156 ± 0.001 | unsupported | 0.143 ± 0.002 | 0.237 ± 0.002 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.728 ± 0.017 | 1.797 ± 0.062 | unsupported | 2.171 ± 0.026 | 2.487 ± 0.059 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.134 ± 0.006 | 0.208 ± 0.006 | unsupported | 0.064 ± 0.001 | 0.215 ± 0.012 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.946 ± 0.038 | 1.007 ± 0.033 | unsupported | 0.497 ± 0.009 | 1.472 ± 0.062 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.300 ± 0.006 | 0.347 ± 0.006 | unsupported | unsupported | 0.256 ± 0.019 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 4.490 ± 0.047 | 4.540 ± 0.099 | unsupported | unsupported | 3.489 ± 0.072 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.330 ± 0.007 | 0.399 ± 0.013 | unsupported | unsupported | 0.252 ± 0.007 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 1.987 ± 0.056 | 2.115 ± 0.050 | unsupported | unsupported | 2.126 ± 0.143 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.012 ± 0.001 | 0.021 ± 0.000 | unsupported | 0.013 ± 0.001 | 0.051 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.116 ± 0.044 | 0.115 ± 0.017 | unsupported | 0.109 ± 0.015 | 0.251 ± 0.006 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.011 ± 0.002 | 0.033 ± 0.002 | unsupported | 0.015 ± 0.000 | 0.038 ± 0.000 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.128 ± 0.044 | 0.119 ± 0.032 | unsupported | 0.098 ± 0.003 | 0.159 ± 0.008 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.031 ± 0.003 | 0.056 ± 0.008 | unsupported | 0.112 ± 0.002 | 0.177 ± 0.004 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.383 ± 0.011 | 0.428 ± 0.044 | unsupported | 1.501 ± 0.041 | 2.022 ± 0.083 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.049 ± 0.004 | 0.088 ± 0.007 | unsupported | 0.106 ± 0.005 | 0.159 ± 0.005 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.249 ± 0.062 | 0.311 ± 0.031 | unsupported | 0.286 ± 0.049 | 1.132 ± 0.076 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.055 ± 0.000 | 0.055 ± 0.001 | unsupported | 0.126 ± 0.002 | 0.201 ± 0.024 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.856 ± 0.006 | 0.867 ± 0.007 | unsupported | 1.852 ± 0.014 | 1.881 ± 0.045 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.028 ± 0.000 | 0.028 ± 0.001 | unsupported | 0.099 ± 0.002 | 0.146 ± 0.015 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.391 ± 0.006 | 0.389 ± 0.004 | unsupported | 0.712 ± 0.014 | 1.005 ± 0.025 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.056 ± 0.000 | 0.057 ± 0.000 | unsupported | 0.128 ± 0.003 | 0.184 ± 0.001 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.887 ± 0.003 | 0.897 ± 0.007 | unsupported | 1.915 ± 0.021 | 1.879 ± 0.046 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.028 ± 0.001 | 0.029 ± 0.001 | unsupported | 0.102 ± 0.003 | 0.146 ± 0.004 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.408 ± 0.004 | 0.406 ± 0.010 | unsupported | 0.755 ± 0.019 | 1.021 ± 0.063 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.129 ± 0.002 | 0.161 ± 0.003 | unsupported | 0.146 ± 0.001 | 0.230 ± 0.006 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.758 ± 0.034 | 1.796 ± 0.056 | unsupported | 1.239 ± 0.016 | 2.511 ± 0.147 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.130 ± 0.006 | 0.201 ± 0.004 | unsupported | 0.062 ± 0.001 | 0.209 ± 0.004 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 1.001 ± 0.060 | 1.093 ± 0.625 | unsupported | 0.475 ± 0.002 | 1.392 ± 0.088 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.183 ± 0.001 | 0.184 ± 0.001 | unsupported | 0.280 ± 0.003 | 0.331 ± 0.003 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 2.920 ± 0.026 | 2.921 ± 0.023 | unsupported | 4.382 ± 0.014 | 4.133 ± 0.042 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.074 ± 0.002 | 0.074 ± 0.001 | unsupported | 0.203 ± 0.002 | 0.219 ± 0.009 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 1.129 ± 0.003 | 1.129 ± 0.003 | unsupported | 1.585 ± 0.024 | 2.104 ± 0.025 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.741 ± 0.012 | 1.443 ± 0.012 | unsupported | 0.276 ± 0.003 | 0.231 ± 0.003 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 18.616 ± 0.115 | 34.232 ± 0.311 | unsupported | 6.300 ± 0.070 | 2.587 ± 0.046 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.927 ± 0.037 | 1.793 ± 0.025 | unsupported | 0.300 ± 0.005 | 0.172 ± 0.003 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 10.183 ± 0.259 | 23.559 ± 0.476 | unsupported | 2.558 ± 0.168 | 1.274 ± 0.026 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 31.977 ± 0.872 | - | unsupported | 29.562 ± 0.670 | - | 33.619 ± 0.510 | 41.471 ± 0.800 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 23.825 ± 1.206 | - | unsupported | 24.441 ± 0.751 | - | 35.652 ± 1.050 | 23.593 ± 0.865 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 25.095 ± 1.419 | - | unsupported | 27.932 ± 0.971 | - | 23.086 ± 0.637 | 24.329 ± 0.240 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 16.537 ± 0.236 | - | unsupported | 21.369 ± 0.648 | - | 27.334 ± 1.688 | 20.299 ± 0.532 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 17.033 ± 0.923 | - | unsupported | 23.235 ± 0.498 | - | 14.675 ± 0.220 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 16.435 ± 0.306 | - | unsupported | 21.830 ± 1.229 | - | 18.028 ± 2.067 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 31.440 ± 1.817 | - | unsupported | 28.932 ± 1.289 | - | 33.801 ± 0.373 | 46.482 ± 2.500 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 23.776 ± 0.499 | - | unsupported | 27.022 ± 3.194 | - | 33.787 ± 0.804 | 24.692 ± 0.984 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 51.090 ± 0.401 | - | unsupported | 52.528 ± 0.734 | - | 48.440 ± 0.275 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 24.915 ± 0.354 | - | unsupported | 25.491 ± 0.258 | - | 23.977 ± 0.331 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 49.642 ± 0.588 | - | unsupported | 53.056 ± 0.508 | - | 48.126 ± 0.324 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 24.556 ± 0.459 | - | unsupported | 25.573 ± 0.288 | - | 23.457 ± 0.122 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 31.408 ± 1.274 | - | unsupported | 29.054 ± 0.591 | - | 34.355 ± 0.791 | 41.520 ± 0.426 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 23.417 ± 0.712 | - | unsupported | 24.026 ± 0.948 | - | 34.279 ± 0.597 | 24.734 ± 1.285 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 26.300 ± 1.324 | - | unsupported | 23.257 ± 0.351 | - | 31.484 ± 0.387 | 32.679 ± 0.928 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 16.575 ± 0.758 | - | unsupported | 20.156 ± 0.533 | - | 28.295 ± 0.580 | 19.508 ± 2.385 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.749 ± 0.003 | unsupported | unsupported | 0.909 ± 0.007 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 11.250 ± 0.207 | unsupported | unsupported | 13.240 ± 0.116 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.513 ± 0.010 | unsupported | unsupported | 0.615 ± 0.009 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 6.014 ± 0.020 | unsupported | unsupported | 6.719 ± 0.149 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 30.782 ± 1.044 | - | unsupported | 30.013 ± 0.503 | - | 33.615 ± 1.117 | 41.016 ± 0.376 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 24.186 ± 1.253 | - | unsupported | 24.816 ± 0.400 | - | 34.732 ± 0.783 | 24.394 ± 1.084 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 115.178 ± 7.833 | - | 0.003 ± 0.000 | 114.832 ± 1.974 | 113.144 ± 2.621 | 143.408 ± 4.405 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 45.660 ± 0.590 | - | 0.004 ± 0.000 | 37.160 ± 0.602 | 61.104 ± 1.022 | 144.941 ± 6.490 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 73.948 ± 5.467 | - | 223.744 ± 10.089 | 74.090 ± 1.491 | 73.423 ± 0.769 | 87.152 ± 1.568 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 24.700 ± 1.134 | - | 94.345 ± 3.259 | 24.006 ± 0.375 | 37.706 ± 0.303 | 82.834 ± 2.553 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 234.398 ± 4.025 | - | 234.838 ± 4.074 | 227.852 ± 3.234 | 233.794 ± 1.169 | 272.536 ± 12.452 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 74.833 ± 2.006 | - | 75.053 ± 2.363 | 76.050 ± 2.520 | 118.352 ± 0.482 | 269.954 ± 5.427 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 72.861 ± 1.730 | - | 74.137 ± 3.159 | 77.406 ± 3.327 | 72.455 ± 0.424 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 39.673 ± 0.627 | - | 38.270 ± 1.092 | 25.935 ± 0.599 | 38.717 ± 0.753 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 148.513 ± 5.177 | - | 0.003 ± 0.000 | 137.244 ± 4.506 | 132.341 ± 5.690 | 168.340 ± 5.631 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 43.460 ± 0.389 | - | 0.004 ± 0.000 | 42.584 ± 1.266 | 48.563 ± 7.536 | 166.285 ± 2.813 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 95.515 ± 1.429 | - | 0.003 ± 0.000 | 83.534 ± 1.400 | 87.277 ± 0.660 | 218.899 ± 2.711 | 134.135 ± 0.947 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 32.566 ± 0.454 | - | 0.004 ± 0.000 | 85.282 ± 1.541 | 47.385 ± 0.664 | 217.880 ± 1.787 | 54.069 ± 1.919 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 64.386 ± 2.254 | - | 65.494 ± 2.177 | 68.223 ± 2.635 | 71.114 ± 2.679 | 86.513 ± 2.082 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 22.284 ± 0.092 | - | 22.395 ± 0.093 | 24.648 ± 1.112 | 36.089 ± 1.947 | 86.297 ± 3.879 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 66.990 ± 2.477 | - | 65.031 ± 1.501 | 68.464 ± 2.264 | 70.135 ± 0.638 | 87.414 ± 5.150 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 21.588 ± 0.418 | - | 21.779 ± 0.664 | 23.900 ± 0.562 | 35.693 ± 0.580 | 88.791 ± 4.369 | - |
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
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 8.624080 | 539005.00 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 7.491885 | 468242.81 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 8.248547 | 515534.19 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 6.949210 | 434325.62 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 5.674713 | 354669.56 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 7.989747 | 499359.19 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 22.841287 | 1427580.44 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 23.088288 | 1443018.00 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 6.814852 | 425928.25 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 6.998706 | 437419.12 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 12.091430 | 755714.38 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 10.077601 | 629850.06 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 8.185185 | 511574.06 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 8.439539 | 527471.19 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 165.932749 | 10370796.81 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 142.597222 | 8912326.38 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 127.870425 | 7991901.56 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 130.747450 | 8171715.62 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 21.227468 | 1326716.75 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 8.400544 | 525034.00 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 6.307597 | 394224.81 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 6.691744 | 418234.00 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 153.915465 | 9619716.56 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 90.680126 | 5667507.88 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 105.242971 | 6577685.69 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 267.040718 | 16690044.88 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 19.139633 | 1196227.06 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 19.613647 | 1225852.94 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 371.163609 | 23197725.56 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 375.809617 | 23488101.06 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 21.585912 | 1349119.50 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 22.300466 | 1393779.12 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 314.748191 | 19671761.94 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 379.360891 | 23710055.69 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 17.790427 | 1111901.69 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 17.981603 | 1123850.19 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 356.726860 | 22295428.75 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 358.898300 | 22431143.75 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 19.985814 | 1249113.38 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 20.540039 | 1283752.44 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 289.686302 | 18105393.88 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 351.937539 | 21996096.19 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.038162 | 156846.28 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 10.656371 | 166505.80 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.959368 | 139990.12 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.972632 | 140197.38 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.250798 | 656349.75 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 20.043874 | 2505484.25 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 17.780266 | 2222533.25 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 17.746733 | 2218341.62 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.130638 | 127041.22 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.022566 | 78477.59 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.748375 | 74193.36 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.840808 | 75637.62 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.640411 | 580051.38 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.010832 | 751354.00 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.159394 | 894924.25 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.179732 | 897466.50 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 15.459347 | 241552.30 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 9.335006 | 145859.47 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.015954 | 78374.28 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.078430 | 79350.47 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 22.265746 | 2783218.25 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.703106 | 1837888.25 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.913001 | 1239125.12 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 9.954248 | 1244281.00 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.460371 | 163443.30 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.158857 | 96232.14 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.435147 | 38049.17 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.469410 | 38584.53 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.050285 | 1131285.62 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.189214 | 773651.75 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.498564 | 562320.50 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.495859 | 561982.38 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.441372 | 194396.44 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 18.778294 | 293410.84 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 12.586210 | 196659.53 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 12.694883 | 198357.55 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 15.764797 | 1970599.62 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 36.136592 | 4517074.00 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 25.078295 | 3134786.88 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 25.021348 | 3127668.50 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.458703 | 147792.23 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.922573 | 108165.20 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.958357 | 77474.33 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.996509 | 78070.45 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.035451 | 1004431.38 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.332473 | 1541559.12 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.477084 | 1184635.50 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 9.474239 | 1184279.88 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.151215 | 205487.73 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 10.676728 | 166823.88 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.750914 | 58608.03 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.257078 | 97766.84 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.094009 | 2261751.12 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.670741 | 1833842.62 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.702381 | 712797.62 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.282814 | 785351.75 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.845366 | 185083.84 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.501885 | 70341.95 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.333364 | 98958.81 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 10.231782 | 159871.59 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.162641 | 1270330.12 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.979231 | 372403.88 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.457779 | 432222.38 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.300283 | 537535.38 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 15.184525 | 237258.20 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 9.145333 | 142895.83 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.490242 | 132660.03 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 10.013653 | 156463.33 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 19.892873 | 2486609.12 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 17.364629 | 2170578.62 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.823872 | 1727984.00 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 14.372495 | 1796561.88 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.744497 | 214757.77 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.106576 | 64165.25 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.555963 | 133686.92 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 13.321801 | 208153.14 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.778311 | 1472288.88 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.978046 | 497255.75 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.571033 | 946379.12 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 8.056530 | 1007066.25 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 16.406773 | 256355.83 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 19.180006 | 299687.59 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 22.214565 | 347102.58 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 27.909095 | 3488636.88 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 35.917362 | 4489670.25 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 36.316837 | 4539604.62 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 16.129478 | 252023.09 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 21.108759 | 329824.36 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 25.565640 | 399463.12 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 17.010566 | 2126320.75 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 15.894012 | 1986751.50 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 16.923264 | 2115408.00 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 3.248215 | 50753.36 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.859422 | 13428.47 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.758234 | 11847.41 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.343986 | 20999.78 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 2.006009 | 250751.13 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.871916 | 108989.50 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.924353 | 115544.12 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.920948 | 115118.50 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.446852 | 38232.06 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.953450 | 14897.66 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.728129 | 11377.02 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.097326 | 32770.72 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.270719 | 158839.88 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.786548 | 98318.50 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.024634 | 128079.25 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 0.948111 | 118513.88 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.354576 | 177415.25 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.195777 | 112434.02 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.973641 | 30838.14 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.555831 | 55559.86 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 16.173789 | 2021723.62 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 12.010040 | 1501255.00 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.065405 | 383175.62 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.426187 | 428273.38 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.194132 | 159283.31 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.755211 | 105550.17 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.105799 | 48528.11 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.618780 | 87793.44 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.055515 | 1131939.38 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.284174 | 285521.75 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.991169 | 248896.12 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.487977 | 310997.12 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.885535 | 201336.48 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.081208 | 126268.88 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.501078 | 54704.34 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.531386 | 55177.91 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 15.049533 | 1881191.62 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.818992 | 1852374.00 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.844323 | 855540.38 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.933970 | 866746.25 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.356110 | 146189.22 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.317314 | 98708.03 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.760527 | 27508.23 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.809028 | 28266.06 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.038756 | 1004844.50 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.698418 | 712302.25 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.126069 | 390758.62 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.114919 | 389364.88 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.796280 | 184316.88 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.216811 | 128387.67 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.603800 | 56309.38 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.626603 | 56665.67 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 15.032783 | 1879097.88 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 15.322941 | 1915367.62 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.097645 | 887205.62 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.174355 | 896794.38 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.346104 | 146032.88 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.503532 | 101617.69 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.818515 | 28414.30 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.856346 | 29005.41 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.166304 | 1020788.00 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.041298 | 755162.25 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.267033 | 408379.12 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.249129 | 406141.12 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.694001 | 229593.77 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 9.331250 | 145800.78 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.270062 | 129219.72 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 10.289608 | 160775.12 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 20.089438 | 2511179.75 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.913485 | 1239185.62 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 14.066745 | 1758343.12 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 14.369550 | 1796193.75 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.370898 | 208920.28 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.976824 | 62137.88 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.326956 | 130108.69 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 12.890797 | 201418.70 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.138756 | 1392344.50 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.796698 | 474587.25 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.004263 | 1000532.88 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 8.745758 | 1093219.75 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 21.154764 | 330543.19 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.888375 | 279505.86 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 11.691943 | 182686.61 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 11.789595 | 184212.42 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 33.066569 | 4133321.12 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 35.055605 | 4381950.62 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 23.363718 | 2920464.75 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 23.365552 | 2920694.00 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 14.011942 | 218936.59 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 12.982757 | 202855.58 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.714893 | 73670.20 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.738407 | 74037.61 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 16.830307 | 2103788.38 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.681645 | 1585205.62 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.035800 | 1129475.00 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 9.030199 | 1128774.88 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.778889 | 230920.14 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.659839 | 275934.98 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 47.436678 | 741198.09 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 92.320065 | 1442501.02 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 20.697242 | 2587155.25 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 50.399799 | 6299974.88 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 148.928354 | 18616044.25 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 273.854139 | 34231767.38 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.030124 | 172345.69 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 19.227915 | 300436.17 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 59.333183 | 927080.98 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 114.780781 | 1793449.70 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.188409 | 1273551.12 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 20.465046 | 2558130.75 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 81.467048 | 10183381.00 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 188.470280 | 23558785.00 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 14.537897 | 908618.56 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 11.987354 | 749209.62 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 211.838732 | 13239920.75 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 179.997777 | 11249861.06 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 9.839629 | 614976.81 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 8.209647 | 513102.94 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 107.507170 | 6719198.12 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 96.222516 | 6013907.25 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.023874 | 1492.12 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002505 | 156.56 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.023474 | 1467.12 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002375 | 148.44 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000130 | 8.12 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.022912 | 1432.00 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.007153 | 447.06 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000121 | 7.56 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.022001 | 1375.06 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.007485 | 467.81 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000151 | 9.44 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.023314 | 1457.12 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.006211 | 388.19 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000281 | 17.56 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.023193 | 1449.56 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.005159 | 322.44 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000111 | 6.94 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.021410 | 1338.12 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003286 | 205.38 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000120 | 7.50 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.020579 | 1286.19 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003236 | 202.25 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 11.336 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 24.718 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 24.718 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 24.718 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 11.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 13.6x faster than the slowest successful cell (`pytorch-cpu`, 11.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 23.809 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 23.809 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 23.809 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 10.720 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 10.720 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `pytorch-cpu` is 17.0x faster than the slowest successful cell (`julia-base`, 8.255 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `tenferro-direct` is 15.6x faster than the slowest successful cell (`julia-base`, 8.255 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `tenferro-trace` is 15.1x faster than the slowest successful cell (`julia-base`, 8.255 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 12.0x faster than the slowest successful cell (`julia-base`, 8.720 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`julia-base`, 8.720 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `jax-cpu` is 12.8x faster than the slowest successful cell (`julia-base`, 9.320 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 17.8x faster than the slowest successful cell (`julia-base`, 9.320 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 17.6x faster than the slowest successful cell (`julia-base`, 9.320 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-trace` is 16.9x faster than the slowest successful cell (`julia-base`, 9.320 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 3724.9x faster than the slowest successful cell (`julia-base`, 10.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `pytorch-cpu` is 11.1x faster than the slowest successful cell (`julia-base`, 10.588 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-direct` is 12.9x faster than the slowest successful cell (`julia-base`, 10.588 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 2596.3x faster than the slowest successful cell (`julia-base`, 10.588 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 15.5x faster than the slowest successful cell (`jax-cpu`, 0.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 12.8x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x8x8`): `tenferro-direct` is 11.1x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`3x2x2`): `tenferro-direct` is 14.5x faster than the slowest successful cell (`jax-cpu`, 0.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`3x4x4`): `tenferro-direct` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x2x2`): `tenferro-direct` is 14.3x faster than the slowest successful cell (`jax-cpu`, 0.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x4x4`): `tenferro-direct` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.013 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 17.4x faster than the slowest successful cell (`jax-cpu`, 0.031 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x2x2`): `tenferro-direct` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 14.4x faster than the slowest successful cell (`jax-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=1, shape=`1x2x2`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.024 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 19.1x faster than the slowest successful cell (`jax-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 12.8x faster than the slowest successful cell (`jax-cpu`, 0.024 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x4x4,rhs=1`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.019 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `pytorch-cpu` is 15.4x faster than the slowest successful cell (`jax-cpu`, 22.714 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 11.9x faster than the slowest successful cell (`jax-cpu`, 22.714 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 13.6x faster than the slowest successful cell (`jax-cpu`, 22.714 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 13.2x faster than the slowest successful cell (`tenferro-eager`, 34.232 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 10.4x faster than the slowest successful cell (`tenferro-eager`, 1.793 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 18.5x faster than the slowest successful cell (`tenferro-eager`, 23.559 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 51865.4x faster than the slowest successful cell (`julia-base`, 143.408 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 35717.4x faster than the slowest successful cell (`julia-base`, 144.941 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 61550.3x faster than the slowest successful cell (`julia-base`, 168.340 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 39800.2x faster than the slowest successful cell (`julia-base`, 166.285 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 86727.2x faster than the slowest successful cell (`julia-base`, 218.899 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 53038.0x faster than the slowest successful cell (`julia-base`, 217.880 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 179.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 171.9x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 161.9x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 80.6x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 191.1x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 160.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 6199.3 GB/s at 0.003 ms. The conservative physical lower bound is 0.168 ms, so the cell is 62.0x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 4112.1 GB/s at 0.004 ms. The conservative physical lower bound is 0.042 ms, so the cell is 10.3x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 97083.3 GB/s at 0.003 ms. The conservative physical lower bound is 2.684 ms, so the cell is 970.8x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 66149.7 GB/s at 0.004 ms. The conservative physical lower bound is 0.671 ms, so the cell is 165.4x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 98148.2 GB/s at 0.003 ms. The conservative physical lower bound is 2.684 ms, so the cell is 981.5x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 64249.8 GB/s at 0.004 ms. The conservative physical lower bound is 0.671 ms, so the cell is 160.6x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 53176.6 GB/s at 0.003 ms. The conservative physical lower bound is 1.342 ms, so the cell is 531.8x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 32672.3 GB/s at 0.004 ms. The conservative physical lower bound is 0.336 ms, so the cell is 81.7x faster than the bound.

# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/amd-cpu/cpu/public_api/20261031_230828/run.yaml`
- Timestamp: `2026-10-31T23:08:28.314699+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/amd-cpu/cpu/public_api/20261031_230828`.

- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`

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
- Python platform: `Linux-7.0.0-1017-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`

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

- CSV: `data/results/amd-cpu/cpu/public_api/20261031_230828/cpu_public_api_t1_20261031_230828.csv`
- CSV: `data/results/amd-cpu/cpu/public_api/20261031_230828/cpu_public_api_t4_20261031_230828.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 3.446 ± 0.039 | - | 3.549 ± 0.030 | 3.695 ± 0.021 | 3.515 ± 0.016 | 3.644 ± 2.006 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.104 ± 0.014 | - | 5.999 ± 0.001 | 1.893 ± 0.015 | 2.559 ± 0.134 | 2.629 ± 0.046 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 110.667 ± 7.492 | - | 110.830 ± 7.004 | 108.628 ± 3.791 | 116.639 ± 2.738 | 121.042 ± 5.678 | 123.925 ± 5.310 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 35.483 ± 0.437 | - | 37.642 ± 1.985 | 35.465 ± 0.474 | 56.683 ± 4.697 | 120.132 ± 4.388 | 55.782 ± 1.648 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 95.868 ± 3.641 | - | 105.591 ± 5.947 | 86.894 ± 2.614 | 82.139 ± 2.167 | 116.472 ± 2.480 | 123.102 ± 3.306 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 33.729 ± 0.657 | - | 35.169 ± 0.690 | 26.473 ± 0.228 | 41.589 ± 1.476 | 102.706 ± 2.980 | 47.610 ± 0.763 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 40.194 ± 0.190 | - | 40.214 ± 0.241 | 39.752 ± 0.068 | 57.203 ± 0.105 | 42.920 ± 0.166 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 25.134 ± 0.027 | - | 25.147 ± 0.056 | 25.260 ± 0.028 | 25.685 ± 0.248 | 21.108 ± 0.060 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 41.205 ± 0.348 | - | unsupported | 40.080 ± 0.041 | 57.723 ± 0.283 | 43.113 ± 5.937 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 26.033 ± 0.299 | - | unsupported | 25.624 ± 0.094 | 26.547 ± 1.527 | 20.744 ± 4.158 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 9.058 ± 0.024 | - | 9.060 ± 0.026 | 9.467 ± 0.029 | 9.677 ± 0.061 | 9.506 ± 0.051 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 11.891 ± 0.046 | - | 19.781 ± 1.030 | 11.638 ± 0.056 | 14.665 ± 0.032 | 14.619 ± 0.071 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 55.693 ± 0.203 | - | 57.678 ± 0.180 | 95.826 ± 1.970 | 84.254 ± 1.025 | 63.779 ± 1.876 | 64.641 ± 7.495 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 25.705 ± 0.188 | - | 25.550 ± 0.037 | 37.100 ± 0.373 | 43.016 ± 0.994 | 64.104 ± 2.024 | 30.118 ± 9.969 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 91.116 ± 0.147 | - | 91.600 ± 1.233 | 125.386 ± 2.127 | 137.522 ± 1.047 | 116.342 ± 1.191 | 117.071 ± 3.851 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 38.719 ± 0.081 | - | 38.766 ± 0.242 | 52.130 ± 0.147 | 59.694 ± 0.985 | 116.722 ± 4.155 | 50.012 ± 1.414 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 63.237 ± 1.089 | - | 63.558 ± 3.853 | 63.440 ± 2.439 | 65.186 ± 3.221 | 69.641 ± 4.354 | 72.357 ± 3.062 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 20.731 ± 0.385 | - | 20.379 ± 0.290 | 21.292 ± 0.615 | 33.347 ± 1.835 | 68.830 ± 1.062 | 29.948 ± 1.398 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.317 ± 0.022 | - | 15.861 ± 0.334 | 18.171 ± 0.064 | 15.741 ± 0.171 | 3.466 ± 0.015 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 5.999 ± 0.004 | - | 8.370 ± 0.060 | 9.457 ± 0.238 | 2.662 ± 0.053 | 3.459 ± 0.010 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 7.114 ± 0.016 | - | 7.050 ± 0.041 | 7.919 ± 0.035 | 6.579 ± 0.023 | 8.048 ± 0.022 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 4.865 ± 0.034 | - | 12.668 ± 0.426 | 4.911 ± 0.022 | 6.341 ± 0.038 | 6.188 ± 0.148 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 4.650 ± 0.143 | - | 4.635 ± 0.123 | 4.798 ± 0.071 | 4.499 ± 0.014 | 4.248 ± 0.024 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.227 ± 0.143 | - | 13.570 ± 2.843 | 2.907 ± 0.038 | 3.388 ± 0.832 | 2.807 ± 0.032 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 5.198 ± 0.032 | - | 5.193 ± 0.017 | 6.259 ± 0.045 | 4.347 ± 0.023 | 4.319 ± 0.080 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 4.702 ± 0.112 | - | 15.070 ± 1.160 | 5.077 ± 0.025 | 5.427 ± 0.030 | 5.024 ± 0.031 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 40.931 ± 0.691 | - | 40.122 ± 0.043 | 39.876 ± 0.129 | 57.258 ± 0.085 | 41.401 ± 2.804 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 25.179 ± 0.140 | - | 24.974 ± 0.082 | 25.483 ± 0.383 | 25.471 ± 0.188 | 20.335 ± 0.064 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 42.619 ± 0.320 | - | unsupported | 43.550 ± 0.124 | 44.730 ± 0.206 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 21.755 ± 0.254 | - | unsupported | 22.476 ± 0.858 | 26.425 ± 0.139 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 109.676 ± 9.557 | - | 108.336 ± 3.446 | 106.742 ± 4.240 | 112.457 ± 5.672 | 123.388 ± 10.126 | 160.535 ± 5.715 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 37.375 ± 1.983 | - | 35.211 ± 0.335 | 35.273 ± 0.264 | 59.129 ± 2.842 | 114.032 ± 10.492 | 64.264 ± 1.137 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 121.507 ± 7.557 | - | 123.253 ± 5.796 | 119.053 ± 5.543 | 127.419 ± 4.722 | 128.832 ± 7.911 | 175.162 ± 0.379 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 40.254 ± 0.219 | - | 40.190 ± 0.319 | 41.113 ± 2.758 | 65.323 ± 1.166 | 135.061 ± 4.017 | 72.508 ± 2.592 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 13.676 ± 0.180 | - | 40.427 ± 1.058 | 57.105 ± 1.102 | 42.209 ± 0.355 | 69.698 ± 10.412 | 67.439 ± 8.988 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 7.521 ± 0.035 | - | 18.180 ± 0.312 | 24.084 ± 0.399 | 22.175 ± 0.801 | 62.474 ± 1.581 | 35.326 ± 14.435 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 7.946 ± 0.198 | - | 8.106 ± 0.193 | 31.199 ± 1.906 | 34.937 ± 1.391 | 40.379 ± 9.521 | 44.893 ± 11.006 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 6.960 ± 0.463 | - | 6.370 ± 0.226 | 10.877 ± 0.149 | 17.612 ± 0.134 | 39.076 ± 7.952 | 17.951 ± 19.620 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 16.575 ± 0.109 | - | 17.206 ± 0.412 | 41.646 ± 0.408 | 39.292 ± 0.309 | 18.915 ± 0.185 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 14.644 ± 0.138 | - | 14.716 ± 0.078 | 16.603 ± 0.069 | 22.469 ± 0.749 | 19.188 ± 0.252 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 13.415 ± 0.021 | - | 13.570 ± 0.032 | 49.600 ± 1.772 | 91.945 ± 1.182 | 53.102 ± 11.437 | 58.360 ± 10.080 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 5.879 ± 0.028 | - | 5.972 ± 0.015 | 18.781 ± 0.491 | 47.690 ± 0.294 | 57.108 ± 14.567 | 32.067 ± 9.005 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 123.844 ± 8.122 | - | 124.267 ± 7.214 | 116.898 ± 5.035 | 121.013 ± 6.473 | 137.226 ± 5.448 | 177.795 ± 7.594 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 39.889 ± 0.264 | - | 39.781 ± 0.159 | 39.731 ± 0.352 | 64.571 ± 1.775 | 131.822 ± 5.398 | 72.199 ± 1.778 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 8.006 ± 0.060 | - | 8.094 ± 0.032 | 39.354 ± 1.544 | 32.391 ± 1.372 | 50.542 ± 11.114 | 54.775 ± 10.716 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 3.829 ± 0.215 | - | 4.128 ± 1.056 | 14.219 ± 0.266 | 16.717 ± 0.444 | 56.007 ± 6.915 | 30.476 ± 8.707 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 10.932 ± 0.028 | - | 10.794 ± 0.034 | 21.768 ± 0.754 | 19.855 ± 0.206 | 30.266 ± 8.228 | 32.051 ± 9.229 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 4.616 ± 0.028 | - | 4.634 ± 0.018 | 7.412 ± 0.089 | 10.379 ± 0.376 | 29.663 ± 0.587 | 12.649 ± 9.450 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 13.857 ± 0.050 | - | 13.942 ± 0.017 | 43.136 ± 0.140 | 55.839 ± 0.424 | 61.912 ± 22.347 | 73.248 ± 5.823 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 6.859 ± 0.026 | - | 6.887 ± 0.028 | 15.414 ± 0.603 | 27.648 ± 0.598 | 62.334 ± 13.602 | 27.471 ± 13.250 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 6.856 ± 0.036 | - | 6.885 ± 0.016 | 19.050 ± 0.791 | 30.054 ± 0.170 | 35.428 ± 7.777 | 37.415 ± 8.563 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 3.401 ± 0.010 | - | 3.414 ± 0.010 | 6.920 ± 0.335 | 16.342 ± 0.442 | 35.217 ± 1.000 | 15.377 ± 9.468 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.293 ± 0.009 | 0.319 ± 0.004 | unsupported | unsupported | 0.421 ± 0.016 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.214 ± 0.006 | 0.294 ± 0.003 | unsupported | unsupported | 0.192 ± 0.002 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 117.436 ± 7.759 | - | 125.200 ± 11.232 | 121.135 ± 2.779 | 121.687 ± 1.055 | 139.240 ± 8.291 | 179.806 ± 1.495 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 39.551 ± 0.362 | - | 39.105 ± 0.255 | 40.465 ± 1.311 | 62.066 ± 2.762 | 131.743 ± 3.278 | 71.111 ± 1.474 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 124.344 ± 8.060 | - | 125.896 ± 9.895 | 120.760 ± 1.627 | 121.084 ± 0.293 | 140.341 ± 7.782 | 180.167 ± 5.373 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 39.626 ± 0.323 | - | 39.451 ± 0.398 | 40.757 ± 9.752 | 62.231 ± 5.517 | 134.143 ± 5.442 | 71.594 ± 0.793 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 124.393 ± 6.510 | - | 116.556 ± 0.630 | 117.124 ± 4.257 | 126.741 ± 2.003 | 136.868 ± 8.828 | 178.527 ± 4.338 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 40.170 ± 0.265 | - | 39.937 ± 0.128 | 40.019 ± 0.223 | 65.795 ± 0.519 | 131.734 ± 3.606 | 72.833 ± 2.187 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 100.154 ± 9.669 | - | 108.341 ± 3.198 | 106.995 ± 5.133 | 112.338 ± 2.683 | 121.247 ± 3.702 | 158.429 ± 7.054 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 37.397 ± 1.946 | - | 35.241 ± 0.372 | 37.069 ± 1.514 | 60.650 ± 2.338 | 118.644 ± 3.082 | 65.569 ± 1.568 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 43.530 ± 0.103 | - | 44.716 ± 0.217 | 34.572 ± 0.838 | 57.004 ± 0.337 | 108.611 ± 12.010 | 96.562 ± 11.247 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 20.001 ± 0.046 | - | 20.174 ± 0.392 | 12.449 ± 0.231 | 29.915 ± 0.076 | 108.478 ± 0.810 | 36.837 ± 9.508 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 11.511 ± 0.040 | - | 11.550 ± 0.064 | 7.781 ± 0.276 | 37.951 ± 0.016 | 7.630 ± 0.048 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 6.669 ± 0.310 | - | 6.641 ± 0.033 | 6.649 ± 0.033 | 18.907 ± 0.055 | 8.199 ± 0.099 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.490 ± 0.012 | - | 1.502 ± 0.037 | 21.964 ± 0.142 | 2.069 ± 0.027 | 0.902 ± 0.050 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.687 ± 0.014 | - | 0.704 ± 0.024 | 9.734 ± 0.562 | 1.451 ± 0.077 | 0.900 ± 0.040 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 1.601 ± 0.082 | - | 1.703 ± 0.028 | 4.637 ± 0.009 | 3.370 ± 0.012 | 0.891 ± 0.069 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 0.772 ± 0.167 | - | 0.861 ± 0.061 | 2.351 ± 0.025 | 1.804 ± 0.762 | 1.015 ± 0.330 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 11.085 ± 0.121 | - | 11.095 ± 0.026 | 7.520 ± 0.080 | 37.970 ± 0.035 | 7.877 ± 0.081 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 6.615 ± 0.034 | - | 6.739 ± 0.083 | 6.695 ± 0.308 | 18.952 ± 0.062 | 7.661 ± 0.126 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.430 ± 0.009 | - | 1.436 ± 0.010 | 22.043 ± 0.114 | 2.016 ± 0.016 | 0.898 ± 0.032 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.681 ± 0.017 | - | 0.700 ± 0.021 | 9.342 ± 0.318 | 1.121 ± 0.046 | 0.910 ± 0.030 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 8.420 ± 0.090 | - | 8.587 ± 0.340 | 13.883 ± 0.024 | 13.446 ± 0.021 | 7.437 ± 0.078 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 4.245 ± 0.217 | - | 4.326 ± 0.081 | 6.984 ± 0.043 | 6.832 ± 0.694 | 7.322 ± 0.169 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 7.580 ± 0.146 | - | 7.998 ± 0.461 | 7.554 ± 0.212 | 27.422 ± 0.062 | 7.376 ± 0.177 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 6.967 ± 0.209 | - | 7.084 ± 0.037 | 6.727 ± 0.162 | 14.008 ± 0.051 | 8.264 ± 0.175 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.639 ± 0.031 | - | 0.597 ± 0.058 | 1.838 ± 0.064 | 1.538 ± 0.020 | 0.600 ± 0.070 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.263 ± 0.015 | - | 0.279 ± 0.077 | 0.912 ± 0.061 | 1.536 ± 0.014 | 0.572 ± 0.070 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.173 ± 0.143 | - | 1.195 ± 0.217 | 0.543 ± 0.017 | 1.033 ± 0.019 | 0.827 ± 0.246 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.613 ± 0.037 | - | 0.643 ± 0.079 | 0.248 ± 0.013 | 0.609 ± 0.138 | 0.874 ± 0.070 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 7.526 ± 0.296 | - | 7.542 ± 0.511 | 7.576 ± 0.061 | 27.563 ± 0.067 | 7.312 ± 0.080 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 7.042 ± 0.067 | - | 7.062 ± 0.039 | 6.787 ± 0.389 | 13.837 ± 0.040 | 7.348 ± 0.065 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.643 ± 0.068 | - | 0.625 ± 0.081 | 1.866 ± 0.049 | 1.524 ± 0.006 | 0.581 ± 0.062 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.251 ± 0.025 | - | 0.263 ± 0.046 | 0.890 ± 0.040 | 1.522 ± 0.061 | 0.574 ± 0.053 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.163 ± 0.050 | - | 1.180 ± 0.317 | 0.558 ± 0.022 | 1.030 ± 0.028 | 0.738 ± 0.106 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.575 ± 0.026 | - | 0.618 ± 0.052 | 0.239 ± 0.024 | 0.943 ± 0.036 | 0.769 ± 0.205 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.645 ± 0.040 | 0.646 ± 0.015 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.238 ± 0.018 | 0.236 ± 0.007 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 32.940 ± 0.309 | - | 35.274 ± 0.225 | 51.003 ± 0.225 | 58.935 ± 1.064 | 57.217 ± 10.368 | 70.014 ± 7.799 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 14.747 ± 0.080 | - | 13.792 ± 0.064 | 17.763 ± 0.157 | 32.027 ± 0.204 | 57.927 ± 8.732 | 33.923 ± 8.824 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 122.117 ± 9.772 | - | 120.183 ± 7.923 | 121.057 ± 1.905 | 168.661 ± 6.907 | 230.331 ± 8.255 | 232.506 ± 8.030 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 37.240 ± 0.727 | - | 36.920 ± 3.130 | 37.896 ± 1.865 | 85.788 ± 2.516 | 229.221 ± 5.935 | 91.742 ± 10.693 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 123.153 ± 2.146 | - | 123.780 ± 7.903 | 134.717 ± 5.199 | 155.333 ± 1.691 | 139.835 ± 7.448 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 42.679 ± 1.654 | - | 41.869 ± 0.993 | 44.568 ± 0.808 | 70.085 ± 0.712 | 134.986 ± 2.514 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 112.561 ± 8.692 | - | 113.404 ± 8.815 | 111.603 ± 2.250 | 108.597 ± 0.822 | 129.261 ± 8.998 | 162.144 ± 8.101 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 37.300 ± 2.130 | - | 35.731 ± 1.328 | 36.495 ± 1.803 | 58.256 ± 5.244 | 121.073 ± 2.559 | 65.908 ± 4.169 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 10.801 ± 0.030 | - | 10.992 ± 0.035 | 51.422 ± 1.351 | 92.564 ± 0.352 | 57.328 ± 7.530 | 56.251 ± 11.020 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 4.990 ± 0.008 | - | 5.113 ± 0.025 | 19.367 ± 0.384 | 47.928 ± 0.813 | 55.627 ± 11.935 | 22.576 ± 12.809 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 111.567 ± 7.956 | - | 111.557 ± 8.021 | 155.010 ± 1.692 | 116.475 ± 6.443 | 185.058 ± 8.006 | 188.738 ± 6.877 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 36.486 ± 0.834 | - | 36.643 ± 1.525 | 55.342 ± 0.610 | 59.210 ± 2.301 | 177.995 ± 7.507 | 73.977 ± 5.690 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 115.267 ± 8.036 | - | 116.981 ± 0.708 | 119.063 ± 5.907 | 126.842 ± 3.736 | 133.146 ± 7.759 | 177.353 ± 5.716 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 40.638 ± 0.789 | - | 40.343 ± 0.428 | 41.561 ± 2.871 | 65.785 ± 1.519 | 135.092 ± 5.727 | 70.369 ± 1.355 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 27.064 ± 0.040 | - | 27.575 ± 0.040 | 67.759 ± 1.771 | 30.519 ± 1.675 | 62.356 ± 10.402 | 68.281 ± 17.675 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 11.822 ± 0.029 | - | 11.909 ± 0.100 | 24.350 ± 0.182 | 16.436 ± 0.193 | 64.449 ± 8.319 | 26.537 ± 14.576 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 116.560 ± 7.788 | 118.887 ± 8.078 | unsupported | 140.636 ± 6.413 | 151.320 ± 0.986 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 42.148 ± 2.032 | 41.765 ± 1.276 | unsupported | 44.510 ± 0.228 | 72.695 ± 5.285 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.702 ± 0.041 | - | 0.712 ± 0.094 | 0.611 ± 0.021 | 0.641 ± 0.027 | 6.097 ± 3.824 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.262 ± 0.021 | - | 0.275 ± 0.067 | 0.247 ± 0.023 | 0.634 ± 0.040 | 6.868 ± 4.895 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.640 ± 0.108 | - | 0.727 ± 0.113 | 0.579 ± 0.037 | 0.666 ± 0.128 | 7.068 ± 4.423 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.280 ± 0.071 | - | 0.280 ± 0.034 | 0.241 ± 0.034 | 0.577 ± 0.230 | 0.946 ± 5.468 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.071 ± 0.122 | - | unsupported | 0.973 ± 0.123 | 0.895 ± 0.049 | 6.690 ± 3.757 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.483 ± 0.101 | - | unsupported | 0.605 ± 0.090 | 0.768 ± 0.151 | 7.896 ± 5.035 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.274 ± 0.016 | - | 0.334 ± 0.027 | 0.661 ± 0.026 | 0.260 ± 0.037 | 1.122 ± 0.090 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.278 ± 0.065 | - | 0.322 ± 0.013 | 0.321 ± 0.011 | 0.176 ± 0.075 | 1.017 ± 0.106 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.706 ± 0.047 | - | 0.673 ± 0.111 | 0.888 ± 0.047 | 0.872 ± 0.044 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.262 ± 0.122 | - | 0.239 ± 0.028 | 0.414 ± 0.084 | 0.848 ± 0.006 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.681 ± 0.129 | - | 0.699 ± 0.062 | 0.728 ± 0.067 | 0.635 ± 0.086 | 0.972 ± 6.569 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.261 ± 0.048 | - | 0.237 ± 0.026 | 0.320 ± 0.110 | 0.583 ± 0.173 | 6.269 ± 6.635 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.631 ± 0.019 | - | 0.689 ± 0.006 | 0.307 ± 0.010 | 0.313 ± 0.024 | 0.301 ± 0.861 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.843 ± 0.012 | - | 0.872 ± 0.026 | 0.367 ± 0.012 | 0.303 ± 0.021 | 1.291 ± 0.286 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.304 ± 0.045 | - | 0.002 ± 0.000 | 1.287 ± 0.017 | 1.115 ± 0.231 | 8.703 ± 6.290 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.664 ± 0.018 | - | 0.003 ± 0.000 | 0.705 ± 0.185 | 0.907 ± 0.107 | 8.783 ± 0.921 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.042 ± 0.000 | - | 0.046 ± 0.001 | 0.058 ± 0.000 | 0.075 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.074 ± 0.000 | - | 0.079 ± 0.001 | 0.080 ± 0.000 | 0.126 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.187 ± 0.002 | - | 0.203 ± 0.011 | 0.232 ± 0.010 | 0.270 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.069 ± 0.001 | - | 0.072 ± 0.000 | 0.116 ± 0.002 | 0.104 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.001 | 0.034 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.529 ± 0.011 | - | 0.571 ± 0.011 | 0.937 ± 0.005 | 0.601 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.019 ± 0.000 | - | 0.024 ± 0.000 | 0.031 ± 0.000 | 0.053 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.029 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.043 ± 0.001 | - | 0.074 ± 0.003 | 0.070 ± 0.000 | 0.058 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.074 ± 0.002 | - | 0.130 ± 0.002 | 0.100 ± 0.000 | 0.119 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.200 ± 0.024 | - | 0.311 ± 0.010 | 0.246 ± 0.011 | 0.271 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.068 ± 0.001 | - | 0.110 ± 0.002 | 0.143 ± 0.001 | 0.104 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.005 ± 0.001 | - | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.013 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.007 ± 0.001 | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.003 ± 0.000 | - | 0.007 ± 0.001 | 0.007 ± 0.001 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.003 ± 0.000 | - | 0.007 ± 0.001 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.547 ± 0.011 | - | 0.753 ± 0.004 | 1.133 ± 0.016 | 0.981 ± 0.171 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.021 ± 0.000 | - | 0.038 ± 0.002 | 0.046 ± 0.001 | 0.044 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.414 ± 0.011 | - | 0.420 ± 0.010 | 0.526 ± 0.009 | 0.661 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.643 ± 0.012 | - | 1.665 ± 0.008 | 1.688 ± 0.014 | 2.119 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 4.991 ± 0.014 | - | 5.048 ± 0.034 | 5.158 ± 0.035 | 7.124 ± 0.086 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 2.999 ± 0.028 | - | 2.994 ± 0.010 | 2.674 ± 0.008 | 2.158 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.090 ± 0.009 | - | 0.094 ± 0.002 | 0.093 ± 0.003 | 0.117 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.001 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.016 ± 0.000 | - | 0.020 ± 0.001 | 0.022 ± 0.001 | 0.061 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.021 ± 0.000 | - | 0.025 ± 0.000 | 0.026 ± 0.000 | 0.070 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 23.900 ± 0.102 | - | 23.985 ± 0.092 | 21.288 ± 0.089 | 16.754 ± 0.089 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.689 ± 0.018 | - | 0.697 ± 0.010 | 0.696 ± 0.007 | 0.786 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.014 ± 0.000 | - | 0.017 ± 0.000 | 0.019 ± 0.000 | 0.028 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.041 ± 0.000 | - | 0.045 ± 0.001 | 0.047 ± 0.001 | 0.082 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.695 ± 0.013 | - | 0.720 ± 0.012 | 0.798 ± 0.275 | 0.699 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 2.506 ± 0.005 | - | 2.532 ± 0.016 | 1.675 ± 0.004 | 1.958 ± 0.168 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 7.489 ± 0.033 | - | 7.493 ± 0.016 | 7.389 ± 0.045 | 5.024 ± 0.125 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 3.109 ± 0.024 | - | 11.759 ± 0.097 | 1.939 ± 0.049 | 2.996 ± 0.098 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.090 ± 0.002 | - | 0.143 ± 0.003 | 0.139 ± 0.005 | 0.135 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.011 ± 0.000 | - | 0.015 ± 0.002 | 0.011 ± 0.001 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.011 ± 0.000 | - | 0.014 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.026 ± 0.000 | - | 0.030 ± 0.002 | 0.021 ± 0.000 | 0.052 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.007 ± 0.001 | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.014 ± 0.000 | - | 0.017 ± 0.001 | 0.012 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.033 ± 0.000 | - | 0.037 ± 0.001 | 0.026 ± 0.001 | 0.046 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 24.617 ± 0.172 | - | 33.852 ± 8.539 | 15.191 ± 0.119 | 22.984 ± 2.931 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.009 ± 0.000 | - | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.690 ± 0.013 | - | 1.071 ± 0.300 | 1.044 ± 0.004 | 1.901 ± 0.957 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.024 ± 0.001 | - | 0.027 ± 0.000 | 0.018 ± 0.000 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.062 ± 0.001 | - | 0.067 ± 0.002 | 0.046 ± 0.001 | 0.073 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.482 ± 0.031 | - | 0.482 ± 0.009 | 0.329 ± 0.009 | 0.293 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.699 ± 0.006 | - | 1.709 ± 0.005 | 1.473 ± 0.007 | 1.437 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 4.522 ± 0.034 | - | 4.521 ± 0.056 | 4.379 ± 0.026 | 5.014 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.090 ± 0.022 | - | 1.108 ± 0.013 | 1.081 ± 0.005 | 1.124 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.002 ± 0.001 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.056 ± 0.001 | - | 0.060 ± 0.000 | 0.065 ± 0.001 | 0.112 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.015 ± 0.000 | - | 0.018 ± 0.000 | 0.019 ± 0.001 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.019 ± 0.000 | - | 0.022 ± 0.000 | 0.023 ± 0.001 | 0.031 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 8.564 ± 0.079 | - | 8.621 ± 0.138 | 8.515 ± 0.044 | 8.821 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.445 ± 0.009 | - | 0.445 ± 0.006 | 0.480 ± 0.012 | 0.610 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.014 ± 0.000 | - | 0.018 ± 0.000 | 0.016 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.036 ± 0.000 | - | 0.040 ± 0.000 | 0.040 ± 0.000 | 0.077 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.764 ± 0.012 | - | 0.784 ± 0.064 | 0.534 ± 0.009 | 0.296 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 2.571 ± 0.010 | - | 2.599 ± 0.009 | 1.456 ± 0.007 | 0.951 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 6.673 ± 0.619 | - | 6.688 ± 0.014 | 6.219 ± 0.007 | 2.933 ± 0.456 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.050 ± 0.014 | - | 9.397 ± 0.569 | 1.004 ± 0.013 | 1.381 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.005 ± 0.001 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.092 ± 0.003 | - | 0.103 ± 0.011 | 0.099 ± 0.004 | 0.102 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.010 ± 0.000 | - | 0.014 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.011 ± 0.000 | - | 0.015 ± 0.000 | 0.013 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.023 ± 0.000 | - | 0.027 ± 0.002 | 0.018 ± 0.000 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.014 ± 0.000 | - | 0.018 ± 0.000 | 0.016 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.030 ± 0.000 | - | 0.033 ± 0.002 | 0.023 ± 0.000 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 8.274 ± 0.111 | - | 17.240 ± 0.440 | 7.848 ± 0.048 | 10.990 ± 4.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.009 ± 0.000 | - | 0.013 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.701 ± 0.017 | - | 0.711 ± 0.009 | 0.721 ± 0.009 | 0.348 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.023 ± 0.000 | - | 0.028 ± 0.000 | 0.023 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.056 ± 0.000 | - | 0.059 ± 0.002 | 0.040 ± 0.000 | 0.055 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.187 ± 0.003 | - | 0.194 ± 0.001 | 0.168 ± 0.008 | 0.290 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 0.836 ± 0.012 | - | 0.843 ± 0.003 | 0.721 ± 0.010 | 1.445 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 2.414 ± 0.003 | - | 2.434 ± 0.008 | 2.276 ± 0.010 | 4.970 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.428 ± 0.014 | - | 0.442 ± 0.018 | 0.445 ± 0.008 | 1.141 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.002 ± 0.002 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.025 ± 0.000 | - | 0.028 ± 0.000 | 0.031 ± 0.001 | 0.108 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.001 | 0.007 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.014 ± 0.001 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.011 ± 0.000 | - | 0.014 ± 0.000 | 0.015 ± 0.001 | 0.027 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 3.414 ± 0.064 | - | 3.416 ± 0.012 | 3.527 ± 0.011 | 8.632 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.186 ± 0.001 | - | 0.191 ± 0.002 | 0.196 ± 0.009 | 0.606 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.020 ± 0.000 | - | 0.023 ± 0.000 | 0.024 ± 0.000 | 0.061 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.329 ± 0.011 | - | 0.332 ± 0.011 | 0.283 ± 0.010 | 0.291 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 1.161 ± 0.013 | - | 1.169 ± 0.008 | 0.723 ± 0.021 | 0.947 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 3.118 ± 0.005 | - | 3.153 ± 0.007 | 2.755 ± 0.010 | 2.921 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.568 ± 0.017 | - | 2.996 ± 0.982 | 0.589 ± 0.010 | 1.305 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.004 ± 0.002 | - | 0.007 ± 0.000 | 0.009 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.033 ± 0.001 | - | 0.036 ± 0.001 | 0.039 ± 0.001 | 0.097 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.006 ± 0.000 | - | 0.010 ± 0.001 | 0.011 ± 0.003 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.012 ± 0.000 | - | 0.016 ± 0.001 | 0.014 ± 0.001 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.003 | 0.012 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.015 ± 0.000 | - | 0.019 ± 0.001 | 0.015 ± 0.000 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 4.442 ± 0.018 | - | 14.986 ± 2.246 | 4.476 ± 0.012 | 10.998 ± 0.999 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.232 ± 0.009 | - | 0.239 ± 0.001 | 0.244 ± 0.010 | 0.332 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.012 ± 0.000 | - | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.028 ± 0.000 | - | 0.031 ± 0.002 | 0.024 ± 0.000 | 0.054 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.143 ± 0.002 | - | 0.145 ± 0.003 | 0.146 ± 0.003 | 0.102 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.181 ± 0.002 | - | 0.185 ± 0.002 | 0.203 ± 0.004 | 0.175 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.414 ± 0.010 | - | 0.424 ± 0.009 | 0.423 ± 0.019 | 0.460 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.105 ± 0.002 | - | 0.114 ± 0.003 | 0.136 ± 0.004 | 0.180 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.015 ± 0.000 | 0.045 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.011 ± 0.000 | - | 0.014 ± 0.000 | 0.017 ± 0.000 | 0.047 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.036 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.038 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.833 ± 0.008 | - | 0.861 ± 0.008 | 1.029 ± 0.012 | 1.016 ± 0.062 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.057 ± 0.001 | - | 0.064 ± 0.003 | 0.062 ± 0.005 | 0.097 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.035 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.043 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.173 ± 0.010 | - | 0.177 ± 0.011 | 0.123 ± 0.001 | 0.083 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.255 ± 0.010 | - | 0.270 ± 0.011 | 0.165 ± 0.006 | 0.172 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.620 ± 0.011 | - | 0.629 ± 0.012 | 0.280 ± 0.012 | 0.485 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.154 ± 0.005 | - | 0.165 ± 0.002 | 0.167 ± 0.005 | 0.167 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.015 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.016 ± 0.001 | - | 0.019 ± 0.000 | 0.025 ± 0.001 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.015 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.019 ± 0.000 | - | 0.022 ± 0.001 | 0.027 ± 0.001 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.015 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.002 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.002 | 0.012 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.017 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.007 ± 0.000 | - | 0.010 ± 0.001 | 0.019 ± 0.001 | 0.017 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 1.189 ± 0.007 | - | 1.203 ± 0.007 | 0.497 ± 0.027 | 0.735 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.001 | 0.018 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.091 ± 0.004 | - | 0.095 ± 0.002 | 0.052 ± 0.000 | 0.121 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.019 ± 0.000 | 0.012 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.009 ± 0.000 | - | 0.013 ± 0.000 | 0.021 ± 0.000 | 0.021 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.201 ± 0.010 | - | 0.208 ± 0.014 | 0.169 ± 0.001 | 0.243 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.429 ± 0.010 | - | 0.436 ± 0.009 | 0.410 ± 0.016 | 0.642 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.018 ± 0.008 | - | 1.044 ± 0.022 | 1.041 ± 0.001 | 1.657 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.399 ± 0.006 | - | 0.415 ± 0.008 | 0.446 ± 0.013 | 0.500 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.013 ± 0.000 | - | 0.016 ± 0.000 | 0.019 ± 0.000 | 0.040 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.015 ± 0.000 | - | 0.018 ± 0.001 | 0.022 ± 0.000 | 0.041 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 3.071 ± 0.028 | - | 3.863 ± 0.025 | 3.478 ± 0.028 | 3.772 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.094 ± 0.000 | - | 0.099 ± 0.000 | 0.127 ± 0.001 | 0.152 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.207 ± 0.017 | - | 0.293 ± 0.013 | 0.246 ± 0.009 | 0.218 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.434 ± 0.011 | - | 0.656 ± 0.012 | 0.587 ± 0.011 | 0.597 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.019 ± 0.011 | - | 1.545 ± 0.012 | 1.548 ± 0.013 | 0.992 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.462 ± 0.012 | - | 2.997 ± 0.013 | 0.410 ± 0.012 | 0.974 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.015 ± 0.000 | - | 0.027 ± 0.002 | 0.032 ± 0.000 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.025 ± 0.001 | - | 0.029 ± 0.001 | 0.034 ± 0.000 | 0.032 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.009 ± 0.001 | 0.008 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.009 ± 0.001 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.006 ± 0.000 | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.010 ± 0.001 | 0.009 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.007 ± 0.004 | - | 0.013 ± 0.000 | 0.013 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 3.581 ± 0.029 | - | 14.966 ± 2.345 | 3.193 ± 0.056 | 8.301 ± 1.345 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.095 ± 0.001 | - | 0.162 ± 0.010 | 0.212 ± 0.001 | 0.151 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.006 ± 0.000 | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.011 ± 0.000 | - | 0.019 ± 0.001 | 0.019 ± 0.000 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.139 ± 0.002 | - | 0.126 ± 0.002 | 0.074 ± 0.001 | 0.183 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.188 ± 0.002 | - | 0.130 ± 0.002 | 0.116 ± 0.001 | 0.271 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.438 ± 0.014 | - | 0.380 ± 0.012 | 0.336 ± 0.008 | 0.606 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.092 ± 0.001 | - | 0.102 ± 0.004 | 0.112 ± 0.003 | 0.154 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.008 ± 0.001 | - | 0.013 ± 0.000 | 0.018 ± 0.001 | 0.042 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.014 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.725 ± 0.009 | - | 0.759 ± 0.012 | 0.824 ± 0.007 | 0.859 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.053 ± 0.001 | - | 0.059 ± 0.001 | 0.057 ± 0.001 | 0.095 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.015 ± 0.001 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.139 ± 0.001 | - | 0.146 ± 0.002 | 0.097 ± 0.001 | 0.183 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.186 ± 0.001 | - | 0.200 ± 0.013 | 0.133 ± 0.001 | 0.266 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.420 ± 0.009 | - | 0.562 ± 0.014 | 0.305 ± 0.013 | 0.653 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.095 ± 0.004 | - | 0.154 ± 0.005 | 0.146 ± 0.006 | 0.125 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.012 ± 0.001 | 0.017 ± 0.000 | 0.008 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.010 ± 0.001 | - | 0.022 ± 0.000 | 0.026 ± 0.002 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.012 ± 0.000 | 0.017 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.012 ± 0.000 | 0.017 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.012 ± 0.000 | 0.017 ± 0.002 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.012 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.013 ± 0.000 | 0.019 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.012 ± 0.000 | 0.017 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.012 ± 0.000 | 0.017 ± 0.001 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.014 ± 0.000 | 0.019 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.732 ± 0.010 | - | 1.089 ± 0.009 | 0.446 ± 0.010 | 0.593 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.013 ± 0.000 | 0.019 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.055 ± 0.003 | - | 0.094 ± 0.004 | 0.047 ± 0.003 | 0.091 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.013 ± 0.000 | 0.020 ± 0.003 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.016 ± 0.000 | 0.022 ± 0.001 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.396 ± 0.017 | - | 1.397 ± 0.009 | 1.180 ± 0.007 | 1.581 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 4.548 ± 0.018 | - | 4.562 ± 0.017 | 3.970 ± 0.014 | 4.033 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 13.609 ± 0.029 | - | 13.666 ± 0.076 | 11.972 ± 0.017 | 12.000 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 4.059 ± 0.041 | - | 4.087 ± 0.027 | 4.649 ± 0.022 | 3.428 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.157 ± 0.003 | - | 0.164 ± 0.018 | 0.180 ± 0.008 | 0.174 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.140 ± 0.005 | - | 0.150 ± 0.006 | 0.161 ± 0.007 | 0.166 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.016 ± 0.000 | - | 0.020 ± 0.000 | 0.020 ± 0.001 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.015 ± 0.000 | - | 0.019 ± 0.000 | 0.018 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.043 ± 0.001 | - | 0.046 ± 0.001 | 0.044 ± 0.000 | 0.048 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.020 ± 0.000 | - | 0.023 ± 0.000 | 0.022 ± 0.001 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.057 ± 0.000 | - | 0.061 ± 0.001 | 0.057 ± 0.002 | 0.060 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 32.535 ± 0.075 | - | 33.676 ± 0.227 | 36.963 ± 0.165 | 27.053 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.013 ± 0.000 | - | 0.016 ± 0.000 | 0.016 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.228 ± 0.003 | - | 1.238 ± 0.008 | 1.339 ± 0.012 | 1.202 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.038 ± 0.002 | - | 0.042 ± 0.000 | 0.040 ± 0.001 | 0.044 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.110 ± 0.001 | - | 0.115 ± 0.001 | 0.103 ± 0.001 | 0.119 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 2.152 ± 0.014 | - | 2.171 ± 0.015 | 1.179 ± 0.009 | 1.565 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 6.894 ± 0.024 | - | 6.997 ± 0.014 | 3.967 ± 0.051 | 2.003 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 13.504 ± 0.012 | - | 16.449 ± 0.010 | 14.402 ± 0.048 | 6.914 ± 0.672 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 4.076 ± 0.089 | - | 12.545 ± 0.058 | 4.056 ± 0.012 | 4.586 ± 0.397 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.007 ± 0.000 | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.250 ± 0.013 | - | 0.258 ± 0.012 | 0.268 ± 0.012 | 0.165 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.012 ± 0.000 | - | 0.016 ± 0.001 | 0.011 ± 0.001 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.216 ± 0.010 | - | 0.226 ± 0.014 | 0.234 ± 0.011 | 0.155 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.026 ± 0.000 | - | 0.029 ± 0.000 | 0.020 ± 0.003 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.011 ± 0.001 | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.025 ± 0.000 | - | 0.028 ± 0.002 | 0.018 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.065 ± 0.001 | - | 0.069 ± 0.002 | 0.043 ± 0.000 | 0.048 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.013 ± 0.001 | - | 0.017 ± 0.000 | 0.015 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.032 ± 0.000 | - | 0.036 ± 0.002 | 0.022 ± 0.001 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.086 ± 0.001 | - | 0.090 ± 0.004 | 0.056 ± 0.001 | 0.061 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 32.402 ± 0.222 | - | 48.919 ± 9.135 | 31.558 ± 0.368 | 50.761 ± 25.983 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.022 ± 0.001 | - | 0.025 ± 0.002 | 0.022 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 1.868 ± 0.013 | - | 1.895 ± 0.015 | 1.996 ± 0.018 | 0.668 ± 0.086 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.059 ± 0.000 | - | 0.063 ± 0.002 | 0.038 ± 0.001 | 0.044 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.111 ± 0.001 | - | 0.166 ± 0.002 | 0.103 ± 0.004 | 0.111 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 1.047 ± 0.003 | - | 1.048 ± 0.002 | 0.830 ± 0.005 | 0.986 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 3.879 ± 0.033 | - | 3.889 ± 0.007 | 3.352 ± 0.015 | 3.099 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 11.586 ± 0.095 | - | 11.593 ± 0.056 | 10.451 ± 0.039 | 9.566 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.890 ± 0.013 | - | 1.918 ± 0.038 | 1.976 ± 0.014 | 1.853 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.108 ± 0.005 | - | 0.110 ± 0.003 | 0.125 ± 0.006 | 0.164 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.001 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.087 ± 0.003 | - | 0.090 ± 0.004 | 0.099 ± 0.005 | 0.140 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.014 ± 0.000 | - | 0.017 ± 0.000 | 0.018 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.001 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.013 ± 0.000 | - | 0.017 ± 0.000 | 0.017 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.034 ± 0.001 | - | 0.038 ± 0.000 | 0.037 ± 0.000 | 0.039 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.017 ± 0.000 | - | 0.021 ± 0.000 | 0.020 ± 0.006 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.045 ± 0.000 | - | 0.049 ± 0.000 | 0.047 ± 0.000 | 0.049 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 14.968 ± 0.071 | - | 15.033 ± 0.027 | 15.679 ± 0.199 | 14.419 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.013 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 0.829 ± 0.011 | - | 0.837 ± 0.008 | 0.916 ± 0.006 | 0.947 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.033 ± 0.000 | - | 0.036 ± 0.000 | 0.034 ± 0.002 | 0.036 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.089 ± 0.000 | - | 0.093 ± 0.000 | 0.086 ± 0.001 | 0.117 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 1.580 ± 0.036 | - | 1.582 ± 0.016 | 0.792 ± 0.391 | 1.004 ± 0.331 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 5.855 ± 0.006 | - | 5.895 ± 0.007 | 3.297 ± 0.018 | 1.553 ± 0.258 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 11.544 ± 0.014 | - | 14.430 ± 0.023 | 12.929 ± 0.060 | 4.889 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 2.227 ± 0.014 | - | 5.980 ± 4.867 | 2.375 ± 0.009 | 2.515 ± 0.521 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.164 ± 0.006 | - | 0.177 ± 0.018 | 0.190 ± 0.006 | 0.130 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.010 ± 0.000 | - | 0.014 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.131 ± 0.005 | - | 0.141 ± 0.009 | 0.147 ± 0.008 | 0.145 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.022 ± 0.001 | - | 0.025 ± 0.002 | 0.017 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.022 ± 0.000 | - | 0.025 ± 0.002 | 0.016 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.052 ± 0.001 | - | 0.055 ± 0.001 | 0.036 ± 0.001 | 0.041 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.011 ± 0.001 | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.028 ± 0.000 | - | 0.031 ± 0.002 | 0.020 ± 0.001 | 0.025 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.068 ± 0.001 | - | 0.073 ± 0.002 | 0.046 ± 0.001 | 0.049 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 17.520 ± 0.087 | - | 26.458 ± 0.362 | 18.610 ± 0.069 | 22.030 ± 3.639 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.017 ± 0.000 | - | 0.020 ± 0.000 | 0.018 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 1.264 ± 0.010 | - | 1.288 ± 0.007 | 1.409 ± 0.011 | 0.440 ± 0.075 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.051 ± 0.002 | - | 0.054 ± 0.002 | 0.033 ± 0.000 | 0.037 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.132 ± 0.001 | - | 0.136 ± 0.001 | 0.086 ± 0.002 | 0.105 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.366 ± 0.006 | - | unsupported | 0.271 ± 0.020 | 0.528 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.682 ± 0.037 | - | unsupported | 1.440 ± 0.015 | 2.002 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 4.153 ± 0.035 | - | unsupported | 3.966 ± 0.014 | 5.179 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.809 ± 0.022 | - | unsupported | 0.923 ± 0.004 | 0.927 ± 0.068 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.002 ± 0.001 | - | unsupported | 0.006 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.039 ± 0.001 | - | unsupported | 0.048 ± 0.002 | 0.054 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.031 ± 0.000 | - | unsupported | 0.041 ± 0.005 | 0.047 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.007 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | unsupported | 0.010 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.013 ± 0.000 | - | unsupported | 0.018 ± 0.001 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.007 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.017 ± 0.000 | - | unsupported | 0.022 ± 0.000 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 6.366 ± 0.033 | - | unsupported | 7.250 ± 0.054 | 7.314 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.299 ± 0.008 | - | unsupported | 0.332 ± 0.007 | 0.397 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.014 ± 0.000 | - | unsupported | 0.017 ± 0.000 | 0.024 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.033 ± 0.000 | - | unsupported | 0.037 ± 0.001 | 0.049 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.560 ± 0.010 | - | unsupported | 0.395 ± 0.012 | 0.530 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 2.636 ± 0.015 | - | unsupported | 1.432 ± 0.008 | 1.028 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 6.117 ± 0.059 | - | unsupported | 5.615 ± 0.019 | 2.676 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.032 ± 0.012 | - | unsupported | 1.187 ± 0.012 | 1.371 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.004 ± 0.002 | - | unsupported | 0.008 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.053 ± 0.003 | - | unsupported | 0.064 ± 0.002 | 0.055 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | unsupported | 0.010 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.045 ± 0.001 | - | unsupported | 0.055 ± 0.003 | 0.060 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.009 ± 0.000 | - | unsupported | 0.010 ± 0.001 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.005 ± 0.000 | - | unsupported | 0.010 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.011 ± 0.000 | - | unsupported | 0.014 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.021 ± 0.000 | - | unsupported | 0.018 ± 0.001 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.006 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.013 ± 0.000 | - | unsupported | 0.016 ± 0.002 | 0.017 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.027 ± 0.000 | - | unsupported | 0.022 ± 0.000 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 8.147 ± 0.197 | - | unsupported | 9.216 ± 0.356 | 11.184 ± 0.236 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.008 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.398 ± 0.011 | - | unsupported | 0.332 ± 0.017 | 0.236 ± 0.064 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.023 ± 0.000 | - | unsupported | 0.024 ± 0.002 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.050 ± 0.000 | - | unsupported | 0.038 ± 0.001 | 0.048 ± 0.003 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.834 ± 0.016 | - | unsupported | 0.779 ± 0.004 | 2.775 ± 0.147 | 2.204 ± 0.059 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.103 ± 0.001 | - | unsupported | 0.049 ± 0.031 | 0.194 ± 0.135 | 0.112 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.135 ± 0.002 | - | unsupported | 0.071 ± 0.003 | 0.135 ± 0.003 | 0.180 ± 0.010 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.277 ± 0.008 | - | unsupported | 0.201 ± 0.005 | 0.372 ± 0.007 | 0.636 ± 0.022 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.844 ± 0.016 | - | unsupported | 0.393 ± 0.012 | 0.810 ± 0.017 | 2.112 ± 0.928 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.101 ± 0.001 | - | unsupported | 0.048 ± 0.004 | 0.085 ± 0.006 | 0.088 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.138 ± 0.001 | - | unsupported | 0.058 ± 0.001 | 0.141 ± 0.005 | 0.147 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.280 ± 0.010 | - | unsupported | 0.118 ± 0.010 | 0.422 ± 0.042 | 0.611 ± 0.023 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.245 ± 0.004 | - | unsupported | 0.300 ± 0.010 | 0.634 ± 0.022 | 0.350 ± 0.008 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.027 ± 0.000 | - | unsupported | 0.033 ± 0.001 | 0.130 ± 0.003 | 0.067 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.050 ± 0.000 | - | unsupported | 0.050 ± 0.000 | 0.173 ± 0.016 | 0.101 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.163 ± 0.001 | - | unsupported | 0.138 ± 0.005 | 0.283 ± 0.014 | 0.177 ± 0.007 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.248 ± 0.010 | - | unsupported | 0.431 ± 0.011 | 0.749 ± 0.076 | 0.297 ± 0.008 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.029 ± 0.000 | - | unsupported | 0.056 ± 0.002 | 0.127 ± 0.003 | 0.062 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.051 ± 0.000 | - | unsupported | 0.077 ± 0.003 | 0.155 ± 0.014 | 0.090 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.165 ± 0.009 | - | unsupported | 0.181 ± 0.001 | 0.263 ± 0.015 | 0.153 ± 0.003 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.161 ± 0.010 | - | 0.223 ± 0.012 | 0.160 ± 0.002 | 0.321 ± 0.025 | 0.222 ± 0.009 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.089 ± 0.035 | - | 0.066 ± 0.003 | 0.056 ± 0.001 | 0.055 ± 0.001 | 0.050 ± 0.012 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.072 ± 0.001 | - | 0.080 ± 0.002 | 0.067 ± 0.001 | 0.070 ± 0.002 | 0.077 ± 0.005 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.098 ± 0.001 | - | 0.119 ± 0.001 | 0.096 ± 0.002 | 0.126 ± 0.003 | 0.130 ± 0.012 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.157 ± 0.004 | - | 0.275 ± 0.014 | 0.207 ± 0.010 | 0.322 ± 0.012 | 0.171 ± 0.003 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.064 ± 0.000 | - | 0.122 ± 0.001 | 0.095 ± 0.003 | 0.058 ± 0.002 | 0.046 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.074 ± 0.000 | - | 0.151 ± 0.002 | 0.109 ± 0.009 | 0.071 ± 0.002 | 0.065 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.100 ± 0.001 | - | 0.191 ± 0.003 | 0.141 ± 0.002 | 0.154 ± 0.002 | 0.101 ± 0.001 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 36.816 ± 0.368 | - | 39.843 ± 11.902 | 53.925 ± 0.343 | 58.215 ± 0.215 | 49.533 ± 4.748 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 18.464 ± 0.201 | - | 25.014 ± 0.155 | 19.999 ± 0.249 | 42.269 ± 15.055 | 32.343 ± 5.294 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 21.603 ± 0.467 | - | 20.452 ± 0.450 | 22.720 ± 0.082 | 19.952 ± 0.042 | 18.054 ± 0.551 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 10.975 ± 0.409 | - | 30.006 ± 3.070 | 9.635 ± 0.132 | 12.198 ± 0.867 | 10.730 ± 0.542 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 6.161 ± 0.038 | - | 6.177 ± 0.038 | 6.462 ± 0.110 | 6.061 ± 0.041 | 5.916 ± 0.258 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 9.559 ± 0.023 | - | 18.152 ± 0.296 | 9.130 ± 0.084 | 8.962 ± 0.133 | 8.506 ± 0.231 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 7.100 ± 0.072 | - | 7.120 ± 0.013 | 7.121 ± 0.015 | 7.199 ± 0.023 | 7.000 ± 0.077 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 10.721 ± 0.060 | - | 18.804 ± 0.401 | 9.742 ± 0.025 | 10.908 ± 0.683 | 10.411 ± 0.020 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 17.100 ± 0.047 | - | 17.290 ± 0.111 | 15.538 ± 0.035 | 31.620 ± 0.201 | 11.097 ± 0.405 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 12.099 ± 0.072 | - | 23.442 ± 2.843 | 12.553 ± 0.391 | 25.708 ± 0.105 | 9.229 ± 0.110 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.409 ± 0.004 | 0.418 ± 0.005 | unsupported | 0.497 ± 0.003 | 0.610 ± 0.020 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 6.482 ± 0.067 | 6.553 ± 0.029 | unsupported | 7.089 ± 0.133 | 7.773 ± 0.061 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.321 ± 0.001 | 3.936 ± 2.624 | unsupported | 0.416 ± 0.003 | 1.408 ± 0.024 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 5.377 ± 0.095 | 13.538 ± 0.565 | unsupported | 4.591 ± 0.022 | 8.857 ± 0.057 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 34.811 ± 0.446 | - | 34.729 ± 0.349 | 36.369 ± 0.068 | 34.484 ± 0.109 | 24.010 ± 0.205 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 27.419 ± 8.501 | - | 31.534 ± 4.367 | 18.662 ± 0.371 | 20.148 ± 0.610 | 13.773 ± 2.724 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 14.832 ± 0.026 | 14.416 ± 0.061 | 7.478 ± 0.022 | 35.759 ± 0.106 | 6.120 ± 0.105 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 31.939 ± 2.680 | 18.107 ± 0.417 | 5.145 ± 0.085 | 34.987 ± 0.997 | 4.733 ± 0.142 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 21.064 ± 0.237 | - | 21.537 ± 0.045 | 27.935 ± 0.175 | 26.771 ± 0.197 | 26.301 ± 14.914 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 11.035 ± 0.542 | - | 22.083 ± 0.832 | 13.910 ± 0.528 | 19.312 ± 0.330 | 22.648 ± 9.337 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.737 ± 0.072 | - | 0.588 ± 0.027 | 0.939 ± 0.034 | 4.769 ± 0.056 | 2.308 ± 0.011 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 5.999 ± 0.007 | - | 0.308 ± 0.078 | 0.940 ± 0.056 | 11.974 ± 2.019 | 2.299 ± 0.013 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 16.674 ± 0.158 | - | 16.594 ± 0.157 | 17.154 ± 0.063 | 16.049 ± 0.038 | 17.622 ± 2.862 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 21.076 ± 2.846 | - | 24.724 ± 15.052 | 13.886 ± 0.217 | 17.100 ± 1.168 | 15.965 ± 1.394 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 16.821 ± 0.221 | - | 16.641 ± 0.145 | 17.204 ± 0.052 | 16.228 ± 0.091 | 15.801 ± 0.214 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 21.001 ± 1.015 | - | 42.014 ± 3.965 | 14.242 ± 1.134 | 17.435 ± 0.747 | 15.770 ± 0.262 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 0.962 ± 0.007 | 0.983 ± 0.008 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 17.493 ± 0.114 | 17.621 ± 0.079 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.160 ± 0.018 | 10.891 ± 1.416 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 16.766 ± 0.069 | 20.413 ± 7.104 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 0.899 ± 0.010 | 0.918 ± 0.011 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 16.628 ± 0.085 | 16.749 ± 0.119 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.064 ± 0.014 | 11.125 ± 0.697 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 15.781 ± 0.203 | 19.061 ± 0.311 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 21.923 ± 0.579 | - | 20.339 ± 0.051 | 22.788 ± 0.085 | 20.004 ± 0.043 | 17.994 ± 3.425 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 10.984 ± 0.285 | - | 29.968 ± 10.671 | 9.691 ± 0.160 | 12.324 ± 0.124 | 10.694 ± 0.412 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 48.640 ± 0.373 | 48.798 ± 1.355 | 49.982 ± 0.277 | 49.578 ± 0.112 | 48.657 ± 0.964 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 41.481 ± 0.216 | 39.486 ± 0.495 | 32.848 ± 0.402 | 43.017 ± 0.234 | 44.620 ± 4.445 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 30.116 ± 0.137 | - | 94.755 ± 1.034 | 25.568 ± 0.098 | 126.897 ± 3.389 | 26.053 ± 0.308 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 23.239 ± 0.163 | - | 58.682 ± 21.676 | 21.500 ± 1.913 | 122.148 ± 7.005 | 17.267 ± 0.378 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.080 ± 0.000 | 0.079 ± 0.000 | unsupported | 0.133 ± 0.025 | 0.115 ± 0.007 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 1.257 ± 0.003 | 1.254 ± 0.002 | unsupported | 2.070 ± 0.017 | 0.530 ± 0.572 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.040 ± 0.000 | 0.041 ± 0.000 | unsupported | 0.064 ± 0.003 | 0.035 ± 0.005 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.567 ± 0.003 | 0.568 ± 0.002 | unsupported | 0.880 ± 0.021 | 0.274 ± 0.304 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.038 ± 0.000 | 0.038 ± 0.000 | unsupported | 0.085 ± 0.007 | 0.188 ± 0.011 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 0.589 ± 0.000 | 0.616 ± 0.005 | unsupported | 1.252 ± 0.007 | 2.052 ± 0.058 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.020 ± 0.000 | 0.021 ± 0.000 | unsupported | 0.068 ± 0.024 | 0.116 ± 0.003 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.282 ± 0.001 | 0.287 ± 0.006 | unsupported | 0.480 ± 0.008 | 0.866 ± 0.049 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.105 ± 0.000 | 0.105 ± 0.001 | unsupported | 0.111 ± 0.000 | 0.148 ± 0.005 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 1.671 ± 0.002 | 1.682 ± 0.005 | unsupported | 2.571 ± 0.024 | 1.546 ± 0.103 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.050 ± 0.000 | 0.051 ± 0.001 | unsupported | 0.072 ± 0.003 | 0.108 ± 0.005 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 0.763 ± 0.001 | 0.763 ± 0.021 | unsupported | 1.003 ± 0.023 | 1.084 ± 0.121 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.044 ± 0.008 | 0.069 ± 0.009 | unsupported | 0.130 ± 0.001 | 0.154 ± 0.007 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.587 ± 0.012 | 0.623 ± 0.023 | unsupported | 0.690 ± 0.006 | 1.655 ± 0.114 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.078 ± 0.005 | 0.125 ± 0.013 | unsupported | 0.054 ± 0.001 | 0.142 ± 0.001 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.367 ± 0.009 | 0.418 ± 0.028 | unsupported | 0.295 ± 0.004 | 1.002 ± 0.052 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.066 ± 0.008 | 0.081 ± 0.003 | unsupported | 0.090 ± 0.001 | 0.212 ± 0.014 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 0.821 ± 0.022 | 0.866 ± 0.015 | unsupported | 0.726 ± 0.006 | 2.438 ± 0.071 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.096 ± 0.010 | 0.126 ± 0.004 | unsupported | 0.044 ± 0.001 | 0.187 ± 0.001 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.502 ± 0.034 | 0.550 ± 0.021 | unsupported | 0.335 ± 0.011 | 1.398 ± 0.056 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.220 ± 0.004 | 0.246 ± 0.001 | unsupported | unsupported | 0.215 ± 0.003 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 3.225 ± 0.003 | 3.296 ± 0.043 | unsupported | unsupported | 3.207 ± 0.044 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.179 ± 0.007 | 0.214 ± 0.032 | unsupported | unsupported | 0.223 ± 0.005 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 1.398 ± 0.015 | 1.476 ± 0.142 | unsupported | unsupported | 1.785 ± 0.134 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.009 ± 0.000 | 0.015 ± 0.000 | unsupported | 0.010 ± 0.000 | 0.041 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.092 ± 0.004 | 0.098 ± 0.002 | unsupported | 0.061 ± 0.000 | 0.256 ± 0.004 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.012 ± 0.000 | 0.020 ± 0.000 | unsupported | 0.011 ± 0.000 | 0.034 ± 0.000 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.059 ± 0.001 | 0.064 ± 0.002 | unsupported | 0.039 ± 0.002 | 0.224 ± 0.004 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.025 ± 0.008 | 0.038 ± 0.005 | unsupported | 0.086 ± 0.004 | 0.152 ± 0.022 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.321 ± 0.017 | 0.347 ± 0.038 | unsupported | 0.483 ± 0.014 | 1.495 ± 0.045 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.037 ± 0.008 | 0.056 ± 0.006 | unsupported | 0.087 ± 0.001 | 0.116 ± 0.006 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.195 ± 0.010 | 0.230 ± 0.018 | unsupported | 0.220 ± 0.004 | 1.210 ± 0.045 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.026 ± 0.000 | 0.026 ± 0.000 | unsupported | 0.076 ± 0.004 | 0.139 ± 0.007 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.400 ± 0.001 | 0.407 ± 0.003 | unsupported | 1.261 ± 0.021 | 1.400 ± 0.083 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.015 ± 0.000 | 0.015 ± 0.000 | unsupported | 0.059 ± 0.002 | 0.096 ± 0.007 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.193 ± 0.002 | 0.202 ± 0.003 | unsupported | 0.465 ± 0.015 | 0.732 ± 0.019 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.027 ± 0.000 | 0.028 ± 0.000 | unsupported | 0.085 ± 0.004 | 0.142 ± 0.003 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.416 ± 0.002 | 0.426 ± 0.002 | unsupported | 1.264 ± 0.010 | 1.435 ± 0.028 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.015 ± 0.000 | 0.016 ± 0.001 | unsupported | 0.060 ± 0.000 | 0.101 ± 0.004 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.201 ± 0.001 | 0.211 ± 0.008 | unsupported | 0.482 ± 0.058 | 0.757 ± 0.012 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.064 ± 0.001 | 0.084 ± 0.002 | unsupported | 0.098 ± 0.001 | 0.206 ± 0.003 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 0.833 ± 0.106 | 0.865 ± 0.010 | unsupported | 0.792 ± 0.003 | 2.374 ± 0.114 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.092 ± 0.001 | 0.117 ± 0.001 | unsupported | 0.045 ± 0.001 | 0.181 ± 0.001 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 0.526 ± 0.059 | 0.588 ± 0.012 | unsupported | 0.317 ± 0.006 | 1.400 ± 0.072 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.092 ± 0.001 | 0.092 ± 0.001 | unsupported | 0.151 ± 0.006 | 0.273 ± 0.005 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 1.456 ± 0.005 | 1.458 ± 0.003 | unsupported | 2.351 ± 0.008 | 3.395 ± 0.022 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.041 ± 0.000 | 0.041 ± 0.001 | unsupported | 0.114 ± 0.003 | 0.162 ± 0.004 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 0.608 ± 0.001 | 0.610 ± 0.005 | unsupported | 0.922 ± 0.015 | 1.767 ± 0.022 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.533 ± 0.002 | 1.141 ± 0.028 | unsupported | 0.225 ± 0.004 | 0.154 ± 0.018 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 10.400 ± 0.024 | 19.064 ± 0.290 | unsupported | 4.694 ± 0.085 | 1.438 ± 0.078 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.418 ± 0.009 | 1.157 ± 0.054 | unsupported | 0.215 ± 0.001 | 0.101 ± 0.003 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 6.618 ± 0.147 | 17.159 ± 0.307 | unsupported | 1.859 ± 0.026 | 0.757 ± 0.030 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 23.242 ± 0.172 | - | unsupported | 21.114 ± 0.451 | - | 22.918 ± 0.578 | 69.181 ± 1.328 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 20.163 ± 0.290 | - | unsupported | 19.594 ± 0.102 | - | 22.723 ± 0.292 | 27.056 ± 1.517 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 16.860 ± 0.122 | - | unsupported | 15.968 ± 1.308 | - | 16.755 ± 1.664 | 15.793 ± 0.306 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 13.626 ± 0.146 | - | unsupported | 13.692 ± 0.321 | - | 15.955 ± 0.314 | 13.413 ± 0.217 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 18.112 ± 0.032 | - | unsupported | 16.003 ± 1.591 | - | 18.508 ± 0.225 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 13.670 ± 0.086 | - | unsupported | 14.145 ± 0.290 | - | 18.290 ± 0.182 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 21.215 ± 0.213 | - | unsupported | 21.885 ± 0.425 | - | 26.208 ± 0.324 | 73.156 ± 1.991 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 20.204 ± 0.358 | - | unsupported | 19.755 ± 0.268 | - | 23.937 ± 0.391 | 26.887 ± 0.456 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 41.986 ± 0.168 | - | unsupported | 44.002 ± 0.224 | - | 39.828 ± 0.179 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 20.985 ± 0.107 | - | unsupported | 22.276 ± 0.222 | - | 19.437 ± 0.112 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 41.795 ± 0.143 | - | unsupported | 43.400 ± 0.698 | - | 39.717 ± 0.191 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 20.891 ± 0.142 | - | unsupported | 22.269 ± 0.425 | - | 19.442 ± 0.117 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 23.228 ± 0.095 | - | unsupported | 21.127 ± 0.206 | - | 22.853 ± 0.443 | 70.160 ± 0.967 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 20.114 ± 0.266 | - | unsupported | 20.206 ± 0.355 | - | 22.634 ± 0.158 | 26.928 ± 0.148 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 16.951 ± 0.092 | - | unsupported | 14.617 ± 0.273 | - | 14.902 ± 0.353 | 55.609 ± 0.412 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 13.763 ± 0.112 | - | unsupported | 13.451 ± 0.063 | - | 15.010 ± 0.394 | 20.363 ± 1.713 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.593 ± 0.003 | unsupported | unsupported | 0.775 ± 0.003 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 9.224 ± 0.020 | unsupported | unsupported | 11.011 ± 0.099 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.472 ± 0.013 | unsupported | unsupported | 0.517 ± 0.002 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 5.283 ± 0.118 | unsupported | unsupported | 5.484 ± 0.133 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 23.212 ± 0.170 | - | unsupported | 20.761 ± 0.644 | - | 22.719 ± 0.494 | 69.982 ± 1.660 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 20.160 ± 0.207 | - | unsupported | 19.715 ± 0.090 | - | 22.757 ± 0.299 | 27.015 ± 2.117 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 90.931 ± 7.582 | - | 0.002 ± 0.000 | 89.794 ± 3.919 | 88.157 ± 5.994 | 116.498 ± 6.813 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 38.712 ± 0.355 | - | 0.003 ± 0.000 | 30.093 ± 1.071 | 49.453 ± 5.374 | 115.973 ± 4.285 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 57.742 ± 3.493 | - | 183.508 ± 7.957 | 59.689 ± 3.959 | 66.673 ± 3.665 | 72.401 ± 0.332 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 21.377 ± 1.001 | - | 76.075 ± 3.818 | 20.540 ± 0.292 | 32.149 ± 0.803 | 73.024 ± 2.217 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 183.083 ± 3.685 | - | 180.183 ± 5.160 | 182.450 ± 4.425 | 177.201 ± 4.555 | 216.179 ± 9.762 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 60.673 ± 1.187 | - | 60.224 ± 1.049 | 59.874 ± 0.609 | 97.613 ± 7.367 | 214.416 ± 11.072 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 58.555 ± 2.433 | - | 59.511 ± 7.021 | 61.638 ± 5.477 | 64.200 ± 3.978 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 30.492 ± 0.622 | - | 30.328 ± 1.527 | 21.747 ± 0.104 | 33.443 ± 0.411 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 118.776 ± 7.081 | - | 0.002 ± 0.000 | 106.490 ± 18.265 | 104.635 ± 4.561 | 135.071 ± 4.455 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 37.082 ± 1.206 | - | 0.003 ± 0.000 | 35.324 ± 0.029 | 38.116 ± 6.159 | 140.025 ± 11.556 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 83.673 ± 2.829 | - | 0.002 ± 0.000 | 74.084 ± 1.843 | 66.704 ± 0.846 | 179.971 ± 10.102 | 118.112 ± 3.854 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 25.718 ± 0.616 | - | 0.003 ± 0.000 | 74.709 ± 0.612 | 35.305 ± 0.276 | 179.167 ± 2.951 | 43.942 ± 1.298 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 50.592 ± 2.633 | - | 51.614 ± 2.877 | 60.540 ± 2.629 | 53.417 ± 0.678 | 69.665 ± 2.592 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 20.221 ± 1.080 | - | 20.173 ± 0.149 | 19.507 ± 0.544 | 30.636 ± 2.002 | 70.808 ± 4.633 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 51.456 ± 2.207 | - | 50.773 ± 1.803 | 56.733 ± 2.071 | 52.624 ± 0.791 | 71.113 ± 3.294 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 18.849 ± 0.172 | - | 19.057 ± 0.260 | 18.425 ± 0.199 | 29.029 ± 1.834 | 70.726 ± 5.718 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.001 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 6.733881 | 420867.56 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 4.691071 | 293191.94 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 5.097068 | 318566.75 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 3.075456 | 192216.00 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 3.430041 | 214377.56 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 4.709101 | 294318.81 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 10.325479 | 645342.44 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 10.336876 | 646054.75 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 3.801407 | 237587.94 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 3.782269 | 236391.81 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 9.762151 | 610134.44 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 7.953139 | 497071.19 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 6.536936 | 408558.50 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 6.687308 | 417956.75 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 124.373320 | 7773332.50 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 113.420231 | 7088764.44 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 103.719641 | 6482477.56 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 104.845507 | 6552844.19 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 22.523768 | 1407735.50 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 6.656106 | 416006.62 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 5.133809 | 320863.06 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 62.979349 | 3936209.31 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 141.714984 | 8857186.50 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 73.458805 | 4591175.31 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 86.027310 | 5376706.88 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 216.602635 | 13537664.69 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 15.384151 | 961509.44 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 15.727517 | 982969.81 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 279.889556 | 17493097.25 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 281.931319 | 17620707.44 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 18.553423 | 1159588.94 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 174.251850 | 10890740.62 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 268.259598 | 16766224.87 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 326.603356 | 20412709.75 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 14.388757 | 899297.31 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 14.694707 | 918419.19 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 266.044049 | 16627753.06 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 267.988298 | 16749268.62 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 17.028330 | 1064270.62 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 178.005299 | 11125331.19 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 252.491986 | 15780749.12 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 304.971307 | 19060706.69 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.344613 | 114759.58 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.529027 | 133266.05 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.088245 | 79503.83 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.032063 | 78625.98 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 4.241587 | 530198.38 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 16.559360 | 2069920.00 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.055963 | 1256995.38 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.028372 | 1253546.50 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.225002 | 34765.66 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.117774 | 64340.22 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.560205 | 40003.20 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.625230 | 41019.22 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.192313 | 274039.12 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.042795 | 880349.38 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.534748 | 566843.50 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.542740 | 567842.50 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.038233 | 188097.39 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.417620 | 84650.31 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.401843 | 37528.80 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.431487 | 37991.98 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 16.418034 | 2052254.25 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.012710 | 1251588.75 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.714676 | 589334.50 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.929783 | 616222.88 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.421498 | 115960.91 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.376244 | 68378.81 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.286271 | 20097.98 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.321333 | 20645.83 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.927766 | 865970.75 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.843210 | 480401.25 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.253864 | 281733.00 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.293973 | 286746.62 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.458155 | 147783.67 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.091352 | 110802.38 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.725434 | 105084.91 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.744241 | 105378.77 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.364122 | 1545515.25 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 20.568978 | 2571122.25 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.367244 | 1670905.50 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 13.455965 | 1681995.62 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.927576 | 108243.38 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.615716 | 72120.56 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.218652 | 50291.44 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3.256899 | 50889.05 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.675576 | 1084447.00 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.023757 | 1002969.62 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.102704 | 762838.00 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 6.105628 | 763203.50 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.842731 | 153792.67 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.347099 | 130423.42 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.793770 | 43652.66 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.409447 | 68897.61 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.241676 | 1655209.50 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 5.517957 | 689744.62 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.695407 | 586925.88 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.986646 | 623330.75 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.095554 | 142118.03 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.448330 | 53880.16 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.005210 | 78206.41 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 8.024049 | 125375.77 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.018281 | 1002285.12 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.356435 | 294554.38 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.938921 | 367365.12 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.341653 | 417706.62 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.542682 | 211604.41 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.730680 | 89541.88 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.213999 | 65843.73 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.190175 | 81096.48 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 19.500051 | 2437506.38 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 5.805780 | 725722.50 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.565568 | 820696.00 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.927291 | 865911.38 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.943721 | 186620.64 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.808940 | 43889.69 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.131957 | 95811.83 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 8.058049 | 125907.02 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.185769 | 1398221.12 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.677207 | 334650.88 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.013802 | 501725.25 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.402834 | 550354.25 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.755114 | 214923.66 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 14.110138 | 220470.91 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 15.733094 | 245829.59 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 25.656416 | 3207052.00 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 25.799524 | 3224940.50 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 26.366758 | 3295844.75 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 14.276833 | 223075.52 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 11.448684 | 178885.69 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 13.669770 | 213590.16 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 14.280029 | 1785003.62 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 11.180390 | 1397548.75 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 11.806541 | 1475817.62 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.619886 | 40935.72 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.634243 | 9910.05 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.555778 | 8684.03 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 0.931841 | 14560.02 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 2.046344 | 255793.00 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.486386 | 60798.25 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.737925 | 92240.62 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.780819 | 97602.38 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.180937 | 34077.14 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.688513 | 10758.02 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.768390 | 12006.09 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.289977 | 20155.89 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.790693 | 223836.62 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.315134 | 39391.75 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.471373 | 58921.62 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 0.508007 | 63500.88 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.723665 | 151932.27 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.489806 | 85778.22 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.622808 | 25356.38 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.455172 | 38362.06 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.957679 | 1494709.88 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 3.863965 | 482995.62 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.564202 | 320525.25 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 2.778357 | 347294.62 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.402935 | 115670.86 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.587735 | 87308.36 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.358919 | 36858.11 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3.614174 | 56471.47 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.682123 | 1210265.38 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.761009 | 220126.12 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.560064 | 195008.00 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.841978 | 230247.25 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.896972 | 139015.19 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.888331 | 76380.17 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.655996 | 25874.94 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.660142 | 25939.72 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.197499 | 1399687.38 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.085076 | 1260634.50 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.198605 | 399825.62 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.259154 | 407394.25 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.175060 | 96485.31 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.771685 | 58932.58 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.931990 | 14562.34 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 0.957638 | 14963.09 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.856631 | 732078.88 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.721992 | 465249.00 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.545703 | 193212.88 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.617848 | 202231.00 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.111527 | 142367.61 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.440905 | 85014.14 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.705469 | 26647.95 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.765537 | 27586.52 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.483920 | 1435490.00 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.114880 | 1264360.00 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.324371 | 415546.38 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.408504 | 426063.00 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.449493 | 100773.33 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.865313 | 60395.52 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.961694 | 15026.47 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.000451 | 15632.05 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.055443 | 756930.38 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.853375 | 481671.88 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.604870 | 200608.75 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.688693 | 211086.62 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.212748 | 206449.19 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.298806 | 98418.84 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.126541 | 64477.20 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.388647 | 84197.61 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.992222 | 2374027.75 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 6.337653 | 792206.62 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.665907 | 833238.38 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.916004 | 864500.50 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.580713 | 180948.64 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.866214 | 44784.59 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.882429 | 91912.95 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.502321 | 117223.77 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.198656 | 1399832.00 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.537200 | 317150.00 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.209100 | 526137.50 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.702355 | 587794.38 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 17.502476 | 273476.19 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 9.640302 | 150629.72 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.876565 | 91821.33 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.877837 | 91841.20 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 27.162016 | 3395252.00 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 18.809160 | 2351145.00 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 11.651901 | 1456487.62 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.665070 | 1458133.75 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.339683 | 161557.55 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 7.288546 | 113883.53 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.608275 | 40754.30 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.636607 | 41196.98 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 14.134212 | 1766776.50 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.379280 | 922410.00 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.865224 | 608153.00 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.878043 | 609755.38 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.885413 | 154459.58 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 14.371992 | 224562.38 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 34.090300 | 532660.94 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 73.041340 | 1141270.94 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.501689 | 1437711.12 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 37.550820 | 4693852.50 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 83.203159 | 10400394.88 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 152.510025 | 19063753.12 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.446228 | 100722.31 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 13.753582 | 214899.72 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 26.764045 | 418188.20 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 74.020592 | 1156571.75 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.055605 | 756950.62 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 14.870523 | 1858815.38 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 52.946254 | 6618281.75 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 137.269897 | 17158737.12 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 12.404968 | 775310.50 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 9.480066 | 592504.12 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 176.179274 | 11011204.62 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 147.585621 | 9224101.31 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 8.276339 | 517271.19 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 7.553275 | 472079.69 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 87.745700 | 5484106.25 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 84.525950 | 5282871.88 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.017886 | 1117.88 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002123 | 132.69 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.017856 | 1116.00 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002043 | 127.69 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000110 | 6.88 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.016515 | 1032.19 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.008012 | 500.75 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000170 | 10.62 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.017035 | 1064.69 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.007481 | 467.56 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000170 | 10.62 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.017696 | 1106.00 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.005458 | 341.12 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000271 | 16.94 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.017626 | 1101.62 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.004877 | 304.81 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000090 | 5.62 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.016174 | 1010.88 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003085 | 192.81 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000090 | 5.62 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.016054 | 1003.37 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003204 | 200.25 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/norm_fro` (c64, threads=1, shape=`2048x1536`): `tenferro-direct` is 13.8x faster than the slowest successful cell (`pytorch-cpu`, 18.171 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/exp` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 14.6x faster than the slowest successful cell (`julia-base`, 56.007 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/exp` (f64, threads=4, shape=`8388608`): `tenferro-trace` is 13.6x faster than the slowest successful cell (`julia-base`, 56.007 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log1p` (f64, threads=4, shape=`4194304`): `tenferro-direct` is 10.4x faster than the slowest successful cell (`julia-base`, 35.217 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/log1p` (f64, threads=4, shape=`4194304`): `tenferro-trace` is 10.3x faster than the slowest successful cell (`julia-base`, 35.217 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 21.964 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 24.3x faster than the slowest successful cell (`pytorch-cpu`, 21.964 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 21.964 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 14.6x faster than the slowest successful cell (`pytorch-cpu`, 21.964 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 9.734 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 9.734 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 13.8x faster than the slowest successful cell (`pytorch-cpu`, 9.734 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 22.043 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 24.5x faster than the slowest successful cell (`pytorch-cpu`, 22.043 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 22.043 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 22.043 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 9.342 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 13.7x faster than the slowest successful cell (`pytorch-cpu`, 9.342 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 13.4x faster than the slowest successful cell (`pytorch-cpu`, 9.342 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sin` (f64, threads=4, shape=`8388608`): `tenferro-direct` is 11.1x faster than the slowest successful cell (`julia-base`, 55.627 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sin` (f64, threads=4, shape=`8388608`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`julia-base`, 55.627 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `jax-cpu` is 10.8x faster than the slowest successful cell (`julia-base`, 6.868 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `pytorch-cpu` is 27.8x faster than the slowest successful cell (`julia-base`, 6.868 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `tenferro-direct` is 26.2x faster than the slowest successful cell (`julia-base`, 6.868 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `tenferro-trace` is 25.0x faster than the slowest successful cell (`julia-base`, 6.868 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=1, shape=`4194304 -> 2097152`): `jax-cpu` is 10.6x faster than the slowest successful cell (`julia-base`, 7.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=1, shape=`4194304 -> 2097152`): `pytorch-cpu` is 12.2x faster than the slowest successful cell (`julia-base`, 7.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=1, shape=`4194304 -> 2097152`): `tenferro-direct` is 11.0x faster than the slowest successful cell (`julia-base`, 7.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `jax-cpu` is 10.3x faster than the slowest successful cell (`julia-base`, 7.896 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 13.0x faster than the slowest successful cell (`julia-base`, 7.896 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 16.3x faster than the slowest successful cell (`julia-base`, 7.896 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `jax-cpu` is 10.7x faster than the slowest successful cell (`julia-base`, 6.269 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 19.6x faster than the slowest successful cell (`julia-base`, 6.269 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 24.0x faster than the slowest successful cell (`julia-base`, 6.269 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-trace` is 26.5x faster than the slowest successful cell (`julia-base`, 6.269 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 3761.2x faster than the slowest successful cell (`julia-base`, 8.703 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `pytorch-cpu` is 12.5x faster than the slowest successful cell (`julia-base`, 8.783 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-direct` is 13.2x faster than the slowest successful cell (`julia-base`, 8.783 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 3055.9x faster than the slowest successful cell (`julia-base`, 8.783 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 14.7x faster than the slowest successful cell (`jax-cpu`, 0.023 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.029 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.038 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`256x128`): `tenferro-direct` is 12.3x faster than the slowest successful cell (`tenferro-eager`, 3.936 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 12.7x faster than the slowest successful cell (`jax-cpu`, 11.974 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 38.9x faster than the slowest successful cell (`jax-cpu`, 11.974 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/rank_revealing_qr_deficient` (f64, threads=4, shape=`256x128`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`tenferro-eager`, 11.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 13.3x faster than the slowest successful cell (`tenferro-eager`, 19.064 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 11.5x faster than the slowest successful cell (`tenferro-eager`, 1.157 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 22.7x faster than the slowest successful cell (`tenferro-eager`, 17.159 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 53122.7x faster than the slowest successful cell (`julia-base`, 116.498 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 41656.8x faster than the slowest successful cell (`julia-base`, 115.973 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 59660.4x faster than the slowest successful cell (`julia-base`, 135.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 48721.1x faster than the slowest successful cell (`julia-base`, 140.025 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 88091.7x faster than the slowest successful cell (`julia-base`, 179.971 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 67255.0x faster than the slowest successful cell (`julia-base`, 179.167 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 147.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 96.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 100.5x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 64.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 168.5x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 167.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 7246.8 GB/s at 0.002 ms. The conservative physical lower bound is 0.168 ms, so the cell is 72.5x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 5834.7 GB/s at 0.003 ms. The conservative physical lower bound is 0.042 ms, so the cell is 14.6x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 122405.6 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1224.1x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 96420.8 GB/s at 0.003 ms. The conservative physical lower bound is 0.671 ms, so the cell is 241.1x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 118566.9 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1185.7x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 93401.3 GB/s at 0.003 ms. The conservative physical lower bound is 0.671 ms, so the cell is 233.5x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 65696.4 GB/s at 0.002 ms. The conservative physical lower bound is 1.342 ms, so the cell is 657.0x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 50382.0 GB/s at 0.003 ms. The conservative physical lower bound is 0.336 ms, so the cell is 126.0x faster than the bound.

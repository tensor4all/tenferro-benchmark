# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20261106_182046/run.yaml`
- Timestamp: `2026-11-06T18:20:46.704878+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20261106_182046`.

- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

## CPU Information

- Model: `Apple M1 (Virtual)`
- Vendor: `Apple`
- Logical CPUs: `3`
- Physical CPUs: `3`
- Sockets: `1`
- Cores per socket: `3`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Standard: 3 physical / 3 logical (L1i 192 KiB, L1d 128 KiB, L2 12 MiB, 3 CPUs/L2)`
- Python platform: `macOS-26.6.2-arm64-arm-64bit-Mach-O`
- Container: `none`

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

- CSV: `data/results/mac-cpu/cpu/public_api/20261106_182046/cpu_public_api_t1_20261106_182046.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20261106_182046/cpu_public_api_t4_20261106_182046.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 3.749 ± 0.158 | - | 3.837 ± 0.285 | 3.863 ± 0.468 | 1.579 ± 0.052 | 7.613 ± 0.853 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 3.888 ± 0.483 | - | 3.941 ± 0.325 | 4.417 ± 0.388 | 1.365 ± 0.065 | 75.740 ± 13.878 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 13.474 ± 2.466 | - | 12.483 ± 0.576 | 11.770 ± 1.030 | 12.561 ± 0.249 | 16.299 ± 9.275 | 10.449 ± 0.351 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 11.003 ± 2.056 | - | 11.289 ± 3.706 | 17.626 ± 4.671 | 14.646 ± 1.065 | 12.195 ± 3.128 | 13.851 ± 1.204 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 48.940 ± 5.639 | - | 45.302 ± 1.679 | 18.384 ± 3.273 | 36.917 ± 1.384 | 58.509 ± 5.574 | 55.251 ± 3.592 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 23.880 ± 2.291 | - | 20.029 ± 7.732 | 10.909 ± 2.959 | 18.997 ± 4.349 | 52.333 ± 7.495 | 28.646 ± 15.274 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 10.950 ± 2.937 | - | 10.030 ± 0.448 | 9.895 ± 0.860 | 56.695 ± 13.715 | 64.113 ± 4.036 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 10.053 ± 0.741 | - | 10.836 ± 0.690 | 11.444 ± 0.476 | 22.022 ± 4.857 | 40.135 ± 21.973 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 11.420 ± 1.517 | - | unsupported | 10.541 ± 2.586 | 59.950 ± 9.584 | 83.290 ± 23.114 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 13.081 ± 0.961 | - | unsupported | 11.264 ± 1.232 | 17.849 ± 1.687 | 53.619 ± 18.746 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 11.931 ± 0.638 | - | 12.099 ± 1.481 | 11.187 ± 0.946 | 22.849 ± 1.380 | 9.049 ± 1.195 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 11.475 ± 1.413 | - | 11.404 ± 5.830 | 11.951 ± 1.265 | 21.269 ± 1.414 | 508.308 ± 598.234 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 37.716 ± 3.622 | - | 37.422 ± 5.362 | 37.123 ± 5.923 | 47.050 ± 5.295 | 54.896 ± 8.201 | 54.588 ± 5.838 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 14.095 ± 2.362 | - | 16.079 ± 3.281 | 20.480 ± 8.964 | 31.739 ± 13.988 | 47.602 ± 2.726 | 22.694 ± 13.505 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 47.177 ± 2.625 | - | 49.042 ± 3.156 | 65.796 ± 5.453 | 94.479 ± 8.979 | 77.559 ± 10.610 | 76.081 ± 2.306 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 22.356 ± 0.844 | - | 21.815 ± 1.063 | 26.527 ± 4.052 | 43.320 ± 16.364 | 80.184 ± 5.996 | 49.939 ± 12.251 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 10.180 ± 0.896 | - | 8.947 ± 1.002 | 10.549 ± 1.591 | 9.086 ± 0.851 | 10.551 ± 1.015 | 10.424 ± 1.114 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 8.306 ± 1.330 | - | 8.644 ± 1.468 | 10.291 ± 0.882 | 11.805 ± 1.744 | 11.028 ± 0.854 | 11.313 ± 2.066 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.265 ± 0.109 | - | 7.600 ± 0.265 | 6.296 ± 0.641 | 5.298 ± 0.732 | 8.933 ± 0.696 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 0.904 ± 0.019 | - | 5.334 ± 1.378 | 3.581 ± 0.367 | 2.801 ± 0.112 | 9.085 ± 5.099 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 8.334 ± 0.499 | - | 7.633 ± 0.671 | 7.260 ± 0.755 | 7.312 ± 0.964 | 10.490 ± 4.796 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 8.794 ± 0.794 | - | 7.442 ± 0.091 | 7.601 ± 0.223 | 6.965 ± 0.865 | 175.469 ± 52.941 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 5.074 ± 0.229 | - | 5.309 ± 0.349 | 4.980 ± 0.435 | 2.332 ± 0.091 | 7.285 ± 2.016 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 5.896 ± 0.205 | - | 6.123 ± 0.535 | 5.835 ± 0.330 | 2.517 ± 0.363 | 151.977 ± 32.992 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 8.493 ± 0.798 | - | 7.630 ± 0.141 | 7.161 ± 0.140 | 7.210 ± 0.465 | 11.254 ± 3.407 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 7.890 ± 0.144 | - | 7.882 ± 0.629 | 7.473 ± 0.296 | 6.495 ± 0.805 | 624.846 ± 40.052 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 11.908 ± 1.208 | - | 12.864 ± 0.761 | 10.244 ± 0.345 | 63.137 ± 10.945 | 74.488 ± 10.821 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 10.675 ± 0.655 | - | 11.727 ± 1.197 | 10.703 ± 0.806 | 16.668 ± 2.043 | 49.306 ± 27.067 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 13.772 ± 2.830 | - | unsupported | 16.480 ± 2.130 | 63.283 ± 10.031 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 16.684 ± 4.665 | - | unsupported | 12.834 ± 2.791 | 27.072 ± 4.163 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 16.430 ± 2.862 | - | 13.018 ± 3.137 | 12.243 ± 1.484 | 21.326 ± 6.345 | 12.343 ± 2.493 | 13.769 ± 0.854 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 14.088 ± 0.899 | - | 13.584 ± 1.932 | 12.501 ± 2.439 | 10.573 ± 1.407 | 15.243 ± 4.072 | 12.617 ± 4.077 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 19.620 ± 7.262 | - | 21.284 ± 8.114 | 20.251 ± 3.678 | 25.920 ± 2.544 | 22.633 ± 23.114 | 28.980 ± 6.417 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 20.196 ± 3.859 | - | 23.640 ± 7.479 | 21.360 ± 3.175 | 28.177 ± 2.627 | 30.199 ± 36.448 | 21.354 ± 5.771 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 39.416 ± 13.782 | - | 39.101 ± 4.132 | 39.465 ± 1.320 | 30.588 ± 1.518 | 53.909 ± 6.226 | 69.440 ± 11.128 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 14.525 ± 7.467 | - | 17.588 ± 2.336 | 18.418 ± 1.930 | 17.323 ± 5.831 | 47.524 ± 7.443 | 26.447 ± 4.188 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 11.567 ± 2.997 | - | 7.186 ± 0.782 | 9.130 ± 2.337 | 7.098 ± 1.631 | 7.796 ± 2.705 | 7.707 ± 35.526 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 7.179 ± 2.151 | - | 6.106 ± 1.228 | 7.162 ± 1.478 | 7.857 ± 1.505 | 7.490 ± 1.522 | 7.991 ± 4.132 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 14.258 ± 1.661 | - | 10.790 ± 0.835 | 20.000 ± 1.311 | 20.529 ± 1.865 | 26.692 ± 10.388 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 14.129 ± 1.507 | - | 13.432 ± 1.132 | 12.778 ± 1.292 | 15.566 ± 1.639 | 25.620 ± 10.616 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 37.014 ± 5.564 | - | 31.950 ± 6.027 | 34.509 ± 1.475 | 28.816 ± 1.475 | 52.202 ± 5.522 | 44.037 ± 23.791 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 20.888 ± 7.402 | - | 14.042 ± 2.193 | 18.706 ± 5.633 | 14.883 ± 1.836 | 38.741 ± 11.643 | 14.242 ± 1.792 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 18.354 ± 1.905 | - | 19.920 ± 2.474 | 19.669 ± 1.579 | 37.267 ± 4.909 | 17.481 ± 3.780 | 17.768 ± 5.650 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 23.248 ± 3.421 | - | 23.891 ± 2.969 | 16.989 ± 2.825 | 25.296 ± 3.594 | 28.557 ± 25.993 | 19.890 ± 4.347 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 21.118 ± 3.620 | - | 21.017 ± 5.858 | 40.931 ± 12.790 | 16.195 ± 3.287 | 44.020 ± 2.869 | 36.665 ± 2.984 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 10.840 ± 7.493 | - | 12.246 ± 5.357 | 13.005 ± 1.697 | 9.047 ± 0.389 | 34.336 ± 1.581 | 18.627 ± 3.131 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 28.340 ± 4.736 | - | 27.529 ± 0.562 | 25.954 ± 2.411 | 17.473 ± 1.494 | 33.747 ± 6.582 | 58.461 ± 18.586 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 15.292 ± 2.592 | - | 11.091 ± 0.673 | 16.205 ± 2.167 | 9.642 ± 1.841 | 24.197 ± 1.439 | 10.981 ± 2.421 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 36.855 ± 11.693 | - | 40.171 ± 7.414 | 44.040 ± 5.040 | 31.494 ± 7.016 | 48.574 ± 13.657 | 47.009 ± 5.072 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 19.738 ± 3.264 | - | 21.864 ± 6.081 | 24.959 ± 5.264 | 13.869 ± 1.476 | 46.497 ± 1.011 | 19.529 ± 6.246 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 18.811 ± 1.848 | - | 20.175 ± 7.610 | 20.238 ± 2.512 | 20.506 ± 2.134 | 36.856 ± 22.536 | 40.202 ± 43.112 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 7.847 ± 0.456 | - | 7.796 ± 0.368 | 11.920 ± 4.867 | 10.320 ± 0.373 | 20.647 ± 2.046 | 14.541 ± 5.541 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.278 ± 0.021 | 0.260 ± 0.028 | unsupported | unsupported | 0.305 ± 0.153 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.397 ± 0.256 | 0.372 ± 0.079 | unsupported | unsupported | 0.202 ± 0.017 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 19.722 ± 1.529 | - | 18.214 ± 3.500 | 24.044 ± 7.239 | 35.171 ± 4.574 | 31.549 ± 51.381 | 18.353 ± 3.657 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 20.657 ± 3.699 | - | 22.608 ± 2.227 | 16.313 ± 4.805 | 25.048 ± 2.873 | 115.035 ± 202.376 | 21.974 ± 8.838 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 17.958 ± 2.349 | - | 16.940 ± 2.570 | 25.348 ± 2.959 | 34.997 ± 1.503 | 23.132 ± 9.423 | 20.874 ± 3.657 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 22.920 ± 6.346 | - | 20.212 ± 3.252 | 16.640 ± 2.215 | 25.171 ± 1.015 | 23.972 ± 9.325 | 26.999 ± 10.648 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 23.624 ± 9.448 | - | 21.541 ± 5.668 | 19.779 ± 1.647 | 32.152 ± 4.821 | 24.416 ± 6.402 | 18.939 ± 2.549 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 20.439 ± 6.171 | - | 21.696 ± 1.771 | 19.117 ± 2.610 | 23.097 ± 2.144 | 17.589 ± 5.495 | 20.498 ± 5.117 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 11.792 ± 2.347 | - | 17.297 ± 7.896 | 10.702 ± 1.865 | 20.931 ± 5.652 | 12.630 ± 2.525 | 14.961 ± 0.996 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 15.515 ± 3.811 | - | 11.756 ± 1.385 | 14.714 ± 3.557 | 13.077 ± 2.905 | 19.792 ± 16.569 | 12.725 ± 3.032 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 36.286 ± 4.171 | - | 35.833 ± 6.062 | 70.697 ± 1.554 | 44.404 ± 4.087 | 111.023 ± 33.561 | 80.085 ± 6.637 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 21.674 ± 4.021 | - | 22.711 ± 6.237 | 32.900 ± 7.141 | 19.396 ± 3.132 | 80.113 ± 8.726 | 28.146 ± 4.224 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 15.983 ± 6.815 | - | 10.487 ± 5.982 | 12.911 ± 1.535 | 27.547 ± 3.023 | 8.227 ± 3.616 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 6.296 ± 0.973 | - | 6.976 ± 1.880 | 6.911 ± 0.417 | 12.706 ± 1.082 | 5.883 ± 0.410 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.009 ± 0.150 | - | 1.218 ± 0.277 | 13.517 ± 4.029 | 2.672 ± 0.161 | 0.924 ± 0.513 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.829 ± 0.119 | - | 0.915 ± 0.075 | 11.600 ± 1.002 | 1.778 ± 0.407 | 0.700 ± 0.134 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 0.909 ± 0.089 | - | 1.059 ± 0.062 | 1.729 ± 0.159 | 0.890 ± 0.169 | 0.803 ± 0.116 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 1.851 ± 0.230 | - | 2.107 ± 0.588 | 1.123 ± 0.226 | 0.803 ± 0.172 | 0.714 ± 0.033 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 10.025 ± 0.719 | - | 13.856 ± 6.424 | 12.181 ± 0.714 | 32.775 ± 9.975 | 6.777 ± 0.678 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 6.446 ± 4.404 | - | 5.870 ± 0.624 | 7.154 ± 0.494 | 13.427 ± 1.376 | 5.676 ± 0.506 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.010 ± 0.119 | - | 1.128 ± 0.148 | 15.797 ± 2.255 | 1.980 ± 0.231 | 0.683 ± 0.032 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.832 ± 0.134 | - | 0.767 ± 0.068 | 12.458 ± 2.838 | 1.780 ± 0.334 | 1.073 ± 0.305 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 4.238 ± 0.257 | - | 4.077 ± 1.189 | 5.837 ± 0.353 | 3.439 ± 0.554 | 4.039 ± 1.457 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 6.028 ± 1.472 | - | 6.234 ± 1.254 | 4.454 ± 1.846 | 4.206 ± 1.897 | 3.128 ± 0.183 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 8.760 ± 1.142 | - | 10.099 ± 2.110 | 6.657 ± 0.242 | 52.238 ± 2.665 | 6.947 ± 0.607 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 6.625 ± 1.207 | - | 7.660 ± 3.417 | 5.699 ± 0.251 | 19.880 ± 2.281 | 5.558 ± 1.626 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 1.116 ± 0.118 | - | 1.133 ± 0.288 | 4.352 ± 0.280 | 2.387 ± 0.495 | 0.705 ± 0.086 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 1.531 ± 0.791 | - | 0.982 ± 0.297 | 2.477 ± 0.117 | 1.399 ± 0.219 | 0.931 ± 0.277 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 0.949 ± 0.209 | - | 1.224 ± 0.807 | 0.689 ± 0.073 | 1.076 ± 0.272 | 0.734 ± 0.038 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 1.632 ± 0.310 | - | 1.906 ± 1.007 | 0.839 ± 0.084 | 0.843 ± 0.079 | 0.939 ± 0.067 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 7.018 ± 0.331 | - | 7.397 ± 1.869 | 6.786 ± 1.360 | 33.913 ± 11.783 | 6.691 ± 1.223 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 6.757 ± 2.827 | - | 6.970 ± 1.132 | 5.266 ± 0.634 | 16.738 ± 0.188 | 6.909 ± 0.861 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 1.074 ± 0.119 | - | 1.087 ± 0.083 | 4.546 ± 0.111 | 2.845 ± 0.708 | 1.004 ± 0.050 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.816 ± 0.150 | - | 0.709 ± 0.049 | 2.600 ± 0.327 | 1.456 ± 0.539 | 0.840 ± 0.051 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.056 ± 0.145 | - | 1.172 ± 0.202 | 0.626 ± 0.018 | 1.169 ± 0.080 | 0.804 ± 0.150 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 1.662 ± 0.489 | - | 1.854 ± 0.344 | 0.827 ± 0.041 | 0.944 ± 0.110 | 0.969 ± 0.115 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 1.381 ± 0.284 | 1.165 ± 0.221 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.899 ± 0.440 | 1.195 ± 0.178 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 33.486 ± 7.730 | - | 35.368 ± 3.175 | 32.652 ± 3.129 | 38.243 ± 2.419 | 30.262 ± 8.201 | 34.142 ± 1.184 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 11.617 ± 0.382 | - | 11.274 ± 2.764 | 20.858 ± 6.836 | 22.027 ± 1.613 | 28.903 ± 25.458 | 12.873 ± 3.348 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 24.715 ± 7.147 | - | 24.173 ± 6.709 | 19.666 ± 1.886 | 43.557 ± 5.713 | 44.633 ± 5.168 | 49.047 ± 3.384 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 11.274 ± 2.917 | - | 11.528 ± 1.588 | 13.608 ± 2.823 | 31.851 ± 11.198 | 41.149 ± 4.973 | 31.097 ± 17.073 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 27.224 ± 2.767 | - | 28.100 ± 6.182 | 25.682 ± 2.418 | 31.153 ± 5.688 | 32.951 ± 7.875 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 25.608 ± 1.842 | - | 25.924 ± 5.614 | 19.335 ± 1.968 | 23.733 ± 1.728 | 114.900 ± 188.446 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 12.208 ± 3.438 | - | 12.432 ± 0.962 | 12.643 ± 0.701 | 21.723 ± 3.413 | 11.332 ± 3.855 | 22.114 ± 1.394 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 13.096 ± 1.117 | - | 14.548 ± 2.075 | 14.046 ± 3.099 | 14.367 ± 2.291 | 44.353 ± 176.628 | 14.374 ± 1.892 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 28.781 ± 5.609 | - | 28.187 ± 5.538 | 29.908 ± 2.271 | 27.748 ± 2.136 | 31.049 ± 2.620 | 35.134 ± 3.815 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 15.445 ± 6.581 | - | 16.986 ± 10.030 | 14.774 ± 3.307 | 13.661 ± 0.812 | 29.288 ± 5.745 | 11.951 ± 1.058 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 16.204 ± 1.564 | - | 17.660 ± 3.194 | 13.570 ± 1.677 | 22.317 ± 3.394 | 32.704 ± 4.517 | 29.699 ± 20.819 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 11.703 ± 1.608 | - | 10.330 ± 1.843 | 14.443 ± 2.072 | 17.180 ± 2.947 | 30.931 ± 3.333 | 15.444 ± 4.002 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 35.861 ± 13.143 | - | 25.288 ± 6.967 | 17.441 ± 2.148 | 26.661 ± 3.892 | 27.753 ± 41.895 | 18.987 ± 5.303 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 23.000 ± 4.801 | - | 21.469 ± 2.840 | 22.879 ± 2.639 | 23.292 ± 3.633 | 18.267 ± 40.481 | 18.744 ± 3.476 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 69.594 ± 4.935 | - | 68.086 ± 11.723 | 73.616 ± 6.830 | 12.807 ± 0.461 | 48.355 ± 7.636 | 64.114 ± 24.059 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 25.101 ± 0.408 | - | 33.843 ± 6.554 | 36.144 ± 8.639 | 7.303 ± 0.398 | 44.907 ± 1.719 | 16.943 ± 1.954 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 39.835 ± 24.018 | 35.105 ± 6.182 | unsupported | 33.074 ± 8.109 | 31.565 ± 8.158 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 26.443 ± 1.584 | 26.957 ± 2.011 | unsupported | 21.574 ± 3.075 | 37.089 ± 7.788 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.910 ± 0.056 | - | 0.885 ± 0.055 | 0.689 ± 0.167 | 0.748 ± 0.333 | 0.791 ± 0.125 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.698 ± 0.116 | - | 0.887 ± 0.301 | 1.005 ± 0.178 | 0.813 ± 0.057 | 0.770 ± 0.021 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.825 ± 0.304 | - | 0.852 ± 0.335 | 0.723 ± 0.052 | 0.866 ± 0.128 | 0.871 ± 0.082 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.729 ± 0.113 | - | 0.885 ± 0.483 | 0.737 ± 0.103 | 1.446 ± 0.785 | 0.866 ± 0.168 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.237 ± 0.144 | - | unsupported | 1.126 ± 0.180 | 1.188 ± 0.078 | 1.424 ± 0.284 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 1.209 ± 0.393 | - | unsupported | 1.465 ± 0.107 | 1.215 ± 0.115 | 1.133 ± 0.348 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.346 ± 0.016 | - | 0.372 ± 0.011 | 0.335 ± 0.233 | 0.308 ± 0.007 | 1.221 ± 0.097 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.611 ± 0.068 | - | 0.938 ± 0.430 | 0.335 ± 0.015 | 0.243 ± 0.039 | 1.306 ± 1.101 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.738 ± 0.060 | - | 0.808 ± 0.100 | 0.874 ± 0.114 | 0.733 ± 0.070 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.962 ± 0.251 | - | 0.709 ± 0.142 | 1.299 ± 0.064 | 0.880 ± 0.270 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.836 ± 0.223 | - | 0.863 ± 0.141 | 1.091 ± 0.086 | 0.793 ± 0.071 | 1.446 ± 0.366 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.956 ± 0.327 | - | 0.915 ± 0.081 | 1.013 ± 0.611 | 0.808 ± 0.110 | 1.485 ± 0.343 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.609 ± 0.043 | - | 0.772 ± 0.129 | 0.450 ± 0.034 | 0.376 ± 0.034 | 0.350 ± 0.022 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 2.659 ± 1.532 | - | 1.694 ± 1.923 | 0.545 ± 0.052 | 0.444 ± 0.072 | 1.195 ± 0.782 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.272 ± 0.211 | - | 0.002 ± 0.000 | 1.195 ± 0.180 | 1.202 ± 0.071 | 1.790 ± 0.412 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 1.537 ± 0.465 | - | 0.002 ± 0.000 | 1.368 ± 0.066 | 1.437 ± 0.064 | 1.530 ± 0.095 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.051 ± 0.006 | - | 0.060 ± 0.001 | 0.269 ± 0.004 | 0.080 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.086 ± 0.006 | - | 0.097 ± 0.008 | 0.171 ± 0.019 | 0.136 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.232 ± 0.058 | - | 0.227 ± 0.037 | 0.456 ± 0.058 | 0.243 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.025 ± 0.002 | - | 0.033 ± 0.000 | 0.095 ± 0.046 | 0.084 ± 0.074 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.005 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.010 | 0.062 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.011 ± 0.003 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.004 ± 0.001 | 0.005 ± 0.003 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.002 | 0.012 ± 0.001 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.001 | 0.012 ± 0.001 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.057 ± 0.399 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.152 ± 0.001 | - | 0.200 ± 0.001 | 0.317 ± 0.150 | 0.276 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.010 ± 0.000 | - | 0.015 ± 0.001 | 0.037 ± 0.014 | 0.064 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.012 ± 0.001 | 0.036 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.048 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.061 ± 0.009 | - | 0.065 ± 0.002 | 0.172 ± 0.003 | 0.118 ± 0.068 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.100 ± 0.002 | - | 0.106 ± 0.006 | 0.230 ± 0.127 | 0.253 ± 0.069 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.203 ± 0.014 | - | 0.227 ± 0.005 | 0.285 ± 0.017 | 0.251 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.026 ± 0.001 | - | 0.040 ± 0.006 | 0.143 ± 0.045 | 0.090 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.074 ± 0.011 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.103 ± 0.017 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.004 ± 0.007 | - | 0.005 ± 0.001 | 0.093 ± 0.055 | 0.008 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.080 ± 0.011 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.103 ± 0.034 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.080 ± 0.051 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.076 ± 0.009 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.081 ± 0.007 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.064 ± 0.051 | 0.009 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.075 ± 0.050 | 0.053 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.176 ± 0.021 | - | 0.343 ± 0.117 | 0.474 ± 0.093 | 0.229 ± 0.114 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.080 ± 0.006 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.012 ± 0.001 | - | 0.018 ± 0.001 | 0.105 ± 0.039 | 0.032 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.002 ± 0.003 | - | 0.006 ± 0.001 | 0.119 ± 0.029 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.110 ± 0.037 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.289 ± 0.009 | - | 0.296 ± 0.029 | 0.452 ± 0.028 | 0.771 ± 0.072 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.586 ± 0.039 | - | 1.506 ± 0.169 | 1.886 ± 0.784 | 2.159 ± 0.267 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 7.428 ± 2.109 | - | 6.341 ± 0.160 | 7.215 ± 1.238 | 7.939 ± 2.066 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 6.119 ± 0.690 | - | 4.523 ± 0.230 | 4.289 ± 0.912 | 4.317 ± 0.367 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.086 ± 0.001 | - | 0.109 ± 0.010 | 0.093 ± 0.002 | 0.132 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.019 ± 0.018 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.017 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.016 ± 0.000 | - | 0.027 ± 0.000 | 0.010 ± 0.005 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.019 ± 0.004 | - | 0.025 ± 0.006 | 0.028 ± 0.006 | 0.088 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.020 ± 0.000 | - | 0.031 ± 0.000 | 0.014 ± 0.003 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.027 ± 0.007 | - | 0.028 ± 0.000 | 0.034 ± 0.006 | 0.108 ± 0.188 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 53.452 ± 3.573 | - | 37.991 ± 1.223 | 29.083 ± 2.231 | 28.671 ± 1.658 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.673 ± 0.019 | - | 0.697 ± 0.119 | 0.780 ± 0.092 | 0.922 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.036 ± 0.000 | - | 0.018 ± 0.004 | 0.018 ± 0.004 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.050 ± 0.019 | - | 0.053 ± 0.002 | 0.055 ± 0.002 | 0.103 ± 0.101 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.239 ± 0.009 | - | 0.246 ± 0.018 | 1.153 ± 0.517 | 0.856 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 1.430 ± 0.043 | - | 1.417 ± 0.009 | 1.913 ± 0.419 | 1.427 ± 0.101 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 5.917 ± 0.378 | - | 6.310 ± 0.324 | 9.934 ± 1.073 | 4.190 ± 0.762 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 7.038 ± 0.264 | - | 4.858 ± 0.315 | 3.780 ± 0.726 | 4.383 ± 1.357 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.018 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.091 ± 0.021 | - | 0.095 ± 0.002 | 0.098 ± 0.006 | 0.127 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.002 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.009 ± 0.001 | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.021 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.007 ± 0.001 | - | 0.010 ± 0.002 | 0.012 ± 0.003 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.018 ± 0.000 | - | 0.022 ± 0.000 | 0.023 ± 0.003 | 0.117 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.001 | 0.022 ± 0.001 | 0.043 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.008 ± 0.002 | - | 0.011 ± 0.000 | 0.011 ± 0.003 | 0.021 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.024 ± 0.001 | - | 0.027 ± 0.000 | 0.036 ± 0.158 | 0.078 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 50.848 ± 7.908 | - | 40.437 ± 11.139 | 39.670 ± 7.346 | 16.685 ± 1.236 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.001 | 0.027 ± 0.002 | 0.022 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.699 ± 0.033 | - | 0.748 ± 0.060 | 0.727 ± 0.030 | 0.525 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.013 ± 0.002 | - | 0.017 ± 0.000 | 0.022 ± 0.001 | 0.027 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.047 ± 0.002 | - | 0.053 ± 0.001 | 0.101 ± 0.025 | 0.226 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.220 ± 0.001 | - | 0.224 ± 0.001 | 0.264 ± 0.221 | 0.566 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.148 ± 0.013 | - | 1.233 ± 0.102 | 1.586 ± 0.387 | 1.472 ± 0.195 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 5.425 ± 0.611 | - | 5.708 ± 0.365 | 6.814 ± 2.241 | 5.112 ± 0.633 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 0.873 ± 0.029 | - | 0.865 ± 0.016 | 1.438 ± 0.540 | 1.130 ± 0.068 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.069 ± 0.003 | - | 0.074 ± 0.002 | 0.079 ± 0.011 | 0.144 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.003 | 0.018 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.002 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.015 ± 0.000 | - | 0.018 ± 0.004 | 0.019 ± 0.002 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.017 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.020 ± 0.004 | - | 0.023 ± 0.000 | 0.023 ± 0.001 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 7.755 ± 0.547 | - | 8.133 ± 0.748 | 9.721 ± 5.274 | 8.959 ± 0.755 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.519 ± 0.041 | - | 0.526 ± 0.084 | 0.576 ± 0.335 | 0.557 ± 0.122 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.043 ± 0.002 | 0.029 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.038 ± 0.000 | - | 0.041 ± 0.001 | 0.050 ± 0.010 | 0.081 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.222 ± 0.021 | - | 0.225 ± 0.001 | 0.260 ± 0.034 | 0.937 ± 0.791 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.158 ± 0.042 | - | 1.199 ± 0.121 | 1.714 ± 0.860 | 1.012 ± 0.386 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 4.490 ± 0.083 | - | 4.536 ± 0.319 | 6.237 ± 2.109 | 2.843 ± 0.178 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.022 ± 0.172 | - | 1.151 ± 0.165 | 1.030 ± 0.050 | 1.074 ± 0.128 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.059 ± 0.000 | - | 0.063 ± 0.002 | 0.074 ± 0.151 | 0.125 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.001 | 0.006 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.007 ± 0.001 | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.002 | 0.005 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.006 ± 0.001 | - | 0.009 ± 0.000 | 0.010 ± 0.002 | 0.015 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.015 ± 0.002 | - | 0.018 ± 0.000 | 0.019 ± 0.000 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.002 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.008 ± 0.001 | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.019 ± 0.000 | - | 0.024 ± 0.007 | 0.029 ± 0.002 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 7.985 ± 0.087 | - | 7.892 ± 1.431 | 12.667 ± 12.526 | 4.660 ± 0.422 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.454 ± 0.015 | - | 0.460 ± 0.001 | 0.624 ± 0.317 | 0.377 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.011 ± 0.002 | - | 0.015 ± 0.000 | 0.013 ± 0.000 | 0.026 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.038 ± 0.007 | - | 0.040 ± 0.002 | 0.043 ± 0.007 | 0.206 ± 0.066 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.103 ± 0.001 | - | 0.107 ± 0.000 | 0.129 ± 0.029 | 0.792 ± 0.082 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 0.782 ± 0.103 | - | 0.680 ± 0.006 | 0.765 ± 0.035 | 1.706 ± 0.262 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 2.540 ± 0.349 | - | 2.843 ± 0.585 | 2.484 ± 0.105 | 4.927 ± 0.147 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.457 ± 0.026 | - | 0.451 ± 0.015 | 0.491 ± 0.014 | 0.962 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.001 ± 0.001 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.025 ± 0.000 | - | 0.033 ± 0.006 | 0.032 ± 0.006 | 0.168 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.002 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.008 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.013 ± 0.001 | 0.026 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.011 ± 0.012 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.016 ± 0.003 | 0.033 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 4.006 ± 1.122 | - | 3.905 ± 0.385 | 4.153 ± 0.647 | 9.035 ± 0.447 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.022 ± 0.025 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.211 ± 0.014 | - | 0.217 ± 0.012 | 0.204 ± 0.008 | 0.509 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.018 ± 0.000 | - | 0.020 ± 0.000 | 0.024 ± 0.001 | 0.091 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.163 ± 0.053 | - | 0.184 ± 0.017 | 0.148 ± 0.030 | 0.618 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 0.857 ± 0.058 | - | 0.899 ± 0.145 | 0.816 ± 0.089 | 1.169 ± 0.073 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 2.729 ± 0.142 | - | 2.689 ± 0.453 | 2.757 ± 1.435 | 2.667 ± 0.082 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.558 ± 0.034 | - | 0.557 ± 0.004 | 1.165 ± 0.220 | 1.011 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.003 ± 0.001 | - | 0.007 ± 0.000 | 0.042 ± 0.035 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.030 ± 0.006 | - | 0.033 ± 0.000 | 0.031 ± 0.001 | 0.132 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.003 ± 0.001 | - | 0.018 ± 0.000 | 0.006 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.009 ± 0.003 | 0.032 ± 0.011 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.015 ± 0.000 | 0.012 ± 0.019 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.005 ± 0.010 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.009 ± 0.001 | - | 0.014 ± 0.004 | 0.040 ± 0.050 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.001 | 0.016 ± 0.084 | 0.013 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.011 ± 0.003 | - | 0.017 ± 0.004 | 0.015 ± 0.003 | 0.027 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 4.625 ± 0.921 | - | 5.337 ± 2.577 | 4.842 ± 2.983 | 4.455 ± 0.712 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.008 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.210 ± 0.037 | - | 0.225 ± 0.009 | 0.203 ± 0.016 | 0.342 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.007 ± 0.000 | - | 0.013 ± 0.001 | 0.010 ± 0.000 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.020 ± 0.007 | - | 0.031 ± 0.024 | 0.025 ± 0.013 | 0.062 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.094 ± 0.007 | - | 0.093 ± 0.001 | 0.184 ± 0.028 | 0.149 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.184 ± 0.004 | - | 0.196 ± 0.008 | 0.257 ± 0.001 | 0.273 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.500 ± 0.134 | - | 0.469 ± 0.048 | 0.526 ± 0.065 | 0.412 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.061 ± 0.002 | - | 0.066 ± 0.005 | 0.068 ± 0.002 | 0.145 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.008 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.059 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.011 ± 0.003 | - | 0.014 ± 0.003 | 0.013 ± 0.002 | 0.055 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.001 | 0.008 ± 0.000 | 0.013 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.041 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.001 | 0.009 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.002 | 0.039 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.508 ± 0.028 | - | 0.539 ± 0.234 | 0.450 ± 0.045 | 0.701 ± 0.209 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.032 ± 0.013 | - | 0.042 ± 0.002 | 0.039 ± 0.001 | 0.093 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.004 ± 0.001 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.058 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.003 | 0.041 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.088 ± 0.001 | - | 0.090 ± 0.002 | 0.301 ± 0.024 | 0.168 ± 0.130 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.213 ± 0.015 | - | 0.229 ± 0.010 | 0.307 ± 0.065 | 0.237 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.417 ± 0.007 | - | 0.433 ± 0.025 | 0.546 ± 0.095 | 0.577 ± 0.187 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.059 ± 0.006 | - | 0.064 ± 0.004 | 0.138 ± 0.040 | 0.165 ± 0.067 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.081 ± 0.033 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.007 ± 0.001 | - | 0.009 ± 0.000 | 0.076 ± 0.018 | 0.064 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.001 | 0.074 ± 0.017 | 0.022 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.012 ± 0.003 | - | 0.013 ± 0.003 | 0.049 ± 0.029 | 0.071 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.093 ± 0.073 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.002 | 0.084 ± 0.028 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.066 ± 0.011 | 0.030 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.073 ± 0.056 | 0.067 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.001 | 0.044 ± 0.042 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.001 | 0.062 ± 0.009 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.005 ± 0.001 | - | 0.007 ± 0.002 | 0.070 ± 0.014 | 0.038 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.456 ± 0.046 | - | 0.502 ± 0.009 | 0.548 ± 0.320 | 0.511 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.088 ± 0.043 | 0.022 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.032 ± 0.001 | - | 0.037 ± 0.001 | 0.196 ± 0.046 | 0.122 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.002 | 0.106 ± 0.029 | 0.059 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.006 ± 0.001 | - | 0.008 ± 0.001 | 0.153 ± 0.081 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.198 ± 0.022 | - | 0.221 ± 0.025 | 0.407 ± 0.095 | 0.307 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.553 ± 0.065 | - | 0.597 ± 0.052 | 1.474 ± 0.270 | 0.623 ± 0.118 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.175 ± 0.186 | - | 1.159 ± 0.007 | 2.272 ± 1.179 | 1.602 ± 0.224 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.276 ± 0.001 | - | 0.285 ± 0.009 | 0.369 ± 0.050 | 0.397 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.011 ± 0.000 | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.084 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.011 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.019 ± 0.005 | - | 0.020 ± 0.004 | 0.055 ± 0.011 | 0.132 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.025 ± 0.008 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.005 ± 0.002 | - | 0.008 ± 0.002 | 0.009 ± 0.001 | 0.016 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.004 ± 0.001 | - | 0.007 ± 0.000 | 0.028 ± 0.002 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 2.194 ± 0.033 | - | 2.291 ± 0.345 | 4.801 ± 3.063 | 2.698 ± 0.288 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.002 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.081 ± 0.000 | - | 0.085 ± 0.000 | 0.094 ± 0.022 | 0.440 ± 0.158 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.006 ± 0.001 | - | 0.009 ± 0.001 | 0.034 ± 0.016 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.011 ± 0.003 | - | 0.014 ± 0.001 | 0.015 ± 0.003 | 0.023 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.349 ± 0.317 | - | 0.298 ± 0.077 | 0.423 ± 0.067 | 0.376 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.593 ± 0.157 | - | 0.665 ± 0.294 | 0.778 ± 0.186 | 0.679 ± 0.061 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.800 ± 0.864 | - | 1.709 ± 0.272 | 1.826 ± 1.151 | 0.832 ± 0.074 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.309 ± 0.025 | - | 0.311 ± 0.009 | 0.846 ± 0.779 | 0.418 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.003 | - | 0.010 ± 0.002 | 0.067 ± 0.014 | 0.016 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.019 ± 0.003 | - | 0.029 ± 0.004 | 0.097 ± 0.028 | 0.136 ± 0.083 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.004 ± 0.001 | - | 0.011 ± 0.003 | 0.030 ± 0.005 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.014 ± 0.000 | - | 0.018 ± 0.000 | 0.045 ± 0.041 | 0.089 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.013 ± 0.005 | - | 0.009 ± 0.004 | 0.080 ± 0.029 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.004 ± 0.001 | - | 0.011 ± 0.004 | 0.061 ± 0.019 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.065 ± 0.031 | 0.039 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.009 ± 0.002 | - | 0.012 ± 0.005 | 0.071 ± 0.025 | 0.017 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.004 ± 0.002 | - | 0.009 ± 0.008 | 0.074 ± 0.026 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.004 ± 0.001 | - | 0.008 ± 0.002 | 0.046 ± 0.056 | 0.014 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.010 ± 0.003 | - | 0.010 ± 0.000 | 0.066 ± 0.005 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 2.740 ± 0.499 | - | 2.305 ± 0.048 | 3.261 ± 1.430 | 2.233 ± 0.827 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.004 ± 0.001 | - | 0.008 ± 0.002 | 0.050 ± 0.056 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.146 ± 0.083 | - | 0.142 ± 0.024 | 0.197 ± 0.013 | 0.153 ± 0.139 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.008 ± 0.001 | - | 0.009 ± 0.002 | 0.039 ± 0.029 | 0.042 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.019 ± 0.070 | - | 0.053 ± 0.008 | 0.086 ± 0.051 | 0.022 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.131 ± 0.000 | - | 0.127 ± 0.011 | 0.159 ± 0.224 | 0.195 ± 0.079 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.192 ± 0.017 | - | 0.172 ± 0.001 | 0.182 ± 0.100 | 0.255 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.349 ± 0.038 | - | 0.408 ± 0.057 | 0.498 ± 0.147 | 0.511 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.047 ± 0.000 | - | 0.054 ± 0.000 | 0.069 ± 0.173 | 0.103 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.006 ± 0.006 | 0.011 ± 0.003 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.035 ± 0.024 | 0.045 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.057 ± 0.053 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.008 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.008 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.035 ± 0.003 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.013 ± 0.003 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.361 ± 0.020 | - | 0.358 ± 0.080 | 1.035 ± 0.456 | 0.666 ± 0.080 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.003 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.029 ± 0.005 | - | 0.034 ± 0.000 | 0.095 ± 0.040 | 0.092 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.004 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.131 ± 0.006 | - | 0.115 ± 0.004 | 0.159 ± 0.001 | 0.218 ± 0.078 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.203 ± 0.023 | - | 0.182 ± 0.012 | 0.228 ± 0.004 | 0.256 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.347 ± 0.035 | - | 0.316 ± 0.004 | 0.415 ± 0.049 | 0.672 ± 0.344 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.044 ± 0.001 | - | 0.053 ± 0.001 | 0.174 ± 0.141 | 0.131 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.009 ± 0.007 | 0.010 ± 0.001 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.006 ± 0.001 | - | 0.010 ± 0.000 | 0.013 ± 0.002 | 0.063 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.030 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.002 | 0.010 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.012 ± 0.002 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.001 | 0.010 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.339 ± 0.009 | - | 0.508 ± 0.028 | 0.523 ± 0.066 | 0.551 ± 0.171 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.066 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.026 ± 0.001 | - | 0.032 ± 0.000 | 0.135 ± 0.068 | 0.165 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.066 ± 0.002 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.005 ± 0.001 | - | 0.010 ± 0.000 | 0.038 ± 0.004 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.741 ± 0.772 | - | 1.427 ± 0.169 | 1.467 ± 0.490 | 3.372 ± 0.509 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 4.563 ± 0.832 | - | 4.895 ± 2.854 | 4.770 ± 0.363 | 5.884 ± 0.345 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 17.897 ± 3.839 | - | 16.244 ± 10.188 | 15.447 ± 1.195 | 16.932 ± 0.675 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 5.321 ± 2.200 | - | 5.950 ± 0.425 | 4.745 ± 0.507 | 5.496 ± 0.744 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.005 ± 0.003 | - | 0.006 ± 0.000 | 0.021 ± 0.008 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.203 ± 0.016 | - | 0.198 ± 0.012 | 0.193 ± 0.011 | 0.582 ± 0.407 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.003 | 0.012 ± 0.002 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.164 ± 0.003 | - | 0.167 ± 0.003 | 0.174 ± 0.016 | 0.272 ± 0.436 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.020 ± 0.004 | - | 0.020 ± 0.000 | 0.026 ± 0.004 | 0.034 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.021 ± 0.005 | - | 0.008 ± 0.000 | 0.038 ± 0.040 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.014 ± 0.000 | - | 0.017 ± 0.004 | 0.022 ± 0.003 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.046 ± 0.001 | - | 0.052 ± 0.006 | 0.050 ± 0.001 | 0.060 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.007 ± 0.001 | - | 0.010 ± 0.002 | 0.033 ± 0.033 | 0.026 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.018 ± 0.000 | - | 0.022 ± 0.000 | 0.025 ± 0.004 | 0.037 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.065 ± 0.008 | - | 0.070 ± 0.009 | 0.069 ± 0.010 | 0.086 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 48.058 ± 11.559 | - | 46.791 ± 2.007 | 36.851 ± 1.824 | 40.650 ± 5.538 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.012 ± 0.002 | - | 0.015 ± 0.001 | 0.025 ± 0.020 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.707 ± 0.576 | - | 1.434 ± 0.039 | 1.515 ± 0.058 | 1.766 ± 0.184 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.035 ± 0.001 | - | 0.041 ± 0.007 | 0.044 ± 0.007 | 0.057 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.130 ± 0.014 | - | 0.151 ± 0.045 | 0.131 ± 0.015 | 0.179 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.254 ± 0.073 | - | 1.226 ± 0.010 | 1.761 ± 0.323 | 2.774 ± 0.096 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 4.589 ± 0.224 | - | 4.596 ± 0.200 | 6.964 ± 3.921 | 2.615 ± 0.498 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 15.490 ± 0.535 | - | 16.221 ± 1.486 | 16.832 ± 10.286 | 15.966 ± 8.474 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 4.872 ± 0.141 | - | 6.236 ± 0.418 | 4.836 ± 0.183 | 6.606 ± 1.695 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.008 ± 0.001 | 0.016 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.164 ± 0.003 | - | 0.166 ± 0.002 | 0.204 ± 0.015 | 0.291 ± 0.153 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.007 ± 0.002 | - | 0.010 ± 0.000 | 0.014 ± 0.001 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.169 ± 0.002 | - | 0.173 ± 0.001 | 0.179 ± 0.012 | 0.265 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.018 ± 0.003 | - | 0.022 ± 0.005 | 0.025 ± 0.021 | 0.040 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.009 ± 0.001 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.018 ± 0.003 | - | 0.019 ± 0.001 | 0.019 ± 0.003 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.046 ± 0.007 | - | 0.053 ± 0.011 | 0.051 ± 0.029 | 0.108 ± 0.133 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.009 ± 0.002 | - | 0.013 ± 0.002 | 0.011 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.020 ± 0.004 | - | 0.025 ± 0.005 | 0.022 ± 0.004 | 0.045 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.063 ± 0.009 | - | 0.069 ± 0.007 | 0.077 ± 0.007 | 0.082 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 38.663 ± 3.260 | - | 53.276 ± 5.175 | 37.325 ± 5.803 | 38.158 ± 8.632 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.012 ± 0.003 | - | 0.015 ± 0.004 | 0.015 ± 0.001 | 0.034 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 1.409 ± 0.055 | - | 1.440 ± 0.026 | 1.615 ± 1.832 | 1.302 ± 0.289 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.036 ± 0.004 | - | 0.039 ± 0.004 | 0.046 ± 0.009 | 0.101 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.118 ± 0.010 | - | 0.130 ± 0.011 | 0.144 ± 0.009 | 0.246 ± 0.065 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.624 ± 0.102 | - | 0.610 ± 0.026 | 0.800 ± 0.478 | 2.091 ± 0.159 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 2.896 ± 0.143 | - | 3.093 ± 0.581 | 3.617 ± 0.959 | 4.503 ± 0.528 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 11.287 ± 0.506 | - | 11.028 ± 0.643 | 11.985 ± 0.998 | 13.513 ± 0.890 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.572 ± 0.027 | - | 1.668 ± 0.141 | 1.564 ± 0.075 | 2.190 ± 1.920 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.003 ± 0.001 | - | 0.006 ± 0.001 | 0.008 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.122 ± 0.004 | - | 0.150 ± 0.040 | 0.126 ± 0.001 | 0.236 ± 0.096 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.005 ± 0.000 | - | 0.009 ± 0.002 | 0.011 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.167 ± 0.064 | - | 0.179 ± 0.206 | 0.132 ± 0.045 | 0.186 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.013 ± 0.003 | - | 0.015 ± 0.004 | 0.019 ± 0.003 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.020 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.010 ± 0.045 | - | 0.014 ± 0.003 | 0.015 ± 0.002 | 0.023 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.029 ± 0.005 | - | 0.037 ± 0.006 | 0.038 ± 0.008 | 0.054 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.014 ± 0.006 | - | 0.018 ± 0.002 | 0.023 ± 0.005 | 0.034 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.108 ± 0.002 | - | 0.043 ± 0.000 | 0.049 ± 0.005 | 0.075 ± 0.053 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 13.377 ± 0.584 | - | 14.754 ± 1.624 | 13.572 ± 1.042 | 13.721 ± 0.829 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.007 ± 0.002 | - | 0.009 ± 0.000 | 0.035 ± 0.023 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 0.998 ± 0.430 | - | 1.001 ± 0.024 | 1.009 ± 0.063 | 1.335 ± 0.471 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.024 ± 0.005 | - | 0.028 ± 0.006 | 0.030 ± 0.004 | 0.043 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.080 ± 0.003 | - | 0.081 ± 0.000 | 0.099 ± 0.008 | 0.145 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.744 ± 0.011 | - | 0.777 ± 0.017 | 0.602 ± 0.431 | 1.936 ± 0.117 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 2.939 ± 0.195 | - | 2.600 ± 0.541 | 3.135 ± 0.476 | 1.786 ± 0.160 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 9.705 ± 0.245 | - | 10.702 ± 0.217 | 13.331 ± 2.701 | 6.697 ± 0.238 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 1.569 ± 0.038 | - | 1.720 ± 0.321 | 1.655 ± 0.163 | 1.655 ± 0.070 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.010 ± 0.002 | - | 0.012 ± 0.002 | 0.037 ± 0.011 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.117 ± 0.002 | - | 0.127 ± 0.007 | 0.134 ± 0.381 | 0.259 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.011 ± 0.000 | 0.009 ± 0.001 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.099 ± 0.005 | - | 0.103 ± 0.010 | 0.107 ± 0.019 | 0.158 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.012 ± 0.000 | - | 0.016 ± 0.001 | 0.017 ± 0.004 | 0.028 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.007 ± 0.001 | - | 0.024 ± 0.000 | 0.068 ± 0.031 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.012 ± 0.003 | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.028 ± 0.001 | - | 0.032 ± 0.007 | 0.034 ± 0.001 | 0.045 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.007 ± 0.000 | - | 0.025 ± 0.000 | 0.026 ± 0.001 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.014 ± 0.000 | - | 0.016 ± 0.005 | 0.017 ± 0.000 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.039 ± 0.001 | - | 0.043 ± 0.006 | 0.045 ± 0.001 | 0.068 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 10.710 ± 0.638 | - | 10.937 ± 1.436 | 14.959 ± 4.665 | 6.159 ± 0.180 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.009 ± 0.000 | - | 0.015 ± 0.003 | 0.031 ± 0.001 | 0.026 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 0.939 ± 0.023 | - | 0.977 ± 0.033 | 1.089 ± 0.405 | 0.603 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.024 ± 0.005 | - | 0.027 ± 0.000 | 0.027 ± 0.001 | 0.043 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.076 ± 0.008 | - | 0.077 ± 0.000 | 0.233 ± 0.022 | 0.171 ± 0.078 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.209 ± 0.017 | - | unsupported | 0.228 ± 0.028 | 1.385 ± 0.093 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.446 ± 0.041 | - | unsupported | 1.509 ± 0.193 | 2.478 ± 0.360 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 4.828 ± 0.841 | - | unsupported | 4.117 ± 0.046 | 5.123 ± 0.674 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.782 ± 0.068 | - | unsupported | 0.884 ± 0.061 | 1.044 ± 0.102 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | unsupported | 0.005 ± 0.001 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.045 ± 0.006 | - | unsupported | 0.040 ± 0.000 | 0.046 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.029 ± 0.001 | - | unsupported | 0.038 ± 0.005 | 0.041 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | unsupported | 0.009 ± 0.002 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.007 ± 0.001 | 0.014 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.005 ± 0.001 | - | unsupported | 0.009 ± 0.000 | 0.020 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.012 ± 0.000 | - | unsupported | 0.018 ± 0.003 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.016 ± 0.000 | - | unsupported | 0.025 ± 0.003 | 0.033 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 6.983 ± 0.974 | - | unsupported | 6.524 ± 0.428 | 5.899 ± 0.703 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.002 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.344 ± 0.077 | - | unsupported | 0.281 ± 0.004 | 0.290 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.012 ± 0.002 | - | unsupported | 0.015 ± 0.003 | 0.075 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.030 ± 0.000 | - | unsupported | 0.040 ± 0.003 | 0.044 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.210 ± 0.010 | - | unsupported | 0.258 ± 0.060 | 1.815 ± 0.457 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 1.519 ± 0.180 | - | unsupported | 1.380 ± 0.054 | 1.659 ± 0.676 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 4.469 ± 0.660 | - | unsupported | 3.958 ± 0.325 | 2.619 ± 0.072 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.018 ± 0.269 | - | unsupported | 0.708 ± 0.010 | 0.799 ± 0.073 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.002 ± 0.001 | - | unsupported | 0.006 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.039 ± 0.004 | - | unsupported | 0.044 ± 0.009 | 0.074 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.008 ± 0.001 | 0.019 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.030 ± 0.007 | - | unsupported | 0.033 ± 0.001 | 0.110 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.006 ± 0.000 | - | unsupported | 0.011 ± 0.002 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.006 ± 0.000 | - | unsupported | 0.010 ± 0.002 | 0.037 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.014 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.029 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.022 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.007 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.051 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.018 ± 0.001 | - | unsupported | 0.019 ± 0.000 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 6.549 ± 0.184 | - | unsupported | 6.819 ± 0.721 | 3.806 ± 0.778 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.676 ± 0.133 | - | unsupported | 0.271 ± 0.000 | 0.233 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.012 ± 0.000 | - | unsupported | 0.015 ± 0.000 | 0.035 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.033 ± 0.000 | - | unsupported | 0.033 ± 0.000 | 0.044 ± 0.001 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.885 ± 0.041 | - | unsupported | 1.065 ± 0.215 | 0.881 ± 0.028 | 2.850 ± 1.170 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.059 ± 0.004 | - | unsupported | 0.085 ± 0.019 | 0.133 ± 0.015 | 0.082 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.128 ± 0.009 | - | unsupported | 0.130 ± 0.008 | 0.198 ± 0.026 | 0.140 ± 0.016 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.316 ± 0.019 | - | unsupported | 0.375 ± 0.075 | 0.383 ± 0.027 | 0.690 ± 0.081 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 1.096 ± 0.624 | - | unsupported | 0.713 ± 0.016 | 1.226 ± 0.414 | 3.364 ± 0.526 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.056 ± 0.002 | - | unsupported | 0.133 ± 0.012 | 0.182 ± 0.048 | 0.076 ± 0.014 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.109 ± 0.000 | - | unsupported | 0.179 ± 0.018 | 0.207 ± 0.012 | 0.216 ± 0.014 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.254 ± 0.049 | - | unsupported | 0.324 ± 0.030 | 0.474 ± 0.359 | 0.795 ± 0.065 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.227 ± 0.027 | - | unsupported | 0.297 ± 0.059 | 0.516 ± 0.034 | 0.571 ± 0.055 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.067 ± 0.007 | - | unsupported | 0.055 ± 0.005 | 0.080 ± 0.003 | 0.069 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.094 ± 0.011 | - | unsupported | 0.072 ± 0.131 | 0.109 ± 0.008 | 0.108 ± 0.011 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.105 ± 0.011 | - | unsupported | 0.120 ± 0.035 | 0.155 ± 0.013 | 0.263 ± 0.026 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.261 ± 0.057 | - | unsupported | 0.199 ± 0.018 | 0.536 ± 0.083 | 0.535 ± 0.101 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.122 ± 0.113 | - | unsupported | 0.044 ± 0.002 | 0.199 ± 0.080 | 0.074 ± 0.010 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.083 ± 0.004 | - | unsupported | 0.051 ± 0.003 | 0.087 ± 0.009 | 0.099 ± 0.005 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.128 ± 0.070 | - | unsupported | 0.094 ± 0.033 | 0.211 ± 0.047 | 0.239 ± 0.006 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.135 ± 0.039 | - | 0.220 ± 0.065 | 0.086 ± 0.011 | 0.189 ± 0.016 | 0.340 ± 0.032 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.046 ± 0.000 | - | 0.059 ± 0.016 | 0.040 ± 0.018 | 0.025 ± 0.006 | 0.050 ± 0.003 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.044 ± 0.007 | - | 0.052 ± 0.000 | 0.029 ± 0.004 | 0.034 ± 0.006 | 0.055 ± 0.003 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.056 ± 0.000 | - | 0.074 ± 0.001 | 0.038 ± 0.006 | 0.064 ± 0.007 | 0.150 ± 0.018 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.103 ± 0.001 | - | 0.202 ± 0.051 | 0.094 ± 0.034 | 0.216 ± 0.012 | 0.255 ± 0.021 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.151 ± 0.195 | - | 0.060 ± 0.011 | 0.033 ± 0.028 | 0.025 ± 0.001 | 0.041 ± 0.007 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.052 ± 0.009 | - | 0.053 ± 0.000 | 0.027 ± 0.011 | 0.031 ± 0.002 | 0.048 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.089 ± 0.015 | - | 0.109 ± 0.061 | 0.041 ± 0.014 | 0.098 ± 0.045 | 0.097 ± 0.000 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 10.876 ± 5.598 | - | 15.159 ± 3.350 | 26.335 ± 6.382 | 18.035 ± 4.388 | 39.873 ± 8.787 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 16.451 ± 2.664 | - | 22.211 ± 11.187 | 17.528 ± 1.056 | 13.686 ± 0.759 | 348.422 ± 94.257 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 8.968 ± 0.560 | - | 7.519 ± 0.779 | 6.711 ± 1.244 | 9.296 ± 2.695 | 19.516 ± 8.839 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 9.835 ± 2.050 | - | 8.379 ± 1.230 | 6.665 ± 1.205 | 7.346 ± 0.426 | 246.081 ± 38.641 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 6.971 ± 0.738 | - | 7.136 ± 0.614 | 6.733 ± 0.987 | 8.641 ± 3.562 | 8.133 ± 1.606 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 8.021 ± 2.282 | - | 7.998 ± 1.847 | 6.982 ± 0.457 | 9.418 ± 3.712 | 294.197 ± 18.545 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 7.973 ± 0.862 | - | 7.704 ± 1.331 | 8.206 ± 0.735 | 7.417 ± 0.574 | 10.739 ± 2.608 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 8.665 ± 1.646 | - | 8.754 ± 0.899 | 8.131 ± 0.485 | 7.900 ± 0.782 | 250.631 ± 6.591 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 21.255 ± 2.009 | - | 21.848 ± 2.251 | 22.354 ± 1.649 | 24.022 ± 4.820 | 12.341 ± 1.435 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 21.628 ± 1.803 | - | 27.146 ± 13.054 | 21.780 ± 1.248 | 26.435 ± 3.749 | 834.916 ± 671.669 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.296 ± 0.053 | 0.304 ± 0.008 | unsupported | 0.304 ± 0.031 | 0.377 ± 0.045 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 3.681 ± 0.669 | 3.859 ± 0.429 | unsupported | 3.798 ± 0.445 | 5.303 ± 0.275 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.366 ± 0.073 | 0.323 ± 0.062 | unsupported | 0.324 ± 0.039 | 0.429 ± 0.024 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 3.986 ± 1.212 | 3.387 ± 0.431 | unsupported | 3.613 ± 0.329 | 4.761 ± 0.108 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 8.814 ± 0.505 | - | 8.331 ± 1.120 | 8.295 ± 1.864 | 12.051 ± 0.742 | 30.176 ± 3.597 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 8.732 ± 0.323 | - | 9.522 ± 2.114 | 10.032 ± 1.079 | 9.920 ± 1.111 | 408.992 ± 431.103 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 8.158 ± 0.457 | 8.498 ± 2.008 | 5.457 ± 0.320 | 25.376 ± 1.060 | 9.604 ± 2.226 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 11.301 ± 1.322 | 8.301 ± 0.575 | 5.052 ± 0.151 | 27.720 ± 2.676 | 141.532 ± 31.002 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 8.266 ± 0.562 | - | 8.744 ± 0.464 | 13.650 ± 1.424 | 11.002 ± 2.689 | 42.135 ± 27.818 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 6.906 ± 0.364 | - | 7.201 ± 0.958 | 10.903 ± 1.923 | 11.731 ± 6.202 | 227.964 ± 46.365 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.654 ± 0.104 | - | 0.797 ± 0.035 | 2.352 ± 0.380 | 6.891 ± 0.457 | 6.695 ± 0.161 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.840 ± 0.174 | - | 0.861 ± 0.093 | 2.489 ± 0.094 | 4.779 ± 1.166 | 8.915 ± 1.913 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 10.774 ± 0.282 | - | 10.164 ± 0.265 | 10.059 ± 1.398 | 13.603 ± 1.193 | 21.505 ± 4.015 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 11.518 ± 1.480 | - | 10.767 ± 0.648 | 12.323 ± 3.402 | 13.912 ± 4.070 | 1057.507 ± 180.901 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 10.218 ± 0.742 | - | 10.113 ± 0.713 | 9.674 ± 0.312 | 14.057 ± 0.712 | 23.483 ± 1.135 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 11.021 ± 0.338 | - | 10.299 ± 1.197 | 11.717 ± 0.872 | 15.693 ± 2.884 | 1002.548 ± 506.270 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.468 ± 0.130 | 1.490 ± 0.208 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 13.606 ± 0.676 | 17.441 ± 2.913 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.538 ± 0.159 | 1.427 ± 0.204 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 14.730 ± 0.854 | 15.596 ± 2.211 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.493 ± 0.076 | 1.488 ± 0.096 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 12.107 ± 0.758 | 12.495 ± 0.992 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.464 ± 0.048 | 1.428 ± 0.173 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 17.442 ± 5.209 | 12.454 ± 0.518 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 9.412 ± 3.966 | - | 8.282 ± 1.490 | 7.671 ± 1.096 | 9.074 ± 0.758 | 22.214 ± 2.497 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 10.427 ± 0.950 | - | 8.416 ± 2.573 | 7.207 ± 0.677 | 7.689 ± 0.525 | 217.482 ± 16.580 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 38.221 ± 8.833 | 36.705 ± 2.520 | 35.134 ± 2.453 | 35.469 ± 1.945 | 74.908 ± 15.943 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 36.083 ± 7.700 | 37.340 ± 4.770 | 37.380 ± 2.931 | 52.513 ± 11.016 | 1723.504 ± 1007.146 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 15.098 ± 0.279 | - | 28.489 ± 4.101 | 12.686 ± 1.498 | 72.717 ± 4.642 | 27.942 ± 8.178 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 18.398 ± 4.026 | - | 29.499 ± 14.117 | 12.832 ± 0.996 | 70.361 ± 13.903 | 25.770 ± 4.350 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.236 ± 0.076 | 0.166 ± 0.022 | unsupported | 0.063 ± 0.030 | 0.054 ± 0.028 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 4.332 ± 1.633 | 3.868 ± 1.967 | unsupported | 0.815 ± 0.016 | 0.514 ± 0.021 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.252 ± 0.138 | 0.188 ± 0.027 | unsupported | 0.120 ± 0.034 | 0.086 ± 0.050 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 1.370 ± 0.031 | 1.800 ± 0.126 | unsupported | 0.497 ± 0.160 | 0.396 ± 0.360 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.074 ± 0.003 | 0.087 ± 0.005 | unsupported | 0.091 ± 0.002 | 0.209 ± 0.044 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.638 ± 0.523 | 1.412 ± 0.047 | unsupported | 1.436 ± 0.082 | 2.198 ± 0.534 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.126 ± 0.013 | 0.069 ± 0.061 | unsupported | 0.087 ± 0.020 | 0.161 ± 0.064 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.817 ± 0.153 | 0.683 ± 0.042 | unsupported | 1.259 ± 0.590 | 1.454 ± 0.401 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.290 ± 0.007 | 0.288 ± 0.021 | unsupported | 0.307 ± 0.006 | 0.100 ± 0.017 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 4.516 ± 0.505 | 4.898 ± 0.733 | unsupported | 4.838 ± 0.108 | 0.917 ± 0.035 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.196 ± 0.097 | 0.245 ± 0.078 | unsupported | 0.204 ± 0.069 | 0.070 ± 0.015 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 2.423 ± 0.317 | 2.382 ± 0.084 | unsupported | 3.168 ± 0.415 | 0.540 ± 0.060 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.052 ± 0.004 | 0.074 ± 0.008 | unsupported | 0.048 ± 0.003 | 0.083 ± 0.006 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.647 ± 0.105 | 0.565 ± 0.049 | unsupported | 0.616 ± 0.084 | 0.743 ± 0.124 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.054 ± 0.001 | 0.082 ± 0.015 | unsupported | 0.089 ± 0.040 | 0.076 ± 0.057 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.869 ± 0.068 | 1.477 ± 0.849 | unsupported | 0.504 ± 0.065 | 0.635 ± 0.326 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.094 ± 0.006 | 0.111 ± 0.003 | unsupported | 0.092 ± 0.006 | 0.118 ± 0.009 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.347 ± 0.117 | 1.494 ± 0.179 | unsupported | 1.477 ± 0.171 | 1.368 ± 0.078 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.111 ± 0.024 | 0.120 ± 0.004 | unsupported | 0.092 ± 0.021 | 0.134 ± 0.031 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 1.260 ± 0.098 | 1.700 ± 0.377 | unsupported | 0.977 ± 0.288 | 0.754 ± 0.074 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.130 ± 0.007 | 0.167 ± 0.074 | unsupported | unsupported | 0.142 ± 0.013 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 1.947 ± 0.074 | 2.043 ± 0.196 | unsupported | unsupported | 1.744 ± 0.424 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.139 ± 0.006 | 0.163 ± 0.002 | unsupported | unsupported | 0.126 ± 0.020 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 1.498 ± 0.113 | 1.842 ± 0.516 | unsupported | unsupported | 1.122 ± 0.236 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.013 ± 0.000 | 0.024 ± 0.009 | unsupported | 0.008 ± 0.002 | 0.069 ± 0.050 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.117 ± 0.004 | 0.124 ± 0.024 | unsupported | 0.064 ± 0.013 | 0.265 ± 0.024 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.013 ± 0.000 | 0.019 ± 0.000 | unsupported | 0.032 ± 0.005 | 0.065 ± 0.033 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.111 ± 0.015 | 0.110 ± 0.015 | unsupported | 0.075 ± 0.011 | 0.149 ± 0.006 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.025 ± 0.002 | 0.041 ± 0.003 | unsupported | 0.029 ± 0.003 | 0.075 ± 0.004 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.285 ± 0.011 | 0.311 ± 0.068 | unsupported | 0.389 ± 0.099 | 0.737 ± 0.264 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.028 ± 0.002 | 0.044 ± 0.004 | unsupported | 0.107 ± 0.024 | 0.097 ± 0.023 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.604 ± 0.123 | 0.739 ± 0.156 | unsupported | 0.437 ± 0.139 | 1.172 ± 0.999 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.046 ± 0.011 | 0.056 ± 0.009 | unsupported | 0.076 ± 0.004 | 0.106 ± 0.009 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.889 ± 0.062 | 0.898 ± 0.229 | unsupported | 1.331 ± 0.241 | 1.019 ± 0.057 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.091 ± 0.011 | 0.076 ± 0.060 | unsupported | 0.077 ± 0.011 | 0.078 ± 0.025 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.654 ± 0.179 | 0.666 ± 0.154 | unsupported | 0.734 ± 0.094 | 0.733 ± 0.352 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.058 ± 0.003 | 0.059 ± 0.007 | unsupported | 0.087 ± 0.012 | 0.102 ± 0.019 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.918 ± 0.117 | 0.946 ± 0.303 | unsupported | 1.224 ± 0.136 | 0.956 ± 0.012 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.091 ± 0.021 | 0.030 ± 0.017 | unsupported | 0.078 ± 0.009 | 0.093 ± 0.012 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.657 ± 0.242 | 0.558 ± 0.091 | unsupported | 0.820 ± 0.369 | 0.633 ± 0.234 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.096 ± 0.015 | 0.110 ± 0.008 | unsupported | 0.102 ± 0.005 | 0.186 ± 0.055 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.249 ± 0.115 | 1.398 ± 0.167 | unsupported | 1.705 ± 0.028 | 1.440 ± 0.327 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.102 ± 0.006 | 0.136 ± 0.079 | unsupported | 0.081 ± 0.026 | 0.124 ± 0.010 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 1.001 ± 0.259 | 1.267 ± 0.098 | unsupported | 1.275 ± 0.211 | 0.768 ± 0.092 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.255 ± 0.027 | 0.225 ± 0.016 | unsupported | 0.290 ± 0.006 | 0.314 ± 0.079 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 4.908 ± 0.868 | 5.171 ± 1.389 | unsupported | 4.694 ± 0.655 | 4.495 ± 0.435 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.171 ± 0.072 | 0.209 ± 0.083 | unsupported | 0.223 ± 0.140 | 0.354 ± 0.093 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 2.140 ± 0.175 | 1.921 ± 0.082 | unsupported | 2.516 ± 0.307 | 3.156 ± 1.128 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.588 ± 0.057 | 0.786 ± 0.204 | unsupported | 0.124 ± 0.003 | 0.103 ± 0.013 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 9.966 ± 0.278 | 11.152 ± 1.531 | unsupported | 2.474 ± 0.252 | 1.803 ± 0.270 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.739 ± 0.060 | 0.911 ± 0.071 | unsupported | 0.198 ± 0.064 | 0.156 ± 0.015 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 10.141 ± 2.101 | 10.707 ± 3.121 | unsupported | 1.832 ± 0.227 | 1.115 ± 0.317 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 30.723 ± 6.240 | - | unsupported | 18.877 ± 1.823 | - | 19.206 ± 2.472 | 18.594 ± 2.440 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 19.220 ± 1.969 | - | unsupported | 20.410 ± 1.355 | - | 24.056 ± 2.787 | 22.604 ± 5.961 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 15.644 ± 6.386 | - | unsupported | 23.410 ± 11.049 | - | 12.141 ± 1.787 | 12.858 ± 1.352 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 16.702 ± 3.720 | - | unsupported | 13.005 ± 0.921 | - | 13.529 ± 1.337 | 13.994 ± 1.831 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 15.631 ± 3.451 | - | unsupported | 18.732 ± 7.008 | - | 10.299 ± 0.817 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 14.919 ± 4.657 | - | unsupported | 13.088 ± 2.260 | - | 13.532 ± 6.456 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 27.912 ± 4.683 | - | unsupported | 22.984 ± 6.460 | - | 18.449 ± 3.118 | 22.076 ± 8.276 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 20.110 ± 1.802 | - | unsupported | 22.769 ± 4.019 | - | 19.240 ± 1.634 | 17.772 ± 1.094 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 14.876 ± 7.386 | - | unsupported | 14.003 ± 2.596 | - | 50.384 ± 3.776 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 11.840 ± 1.477 | - | unsupported | 12.247 ± 1.536 | - | 53.023 ± 5.578 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 12.653 ± 1.812 | - | unsupported | 16.218 ± 5.035 | - | 53.032 ± 1.124 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 15.229 ± 2.755 | - | unsupported | 15.253 ± 4.114 | - | 59.071 ± 17.729 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 34.212 ± 11.309 | - | unsupported | 19.767 ± 2.514 | - | 20.020 ± 1.390 | 23.133 ± 2.927 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 20.807 ± 2.416 | - | unsupported | 20.533 ± 2.512 | - | 23.417 ± 4.749 | 18.064 ± 6.707 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 17.939 ± 2.395 | - | unsupported | 12.085 ± 0.766 | - | 12.002 ± 0.635 | 12.946 ± 2.085 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 12.411 ± 1.013 | - | unsupported | 13.435 ± 2.616 | - | 12.145 ± 2.149 | 13.019 ± 1.281 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.249 ± 0.023 | unsupported | unsupported | 0.250 ± 0.031 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 3.451 ± 0.259 | unsupported | unsupported | 3.482 ± 0.354 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.210 ± 0.014 | unsupported | unsupported | 0.361 ± 0.047 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 2.804 ± 0.485 | unsupported | unsupported | 3.217 ± 0.109 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 30.088 ± 9.796 | - | unsupported | 18.933 ± 2.156 | - | 19.285 ± 1.080 | 20.027 ± 2.394 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 22.166 ± 1.788 | - | unsupported | 19.279 ± 4.335 | - | 24.862 ± 3.113 | 21.033 ± 1.264 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 8.667 ± 2.163 | - | 0.002 ± 0.000 | 5.659 ± 0.628 | 5.671 ± 0.457 | 16.219 ± 3.749 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 13.033 ± 4.861 | - | 0.002 ± 0.000 | 6.232 ± 0.720 | 5.242 ± 0.325 | 15.466 ± 1.175 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 8.450 ± 0.326 | - | 28.782 ± 7.293 | 10.179 ± 0.770 | 9.749 ± 1.216 | 9.304 ± 1.511 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 10.508 ± 2.027 | - | 21.195 ± 5.868 | 10.623 ± 0.711 | 9.941 ± 1.956 | 17.185 ± 6.804 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 13.548 ± 2.598 | - | 17.084 ± 10.940 | 13.454 ± 0.915 | 17.179 ± 0.876 | 11.702 ± 0.647 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 12.681 ± 1.692 | - | 10.786 ± 0.999 | 12.555 ± 0.840 | 12.925 ± 4.100 | 17.730 ± 5.786 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 7.476 ± 1.273 | - | 8.991 ± 1.821 | 10.844 ± 1.207 | 10.109 ± 1.107 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 5.886 ± 0.532 | - | 5.545 ± 0.183 | 11.836 ± 2.026 | 8.897 ± 0.646 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 15.387 ± 3.222 | - | 0.002 ± 0.000 | 11.526 ± 0.236 | 12.591 ± 1.039 | 13.546 ± 3.282 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 14.312 ± 3.160 | - | 0.017 ± 0.006 | 14.818 ± 3.564 | 9.841 ± 0.500 | 22.280 ± 5.571 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 43.501 ± 3.744 | - | 0.003 ± 0.011 | 46.027 ± 3.032 | 67.198 ± 12.773 | 230.907 ± 41.815 | 31.036 ± 4.602 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 28.343 ± 2.476 | - | 0.006 ± 0.014 | 48.062 ± 8.074 | 30.906 ± 12.162 | 206.789 ± 43.981 | 30.013 ± 13.029 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 7.473 ± 1.923 | - | 7.228 ± 0.698 | 10.168 ± 1.110 | 7.039 ± 1.297 | 10.187 ± 2.498 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 6.179 ± 0.370 | - | 6.207 ± 1.407 | 6.632 ± 0.770 | 6.113 ± 1.345 | 10.763 ± 3.518 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 6.991 ± 0.285 | - | 8.004 ± 0.959 | 8.425 ± 1.315 | 5.731 ± 0.566 | 10.777 ± 1.614 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 5.374 ± 0.756 | - | 5.782 ± 0.582 | 6.215 ± 1.185 | 5.408 ± 0.342 | 10.522 ± 2.210 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.002 ± 0.003 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.005 ± 0.007 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 4.883958 | 305247.38 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 4.449417 | 278088.56 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 4.165209 | 260325.56 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 3.231166 | 201947.88 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 6.349417 | 396838.56 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 5.954916 | 372182.25 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 22.092750 | 1380796.88 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 18.645959 | 1165372.44 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 14.391667 | 899479.19 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 19.117416 | 1194838.50 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 6.034209 | 377138.06 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 4.862084 | 303880.25 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 4.732167 | 295760.44 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 4.866000 | 304125.00 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 84.845666 | 5302854.12 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 60.768750 | 3798046.88 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 58.894417 | 3680901.06 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 61.741416 | 3858838.50 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 6.860541 | 428783.81 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 5.185125 | 324070.31 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 5.850292 | 365643.25 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 5.163041 | 322690.06 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 76.174958 | 4760934.88 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 57.801792 | 3612612.00 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 63.775417 | 3985963.56 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 54.185042 | 3386565.12 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 23.491375 | 1468210.94 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 23.842500 | 1490156.25 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 217.695750 | 13605984.38 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 279.060500 | 17441281.25 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 24.601666 | 1537604.12 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 22.826834 | 1426677.12 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 235.686791 | 14730424.44 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 249.533667 | 15595854.19 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 23.888416 | 1493026.00 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 23.815625 | 1488476.56 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 193.710667 | 12106916.69 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 199.923375 | 12495210.94 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 23.430958 | 1464434.88 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 22.843000 | 1427687.50 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 279.077417 | 17442338.56 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 199.264375 | 12454023.44 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 3.484959 | 54452.48 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.015291 | 62738.92 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 15.134334 | 236473.97 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 10.635167 | 166174.48 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 4.109000 | 513625.00 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 6.518458 | 814807.25 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 34.657791 | 4332223.88 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 30.945334 | 3868166.75 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.533416 | 86459.62 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 7.666084 | 119782.56 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 16.100667 | 251572.92 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 12.019667 | 187807.30 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.166958 | 395869.75 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.975916 | 496989.50 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 10.961709 | 1370213.62 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 14.402500 | 1800312.50 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.350875 | 208607.42 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.834417 | 91162.77 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.712333 | 73630.20 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.587542 | 87305.34 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 17.587250 | 2198406.25 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.484583 | 1435572.88 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.103167 | 1637895.88 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.296708 | 1412088.50 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.287833 | 160747.39 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.559792 | 86871.75 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.092792 | 126449.87 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.434917 | 69295.58 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.630459 | 1453807.38 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 10.068375 | 1258546.88 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.539750 | 817468.75 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.463625 | 682953.12 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.419834 | 100309.91 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 19.650875 | 307044.92 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 18.574541 | 290227.20 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 18.439000 | 288109.38 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.335959 | 916994.88 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 38.700250 | 4837531.25 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 36.129542 | 4516192.75 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 39.186250 | 4898281.25 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.449625 | 69525.39 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 13.066583 | 204165.36 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 12.566292 | 196348.31 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 15.702042 | 245344.41 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.320375 | 540046.88 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 25.341208 | 3167651.00 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 19.382666 | 2422833.25 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 19.056625 | 2382078.12 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.335709 | 83370.45 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.056250 | 47753.91 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.341875 | 52216.80 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.717625 | 73712.89 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.945083 | 743135.38 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.927834 | 615979.25 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.173959 | 646744.88 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.522500 | 565312.50 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.846666 | 75729.16 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.668375 | 88568.36 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.484958 | 54452.47 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.267458 | 82304.03 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.083042 | 635380.25 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.031000 | 503875.00 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.955125 | 869390.62 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 11.814667 | 1476833.38 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.581917 | 118467.45 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.912542 | 92383.47 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.041292 | 94395.19 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 7.121500 | 111273.44 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.946917 | 1368364.62 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.818167 | 1477270.88 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.779250 | 1347406.25 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.950792 | 1493849.00 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.552292 | 133629.56 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.890792 | 92043.62 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 7.080584 | 110634.12 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.707458 | 120429.03 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.032542 | 754067.75 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.815625 | 976953.12 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 10.077167 | 1259645.88 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 13.600541 | 1700067.62 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.072750 | 141761.72 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.297250 | 129644.53 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 10.693292 | 167082.69 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.948833 | 1743604.12 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 15.573125 | 1946640.62 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 16.347250 | 2043406.25 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.044125 | 125689.45 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.922709 | 139417.33 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 10.406625 | 162603.52 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.973208 | 1121651.00 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 11.982416 | 1497802.00 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 14.738333 | 1842291.62 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.389417 | 68584.64 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.522500 | 8164.06 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.813583 | 12712.23 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.523291 | 23801.42 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 2.117000 | 264625.00 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.513041 | 64130.12 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.937750 | 117218.75 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.994375 | 124296.88 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.188875 | 65451.17 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.042708 | 31917.31 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.823500 | 12867.19 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.244833 | 19450.52 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.194208 | 149276.00 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.598166 | 74770.75 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.889792 | 111224.00 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 0.877750 | 109718.75 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.772250 | 74566.41 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 1.825750 | 28527.34 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.604083 | 25063.80 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.631333 | 41114.58 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.895625 | 736953.12 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 3.108542 | 388567.75 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.282792 | 285349.00 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 2.485875 | 310734.38 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.216250 | 97128.91 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.869083 | 107329.42 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.812000 | 28312.50 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.824917 | 44139.33 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.375916 | 1171989.50 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.496084 | 437010.50 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.832292 | 604036.50 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.913583 | 739197.88 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.778583 | 105915.36 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.850333 | 75786.45 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.952500 | 46132.81 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.582125 | 55970.70 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 8.154208 | 1019276.00 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.649167 | 1331145.88 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.110959 | 888869.88 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.187291 | 898411.38 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.992250 | 78003.91 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.896167 | 76502.61 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.794125 | 90533.20 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.886000 | 76343.75 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.864500 | 733062.50 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.868333 | 733541.62 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.228667 | 653583.38 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.331875 | 666484.38 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.499834 | 101559.91 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.565916 | 86967.44 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.717000 | 58078.12 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.782375 | 59099.61 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.647750 | 955968.75 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.790834 | 1223854.25 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.347917 | 918489.62 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.571417 | 946427.12 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.979542 | 93430.34 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.971583 | 77680.98 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.820542 | 90945.97 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.948250 | 30441.41 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.066958 | 633369.75 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.562209 | 820276.12 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.259000 | 657375.00 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.467792 | 558474.00 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.899500 | 185929.69 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.511959 | 101749.36 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.146792 | 96043.62 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 7.019375 | 109677.73 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.522792 | 1440349.00 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 13.641708 | 1705213.50 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.994500 | 1249312.50 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.187458 | 1398432.25 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.961167 | 124393.23 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.210500 | 81414.06 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.536375 | 102130.86 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 8.731334 | 136427.09 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.143167 | 767895.88 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 10.197333 | 1274666.62 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.006458 | 1000807.25 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 10.139375 | 1267421.88 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 20.081500 | 313773.44 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 18.587250 | 290425.78 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 16.316334 | 254942.72 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 14.377750 | 224652.34 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 35.962125 | 4495265.62 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 37.553458 | 4694182.25 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 39.263750 | 4907968.75 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 41.365667 | 5170708.38 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 22.631042 | 353610.03 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 14.249792 | 222653.00 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 10.964833 | 171325.52 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 13.369583 | 208899.73 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 25.245708 | 3155713.50 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 20.130958 | 2516369.75 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 17.120375 | 2140046.88 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 15.367500 | 1920937.50 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.577125 | 102767.58 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.939792 | 124059.25 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 37.658750 | 588417.97 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 50.311500 | 786117.19 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.422958 | 1802869.75 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 19.790750 | 2473843.75 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 79.729875 | 9966234.38 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 89.214917 | 11151864.62 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.981125 | 155955.08 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 12.650125 | 197658.20 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 47.289208 | 738893.88 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 58.282459 | 910663.42 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.921917 | 1115239.62 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 14.659125 | 1832390.62 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 81.125292 | 10140661.50 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 85.658791 | 10707348.88 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 3.992792 | 249549.50 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 3.989167 | 249322.94 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 55.707416 | 3481713.50 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 55.219375 | 3451210.94 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 5.783417 | 361463.56 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 3.354833 | 209677.06 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 51.478917 | 3217432.31 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 44.858542 | 2803658.88 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.014000 | 875.00 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.001292 | 80.75 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.015750 | 984.38 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.001458 | 91.12 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000084 | 5.25 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.012209 | 763.06 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.036917 | 2307.31 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.013833 | 864.56 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.073875 | 4617.19 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000084 | 5.25 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.011208 | 700.50 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.002833 | 177.06 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000125 | 7.81 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.010250 | 640.62 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.002750 | 171.88 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000209 | 13.06 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.011541 | 721.31 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002542 | 158.88 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000084 | 5.25 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.013209 | 825.56 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002584 | 161.50 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `jax-cpu` is 55.5x faster than the slowest successful cell (`julia-base`, 75.740 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `pytorch-cpu` is 17.1x faster than the slowest successful cell (`julia-base`, 75.740 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-direct` is 19.5x faster than the slowest successful cell (`julia-base`, 75.740 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-trace` is 19.2x faster than the slowest successful cell (`julia-base`, 75.740 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `jax-cpu` is 23.9x faster than the slowest successful cell (`julia-base`, 508.308 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `pytorch-cpu` is 42.5x faster than the slowest successful cell (`julia-base`, 508.308 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-direct` is 44.3x faster than the slowest successful cell (`julia-base`, 508.308 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-trace` is 44.6x faster than the slowest successful cell (`julia-base`, 508.308 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 10.0x faster than the slowest successful cell (`julia-base`, 9.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 25.2x faster than the slowest successful cell (`julia-base`, 175.469 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 23.1x faster than the slowest successful cell (`julia-base`, 175.469 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-direct` is 20.0x faster than the slowest successful cell (`julia-base`, 175.469 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-trace` is 23.6x faster than the slowest successful cell (`julia-base`, 175.469 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 60.4x faster than the slowest successful cell (`julia-base`, 151.977 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 26.0x faster than the slowest successful cell (`julia-base`, 151.977 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-direct` is 25.8x faster than the slowest successful cell (`julia-base`, 151.977 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-trace` is 24.8x faster than the slowest successful cell (`julia-base`, 151.977 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `jax-cpu` is 96.2x faster than the slowest successful cell (`julia-base`, 624.846 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `pytorch-cpu` is 83.6x faster than the slowest successful cell (`julia-base`, 624.846 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-direct` is 79.2x faster than the slowest successful cell (`julia-base`, 624.846 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-trace` is 79.3x faster than the slowest successful cell (`julia-base`, 624.846 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 14.6x faster than the slowest successful cell (`pytorch-cpu`, 13.517 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 13.4x faster than the slowest successful cell (`pytorch-cpu`, 13.517 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 13.517 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 16.6x faster than the slowest successful cell (`pytorch-cpu`, 11.600 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 14.0x faster than the slowest successful cell (`pytorch-cpu`, 11.600 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 11.600 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 23.1x faster than the slowest successful cell (`pytorch-cpu`, 15.797 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 15.6x faster than the slowest successful cell (`pytorch-cpu`, 15.797 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 14.0x faster than the slowest successful cell (`pytorch-cpu`, 15.797 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 12.458 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 15.0x faster than the slowest successful cell (`pytorch-cpu`, 12.458 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 16.2x faster than the slowest successful cell (`pytorch-cpu`, 12.458 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 876.5x faster than the slowest successful cell (`julia-base`, 1.790 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 625.3x faster than the slowest successful cell (`tenferro-direct`, 1.537 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.008 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 32.5x faster than the slowest successful cell (`jax-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-trace` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 0.011 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`3x4x4`): `tenferro-direct` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x4x4`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `pytorch-cpu` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 37.8x faster than the slowest successful cell (`jax-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 24.4x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 21.6x faster than the slowest successful cell (`jax-cpu`, 0.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 45.3x faster than the slowest successful cell (`pytorch-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 13.1x faster than the slowest successful cell (`pytorch-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `jax-cpu` is 14.1x faster than the slowest successful cell (`pytorch-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 37.9x faster than the slowest successful cell (`pytorch-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-trace` is 15.8x faster than the slowest successful cell (`pytorch-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `jax-cpu` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 26.6x faster than the slowest successful cell (`pytorch-cpu`, 0.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 17.8x faster than the slowest successful cell (`pytorch-cpu`, 0.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 47.9x faster than the slowest successful cell (`pytorch-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 15.1x faster than the slowest successful cell (`pytorch-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `jax-cpu` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 61.8x faster than the slowest successful cell (`pytorch-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 17.7x faster than the slowest successful cell (`pytorch-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `jax-cpu` is 10.0x faster than the slowest successful cell (`pytorch-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 44.5x faster than the slowest successful cell (`pytorch-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 14.6x faster than the slowest successful cell (`pytorch-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 33.4x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 41.6x faster than the slowest successful cell (`pytorch-cpu`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 13.9x faster than the slowest successful cell (`pytorch-cpu`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 33.4x faster than the slowest successful cell (`pytorch-cpu`, 0.064 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.064 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 32.8x faster than the slowest successful cell (`pytorch-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 38.3x faster than the slowest successful cell (`pytorch-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 13.6x faster than the slowest successful cell (`pytorch-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `jax-cpu` is 13.1x faster than the slowest successful cell (`pytorch-cpu`, 0.119 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 52.0x faster than the slowest successful cell (`pytorch-cpu`, 0.119 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 19.3x faster than the slowest successful cell (`pytorch-cpu`, 0.119 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `jax-cpu` is 14.6x faster than the slowest successful cell (`pytorch-cpu`, 0.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 35.7x faster than the slowest successful cell (`pytorch-cpu`, 0.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 16.0x faster than the slowest successful cell (`pytorch-cpu`, 0.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=4, shape=`4x2x2`): `tenferro-direct` is 13.5x faster than the slowest successful cell (`jax-cpu`, 0.043 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`8x2x2`): `tenferro-direct` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 0.022 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 16.8x faster than the slowest successful cell (`pytorch-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 11.1x faster than the slowest successful cell (`jax-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 14.3x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 12.2x faster than the slowest successful cell (`jax-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 14.4x faster than the slowest successful cell (`jax-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 31.8x faster than the slowest successful cell (`pytorch-cpu`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 15.3x faster than the slowest successful cell (`pytorch-cpu`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 27.8x faster than the slowest successful cell (`pytorch-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 29.8x faster than the slowest successful cell (`pytorch-cpu`, 0.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 0.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 24.8x faster than the slowest successful cell (`pytorch-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 15.1x faster than the slowest successful cell (`pytorch-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 17.4x faster than the slowest successful cell (`pytorch-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 20.3x faster than the slowest successful cell (`pytorch-cpu`, 0.073 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.073 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 15.3x faster than the slowest successful cell (`pytorch-cpu`, 0.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 16.8x faster than the slowest successful cell (`pytorch-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 27.4x faster than the slowest successful cell (`pytorch-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 24.0x faster than the slowest successful cell (`pytorch-cpu`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 15.3x faster than the slowest successful cell (`pytorch-cpu`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 27.7x faster than the slowest successful cell (`pytorch-cpu`, 0.153 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 18.1x faster than the slowest successful cell (`pytorch-cpu`, 0.153 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 32.0x faster than the slowest successful cell (`pytorch-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 16.7x faster than the slowest successful cell (`pytorch-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 15.2x faster than the slowest successful cell (`pytorch-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 18.4x faster than the slowest successful cell (`pytorch-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x4x4,rhs=1`): `tenferro-direct` is 41.5x faster than the slowest successful cell (`pytorch-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x4x4,rhs=1`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`3x8x8,rhs=1`): `tenferro-direct` is 16.8x faster than the slowest successful cell (`pytorch-cpu`, 0.035 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x32x32,rhs=1`): `tenferro-direct` is 11.1x faster than the slowest successful cell (`jax-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x8x8,rhs=1`): `tenferro-direct` is 13.9x faster than the slowest successful cell (`jax-cpu`, 0.030 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x2x2,rhs=1`): `tenferro-direct` is 22.8x faster than the slowest successful cell (`pytorch-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x4x4,rhs=1`): `tenferro-direct` is 15.7x faster than the slowest successful cell (`pytorch-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `jax-cpu` is 25.5x faster than the slowest successful cell (`julia-base`, 348.422 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `pytorch-cpu` is 19.9x faster than the slowest successful cell (`julia-base`, 348.422 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-direct` is 21.2x faster than the slowest successful cell (`julia-base`, 348.422 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-trace` is 15.7x faster than the slowest successful cell (`julia-base`, 348.422 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 33.5x faster than the slowest successful cell (`julia-base`, 246.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 36.9x faster than the slowest successful cell (`julia-base`, 246.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 25.0x faster than the slowest successful cell (`julia-base`, 246.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 29.4x faster than the slowest successful cell (`julia-base`, 246.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `jax-cpu` is 31.2x faster than the slowest successful cell (`julia-base`, 294.197 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `pytorch-cpu` is 42.1x faster than the slowest successful cell (`julia-base`, 294.197 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-direct` is 36.7x faster than the slowest successful cell (`julia-base`, 294.197 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-trace` is 36.8x faster than the slowest successful cell (`julia-base`, 294.197 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `jax-cpu` is 31.7x faster than the slowest successful cell (`julia-base`, 250.631 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `pytorch-cpu` is 30.8x faster than the slowest successful cell (`julia-base`, 250.631 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-direct` is 28.9x faster than the slowest successful cell (`julia-base`, 250.631 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-trace` is 28.6x faster than the slowest successful cell (`julia-base`, 250.631 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `jax-cpu` is 31.6x faster than the slowest successful cell (`julia-base`, 834.916 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 38.3x faster than the slowest successful cell (`julia-base`, 834.916 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-direct` is 38.6x faster than the slowest successful cell (`julia-base`, 834.916 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-trace` is 30.8x faster than the slowest successful cell (`julia-base`, 834.916 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 41.2x faster than the slowest successful cell (`julia-base`, 408.992 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 40.8x faster than the slowest successful cell (`julia-base`, 408.992 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-direct` is 46.8x faster than the slowest successful cell (`julia-base`, 408.992 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-trace` is 43.0x faster than the slowest successful cell (`julia-base`, 408.992 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 28.0x faster than the slowest successful cell (`julia-base`, 141.532 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-eager` is 12.5x faster than the slowest successful cell (`julia-base`, 141.532 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-trace` is 17.0x faster than the slowest successful cell (`julia-base`, 141.532 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 19.4x faster than the slowest successful cell (`julia-base`, 227.964 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 20.9x faster than the slowest successful cell (`julia-base`, 227.964 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 33.0x faster than the slowest successful cell (`julia-base`, 227.964 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 31.7x faster than the slowest successful cell (`julia-base`, 227.964 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`jax-cpu`, 6.891 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 10.6x faster than the slowest successful cell (`julia-base`, 8.915 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 10.4x faster than the slowest successful cell (`julia-base`, 8.915 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `jax-cpu` is 76.0x faster than the slowest successful cell (`julia-base`, 1057.507 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 85.8x faster than the slowest successful cell (`julia-base`, 1057.507 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 91.8x faster than the slowest successful cell (`julia-base`, 1057.507 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 98.2x faster than the slowest successful cell (`julia-base`, 1057.507 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `jax-cpu` is 63.9x faster than the slowest successful cell (`julia-base`, 1002.548 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 85.6x faster than the slowest successful cell (`julia-base`, 1002.548 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 91.0x faster than the slowest successful cell (`julia-base`, 1002.548 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 97.3x faster than the slowest successful cell (`julia-base`, 1002.548 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 28.3x faster than the slowest successful cell (`julia-base`, 217.482 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 30.2x faster than the slowest successful cell (`julia-base`, 217.482 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 20.9x faster than the slowest successful cell (`julia-base`, 217.482 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 25.8x faster than the slowest successful cell (`julia-base`, 217.482 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 32.8x faster than the slowest successful cell (`julia-base`, 1723.504 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 46.1x faster than the slowest successful cell (`julia-base`, 1723.504 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-eager` is 47.8x faster than the slowest successful cell (`julia-base`, 1723.504 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-trace` is 46.2x faster than the slowest successful cell (`julia-base`, 1723.504 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 7345.4x faster than the slowest successful cell (`julia-base`, 16.219 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 6629.4x faster than the slowest successful cell (`julia-base`, 15.466 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 6716.4x faster than the slowest successful cell (`tenferro-direct`, 15.387 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 1336.7x faster than the slowest successful cell (`julia-base`, 22.280 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 72910.5x faster than the slowest successful cell (`julia-base`, 230.907 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 34464.9x faster than the slowest successful cell (`julia-base`, 206.789 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 461.4x faster than the slowest successful cell (`tenferro-direct`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 923.4x faster than the slowest successful cell (`tenferro-direct`, 0.005 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 140.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 80.1x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 55.5x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 165.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 8212.1 GB/s at 0.002 ms. The conservative physical lower bound is 0.168 ms, so the cell is 82.1x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 6822.2 GB/s at 0.002 ms. The conservative physical lower bound is 0.042 ms, so the cell is 17.1x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 121574.0 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1215.7x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 115060.2 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 287.7x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 117169.6 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1171.7x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 16105.8 GB/s at 0.017 ms. The conservative physical lower bound is 0.671 ms, so the cell is 40.3x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 42380.1 GB/s at 0.003 ms. The conservative physical lower bound is 1.342 ms, so the cell is 423.8x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 22369.6 GB/s at 0.006 ms. The conservative physical lower bound is 0.336 ms, so the cell is 55.9x faster than the bound.

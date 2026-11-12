# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20261112_182103/run.yaml`
- Timestamp: `2026-11-12T18:21:03.775436+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20261112_182103`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

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
- Python platform: `macOS-26.7.2-arm64-arm-64bit-Mach-O`
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

- CSV: `data/results/mac-cpu/cpu/public_api/20261112_182103/cpu_public_api_t1_20261112_182103.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20261112_182103/cpu_public_api_t4_20261112_182103.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 4.159 ± 0.543 | - | 4.145 ± 0.334 | 4.070 ± 0.169 | 1.551 ± 0.054 | 4.054 ± 2.613 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 3.932 ± 0.299 | - | 3.797 ± 0.276 | 4.120 ± 0.242 | 3.102 ± 0.465 | 75.298 ± 6.789 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 19.194 ± 4.793 | - | 13.017 ± 1.940 | 12.668 ± 0.275 | 15.515 ± 4.137 | 13.312 ± 3.294 | 21.509 ± 7.295 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 12.827 ± 0.514 | - | 12.997 ± 0.733 | 15.595 ± 1.054 | 13.028 ± 2.950 | 11.254 ± 1.034 | 9.582 ± 0.405 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 45.816 ± 7.203 | - | 49.994 ± 3.007 | 21.121 ± 4.934 | 36.153 ± 3.295 | 52.836 ± 4.673 | 59.881 ± 22.185 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 22.568 ± 3.181 | - | 22.760 ± 2.897 | 12.289 ± 1.484 | 24.926 ± 2.975 | 45.162 ± 0.931 | 18.470 ± 6.203 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 11.018 ± 1.767 | - | 11.393 ± 1.920 | 10.970 ± 0.393 | 49.953 ± 1.160 | 61.901 ± 13.508 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 10.474 ± 1.061 | - | 10.715 ± 1.351 | 12.437 ± 1.053 | 23.986 ± 3.526 | 29.876 ± 4.379 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 12.685 ± 1.230 | - | unsupported | 11.339 ± 0.651 | 51.699 ± 1.889 | 62.286 ± 2.410 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 11.703 ± 0.279 | - | unsupported | 11.782 ± 1.028 | 23.763 ± 2.473 | 26.178 ± 4.026 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 12.778 ± 0.939 | - | 10.458 ± 0.390 | 11.453 ± 0.284 | 19.817 ± 0.567 | 8.952 ± 2.207 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 12.039 ± 1.725 | - | 10.013 ± 0.286 | 11.678 ± 0.183 | 21.167 ± 1.549 | 440.327 ± 24.169 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 39.714 ± 4.021 | - | 46.410 ± 18.339 | 38.974 ± 3.185 | 43.216 ± 2.064 | 47.450 ± 6.662 | 51.190 ± 2.588 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 19.688 ± 3.106 | - | 19.684 ± 4.762 | 22.932 ± 3.036 | 29.956 ± 3.383 | 48.700 ± 1.300 | 19.445 ± 2.810 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 52.480 ± 5.280 | - | 62.413 ± 12.655 | 49.613 ± 17.149 | 86.553 ± 8.988 | 78.895 ± 6.529 | 79.565 ± 1.093 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 22.974 ± 4.880 | - | 23.433 ± 5.644 | 29.906 ± 2.753 | 43.294 ± 5.301 | 68.681 ± 3.164 | 30.589 ± 8.708 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 9.780 ± 1.454 | - | 9.377 ± 1.389 | 10.370 ± 0.972 | 9.508 ± 1.334 | 9.040 ± 0.440 | 10.160 ± 2.114 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 9.830 ± 0.645 | - | 9.814 ± 0.436 | 12.338 ± 2.774 | 13.443 ± 4.969 | 7.530 ± 1.819 | 9.592 ± 2.333 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.535 ± 0.261 | - | 8.171 ± 1.005 | 6.231 ± 0.182 | 4.867 ± 0.263 | 9.350 ± 0.391 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 1.220 ± 0.081 | - | 5.108 ± 0.326 | 5.158 ± 1.097 | 6.514 ± 2.198 | 11.740 ± 2.163 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 8.418 ± 0.765 | - | 7.027 ± 0.511 | 7.565 ± 0.024 | 6.921 ± 1.166 | 9.187 ± 2.473 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 8.382 ± 0.276 | - | 6.883 ± 0.211 | 8.082 ± 0.468 | 7.449 ± 0.135 | 180.646 ± 158.929 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 5.656 ± 0.708 | - | 5.998 ± 0.710 | 5.319 ± 0.168 | 2.245 ± 0.035 | 6.324 ± 4.484 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 5.523 ± 0.801 | - | 5.977 ± 0.394 | 6.643 ± 0.800 | 3.246 ± 0.909 | 119.072 ± 16.955 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 6.824 ± 0.186 | - | 8.446 ± 0.316 | 7.603 ± 0.158 | 6.583 ± 0.769 | 7.783 ± 0.286 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 6.556 ± 0.264 | - | 8.057 ± 0.198 | 7.910 ± 0.188 | 7.211 ± 0.246 | 617.766 ± 114.750 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 11.397 ± 0.973 | - | 12.017 ± 4.032 | 11.727 ± 1.215 | 48.003 ± 7.342 | 61.926 ± 2.021 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 10.416 ± 0.384 | - | 10.398 ± 0.121 | 11.395 ± 0.714 | 22.952 ± 2.356 | 28.390 ± 2.731 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 12.619 ± 1.117 | - | unsupported | 11.315 ± 0.378 | 60.463 ± 5.013 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 11.951 ± 1.941 | - | unsupported | 12.845 ± 1.299 | 24.672 ± 3.496 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 38.822 ± 88.240 | - | 23.054 ± 5.426 | 14.937 ± 1.799 | 15.856 ± 1.026 | 11.502 ± 6.557 | 15.001 ± 4.038 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 13.538 ± 0.925 | - | 12.807 ± 1.047 | 15.029 ± 1.817 | 15.460 ± 3.226 | 12.028 ± 1.386 | 15.825 ± 2.995 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 20.009 ± 6.091 | - | 19.238 ± 11.242 | 21.438 ± 2.403 | 29.407 ± 5.721 | 35.444 ± 19.838 | 20.930 ± 1.201 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 19.295 ± 3.943 | - | 23.166 ± 4.059 | 20.376 ± 1.652 | 23.058 ± 0.855 | 23.026 ± 8.164 | 19.064 ± 3.114 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 25.939 ± 1.700 | - | 41.408 ± 9.799 | 41.251 ± 3.012 | 31.135 ± 1.271 | 63.323 ± 7.962 | 70.970 ± 32.298 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 16.436 ± 6.900 | - | 18.623 ± 7.774 | 26.695 ± 4.628 | 22.472 ± 4.598 | 54.971 ± 4.671 | 28.156 ± 9.010 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 7.570 ± 0.516 | - | 6.630 ± 0.714 | 7.639 ± 0.676 | 8.654 ± 0.977 | 8.324 ± 1.685 | 7.851 ± 1.682 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 8.705 ± 1.808 | - | 6.971 ± 0.826 | 7.263 ± 0.135 | 7.613 ± 0.887 | 9.263 ± 2.617 | 10.733 ± 6.506 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 16.387 ± 3.031 | - | 14.181 ± 2.320 | 17.424 ± 0.664 | 18.140 ± 0.512 | 15.142 ± 2.169 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 13.925 ± 1.382 | - | 12.723 ± 0.849 | 15.793 ± 1.711 | 17.775 ± 4.833 | 12.110 ± 3.013 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 30.703 ± 0.065 | - | 30.858 ± 0.877 | 36.868 ± 1.692 | 29.999 ± 1.965 | 39.278 ± 4.284 | 38.565 ± 8.770 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 23.894 ± 7.735 | - | 21.517 ± 6.430 | 19.812 ± 6.850 | 20.490 ± 4.627 | 41.695 ± 6.561 | 21.839 ± 5.262 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 18.008 ± 5.324 | - | 19.569 ± 3.591 | 20.843 ± 3.700 | 26.319 ± 0.674 | 19.051 ± 0.944 | 21.289 ± 1.433 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 19.536 ± 1.934 | - | 19.038 ± 1.856 | 20.614 ± 1.176 | 27.468 ± 1.465 | 18.445 ± 0.659 | 21.466 ± 3.191 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 18.021 ± 1.118 | - | 16.892 ± 1.301 | 28.633 ± 4.540 | 18.595 ± 1.178 | 36.803 ± 4.538 | 51.381 ± 25.103 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 13.079 ± 6.119 | - | 17.970 ± 11.788 | 15.549 ± 1.485 | 13.389 ± 3.830 | 38.163 ± 9.770 | 24.796 ± 12.449 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 28.634 ± 6.843 | - | 26.282 ± 3.482 | 29.097 ± 0.541 | 22.536 ± 6.083 | 35.035 ± 17.884 | 37.151 ± 9.086 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 14.652 ± 2.333 | - | 18.403 ± 2.490 | 17.363 ± 2.461 | 14.326 ± 0.176 | 26.286 ± 2.880 | 12.412 ± 1.503 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 31.456 ± 6.146 | - | 30.578 ± 0.664 | 44.720 ± 1.327 | 27.889 ± 0.777 | 46.559 ± 12.874 | 44.719 ± 1.081 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 21.160 ± 5.794 | - | 22.638 ± 3.837 | 24.687 ± 6.993 | 22.060 ± 2.431 | 53.794 ± 13.709 | 24.352 ± 7.177 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 15.768 ± 1.089 | - | 15.315 ± 1.103 | 25.230 ± 2.603 | 19.592 ± 0.537 | 32.407 ± 7.641 | 32.671 ± 12.261 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 8.300 ± 0.405 | - | 10.724 ± 3.176 | 13.211 ± 2.212 | 15.855 ± 1.316 | 25.676 ± 3.437 | 14.366 ± 6.369 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.272 ± 0.019 | 0.290 ± 0.011 | unsupported | unsupported | 0.291 ± 0.136 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.248 ± 0.014 | 0.304 ± 0.039 | unsupported | unsupported | 0.409 ± 0.216 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 26.695 ± 16.545 | - | 24.058 ± 9.353 | 20.976 ± 1.833 | 27.145 ± 1.284 | 16.202 ± 2.911 | 20.501 ± 0.650 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 18.950 ± 0.773 | - | 18.870 ± 1.522 | 22.748 ± 2.299 | 27.041 ± 1.190 | 24.612 ± 4.263 | 23.179 ± 3.696 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 25.672 ± 6.959 | - | 25.857 ± 7.631 | 20.304 ± 1.971 | 27.894 ± 2.500 | 22.709 ± 4.271 | 23.090 ± 1.592 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 18.380 ± 1.259 | - | 19.145 ± 1.990 | 22.052 ± 2.190 | 28.631 ± 5.734 | 21.077 ± 5.821 | 19.946 ± 1.921 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 18.165 ± 6.299 | - | 33.113 ± 10.045 | 19.871 ± 0.793 | 28.794 ± 2.099 | 20.786 ± 3.862 | 19.179 ± 2.610 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 20.604 ± 3.660 | - | 18.101 ± 1.342 | 20.670 ± 1.166 | 24.723 ± 2.208 | 19.312 ± 1.636 | 19.792 ± 1.903 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 12.896 ± 5.618 | - | 19.256 ± 5.403 | 15.319 ± 1.505 | 15.532 ± 1.158 | 12.491 ± 0.728 | 15.277 ± 12.923 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 11.307 ± 0.984 | - | 13.113 ± 1.514 | 13.731 ± 1.115 | 14.336 ± 2.979 | 12.038 ± 0.569 | 12.872 ± 1.880 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 37.917 ± 7.568 | - | 46.179 ± 21.339 | 71.109 ± 0.956 | 51.627 ± 31.693 | 93.283 ± 34.852 | 129.026 ± 47.585 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 17.065 ± 15.052 | - | 20.357 ± 1.765 | 42.764 ± 11.449 | 25.902 ± 0.483 | 73.450 ± 13.870 | 37.476 ± 8.719 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 9.545 ± 2.878 | - | 7.599 ± 3.168 | 12.286 ± 0.117 | 22.961 ± 0.897 | 7.150 ± 1.879 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 9.066 ± 3.827 | - | 6.060 ± 0.756 | 9.696 ± 1.406 | 16.495 ± 2.554 | 4.743 ± 0.248 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.043 ± 0.642 | - | 2.147 ± 1.478 | 17.378 ± 2.401 | 2.589 ± 0.171 | 0.853 ± 0.143 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.954 ± 0.982 | - | 0.732 ± 0.077 | 13.428 ± 2.669 | 2.169 ± 0.212 | 0.937 ± 0.198 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 1.135 ± 0.084 | - | 1.231 ± 0.762 | 1.856 ± 0.147 | 2.611 ± 1.533 | 1.342 ± 0.244 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 1.736 ± 0.200 | - | 1.508 ± 0.096 | 1.187 ± 0.270 | 1.471 ± 1.009 | 0.905 ± 0.423 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 8.031 ± 1.520 | - | 7.444 ± 0.604 | 12.421 ± 0.294 | 23.001 ± 0.455 | 7.335 ± 0.897 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 7.535 ± 3.101 | - | 6.991 ± 1.419 | 7.652 ± 0.600 | 16.604 ± 2.811 | 5.250 ± 0.393 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.055 ± 0.207 | - | 1.072 ± 0.084 | 21.128 ± 3.206 | 2.353 ± 0.124 | 1.134 ± 0.240 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.732 ± 0.035 | - | 0.758 ± 0.048 | 17.874 ± 1.058 | 3.354 ± 0.817 | 0.926 ± 0.128 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 5.742 ± 1.489 | - | 7.180 ± 1.760 | 6.498 ± 0.297 | 3.985 ± 0.552 | 3.975 ± 0.928 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 5.300 ± 0.279 | - | 6.649 ± 0.681 | 4.681 ± 0.971 | 5.298 ± 0.896 | 3.864 ± 0.288 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 8.038 ± 3.332 | - | 7.991 ± 0.859 | 8.264 ± 1.411 | 42.731 ± 1.294 | 8.200 ± 0.505 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 6.027 ± 0.675 | - | 7.406 ± 3.242 | 6.935 ± 0.599 | 25.565 ± 2.680 | 5.627 ± 1.000 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.972 ± 0.800 | - | 1.164 ± 0.769 | 5.948 ± 1.599 | 2.617 ± 0.221 | 0.842 ± 0.088 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.722 ± 0.072 | - | 0.698 ± 0.011 | 6.483 ± 2.287 | 2.546 ± 1.166 | 0.844 ± 0.227 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 2.253 ± 0.855 | - | 1.016 ± 0.094 | 0.948 ± 0.580 | 1.248 ± 0.268 | 0.950 ± 0.053 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 1.425 ± 0.083 | - | 1.465 ± 0.223 | 1.074 ± 0.316 | 1.182 ± 0.179 | 0.935 ± 1.119 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 7.746 ± 6.705 | - | 6.879 ± 2.966 | 7.403 ± 1.526 | 32.730 ± 0.990 | 11.944 ± 1.864 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 5.886 ± 0.759 | - | 7.190 ± 1.779 | 6.963 ± 0.430 | 19.296 ± 4.505 | 4.612 ± 0.091 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 1.379 ± 1.149 | - | 0.934 ± 0.059 | 5.755 ± 2.117 | 2.506 ± 0.254 | 0.787 ± 0.047 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.637 ± 0.004 | - | 0.767 ± 0.034 | 5.374 ± 1.858 | 3.252 ± 0.719 | 0.728 ± 0.034 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.292 ± 0.206 | - | 0.932 ± 0.093 | 1.346 ± 0.894 | 1.174 ± 0.193 | 0.722 ± 0.034 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 1.571 ± 0.102 | - | 1.312 ± 0.164 | 1.187 ± 0.675 | 1.893 ± 0.769 | 0.905 ± 0.960 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.890 ± 0.123 | 0.920 ± 0.129 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.720 ± 0.034 | 0.712 ± 0.026 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 28.450 ± 1.137 | - | 28.910 ± 1.476 | 33.468 ± 0.634 | 39.252 ± 0.888 | 30.425 ± 45.657 | 32.127 ± 1.885 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 19.218 ± 1.849 | - | 20.204 ± 4.484 | 20.214 ± 3.109 | 25.562 ± 2.239 | 28.618 ± 45.684 | 17.598 ± 3.848 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 23.763 ± 4.333 | - | 19.484 ± 3.124 | 19.593 ± 0.846 | 39.386 ± 2.435 | 45.375 ± 2.064 | 42.725 ± 5.568 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 12.726 ± 0.969 | - | 13.816 ± 1.432 | 19.051 ± 3.721 | 21.647 ± 4.284 | 41.488 ± 0.168 | 22.767 ± 2.081 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 34.765 ± 12.404 | - | 29.149 ± 1.412 | 21.528 ± 1.411 | 28.495 ± 4.656 | 33.825 ± 3.440 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 24.330 ± 2.537 | - | 24.349 ± 1.190 | 24.503 ± 3.027 | 29.168 ± 3.187 | 24.335 ± 1.208 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 17.447 ± 5.700 | - | 21.545 ± 11.553 | 14.157 ± 1.478 | 16.784 ± 0.739 | 11.908 ± 3.930 | 18.972 ± 0.871 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 12.916 ± 0.844 | - | 12.848 ± 0.734 | 15.495 ± 3.412 | 16.253 ± 1.837 | 14.303 ± 4.567 | 28.929 ± 100.002 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 24.343 ± 0.689 | - | 23.757 ± 0.551 | 30.663 ± 2.292 | 27.656 ± 1.406 | 32.200 ± 3.563 | 33.667 ± 2.408 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 16.875 ± 4.376 | - | 18.974 ± 5.836 | 19.485 ± 7.303 | 21.040 ± 1.480 | 32.418 ± 6.669 | 13.028 ± 1.152 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 12.936 ± 0.458 | - | 16.296 ± 7.328 | 13.712 ± 0.767 | 22.688 ± 0.968 | 28.615 ± 1.860 | 27.058 ± 4.429 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 12.566 ± 0.861 | - | 13.138 ± 0.447 | 16.666 ± 5.741 | 13.618 ± 2.641 | 26.806 ± 0.784 | 16.787 ± 2.998 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 17.803 ± 4.822 | - | 18.554 ± 3.733 | 19.048 ± 2.439 | 26.288 ± 3.269 | 25.205 ± 1.542 | 20.051 ± 1.187 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 21.048 ± 2.248 | - | 19.884 ± 1.754 | 19.944 ± 0.728 | 23.296 ± 0.234 | 17.643 ± 3.104 | 19.796 ± 1.264 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 68.666 ± 3.461 | - | 67.917 ± 2.925 | 71.203 ± 1.543 | 13.893 ± 1.004 | 44.117 ± 2.537 | 49.696 ± 8.439 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 34.482 ± 15.205 | - | 34.163 ± 8.383 | 41.803 ± 4.402 | 9.678 ± 3.738 | 52.394 ± 1.994 | 24.597 ± 6.320 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 26.903 ± 4.420 | 31.480 ± 11.217 | unsupported | 22.442 ± 2.328 | 30.134 ± 1.637 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 24.199 ± 2.254 | 25.080 ± 0.295 | unsupported | 21.895 ± 1.765 | 25.267 ± 1.152 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.822 ± 0.122 | - | 0.701 ± 0.024 | 1.366 ± 0.142 | 0.781 ± 0.066 | 1.037 ± 0.150 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.633 ± 0.102 | - | 0.625 ± 0.066 | 1.117 ± 0.201 | 0.713 ± 0.045 | 0.780 ± 0.104 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.783 ± 0.049 | - | 0.756 ± 0.046 | 1.162 ± 0.310 | 0.758 ± 0.071 | 1.774 ± 0.575 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.852 ± 0.350 | - | 0.805 ± 0.108 | 1.082 ± 0.144 | 0.869 ± 0.026 | 0.870 ± 0.060 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.067 ± 0.089 | - | unsupported | 1.768 ± 0.706 | 1.208 ± 0.136 | 1.548 ± 0.766 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 1.149 ± 0.072 | - | unsupported | 1.386 ± 0.301 | 1.266 ± 0.082 | 1.446 ± 0.708 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.328 ± 0.034 | - | 0.455 ± 0.045 | 1.229 ± 1.454 | 0.671 ± 0.414 | 2.725 ± 3.353 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.444 ± 0.117 | - | 0.346 ± 0.015 | 0.485 ± 0.257 | 0.425 ± 0.037 | 1.165 ± 3.319 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.689 ± 0.031 | - | 0.713 ± 0.069 | 1.360 ± 1.938 | 0.863 ± 0.088 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.650 ± 0.028 | - | 0.584 ± 0.006 | 1.518 ± 0.240 | 1.016 ± 0.033 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.708 ± 0.003 | - | 0.692 ± 0.041 | 1.161 ± 0.811 | 0.834 ± 0.139 | 3.197 ± 1.962 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.614 ± 0.066 | - | 0.758 ± 0.246 | 1.060 ± 0.166 | 0.920 ± 0.041 | 1.184 ± 0.353 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.605 ± 0.041 | - | 0.711 ± 0.054 | 0.613 ± 0.468 | 0.416 ± 0.077 | 2.949 ± 2.489 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.558 ± 0.014 | - | 0.940 ± 0.247 | 1.093 ± 0.464 | 0.538 ± 0.020 | 1.126 ± 1.877 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.347 ± 0.140 | - | 0.001 ± 0.000 | 1.462 ± 0.762 | 1.455 ± 0.091 | 3.051 ± 4.213 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 1.431 ± 0.230 | - | 0.004 ± 0.000 | 1.463 ± 0.104 | 1.434 ± 0.069 | 1.501 ± 0.102 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.054 ± 0.000 | - | 0.075 ± 0.000 | 0.113 ± 0.003 | 0.097 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.092 ± 0.003 | - | 0.115 ± 0.001 | 0.167 ± 0.018 | 0.148 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.203 ± 0.005 | - | 0.218 ± 0.002 | 0.272 ± 0.017 | 0.238 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.025 ± 0.001 | - | 0.028 ± 0.002 | 0.039 ± 0.004 | 0.096 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.057 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.060 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.181 ± 0.039 | - | 0.184 ± 0.010 | 0.260 ± 0.228 | 0.345 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.011 ± 0.000 | - | 0.013 ± 0.001 | 0.017 ± 0.002 | 0.052 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.037 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.054 ± 0.000 | - | 0.076 ± 0.000 | 0.215 ± 0.069 | 0.119 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.095 ± 0.003 | - | 0.116 ± 0.001 | 0.246 ± 0.107 | 0.255 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.201 ± 0.001 | - | 0.220 ± 0.005 | 0.466 ± 0.085 | 0.274 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.026 ± 0.000 | - | 0.028 ± 0.002 | 0.162 ± 0.024 | 0.157 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.089 ± 0.040 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.088 ± 0.040 | 0.094 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.009 ± 0.003 | 0.069 ± 0.046 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.100 ± 0.037 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.103 ± 0.029 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.004 ± 0.000 | 0.102 ± 0.026 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.114 ± 0.081 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.087 ± 0.025 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.106 ± 0.023 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.125 ± 0.047 | 0.065 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.173 ± 0.005 | - | 0.178 ± 0.007 | 0.443 ± 0.240 | 0.497 ± 0.086 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.085 ± 0.024 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.012 ± 0.001 | - | 0.013 ± 0.000 | 0.131 ± 0.059 | 0.078 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.002 ± 0.004 | - | 0.005 ± 0.000 | 0.076 ± 0.060 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.003 ± 0.000 | - | 0.011 ± 0.000 | 0.086 ± 0.077 | 0.062 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.320 ± 0.027 | - | 0.314 ± 0.032 | 0.450 ± 0.049 | 0.905 ± 0.153 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.712 ± 0.073 | - | 1.540 ± 0.075 | 1.805 ± 0.083 | 2.300 ± 0.104 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 6.396 ± 0.502 | - | 6.405 ± 0.234 | 6.884 ± 0.183 | 7.651 ± 0.139 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 4.965 ± 0.559 | - | 6.799 ± 0.341 | 3.393 ± 0.090 | 4.422 ± 0.222 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.105 ± 0.024 | - | 0.097 ± 0.010 | 0.095 ± 0.001 | 0.173 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.003 | 0.021 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.019 ± 0.005 | - | 0.020 ± 0.000 | 0.028 ± 0.006 | 0.094 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.002 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.003 | 0.011 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.029 ± 0.005 | - | 0.026 ± 0.004 | 0.029 ± 0.006 | 0.081 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 38.226 ± 4.183 | - | 52.348 ± 4.248 | 27.072 ± 0.739 | 34.010 ± 1.905 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.019 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.726 ± 0.049 | - | 0.722 ± 0.051 | 0.742 ± 0.032 | 0.925 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.015 ± 0.003 | - | 0.017 ± 0.004 | 0.017 ± 0.000 | 0.031 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.054 ± 0.009 | - | 0.060 ± 0.004 | 0.054 ± 0.002 | 0.104 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.251 ± 0.010 | - | 0.254 ± 0.013 | 0.439 ± 0.114 | 1.332 ± 0.739 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 1.498 ± 0.016 | - | 1.525 ± 0.167 | 1.798 ± 0.017 | 1.571 ± 0.104 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 6.206 ± 0.104 | - | 6.270 ± 0.077 | 7.574 ± 0.665 | 4.630 ± 1.219 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 4.549 ± 0.041 | - | 6.376 ± 0.020 | 3.552 ± 0.308 | 6.651 ± 1.543 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.103 ± 0.160 | - | 0.092 ± 0.001 | 0.104 ± 0.014 | 0.202 ± 0.090 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.008 ± 0.001 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.001 | 0.027 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.024 ± 0.003 | - | 0.020 ± 0.000 | 0.029 ± 0.004 | 0.070 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.039 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.024 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.025 ± 0.001 | - | 0.026 ± 0.001 | 0.029 ± 0.000 | 0.131 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 37.000 ± 3.333 | - | 51.882 ± 3.282 | 27.227 ± 1.002 | 21.866 ± 1.626 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.010 ± 0.006 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.711 ± 0.051 | - | 0.717 ± 0.020 | 0.765 ± 0.056 | 0.937 ± 0.510 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.018 ± 0.000 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.049 ± 0.004 | - | 0.050 ± 0.003 | 0.054 ± 0.006 | 0.200 ± 0.090 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.454 ± 0.348 | - | 0.262 ± 0.063 | 0.331 ± 0.010 | 0.628 ± 0.084 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.485 ± 0.579 | - | 1.263 ± 0.093 | 1.398 ± 0.112 | 1.652 ± 0.113 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 5.743 ± 0.556 | - | 5.146 ± 1.047 | 5.283 ± 1.144 | 5.725 ± 0.403 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.062 ± 0.038 | - | 1.046 ± 0.069 | 1.013 ± 0.059 | 1.169 ± 0.129 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.058 ± 0.000 | - | 0.060 ± 0.000 | 0.073 ± 0.001 | 0.124 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.003 ± 0.002 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | 0.008 ± 0.001 | 0.009 ± 0.000 | 0.018 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.001 | 0.008 ± 0.002 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.016 ± 0.008 | - | 0.017 ± 0.004 | 0.019 ± 0.001 | 0.031 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.012 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.004 | 0.010 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.024 ± 0.005 | - | 0.022 ± 0.008 | 0.024 ± 0.005 | 0.039 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 8.662 ± 0.597 | - | 8.703 ± 0.400 | 8.094 ± 0.120 | 8.913 ± 0.172 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.452 ± 0.034 | - | 0.536 ± 0.058 | 0.560 ± 0.018 | 0.642 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.011 ± 0.000 | - | 0.012 ± 0.003 | 0.014 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.039 ± 0.000 | - | 0.041 ± 0.009 | 0.044 ± 0.003 | 0.104 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.239 ± 0.014 | - | 0.252 ± 0.025 | 0.250 ± 0.008 | 1.288 ± 0.626 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.419 ± 1.147 | - | 1.285 ± 0.026 | 1.372 ± 0.114 | 2.003 ± 0.763 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 4.942 ± 0.091 | - | 5.020 ± 0.117 | 5.554 ± 0.927 | 4.421 ± 1.139 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 0.992 ± 0.019 | - | 1.017 ± 0.043 | 1.043 ± 0.050 | 2.111 ± 1.550 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.020 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.071 ± 0.001 | - | 0.172 ± 0.002 | 0.076 ± 0.002 | 0.153 ± 0.069 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.002 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.007 ± 0.000 | - | 0.025 ± 0.001 | 0.009 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.006 ± 0.001 | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.044 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.016 ± 0.001 | - | 0.018 ± 0.000 | 0.019 ± 0.003 | 0.027 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.004 ± 0.001 | - | 0.005 ± 0.000 | 0.006 ± 0.001 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.021 ± 0.000 | - | 0.063 ± 0.000 | 0.023 ± 0.005 | 0.085 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 7.855 ± 0.078 | - | 7.857 ± 1.251 | 8.666 ± 0.668 | 5.197 ± 0.403 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.004 ± 0.001 | - | 0.005 ± 0.000 | 0.007 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.552 ± 0.008 | - | 0.558 ± 0.019 | 0.559 ± 0.019 | 0.570 ± 0.262 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.011 ± 0.000 | - | 0.015 ± 0.003 | 0.014 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.106 ± 0.001 | - | 0.113 ± 0.000 | 0.047 ± 0.006 | 0.105 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.153 ± 0.051 | - | 0.185 ± 0.013 | 0.128 ± 0.016 | 0.562 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 0.828 ± 0.112 | - | 0.755 ± 0.048 | 0.783 ± 0.066 | 1.654 ± 0.139 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 2.390 ± 0.091 | - | 2.396 ± 0.053 | 2.594 ± 0.150 | 6.245 ± 0.295 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.455 ± 0.064 | - | 0.444 ± 0.019 | 0.474 ± 0.052 | 1.123 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.001 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.025 ± 0.000 | - | 0.028 ± 0.005 | 0.030 ± 0.001 | 0.105 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.001 | - | 0.005 ± 0.000 | 0.007 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.001 | 0.013 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.001 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.002 | 0.008 ± 0.000 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.013 ± 0.006 | - | 0.011 ± 0.000 | 0.015 ± 0.000 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 4.109 ± 0.472 | - | 3.816 ± 0.328 | 3.879 ± 0.108 | 8.879 ± 0.595 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.002 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.189 ± 0.001 | - | 0.191 ± 0.001 | 0.214 ± 0.017 | 0.682 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.019 ± 0.000 | - | 0.020 ± 0.000 | 0.023 ± 0.001 | 0.108 ± 0.067 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.117 ± 0.021 | - | 0.122 ± 0.012 | 0.128 ± 0.002 | 0.564 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 0.760 ± 0.055 | - | 0.822 ± 0.086 | 0.774 ± 0.036 | 1.466 ± 0.387 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 2.410 ± 0.071 | - | 2.431 ± 0.083 | 2.748 ± 0.169 | 2.688 ± 0.304 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.554 ± 0.048 | - | 0.536 ± 0.025 | 0.485 ± 0.040 | 1.084 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.026 ± 0.000 | - | 0.029 ± 0.000 | 0.030 ± 0.001 | 0.155 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.005 ± 0.001 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.005 ± 0.001 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.013 ± 0.003 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.005 ± 0.001 | - | 0.006 ± 0.001 | 0.008 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.011 ± 0.000 | - | 0.015 ± 0.003 | 0.014 ± 0.002 | 0.027 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 4.129 ± 0.026 | - | 4.202 ± 0.067 | 3.989 ± 0.253 | 4.594 ± 0.518 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.001 | 0.006 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.205 ± 0.007 | - | 0.249 ± 0.115 | 0.208 ± 0.024 | 0.366 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.007 ± 0.000 | - | 0.010 ± 0.001 | 0.011 ± 0.003 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.020 ± 0.000 | - | 0.021 ± 0.000 | 0.028 ± 0.006 | 0.102 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.153 ± 0.030 | - | 0.193 ± 0.057 | 0.174 ± 0.015 | 0.140 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.214 ± 0.028 | - | 0.228 ± 0.096 | 0.265 ± 0.037 | 0.205 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.614 ± 0.171 | - | 0.576 ± 0.185 | 0.546 ± 0.057 | 0.487 ± 0.077 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.064 ± 0.017 | - | 0.082 ± 0.163 | 0.071 ± 0.003 | 0.130 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.029 | 0.007 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.009 ± 0.000 | - | 0.011 ± 0.003 | 0.010 ± 0.000 | 0.059 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.009 | 0.006 ± 0.002 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.014 ± 0.001 | 0.049 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.008 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.001 | 0.009 ± 0.002 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.004 ± 0.001 | - | 0.006 ± 0.001 | 0.009 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.074 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.001 | 0.010 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.001 | 0.011 ± 0.002 | 0.063 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.546 ± 0.069 | - | 0.731 ± 0.196 | 0.530 ± 0.078 | 0.634 ± 0.067 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.004 ± 0.001 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.042 ± 0.018 | - | 0.093 ± 0.081 | 0.038 ± 0.002 | 0.088 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.002 | 0.044 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.002 | 0.043 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.091 ± 0.006 | - | 0.092 ± 0.001 | 0.377 ± 0.159 | 0.234 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.191 ± 0.011 | - | 0.193 ± 0.012 | 0.425 ± 0.106 | 0.249 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.442 ± 0.021 | - | 0.448 ± 0.032 | 0.890 ± 0.129 | 0.609 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.063 ± 0.001 | - | 0.064 ± 0.000 | 0.187 ± 0.041 | 0.147 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.070 ± 0.038 | 0.010 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.143 ± 0.041 | 0.104 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.091 ± 0.026 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.009 ± 0.001 | - | 0.010 ± 0.000 | 0.078 ± 0.101 | 0.079 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.105 ± 0.046 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.001 | 0.091 ± 0.048 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.004 ± 0.000 | 0.098 ± 0.021 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.005 ± 0.001 | - | 0.005 ± 0.000 | 0.096 ± 0.070 | 0.099 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.004 ± 0.001 | 0.069 ± 0.070 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.009 ± 0.002 | - | 0.005 ± 0.001 | 0.143 ± 0.100 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.004 ± 0.001 | - | 0.005 ± 0.003 | 0.097 ± 0.082 | 0.085 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.480 ± 0.006 | - | 0.488 ± 0.006 | 0.884 ± 0.240 | 0.831 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.004 ± 0.001 | - | 0.004 ± 0.001 | 0.130 ± 0.026 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.033 ± 0.000 | - | 0.033 ± 0.000 | 0.173 ± 0.073 | 0.196 ± 0.163 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.005 ± 0.001 | - | 0.005 ± 0.000 | 0.224 ± 0.020 | 0.073 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.006 ± 0.001 | - | 0.007 ± 0.000 | 0.162 ± 0.078 | 0.068 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.270 ± 0.009 | - | 0.319 ± 0.210 | 0.344 ± 0.052 | 0.429 ± 0.141 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.518 ± 0.023 | - | 0.501 ± 0.030 | 0.718 ± 0.040 | 0.947 ± 0.254 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.285 ± 0.080 | - | 1.341 ± 0.316 | 1.672 ± 0.106 | 1.566 ± 0.428 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.333 ± 0.047 | - | 0.335 ± 0.045 | 0.379 ± 0.138 | 1.283 ± 0.159 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.001 | - | 0.005 ± 0.001 | 0.003 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.013 ± 0.003 | - | 0.013 ± 0.000 | 0.024 ± 0.002 | 0.063 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.006 ± 0.004 | - | 0.006 ± 0.005 | 0.003 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.018 ± 0.009 | - | 0.019 ± 0.004 | 0.019 ± 0.004 | 0.152 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.009 ± 0.002 | - | 0.004 ± 0.000 | 0.006 ± 0.004 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.012 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.004 ± 0.001 | - | 0.006 ± 0.001 | 0.005 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.006 ± 0.001 | - | 0.008 ± 0.002 | 0.010 ± 0.005 | 0.044 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.006 ± 0.000 | - | 0.009 ± 0.001 | 0.013 ± 0.008 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 2.654 ± 0.255 | - | 3.966 ± 2.344 | 2.888 ± 0.121 | 3.558 ± 1.301 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.088 ± 0.000 | - | 0.100 ± 0.026 | 0.138 ± 0.008 | 0.183 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.006 ± 0.002 | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.014 ± 0.003 | - | 0.025 ± 0.000 | 0.018 ± 0.006 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.196 ± 0.018 | - | 0.196 ± 0.001 | 0.386 ± 0.144 | 0.606 ± 0.075 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.511 ± 0.013 | - | 0.515 ± 0.017 | 0.747 ± 0.048 | 1.050 ± 0.107 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.246 ± 0.042 | - | 1.586 ± 0.069 | 1.926 ± 1.269 | 1.353 ± 0.289 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.313 ± 0.019 | - | 0.312 ± 0.017 | 0.467 ± 0.274 | 0.498 ± 0.073 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.001 | - | 0.004 ± 0.009 | 0.062 ± 0.012 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.023 ± 0.004 | - | 0.021 ± 0.000 | 0.076 ± 0.010 | 0.116 ± 0.061 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.003 ± 0.001 | - | 0.004 ± 0.017 | 0.062 ± 0.037 | 0.011 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.017 ± 0.004 | - | 0.016 ± 0.000 | 0.036 ± 0.016 | 0.089 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.009 ± 0.001 | - | 0.004 ± 0.000 | 0.059 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.066 ± 0.035 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.004 ± 0.001 | - | 0.005 ± 0.000 | 0.060 ± 0.001 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.007 ± 0.001 | - | 0.007 ± 0.002 | 0.071 ± 0.033 | 0.017 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.013 ± 0.001 | 0.062 ± 0.024 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.061 ± 0.041 | 0.038 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.007 ± 0.000 | - | 0.008 ± 0.002 | 0.066 ± 0.003 | 0.017 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 2.568 ± 0.086 | - | 2.543 ± 0.077 | 3.750 ± 2.050 | 2.927 ± 2.205 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.009 ± 0.001 | - | 0.016 ± 0.011 | 0.037 ± 0.018 | 0.017 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.147 ± 0.021 | - | 0.138 ± 0.005 | 0.174 ± 0.030 | 0.460 ± 0.131 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.063 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.012 ± 0.000 | - | 0.013 ± 0.000 | 0.083 ± 0.041 | 0.033 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.134 ± 0.006 | - | 0.288 ± 0.001 | 0.123 ± 0.014 | 0.167 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.518 ± 0.058 | - | 0.440 ± 0.304 | 0.216 ± 0.021 | 0.245 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.333 ± 0.034 | - | 0.338 ± 0.014 | 0.415 ± 0.179 | 0.508 ± 0.147 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.051 ± 0.004 | - | 0.061 ± 0.002 | 0.067 ± 0.006 | 0.122 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.017 ± 0.035 | 0.010 ± 0.001 | 0.011 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.001 | 0.046 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.013 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.002 | 0.011 ± 0.002 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.012 ± 0.003 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.005 ± 0.000 | - | 0.015 ± 0.001 | 0.012 ± 0.003 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.003 | 0.012 ± 0.002 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.001 | 0.012 ± 0.001 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.005 ± 0.000 | - | 0.015 ± 0.000 | 0.012 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.398 ± 0.013 | - | 0.448 ± 0.063 | 0.415 ± 0.020 | 0.696 ± 0.201 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.027 ± 0.000 | - | 0.032 ± 0.000 | 0.085 ± 0.093 | 0.100 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.007 ± 0.000 | - | 0.016 ± 0.000 | 0.013 ± 0.001 | 0.011 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.003 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.140 ± 0.000 | - | 0.091 ± 0.001 | 0.161 ± 0.004 | 0.230 ± 0.070 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.166 ± 0.016 | - | 0.154 ± 0.013 | 0.232 ± 0.026 | 0.657 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.344 ± 0.029 | - | 0.352 ± 0.015 | 0.442 ± 0.028 | 0.661 ± 0.193 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.052 ± 0.003 | - | 0.062 ± 0.000 | 0.068 ± 0.003 | 0.259 ± 0.072 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.005 ± 0.000 | - | 0.009 ± 0.014 | 0.010 ± 0.001 | 0.017 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.013 ± 0.001 | 0.058 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.002 | 0.010 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.042 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.003 ± 0.001 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.021 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.003 ± 0.001 | - | 0.021 ± 0.000 | 0.012 ± 0.000 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.001 | 0.015 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.389 ± 0.011 | - | 0.535 ± 0.678 | 0.498 ± 0.024 | 0.918 ± 0.325 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.068 ± 0.005 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.028 ± 0.000 | - | 0.034 ± 0.000 | 0.125 ± 0.037 | 0.162 ± 0.191 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.003 ± 0.001 | - | 0.007 ± 0.000 | 0.069 ± 0.002 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.005 ± 0.001 | - | 0.008 ± 0.000 | 0.068 ± 0.005 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.228 ± 0.064 | - | 1.257 ± 0.037 | 1.247 ± 0.043 | 3.151 ± 0.133 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 4.547 ± 0.664 | - | 4.506 ± 0.626 | 4.670 ± 0.306 | 6.333 ± 0.335 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 15.017 ± 0.376 | - | 15.495 ± 0.797 | 15.457 ± 0.259 | 20.692 ± 4.214 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 6.062 ± 0.064 | - | 6.102 ± 0.208 | 4.729 ± 0.220 | 6.908 ± 0.582 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.182 ± 0.008 | - | 0.198 ± 0.014 | 0.195 ± 0.013 | 0.389 ± 0.088 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.002 | 0.010 ± 0.000 | 0.021 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.462 ± 0.021 | - | 0.180 ± 0.262 | 0.182 ± 0.060 | 0.573 ± 0.172 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.019 ± 0.004 | - | 0.019 ± 0.005 | 0.026 ± 0.005 | 0.032 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.014 ± 0.003 | - | 0.017 ± 0.004 | 0.018 ± 0.000 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.049 ± 0.011 | - | 0.049 ± 0.008 | 0.054 ± 0.007 | 0.064 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.011 ± 0.002 | - | 0.012 ± 0.003 | 0.010 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.019 ± 0.005 | - | 0.020 ± 0.005 | 0.026 ± 0.005 | 0.149 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.063 ± 0.010 | - | 0.062 ± 0.011 | 0.065 ± 0.001 | 0.149 ± 0.244 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 48.912 ± 0.903 | - | 60.003 ± 8.270 | 37.256 ± 0.556 | 48.927 ± 6.226 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.018 ± 0.003 | - | 0.017 ± 0.001 | 0.015 ± 0.000 | 0.037 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.438 ± 0.033 | - | 1.516 ± 0.107 | 1.465 ± 0.057 | 2.335 ± 1.145 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.038 ± 0.009 | - | 0.037 ± 0.003 | 0.042 ± 0.006 | 0.059 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.116 ± 0.004 | - | 0.118 ± 0.002 | 0.123 ± 0.013 | 0.427 ± 0.217 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.286 ± 0.036 | - | 1.315 ± 0.106 | 1.731 ± 0.478 | 3.851 ± 0.810 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 4.718 ± 0.125 | - | 4.713 ± 0.170 | 4.817 ± 0.209 | 4.384 ± 1.660 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 15.245 ± 0.832 | - | 14.762 ± 0.508 | 17.461 ± 1.132 | 9.660 ± 1.910 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 6.057 ± 0.135 | - | 6.014 ± 0.036 | 4.806 ± 1.256 | 5.586 ± 0.369 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.008 ± 0.002 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.186 ± 0.008 | - | 0.184 ± 0.002 | 0.188 ± 0.014 | 0.297 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.009 ± 0.017 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.174 ± 0.017 | - | 0.172 ± 0.022 | 0.177 ± 0.009 | 0.283 ± 0.058 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.028 ± 0.009 | - | 0.024 ± 0.008 | 0.022 ± 0.006 | 0.033 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.008 ± 0.005 | - | 0.010 ± 0.000 | 0.012 ± 0.002 | 0.021 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.015 ± 0.004 | - | 0.017 ± 0.004 | 0.019 ± 0.004 | 0.035 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.051 ± 0.010 | - | 0.049 ± 0.003 | 0.050 ± 0.001 | 0.065 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.009 ± 0.003 | - | 0.011 ± 0.002 | 0.014 ± 0.003 | 0.065 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.019 ± 0.005 | - | 0.020 ± 0.005 | 0.024 ± 0.000 | 0.045 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.081 ± 0.008 | - | 0.090 ± 0.023 | 0.066 ± 0.001 | 0.092 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 47.784 ± 0.515 | - | 47.982 ± 0.391 | 39.461 ± 0.993 | 24.263 ± 1.416 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.011 ± 0.002 | - | 0.013 ± 0.003 | 0.018 ± 0.004 | 0.071 ± 0.154 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 1.467 ± 0.059 | - | 1.514 ± 0.280 | 1.580 ± 0.247 | 1.214 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.041 ± 0.004 | - | 0.046 ± 0.008 | 0.040 ± 0.001 | 0.060 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.146 ± 0.018 | - | 0.120 ± 0.012 | 0.124 ± 0.023 | 0.222 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.570 ± 0.126 | - | 0.837 ± 0.509 | 0.598 ± 0.056 | 2.215 ± 0.297 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 2.712 ± 0.091 | - | 2.752 ± 0.035 | 3.430 ± 0.666 | 5.096 ± 2.360 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 12.481 ± 2.081 | - | 10.748 ± 0.883 | 10.933 ± 0.151 | 13.539 ± 1.518 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.514 ± 0.125 | - | 1.533 ± 0.099 | 1.589 ± 0.123 | 2.137 ± 0.416 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.003 ± 0.003 | - | 0.003 ± 0.000 | 0.006 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.123 ± 0.002 | - | 0.123 ± 0.002 | 0.130 ± 0.012 | 0.205 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.002 | 0.008 ± 0.001 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.098 ± 0.006 | - | 0.104 ± 0.004 | 0.106 ± 0.017 | 0.140 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.012 ± 0.000 | - | 0.013 ± 0.000 | 0.021 ± 0.004 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.011 ± 0.000 | - | 0.013 ± 0.002 | 0.013 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.029 ± 0.001 | - | 0.031 ± 0.004 | 0.036 ± 0.001 | 0.047 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.014 ± 0.004 | - | 0.016 ± 0.000 | 0.017 ± 0.003 | 0.030 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.039 ± 0.001 | - | 0.041 ± 0.000 | 0.044 ± 0.002 | 0.069 ± 0.145 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 13.033 ± 0.849 | - | 12.609 ± 1.147 | 12.458 ± 0.357 | 13.227 ± 0.669 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.002 | 0.010 ± 0.000 | 0.033 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 0.996 ± 0.028 | - | 0.980 ± 0.029 | 0.994 ± 0.007 | 1.147 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.026 ± 0.000 | - | 0.028 ± 0.000 | 0.027 ± 0.003 | 0.046 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.079 ± 0.004 | - | 0.080 ± 0.001 | 0.086 ± 0.010 | 0.181 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.586 ± 0.024 | - | 0.583 ± 0.048 | 0.792 ± 0.147 | 2.085 ± 0.200 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 2.837 ± 0.227 | - | 2.810 ± 0.264 | 3.163 ± 0.460 | 2.167 ± 0.375 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 11.913 ± 0.861 | - | 10.818 ± 0.393 | 12.207 ± 1.110 | 6.936 ± 0.434 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 1.615 ± 0.073 | - | 1.591 ± 0.111 | 1.704 ± 0.247 | 1.927 ± 0.407 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.124 ± 0.002 | - | 0.125 ± 0.001 | 0.134 ± 0.017 | 0.216 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.007 ± 0.007 | - | 0.019 ± 0.007 | 0.008 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.098 ± 0.017 | - | 0.099 ± 0.008 | 0.109 ± 0.003 | 0.140 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.013 ± 0.000 | - | 0.014 ± 0.000 | 0.016 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.005 ± 0.001 | - | 0.005 ± 0.000 | 0.007 ± 0.002 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.025 ± 0.001 | - | 0.014 ± 0.009 | 0.014 ± 0.003 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.034 ± 0.007 | - | 0.031 ± 0.000 | 0.034 ± 0.001 | 0.044 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.005 ± 0.001 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.019 ± 0.008 | - | 0.015 ± 0.000 | 0.020 ± 0.002 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.040 ± 0.006 | - | 0.042 ± 0.004 | 0.045 ± 0.001 | 0.058 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 13.383 ± 0.571 | - | 13.419 ± 5.429 | 13.789 ± 1.185 | 8.692 ± 1.106 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.001 | 0.028 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 1.062 ± 0.101 | - | 1.044 ± 0.030 | 0.977 ± 0.053 | 0.863 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.024 ± 0.000 | - | 0.025 ± 0.001 | 0.029 ± 0.014 | 0.045 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.079 ± 0.003 | - | 0.080 ± 0.000 | 0.090 ± 0.019 | 0.184 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.209 ± 0.003 | - | unsupported | 0.277 ± 0.039 | 3.141 ± 0.993 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.568 ± 0.059 | - | unsupported | 1.468 ± 0.252 | 2.792 ± 0.403 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 4.589 ± 0.690 | - | unsupported | 4.158 ± 0.165 | 5.525 ± 0.138 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 1.106 ± 0.061 | - | unsupported | 0.901 ± 0.070 | 0.855 ± 0.085 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | unsupported | 0.005 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.038 ± 0.001 | - | unsupported | 0.051 ± 0.038 | 0.047 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.037 ± 0.001 | - | unsupported | 0.034 ± 0.003 | 0.048 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.005 ± 0.001 | - | unsupported | 0.009 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | unsupported | 0.011 ± 0.001 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.013 ± 0.000 | - | unsupported | 0.016 ± 0.003 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.014 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.006 ± 0.002 | - | unsupported | 0.012 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.020 ± 0.004 | - | unsupported | 0.020 ± 0.000 | 0.033 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 8.607 ± 0.785 | - | unsupported | 6.636 ± 2.770 | 6.924 ± 0.234 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.008 ± 0.001 | 0.023 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.332 ± 0.054 | - | unsupported | 0.310 ± 0.039 | 0.327 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.011 ± 0.000 | - | unsupported | 0.018 ± 0.003 | 0.030 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.034 ± 0.008 | - | unsupported | 0.034 ± 0.000 | 0.048 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.209 ± 0.015 | - | unsupported | 0.240 ± 0.027 | 1.418 ± 0.109 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 1.354 ± 0.029 | - | unsupported | 1.477 ± 0.241 | 1.945 ± 0.200 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 3.904 ± 0.093 | - | unsupported | 4.441 ± 0.368 | 3.175 ± 0.459 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 0.779 ± 0.023 | - | unsupported | 0.817 ± 0.065 | 0.819 ± 0.067 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.038 ± 0.000 | - | unsupported | 0.046 ± 0.010 | 0.047 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.031 ± 0.008 | - | unsupported | 0.035 ± 0.001 | 0.042 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | unsupported | 0.009 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.012 ± 0.000 | - | unsupported | 0.020 ± 0.002 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.007 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.016 ± 0.000 | - | unsupported | 0.020 ± 0.000 | 0.030 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 6.307 ± 0.032 | - | unsupported | 6.781 ± 0.737 | 4.143 ± 0.725 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.023 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.293 ± 0.009 | - | unsupported | 0.308 ± 0.066 | 0.283 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.012 ± 0.000 | - | unsupported | 0.017 ± 0.004 | 0.039 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.031 ± 0.004 | - | unsupported | 0.036 ± 0.002 | 0.049 ± 0.002 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.913 ± 0.045 | - | unsupported | 0.892 ± 0.077 | 1.205 ± 0.021 | 5.996 ± 4.080 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.065 ± 0.009 | - | unsupported | 0.077 ± 0.012 | 0.136 ± 0.007 | 0.151 ± 0.091 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.256 ± 0.186 | - | unsupported | 0.120 ± 0.036 | 0.199 ± 0.019 | 0.234 ± 0.042 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.262 ± 0.119 | - | unsupported | 0.328 ± 0.036 | 0.398 ± 0.019 | 1.208 ± 1.078 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.819 ± 0.026 | - | unsupported | 0.943 ± 0.752 | 1.063 ± 0.040 | 3.657 ± 0.308 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.057 ± 0.002 | - | unsupported | 0.203 ± 0.100 | 0.175 ± 0.022 | 0.092 ± 0.007 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.113 ± 0.009 | - | unsupported | 0.279 ± 0.069 | 0.471 ± 0.238 | 0.165 ± 0.008 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.263 ± 0.036 | - | unsupported | 0.401 ± 0.085 | 0.517 ± 0.093 | 0.808 ± 1.015 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.251 ± 0.027 | - | unsupported | 0.220 ± 0.023 | 0.490 ± 0.149 | 0.604 ± 0.078 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.069 ± 0.011 | - | unsupported | 0.049 ± 0.007 | 0.090 ± 0.016 | 0.126 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.065 ± 0.016 | - | unsupported | 0.057 ± 0.009 | 0.103 ± 0.008 | 0.094 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.100 ± 0.006 | - | unsupported | 0.085 ± 0.012 | 0.179 ± 0.014 | 0.251 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.234 ± 0.076 | - | unsupported | 0.222 ± 0.031 | 0.943 ± 0.777 | 1.023 ± 0.314 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.067 ± 0.001 | - | unsupported | 0.138 ± 0.001 | 0.144 ± 0.098 | 0.078 ± 0.005 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.098 ± 0.010 | - | unsupported | 0.051 ± 0.003 | 0.321 ± 0.042 | 0.095 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.103 ± 0.010 | - | unsupported | 0.085 ± 0.010 | 0.478 ± 0.029 | 0.391 ± 0.086 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.116 ± 0.017 | - | 0.127 ± 0.027 | 0.100 ± 0.010 | 0.366 ± 0.244 | 0.424 ± 0.175 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.134 ± 0.140 | - | 0.050 ± 0.010 | 0.042 ± 0.013 | 0.025 ± 0.001 | 0.047 ± 0.011 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.054 ± 0.010 | - | 0.051 ± 0.008 | 0.033 ± 0.020 | 0.032 ± 0.004 | 0.106 ± 0.029 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.065 ± 0.010 | - | 0.058 ± 0.014 | 0.044 ± 0.011 | 0.062 ± 0.008 | 0.248 ± 0.079 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.112 ± 0.012 | - | 0.119 ± 0.018 | 0.093 ± 0.028 | 0.300 ± 0.119 | 0.291 ± 0.057 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.048 ± 0.000 | - | 0.045 ± 0.010 | 0.028 ± 0.007 | 0.025 ± 0.001 | 0.045 ± 0.004 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.045 ± 0.003 | - | 0.048 ± 0.001 | 0.033 ± 0.015 | 0.032 ± 0.002 | 0.065 ± 0.006 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.056 ± 0.001 | - | 0.059 ± 0.009 | 0.048 ± 0.019 | 0.077 ± 0.005 | 0.127 ± 0.002 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 9.606 ± 0.105 | - | 9.330 ± 0.793 | 25.038 ± 0.159 | 18.687 ± 4.749 | 37.940 ± 7.473 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 10.571 ± 3.598 | - | 10.597 ± 1.691 | 21.624 ± 1.891 | 15.273 ± 2.915 | 346.474 ± 5.948 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 8.255 ± 0.335 | - | 8.135 ± 0.353 | 7.978 ± 0.534 | 11.803 ± 2.502 | 21.692 ± 5.283 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 10.489 ± 3.590 | - | 10.086 ± 0.753 | 8.326 ± 0.832 | 8.584 ± 0.444 | 253.859 ± 42.134 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 6.864 ± 0.091 | - | 6.877 ± 0.098 | 7.108 ± 0.267 | 8.770 ± 0.070 | 7.656 ± 1.294 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 6.836 ± 0.475 | - | 6.790 ± 0.541 | 7.308 ± 0.095 | 8.360 ± 0.087 | 298.090 ± 11.092 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 7.779 ± 0.454 | - | 7.738 ± 0.056 | 8.074 ± 0.119 | 7.540 ± 0.612 | 7.991 ± 2.791 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 7.630 ± 0.566 | - | 7.718 ± 0.347 | 8.464 ± 0.340 | 8.594 ± 3.759 | 327.117 ± 293.861 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 21.158 ± 0.210 | - | 27.124 ± 4.570 | 22.326 ± 1.265 | 28.863 ± 0.840 | 14.014 ± 3.412 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 21.196 ± 0.594 | - | 21.347 ± 0.967 | 22.658 ± 0.549 | 32.449 ± 3.359 | 658.409 ± 43.673 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.294 ± 0.037 | 0.308 ± 0.029 | unsupported | 0.321 ± 0.006 | 0.407 ± 0.124 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 3.684 ± 0.107 | 4.381 ± 0.122 | unsupported | 4.079 ± 0.297 | 4.945 ± 0.692 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.296 ± 0.010 | 0.307 ± 0.011 | unsupported | 0.325 ± 0.012 | 0.472 ± 0.042 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 3.835 ± 0.119 | 4.825 ± 1.711 | unsupported | 4.196 ± 1.044 | 4.752 ± 0.444 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 8.489 ± 0.320 | - | 9.236 ± 0.107 | 9.873 ± 2.514 | 12.967 ± 4.484 | 36.092 ± 6.389 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 11.713 ± 3.313 | - | 14.755 ± 6.289 | 11.367 ± 1.501 | 10.442 ± 1.054 | 389.343 ± 223.322 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 9.032 ± 0.279 | 8.711 ± 0.135 | 5.455 ± 0.339 | 35.061 ± 9.709 | 12.888 ± 5.367 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 12.852 ± 2.542 | 11.346 ± 1.100 | 5.556 ± 0.350 | 46.021 ± 6.002 | 136.062 ± 3.718 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 8.629 ± 0.204 | - | 8.804 ± 0.745 | 13.550 ± 1.023 | 15.955 ± 2.596 | 33.972 ± 9.288 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 15.172 ± 7.167 | - | 9.530 ± 0.523 | 13.256 ± 1.664 | 16.885 ± 8.338 | 248.923 ± 30.188 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.872 ± 0.082 | - | 0.792 ± 0.120 | 2.336 ± 0.070 | 6.082 ± 1.850 | 6.142 ± 0.180 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.723 ± 0.059 | - | 0.824 ± 0.091 | 2.299 ± 0.206 | 5.194 ± 1.320 | 6.809 ± 2.358 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 11.730 ± 2.725 | - | 10.667 ± 0.104 | 11.476 ± 0.600 | 16.764 ± 2.457 | 21.899 ± 1.051 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 15.050 ± 6.270 | - | 13.354 ± 3.852 | 11.948 ± 1.755 | 13.955 ± 1.821 | 996.351 ± 59.399 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 10.774 ± 0.082 | - | 10.653 ± 0.421 | 11.170 ± 0.301 | 13.411 ± 0.226 | 22.132 ± 1.626 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 16.135 ± 3.372 | - | 21.390 ± 7.983 | 11.862 ± 0.159 | 19.605 ± 5.197 | 960.302 ± 54.012 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.460 ± 0.030 | 1.467 ± 0.022 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 16.326 ± 4.986 | 13.728 ± 0.389 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.479 ± 0.160 | 1.573 ± 0.160 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 16.517 ± 0.554 | 17.788 ± 2.283 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.476 ± 0.059 | 1.471 ± 0.020 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 12.710 ± 0.550 | 12.425 ± 1.130 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.587 ± 0.188 | 1.528 ± 0.107 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 13.034 ± 0.396 | 16.221 ± 2.679 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 8.270 ± 0.180 | - | 7.752 ± 0.181 | 9.226 ± 3.839 | 11.717 ± 6.157 | 21.465 ± 4.342 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 9.698 ± 0.899 | - | 9.574 ± 2.782 | 8.030 ± 0.522 | 8.708 ± 0.350 | 280.611 ± 223.696 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 34.657 ± 0.698 | 34.690 ± 1.692 | 36.218 ± 0.912 | 36.400 ± 1.213 | 84.606 ± 7.731 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 45.976 ± 5.528 | 45.445 ± 4.213 | 48.064 ± 2.085 | 57.181 ± 4.702 | 1695.042 ± 22.932 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 16.187 ± 3.719 | - | 15.995 ± 0.192 | 13.482 ± 0.497 | 71.573 ± 4.791 | 32.721 ± 0.703 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 19.064 ± 4.106 | - | 28.084 ± 7.257 | 13.959 ± 0.942 | 49.948 ± 2.580 | 22.403 ± 3.798 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.210 ± 0.112 | 0.219 ± 0.086 | unsupported | 0.060 ± 0.003 | 0.039 ± 0.005 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.809 ± 0.061 | 2.801 ± 0.067 | unsupported | 0.957 ± 0.166 | 0.493 ± 0.042 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.218 ± 0.082 | 0.208 ± 0.131 | unsupported | 0.090 ± 0.031 | 0.072 ± 0.053 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 1.627 ± 0.109 | 1.931 ± 0.334 | unsupported | 1.171 ± 0.246 | 0.921 ± 0.717 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.103 ± 0.042 | 0.096 ± 0.053 | unsupported | 0.097 ± 0.013 | 0.205 ± 0.004 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.261 ± 0.011 | 1.282 ± 0.022 | unsupported | 1.444 ± 0.091 | 2.511 ± 0.055 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.119 ± 0.025 | 0.121 ± 0.007 | unsupported | 0.141 ± 0.009 | 0.281 ± 0.003 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 1.391 ± 0.591 | 1.379 ± 0.397 | unsupported | 1.622 ± 0.447 | 3.210 ± 0.958 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.313 ± 0.037 | 0.278 ± 0.077 | unsupported | 0.317 ± 0.020 | 0.097 ± 0.003 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 4.287 ± 0.084 | 4.262 ± 0.245 | unsupported | 6.728 ± 1.632 | 0.951 ± 0.030 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.263 ± 0.011 | 0.262 ± 0.005 | unsupported | 0.284 ± 0.028 | 0.144 ± 0.035 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 4.311 ± 1.257 | 3.872 ± 2.237 | unsupported | 4.333 ± 1.214 | 1.457 ± 0.908 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.047 ± 0.001 | 0.066 ± 0.007 | unsupported | 0.048 ± 0.004 | 0.077 ± 0.001 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.587 ± 0.008 | 0.633 ± 0.010 | unsupported | 0.849 ± 0.219 | 0.711 ± 0.013 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.055 ± 0.027 | 0.086 ± 0.015 | unsupported | 0.106 ± 0.021 | 0.164 ± 0.051 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 1.857 ± 0.501 | 1.937 ± 0.216 | unsupported | 0.515 ± 0.136 | 0.905 ± 0.106 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.111 ± 0.074 | 0.117 ± 0.016 | unsupported | 0.105 ± 0.023 | 0.107 ± 0.002 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.213 ± 0.073 | 1.380 ± 0.348 | unsupported | 2.161 ± 0.950 | 1.276 ± 0.051 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.099 ± 0.025 | 0.120 ± 0.007 | unsupported | 0.139 ± 0.003 | 0.213 ± 0.004 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 2.059 ± 0.261 | 3.187 ± 0.687 | unsupported | 1.934 ± 0.725 | 1.455 ± 0.027 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.138 ± 0.016 | 0.151 ± 0.014 | unsupported | unsupported | 0.139 ± 0.010 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 1.813 ± 0.015 | 1.867 ± 0.009 | unsupported | unsupported | 1.685 ± 0.094 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.147 ± 0.026 | 0.165 ± 0.015 | unsupported | unsupported | 0.233 ± 0.017 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 3.584 ± 1.197 | 3.958 ± 1.123 | unsupported | unsupported | 2.066 ± 0.209 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.011 ± 0.000 | 0.017 ± 0.003 | unsupported | 0.008 ± 0.007 | 0.053 ± 0.004 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.104 ± 0.004 | 0.112 ± 0.006 | unsupported | 0.108 ± 0.221 | 0.288 ± 0.051 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.013 ± 0.001 | 0.019 ± 0.001 | unsupported | 0.061 ± 0.002 | 0.067 ± 0.004 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.102 ± 0.018 | 0.177 ± 0.141 | unsupported | 0.147 ± 0.084 | 0.349 ± 0.048 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.027 ± 0.006 | 0.038 ± 0.002 | unsupported | 0.029 ± 0.002 | 0.078 ± 0.002 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.333 ± 0.026 | 0.345 ± 0.041 | unsupported | 0.408 ± 0.256 | 0.641 ± 0.069 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.032 ± 0.005 | 0.044 ± 0.006 | unsupported | 0.147 ± 0.016 | 0.179 ± 0.022 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.735 ± 0.180 | 1.028 ± 0.291 | unsupported | 0.705 ± 0.373 | 0.815 ± 0.056 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.069 ± 0.022 | 0.057 ± 0.067 | unsupported | 0.083 ± 0.002 | 0.089 ± 0.005 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.791 ± 0.032 | 0.769 ± 0.015 | unsupported | 1.347 ± 0.443 | 0.857 ± 0.012 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.087 ± 0.123 | 0.097 ± 0.016 | unsupported | 0.132 ± 0.031 | 0.174 ± 0.002 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.668 ± 0.200 | 0.830 ± 0.091 | unsupported | 1.334 ± 0.205 | 0.980 ± 0.435 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.058 ± 0.021 | 0.052 ± 0.004 | unsupported | 0.083 ± 0.001 | 0.093 ± 0.012 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.800 ± 0.025 | 0.805 ± 0.027 | unsupported | 1.339 ± 0.021 | 0.931 ± 0.025 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.088 ± 0.018 | 0.095 ± 0.027 | unsupported | 0.133 ± 0.003 | 0.176 ± 0.002 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.781 ± 0.348 | 0.721 ± 0.103 | unsupported | 1.252 ± 0.263 | 1.143 ± 0.539 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.088 ± 0.025 | 0.106 ± 0.007 | unsupported | 0.106 ± 0.002 | 0.122 ± 0.008 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.233 ± 0.069 | 1.305 ± 0.046 | unsupported | 2.250 ± 0.359 | 1.306 ± 0.037 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.097 ± 0.007 | 0.119 ± 0.005 | unsupported | 0.143 ± 0.004 | 0.215 ± 0.007 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 3.251 ± 1.706 | 2.815 ± 1.115 | unsupported | 1.553 ± 0.525 | 1.346 ± 0.246 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.300 ± 0.091 | 0.268 ± 0.102 | unsupported | 0.297 ± 0.011 | 0.337 ± 0.224 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 3.555 ± 0.065 | 3.584 ± 0.123 | unsupported | 4.703 ± 0.171 | 4.465 ± 0.391 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.245 ± 0.010 | 0.245 ± 0.010 | unsupported | 0.316 ± 0.008 | 0.336 ± 0.062 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 3.441 ± 1.724 | 4.003 ± 1.119 | unsupported | 4.013 ± 0.459 | 5.815 ± 2.136 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.578 ± 0.017 | 0.712 ± 0.035 | unsupported | 0.127 ± 0.003 | 0.105 ± 0.006 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 9.564 ± 0.229 | 15.227 ± 1.385 | unsupported | 2.745 ± 0.920 | 1.359 ± 0.124 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.868 ± 0.202 | 1.012 ± 0.118 | unsupported | 0.256 ± 0.035 | 0.222 ± 0.023 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 12.443 ± 2.136 | 13.323 ± 2.067 | unsupported | 2.763 ± 0.243 | 1.101 ± 0.065 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 19.699 ± 2.149 | - | unsupported | 20.752 ± 5.830 | - | 20.902 ± 2.664 | 33.861 ± 14.991 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 22.397 ± 4.063 | - | unsupported | 26.152 ± 5.867 | - | 14.112 ± 2.017 | 14.457 ± 0.559 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 11.500 ± 0.674 | - | unsupported | 12.421 ± 0.457 | - | 12.208 ± 5.576 | 15.778 ± 7.489 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 14.856 ± 1.385 | - | unsupported | 13.344 ± 4.601 | - | 9.136 ± 0.822 | 10.332 ± 1.437 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 10.666 ± 0.695 | - | unsupported | 11.964 ± 0.417 | - | 11.228 ± 0.474 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 15.478 ± 2.857 | - | unsupported | 14.635 ± 2.912 | - | 11.081 ± 0.291 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 18.489 ± 1.221 | - | unsupported | 20.610 ± 3.115 | - | 30.482 ± 4.156 | 22.151 ± 6.560 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 22.628 ± 4.133 | - | unsupported | 23.469 ± 1.764 | - | 15.811 ± 2.192 | 14.440 ± 0.252 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 10.597 ± 0.853 | - | unsupported | 11.267 ± 0.423 | - | 54.818 ± 0.743 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 11.973 ± 7.833 | - | unsupported | 19.369 ± 10.513 | - | 22.142 ± 2.946 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 11.868 ± 0.509 | - | unsupported | 12.345 ± 0.605 | - | 56.792 ± 5.163 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 12.184 ± 1.265 | - | unsupported | 13.664 ± 5.049 | - | 21.545 ± 1.772 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 19.139 ± 0.715 | - | unsupported | 22.932 ± 2.952 | - | 25.867 ± 6.782 | 26.080 ± 6.689 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 22.814 ± 3.329 | - | unsupported | 22.031 ± 15.431 | - | 14.335 ± 1.090 | 14.435 ± 0.664 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 12.094 ± 0.728 | - | unsupported | 11.928 ± 1.193 | - | 16.331 ± 8.649 | 17.135 ± 4.074 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 13.116 ± 0.612 | - | unsupported | 14.243 ± 2.021 | - | 9.182 ± 0.596 | 9.971 ± 0.552 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.205 ± 0.005 | unsupported | unsupported | 0.280 ± 0.023 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 2.737 ± 0.120 | unsupported | unsupported | 4.044 ± 0.590 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.231 ± 0.035 | unsupported | unsupported | 0.357 ± 0.073 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 3.469 ± 0.427 | unsupported | unsupported | 3.481 ± 0.170 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 18.781 ± 1.533 | - | unsupported | 20.362 ± 1.399 | - | 34.339 ± 13.835 | 26.163 ± 4.514 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 21.608 ± 7.692 | - | unsupported | 22.329 ± 2.826 | - | 14.438 ± 1.655 | 14.665 ± 0.830 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 8.729 ± 2.774 | - | 0.003 ± 0.000 | 6.709 ± 9.217 | 7.631 ± 1.728 | 14.500 ± 0.179 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 9.055 ± 1.724 | - | 0.001 ± 0.000 | 7.986 ± 2.439 | 5.702 ± 0.217 | 14.238 ± 4.851 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 9.522 ± 0.984 | - | 21.661 ± 4.145 | 14.945 ± 4.227 | 12.638 ± 5.509 | 8.777 ± 0.538 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 8.408 ± 0.648 | - | 18.847 ± 3.365 | 10.824 ± 1.368 | 10.420 ± 0.289 | 9.357 ± 0.920 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 11.433 ± 2.369 | - | 12.154 ± 0.685 | 15.823 ± 4.107 | 19.281 ± 2.468 | 9.865 ± 0.993 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 11.541 ± 1.905 | - | 9.827 ± 0.856 | 13.165 ± 0.797 | 13.022 ± 2.315 | 12.127 ± 1.779 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 6.061 ± 1.003 | - | 5.744 ± 0.668 | 10.902 ± 1.264 | 11.738 ± 2.945 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 5.097 ± 0.434 | - | 4.769 ± 0.351 | 12.853 ± 2.720 | 11.198 ± 1.220 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 10.803 ± 1.450 | - | 0.001 ± 0.000 | 15.862 ± 3.453 | 13.506 ± 4.569 | 10.081 ± 0.581 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 10.452 ± 0.446 | - | 0.001 ± 0.000 | 15.548 ± 3.805 | 11.456 ± 0.496 | 16.574 ± 7.777 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 42.798 ± 5.444 | - | 0.002 ± 0.008 | 67.765 ± 22.243 | 55.323 ± 3.785 | 226.515 ± 28.982 | 31.916 ± 0.364 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 21.312 ± 2.696 | - | 0.001 ± 0.000 | 55.062 ± 1.448 | 31.862 ± 3.143 | 219.693 ± 26.535 | 15.306 ± 6.956 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 8.698 ± 0.437 | - | 9.275 ± 2.500 | 9.596 ± 0.672 | 6.699 ± 0.672 | 8.065 ± 0.129 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 5.318 ± 0.323 | - | 5.680 ± 0.306 | 8.084 ± 1.633 | 6.579 ± 1.316 | 9.152 ± 0.412 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 9.792 ± 1.467 | - | 8.857 ± 2.903 | 10.602 ± 2.146 | 6.683 ± 0.490 | 8.812 ± 0.751 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 5.053 ± 0.393 | - | 5.156 ± 0.253 | 7.414 ± 0.810 | 6.623 ± 0.233 | 10.606 ± 3.817 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.003 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.002 ± 0.001 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.002 ± 0.001 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 4.650125 | 290632.81 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 4.353416 | 272088.50 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 4.639250 | 289953.12 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 6.548666 | 409291.62 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 3.971792 | 248237.00 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 4.867750 | 304234.38 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 14.241917 | 890119.81 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 14.723833 | 920239.56 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 11.514208 | 719638.00 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 11.399250 | 712453.12 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 6.514667 | 407166.69 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 5.132500 | 320781.25 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 4.702166 | 293885.38 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 4.930333 | 308145.81 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 79.120292 | 4945018.25 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 65.263083 | 4078942.69 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 58.947250 | 3684203.12 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 70.088542 | 4380533.88 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 7.557292 | 472330.75 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 5.195667 | 324729.19 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 4.740417 | 296276.06 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 4.914000 | 307125.00 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 76.029833 | 4751864.56 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 67.135291 | 4195955.69 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 61.367500 | 3835468.75 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 77.203750 | 4825234.38 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 23.359625 | 1459976.56 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 23.475542 | 1467221.38 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 261.211959 | 16325747.44 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 219.652708 | 13728294.25 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 23.670416 | 1479401.00 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 25.170792 | 1573174.50 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 264.275125 | 16517195.31 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 284.603167 | 17787697.94 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 23.613459 | 1475841.19 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 23.539542 | 1471221.38 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 203.363958 | 12710247.38 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 198.802833 | 12425177.06 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 25.386584 | 1586661.50 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 24.442250 | 1527640.62 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 208.536125 | 13033507.81 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 259.541792 | 16221362.00 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.475583 | 38680.98 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.812833 | 59575.52 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 13.418209 | 209659.52 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 14.016458 | 219007.16 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 3.946458 | 493307.25 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.652375 | 956546.88 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 22.470500 | 2808812.50 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 22.404334 | 2800541.75 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.593708 | 71776.69 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.782334 | 90348.97 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 13.933167 | 217705.73 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 13.327792 | 208246.75 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.370917 | 921364.62 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 9.368792 | 1171099.00 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 13.016916 | 1627114.50 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 15.450750 | 1931343.75 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.122625 | 205041.02 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.219375 | 97177.73 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.615292 | 103363.94 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.173500 | 96460.94 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 20.090500 | 2511312.50 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.552750 | 1444093.75 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.091667 | 1261458.38 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.254583 | 1281822.88 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 17.984500 | 281007.81 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 9.018500 | 140914.06 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 7.641375 | 119396.48 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.740750 | 120949.22 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 25.682709 | 3210338.62 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.973833 | 1621729.12 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 11.126291 | 1390786.38 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 11.029209 | 1378651.12 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.202750 | 96917.97 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 20.287042 | 316985.03 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 20.040208 | 313128.25 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 17.816750 | 278386.72 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.604000 | 950500.00 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 53.823666 | 6727958.25 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 34.294834 | 4286854.25 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 34.096750 | 4262093.75 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.207042 | 143860.03 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 18.144667 | 283510.42 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 16.851750 | 263308.59 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 16.796041 | 262438.14 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.657667 | 1457208.38 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 34.665500 | 4333187.50 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 34.490584 | 4311323.00 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 30.976250 | 3872031.25 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.951334 | 77364.59 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.071625 | 47994.14 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.991000 | 46734.38 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.225750 | 66027.34 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.691458 | 711432.25 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 6.795167 | 849395.88 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.698292 | 587286.50 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.064750 | 633093.75 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.526416 | 164475.25 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.757833 | 105591.14 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.516625 | 54947.27 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.500792 | 85949.88 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.239083 | 904885.38 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.116250 | 514531.25 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 14.855167 | 1856895.88 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 15.498250 | 1937281.25 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.857833 | 107153.64 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.736416 | 105256.50 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 7.088667 | 110760.42 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 7.494917 | 117108.08 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.206917 | 1275864.62 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 17.284417 | 2160552.12 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.701792 | 1212724.00 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.036125 | 1379515.62 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.621083 | 212829.42 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 8.870916 | 138608.06 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.346708 | 99167.31 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.672750 | 119886.72 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.640500 | 1455062.50 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 15.470875 | 1933859.38 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 16.475291 | 2059411.37 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 25.492875 | 3186609.38 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.927625 | 139494.14 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.808708 | 137636.06 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.638083 | 150595.05 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.476625 | 1684578.12 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 14.507583 | 1813447.88 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 14.933291 | 1866661.38 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 14.910458 | 232975.91 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 9.423333 | 147239.58 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 10.550541 | 164852.20 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 16.526708 | 2065838.50 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 28.669459 | 3583682.38 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 31.662625 | 3957828.12 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 3.369667 | 52651.05 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.529541 | 8274.08 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.734000 | 11468.75 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.073625 | 16775.39 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 2.305125 | 288140.62 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.866959 | 108369.88 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.834333 | 104291.62 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.894333 | 111791.62 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.309208 | 67331.38 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.931750 | 61433.59 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.855000 | 13359.38 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.203000 | 18796.88 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.791208 | 348901.00 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.175209 | 146901.12 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.817250 | 102156.25 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.419959 | 177494.88 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.971375 | 77677.73 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 1.869958 | 29218.09 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.754792 | 27418.62 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.400250 | 37503.91 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.125292 | 640661.50 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 3.262875 | 407859.38 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.666375 | 333296.88 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 2.762917 | 345364.62 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.467000 | 179171.88 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 9.412958 | 147077.47 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.025292 | 31645.19 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.785958 | 43530.59 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.522167 | 815270.88 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.642417 | 705302.12 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.876417 | 734552.12 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 8.224875 | 1028109.38 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.720417 | 89381.52 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.303833 | 82872.39 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.434750 | 69292.97 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.673958 | 57405.59 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.853333 | 856666.62 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.773125 | 1346640.62 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.328917 | 791114.62 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.148417 | 768552.12 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.137083 | 174016.92 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 8.429083 | 131704.42 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.567500 | 86992.19 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6.218292 | 97160.81 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.842708 | 980338.50 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 10.669666 | 1333708.25 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.343959 | 667994.88 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 6.636416 | 829552.00 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.940167 | 92815.11 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.291208 | 82675.12 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.728333 | 58255.20 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.300125 | 51564.45 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.450292 | 931286.50 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.708250 | 1338531.25 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.402167 | 800270.88 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.442666 | 805333.25 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.294209 | 176472.02 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 8.509125 | 132955.08 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.643333 | 88177.08 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6.079583 | 94993.48 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.147750 | 1143468.75 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 10.015250 | 1251906.25 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.247958 | 780994.75 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.765833 | 720729.12 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.805291 | 121957.67 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.758000 | 105593.75 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.610375 | 87662.11 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.752625 | 105509.77 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.447500 | 1305937.50 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 17.996375 | 2249546.88 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.864583 | 1233072.88 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.441083 | 1305135.38 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.744084 | 214751.31 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 9.141791 | 142840.48 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.237833 | 97466.14 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.586834 | 118544.28 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.771083 | 1346385.38 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.420208 | 1552526.00 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 26.010791 | 3251348.88 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 22.517875 | 2814734.38 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 21.550250 | 336722.66 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 19.036583 | 297446.61 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 19.189083 | 299829.42 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 17.158458 | 268100.91 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 35.722375 | 4465296.88 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 37.624000 | 4703000.00 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 28.443208 | 3555401.00 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 28.668959 | 3583619.88 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 21.492583 | 335821.61 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 20.237417 | 316209.64 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 15.682208 | 245034.50 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 15.688750 | 245136.72 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 46.521500 | 5815187.50 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 32.102583 | 4012822.88 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 27.524167 | 3440520.88 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 32.023750 | 4002968.75 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.696000 | 104625.00 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.135084 | 127110.69 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 37.016917 | 578389.33 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 45.541791 | 711590.48 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.871584 | 1358948.00 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 21.963958 | 2745494.75 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 76.511209 | 9563901.12 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 121.814042 | 15226755.25 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 14.184834 | 221638.03 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 16.392958 | 256139.97 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 55.521875 | 867529.30 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 64.782666 | 1012229.16 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.810334 | 1101291.75 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 22.104750 | 2763093.75 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 99.542625 | 12442828.12 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 106.585459 | 13323182.38 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 4.487167 | 280447.94 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 3.281125 | 205070.31 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 64.701333 | 4043833.31 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 43.799500 | 2737468.75 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 5.704541 | 356533.81 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 3.701042 | 231315.12 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 55.702708 | 3481419.25 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 55.506708 | 3469169.25 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.012792 | 799.50 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.001250 | 78.12 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.012959 | 809.94 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.001375 | 85.94 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000084 | 5.25 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.014417 | 901.06 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.035959 | 2247.44 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000125 | 7.81 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.012583 | 786.44 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.025750 | 1609.38 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000084 | 5.25 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.014209 | 888.06 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.002291 | 143.19 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000125 | 7.81 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.010750 | 671.88 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.004250 | 265.62 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000208 | 13.00 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.010625 | 664.06 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002666 | 166.62 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.011042 | 690.12 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002666 | 166.62 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `jax-cpu` is 24.3x faster than the slowest successful cell (`julia-base`, 75.298 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `pytorch-cpu` is 18.3x faster than the slowest successful cell (`julia-base`, 75.298 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-direct` is 19.1x faster than the slowest successful cell (`julia-base`, 75.298 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-trace` is 19.8x faster than the slowest successful cell (`julia-base`, 75.298 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `jax-cpu` is 20.8x faster than the slowest successful cell (`julia-base`, 440.327 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `pytorch-cpu` is 37.7x faster than the slowest successful cell (`julia-base`, 440.327 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-direct` is 36.6x faster than the slowest successful cell (`julia-base`, 440.327 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-trace` is 44.0x faster than the slowest successful cell (`julia-base`, 440.327 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 24.3x faster than the slowest successful cell (`julia-base`, 180.646 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 22.4x faster than the slowest successful cell (`julia-base`, 180.646 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-direct` is 21.6x faster than the slowest successful cell (`julia-base`, 180.646 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-trace` is 26.2x faster than the slowest successful cell (`julia-base`, 180.646 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 36.7x faster than the slowest successful cell (`julia-base`, 119.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 17.9x faster than the slowest successful cell (`julia-base`, 119.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-direct` is 21.6x faster than the slowest successful cell (`julia-base`, 119.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-trace` is 19.9x faster than the slowest successful cell (`julia-base`, 119.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `jax-cpu` is 85.7x faster than the slowest successful cell (`julia-base`, 617.766 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `pytorch-cpu` is 78.1x faster than the slowest successful cell (`julia-base`, 617.766 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-direct` is 94.2x faster than the slowest successful cell (`julia-base`, 617.766 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-trace` is 76.7x faster than the slowest successful cell (`julia-base`, 617.766 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 20.4x faster than the slowest successful cell (`pytorch-cpu`, 17.378 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 16.7x faster than the slowest successful cell (`pytorch-cpu`, 17.378 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 14.3x faster than the slowest successful cell (`pytorch-cpu`, 13.428 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 14.1x faster than the slowest successful cell (`pytorch-cpu`, 13.428 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 18.3x faster than the slowest successful cell (`pytorch-cpu`, 13.428 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 18.6x faster than the slowest successful cell (`pytorch-cpu`, 21.128 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 20.0x faster than the slowest successful cell (`pytorch-cpu`, 21.128 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 19.7x faster than the slowest successful cell (`pytorch-cpu`, 21.128 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 19.3x faster than the slowest successful cell (`pytorch-cpu`, 17.874 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 24.4x faster than the slowest successful cell (`pytorch-cpu`, 17.874 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 23.6x faster than the slowest successful cell (`pytorch-cpu`, 17.874 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 2153.1x faster than the slowest successful cell (`julia-base`, 3.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 395.9x faster than the slowest successful cell (`julia-base`, 1.501 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `pytorch-cpu` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 29.6x faster than the slowest successful cell (`jax-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-trace` is 15.7x faster than the slowest successful cell (`jax-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `pytorch-cpu` is 13.4x faster than the slowest successful cell (`jax-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 39.0x faster than the slowest successful cell (`jax-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 18.3x faster than the slowest successful cell (`jax-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 15.8x faster than the slowest successful cell (`jax-cpu`, 0.037 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 57.4x faster than the slowest successful cell (`pytorch-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 25.9x faster than the slowest successful cell (`pytorch-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 36.3x faster than the slowest successful cell (`jax-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-trace` is 20.9x faster than the slowest successful cell (`jax-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 42.3x faster than the slowest successful cell (`pytorch-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 52.4x faster than the slowest successful cell (`pytorch-cpu`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 25.1x faster than the slowest successful cell (`pytorch-cpu`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `jax-cpu` is 12.2x faster than the slowest successful cell (`pytorch-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 54.0x faster than the slowest successful cell (`pytorch-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 29.9x faster than the slowest successful cell (`pytorch-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `jax-cpu` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 27.2x faster than the slowest successful cell (`pytorch-cpu`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 27.5x faster than the slowest successful cell (`pytorch-cpu`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `jax-cpu` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 0.114 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 47.4x faster than the slowest successful cell (`pytorch-cpu`, 0.114 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 29.2x faster than the slowest successful cell (`pytorch-cpu`, 0.114 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 45.6x faster than the slowest successful cell (`pytorch-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 23.8x faster than the slowest successful cell (`pytorch-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `jax-cpu` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 51.8x faster than the slowest successful cell (`pytorch-cpu`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 25.1x faster than the slowest successful cell (`pytorch-cpu`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 47.0x faster than the slowest successful cell (`pytorch-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 30.7x faster than the slowest successful cell (`pytorch-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `jax-cpu` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 45.2x faster than the slowest successful cell (`pytorch-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 22.1x faster than the slowest successful cell (`pytorch-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x32x32`): `tenferro-direct` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.131 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 30.9x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 16.4x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 28.2x faster than the slowest successful cell (`pytorch-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=4, shape=`4x2x2`): `tenferro-direct` is 12.5x faster than the slowest successful cell (`jax-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.020 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 24.4x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-trace` is 18.0x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 14.9x faster than the slowest successful cell (`jax-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 12.7x faster than the slowest successful cell (`jax-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 10.0x faster than the slowest successful cell (`jax-cpu`, 0.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 26.8x faster than the slowest successful cell (`pytorch-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 19.4x faster than the slowest successful cell (`pytorch-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 22.0x faster than the slowest successful cell (`pytorch-cpu`, 0.143 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x32x32`): `tenferro-trace` is 20.3x faster than the slowest successful cell (`pytorch-cpu`, 0.143 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 29.1x faster than the slowest successful cell (`pytorch-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 25.4x faster than the slowest successful cell (`pytorch-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 37.1x faster than the slowest successful cell (`pytorch-cpu`, 0.105 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 26.8x faster than the slowest successful cell (`pytorch-cpu`, 0.105 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 27.3x faster than the slowest successful cell (`pytorch-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 24.8x faster than the slowest successful cell (`pytorch-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 26.1x faster than the slowest successful cell (`pytorch-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 23.7x faster than the slowest successful cell (`pytorch-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 20.7x faster than the slowest successful cell (`jax-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 20.9x faster than the slowest successful cell (`jax-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 19.1x faster than the slowest successful cell (`pytorch-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 18.3x faster than the slowest successful cell (`pytorch-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `jax-cpu` is 12.3x faster than the slowest successful cell (`pytorch-cpu`, 0.143 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 16.5x faster than the slowest successful cell (`pytorch-cpu`, 0.143 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 27.1x faster than the slowest successful cell (`pytorch-cpu`, 0.143 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 21.5x faster than the slowest successful cell (`pytorch-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 18.8x faster than the slowest successful cell (`pytorch-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `jax-cpu` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.130 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 30.0x faster than the slowest successful cell (`pytorch-cpu`, 0.130 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 31.5x faster than the slowest successful cell (`pytorch-cpu`, 0.130 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 46.4x faster than the slowest successful cell (`pytorch-cpu`, 0.224 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 45.6x faster than the slowest successful cell (`pytorch-cpu`, 0.224 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 29.0x faster than the slowest successful cell (`pytorch-cpu`, 0.162 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 24.3x faster than the slowest successful cell (`pytorch-cpu`, 0.162 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 31.0x faster than the slowest successful cell (`pytorch-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 14.0x faster than the slowest successful cell (`pytorch-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 22.3x faster than the slowest successful cell (`pytorch-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 16.4x faster than the slowest successful cell (`pytorch-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 13.1x faster than the slowest successful cell (`pytorch-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 27.1x faster than the slowest successful cell (`pytorch-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 17.1x faster than the slowest successful cell (`pytorch-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 16.0x faster than the slowest successful cell (`pytorch-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 24.7x faster than the slowest successful cell (`pytorch-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x32x32,rhs=1`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x32x32,rhs=1`): `tenferro-direct` is 11.7x faster than the slowest successful cell (`jax-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`3x4x4,rhs=1`): `tenferro-direct` is 14.0x faster than the slowest successful cell (`jax-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x2x2,rhs=1`): `tenferro-direct` is 20.3x faster than the slowest successful cell (`pytorch-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x4x4,rhs=1`): `tenferro-direct` is 22.0x faster than the slowest successful cell (`pytorch-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x8x8,rhs=1`): `tenferro-direct` is 14.3x faster than the slowest successful cell (`pytorch-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svdvals` (f64, threads=1, shape=`1024x2x2`): `pytorch-cpu` is 11.3x faster than the slowest successful cell (`jax-cpu`, 3.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svdvals` (f64, threads=1, shape=`1024x2x2`): `tenferro-direct` is 15.0x faster than the slowest successful cell (`jax-cpu`, 3.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `jax-cpu` is 22.7x faster than the slowest successful cell (`julia-base`, 346.474 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `pytorch-cpu` is 16.0x faster than the slowest successful cell (`julia-base`, 346.474 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-direct` is 32.8x faster than the slowest successful cell (`julia-base`, 346.474 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-trace` is 32.7x faster than the slowest successful cell (`julia-base`, 346.474 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 29.6x faster than the slowest successful cell (`julia-base`, 253.859 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 30.5x faster than the slowest successful cell (`julia-base`, 253.859 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 24.2x faster than the slowest successful cell (`julia-base`, 253.859 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 25.2x faster than the slowest successful cell (`julia-base`, 253.859 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `jax-cpu` is 35.7x faster than the slowest successful cell (`julia-base`, 298.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `pytorch-cpu` is 40.8x faster than the slowest successful cell (`julia-base`, 298.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-direct` is 43.6x faster than the slowest successful cell (`julia-base`, 298.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-trace` is 43.9x faster than the slowest successful cell (`julia-base`, 298.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `jax-cpu` is 38.1x faster than the slowest successful cell (`julia-base`, 327.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `pytorch-cpu` is 38.6x faster than the slowest successful cell (`julia-base`, 327.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-direct` is 42.9x faster than the slowest successful cell (`julia-base`, 327.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-trace` is 42.4x faster than the slowest successful cell (`julia-base`, 327.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `jax-cpu` is 20.3x faster than the slowest successful cell (`julia-base`, 658.409 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 29.1x faster than the slowest successful cell (`julia-base`, 658.409 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-direct` is 31.1x faster than the slowest successful cell (`julia-base`, 658.409 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-trace` is 30.8x faster than the slowest successful cell (`julia-base`, 658.409 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 37.3x faster than the slowest successful cell (`julia-base`, 389.343 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 34.3x faster than the slowest successful cell (`julia-base`, 389.343 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-direct` is 33.2x faster than the slowest successful cell (`julia-base`, 389.343 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-trace` is 26.4x faster than the slowest successful cell (`julia-base`, 389.343 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 24.5x faster than the slowest successful cell (`julia-base`, 136.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-eager` is 10.6x faster than the slowest successful cell (`julia-base`, 136.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-trace` is 12.0x faster than the slowest successful cell (`julia-base`, 136.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 14.7x faster than the slowest successful cell (`julia-base`, 248.923 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 18.8x faster than the slowest successful cell (`julia-base`, 248.923 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 16.4x faster than the slowest successful cell (`julia-base`, 248.923 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 26.1x faster than the slowest successful cell (`julia-base`, 248.923 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `jax-cpu` is 71.4x faster than the slowest successful cell (`julia-base`, 996.351 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 83.4x faster than the slowest successful cell (`julia-base`, 996.351 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 66.2x faster than the slowest successful cell (`julia-base`, 996.351 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 74.6x faster than the slowest successful cell (`julia-base`, 996.351 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `jax-cpu` is 49.0x faster than the slowest successful cell (`julia-base`, 960.302 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 81.0x faster than the slowest successful cell (`julia-base`, 960.302 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 59.5x faster than the slowest successful cell (`julia-base`, 960.302 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 44.9x faster than the slowest successful cell (`julia-base`, 960.302 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 32.2x faster than the slowest successful cell (`julia-base`, 280.611 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 34.9x faster than the slowest successful cell (`julia-base`, 280.611 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 28.9x faster than the slowest successful cell (`julia-base`, 280.611 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 29.3x faster than the slowest successful cell (`julia-base`, 280.611 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 29.6x faster than the slowest successful cell (`julia-base`, 1695.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 35.3x faster than the slowest successful cell (`julia-base`, 1695.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-eager` is 36.9x faster than the slowest successful cell (`julia-base`, 1695.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-trace` is 37.3x faster than the slowest successful cell (`julia-base`, 1695.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 11.2x faster than the slowest successful cell (`tenferro-eager`, 15.227 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 12.1x faster than the slowest successful cell (`tenferro-eager`, 13.323 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 4192.0x faster than the slowest successful cell (`julia-base`, 14.500 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 10681.2x faster than the slowest successful cell (`julia-base`, 14.238 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 11536.3x faster than the slowest successful cell (`pytorch-cpu`, 15.862 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 12424.4x faster than the slowest successful cell (`julia-base`, 16.574 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 129437.1x faster than the slowest successful cell (`julia-base`, 226.515 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `strided-jl` is 14.4x faster than the slowest successful cell (`julia-base`, 219.693 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`julia-base`, 219.693 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 159776.6x faster than the slowest successful cell (`julia-base`, 219.693 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 449.4x faster than the slowest successful cell (`tenferro-direct`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 201.1x faster than the slowest successful cell (`tenferro-direct`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 177.6x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 84.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 51.1x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 138.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 11834.2 GB/s at 0.001 ms. The conservative physical lower bound is 0.168 ms, so the cell is 118.3x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 4423.4 GB/s at 0.004 ms. The conservative physical lower bound is 0.042 ms, so the cell is 11.1x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 77604.9 GB/s at 0.003 ms. The conservative physical lower bound is 2.684 ms, so the cell is 776.0x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 201376.9 GB/s at 0.001 ms. The conservative physical lower bound is 0.671 ms, so the cell is 503.4x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 195225.8 GB/s at 0.001 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1952.3x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 201226.0 GB/s at 0.001 ms. The conservative physical lower bound is 0.671 ms, so the cell is 503.1x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 76695.8 GB/s at 0.002 ms. The conservative physical lower bound is 1.342 ms, so the cell is 767.0x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 97612.9 GB/s at 0.001 ms. The conservative physical lower bound is 0.336 ms, so the cell is 244.0x faster than the bound.

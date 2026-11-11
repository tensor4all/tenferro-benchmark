# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20261111_181810/run.yaml`
- Timestamp: `2026-11-11T18:18:10.471772+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20261111_181810`.

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

- CSV: `data/results/mac-cpu/cpu/public_api/20261111_181810/cpu_public_api_t1_20261111_181810.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20261111_181810/cpu_public_api_t4_20261111_181810.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 4.015 ± 0.278 | - | 3.993 ± 0.368 | 4.678 ± 0.206 | 1.869 ± 0.189 | 9.898 ± 5.186 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 3.949 ± 1.161 | - | 3.839 ± 0.317 | 3.718 ± 0.084 | 2.197 ± 0.722 | 77.662 ± 11.957 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 18.682 ± 5.251 | - | 15.402 ± 4.538 | 18.628 ± 9.325 | 13.507 ± 2.202 | 11.999 ± 0.802 | 16.675 ± 13.722 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 12.706 ± 0.560 | - | 12.502 ± 1.210 | 9.760 ± 1.573 | 17.185 ± 4.661 | 13.126 ± 3.544 | 14.732 ± 2.652 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 60.915 ± 14.978 | - | 56.405 ± 12.414 | 22.237 ± 2.787 | 36.009 ± 0.683 | 50.914 ± 5.418 | 51.441 ± 6.706 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 21.192 ± 3.554 | - | 22.766 ± 5.602 | 11.534 ± 2.526 | 21.235 ± 2.290 | 64.206 ± 11.124 | 39.918 ± 9.425 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 11.726 ± 1.334 | - | 11.459 ± 1.147 | 14.648 ± 1.804 | 49.562 ± 11.883 | 73.613 ± 14.884 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 11.137 ± 0.662 | - | 13.459 ± 3.109 | 9.948 ± 0.926 | 28.612 ± 7.952 | 37.038 ± 11.396 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 12.712 ± 0.933 | - | unsupported | 14.172 ± 0.898 | 49.468 ± 4.958 | 72.922 ± 3.235 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 13.044 ± 0.418 | - | unsupported | 10.396 ± 1.337 | 19.305 ± 1.012 | 31.705 ± 6.484 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 11.984 ± 1.313 | - | 15.959 ± 5.831 | 13.636 ± 0.229 | 20.582 ± 1.644 | 10.401 ± 0.896 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 12.000 ± 0.957 | - | 10.130 ± 1.588 | 11.883 ± 0.966 | 26.845 ± 5.673 | 461.518 ± 483.913 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 41.700 ± 4.501 | - | 47.680 ± 10.269 | 45.291 ± 3.091 | 47.473 ± 3.302 | 52.666 ± 6.338 | 52.724 ± 1.179 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 20.334 ± 3.172 | - | 24.225 ± 5.614 | 18.318 ± 5.304 | 21.702 ± 1.122 | 54.008 ± 1.105 | 24.982 ± 3.079 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 46.105 ± 0.385 | - | 45.926 ± 0.445 | 63.659 ± 4.428 | 87.383 ± 2.034 | 83.169 ± 3.799 | 89.261 ± 14.781 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 26.410 ± 6.708 | - | 26.828 ± 8.766 | 15.623 ± 2.132 | 47.522 ± 10.294 | 71.890 ± 6.023 | 38.425 ± 8.087 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 11.440 ± 5.220 | - | 10.961 ± 4.394 | 11.574 ± 1.775 | 10.579 ± 2.736 | 10.558 ± 1.769 | 11.341 ± 1.652 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 10.365 ± 0.733 | - | 10.291 ± 0.525 | 10.560 ± 2.714 | 12.644 ± 0.855 | 13.617 ± 2.058 | 15.573 ± 2.155 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.341 ± 0.187 | - | 8.107 ± 0.619 | 7.505 ± 0.406 | 6.172 ± 1.188 | 9.432 ± 1.563 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 1.324 ± 0.100 | - | 6.617 ± 1.345 | 3.597 ± 0.644 | 4.059 ± 1.094 | 12.819 ± 3.099 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 9.222 ± 0.230 | - | 7.485 ± 0.848 | 8.407 ± 2.126 | 7.027 ± 0.121 | 11.522 ± 5.023 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 8.976 ± 0.721 | - | 6.932 ± 0.617 | 7.636 ± 0.567 | 7.148 ± 0.153 | 169.371 ± 18.059 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 5.368 ± 0.856 | - | 5.715 ± 0.490 | 6.259 ± 0.501 | 3.051 ± 0.683 | 7.650 ± 0.518 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 5.395 ± 1.183 | - | 5.471 ± 0.118 | 5.266 ± 0.758 | 2.948 ± 0.192 | 152.849 ± 9.879 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 7.045 ± 0.709 | - | 8.588 ± 1.344 | 8.082 ± 1.466 | 6.960 ± 0.420 | 9.100 ± 2.224 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 7.158 ± 0.816 | - | 8.434 ± 0.707 | 7.459 ± 0.618 | 7.042 ± 0.154 | 625.633 ± 56.600 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 12.645 ± 2.031 | - | 13.362 ± 5.470 | 13.607 ± 0.935 | 47.608 ± 3.712 | 69.321 ± 3.450 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 11.755 ± 2.074 | - | 13.454 ± 3.945 | 11.054 ± 0.999 | 18.082 ± 2.005 | 29.313 ± 4.462 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 13.503 ± 1.606 | - | unsupported | 11.113 ± 0.901 | 61.224 ± 3.510 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 14.563 ± 1.760 | - | unsupported | 13.809 ± 1.365 | 28.276 ± 3.019 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 13.396 ± 2.449 | - | 15.171 ± 5.012 | 14.294 ± 1.449 | 17.771 ± 2.489 | 11.247 ± 1.127 | 12.563 ± 0.259 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 14.070 ± 2.632 | - | 13.124 ± 1.223 | 14.128 ± 5.891 | 14.181 ± 1.544 | 13.632 ± 2.837 | 13.480 ± 0.761 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 22.879 ± 4.379 | - | 23.087 ± 6.915 | 26.690 ± 4.682 | 32.878 ± 5.447 | 20.541 ± 19.723 | 18.749 ± 2.319 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 21.367 ± 2.008 | - | 22.758 ± 5.552 | 18.446 ± 1.923 | 25.107 ± 1.375 | 23.943 ± 40.690 | 24.247 ± 10.273 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 36.391 ± 5.217 | - | 47.642 ± 3.625 | 49.768 ± 8.074 | 34.549 ± 3.478 | 48.991 ± 1.732 | 56.622 ± 5.495 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 23.692 ± 16.389 | - | 21.269 ± 5.334 | 18.823 ± 1.060 | 16.807 ± 0.518 | 50.692 ± 3.977 | 25.162 ± 3.183 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 8.450 ± 3.481 | - | 9.052 ± 2.777 | 5.810 ± 0.920 | 9.643 ± 2.977 | 13.688 ± 8.078 | 11.711 ± 3.452 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 7.290 ± 0.720 | - | 7.428 ± 1.567 | 5.654 ± 0.135 | 7.375 ± 0.548 | 7.353 ± 1.078 | 6.682 ± 1.276 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 16.270 ± 0.848 | - | 14.062 ± 2.530 | 21.653 ± 1.943 | 15.601 ± 0.688 | 13.809 ± 1.696 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 14.347 ± 2.195 | - | 13.608 ± 0.825 | 16.828 ± 1.325 | 14.757 ± 0.660 | 16.938 ± 3.906 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 47.349 ± 14.039 | - | 44.553 ± 12.149 | 40.378 ± 5.552 | 30.250 ± 2.500 | 45.317 ± 5.479 | 49.195 ± 22.875 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 19.640 ± 5.398 | - | 17.814 ± 5.731 | 12.438 ± 0.461 | 14.175 ± 0.436 | 41.557 ± 4.366 | 21.428 ± 3.235 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 22.710 ± 11.469 | - | 20.988 ± 1.710 | 19.849 ± 1.820 | 30.855 ± 3.648 | 17.459 ± 2.116 | 21.246 ± 3.461 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 20.428 ± 2.085 | - | 20.131 ± 1.567 | 22.298 ± 7.538 | 29.122 ± 3.522 | 24.352 ± 4.570 | 24.886 ± 3.085 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 26.028 ± 2.614 | - | 30.168 ± 7.867 | 30.293 ± 1.762 | 18.810 ± 4.880 | 47.998 ± 17.240 | 58.173 ± 14.041 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 12.043 ± 6.112 | - | 12.190 ± 3.991 | 10.714 ± 0.575 | 8.488 ± 0.411 | 40.204 ± 11.063 | 20.718 ± 3.800 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 33.767 ± 2.281 | - | 29.274 ± 1.382 | 29.539 ± 3.393 | 19.768 ± 4.175 | 25.529 ± 0.582 | 25.657 ± 0.694 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 15.838 ± 4.323 | - | 16.607 ± 2.581 | 13.813 ± 2.767 | 9.566 ± 0.177 | 32.747 ± 15.548 | 20.751 ± 4.781 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 49.003 ± 5.325 | - | 47.381 ± 9.499 | 41.735 ± 1.705 | 30.430 ± 6.049 | 159.030 ± 80.847 | 129.937 ± 91.918 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 22.573 ± 4.984 | - | 21.469 ± 5.830 | 14.397 ± 0.655 | 13.047 ± 0.211 | 47.112 ± 11.698 | 21.828 ± 13.957 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 17.610 ± 0.982 | - | 16.653 ± 5.838 | 24.893 ± 2.099 | 21.398 ± 1.257 | 21.290 ± 0.961 | 22.649 ± 2.019 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 9.906 ± 2.354 | - | 9.911 ± 2.882 | 9.490 ± 0.112 | 10.272 ± 0.602 | 32.110 ± 9.025 | 17.481 ± 9.596 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.328 ± 0.057 | 0.327 ± 0.074 | unsupported | unsupported | 0.299 ± 0.181 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.332 ± 0.070 | 0.379 ± 0.101 | unsupported | unsupported | 0.376 ± 0.055 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 21.724 ± 2.034 | - | 22.461 ± 1.570 | 22.298 ± 2.532 | 25.623 ± 0.856 | 19.439 ± 3.835 | 17.993 ± 1.566 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 21.509 ± 1.788 | - | 20.812 ± 1.337 | 22.392 ± 3.053 | 30.195 ± 2.283 | 50.952 ± 105.171 | 21.788 ± 3.550 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 23.051 ± 4.141 | - | 21.817 ± 3.890 | 23.933 ± 7.217 | 25.119 ± 3.265 | 18.168 ± 5.782 | 20.886 ± 11.052 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 19.496 ± 0.818 | - | 20.538 ± 1.198 | 22.461 ± 5.293 | 28.460 ± 4.809 | 17.995 ± 2.294 | 19.683 ± 1.560 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 25.339 ± 2.260 | - | 33.037 ± 10.101 | 20.133 ± 6.231 | 32.462 ± 6.416 | 15.090 ± 1.497 | 17.574 ± 1.172 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 20.356 ± 2.095 | - | 21.318 ± 0.511 | 20.396 ± 2.522 | 22.481 ± 0.758 | 32.320 ± 20.468 | 22.037 ± 7.512 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 12.875 ± 1.859 | - | 15.356 ± 7.113 | 13.736 ± 0.698 | 14.640 ± 1.662 | 12.274 ± 1.398 | 13.873 ± 3.381 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 12.983 ± 1.165 | - | 13.268 ± 1.293 | 13.617 ± 0.941 | 15.932 ± 1.699 | 21.283 ± 9.741 | 13.637 ± 0.982 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 41.508 ± 11.846 | - | 44.657 ± 19.139 | 72.014 ± 9.449 | 48.474 ± 2.639 | 87.776 ± 8.187 | 77.588 ± 2.434 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 24.334 ± 3.465 | - | 21.749 ± 4.318 | 30.039 ± 2.912 | 21.370 ± 0.720 | 92.423 ± 16.832 | 38.387 ± 21.004 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 9.935 ± 3.276 | - | 7.800 ± 2.477 | 12.347 ± 0.723 | 26.034 ± 14.707 | 7.792 ± 0.345 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 7.826 ± 1.837 | - | 6.673 ± 0.839 | 6.932 ± 0.352 | 15.264 ± 4.469 | 6.781 ± 1.102 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.136 ± 0.621 | - | 1.257 ± 0.318 | 19.847 ± 2.254 | 2.766 ± 3.113 | 1.358 ± 0.346 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 1.449 ± 4.479 | - | 1.082 ± 0.228 | 12.840 ± 1.882 | 1.576 ± 0.780 | 0.845 ± 0.136 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 0.958 ± 0.064 | - | 0.924 ± 0.156 | 1.668 ± 0.101 | 1.384 ± 0.127 | 1.091 ± 0.209 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 1.773 ± 0.389 | - | 1.296 ± 0.059 | 0.959 ± 0.024 | 1.240 ± 0.198 | 0.947 ± 0.067 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 8.979 ± 1.330 | - | 9.672 ± 0.720 | 13.299 ± 1.320 | 24.969 ± 1.626 | 7.960 ± 1.240 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 7.191 ± 1.298 | - | 7.048 ± 0.318 | 7.185 ± 0.711 | 14.768 ± 2.539 | 7.092 ± 0.619 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.050 ± 0.506 | - | 1.152 ± 0.440 | 18.834 ± 2.600 | 2.692 ± 0.326 | 0.942 ± 0.254 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.798 ± 0.077 | - | 0.725 ± 0.042 | 10.507 ± 0.747 | 1.963 ± 0.310 | 0.846 ± 0.043 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 4.671 ± 0.891 | - | 4.083 ± 1.038 | 6.586 ± 0.469 | 4.047 ± 0.451 | 4.320 ± 0.742 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 7.166 ± 0.374 | - | 6.295 ± 1.211 | 3.659 ± 0.226 | 4.565 ± 0.343 | 3.784 ± 0.322 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 8.503 ± 0.935 | - | 8.800 ± 2.170 | 10.661 ± 4.059 | 48.275 ± 1.144 | 7.874 ± 1.127 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 7.288 ± 0.598 | - | 6.933 ± 0.675 | 6.172 ± 0.286 | 27.716 ± 5.687 | 6.729 ± 0.416 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.971 ± 0.218 | - | 1.048 ± 0.120 | 5.152 ± 0.422 | 2.834 ± 0.406 | 1.160 ± 0.272 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.894 ± 0.172 | - | 0.980 ± 0.160 | 2.326 ± 0.075 | 2.037 ± 0.667 | 0.718 ± 0.118 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.057 ± 0.165 | - | 0.969 ± 0.527 | 1.013 ± 0.045 | 1.496 ± 1.010 | 1.028 ± 0.194 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 2.516 ± 0.690 | - | 1.454 ± 0.121 | 0.740 ± 0.046 | 1.328 ± 0.470 | 0.896 ± 0.163 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 11.232 ± 5.368 | - | 8.290 ± 1.934 | 8.974 ± 2.215 | 36.348 ± 2.132 | 7.844 ± 0.732 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 8.404 ± 2.788 | - | 6.728 ± 0.975 | 6.035 ± 0.642 | 17.817 ± 1.309 | 6.324 ± 0.286 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.942 ± 0.295 | - | 0.978 ± 0.184 | 5.387 ± 0.619 | 2.736 ± 0.356 | 0.974 ± 0.224 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.730 ± 0.013 | - | 0.757 ± 0.100 | 2.232 ± 0.364 | 2.391 ± 0.212 | 0.826 ± 0.213 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 0.920 ± 0.094 | - | 0.903 ± 0.170 | 0.887 ± 0.125 | 1.274 ± 0.341 | 0.982 ± 0.171 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 2.113 ± 0.116 | - | 2.375 ± 0.706 | 0.683 ± 0.015 | 1.143 ± 0.194 | 0.861 ± 0.084 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.941 ± 0.116 | 1.030 ± 0.061 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.852 ± 0.129 | 0.881 ± 0.053 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 41.675 ± 6.723 | - | 38.754 ± 8.566 | 43.688 ± 6.319 | 42.966 ± 5.441 | 43.971 ± 44.206 | 66.054 ± 28.368 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 16.366 ± 3.908 | - | 15.653 ± 2.183 | 22.470 ± 6.748 | 20.899 ± 2.139 | 36.548 ± 14.422 | 15.528 ± 2.258 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 24.095 ± 3.702 | - | 28.293 ± 2.150 | 18.883 ± 0.766 | 37.388 ± 1.705 | 41.098 ± 0.150 | 39.953 ± 2.138 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 13.712 ± 2.138 | - | 14.299 ± 4.492 | 13.694 ± 0.803 | 18.398 ± 0.686 | 53.808 ± 3.733 | 30.341 ± 8.226 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 30.677 ± 5.773 | - | 31.548 ± 8.373 | 29.279 ± 6.191 | 26.528 ± 3.076 | 23.744 ± 1.136 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 32.490 ± 4.924 | - | 33.727 ± 2.773 | 22.251 ± 2.506 | 23.865 ± 1.193 | 25.627 ± 8.287 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 13.622 ± 4.772 | - | 13.603 ± 2.299 | 16.227 ± 3.318 | 17.433 ± 2.130 | 13.543 ± 65.982 | 19.406 ± 1.070 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 13.886 ± 0.801 | - | 12.654 ± 1.926 | 14.023 ± 2.498 | 13.519 ± 1.665 | 12.809 ± 0.701 | 13.333 ± 0.976 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 39.413 ± 10.459 | - | 43.741 ± 11.005 | 30.792 ± 1.463 | 30.103 ± 11.319 | 40.661 ± 5.368 | 42.787 ± 6.740 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 16.192 ± 3.163 | - | 15.744 ± 2.956 | 11.277 ± 0.194 | 13.268 ± 0.459 | 36.916 ± 3.780 | 14.035 ± 1.738 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 15.517 ± 3.422 | - | 17.897 ± 4.308 | 13.453 ± 1.733 | 21.292 ± 5.736 | 26.443 ± 2.575 | 26.348 ± 0.784 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 14.373 ± 2.866 | - | 13.172 ± 2.044 | 14.704 ± 2.287 | 11.595 ± 0.354 | 34.776 ± 7.479 | 23.148 ± 28.749 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 23.772 ± 6.698 | - | 34.926 ± 15.235 | 23.951 ± 3.355 | 33.329 ± 4.828 | 14.870 ± 0.736 | 15.310 ± 0.657 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 19.564 ± 1.873 | - | 20.116 ± 1.039 | 19.605 ± 1.536 | 26.582 ± 1.506 | 26.547 ± 6.197 | 22.717 ± 2.247 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 105.758 ± 17.686 | - | 85.031 ± 5.385 | 84.058 ± 8.401 | 14.653 ± 0.841 | 53.396 ± 35.257 | 66.172 ± 6.433 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 42.137 ± 12.918 | - | 40.681 ± 4.806 | 30.620 ± 4.731 | 8.160 ± 1.393 | 45.629 ± 1.214 | 25.361 ± 6.122 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 31.285 ± 3.970 | 34.833 ± 5.345 | unsupported | 19.104 ± 1.778 | 25.246 ± 2.832 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 31.062 ± 2.493 | 31.666 ± 3.277 | unsupported | 17.489 ± 0.724 | 24.917 ± 1.105 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.734 ± 0.305 | - | 0.780 ± 0.094 | 0.720 ± 0.267 | 0.829 ± 0.134 | 1.059 ± 0.696 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.887 ± 0.102 | - | 0.979 ± 0.102 | 0.684 ± 0.017 | 1.373 ± 0.546 | 1.040 ± 0.944 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.872 ± 0.533 | - | 0.847 ± 0.130 | 0.818 ± 0.030 | 0.963 ± 0.439 | 0.814 ± 0.213 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.835 ± 0.350 | - | 0.915 ± 0.114 | 0.700 ± 0.079 | 1.105 ± 0.431 | 0.994 ± 0.060 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.194 ± 0.109 | - | unsupported | 1.535 ± 0.285 | 1.549 ± 0.089 | 1.250 ± 0.651 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 1.231 ± 0.126 | - | unsupported | 1.019 ± 0.303 | 2.360 ± 1.130 | 2.386 ± 0.922 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.387 ± 0.198 | - | 0.675 ± 0.167 | 0.345 ± 0.079 | 0.396 ± 0.059 | 1.843 ± 1.933 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.359 ± 0.189 | - | 0.650 ± 0.133 | 0.313 ± 0.031 | 1.017 ± 0.134 | 1.411 ± 3.021 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.782 ± 0.045 | - | 0.720 ± 0.200 | 1.286 ± 0.184 | 0.988 ± 0.336 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.850 ± 0.147 | - | 0.758 ± 0.303 | 1.007 ± 0.039 | 1.638 ± 0.538 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.769 ± 0.073 | - | 0.840 ± 0.221 | 1.048 ± 0.339 | 0.976 ± 0.077 | 1.221 ± 0.302 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 1.101 ± 0.221 | - | 0.987 ± 0.111 | 0.696 ± 0.139 | 0.860 ± 0.049 | 2.009 ± 2.500 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 1.215 ± 0.797 | - | 0.814 ± 0.169 | 0.518 ± 0.117 | 0.542 ± 0.090 | 2.058 ± 4.483 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.912 ± 0.248 | - | 1.077 ± 0.381 | 0.442 ± 0.131 | 1.092 ± 0.103 | 1.133 ± 1.596 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.420 ± 0.263 | - | 0.004 ± 0.000 | 1.386 ± 0.184 | 2.588 ± 1.084 | 1.894 ± 0.734 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 1.499 ± 0.429 | - | 0.001 ± 0.000 | 1.266 ± 0.085 | 2.041 ± 0.300 | 1.739 ± 0.427 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.060 ± 0.000 | - | 0.083 ± 0.006 | 0.133 ± 0.009 | 0.092 ± 0.125 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.098 ± 0.003 | - | 0.121 ± 0.004 | 0.162 ± 0.008 | 0.132 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.211 ± 0.011 | - | 0.220 ± 0.002 | 0.250 ± 0.009 | 0.224 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.026 ± 0.001 | - | 0.027 ± 0.002 | 0.058 ± 0.004 | 0.107 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.177 ± 0.088 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.011 ± 0.001 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.012 ± 0.001 | 0.023 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.006 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.011 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.001 | 0.008 ± 0.006 | 0.064 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.179 ± 0.010 | - | 0.184 ± 0.003 | 0.271 ± 0.060 | 0.315 ± 0.158 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.011 ± 0.003 | - | 0.013 ± 0.000 | 0.018 ± 0.001 | 0.054 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.014 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.027 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.055 ± 0.000 | - | 0.077 ± 0.001 | 0.142 ± 0.039 | 0.137 ± 0.071 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.093 ± 0.002 | - | 0.116 ± 0.000 | 0.208 ± 0.025 | 0.169 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.201 ± 0.002 | - | 0.221 ± 0.012 | 0.305 ± 0.134 | 0.259 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.026 ± 0.001 | - | 0.028 ± 0.001 | 0.117 ± 0.069 | 0.054 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.054 ± 0.026 | 0.011 ± 0.053 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.098 ± 0.019 | 0.112 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.047 ± 0.019 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.063 ± 0.041 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.053 ± 0.020 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.052 ± 0.046 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.051 ± 0.019 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.062 ± 0.027 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.059 ± 0.016 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.003 ± 0.000 | - | 0.010 ± 0.000 | 0.049 ± 0.012 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.176 ± 0.017 | - | 0.184 ± 0.011 | 0.363 ± 0.077 | 0.200 ± 0.060 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.052 ± 0.015 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.012 ± 0.000 | - | 0.014 ± 0.001 | 0.128 ± 0.082 | 0.103 ± 0.067 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.087 ± 0.052 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.003 ± 0.005 | - | 0.005 ± 0.000 | 0.063 ± 0.017 | 0.112 ± 0.075 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.283 ± 0.016 | - | 0.268 ± 0.016 | 0.629 ± 0.217 | 0.976 ± 0.459 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.544 ± 0.036 | - | 1.626 ± 0.116 | 1.891 ± 0.308 | 2.163 ± 0.107 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 6.204 ± 0.515 | - | 6.598 ± 0.620 | 7.241 ± 0.846 | 7.649 ± 1.031 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 4.685 ± 0.117 | - | 6.264 ± 0.996 | 3.700 ± 0.691 | 4.773 ± 1.348 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.056 ± 0.037 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.090 ± 0.001 | - | 0.092 ± 0.001 | 0.098 ± 0.025 | 0.152 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.002 | 0.019 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.011 ± 0.016 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.002 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.019 ± 0.005 | - | 0.026 ± 0.009 | 0.023 ± 0.005 | 0.071 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.024 ± 0.003 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.030 ± 0.007 | - | 0.032 ± 0.003 | 0.029 ± 0.002 | 0.073 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 37.337 ± 1.166 | - | 59.864 ± 6.192 | 39.278 ± 5.994 | 34.277 ± 0.956 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.011 ± 0.023 | 0.016 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.802 ± 0.058 | - | 0.720 ± 0.013 | 0.775 ± 0.040 | 0.912 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.013 ± 0.000 | - | 0.014 ± 0.000 | 0.018 ± 0.001 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.061 ± 0.068 | - | 0.061 ± 0.006 | 0.064 ± 0.012 | 0.131 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.254 ± 0.013 | - | 0.276 ± 0.088 | 0.595 ± 0.167 | 1.017 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 1.576 ± 0.141 | - | 1.755 ± 0.262 | 2.368 ± 1.653 | 1.248 ± 0.212 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 7.423 ± 1.873 | - | 6.355 ± 0.676 | 10.301 ± 1.459 | 4.239 ± 1.135 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 4.644 ± 0.573 | - | 7.041 ± 0.480 | 4.101 ± 0.604 | 4.152 ± 0.748 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.007 | 0.034 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.090 ± 0.013 | - | 0.093 ± 0.002 | 0.099 ± 0.069 | 0.120 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.019 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.009 ± 0.002 | - | 0.011 ± 0.001 | 0.014 ± 0.003 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.029 ± 0.028 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.021 ± 0.004 | - | 0.023 ± 0.005 | 0.025 ± 0.007 | 0.092 ± 0.065 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.005 ± 0.001 | 0.013 ± 0.007 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.013 ± 0.001 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.025 ± 0.003 | - | 0.027 ± 0.005 | 0.038 ± 0.008 | 0.078 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 37.578 ± 2.533 | - | 51.430 ± 2.844 | 28.293 ± 2.607 | 18.079 ± 0.961 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.005 ± 0.000 | - | 0.005 ± 0.000 | 0.021 ± 0.011 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.780 ± 0.103 | - | 0.763 ± 0.135 | 1.259 ± 0.713 | 0.460 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.013 ± 0.004 | - | 0.015 ± 0.004 | 0.024 ± 0.005 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.050 ± 0.009 | - | 0.050 ± 0.006 | 0.068 ± 0.010 | 0.210 ± 0.060 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.244 ± 0.001 | - | 0.264 ± 0.027 | 0.327 ± 0.067 | 0.870 ± 0.157 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.289 ± 0.052 | - | 1.290 ± 0.014 | 1.360 ± 0.330 | 1.977 ± 0.112 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 4.934 ± 0.165 | - | 5.053 ± 0.198 | 6.831 ± 0.928 | 6.452 ± 0.604 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 0.982 ± 0.006 | - | 1.026 ± 0.479 | 1.136 ± 0.225 | 1.335 ± 0.120 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.070 ± 0.001 | - | 0.176 ± 0.107 | 0.089 ± 0.008 | 0.191 ± 0.094 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.036 ± 0.013 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.003 | 0.018 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.002 | 0.009 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.015 ± 0.000 | - | 0.018 ± 0.004 | 0.019 ± 0.000 | 0.028 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.020 ± 0.006 | - | 0.022 ± 0.000 | 0.023 ± 0.006 | 0.034 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 8.805 ± 3.254 | - | 8.215 ± 0.761 | 8.947 ± 0.297 | 9.557 ± 0.575 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.552 ± 0.011 | - | 0.551 ± 0.109 | 0.533 ± 0.140 | 0.641 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.011 ± 0.000 | - | 0.012 ± 0.000 | 0.017 ± 0.004 | 0.025 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.039 ± 0.006 | - | 0.040 ± 0.005 | 0.048 ± 0.009 | 0.085 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.280 ± 0.107 | - | 0.244 ± 0.013 | 0.317 ± 0.165 | 0.563 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.432 ± 0.118 | - | 1.431 ± 0.064 | 2.709 ± 1.469 | 1.348 ± 0.833 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 5.628 ± 0.287 | - | 5.626 ± 0.200 | 6.461 ± 1.498 | 3.044 ± 0.091 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.045 ± 0.128 | - | 1.077 ± 0.023 | 1.315 ± 0.739 | 1.088 ± 0.080 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.005 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.071 ± 0.002 | - | 0.073 ± 0.004 | 0.074 ± 0.048 | 0.139 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.020 ± 0.006 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.001 | 0.066 ± 0.076 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.001 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.005 ± 0.002 | - | 0.007 ± 0.000 | 0.019 ± 0.006 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.016 ± 0.001 | - | 0.017 ± 0.001 | 0.077 ± 0.017 | 0.029 ± 0.131 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.009 ± 0.004 | - | 0.009 ± 0.000 | 0.024 ± 0.015 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.023 ± 0.005 | - | 0.024 ± 0.005 | 0.031 ± 0.013 | 0.030 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 8.635 ± 0.734 | - | 9.550 ± 0.973 | 9.458 ± 0.996 | 4.164 ± 0.275 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.004 ± 0.001 | - | 0.006 ± 0.001 | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.587 ± 0.032 | - | 0.680 ± 0.453 | 0.556 ± 0.074 | 0.347 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.011 ± 0.000 | - | 0.016 ± 0.001 | 0.038 ± 0.013 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.044 ± 0.007 | - | 0.041 ± 0.002 | 0.058 ± 0.060 | 0.121 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.166 ± 0.045 | - | 0.181 ± 0.070 | 0.139 ± 0.005 | 0.821 ± 0.105 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 0.794 ± 0.089 | - | 0.826 ± 0.099 | 1.046 ± 0.306 | 1.895 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 3.275 ± 0.953 | - | 3.114 ± 0.797 | 3.124 ± 0.208 | 6.079 ± 0.302 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 1.012 ± 0.107 | - | 1.022 ± 0.064 | 0.532 ± 0.051 | 1.141 ± 0.060 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.016 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.027 ± 0.006 | - | 0.029 ± 0.001 | 0.043 ± 0.044 | 0.143 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.014 ± 0.007 | 0.015 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.001 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.002 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.010 ± 0.002 | - | 0.009 ± 0.000 | 0.022 ± 0.008 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.012 ± 0.002 | - | 0.011 ± 0.000 | 0.041 ± 0.007 | 0.030 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 6.707 ± 2.301 | - | 4.819 ± 0.996 | 5.147 ± 2.239 | 9.511 ± 0.746 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.001 | - | 0.006 ± 0.001 | 0.006 ± 0.001 | 0.013 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.236 ± 0.125 | - | 0.383 ± 0.022 | 0.271 ± 0.047 | 0.618 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.002 | 0.059 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.020 ± 0.050 | - | 0.020 ± 0.004 | 0.053 ± 0.007 | 0.108 ± 0.053 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.117 ± 0.001 | - | 0.134 ± 0.022 | 0.126 ± 0.003 | 0.622 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 0.800 ± 0.075 | - | 0.748 ± 0.032 | 0.957 ± 0.167 | 1.172 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 2.544 ± 0.167 | - | 2.716 ± 0.627 | 2.583 ± 0.125 | 3.214 ± 0.112 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.529 ± 0.025 | - | 0.524 ± 0.077 | 0.487 ± 0.047 | 1.312 ± 0.126 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.026 ± 0.006 | - | 0.032 ± 0.007 | 0.030 ± 0.001 | 0.163 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.002 | 0.007 ± 0.001 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.002 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.011 ± 0.002 | - | 0.010 ± 0.001 | 0.012 ± 0.000 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.002 | 0.011 ± 0.060 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.002 | 0.008 ± 0.002 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.011 ± 0.003 | - | 0.012 ± 0.003 | 0.014 ± 0.000 | 0.035 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 3.967 ± 0.353 | - | 3.791 ± 0.284 | 4.214 ± 0.505 | 5.203 ± 0.456 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.021 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.222 ± 0.031 | - | 0.227 ± 0.041 | 0.202 ± 0.004 | 0.476 ± 0.123 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.002 | 0.010 ± 0.000 | 0.026 ± 0.107 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.020 ± 0.003 | - | 0.021 ± 0.005 | 0.027 ± 0.006 | 0.162 ± 0.063 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.093 ± 0.016 | - | 0.095 ± 0.010 | 0.188 ± 0.017 | 0.174 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.189 ± 0.020 | - | 0.204 ± 0.013 | 0.267 ± 0.001 | 0.245 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.437 ± 0.023 | - | 0.460 ± 0.021 | 0.507 ± 0.018 | 0.520 ± 0.527 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.156 ± 0.015 | - | 0.156 ± 0.003 | 0.072 ± 0.016 | 0.153 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.003 ± 0.000 | 0.008 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.015 ± 0.001 | - | 0.007 ± 0.000 | 0.011 ± 0.001 | 0.071 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.010 ± 0.003 | - | 0.009 ± 0.000 | 0.013 ± 0.000 | 0.060 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.010 ± 0.036 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.007 ± 0.000 | - | 0.012 ± 0.000 | 0.009 ± 0.000 | 0.066 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.003 ± 0.000 | 0.010 ± 0.002 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.009 ± 0.000 | - | 0.013 ± 0.000 | 0.009 ± 0.000 | 0.055 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.493 ± 0.015 | - | 0.502 ± 0.015 | 0.487 ± 0.031 | 0.648 ± 0.106 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.031 ± 0.134 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.032 ± 0.000 | - | 0.089 ± 0.005 | 0.039 ± 0.000 | 0.107 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.060 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.013 ± 0.000 | - | 0.017 ± 0.000 | 0.011 ± 0.000 | 0.065 ± 0.059 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.091 ± 0.011 | - | 0.097 ± 0.009 | 0.268 ± 0.060 | 0.151 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.194 ± 0.005 | - | 0.212 ± 0.042 | 0.308 ± 0.061 | 0.396 ± 0.098 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.474 ± 0.056 | - | 0.464 ± 0.085 | 0.608 ± 0.048 | 0.801 ± 0.464 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.063 ± 0.003 | - | 0.064 ± 0.002 | 0.154 ± 0.037 | 0.160 ± 0.074 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.069 ± 0.034 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.060 ± 0.033 | 0.094 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.062 ± 0.030 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.011 ± 0.002 | - | 0.013 ± 0.002 | 0.076 ± 0.018 | 0.051 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.001 | 0.060 ± 0.014 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.071 ± 0.034 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.063 ± 0.034 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.066 ± 0.006 | 0.039 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.069 ± 0.015 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.065 ± 0.013 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.002 | 0.086 ± 0.044 | 0.013 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.480 ± 0.015 | - | 0.644 ± 0.152 | 0.646 ± 0.397 | 0.591 ± 0.108 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.004 ± 0.000 | 0.091 ± 0.016 | 0.014 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.033 ± 0.005 | - | 0.039 ± 0.008 | 0.132 ± 0.053 | 0.161 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.094 ± 0.039 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.084 ± 0.044 | 0.063 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.304 ± 0.280 | - | 0.210 ± 0.039 | 0.333 ± 0.014 | 0.539 ± 0.102 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.610 ± 0.023 | - | 0.608 ± 0.024 | 0.700 ± 0.045 | 1.663 ± 0.690 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.629 ± 0.479 | - | 1.645 ± 0.204 | 1.360 ± 0.169 | 1.512 ± 0.224 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.329 ± 0.027 | - | 0.355 ± 0.169 | 0.352 ± 0.016 | 0.371 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.003 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.022 ± 0.002 | - | 0.021 ± 0.005 | 0.015 ± 0.004 | 0.150 ± 0.131 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.033 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.015 ± 0.000 | - | 0.016 ± 0.000 | 0.021 ± 0.005 | 0.063 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.002 | 0.005 ± 0.000 | 0.012 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.002 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.005 ± 0.000 | - | 0.010 ± 0.000 | 0.004 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.008 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 2.613 ± 0.225 | - | 2.573 ± 0.061 | 3.031 ± 0.463 | 3.289 ± 0.316 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.007 ± 0.000 | - | 0.012 ± 0.000 | 0.005 ± 0.000 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.093 ± 0.009 | - | 0.103 ± 0.044 | 0.092 ± 0.001 | 0.142 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.002 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.013 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.213 ± 0.040 | - | 0.263 ± 0.019 | 0.458 ± 0.086 | 0.366 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.631 ± 0.110 | - | 0.520 ± 0.162 | 0.681 ± 0.067 | 0.654 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.550 ± 0.169 | - | 1.288 ± 0.161 | 1.813 ± 0.202 | 1.065 ± 0.192 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.314 ± 0.019 | - | 0.314 ± 0.003 | 0.548 ± 0.260 | 0.404 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.051 ± 0.058 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.012 ± 0.003 | - | 0.014 ± 0.004 | 0.104 ± 0.027 | 0.120 ± 0.070 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.038 ± 0.049 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.016 ± 0.006 | - | 0.017 ± 0.002 | 0.094 ± 0.094 | 0.077 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.009 ± 0.000 | - | 0.008 ± 0.002 | 0.038 ± 0.026 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.051 ± 0.035 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.005 ± 0.001 | - | 0.008 ± 0.000 | 0.042 ± 0.016 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.033 ± 0.021 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.030 ± 0.009 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.005 ± 0.001 | - | 0.005 ± 0.000 | 0.045 ± 0.036 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.008 ± 0.000 | - | 0.012 ± 0.003 | 0.034 ± 0.024 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 2.509 ± 0.164 | - | 2.605 ± 0.294 | 3.657 ± 1.293 | 1.644 ± 0.160 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.040 ± 0.006 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.088 ± 0.001 | - | 0.098 ± 0.014 | 0.184 ± 0.024 | 0.180 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.041 ± 0.016 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.046 ± 0.013 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.136 ± 0.005 | - | 0.088 ± 0.001 | 0.183 ± 0.068 | 0.176 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.167 ± 0.008 | - | 0.147 ± 0.001 | 0.169 ± 0.002 | 0.298 ± 0.069 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.342 ± 0.006 | - | 0.341 ± 0.001 | 0.359 ± 0.035 | 0.517 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.052 ± 0.011 | - | 0.059 ± 0.001 | 0.068 ± 0.004 | 0.124 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.008 ± 0.010 | 0.011 ± 0.002 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.014 ± 0.000 | 0.059 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.011 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.012 ± 0.003 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.013 ± 0.002 | 0.036 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.012 ± 0.003 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.002 | 0.013 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.014 ± 0.006 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.384 ± 0.003 | - | 0.435 ± 0.035 | 0.493 ± 0.292 | 0.709 ± 0.310 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.024 ± 0.017 | 0.010 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.027 ± 0.007 | - | 0.032 ± 0.000 | 0.039 ± 0.006 | 0.096 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.003 | 0.014 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.146 ± 0.056 | - | 0.101 ± 0.004 | 0.406 ± 0.343 | 0.268 ± 0.087 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.185 ± 0.014 | - | 0.148 ± 0.011 | 0.428 ± 0.198 | 0.305 ± 0.075 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.358 ± 0.044 | - | 0.337 ± 0.013 | 0.519 ± 0.157 | 0.626 ± 0.243 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.053 ± 0.002 | - | 0.062 ± 0.002 | 0.073 ± 0.027 | 0.130 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.008 | 0.010 ± 0.001 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.005 ± 0.025 | - | 0.010 ± 0.000 | 0.014 ± 0.002 | 0.072 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.001 | - | 0.006 ± 0.000 | 0.022 ± 0.007 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.002 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.001 | 0.061 ± 0.032 | 0.010 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.003 ± 0.001 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.041 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.003 ± 0.001 | - | 0.008 ± 0.000 | 0.012 ± 0.002 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.003 ± 0.001 | - | 0.007 ± 0.034 | 0.023 ± 0.013 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.004 ± 0.001 | - | 0.009 ± 0.002 | 0.018 ± 0.008 | 0.012 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.397 ± 0.004 | - | 0.475 ± 0.017 | 0.480 ± 0.085 | 0.787 ± 0.203 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.003 ± 0.001 | - | 0.009 ± 0.002 | 0.039 ± 0.140 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.028 ± 0.006 | - | 0.039 ± 0.003 | 0.144 ± 0.118 | 0.260 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.001 | 0.045 ± 0.046 | 0.012 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.004 ± 0.001 | - | 0.009 ± 0.000 | 0.091 ± 0.052 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.426 ± 0.112 | - | 1.340 ± 0.247 | 1.565 ± 0.313 | 4.063 ± 1.618 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 4.767 ± 0.789 | - | 4.771 ± 0.457 | 6.130 ± 1.344 | 9.743 ± 2.190 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 15.695 ± 1.129 | - | 17.634 ± 2.309 | 19.500 ± 1.187 | 20.821 ± 1.690 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 6.265 ± 0.379 | - | 6.862 ± 0.937 | 6.120 ± 0.275 | 5.737 ± 0.482 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.003 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.207 ± 0.028 | - | 0.201 ± 0.024 | 0.247 ± 0.075 | 0.288 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.006 ± 0.000 | - | 0.010 ± 0.002 | 0.012 ± 0.003 | 0.027 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.406 ± 0.271 | - | 0.182 ± 0.059 | 0.189 ± 0.011 | 0.306 ± 0.139 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.017 ± 0.002 | - | 0.018 ± 0.000 | 0.024 ± 0.005 | 0.097 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.007 ± 0.001 | - | 0.008 ± 0.002 | 0.010 ± 0.000 | 0.025 ± 0.096 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.014 ± 0.003 | - | 0.018 ± 0.004 | 0.022 ± 0.005 | 0.034 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.052 ± 0.011 | - | 0.046 ± 0.000 | 0.059 ± 0.014 | 0.068 ± 0.166 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.005 | 0.011 ± 0.000 | 0.028 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.018 ± 0.005 | - | 0.020 ± 0.004 | 0.023 ± 0.000 | 0.045 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.061 ± 0.001 | - | 0.061 ± 0.000 | 0.075 ± 0.024 | 0.091 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 49.103 ± 2.822 | - | 53.975 ± 3.081 | 43.839 ± 4.636 | 44.627 ± 2.756 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.011 ± 0.000 | - | 0.013 ± 0.003 | 0.015 ± 0.004 | 0.101 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.476 ± 0.179 | - | 1.448 ± 0.190 | 1.931 ± 0.302 | 1.907 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.038 ± 0.037 | - | 0.041 ± 0.010 | 0.048 ± 0.009 | 0.072 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.116 ± 0.001 | - | 0.124 ± 0.023 | 0.208 ± 0.128 | 0.225 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.185 ± 0.026 | - | 1.230 ± 0.202 | 1.248 ± 0.064 | 3.232 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 4.656 ± 0.464 | - | 4.420 ± 0.163 | 4.608 ± 0.787 | 2.730 ± 2.185 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 15.350 ± 0.570 | - | 15.100 ± 0.238 | 16.828 ± 5.773 | 8.742 ± 0.811 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 6.009 ± 0.103 | - | 6.092 ± 0.076 | 4.717 ± 0.248 | 5.431 ± 0.143 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.183 ± 0.002 | - | 0.185 ± 0.013 | 0.188 ± 0.015 | 0.263 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.009 ± 0.002 | - | 0.009 ± 0.001 | 0.011 ± 0.003 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.180 ± 0.018 | - | 0.189 ± 0.014 | 0.180 ± 0.035 | 0.651 ± 0.261 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.018 ± 0.001 | - | 0.019 ± 0.000 | 0.031 ± 0.004 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.006 ± 0.001 | - | 0.009 ± 0.002 | 0.010 ± 0.003 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.014 ± 0.004 | - | 0.017 ± 0.001 | 0.019 ± 0.004 | 0.032 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.046 ± 0.000 | - | 0.059 ± 0.004 | 0.058 ± 0.007 | 0.064 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.002 | 0.013 ± 0.000 | 0.027 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.019 ± 0.003 | - | 0.021 ± 0.003 | 0.023 ± 0.006 | 0.037 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.061 ± 0.010 | - | 0.064 ± 0.001 | 0.065 ± 0.002 | 0.107 ± 0.215 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 48.327 ± 1.347 | - | 49.071 ± 5.948 | 40.513 ± 1.354 | 21.922 ± 0.650 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.014 ± 0.003 | - | 0.014 ± 0.003 | 0.015 ± 0.000 | 0.036 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 1.462 ± 0.131 | - | 1.461 ± 0.026 | 1.958 ± 0.930 | 1.194 ± 0.148 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.035 ± 0.000 | - | 0.037 ± 0.004 | 0.039 ± 0.000 | 0.060 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.126 ± 0.013 | - | 0.121 ± 0.009 | 0.122 ± 0.007 | 0.221 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.947 ± 0.490 | - | 1.105 ± 0.539 | 0.750 ± 0.123 | 2.908 ± 1.476 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 3.960 ± 0.740 | - | 3.912 ± 1.653 | 3.716 ± 1.326 | 4.039 ± 0.295 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 14.524 ± 3.707 | - | 14.339 ± 3.598 | 15.547 ± 2.155 | 15.523 ± 1.645 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.722 ± 0.486 | - | 1.578 ± 0.141 | 1.823 ± 0.188 | 1.727 ± 0.082 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.001 | 0.015 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.134 ± 0.032 | - | 0.126 ± 0.003 | 0.159 ± 0.212 | 0.225 ± 0.067 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.002 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.100 ± 0.040 | - | 0.236 ± 0.204 | 0.111 ± 0.034 | 0.207 ± 0.106 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.014 ± 0.000 | - | 0.014 ± 0.004 | 0.021 ± 0.005 | 0.025 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.011 ± 0.003 | - | 0.016 ± 0.002 | 0.014 ± 0.001 | 0.026 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.030 ± 0.000 | - | 0.035 ± 0.008 | 0.041 ± 0.004 | 0.054 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.001 | 0.010 ± 0.003 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.014 ± 0.002 | - | 0.017 ± 0.004 | 0.017 ± 0.002 | 0.030 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.094 ± 0.065 | - | 0.047 ± 0.003 | 0.051 ± 0.005 | 0.057 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 18.211 ± 4.580 | - | 15.410 ± 3.922 | 15.375 ± 1.692 | 13.809 ± 1.539 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.013 ± 0.003 | 0.037 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 1.046 ± 0.535 | - | 1.057 ± 0.190 | 1.451 ± 0.385 | 1.153 ± 0.065 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.026 ± 0.006 | - | 0.033 ± 0.007 | 0.028 ± 0.001 | 0.046 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.148 ± 0.092 | - | 0.165 ± 0.087 | 0.243 ± 0.040 | 0.154 ± 0.089 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.573 ± 0.029 | - | 0.624 ± 0.036 | 0.581 ± 0.032 | 2.206 ± 0.081 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 2.924 ± 0.115 | - | 2.827 ± 0.114 | 2.782 ± 0.325 | 2.534 ± 0.709 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 10.710 ± 0.394 | - | 11.680 ± 0.653 | 13.163 ± 3.407 | 6.962 ± 0.746 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 1.538 ± 0.084 | - | 1.574 ± 0.066 | 1.666 ± 0.194 | 1.808 ± 0.087 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.002 | 0.007 ± 0.001 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.124 ± 0.003 | - | 0.125 ± 0.003 | 0.130 ± 0.005 | 0.222 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.001 | 0.008 ± 0.001 | 0.018 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.098 ± 0.001 | - | 0.098 ± 0.001 | 0.100 ± 0.003 | 0.328 ± 0.137 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.013 ± 0.001 | - | 0.017 ± 0.003 | 0.016 ± 0.001 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.001 | 0.008 ± 0.000 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.013 ± 0.002 | - | 0.012 ± 0.003 | 0.013 ± 0.000 | 0.029 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.030 ± 0.001 | - | 0.090 ± 0.001 | 0.034 ± 0.000 | 0.044 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.024 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.015 ± 0.003 | - | 0.015 ± 0.004 | 0.017 ± 0.000 | 0.038 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.041 ± 0.006 | - | 0.059 ± 0.073 | 0.046 ± 0.007 | 0.056 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 12.385 ± 0.849 | - | 12.432 ± 0.479 | 13.721 ± 2.186 | 7.467 ± 0.900 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.007 ± 0.002 | - | 0.010 ± 0.002 | 0.010 ± 0.000 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 0.996 ± 0.053 | - | 0.984 ± 0.064 | 1.132 ± 0.134 | 0.699 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.024 ± 0.003 | - | 0.025 ± 0.007 | 0.027 ± 0.007 | 0.053 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.079 ± 0.001 | - | 0.081 ± 0.001 | 0.084 ± 0.005 | 0.226 ± 0.071 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.215 ± 0.032 | - | unsupported | 0.277 ± 0.033 | 1.774 ± 0.596 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.492 ± 0.088 | - | unsupported | 1.804 ± 0.358 | 2.805 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 3.961 ± 0.264 | - | unsupported | 5.340 ± 0.486 | 5.826 ± 0.317 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.789 ± 0.008 | - | unsupported | 0.946 ± 0.158 | 0.872 ± 1.483 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.082 ± 0.009 | - | unsupported | 0.044 ± 0.010 | 0.057 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | unsupported | 0.021 ± 0.008 | 0.044 ± 0.080 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.072 ± 0.003 | - | unsupported | 0.036 ± 0.005 | 0.043 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | unsupported | 0.012 ± 0.009 | 0.017 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.002 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | unsupported | 0.023 ± 0.004 | 0.016 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.013 ± 0.003 | - | unsupported | 0.017 ± 0.000 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.015 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | unsupported | 0.028 ± 0.008 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.017 ± 0.000 | - | unsupported | 0.021 ± 0.003 | 0.035 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 6.422 ± 0.664 | - | unsupported | 7.183 ± 1.211 | 9.115 ± 2.790 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.019 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.635 ± 0.027 | - | unsupported | 0.391 ± 0.018 | 0.336 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.011 ± 0.000 | - | unsupported | 0.045 ± 0.012 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.031 ± 0.001 | - | unsupported | 0.038 ± 0.005 | 0.053 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.243 ± 0.041 | - | unsupported | 0.228 ± 0.011 | 1.333 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 1.590 ± 0.337 | - | unsupported | 1.474 ± 0.146 | 1.042 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 4.178 ± 0.418 | - | unsupported | 4.255 ± 0.471 | 3.257 ± 0.784 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 0.860 ± 0.125 | - | unsupported | 0.797 ± 0.042 | 0.810 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.039 ± 0.010 | - | unsupported | 0.045 ± 0.018 | 0.047 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.031 ± 0.003 | - | unsupported | 0.034 ± 0.001 | 0.041 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.006 ± 0.003 | - | unsupported | 0.009 ± 0.003 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.016 ± 0.010 | - | unsupported | 0.016 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.007 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.021 ± 0.007 | - | unsupported | 0.020 ± 0.000 | 0.031 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 7.669 ± 1.086 | - | unsupported | 8.211 ± 1.772 | 3.453 ± 0.107 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.001 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.329 ± 0.021 | - | unsupported | 0.297 ± 0.026 | 0.287 ± 0.080 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.012 ± 0.000 | - | unsupported | 0.015 ± 0.001 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.045 ± 0.010 | - | unsupported | 0.034 ± 0.002 | 0.048 ± 0.003 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.837 ± 0.027 | - | unsupported | 0.995 ± 0.076 | 1.205 ± 0.109 | 6.058 ± 2.234 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.063 ± 0.001 | - | unsupported | 0.084 ± 0.019 | 0.256 ± 0.065 | 0.149 ± 0.154 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.117 ± 0.011 | - | unsupported | 0.120 ± 0.054 | 0.234 ± 0.078 | 0.160 ± 0.008 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.295 ± 0.014 | - | unsupported | 0.286 ± 0.024 | 0.411 ± 0.034 | 0.746 ± 0.084 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 1.268 ± 0.353 | - | unsupported | 0.828 ± 0.111 | 1.461 ± 0.927 | 3.544 ± 3.644 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.066 ± 0.008 | - | unsupported | 0.182 ± 0.043 | 0.360 ± 0.166 | 0.093 ± 0.005 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.126 ± 0.030 | - | unsupported | 0.222 ± 0.043 | 0.463 ± 0.284 | 0.218 ± 0.045 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.293 ± 0.040 | - | unsupported | 0.387 ± 0.079 | 0.592 ± 0.151 | 0.834 ± 0.026 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.219 ± 0.001 | - | unsupported | 0.445 ± 0.307 | 0.447 ± 0.024 | 0.690 ± 0.042 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.081 ± 0.076 | - | unsupported | 0.049 ± 0.007 | 0.090 ± 0.033 | 0.082 ± 0.010 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.098 ± 0.002 | - | unsupported | 0.102 ± 0.035 | 0.112 ± 0.001 | 0.105 ± 0.010 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.103 ± 0.002 | - | unsupported | 0.090 ± 0.013 | 0.175 ± 0.012 | 0.293 ± 0.102 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.237 ± 0.004 | - | unsupported | 0.193 ± 0.015 | 0.764 ± 0.631 | 0.550 ± 0.083 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.148 ± 0.001 | - | unsupported | 0.049 ± 0.007 | 0.158 ± 0.097 | 0.067 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.071 ± 0.010 | - | unsupported | 0.054 ± 0.012 | 0.148 ± 0.056 | 0.085 ± 0.012 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.095 ± 0.008 | - | unsupported | 0.094 ± 0.019 | 0.278 ± 0.427 | 0.381 ± 0.180 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.117 ± 0.025 | - | 0.301 ± 0.048 | 0.085 ± 0.011 | 0.186 ± 0.010 | 0.327 ± 0.084 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.040 ± 0.000 | - | 0.050 ± 0.009 | 0.043 ± 0.015 | 0.030 ± 0.004 | 0.047 ± 0.011 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.045 ± 0.010 | - | 0.050 ± 0.009 | 0.054 ± 0.035 | 0.103 ± 0.002 | 0.062 ± 0.040 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.056 ± 0.000 | - | 0.058 ± 0.000 | 0.045 ± 0.010 | 0.069 ± 0.015 | 0.117 ± 0.047 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.124 ± 0.011 | - | 0.109 ± 0.005 | 0.097 ± 0.023 | 0.287 ± 0.249 | 0.264 ± 0.089 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.043 ± 0.004 | - | 0.075 ± 0.009 | 0.044 ± 0.003 | 0.026 ± 0.005 | 0.044 ± 0.005 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.049 ± 0.007 | - | 0.047 ± 0.000 | 0.057 ± 0.016 | 0.031 ± 0.007 | 0.055 ± 0.013 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.063 ± 0.008 | - | 0.070 ± 0.066 | 0.046 ± 0.005 | 0.082 ± 0.008 | 0.115 ± 0.002 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 13.193 ± 4.016 | - | 9.590 ± 1.410 | 30.679 ± 3.274 | 18.087 ± 3.144 | 56.919 ± 20.397 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 11.367 ± 2.325 | - | 11.010 ± 1.741 | 16.246 ± 2.753 | 22.316 ± 5.625 | 366.452 ± 9.779 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 9.839 ± 2.036 | - | 8.231 ± 0.363 | 9.389 ± 1.335 | 10.529 ± 0.585 | 30.370 ± 11.404 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 12.438 ± 4.030 | - | 10.555 ± 1.812 | 8.063 ± 0.874 | 14.878 ± 5.025 | 250.443 ± 44.199 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 6.892 ± 0.209 | - | 6.882 ± 0.168 | 9.358 ± 1.030 | 13.198 ± 4.384 | 13.907 ± 3.746 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 6.926 ± 0.987 | - | 8.164 ± 2.820 | 11.278 ± 3.255 | 11.190 ± 2.676 | 465.588 ± 449.533 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 8.168 ± 0.337 | - | 7.957 ± 0.693 | 7.908 ± 1.528 | 8.467 ± 0.824 | 12.094 ± 4.913 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 8.921 ± 2.604 | - | 8.338 ± 0.765 | 8.306 ± 0.395 | 11.729 ± 1.981 | 267.411 ± 25.790 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 22.765 ± 1.556 | - | 22.957 ± 3.150 | 25.781 ± 1.122 | 29.707 ± 3.784 | 20.500 ± 4.282 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 22.405 ± 1.410 | - | 23.434 ± 3.918 | 24.026 ± 2.687 | 41.159 ± 6.619 | 656.733 ± 346.171 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.321 ± 0.029 | 0.307 ± 0.078 | unsupported | 0.406 ± 0.056 | 0.394 ± 0.029 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 4.676 ± 1.638 | 4.687 ± 0.475 | unsupported | 5.617 ± 1.768 | 5.032 ± 0.675 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.318 ± 0.038 | 0.306 ± 0.019 | unsupported | 0.338 ± 0.034 | 0.769 ± 0.252 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 3.906 ± 0.373 | 4.491 ± 0.510 | unsupported | 4.143 ± 0.841 | 6.578 ± 0.594 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 10.315 ± 2.119 | - | 10.596 ± 3.103 | 10.199 ± 1.044 | 13.488 ± 1.638 | 44.879 ± 6.091 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 9.668 ± 5.963 | - | 8.783 ± 0.248 | 10.493 ± 0.804 | 15.243 ± 4.539 | 394.539 ± 76.399 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 9.408 ± 2.017 | 9.549 ± 1.215 | 5.282 ± 0.313 | 31.808 ± 0.389 | 14.065 ± 3.389 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 9.653 ± 0.290 | 10.059 ± 2.256 | 6.370 ± 1.852 | 41.628 ± 1.979 | 144.630 ± 28.164 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 10.861 ± 3.217 | - | 10.135 ± 1.071 | 12.176 ± 1.662 | 13.249 ± 0.928 | 55.349 ± 34.769 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 9.118 ± 1.886 | - | 8.713 ± 0.233 | 10.609 ± 0.470 | 14.056 ± 2.870 | 328.103 ± 196.999 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.928 ± 0.674 | - | 0.817 ± 0.774 | 2.582 ± 0.315 | 6.627 ± 0.502 | 6.373 ± 1.188 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.760 ± 0.087 | - | 0.807 ± 0.009 | 2.467 ± 0.385 | 5.310 ± 0.481 | 6.747 ± 3.711 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 12.447 ± 1.231 | - | 13.330 ± 1.849 | 11.267 ± 1.226 | 16.044 ± 3.292 | 29.648 ± 6.597 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 10.976 ± 1.227 | - | 11.248 ± 3.041 | 14.069 ± 2.338 | 23.091 ± 5.922 | 943.839 ± 47.197 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 14.116 ± 2.558 | - | 13.800 ± 2.173 | 11.677 ± 1.053 | 13.020 ± 0.571 | 29.749 ± 4.857 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 10.825 ± 0.666 | - | 10.850 ± 0.660 | 15.218 ± 3.193 | 16.651 ± 2.977 | 978.617 ± 56.134 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.644 ± 0.220 | 1.571 ± 0.115 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 15.025 ± 0.733 | 16.624 ± 0.595 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.527 ± 0.145 | 1.500 ± 0.287 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 16.830 ± 0.601 | 17.530 ± 1.996 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.502 ± 0.061 | 1.994 ± 1.258 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 14.140 ± 2.186 | 14.875 ± 1.364 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.558 ± 0.023 | 1.503 ± 0.034 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 15.422 ± 2.054 | 13.840 ± 3.952 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 9.417 ± 2.770 | - | 10.038 ± 1.956 | 8.495 ± 1.305 | 12.272 ± 5.075 | 29.151 ± 6.928 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 9.503 ± 1.385 | - | 7.401 ± 0.021 | 8.693 ± 1.360 | 13.495 ± 2.745 | 247.803 ± 53.065 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 37.746 ± 5.066 | 41.675 ± 3.537 | 37.045 ± 3.768 | 36.216 ± 1.595 | 117.552 ± 9.640 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 34.618 ± 7.562 | 38.664 ± 7.878 | 36.818 ± 6.101 | 60.004 ± 11.716 | 1720.007 ± 178.844 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 16.849 ± 3.585 | - | 19.724 ± 3.442 | 15.395 ± 1.851 | 75.947 ± 2.744 | 48.523 ± 17.889 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 17.169 ± 1.269 | - | 19.874 ± 7.860 | 14.454 ± 4.714 | 62.818 ± 11.115 | 22.071 ± 6.568 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.250 ± 0.049 | 0.266 ± 0.080 | unsupported | 0.060 ± 0.035 | 0.035 ± 0.002 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 3.562 ± 0.971 | 2.987 ± 0.273 | unsupported | 0.875 ± 0.254 | 0.491 ± 0.094 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.230 ± 0.178 | 0.199 ± 0.065 | unsupported | 0.116 ± 0.029 | 0.048 ± 0.011 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 2.175 ± 0.343 | 2.617 ± 0.287 | unsupported | 0.611 ± 0.464 | 0.363 ± 0.116 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.083 ± 0.010 | 0.083 ± 0.003 | unsupported | 0.156 ± 0.128 | 0.193 ± 0.022 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.330 ± 0.452 | 1.289 ± 0.228 | unsupported | 1.467 ± 0.129 | 2.672 ± 0.759 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.128 ± 0.013 | 0.130 ± 0.017 | unsupported | 0.152 ± 0.023 | 0.135 ± 0.028 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.979 ± 0.127 | 0.941 ± 0.398 | unsupported | 1.307 ± 0.568 | 1.388 ± 0.088 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.313 ± 0.045 | 0.286 ± 0.046 | unsupported | 0.348 ± 0.054 | 0.099 ± 0.009 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 4.708 ± 0.462 | 5.193 ± 0.481 | unsupported | 5.239 ± 1.824 | 1.378 ± 0.595 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.189 ± 0.036 | 0.178 ± 0.012 | unsupported | 0.333 ± 0.058 | 0.086 ± 0.014 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 3.057 ± 0.276 | 3.453 ± 1.069 | unsupported | 3.152 ± 0.587 | 0.570 ± 0.098 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.049 ± 0.041 | 0.067 ± 0.003 | unsupported | 0.051 ± 0.016 | 0.087 ± 0.012 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.706 ± 0.101 | 0.821 ± 0.139 | unsupported | 0.574 ± 0.038 | 0.903 ± 0.341 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.055 ± 0.006 | 0.076 ± 0.005 | unsupported | 0.076 ± 0.037 | 0.127 ± 0.057 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 1.592 ± 0.571 | 2.180 ± 0.918 | unsupported | 0.653 ± 0.202 | 0.488 ± 0.061 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.104 ± 0.061 | 0.108 ± 0.025 | unsupported | 0.094 ± 0.025 | 0.113 ± 0.009 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.577 ± 0.559 | 1.392 ± 0.145 | unsupported | 1.726 ± 0.353 | 1.977 ± 0.419 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.102 ± 0.008 | 0.146 ± 0.044 | unsupported | 0.127 ± 0.047 | 0.156 ± 0.051 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 2.702 ± 1.362 | 3.635 ± 0.899 | unsupported | 1.213 ± 0.505 | 1.078 ± 0.376 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.129 ± 0.023 | 0.163 ± 0.010 | unsupported | unsupported | 0.133 ± 0.019 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 2.222 ± 0.484 | 2.363 ± 0.354 | unsupported | unsupported | 2.333 ± 0.867 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.134 ± 0.013 | 0.170 ± 0.025 | unsupported | unsupported | 0.148 ± 0.017 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 3.361 ± 1.368 | 3.165 ± 1.024 | unsupported | unsupported | 1.103 ± 0.848 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.029 ± 0.017 | 0.017 ± 0.009 | unsupported | 0.009 ± 0.003 | 0.050 ± 0.008 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.107 ± 0.040 | 0.152 ± 0.079 | unsupported | 0.090 ± 0.104 | 0.333 ± 0.054 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.013 ± 0.002 | 0.018 ± 0.001 | unsupported | 0.057 ± 0.024 | 0.060 ± 0.008 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.254 ± 0.070 | 0.295 ± 0.230 | unsupported | 0.145 ± 0.041 | 0.177 ± 0.074 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.027 ± 0.002 | 0.038 ± 0.014 | unsupported | 0.028 ± 0.002 | 0.085 ± 0.011 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.454 ± 0.166 | 0.399 ± 0.248 | unsupported | 0.418 ± 0.044 | 0.858 ± 0.215 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.031 ± 0.001 | 0.044 ± 0.003 | unsupported | 0.132 ± 0.020 | 0.106 ± 0.028 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.527 ± 0.107 | 1.171 ± 0.371 | unsupported | 0.718 ± 0.392 | 0.424 ± 0.038 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.052 ± 0.031 | 0.065 ± 0.044 | unsupported | 0.093 ± 0.023 | 0.088 ± 0.003 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.994 ± 0.732 | 0.980 ± 0.312 | unsupported | 1.347 ± 0.117 | 0.879 ± 0.065 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.049 ± 0.028 | 0.043 ± 0.018 | unsupported | 0.189 ± 0.082 | 0.087 ± 0.005 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.651 ± 0.075 | 0.652 ± 0.041 | unsupported | 1.060 ± 0.234 | 0.587 ± 0.115 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.080 ± 0.026 | 0.065 ± 0.034 | unsupported | 0.088 ± 0.016 | 0.090 ± 0.009 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 1.169 ± 0.683 | 1.205 ± 0.688 | unsupported | 1.591 ± 0.424 | 1.068 ± 0.332 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.041 ± 0.003 | 0.043 ± 0.008 | unsupported | 0.192 ± 0.018 | 0.114 ± 0.019 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.661 ± 0.169 | 0.739 ± 0.164 | unsupported | 0.920 ± 0.376 | 0.778 ± 0.199 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.095 ± 0.008 | 0.119 ± 0.034 | unsupported | 0.110 ± 0.008 | 0.116 ± 0.004 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.573 ± 0.446 | 2.392 ± 1.119 | unsupported | 1.837 ± 0.150 | 1.276 ± 0.016 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.096 ± 0.003 | 0.114 ± 0.002 | unsupported | 0.184 ± 0.051 | 0.133 ± 0.004 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 2.608 ± 0.657 | 3.049 ± 0.777 | unsupported | 1.400 ± 0.595 | 1.096 ± 0.259 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.307 ± 0.056 | 0.255 ± 0.034 | unsupported | 0.291 ± 0.102 | 0.320 ± 0.013 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 4.144 ± 1.966 | 4.344 ± 1.129 | unsupported | 5.008 ± 0.805 | 4.737 ± 0.361 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.222 ± 0.119 | 0.240 ± 0.132 | unsupported | 0.387 ± 0.122 | 0.243 ± 0.054 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 2.885 ± 0.302 | 2.507 ± 0.258 | unsupported | 3.136 ± 0.300 | 3.072 ± 0.562 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.572 ± 0.081 | 0.736 ± 0.066 | unsupported | 0.138 ± 0.033 | 0.106 ± 0.004 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 11.941 ± 0.609 | 13.323 ± 1.579 | unsupported | 2.630 ± 0.362 | 1.633 ± 0.401 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.789 ± 0.085 | 1.146 ± 0.090 | unsupported | 0.170 ± 0.035 | 0.127 ± 0.028 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 8.116 ± 2.636 | 9.338 ± 0.795 | unsupported | 2.048 ± 0.386 | 0.591 ± 0.054 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 24.341 ± 8.995 | - | unsupported | 23.575 ± 6.332 | - | 18.322 ± 0.856 | 20.239 ± 7.758 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 20.143 ± 5.413 | - | unsupported | 21.288 ± 2.654 | - | 17.980 ± 1.899 | 20.968 ± 3.451 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 13.668 ± 2.788 | - | unsupported | 13.533 ± 2.519 | - | 14.027 ± 2.694 | 13.805 ± 3.332 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 13.416 ± 0.823 | - | unsupported | 12.611 ± 4.202 | - | 13.964 ± 3.526 | 12.193 ± 1.163 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 12.833 ± 4.733 | - | unsupported | 13.741 ± 0.845 | - | 11.407 ± 1.925 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 13.136 ± 0.893 | - | unsupported | 14.135 ± 0.536 | - | 11.284 ± 0.617 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 23.445 ± 11.115 | - | unsupported | 21.018 ± 2.176 | - | 27.295 ± 8.741 | 26.783 ± 8.396 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 18.442 ± 2.696 | - | unsupported | 21.905 ± 3.580 | - | 21.466 ± 7.435 | 19.809 ± 6.592 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 14.509 ± 2.855 | - | unsupported | 11.141 ± 1.705 | - | 55.584 ± 2.061 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 16.461 ± 8.134 | - | unsupported | 11.540 ± 0.904 | - | 27.778 ± 8.049 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 12.201 ± 11.898 | - | unsupported | 12.197 ± 0.846 | - | 55.837 ± 2.340 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 13.176 ± 1.592 | - | unsupported | 13.232 ± 2.692 | - | 33.169 ± 5.579 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 22.591 ± 1.645 | - | unsupported | 21.329 ± 3.712 | - | 22.869 ± 2.693 | 36.270 ± 19.660 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 19.738 ± 4.101 | - | unsupported | 24.579 ± 4.667 | - | 26.305 ± 6.114 | 20.674 ± 3.502 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 17.061 ± 7.130 | - | unsupported | 12.325 ± 2.253 | - | 13.404 ± 2.002 | 12.726 ± 2.008 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 12.991 ± 1.139 | - | unsupported | 15.496 ± 4.784 | - | 13.045 ± 0.764 | 14.399 ± 3.098 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.208 ± 0.008 | unsupported | unsupported | 0.394 ± 0.313 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 3.173 ± 0.446 | unsupported | unsupported | 4.167 ± 1.035 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.216 ± 0.006 | unsupported | unsupported | 0.419 ± 0.066 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 3.506 ± 0.310 | unsupported | unsupported | 4.625 ± 0.875 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 22.320 ± 6.476 | - | unsupported | 21.547 ± 1.520 | - | 20.248 ± 2.705 | 22.543 ± 3.119 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 18.305 ± 2.380 | - | unsupported | 22.898 ± 5.435 | - | 22.673 ± 4.140 | 21.956 ± 1.040 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 9.814 ± 2.264 | - | 0.004 ± 0.000 | 5.987 ± 1.377 | 5.524 ± 0.423 | 22.749 ± 7.274 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 12.842 ± 2.106 | - | 0.004 ± 0.000 | 4.709 ± 0.821 | 5.718 ± 1.552 | 14.335 ± 4.556 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 12.097 ± 0.559 | - | 28.923 ± 3.894 | 10.210 ± 1.082 | 12.931 ± 2.466 | 15.047 ± 8.349 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 11.321 ± 1.185 | - | 23.652 ± 3.053 | 10.754 ± 1.868 | 13.520 ± 7.238 | 10.325 ± 0.459 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 22.120 ± 5.582 | - | 18.958 ± 5.940 | 17.794 ± 6.822 | 22.708 ± 3.170 | 18.876 ± 6.259 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 15.082 ± 3.008 | - | 19.064 ± 6.484 | 12.594 ± 1.528 | 14.447 ± 2.301 | 18.274 ± 6.309 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 6.357 ± 3.110 | - | 8.265 ± 3.377 | 11.876 ± 2.948 | 12.811 ± 4.291 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 7.670 ± 0.441 | - | 6.957 ± 1.039 | 10.570 ± 0.925 | 10.327 ± 2.341 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 15.948 ± 4.793 | - | 0.002 ± 0.000 | 13.571 ± 2.541 | 11.875 ± 1.185 | 13.637 ± 2.554 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 15.282 ± 1.645 | - | 0.002 ± 0.008 | 10.620 ± 1.755 | 12.966 ± 2.385 | 12.357 ± 0.708 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 49.584 ± 6.460 | - | 0.001 ± 0.000 | 55.105 ± 5.782 | 51.359 ± 4.770 | 230.492 ± 8.114 | 40.536 ± 8.880 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 22.975 ± 2.124 | - | 0.002 ± 0.000 | 45.883 ± 3.481 | 37.105 ± 2.330 | 212.702 ± 23.862 | 17.806 ± 3.456 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 11.075 ± 3.333 | - | 10.287 ± 2.131 | 9.615 ± 0.977 | 6.752 ± 4.928 | 11.764 ± 2.181 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 8.310 ± 2.477 | - | 8.492 ± 5.378 | 6.934 ± 0.771 | 6.744 ± 1.366 | 11.556 ± 1.813 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 9.995 ± 2.137 | - | 12.073 ± 1.525 | 9.779 ± 1.362 | 7.393 ± 1.856 | 11.682 ± 0.516 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 7.788 ± 1.625 | - | 6.766 ± 0.971 | 8.438 ± 2.186 | 6.474 ± 0.983 | 11.791 ± 1.114 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.002 ± 0.001 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.001 ± 0.003 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 4.783458 | 298966.12 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 5.241250 | 327578.12 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 5.234042 | 327127.62 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 6.010417 | 375651.06 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 5.305459 | 331591.19 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 6.059458 | 378716.12 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 15.053834 | 940864.62 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 16.480667 | 1030041.69 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 13.631750 | 851984.38 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 14.100458 | 881278.62 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 6.311834 | 394489.62 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 6.488667 | 405541.69 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 5.139792 | 321237.00 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 4.905792 | 306612.00 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 80.519959 | 5032497.44 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 89.877917 | 5617369.81 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 74.818416 | 4676151.00 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 74.990166 | 4686885.38 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 12.299666 | 768729.12 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 5.403792 | 337737.00 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 5.084125 | 317757.81 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 4.895500 | 305968.75 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 105.248125 | 6578007.81 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 66.286666 | 4142916.62 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 62.499792 | 3906237.00 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 71.858667 | 4491166.69 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 26.304292 | 1644018.25 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 25.128083 | 1570505.19 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 240.397917 | 15024869.81 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 265.979625 | 16623726.56 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 24.439542 | 1527471.38 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 23.995375 | 1499710.94 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 269.287542 | 16830471.38 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 280.474833 | 17529677.06 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 24.033791 | 1502111.94 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 31.898667 | 1993666.69 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 226.246042 | 14140377.62 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 238.002333 | 14875145.81 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 24.921834 | 1557614.62 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 24.040875 | 1502554.69 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 246.747875 | 15421742.19 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 221.433500 | 13839593.75 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.269958 | 35468.09 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.849459 | 60147.80 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 15.986166 | 249783.84 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 17.001167 | 265643.23 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 3.925167 | 490645.88 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 6.999875 | 874984.38 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 28.498084 | 3562260.50 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 23.892125 | 2986515.62 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 3.057416 | 47772.12 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 7.408459 | 115757.17 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 14.738000 | 230281.25 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 12.766750 | 199480.47 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.907042 | 363380.25 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.886625 | 610828.12 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 17.396041 | 2174505.12 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 20.938208 | 2617276.00 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.322375 | 192537.11 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 9.973166 | 155830.72 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.322167 | 83158.86 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.300250 | 82816.41 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 21.373125 | 2671640.62 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.732541 | 1466567.62 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.640000 | 1330000.00 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.309375 | 1288671.88 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.671917 | 135498.70 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 9.708708 | 151698.56 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.198917 | 128108.08 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 8.345625 | 130400.39 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.102666 | 1387833.25 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 10.456417 | 1307052.12 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.833500 | 979187.50 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.528792 | 941099.00 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.317750 | 98714.84 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 22.267250 | 347925.78 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 20.045042 | 313203.78 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 18.329208 | 286393.88 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.022917 | 1377864.62 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 41.912291 | 5239036.38 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 37.665042 | 4708130.25 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 41.545250 | 5193156.25 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.528792 | 86387.38 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 21.281959 | 332530.61 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 12.101708 | 189089.19 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 11.411416 | 178303.38 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.560416 | 570052.00 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 25.213708 | 3151713.50 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 24.457584 | 3057198.00 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 27.622959 | 3452869.88 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.576709 | 87136.08 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.242917 | 50670.58 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.155750 | 49308.59 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.266541 | 66664.70 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.223833 | 902979.12 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.591917 | 573989.62 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.651166 | 706395.75 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.564000 | 820500.00 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.134583 | 127102.86 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.888500 | 76382.81 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.517959 | 54968.11 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.854625 | 75853.52 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.906209 | 488276.12 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.220667 | 652583.38 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 12.732625 | 1591578.12 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 17.437417 | 2179677.12 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.215458 | 112741.53 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.030250 | 94222.66 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.664375 | 104130.86 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.917083 | 108079.42 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 15.815750 | 1976968.75 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 13.806250 | 1725781.25 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 12.615209 | 1576901.12 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.134417 | 1391802.12 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.972375 | 155818.36 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 8.149625 | 127337.89 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.498000 | 101531.25 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 9.362500 | 146289.06 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.620750 | 1077593.75 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 9.704208 | 1213026.00 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 21.617125 | 2702140.62 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 29.078875 | 3634859.38 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.543125 | 133486.33 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.225667 | 128526.05 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 10.449208 | 163268.88 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.667542 | 2333442.75 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 17.778500 | 2222312.50 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 18.901084 | 2362635.50 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.445875 | 147591.80 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.579583 | 134055.98 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 10.891292 | 170176.44 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.825958 | 1103244.75 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 26.888833 | 3361104.12 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 25.319416 | 3164927.00 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 3.197125 | 49955.08 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.581333 | 9083.33 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.853417 | 28959.64 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.117417 | 17459.64 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 2.667375 | 333421.88 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.719292 | 89911.50 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.856750 | 107093.75 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 1.216375 | 152046.88 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 3.818166 | 59658.84 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.673208 | 57393.88 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.843791 | 13184.23 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.136417 | 17756.52 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.415167 | 176895.88 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.162500 | 145312.50 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.032084 | 254010.50 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.357000 | 294625.00 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.438834 | 84981.78 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 1.790958 | 27983.72 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.699333 | 26552.08 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.459208 | 38425.12 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.867375 | 858421.88 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 3.341166 | 417645.75 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.633709 | 454213.62 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.189042 | 398630.25 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.804375 | 106318.36 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 8.430375 | 131724.61 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.959459 | 30616.55 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.796875 | 43701.17 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.396000 | 424500.00 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.741750 | 717718.75 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.214166 | 526770.75 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 9.369583 | 1171197.88 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.611834 | 87684.91 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.946584 | 92915.38 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.299750 | 51558.59 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.163625 | 65056.64 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.035500 | 879437.50 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.774000 | 1346750.00 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.954875 | 994359.38 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.840500 | 980062.50 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.596791 | 87449.86 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 12.067750 | 188558.59 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.110291 | 48598.30 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.728667 | 42635.42 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.695083 | 586885.38 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.476708 | 1059588.50 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.205459 | 650682.38 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.219333 | 652416.62 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.764000 | 90062.50 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.650209 | 88284.52 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.103416 | 79740.88 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.186417 | 65412.77 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 8.541792 | 1067724.00 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 12.725084 | 1590635.50 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.350458 | 1168807.25 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 9.640292 | 1205036.50 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.289708 | 113901.69 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 12.298167 | 192158.86 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.624459 | 41007.17 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.779042 | 43422.53 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.225250 | 778156.25 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.357709 | 919713.62 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.284500 | 660562.50 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.914916 | 739364.50 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.429917 | 116092.45 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.048667 | 110135.42 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.074792 | 94918.62 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 7.619709 | 119057.95 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.204292 | 1275536.50 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.696250 | 1837031.25 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 12.583625 | 1572953.12 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 19.139917 | 2392489.62 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.540208 | 133440.75 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 11.800833 | 184388.02 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.160542 | 96258.47 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.279250 | 113738.28 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.765792 | 1095724.00 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 11.202666 | 1400333.25 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 20.865375 | 2608171.88 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 24.391625 | 3048953.12 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 20.505167 | 320393.23 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 18.596250 | 290566.41 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 19.624750 | 306636.72 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 16.294125 | 254595.70 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 37.897750 | 4737218.75 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 40.065375 | 5008171.88 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 33.154583 | 4144322.88 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 34.748084 | 4343510.50 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 15.554041 | 243031.89 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 24.739917 | 386561.20 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 14.185417 | 221647.14 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 15.346709 | 239792.33 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 24.575625 | 3071953.12 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 25.085208 | 3135651.00 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 23.083792 | 2885474.00 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 20.057084 | 2507135.50 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.766292 | 105723.31 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.801792 | 137528.00 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 36.631958 | 572374.34 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 47.126500 | 736351.56 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.061750 | 1632718.75 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 21.036458 | 2629557.25 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 95.527708 | 11940963.50 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 106.584625 | 13323078.12 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.103625 | 126619.14 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 10.908292 | 170442.06 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 50.482542 | 788789.72 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 73.335417 | 1145865.89 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.729542 | 591192.75 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 16.380209 | 2047526.12 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 64.928708 | 8116088.50 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 74.706000 | 9338250.00 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 6.301542 | 393846.38 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 3.333791 | 208361.94 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 66.672917 | 4167057.31 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 50.762792 | 3172674.50 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 6.696167 | 418510.44 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 3.457791 | 216111.94 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 74.007500 | 4625468.75 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 56.097333 | 3506083.31 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.013625 | 851.56 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.001625 | 101.56 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.013834 | 864.62 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.001292 | 80.75 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.016375 | 1023.44 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.030042 | 1877.62 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000125 | 7.81 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.016000 | 1000.00 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.022667 | 1416.69 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000125 | 7.81 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.014333 | 895.81 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.004708 | 294.25 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000167 | 10.44 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.012625 | 789.06 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.002583 | 161.44 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000084 | 5.25 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.012958 | 809.88 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003292 | 205.75 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000084 | 5.25 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.014458 | 903.62 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003291 | 205.69 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `jax-cpu` is 35.3x faster than the slowest successful cell (`julia-base`, 77.662 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `pytorch-cpu` is 20.9x faster than the slowest successful cell (`julia-base`, 77.662 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-direct` is 19.7x faster than the slowest successful cell (`julia-base`, 77.662 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-trace` is 20.2x faster than the slowest successful cell (`julia-base`, 77.662 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `jax-cpu` is 17.2x faster than the slowest successful cell (`julia-base`, 461.518 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `pytorch-cpu` is 38.8x faster than the slowest successful cell (`julia-base`, 461.518 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-direct` is 38.5x faster than the slowest successful cell (`julia-base`, 461.518 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-trace` is 45.6x faster than the slowest successful cell (`julia-base`, 461.518 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 23.7x faster than the slowest successful cell (`julia-base`, 169.371 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 22.2x faster than the slowest successful cell (`julia-base`, 169.371 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-direct` is 18.9x faster than the slowest successful cell (`julia-base`, 169.371 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-trace` is 24.4x faster than the slowest successful cell (`julia-base`, 169.371 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 51.8x faster than the slowest successful cell (`julia-base`, 152.849 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 29.0x faster than the slowest successful cell (`julia-base`, 152.849 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-direct` is 28.3x faster than the slowest successful cell (`julia-base`, 152.849 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-trace` is 27.9x faster than the slowest successful cell (`julia-base`, 152.849 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `jax-cpu` is 88.8x faster than the slowest successful cell (`julia-base`, 625.633 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `pytorch-cpu` is 83.9x faster than the slowest successful cell (`julia-base`, 625.633 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-direct` is 87.4x faster than the slowest successful cell (`julia-base`, 625.633 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-trace` is 74.2x faster than the slowest successful cell (`julia-base`, 625.633 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 14.6x faster than the slowest successful cell (`pytorch-cpu`, 19.847 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 17.5x faster than the slowest successful cell (`pytorch-cpu`, 19.847 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 15.8x faster than the slowest successful cell (`pytorch-cpu`, 19.847 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 15.2x faster than the slowest successful cell (`pytorch-cpu`, 12.840 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 12.840 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 20.0x faster than the slowest successful cell (`pytorch-cpu`, 18.834 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 17.9x faster than the slowest successful cell (`pytorch-cpu`, 18.834 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 16.3x faster than the slowest successful cell (`pytorch-cpu`, 18.834 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 10.507 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 13.2x faster than the slowest successful cell (`pytorch-cpu`, 10.507 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 10.507 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 690.0x faster than the slowest successful cell (`jax-cpu`, 2.588 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 1399.1x faster than the slowest successful cell (`jax-cpu`, 2.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `pytorch-cpu` is 31.5x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 90.3x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-trace` is 47.7x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 0.011 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`3x4x4`): `tenferro-direct` is 21.0x faster than the slowest successful cell (`jax-cpu`, 0.023 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 43.7x faster than the slowest successful cell (`jax-cpu`, 0.064 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 18.9x faster than the slowest successful cell (`jax-cpu`, 0.064 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.027 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 37.3x faster than the slowest successful cell (`pytorch-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 41.4x faster than the slowest successful cell (`jax-cpu`, 0.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-trace` is 25.4x faster than the slowest successful cell (`jax-cpu`, 0.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 29.1x faster than the slowest successful cell (`pytorch-cpu`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 13.2x faster than the slowest successful cell (`pytorch-cpu`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 37.5x faster than the slowest successful cell (`pytorch-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 16.3x faster than the slowest successful cell (`pytorch-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 27.0x faster than the slowest successful cell (`pytorch-cpu`, 0.053 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 14.1x faster than the slowest successful cell (`pytorch-cpu`, 0.053 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 30.0x faster than the slowest successful cell (`pytorch-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 25.7x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 13.4x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 31.6x faster than the slowest successful cell (`pytorch-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 17.1x faster than the slowest successful cell (`pytorch-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 32.1x faster than the slowest successful cell (`pytorch-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 15.7x faster than the slowest successful cell (`pytorch-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 19.6x faster than the slowest successful cell (`pytorch-cpu`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 29.2x faster than the slowest successful cell (`pytorch-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x32x32`): `tenferro-direct` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.128 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `jax-cpu` is 10.0x faster than the slowest successful cell (`pytorch-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 41.0x faster than the slowest successful cell (`pytorch-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 21.3x faster than the slowest successful cell (`pytorch-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 35.3x faster than the slowest successful cell (`jax-cpu`, 0.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 21.8x faster than the slowest successful cell (`jax-cpu`, 0.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=1, shape=`1x2x2`): `tenferro-direct` is 38.7x faster than the slowest successful cell (`pytorch-cpu`, 0.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=1, shape=`1x2x2`): `tenferro-trace` is 21.2x faster than the slowest successful cell (`pytorch-cpu`, 0.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=1, shape=`4x2x2`): `tenferro-direct` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.024 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=4, shape=`1x2x2`): `tenferro-direct` is 12.9x faster than the slowest successful cell (`jax-cpu`, 0.034 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 15.0x faster than the slowest successful cell (`pytorch-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.016 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`3x2x2`): `tenferro-direct` is 13.6x faster than the slowest successful cell (`jax-cpu`, 0.028 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`1x32x32`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x2x2`): `tenferro-direct` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.031 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 18.1x faster than the slowest successful cell (`jax-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-trace` is 14.0x faster than the slowest successful cell (`jax-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 23.8x faster than the slowest successful cell (`pytorch-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 15.1x faster than the slowest successful cell (`pytorch-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 15.4x faster than the slowest successful cell (`jax-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x32x32`): `tenferro-trace` is 12.9x faster than the slowest successful cell (`jax-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 22.5x faster than the slowest successful cell (`pytorch-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 17.2x faster than the slowest successful cell (`pytorch-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`pytorch-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 20.6x faster than the slowest successful cell (`pytorch-cpu`, 0.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 15.1x faster than the slowest successful cell (`pytorch-cpu`, 0.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 21.4x faster than the slowest successful cell (`pytorch-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 15.2x faster than the slowest successful cell (`pytorch-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 15.2x faster than the slowest successful cell (`pytorch-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 23.9x faster than the slowest successful cell (`pytorch-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 20.3x faster than the slowest successful cell (`pytorch-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 12.2x faster than the slowest successful cell (`pytorch-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 21.5x faster than the slowest successful cell (`pytorch-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 16.5x faster than the slowest successful cell (`pytorch-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 23.1x faster than the slowest successful cell (`pytorch-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 21.5x faster than the slowest successful cell (`pytorch-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `jax-cpu` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 24.1x faster than the slowest successful cell (`pytorch-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 15.2x faster than the slowest successful cell (`pytorch-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 13.8x faster than the slowest successful cell (`jax-cpu`, 0.033 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 25.8x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 14.4x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 13.7x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x32x32,rhs=1`): `tenferro-direct` is 13.5x faster than the slowest successful cell (`jax-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`3x4x4,rhs=1`): `tenferro-direct` is 21.8x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`8x2x2,rhs=1`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.024 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x2x2,rhs=1`): `tenferro-direct` is 15.4x faster than the slowest successful cell (`jax-cpu`, 0.030 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x32x32,rhs=1`): `tenferro-direct` is 13.8x faster than the slowest successful cell (`jax-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x4x4,rhs=1`): `tenferro-direct` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.022 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`3x4x4,rhs=1`): `tenferro-direct` is 26.8x faster than the slowest successful cell (`pytorch-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`3x8x8,rhs=1`): `tenferro-direct` is 14.4x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x2x2,rhs=1`): `tenferro-direct` is 13.6x faster than the slowest successful cell (`pytorch-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x4x4,rhs=1`): `tenferro-direct` is 14.3x faster than the slowest successful cell (`pytorch-cpu`, 0.045 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x8x8,rhs=1`): `tenferro-direct` is 20.6x faster than the slowest successful cell (`pytorch-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x8x8,rhs=1`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svdvals` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 18.6x faster than the slowest successful cell (`jax-cpu`, 0.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `jax-cpu` is 16.4x faster than the slowest successful cell (`julia-base`, 366.452 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `pytorch-cpu` is 22.6x faster than the slowest successful cell (`julia-base`, 366.452 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-direct` is 32.2x faster than the slowest successful cell (`julia-base`, 366.452 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-trace` is 33.3x faster than the slowest successful cell (`julia-base`, 366.452 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 16.8x faster than the slowest successful cell (`julia-base`, 250.443 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 31.1x faster than the slowest successful cell (`julia-base`, 250.443 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 20.1x faster than the slowest successful cell (`julia-base`, 250.443 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 23.7x faster than the slowest successful cell (`julia-base`, 250.443 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `jax-cpu` is 41.6x faster than the slowest successful cell (`julia-base`, 465.588 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `pytorch-cpu` is 41.3x faster than the slowest successful cell (`julia-base`, 465.588 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-direct` is 67.2x faster than the slowest successful cell (`julia-base`, 465.588 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-trace` is 57.0x faster than the slowest successful cell (`julia-base`, 465.588 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `jax-cpu` is 22.8x faster than the slowest successful cell (`julia-base`, 267.411 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `pytorch-cpu` is 32.2x faster than the slowest successful cell (`julia-base`, 267.411 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-direct` is 30.0x faster than the slowest successful cell (`julia-base`, 267.411 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-trace` is 32.1x faster than the slowest successful cell (`julia-base`, 267.411 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `jax-cpu` is 16.0x faster than the slowest successful cell (`julia-base`, 656.733 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 27.3x faster than the slowest successful cell (`julia-base`, 656.733 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-direct` is 29.3x faster than the slowest successful cell (`julia-base`, 656.733 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-trace` is 28.0x faster than the slowest successful cell (`julia-base`, 656.733 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 25.9x faster than the slowest successful cell (`julia-base`, 394.539 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 37.6x faster than the slowest successful cell (`julia-base`, 394.539 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-direct` is 40.8x faster than the slowest successful cell (`julia-base`, 394.539 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-trace` is 44.9x faster than the slowest successful cell (`julia-base`, 394.539 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 22.7x faster than the slowest successful cell (`julia-base`, 144.630 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-eager` is 15.0x faster than the slowest successful cell (`julia-base`, 144.630 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-trace` is 14.4x faster than the slowest successful cell (`julia-base`, 144.630 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 23.3x faster than the slowest successful cell (`julia-base`, 328.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 30.9x faster than the slowest successful cell (`julia-base`, 328.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 36.0x faster than the slowest successful cell (`julia-base`, 328.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 37.7x faster than the slowest successful cell (`julia-base`, 328.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `jax-cpu` is 40.9x faster than the slowest successful cell (`julia-base`, 943.839 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 67.1x faster than the slowest successful cell (`julia-base`, 943.839 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 86.0x faster than the slowest successful cell (`julia-base`, 943.839 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 83.9x faster than the slowest successful cell (`julia-base`, 943.839 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `jax-cpu` is 58.8x faster than the slowest successful cell (`julia-base`, 978.617 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 64.3x faster than the slowest successful cell (`julia-base`, 978.617 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 90.4x faster than the slowest successful cell (`julia-base`, 978.617 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 90.2x faster than the slowest successful cell (`julia-base`, 978.617 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 18.4x faster than the slowest successful cell (`julia-base`, 247.803 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 28.5x faster than the slowest successful cell (`julia-base`, 247.803 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 26.1x faster than the slowest successful cell (`julia-base`, 247.803 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 33.5x faster than the slowest successful cell (`julia-base`, 247.803 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 28.7x faster than the slowest successful cell (`julia-base`, 1720.007 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 46.7x faster than the slowest successful cell (`julia-base`, 1720.007 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-eager` is 49.7x faster than the slowest successful cell (`julia-base`, 1720.007 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-trace` is 44.5x faster than the slowest successful cell (`julia-base`, 1720.007 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 15.8x faster than the slowest successful cell (`tenferro-eager`, 9.338 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 5747.7x faster than the slowest successful cell (`julia-base`, 22.749 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 3865.8x faster than the slowest successful cell (`julia-base`, 14.335 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 9113.0x faster than the slowest successful cell (`tenferro-direct`, 15.948 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 9167.5x faster than the slowest successful cell (`tenferro-direct`, 15.282 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 172782.9x faster than the slowest successful cell (`julia-base`, 230.492 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `strided-jl` is 11.9x faster than the slowest successful cell (`julia-base`, 212.702 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 124532.6x faster than the slowest successful cell (`julia-base`, 212.702 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 375.6x faster than the slowest successful cell (`tenferro-direct`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 177.1x faster than the slowest successful cell (`tenferro-direct`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 112.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 78.9x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 162.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 180.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 4471.7 GB/s at 0.004 ms. The conservative physical lower bound is 0.168 ms, so the cell is 44.7x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 11493.5 GB/s at 0.001 ms. The conservative physical lower bound is 0.042 ms, so the cell is 28.7x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 67821.0 GB/s at 0.004 ms. The conservative physical lower bound is 2.684 ms, so the cell is 678.2x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 72393.6 GB/s at 0.004 ms. The conservative physical lower bound is 0.671 ms, so the cell is 181.0x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 153391.7 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1533.9x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 161029.1 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 402.6x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 100613.0 GB/s at 0.001 ms. The conservative physical lower bound is 1.342 ms, so the cell is 1006.1x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 78581.8 GB/s at 0.002 ms. The conservative physical lower bound is 0.336 ms, so the cell is 196.5x faster than the bound.

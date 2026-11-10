# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20261110_174931/run.yaml`
- Timestamp: `2026-11-10T17:49:31.775148+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20261110_174931`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

## CPU Information

- Model: `Apple M2 Pro (Virtual)`
- Vendor: `Apple`
- Logical CPUs: `5`
- Physical CPUs: `5`
- Sockets: `1`
- Cores per socket: `5`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Standard: 5 physical / 5 logical (L1i 128 KiB, L1d 64 KiB, L2 4 MiB, 5 CPUs/L2)`
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

- CSV: `data/results/mac-cpu/cpu/public_api/20261110_174931/cpu_public_api_t1_20261110_174931.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20261110_174931/cpu_public_api_t4_20261110_174931.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 3.167 ± 0.030 | - | 3.175 ± 0.065 | 3.391 ± 0.074 | 1.182 ± 0.097 | 4.364 ± 0.172 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 3.693 ± 0.138 | - | 3.604 ± 0.306 | 3.235 ± 0.093 | 1.137 ± 0.190 | 3.069 ± 1.342 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 9.365 ± 0.551 | - | 9.387 ± 0.968 | 6.823 ± 0.112 | 8.171 ± 0.532 | 9.353 ± 0.967 | 8.796 ± 0.973 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 3.587 ± 0.658 | - | 4.033 ± 1.007 | 4.273 ± 1.098 | 4.689 ± 0.463 | 7.859 ± 1.776 | 4.803 ± 0.996 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 41.478 ± 6.671 | - | 43.035 ± 5.345 | 22.697 ± 0.787 | 31.961 ± 5.322 | 45.762 ± 2.224 | 53.396 ± 5.399 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 14.058 ± 2.232 | - | 11.757 ± 2.921 | 5.894 ± 0.739 | 10.957 ± 0.341 | 46.043 ± 3.204 | 14.234 ± 1.497 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 9.056 ± 0.568 | - | 8.660 ± 0.767 | 8.480 ± 0.385 | 46.055 ± 3.291 | 56.815 ± 2.614 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 8.514 ± 0.733 | - | 8.748 ± 0.877 | 8.384 ± 0.366 | 13.558 ± 0.778 | 14.951 ± 5.074 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 8.589 ± 0.784 | - | unsupported | 8.393 ± 0.390 | 49.426 ± 3.984 | 57.565 ± 3.311 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 9.698 ± 0.950 | - | unsupported | 8.902 ± 1.649 | 13.667 ± 0.081 | 17.137 ± 4.753 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 9.533 ± 0.164 | - | 9.717 ± 0.173 | 8.062 ± 0.094 | 18.135 ± 0.851 | 8.336 ± 0.812 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 9.363 ± 0.491 | - | 9.653 ± 0.190 | 9.353 ± 1.319 | 16.607 ± 0.997 | 12.722 ± 1.874 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 32.821 ± 1.943 | - | 32.548 ± 3.014 | 30.669 ± 5.790 | 42.570 ± 3.647 | 46.849 ± 2.708 | 47.131 ± 2.522 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 8.214 ± 1.608 | - | 8.618 ± 1.061 | 8.608 ± 0.798 | 19.277 ± 5.184 | 44.735 ± 3.041 | 11.062 ± 0.296 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 46.208 ± 3.064 | - | 45.853 ± 0.743 | 48.584 ± 7.961 | 85.318 ± 4.043 | 72.353 ± 8.452 | 71.218 ± 4.021 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 10.725 ± 0.917 | - | 11.213 ± 1.436 | 18.309 ± 1.155 | 31.927 ± 2.956 | 72.287 ± 6.014 | 24.177 ± 3.545 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 5.851 ± 0.510 | - | 6.773 ± 0.521 | 8.229 ± 1.129 | 5.442 ± 0.902 | 6.548 ± 0.925 | 6.227 ± 1.565 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 3.540 ± 0.425 | - | 3.026 ± 0.381 | 3.615 ± 0.738 | 4.190 ± 0.871 | 7.258 ± 0.796 | 4.013 ± 0.914 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 0.838 ± 0.014 | - | 7.005 ± 1.317 | 4.843 ± 0.320 | 4.439 ± 0.157 | 7.856 ± 0.228 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 0.523 ± 0.105 | - | 1.923 ± 1.121 | 1.774 ± 1.041 | 1.993 ± 0.090 | 8.498 ± 0.752 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 5.932 ± 0.126 | - | 5.480 ± 0.056 | 6.395 ± 0.259 | 5.630 ± 0.359 | 8.291 ± 0.343 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 7.029 ± 0.772 | - | 6.794 ± 0.747 | 5.947 ± 0.402 | 5.219 ± 0.340 | 4.331 ± 0.414 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 4.004 ± 0.198 | - | 4.003 ± 0.121 | 4.081 ± 0.471 | 1.844 ± 0.223 | 5.174 ± 0.604 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 5.187 ± 0.286 | - | 5.090 ± 0.173 | 5.001 ± 0.943 | 1.939 ± 0.042 | 2.342 ± 0.071 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 5.203 ± 0.100 | - | 5.589 ± 0.128 | 5.421 ± 1.206 | 3.597 ± 0.647 | 7.119 ± 0.265 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 6.167 ± 0.248 | - | 6.476 ± 0.294 | 5.462 ± 0.540 | 3.960 ± 0.223 | 5.787 ± 1.298 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 7.935 ± 0.245 | - | 7.692 ± 0.577 | 8.188 ± 0.596 | 47.028 ± 6.518 | 55.393 ± 2.872 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 9.587 ± 1.107 | - | 9.745 ± 0.935 | 8.337 ± 0.384 | 14.040 ± 1.554 | 14.674 ± 6.293 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 6.742 ± 0.474 | - | unsupported | 6.708 ± 0.348 | 43.093 ± 10.045 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 6.820 ± 0.355 | - | unsupported | 7.245 ± 0.373 | 12.274 ± 0.376 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 7.564 ± 0.637 | - | 7.369 ± 1.175 | 10.382 ± 0.719 | 16.709 ± 1.052 | 7.108 ± 1.826 | 13.759 ± 1.327 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 4.008 ± 1.018 | - | 5.749 ± 0.680 | 5.572 ± 1.172 | 5.648 ± 0.241 | 7.920 ± 0.556 | 5.638 ± 0.653 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 9.918 ± 0.214 | - | 11.561 ± 0.886 | 12.378 ± 2.256 | 21.958 ± 2.925 | 11.355 ± 0.569 | 18.814 ± 6.097 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 6.707 ± 1.311 | - | 6.727 ± 0.632 | 6.852 ± 1.870 | 11.812 ± 2.260 | 11.178 ± 0.487 | 8.799 ± 0.452 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 26.950 ± 0.685 | - | 27.581 ± 4.568 | 31.160 ± 6.766 | 24.634 ± 0.087 | 56.549 ± 4.569 | 51.410 ± 8.612 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 7.578 ± 1.499 | - | 7.703 ± 0.465 | 9.318 ± 1.198 | 9.176 ± 1.125 | 50.369 ± 3.825 | 15.357 ± 4.915 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 5.686 ± 1.251 | - | 6.310 ± 1.049 | 4.503 ± 0.305 | 6.090 ± 0.442 | 3.775 ± 0.235 | 5.853 ± 4.245 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 3.475 ± 0.274 | - | 2.920 ± 0.175 | 3.313 ± 0.003 | 3.325 ± 0.462 | 4.282 ± 0.383 | 2.875 ± 0.777 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 10.926 ± 1.127 | - | 10.644 ± 0.536 | 15.388 ± 4.083 | 15.786 ± 1.776 | 13.152 ± 0.569 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 6.410 ± 1.446 | - | 4.454 ± 0.141 | 7.417 ± 0.848 | 6.711 ± 0.364 | 12.300 ± 0.214 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 27.337 ± 0.259 | - | 27.930 ± 0.670 | 28.476 ± 4.480 | 23.674 ± 0.484 | 34.308 ± 5.370 | 35.617 ± 1.990 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 8.140 ± 0.083 | - | 7.026 ± 0.989 | 8.684 ± 0.884 | 9.125 ± 0.261 | 36.849 ± 4.803 | 10.646 ± 3.405 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 10.461 ± 0.225 | - | 10.531 ± 0.798 | 12.652 ± 1.637 | 21.421 ± 1.525 | 11.077 ± 0.839 | 13.385 ± 0.870 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 8.363 ± 1.332 | - | 6.834 ± 1.309 | 5.686 ± 2.074 | 9.511 ± 2.079 | 13.309 ± 0.675 | 8.662 ± 1.548 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 16.315 ± 0.766 | - | 16.116 ± 0.160 | 24.758 ± 6.391 | 14.275 ± 0.662 | 29.637 ± 1.709 | 33.089 ± 1.791 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 3.988 ± 0.165 | - | 4.822 ± 1.257 | 5.974 ± 0.023 | 5.545 ± 0.451 | 32.301 ± 2.628 | 13.474 ± 6.034 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 24.089 ± 2.315 | - | 21.987 ± 3.701 | 27.755 ± 2.984 | 16.857 ± 0.987 | 26.464 ± 0.579 | 22.193 ± 5.055 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 5.924 ± 0.031 | - | 6.335 ± 3.869 | 6.634 ± 0.696 | 5.586 ± 0.303 | 25.181 ± 1.580 | 6.673 ± 1.155 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 27.581 ± 2.460 | - | 28.140 ± 0.647 | 33.433 ± 2.606 | 22.834 ± 5.142 | 39.080 ± 8.835 | 38.372 ± 4.986 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 7.931 ± 0.221 | - | 6.905 ± 0.146 | 12.056 ± 5.659 | 8.233 ± 0.190 | 39.150 ± 1.462 | 10.896 ± 0.787 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 14.316 ± 1.894 | - | 13.863 ± 1.050 | 17.950 ± 2.566 | 14.837 ± 0.049 | 19.050 ± 2.256 | 24.973 ± 5.833 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 6.672 ± 3.579 | - | 4.110 ± 0.295 | 5.191 ± 1.009 | 5.433 ± 0.080 | 21.339 ± 0.837 | 7.244 ± 0.298 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.283 ± 0.027 | 0.263 ± 0.019 | unsupported | unsupported | 0.203 ± 0.002 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.279 ± 0.018 | 0.314 ± 0.026 | unsupported | unsupported | 0.206 ± 0.062 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 12.013 ± 1.082 | - | 12.945 ± 0.768 | 11.034 ± 5.579 | 21.889 ± 3.027 | 13.052 ± 0.763 | 14.115 ± 3.067 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 7.470 ± 1.505 | - | 7.629 ± 0.993 | 8.891 ± 1.234 | 10.149 ± 0.755 | 10.946 ± 0.196 | 7.309 ± 1.903 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 12.781 ± 0.721 | - | 13.310 ± 0.706 | 15.607 ± 0.592 | 19.424 ± 1.685 | 10.106 ± 0.282 | 14.845 ± 3.397 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 8.883 ± 1.250 | - | 7.471 ± 2.397 | 7.884 ± 1.269 | 11.127 ± 3.577 | 13.874 ± 1.619 | 9.283 ± 2.948 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 12.190 ± 1.447 | - | 11.722 ± 2.105 | 11.210 ± 1.578 | 14.397 ± 2.322 | 12.665 ± 1.111 | 21.459 ± 1.348 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 8.727 ± 1.489 | - | 8.373 ± 2.552 | 8.480 ± 1.368 | 9.579 ± 2.593 | 11.427 ± 0.954 | 7.962 ± 1.960 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 7.586 ± 0.388 | - | 7.251 ± 0.186 | 7.024 ± 1.459 | 15.688 ± 1.475 | 9.106 ± 0.310 | 11.316 ± 3.774 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 4.342 ± 0.758 | - | 4.686 ± 1.031 | 6.141 ± 0.719 | 5.463 ± 0.733 | 8.839 ± 0.350 | 4.843 ± 1.468 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 39.420 ± 2.719 | - | 35.459 ± 5.200 | 56.368 ± 9.124 | 30.812 ± 5.953 | 84.721 ± 1.757 | 88.392 ± 11.192 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 8.503 ± 0.195 | - | 8.416 ± 0.073 | 14.702 ± 0.144 | 12.165 ± 0.221 | 75.396 ± 5.104 | 21.489 ± 7.086 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 8.037 ± 0.365 | - | 8.418 ± 0.076 | 12.569 ± 0.435 | 26.522 ± 3.149 | 6.935 ± 0.362 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 2.866 ± 1.525 | - | 4.295 ± 1.020 | 2.793 ± 0.084 | 9.884 ± 0.314 | 6.693 ± 1.157 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 0.711 ± 0.113 | - | 0.705 ± 0.011 | 14.037 ± 0.511 | 2.699 ± 0.746 | 0.667 ± 0.084 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.523 ± 0.009 | - | 0.381 ± 0.988 | 9.494 ± 1.884 | 0.877 ± 0.080 | 0.673 ± 0.021 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 0.780 ± 0.105 | - | 0.820 ± 0.060 | 1.590 ± 0.119 | 0.655 ± 0.034 | 0.609 ± 0.030 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 0.507 ± 0.010 | - | 1.322 ± 2.220 | 0.476 ± 0.072 | 0.557 ± 0.054 | 0.734 ± 0.070 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 6.759 ± 0.338 | - | 6.540 ± 0.448 | 12.518 ± 0.239 | 22.135 ± 3.424 | 6.177 ± 0.539 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 3.522 ± 0.180 | - | 3.644 ± 1.025 | 3.358 ± 0.085 | 9.274 ± 0.098 | 5.754 ± 0.464 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 0.845 ± 0.105 | - | 0.818 ± 0.051 | 15.161 ± 2.565 | 1.559 ± 0.119 | 0.507 ± 0.012 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.358 ± 0.096 | - | 0.293 ± 0.033 | 9.284 ± 1.687 | 0.984 ± 0.040 | 0.655 ± 0.059 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 2.846 ± 0.113 | - | 2.962 ± 0.455 | 6.441 ± 1.017 | 2.761 ± 0.026 | 2.739 ± 0.057 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 4.734 ± 0.577 | - | 3.458 ± 0.921 | 1.501 ± 0.014 | 2.185 ± 0.155 | 3.436 ± 0.221 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 6.453 ± 0.940 | - | 7.249 ± 0.851 | 5.489 ± 0.255 | 36.012 ± 0.517 | 5.307 ± 0.437 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 2.717 ± 1.018 | - | 2.790 ± 0.581 | 4.353 ± 0.015 | 14.790 ± 0.111 | 6.094 ± 0.422 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.815 ± 0.119 | - | 0.833 ± 0.213 | 4.239 ± 0.135 | 1.584 ± 0.062 | 0.537 ± 0.024 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.352 ± 0.024 | - | 0.398 ± 0.041 | 1.678 ± 0.226 | 0.901 ± 0.111 | 0.676 ± 0.087 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 0.900 ± 0.138 | - | 0.955 ± 0.191 | 0.658 ± 0.136 | 0.814 ± 0.022 | 0.851 ± 0.144 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.526 ± 0.052 | - | 0.719 ± 0.458 | 0.401 ± 0.088 | 0.561 ± 0.048 | 0.754 ± 0.069 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 5.157 ± 0.135 | - | 5.648 ± 0.761 | 6.232 ± 1.216 | 35.764 ± 5.513 | 5.083 ± 0.030 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 2.236 ± 0.011 | - | 2.309 ± 0.923 | 4.360 ± 0.019 | 11.535 ± 1.161 | 5.922 ± 0.096 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.683 ± 0.212 | - | 0.705 ± 0.144 | 4.088 ± 0.051 | 1.599 ± 0.088 | 0.477 ± 0.017 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.211 ± 0.037 | - | 0.355 ± 0.127 | 1.385 ± 0.195 | 0.906 ± 0.092 | 0.682 ± 0.083 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 0.710 ± 0.115 | - | 0.703 ± 0.084 | 0.496 ± 0.070 | 0.678 ± 0.005 | 0.589 ± 0.013 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.678 ± 0.068 | - | 0.563 ± 0.083 | 0.386 ± 0.126 | 0.527 ± 0.156 | 0.744 ± 0.104 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.853 ± 0.014 | 0.853 ± 0.041 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.467 ± 0.046 | 0.446 ± 0.154 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 25.126 ± 0.286 | - | 25.899 ± 1.551 | 26.495 ± 1.020 | 33.594 ± 8.441 | 23.679 ± 4.228 | 26.804 ± 1.528 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 6.413 ± 0.391 | - | 8.582 ± 2.746 | 7.288 ± 0.115 | 11.591 ± 0.168 | 24.216 ± 6.356 | 11.035 ± 2.878 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 16.736 ± 0.244 | - | 17.119 ± 1.245 | 19.105 ± 2.783 | 36.327 ± 6.451 | 37.312 ± 1.341 | 37.436 ± 2.672 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 5.998 ± 0.624 | - | 5.790 ± 1.346 | 4.084 ± 0.046 | 11.645 ± 0.774 | 39.355 ± 4.115 | 11.196 ± 3.051 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 15.377 ± 2.237 | - | 15.227 ± 0.310 | 15.019 ± 0.379 | 25.392 ± 1.785 | 21.993 ± 1.566 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 10.214 ± 3.668 | - | 12.669 ± 2.032 | 7.685 ± 2.566 | 11.190 ± 0.850 | 18.007 ± 0.652 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 8.559 ± 1.775 | - | 8.607 ± 0.796 | 10.175 ± 3.567 | 17.701 ± 2.797 | 8.030 ± 0.541 | 20.834 ± 1.861 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 6.082 ± 1.617 | - | 4.698 ± 0.976 | 5.520 ± 1.778 | 6.027 ± 0.865 | 8.015 ± 0.340 | 6.451 ± 0.169 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 21.141 ± 0.194 | - | 21.101 ± 0.254 | 29.057 ± 2.874 | 27.252 ± 6.230 | 26.339 ± 2.115 | 27.290 ± 1.665 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 5.444 ± 0.052 | - | 6.974 ± 1.549 | 11.462 ± 3.933 | 8.374 ± 0.444 | 30.590 ± 3.474 | 10.163 ± 3.621 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 10.723 ± 0.363 | - | 10.744 ± 0.603 | 9.816 ± 0.066 | 21.144 ± 6.369 | 28.228 ± 2.372 | 23.970 ± 3.355 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 4.075 ± 1.449 | - | 5.001 ± 0.247 | 3.303 ± 0.070 | 6.749 ± 1.453 | 26.000 ± 3.280 | 8.793 ± 0.575 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 10.134 ± 2.237 | - | 10.185 ± 0.601 | 10.945 ± 4.233 | 22.409 ± 0.766 | 16.057 ± 1.175 | 14.041 ± 1.888 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 7.127 ± 2.374 | - | 7.819 ± 0.981 | 9.532 ± 0.377 | 10.424 ± 3.646 | 13.776 ± 1.550 | 7.868 ± 1.351 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 58.851 ± 5.851 | - | 60.590 ± 19.741 | 62.748 ± 14.141 | 11.714 ± 0.066 | 43.489 ± 2.119 | 41.471 ± 6.380 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 19.271 ± 3.428 | - | 15.290 ± 0.690 | 14.828 ± 0.104 | 4.455 ± 0.081 | 41.882 ± 2.300 | 11.048 ± 0.567 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 15.805 ± 0.565 | 15.225 ± 0.936 | unsupported | 18.306 ± 3.228 | 18.857 ± 1.511 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 12.392 ± 2.713 | 11.313 ± 2.654 | unsupported | 7.788 ± 1.395 | 13.586 ± 1.265 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.580 ± 0.023 | - | 0.571 ± 0.041 | 0.445 ± 0.078 | 0.349 ± 0.013 | 0.528 ± 0.028 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.167 ± 0.016 | - | 0.163 ± 0.227 | 0.336 ± 0.016 | 0.404 ± 0.069 | 0.473 ± 0.073 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.298 ± 0.012 | - | 0.318 ± 0.032 | 0.594 ± 0.061 | 0.480 ± 0.032 | 0.435 ± 0.048 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.206 ± 0.001 | - | 0.215 ± 0.010 | 0.456 ± 0.018 | 0.520 ± 0.207 | 0.511 ± 0.023 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 0.656 ± 0.062 | - | unsupported | 0.833 ± 0.188 | 0.819 ± 0.059 | 0.733 ± 0.088 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.375 ± 0.043 | - | unsupported | 0.868 ± 0.016 | 0.703 ± 0.034 | 0.759 ± 0.182 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.293 ± 0.014 | - | 0.355 ± 0.015 | 0.319 ± 0.028 | 0.311 ± 0.036 | 0.531 ± 0.052 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.120 ± 0.004 | - | 0.236 ± 0.266 | 0.309 ± 0.018 | 0.290 ± 0.041 | 0.594 ± 0.030 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.304 ± 0.011 | - | 0.380 ± 0.041 | 0.885 ± 0.052 | 0.756 ± 0.084 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.147 ± 0.002 | - | 0.210 ± 0.003 | 0.757 ± 0.043 | 0.630 ± 0.097 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.542 ± 0.086 | - | 0.523 ± 0.048 | 0.776 ± 0.029 | 0.735 ± 0.074 | 1.386 ± 0.361 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.239 ± 0.019 | - | 0.152 ± 0.010 | 0.305 ± 0.021 | 0.556 ± 0.082 | 1.209 ± 0.235 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.477 ± 0.020 | - | 0.533 ± 0.020 | 0.384 ± 0.035 | 0.381 ± 0.018 | 0.398 ± 0.038 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.575 ± 0.059 | - | 3.098 ± 2.944 | 0.432 ± 0.034 | 0.533 ± 0.060 | 0.592 ± 0.027 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 0.697 ± 0.060 | - | 0.002 ± 0.000 | 0.751 ± 0.033 | 0.623 ± 0.033 | 1.273 ± 0.082 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.525 ± 0.010 | - | 0.001 ± 0.000 | 0.608 ± 0.020 | 0.700 ± 0.066 | 1.202 ± 0.130 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.061 ± 0.001 | - | 0.069 ± 0.005 | 0.092 ± 0.000 | 0.074 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.096 ± 0.008 | - | 0.115 ± 0.001 | 0.135 ± 0.002 | 0.141 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.196 ± 0.001 | - | 0.211 ± 0.001 | 0.219 ± 0.000 | 0.171 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.019 ± 0.001 | - | 0.019 ± 0.001 | 0.024 ± 0.003 | 0.111 ± 0.067 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.001 | 0.036 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.001 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.035 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.118 ± 0.002 | - | 0.117 ± 0.003 | 0.155 ± 0.006 | 0.250 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.010 ± 0.001 | - | 0.012 ± 0.001 | 0.014 ± 0.001 | 0.056 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.036 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.058 ± 0.000 | - | 0.120 ± 0.014 | 0.159 ± 0.012 | 0.106 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.091 ± 0.007 | - | 0.126 ± 0.009 | 0.177 ± 0.017 | 0.118 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.204 ± 0.005 | - | 0.229 ± 0.003 | 0.224 ± 0.008 | 0.256 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.018 ± 0.001 | - | 0.020 ± 0.002 | 0.070 ± 0.007 | 0.066 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.082 ± 0.025 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.064 ± 0.029 | 0.016 ± 0.059 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.003 ± 0.001 | - | 0.003 ± 0.000 | 0.070 ± 0.022 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.090 ± 0.024 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.060 ± 0.030 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.001 | - | 0.003 ± 0.000 | 0.080 ± 0.022 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.084 ± 0.028 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.088 ± 0.013 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.083 ± 0.017 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.076 ± 0.032 | 0.063 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.117 ± 0.004 | - | 0.128 ± 0.002 | 0.163 ± 0.003 | 0.249 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.076 ± 0.017 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.076 ± 0.038 | 0.088 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.085 ± 0.010 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.058 ± 0.011 | 0.048 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.264 ± 0.036 | - | 0.277 ± 0.027 | 0.365 ± 0.005 | 0.814 ± 0.095 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.626 ± 0.099 | - | 1.606 ± 0.124 | 1.646 ± 0.211 | 1.873 ± 0.170 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 5.982 ± 0.297 | - | 5.729 ± 0.150 | 5.637 ± 0.655 | 5.532 ± 0.071 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 4.713 ± 0.168 | - | 3.967 ± 0.082 | 3.478 ± 0.108 | 3.069 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.082 ± 0.008 | - | 0.083 ± 0.006 | 0.084 ± 0.004 | 0.131 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.008 ± 0.001 | - | 0.007 ± 0.001 | 0.008 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.001 | 0.005 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.002 | 0.008 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.016 ± 0.000 | - | 0.017 ± 0.000 | 0.021 ± 0.004 | 0.074 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.005 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.009 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.021 ± 0.003 | - | 0.021 ± 0.000 | 0.024 ± 0.000 | 0.072 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 33.914 ± 1.795 | - | 36.892 ± 2.152 | 25.011 ± 2.612 | 23.902 ± 2.580 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.001 | 0.006 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.730 ± 0.049 | - | 0.733 ± 0.030 | 0.711 ± 0.020 | 0.784 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.013 ± 0.004 | - | 0.017 ± 0.001 | 0.015 ± 0.000 | 0.030 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.041 ± 0.001 | - | 0.042 ± 0.004 | 0.046 ± 0.005 | 0.091 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.277 ± 0.034 | - | 0.245 ± 0.014 | 0.369 ± 0.017 | 0.780 ± 0.091 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 1.523 ± 0.140 | - | 1.458 ± 0.087 | 1.468 ± 0.073 | 1.090 ± 0.103 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 5.323 ± 0.273 | - | 5.440 ± 0.397 | 5.850 ± 1.118 | 3.062 ± 0.549 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 4.239 ± 0.192 | - | 3.991 ± 0.292 | 3.292 ± 0.227 | 3.136 ± 0.116 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.078 ± 0.004 | - | 0.081 ± 0.002 | 0.085 ± 0.007 | 0.188 ± 0.059 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.008 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.003 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.006 ± 0.002 | - | 0.007 ± 0.000 | 0.010 ± 0.003 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.021 ± 0.005 | - | 0.018 ± 0.000 | 0.018 ± 0.000 | 0.081 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.009 ± 0.001 | - | 0.011 ± 0.001 | 0.011 ± 0.000 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.020 ± 0.000 | - | 0.022 ± 0.000 | 0.023 ± 0.000 | 0.084 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 33.856 ± 2.275 | - | 36.807 ± 3.802 | 26.851 ± 2.209 | 12.437 ± 0.928 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.001 | 0.007 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.623 ± 0.034 | - | 0.666 ± 0.056 | 0.624 ± 0.054 | 0.363 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.012 ± 0.000 | - | 0.017 ± 0.005 | 0.016 ± 0.000 | 0.028 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.050 ± 0.006 | - | 0.049 ± 0.013 | 0.043 ± 0.000 | 0.095 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.243 ± 0.032 | - | 0.292 ± 0.029 | 0.203 ± 0.003 | 0.536 ± 0.076 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.346 ± 0.544 | - | 1.205 ± 0.050 | 1.092 ± 0.074 | 1.620 ± 0.157 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 5.047 ± 0.354 | - | 4.714 ± 0.586 | 3.998 ± 0.345 | 5.260 ± 0.162 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 0.853 ± 0.006 | - | 0.897 ± 0.051 | 0.857 ± 0.028 | 1.132 ± 0.120 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.060 ± 0.003 | - | 0.067 ± 0.003 | 0.059 ± 0.001 | 0.113 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.006 ± 0.001 | - | 0.008 ± 0.002 | 0.006 ± 0.000 | 0.012 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.015 ± 0.003 | - | 0.015 ± 0.001 | 0.015 ± 0.000 | 0.023 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.006 ± 0.002 | - | 0.010 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.018 ± 0.001 | - | 0.019 ± 0.001 | 0.019 ± 0.000 | 0.033 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 8.291 ± 0.312 | - | 8.138 ± 0.735 | 6.825 ± 0.084 | 8.386 ± 0.409 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.464 ± 0.057 | - | 0.448 ± 0.003 | 0.465 ± 0.037 | 0.566 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.014 ± 0.003 | - | 0.014 ± 0.003 | 0.011 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.034 ± 0.000 | - | 0.035 ± 0.001 | 0.034 ± 0.000 | 0.101 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.236 ± 0.019 | - | 0.220 ± 0.012 | 0.249 ± 0.028 | 0.509 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.423 ± 0.371 | - | 1.223 ± 0.130 | 1.089 ± 0.009 | 0.904 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 4.563 ± 0.116 | - | 4.686 ± 0.367 | 4.644 ± 0.342 | 2.467 ± 1.166 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 0.895 ± 0.013 | - | 0.919 ± 0.024 | 0.998 ± 0.069 | 1.136 ± 0.122 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.062 ± 0.002 | - | 0.068 ± 0.003 | 0.068 ± 0.004 | 0.124 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.001 | 0.006 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.006 ± 0.001 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.003 ± 0.001 | - | 0.003 ± 0.000 | 0.004 ± 0.002 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.006 ± 0.000 | - | 0.006 ± 0.002 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.015 ± 0.003 | - | 0.019 ± 0.005 | 0.015 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.006 ± 0.002 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.008 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.022 ± 0.001 | - | 0.020 ± 0.000 | 0.019 ± 0.000 | 0.030 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 7.069 ± 0.146 | - | 8.046 ± 0.994 | 7.664 ± 1.120 | 4.080 ± 1.701 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.476 ± 0.018 | - | 0.512 ± 0.016 | 0.474 ± 0.120 | 0.279 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.013 ± 0.002 | - | 0.011 ± 0.000 | 0.015 ± 0.003 | 0.025 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.035 ± 0.003 | - | 0.035 ± 0.000 | 0.035 ± 0.000 | 0.086 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.091 ± 0.000 | - | 0.092 ± 0.000 | 0.096 ± 0.001 | 0.452 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 0.615 ± 0.007 | - | 0.685 ± 0.115 | 0.620 ± 0.012 | 1.481 ± 0.228 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 2.080 ± 0.323 | - | 1.998 ± 0.005 | 2.000 ± 0.021 | 5.332 ± 0.651 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.507 ± 0.040 | - | 0.436 ± 0.001 | 0.464 ± 0.041 | 1.076 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.021 ± 0.000 | - | 0.022 ± 0.000 | 0.024 ± 0.000 | 0.099 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.009 ± 0.000 | - | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 3.494 ± 0.073 | - | 3.494 ± 0.054 | 3.647 ± 0.075 | 8.306 ± 0.198 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.161 ± 0.048 | - | 0.171 ± 0.022 | 0.178 ± 0.019 | 0.578 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.006 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.016 ± 0.005 | - | 0.017 ± 0.000 | 0.019 ± 0.000 | 0.076 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.093 ± 0.001 | - | 0.104 ± 0.010 | 0.101 ± 0.007 | 0.483 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 0.717 ± 0.095 | - | 0.686 ± 0.079 | 0.623 ± 0.034 | 1.055 ± 0.165 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 2.213 ± 0.191 | - | 2.274 ± 0.082 | 2.228 ± 0.118 | 2.533 ± 0.084 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.521 ± 0.036 | - | 0.552 ± 0.017 | 0.493 ± 0.054 | 1.113 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.024 ± 0.002 | - | 0.025 ± 0.003 | 0.027 ± 0.001 | 0.083 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.003 ± 0.001 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.004 ± 0.001 | 0.006 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.008 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.010 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.002 ± 0.001 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.006 ± 0.002 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.010 ± 0.004 | - | 0.011 ± 0.004 | 0.012 ± 0.000 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 4.199 ± 0.548 | - | 3.880 ± 0.580 | 3.932 ± 0.490 | 4.521 ± 0.235 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.002 ± 0.001 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.183 ± 0.019 | - | 0.175 ± 0.012 | 0.202 ± 0.031 | 0.397 ± 0.063 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.008 ± 0.002 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.018 ± 0.000 | - | 0.019 ± 0.000 | 0.020 ± 0.000 | 0.075 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.077 ± 0.001 | - | 0.079 ± 0.001 | 0.142 ± 0.017 | 0.166 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.163 ± 0.010 | - | 0.163 ± 0.001 | 0.234 ± 0.007 | 0.246 ± 0.067 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.373 ± 0.003 | - | 0.371 ± 0.013 | 0.504 ± 0.051 | 0.364 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.056 ± 0.003 | - | 0.050 ± 0.001 | 0.065 ± 0.004 | 0.112 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.001 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.001 | 0.040 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.043 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.009 ± 0.003 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.050 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.001 | 0.014 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.009 ± 0.002 | 0.011 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.043 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.376 ± 0.002 | - | 0.378 ± 0.001 | 0.444 ± 0.007 | 0.487 ± 0.088 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.028 ± 0.000 | - | 0.029 ± 0.006 | 0.039 ± 0.004 | 0.080 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.002 | 0.045 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.012 ± 0.001 | 0.044 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.081 ± 0.002 | - | 0.082 ± 0.001 | 0.240 ± 0.022 | 0.136 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.170 ± 0.012 | - | 0.175 ± 0.012 | 0.274 ± 0.030 | 0.225 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.445 ± 0.044 | - | 0.403 ± 0.007 | 0.409 ± 0.106 | 0.377 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.050 ± 0.007 | - | 0.063 ± 0.005 | 0.144 ± 0.009 | 0.158 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.003 ± 0.000 | 0.065 ± 0.030 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.007 ± 0.001 | - | 0.009 ± 0.002 | 0.081 ± 0.025 | 0.047 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.076 ± 0.030 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.008 ± 0.003 | - | 0.010 ± 0.000 | 0.097 ± 0.013 | 0.052 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.059 ± 0.005 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.003 ± 0.000 | 0.065 ± 0.002 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.061 ± 0.002 | 0.021 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.004 ± 0.000 | - | 0.004 ± 0.000 | 0.060 ± 0.004 | 0.075 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.003 ± 0.000 | 0.064 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.065 ± 0.009 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.005 ± 0.000 | - | 0.005 ± 0.000 | 0.062 ± 0.002 | 0.078 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.407 ± 0.018 | - | 0.447 ± 0.012 | 0.431 ± 0.019 | 0.522 ± 0.063 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.097 ± 0.032 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.031 ± 0.004 | - | 0.035 ± 0.006 | 0.157 ± 0.011 | 0.132 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.090 ± 0.011 | 0.016 ± 0.076 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.007 ± 0.000 | - | 0.008 ± 0.003 | 0.090 ± 0.012 | 0.085 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.188 ± 0.014 | - | 0.198 ± 0.023 | 0.209 ± 0.001 | 0.343 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.503 ± 0.055 | - | 0.569 ± 0.080 | 0.465 ± 0.034 | 0.552 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.353 ± 0.079 | - | 1.297 ± 0.151 | 1.071 ± 0.019 | 1.487 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.283 ± 0.023 | - | 0.277 ± 0.018 | 0.293 ± 0.018 | 0.381 ± 0.067 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.001 | 0.003 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.011 ± 0.003 | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.050 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.003 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.014 ± 0.000 | - | 0.020 ± 0.004 | 0.014 ± 0.000 | 0.052 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.003 ± 0.001 | - | 0.004 ± 0.002 | 0.003 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.013 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.005 ± 0.001 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.016 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 2.118 ± 0.216 | - | 2.433 ± 0.111 | 2.242 ± 0.131 | 2.606 ± 0.091 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.080 ± 0.001 | - | 0.082 ± 0.000 | 0.082 ± 0.000 | 0.174 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.002 | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.014 ± 0.000 | - | 0.014 ± 0.004 | 0.011 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.177 ± 0.015 | - | 0.191 ± 0.011 | 0.263 ± 0.004 | 0.300 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.486 ± 0.060 | - | 0.493 ± 0.021 | 0.546 ± 0.030 | 0.610 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.200 ± 0.100 | - | 1.195 ± 0.239 | 1.254 ± 0.053 | 0.724 ± 0.232 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.274 ± 0.024 | - | 0.288 ± 0.059 | 0.376 ± 0.052 | 0.337 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.001 | 0.032 ± 0.005 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.015 ± 0.002 | - | 0.013 ± 0.000 | 0.066 ± 0.008 | 0.055 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.037 ± 0.006 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.013 ± 0.003 | - | 0.015 ± 0.005 | 0.060 ± 0.021 | 0.087 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.004 ± 0.000 | 0.022 ± 0.002 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.032 ± 0.007 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.039 ± 0.011 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.007 ± 0.000 | - | 0.006 ± 0.000 | 0.033 ± 0.004 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.036 ± 0.005 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.007 ± 0.003 | - | 0.005 ± 0.000 | 0.034 ± 0.003 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.007 ± 0.002 | - | 0.007 ± 0.000 | 0.036 ± 0.005 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 2.261 ± 0.182 | - | 2.203 ± 0.182 | 2.602 ± 0.056 | 1.448 ± 0.083 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.004 ± 0.000 | 0.034 ± 0.004 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.081 ± 0.001 | - | 0.085 ± 0.003 | 0.138 ± 0.017 | 0.215 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.001 | 0.038 ± 0.003 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.010 ± 0.003 | - | 0.012 ± 0.004 | 0.041 ± 0.008 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.131 ± 0.004 | - | 0.117 ± 0.009 | 0.103 ± 0.001 | 0.171 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.185 ± 0.004 | - | 0.180 ± 0.043 | 0.149 ± 0.003 | 0.211 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.376 ± 0.053 | - | 0.374 ± 0.024 | 0.313 ± 0.055 | 0.486 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.052 ± 0.006 | - | 0.061 ± 0.002 | 0.054 ± 0.001 | 0.121 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.004 ± 0.002 | 0.008 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.005 ± 0.001 | - | 0.009 ± 0.001 | 0.010 ± 0.000 | 0.039 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.002 | 0.008 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.001 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.001 | 0.009 ± 0.000 | 0.012 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.004 | 0.009 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.361 ± 0.009 | - | 0.380 ± 0.007 | 0.329 ± 0.007 | 0.555 ± 0.150 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.026 ± 0.000 | - | 0.032 ± 0.001 | 0.031 ± 0.000 | 0.093 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.007 ± 0.002 | 0.010 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.107 ± 0.012 | - | 0.110 ± 0.010 | 0.154 ± 0.010 | 0.182 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.157 ± 0.009 | - | 0.149 ± 0.025 | 0.166 ± 0.005 | 0.254 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.386 ± 0.041 | - | 0.366 ± 0.057 | 0.265 ± 0.031 | 0.526 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.053 ± 0.003 | - | 0.062 ± 0.008 | 0.063 ± 0.002 | 0.146 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.010 ± 0.005 | 0.013 ± 0.000 | 0.007 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.056 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.002 | 0.016 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.002 | 0.013 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.006 ± 0.001 | 0.013 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.002 | 0.013 ± 0.002 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.002 | 0.012 ± 0.003 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.001 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.003 ± 0.002 | - | 0.006 ± 0.000 | 0.013 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.004 ± 0.005 | - | 0.009 ± 0.002 | 0.013 ± 0.003 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.401 ± 0.003 | - | 0.440 ± 0.118 | 0.275 ± 0.023 | 0.501 ± 0.090 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.004 | 0.058 ± 0.013 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.027 ± 0.008 | - | 0.032 ± 0.000 | 0.080 ± 0.016 | 0.154 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.004 ± 0.003 | - | 0.007 ± 0.003 | 0.063 ± 0.010 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.055 ± 0.008 | 0.019 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.153 ± 0.136 | - | 1.210 ± 0.057 | 1.094 ± 0.062 | 2.946 ± 0.099 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 4.426 ± 0.557 | - | 3.675 ± 0.218 | 3.631 ± 0.015 | 5.523 ± 0.546 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 11.581 ± 1.565 | - | 14.292 ± 0.588 | 11.416 ± 0.143 | 15.278 ± 0.420 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 3.773 ± 0.014 | - | 3.877 ± 0.015 | 3.634 ± 0.024 | 3.680 ± 0.324 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.176 ± 0.010 | - | 0.184 ± 0.006 | 0.174 ± 0.010 | 0.286 ± 0.075 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.007 ± 0.001 | - | 0.009 ± 0.001 | 0.010 ± 0.002 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.185 ± 0.014 | - | 0.186 ± 0.004 | 0.154 ± 0.002 | 0.224 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.013 ± 0.000 | - | 0.014 ± 0.000 | 0.016 ± 0.000 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.013 ± 0.000 | - | 0.015 ± 0.005 | 0.015 ± 0.000 | 0.029 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.036 ± 0.000 | - | 0.037 ± 0.000 | 0.039 ± 0.000 | 0.052 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.018 ± 0.001 | - | 0.018 ± 0.008 | 0.022 ± 0.004 | 0.032 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.048 ± 0.000 | - | 0.049 ± 0.000 | 0.051 ± 0.000 | 0.070 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 32.164 ± 3.332 | - | 33.191 ± 4.292 | 32.112 ± 5.385 | 27.409 ± 3.432 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.011 ± 0.000 | - | 0.012 ± 0.000 | 0.013 ± 0.000 | 0.035 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.457 ± 0.022 | - | 1.364 ± 0.114 | 1.317 ± 0.021 | 1.767 ± 0.116 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.040 ± 0.007 | - | 0.037 ± 0.008 | 0.033 ± 0.000 | 0.058 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.092 ± 0.000 | - | 0.093 ± 0.000 | 0.095 ± 0.000 | 0.168 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.118 ± 0.151 | - | 1.115 ± 0.014 | 1.403 ± 0.176 | 2.806 ± 0.281 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 3.882 ± 0.052 | - | 3.953 ± 0.442 | 4.065 ± 0.620 | 2.121 ± 0.359 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 12.783 ± 0.478 | - | 13.827 ± 0.948 | 12.470 ± 2.536 | 7.318 ± 1.434 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 4.239 ± 0.213 | - | 4.382 ± 0.324 | 3.902 ± 0.290 | 3.469 ± 0.140 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.017 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.184 ± 0.045 | - | 0.191 ± 0.018 | 0.206 ± 0.011 | 0.263 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.008 ± 0.001 | - | 0.010 ± 0.002 | 0.011 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.190 ± 0.010 | - | 0.181 ± 0.013 | 0.171 ± 0.012 | 0.226 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.015 ± 0.004 | - | 0.016 ± 0.005 | 0.020 ± 0.005 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.007 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.002 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.014 ± 0.004 | - | 0.015 ± 0.004 | 0.021 ± 0.004 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.040 ± 0.004 | - | 0.040 ± 0.000 | 0.054 ± 0.013 | 0.051 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.009 ± 0.002 | - | 0.008 ± 0.000 | 0.013 ± 0.001 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.017 ± 0.004 | - | 0.019 ± 0.001 | 0.020 ± 0.000 | 0.032 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.055 ± 0.005 | - | 0.057 ± 0.014 | 0.058 ± 0.012 | 0.068 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 34.600 ± 4.118 | - | 35.657 ± 1.313 | 31.399 ± 2.369 | 13.900 ± 0.240 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.011 ± 0.002 | - | 0.012 ± 0.001 | 0.018 ± 0.002 | 0.032 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 1.309 ± 0.047 | - | 1.477 ± 0.073 | 1.540 ± 0.033 | 0.814 ± 0.209 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.030 ± 0.000 | - | 0.032 ± 0.002 | 0.035 ± 0.001 | 0.054 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.103 ± 0.005 | - | 0.113 ± 0.010 | 0.119 ± 0.009 | 0.179 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.479 ± 0.001 | - | 0.479 ± 0.021 | 0.482 ± 0.011 | 1.951 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 2.315 ± 0.204 | - | 2.296 ± 0.019 | 2.290 ± 0.018 | 3.760 ± 0.128 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 7.854 ± 0.089 | - | 8.864 ± 1.247 | 9.406 ± 1.339 | 11.089 ± 0.569 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.557 ± 0.077 | - | 1.626 ± 0.128 | 1.340 ± 0.018 | 1.703 ± 0.141 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.114 ± 0.011 | - | 0.116 ± 0.006 | 0.096 ± 0.001 | 0.158 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.100 ± 0.005 | - | 0.104 ± 0.008 | 0.089 ± 0.003 | 0.121 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.011 ± 0.003 | - | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.026 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.022 ± 0.000 | - | 0.023 ± 0.000 | 0.025 ± 0.000 | 0.037 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.018 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.011 ± 0.000 | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.031 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.031 ± 0.000 | - | 0.032 ± 0.000 | 0.034 ± 0.000 | 0.050 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 12.303 ± 1.276 | - | 11.917 ± 1.531 | 10.680 ± 0.647 | 12.944 ± 0.159 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 0.880 ± 0.031 | - | 0.852 ± 0.059 | 0.733 ± 0.008 | 0.927 ± 0.158 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.020 ± 0.000 | - | 0.020 ± 0.000 | 0.022 ± 0.000 | 0.042 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.060 ± 0.001 | - | 0.061 ± 0.003 | 0.063 ± 0.000 | 0.141 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.506 ± 0.001 | - | 0.553 ± 0.048 | 0.573 ± 0.087 | 2.069 ± 0.112 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 2.465 ± 0.135 | - | 2.769 ± 0.029 | 2.348 ± 0.067 | 2.076 ± 0.261 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 8.878 ± 0.883 | - | 8.654 ± 0.129 | 8.334 ± 0.576 | 5.269 ± 0.921 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 1.452 ± 0.101 | - | 1.553 ± 0.052 | 1.624 ± 0.127 | 1.571 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.007 ± 0.001 | 0.014 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.103 ± 0.009 | - | 0.115 ± 0.012 | 0.128 ± 0.047 | 0.165 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.006 ± 0.001 | - | 0.008 ± 0.000 | 0.010 ± 0.002 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.092 ± 0.003 | - | 0.092 ± 0.009 | 0.107 ± 0.004 | 0.109 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.012 ± 0.004 | - | 0.015 ± 0.001 | 0.012 ± 0.000 | 0.022 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.009 ± 0.002 | - | 0.013 ± 0.006 | 0.017 ± 0.001 | 0.027 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.032 ± 0.001 | - | 0.029 ± 0.008 | 0.026 ± 0.000 | 0.040 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.006 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.015 ± 0.002 | - | 0.017 ± 0.005 | 0.015 ± 0.004 | 0.026 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.041 ± 0.009 | - | 0.039 ± 0.017 | 0.034 ± 0.001 | 0.051 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 12.141 ± 0.402 | - | 11.801 ± 0.210 | 11.888 ± 1.181 | 6.720 ± 0.079 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.006 ± 0.003 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.032 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 0.801 ± 0.041 | - | 0.811 ± 0.050 | 0.841 ± 0.121 | 0.559 ± 0.058 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.021 ± 0.000 | - | 0.029 ± 0.005 | 0.025 ± 0.006 | 0.041 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.069 ± 0.018 | - | 0.076 ± 0.013 | 0.063 ± 0.000 | 0.157 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.224 ± 0.024 | - | unsupported | 0.189 ± 0.000 | 1.371 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.503 ± 0.179 | - | unsupported | 1.186 ± 0.043 | 2.399 ± 0.071 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 4.248 ± 0.047 | - | unsupported | 4.048 ± 0.183 | 5.175 ± 0.298 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.959 ± 0.026 | - | unsupported | 0.922 ± 0.048 | 0.920 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | unsupported | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.045 ± 0.004 | - | unsupported | 0.047 ± 0.003 | 0.044 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.002 ± 0.001 | - | unsupported | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.026 ± 0.000 | - | unsupported | 0.034 ± 0.001 | 0.034 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.006 ± 0.001 | - | unsupported | 0.007 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.011 ± 0.000 | - | unsupported | 0.013 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | unsupported | 0.004 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.008 ± 0.001 | - | unsupported | 0.008 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.015 ± 0.000 | - | unsupported | 0.017 ± 0.000 | 0.027 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 7.689 ± 0.746 | - | unsupported | 7.688 ± 0.362 | 6.922 ± 0.173 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.019 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.296 ± 0.035 | - | unsupported | 0.313 ± 0.048 | 0.269 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.012 ± 0.004 | - | unsupported | 0.013 ± 0.000 | 0.031 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.029 ± 0.004 | - | unsupported | 0.029 ± 0.000 | 0.049 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.209 ± 0.022 | - | unsupported | 0.252 ± 0.011 | 1.234 ± 0.060 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 1.240 ± 0.242 | - | unsupported | 1.180 ± 0.077 | 0.804 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 3.316 ± 0.065 | - | unsupported | 3.904 ± 0.395 | 2.590 ± 0.142 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 0.812 ± 0.008 | - | unsupported | 0.949 ± 0.029 | 0.922 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.032 ± 0.001 | - | unsupported | 0.044 ± 0.012 | 0.047 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | unsupported | 0.008 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.032 ± 0.005 | - | unsupported | 0.033 ± 0.005 | 0.037 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.006 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.007 ± 0.001 | - | unsupported | 0.008 ± 0.003 | 0.016 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.011 ± 0.000 | - | unsupported | 0.013 ± 0.000 | 0.033 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.006 ± 0.002 | - | unsupported | 0.011 ± 0.003 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.014 ± 0.000 | - | unsupported | 0.017 ± 0.000 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 6.498 ± 0.157 | - | unsupported | 7.016 ± 0.927 | 3.525 ± 1.622 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.022 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.247 ± 0.003 | - | unsupported | 0.339 ± 0.008 | 0.243 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.015 ± 0.001 | - | unsupported | 0.017 ± 0.004 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.036 ± 0.023 | - | unsupported | 0.037 ± 0.011 | 0.046 ± 0.002 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.817 ± 0.092 | - | unsupported | 0.671 ± 0.074 | 0.906 ± 0.093 | 2.686 ± 0.290 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.053 ± 0.009 | - | unsupported | 0.060 ± 0.005 | 0.142 ± 0.030 | 0.073 ± 0.013 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.104 ± 0.013 | - | unsupported | 0.130 ± 0.027 | 0.259 ± 0.046 | 0.150 ± 0.015 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.247 ± 0.018 | - | unsupported | 0.231 ± 0.069 | 0.394 ± 0.029 | 0.629 ± 0.114 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.930 ± 0.355 | - | unsupported | 0.367 ± 0.027 | 0.593 ± 0.053 | 2.494 ± 0.464 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.064 ± 0.007 | - | unsupported | 0.082 ± 0.009 | 0.148 ± 0.028 | 0.098 ± 0.026 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.118 ± 0.013 | - | unsupported | 0.098 ± 0.005 | 0.217 ± 0.026 | 0.166 ± 0.028 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.319 ± 0.044 | - | unsupported | 0.216 ± 0.036 | 0.269 ± 0.024 | 0.562 ± 0.044 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.242 ± 0.023 | - | unsupported | 0.167 ± 0.021 | 0.428 ± 0.034 | 0.414 ± 0.029 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.068 ± 0.003 | - | unsupported | 0.050 ± 0.014 | 0.111 ± 0.013 | 0.072 ± 0.010 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.056 ± 0.000 | - | unsupported | 0.051 ± 0.009 | 0.131 ± 0.023 | 0.117 ± 0.008 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.096 ± 0.004 | - | unsupported | 0.071 ± 0.004 | 0.174 ± 0.015 | 0.210 ± 0.023 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.209 ± 0.047 | - | unsupported | 0.199 ± 0.039 | 0.965 ± 0.387 | 0.359 ± 0.025 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.078 ± 0.012 | - | unsupported | 0.067 ± 0.008 | 0.112 ± 0.015 | 0.065 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.113 ± 0.083 | - | unsupported | 0.054 ± 0.012 | 0.131 ± 0.017 | 0.103 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.212 ± 0.131 | - | unsupported | 0.083 ± 0.014 | 0.188 ± 0.023 | 0.173 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.103 ± 0.006 | - | 0.124 ± 0.004 | 0.065 ± 0.000 | 0.172 ± 0.028 | 0.162 ± 0.023 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.048 ± 0.009 | - | 0.051 ± 0.003 | 0.021 ± 0.002 | 0.024 ± 0.002 | 0.033 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.045 ± 0.003 | - | 0.047 ± 0.001 | 0.023 ± 0.001 | 0.027 ± 0.002 | 0.051 ± 0.014 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.056 ± 0.000 | - | 0.059 ± 0.008 | 0.032 ± 0.009 | 0.067 ± 0.045 | 0.089 ± 0.013 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.122 ± 0.010 | - | 0.099 ± 0.011 | 0.089 ± 0.005 | 0.236 ± 0.016 | 0.152 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.048 ± 0.007 | - | 0.060 ± 0.010 | 0.022 ± 0.006 | 0.024 ± 0.009 | 0.038 ± 0.012 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.049 ± 0.013 | - | 0.108 ± 0.002 | 0.031 ± 0.001 | 0.027 ± 0.001 | 0.047 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.066 ± 0.012 | - | 0.068 ± 0.001 | 0.034 ± 0.011 | 0.072 ± 0.014 | 0.096 ± 0.014 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 6.280 ± 0.419 | - | 6.862 ± 0.807 | 17.284 ± 1.273 | 14.865 ± 4.554 | 30.397 ± 6.240 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 5.724 ± 0.245 | - | 5.617 ± 0.166 | 11.279 ± 1.825 | 11.925 ± 3.728 | 17.861 ± 3.466 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 5.487 ± 0.377 | - | 6.415 ± 0.439 | 6.052 ± 0.330 | 5.069 ± 0.151 | 17.320 ± 4.072 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 6.182 ± 0.677 | - | 5.502 ± 0.437 | 5.445 ± 0.607 | 5.003 ± 0.775 | 7.856 ± 1.796 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 6.715 ± 0.243 | - | 6.537 ± 0.360 | 6.930 ± 0.980 | 6.764 ± 0.083 | 5.336 ± 0.017 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 5.527 ± 0.136 | - | 6.125 ± 0.705 | 5.743 ± 0.965 | 7.622 ± 0.254 | 6.862 ± 0.513 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 7.461 ± 0.341 | - | 7.706 ± 0.284 | 7.397 ± 0.539 | 7.037 ± 0.338 | 7.404 ± 0.386 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 7.075 ± 0.314 | - | 6.466 ± 0.093 | 6.739 ± 0.390 | 6.664 ± 0.279 | 9.162 ± 2.726 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 15.330 ± 3.077 | - | 18.051 ± 0.618 | 17.311 ± 1.550 | 19.176 ± 0.054 | 11.347 ± 2.388 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 17.974 ± 0.872 | - | 15.754 ± 1.070 | 17.230 ± 1.917 | 22.220 ± 1.213 | 15.838 ± 5.757 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.229 ± 0.005 | 0.241 ± 0.027 | unsupported | 0.284 ± 0.013 | 0.317 ± 0.025 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 2.987 ± 0.239 | 2.963 ± 0.117 | unsupported | 3.400 ± 0.213 | 3.453 ± 0.312 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.233 ± 0.008 | 0.237 ± 0.006 | unsupported | 0.263 ± 0.030 | 0.363 ± 0.031 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 3.030 ± 0.131 | 3.199 ± 0.044 | unsupported | 3.161 ± 0.118 | 3.541 ± 0.177 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 7.242 ± 0.284 | - | 6.790 ± 0.429 | 6.641 ± 0.577 | 7.008 ± 0.709 | 27.195 ± 1.078 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 6.859 ± 0.497 | - | 7.329 ± 0.858 | 7.372 ± 1.339 | 7.157 ± 0.362 | 8.685 ± 3.793 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 7.459 ± 0.404 | 6.556 ± 0.331 | 4.681 ± 0.608 | 18.199 ± 0.109 | 6.824 ± 0.912 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 8.021 ± 0.538 | 7.549 ± 0.253 | 4.960 ± 0.443 | 25.997 ± 1.858 | 5.178 ± 0.538 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 5.995 ± 1.368 | - | 5.737 ± 1.039 | 10.524 ± 0.689 | 8.118 ± 1.498 | 21.686 ± 7.528 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 7.898 ± 0.667 | - | 6.327 ± 0.681 | 8.438 ± 0.405 | 8.113 ± 0.777 | 16.360 ± 3.437 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.550 ± 0.119 | - | 0.598 ± 0.019 | 1.822 ± 0.011 | 5.961 ± 0.253 | 5.778 ± 0.208 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.467 ± 0.031 | - | 0.228 ± 0.051 | 2.717 ± 0.257 | 2.073 ± 0.204 | 5.277 ± 0.048 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 9.595 ± 0.471 | - | 8.929 ± 1.066 | 7.641 ± 0.059 | 9.507 ± 1.630 | 19.387 ± 0.787 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 9.376 ± 0.362 | - | 9.007 ± 0.547 | 9.633 ± 0.737 | 10.392 ± 0.558 | 17.154 ± 3.183 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 8.273 ± 0.361 | - | 8.792 ± 1.316 | 8.986 ± 0.912 | 9.436 ± 0.107 | 19.017 ± 1.935 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 9.237 ± 0.522 | - | 8.645 ± 0.188 | 10.352 ± 0.692 | 9.623 ± 0.242 | 21.220 ± 6.359 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.271 ± 0.136 | 1.304 ± 0.155 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 10.101 ± 0.764 | 10.985 ± 1.071 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.333 ± 0.098 | 1.233 ± 0.009 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 11.425 ± 0.925 | 10.814 ± 0.560 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.353 ± 0.057 | 1.315 ± 0.123 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 9.239 ± 0.508 | 9.159 ± 0.950 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.328 ± 0.084 | 1.344 ± 0.103 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 9.996 ± 0.471 | 10.257 ± 0.853 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 6.688 ± 0.688 | - | 6.396 ± 0.320 | 6.003 ± 0.337 | 5.813 ± 0.873 | 16.693 ± 3.356 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 5.679 ± 0.198 | - | 6.309 ± 0.534 | 5.336 ± 0.346 | 4.959 ± 0.241 | 6.721 ± 0.586 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 27.064 ± 3.776 | 25.261 ± 1.423 | 23.361 ± 0.075 | 23.748 ± 1.920 | 66.848 ± 4.766 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 27.375 ± 2.718 | 26.812 ± 4.420 | 31.348 ± 3.754 | 30.982 ± 1.424 | 45.475 ± 7.337 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 11.961 ± 1.572 | - | 10.881 ± 0.085 | 9.713 ± 0.836 | 53.513 ± 6.463 | 23.707 ± 0.151 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 11.294 ± 0.751 | - | 11.397 ± 1.108 | 9.136 ± 1.421 | 27.216 ± 0.688 | 15.052 ± 4.099 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.149 ± 0.014 | 0.151 ± 0.013 | unsupported | 0.052 ± 0.011 | 0.028 ± 0.001 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.492 ± 0.360 | 2.558 ± 0.282 | unsupported | 0.922 ± 0.033 | 0.423 ± 0.095 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.165 ± 0.007 | 0.171 ± 0.007 | unsupported | 0.078 ± 0.008 | 0.058 ± 0.040 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.657 ± 0.037 | 0.656 ± 0.046 | unsupported | 0.239 ± 0.025 | 0.245 ± 0.179 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.085 ± 0.009 | 0.083 ± 0.005 | unsupported | 0.086 ± 0.017 | 0.178 ± 0.027 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.247 ± 0.094 | 1.189 ± 0.170 | unsupported | 1.543 ± 0.060 | 2.131 ± 0.140 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.023 ± 0.007 | 0.021 ± 0.004 | unsupported | 0.070 ± 0.004 | 0.132 ± 0.011 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.302 ± 0.022 | 0.300 ± 0.010 | unsupported | 0.381 ± 0.028 | 0.661 ± 0.019 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.286 ± 0.017 | 0.233 ± 0.068 | unsupported | 0.269 ± 0.026 | 0.084 ± 0.005 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 4.071 ± 0.933 | 3.716 ± 0.927 | unsupported | 4.792 ± 0.395 | 0.915 ± 0.060 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.070 ± 0.006 | 0.067 ± 0.055 | unsupported | 0.182 ± 0.076 | 0.060 ± 0.016 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 1.108 ± 1.059 | 1.017 ± 0.106 | unsupported | 1.137 ± 0.060 | 0.384 ± 0.039 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.049 ± 0.002 | 0.063 ± 0.006 | unsupported | 0.039 ± 0.000 | 0.071 ± 0.011 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.483 ± 0.019 | 0.510 ± 0.030 | unsupported | 0.442 ± 0.015 | 0.492 ± 0.017 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.052 ± 0.003 | 0.068 ± 0.005 | unsupported | 0.058 ± 0.009 | 0.075 ± 0.004 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.306 ± 0.088 | 0.377 ± 0.086 | unsupported | 0.170 ± 0.016 | 0.417 ± 0.042 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.098 ± 0.007 | 0.088 ± 0.011 | unsupported | 0.077 ± 0.006 | 0.109 ± 0.005 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.171 ± 0.310 | 0.979 ± 0.046 | unsupported | 1.282 ± 0.251 | 1.246 ± 0.030 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.090 ± 0.002 | 0.105 ± 0.008 | unsupported | 0.071 ± 0.005 | 0.106 ± 0.016 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.516 ± 0.110 | 0.560 ± 0.065 | unsupported | 0.365 ± 0.022 | 0.601 ± 0.087 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.126 ± 0.020 | 0.154 ± 0.011 | unsupported | unsupported | 0.136 ± 0.022 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 2.042 ± 0.411 | 1.824 ± 0.138 | unsupported | unsupported | 1.560 ± 0.091 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.126 ± 0.009 | 0.148 ± 0.020 | unsupported | unsupported | 0.131 ± 0.009 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 0.694 ± 0.134 | 0.631 ± 0.054 | unsupported | unsupported | 1.329 ± 0.356 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.010 ± 0.000 | 0.014 ± 0.002 | unsupported | 0.008 ± 0.001 | 0.044 ± 0.008 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.086 ± 0.001 | 0.090 ± 0.002 | unsupported | 0.061 ± 0.008 | 0.222 ± 0.010 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.011 ± 0.000 | 0.016 ± 0.003 | unsupported | 0.049 ± 0.012 | 0.066 ± 0.023 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.066 ± 0.008 | 0.083 ± 0.028 | unsupported | 0.070 ± 0.004 | 0.128 ± 0.048 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.025 ± 0.001 | 0.037 ± 0.004 | unsupported | 0.021 ± 0.001 | 0.068 ± 0.005 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.242 ± 0.058 | 0.288 ± 0.054 | unsupported | 0.242 ± 0.023 | 0.447 ± 0.015 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.026 ± 0.006 | 0.039 ± 0.015 | unsupported | 0.134 ± 0.046 | 0.088 ± 0.006 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.189 ± 0.064 | 0.190 ± 0.053 | unsupported | 0.314 ± 0.058 | 0.337 ± 0.085 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.041 ± 0.003 | 0.041 ± 0.001 | unsupported | 0.085 ± 0.002 | 0.072 ± 0.004 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.715 ± 0.191 | 0.901 ± 0.084 | unsupported | 0.996 ± 0.015 | 0.734 ± 0.013 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.012 ± 0.000 | 0.014 ± 0.004 | unsupported | 0.077 ± 0.011 | 0.056 ± 0.011 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.262 ± 0.288 | 0.201 ± 0.033 | unsupported | 0.312 ± 0.008 | 0.341 ± 0.028 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.045 ± 0.003 | 0.052 ± 0.007 | unsupported | 0.073 ± 0.002 | 0.090 ± 0.002 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.773 ± 0.107 | 0.881 ± 0.083 | unsupported | 1.279 ± 0.040 | 0.875 ± 0.189 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.016 ± 0.002 | 0.018 ± 0.065 | unsupported | 0.079 ± 0.026 | 0.069 ± 0.008 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.192 ± 0.060 | 0.236 ± 0.112 | unsupported | 0.312 ± 0.009 | 0.344 ± 0.010 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.087 ± 0.015 | 0.108 ± 0.013 | unsupported | 0.095 ± 0.012 | 0.110 ± 0.023 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 0.959 ± 0.008 | 0.974 ± 0.003 | unsupported | 1.392 ± 0.125 | 1.149 ± 0.136 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.088 ± 0.000 | 0.100 ± 0.009 | unsupported | 0.070 ± 0.008 | 0.123 ± 0.008 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 0.610 ± 0.016 | 0.548 ± 0.141 | unsupported | 0.453 ± 0.150 | 0.546 ± 0.028 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.215 ± 0.014 | 0.187 ± 0.035 | unsupported | 0.271 ± 0.016 | 0.296 ± 0.022 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 3.816 ± 0.103 | 3.115 ± 0.228 | unsupported | 4.534 ± 0.307 | 3.583 ± 0.626 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.056 ± 0.006 | 0.072 ± 0.061 | unsupported | 0.167 ± 0.022 | 0.178 ± 0.013 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 0.982 ± 0.079 | 0.828 ± 0.051 | unsupported | 2.025 ± 0.385 | 1.500 ± 0.132 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.510 ± 0.105 | 0.710 ± 0.027 | unsupported | 0.103 ± 0.020 | 0.098 ± 0.014 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 8.858 ± 1.101 | 9.178 ± 0.756 | unsupported | 1.518 ± 0.164 | 0.842 ± 0.143 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.479 ± 0.061 | 0.602 ± 0.065 | unsupported | 0.127 ± 0.032 | 0.090 ± 0.005 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 4.892 ± 1.367 | 4.067 ± 0.201 | unsupported | 0.517 ± 0.029 | 0.370 ± 0.029 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 14.507 ± 1.891 | - | unsupported | 15.129 ± 2.534 | - | 13.447 ± 1.568 | 14.915 ± 1.003 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 8.256 ± 1.788 | - | unsupported | 7.103 ± 1.127 | - | 11.203 ± 0.907 | 8.136 ± 2.563 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 8.852 ± 0.783 | - | unsupported | 10.692 ± 0.651 | - | 7.521 ± 0.722 | 7.597 ± 0.253 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 4.637 ± 1.713 | - | unsupported | 4.345 ± 1.048 | - | 8.823 ± 0.308 | 4.765 ± 0.424 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 9.201 ± 0.833 | - | unsupported | 10.988 ± 0.837 | - | 5.805 ± 0.716 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 4.363 ± 0.852 | - | unsupported | 4.419 ± 0.262 | - | 5.742 ± 0.187 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 14.447 ± 3.112 | - | unsupported | 11.684 ± 2.368 | - | 12.098 ± 1.689 | 14.599 ± 0.604 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 9.762 ± 0.800 | - | unsupported | 6.321 ± 0.584 | - | 11.099 ± 0.522 | 7.147 ± 0.352 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 8.404 ± 0.594 | - | unsupported | 7.640 ± 0.425 | - | 48.399 ± 2.518 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 7.027 ± 0.588 | - | unsupported | 7.133 ± 0.393 | - | 14.721 ± 5.143 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 8.703 ± 0.936 | - | unsupported | 6.667 ± 0.266 | - | 47.273 ± 1.672 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 7.254 ± 0.796 | - | unsupported | 7.777 ± 0.598 | - | 11.504 ± 0.730 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 14.336 ± 4.629 | - | unsupported | 11.543 ± 1.085 | - | 13.296 ± 0.487 | 13.281 ± 0.632 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 5.297 ± 0.850 | - | unsupported | 6.988 ± 1.289 | - | 12.198 ± 0.931 | 6.966 ± 1.696 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 6.709 ± 0.107 | - | unsupported | 7.718 ± 0.576 | - | 7.459 ± 0.873 | 12.682 ± 1.404 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 5.728 ± 2.117 | - | unsupported | 4.064 ± 0.714 | - | 8.583 ± 0.694 | 3.804 ± 0.335 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.191 ± 0.039 | unsupported | unsupported | 0.225 ± 0.004 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 2.315 ± 0.254 | unsupported | unsupported | 2.538 ± 0.048 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.173 ± 0.005 | unsupported | unsupported | 0.265 ± 0.029 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 2.180 ± 0.154 | unsupported | unsupported | 2.389 ± 0.241 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 9.985 ± 0.208 | - | unsupported | 10.000 ± 2.440 | - | 11.145 ± 1.816 | 13.939 ± 1.098 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 5.402 ± 1.184 | - | unsupported | 6.493 ± 1.780 | - | 13.354 ± 1.060 | 6.842 ± 1.454 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 5.667 ± 0.896 | - | 0.001 ± 0.000 | 5.523 ± 0.448 | 4.222 ± 0.054 | 11.388 ± 2.960 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 3.640 ± 1.353 | - | 0.003 ± 0.000 | 2.242 ± 0.021 | 4.483 ± 0.840 | 12.348 ± 1.763 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 6.467 ± 0.238 | - | 20.323 ± 1.032 | 7.979 ± 0.190 | 6.483 ± 0.148 | 6.730 ± 0.532 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 4.029 ± 0.702 | - | 8.119 ± 0.669 | 2.989 ± 0.050 | 6.142 ± 0.555 | 6.846 ± 0.594 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 11.739 ± 0.362 | - | 11.670 ± 0.878 | 8.847 ± 1.586 | 17.545 ± 0.836 | 8.882 ± 0.464 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 8.511 ± 2.089 | - | 7.030 ± 0.773 | 8.538 ± 2.373 | 8.193 ± 2.933 | 11.099 ± 0.635 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 4.459 ± 0.151 | - | 4.610 ± 0.110 | 7.527 ± 0.058 | 6.500 ± 0.199 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 2.643 ± 0.333 | - | 2.037 ± 0.506 | 5.174 ± 1.102 | 5.537 ± 1.369 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 5.959 ± 0.298 | - | 0.002 ± 0.000 | 7.663 ± 0.940 | 7.100 ± 1.983 | 5.940 ± 0.725 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 4.597 ± 0.693 | - | 0.001 ± 0.000 | 3.339 ± 0.170 | 4.159 ± 0.621 | 7.299 ± 0.358 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 39.162 ± 2.327 | - | 0.001 ± 0.000 | 48.400 ± 1.356 | 47.193 ± 7.490 | 78.731 ± 5.547 | 27.619 ± 7.545 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 11.952 ± 0.799 | - | 0.001 ± 0.000 | 44.008 ± 0.604 | 17.173 ± 1.036 | 90.627 ± 5.199 | 11.592 ± 0.704 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 7.940 ± 0.569 | - | 7.252 ± 0.829 | 6.977 ± 0.682 | 4.317 ± 0.120 | 5.527 ± 0.493 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 2.328 ± 0.084 | - | 2.453 ± 0.260 | 3.143 ± 0.559 | 2.625 ± 0.429 | 5.504 ± 0.275 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 6.390 ± 0.369 | - | 6.078 ± 0.265 | 7.492 ± 0.306 | 4.654 ± 0.860 | 5.713 ± 1.906 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 2.403 ± 0.022 | - | 2.429 ± 0.393 | 2.571 ± 0.257 | 2.562 ± 0.354 | 5.719 ± 0.290 | - |
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
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 3.250959 | 203184.94 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 4.521375 | 282585.94 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 4.203208 | 262700.50 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 3.299208 | 206200.50 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 4.463333 | 278958.31 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 5.030250 | 314390.62 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 13.652750 | 853296.88 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 13.651667 | 853229.19 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 7.479750 | 467484.38 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 7.138041 | 446127.56 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 5.066458 | 316653.62 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 4.536041 | 283502.56 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 3.660250 | 228765.62 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 3.856333 | 241020.81 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 55.244959 | 3452809.94 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 54.396375 | 3399773.44 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 47.799750 | 2987484.38 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 47.400959 | 2962559.94 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 5.804292 | 362768.25 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 4.203792 | 262737.00 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 3.727000 | 232937.50 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 3.785458 | 236591.12 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 56.662291 | 3541393.19 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 50.570375 | 3160648.44 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 48.481875 | 3030117.19 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 51.187792 | 3199237.00 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 20.330167 | 1270635.44 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 20.862958 | 1303934.88 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 161.621875 | 10101367.19 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 175.760542 | 10985033.88 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 21.331083 | 1333192.69 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 19.728209 | 1233013.06 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 182.794250 | 11424640.62 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 173.022292 | 10813893.25 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 21.642333 | 1352645.81 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 21.044375 | 1315273.44 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 147.822875 | 9238929.69 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 146.536708 | 9158544.25 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 21.249583 | 1328098.94 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 21.511166 | 1344447.88 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 159.938333 | 9996145.81 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 164.112042 | 10257002.62 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 1.817125 | 28392.58 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.313084 | 51766.94 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 9.538875 | 149044.92 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.638000 | 150593.75 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 3.383958 | 422994.75 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.373375 | 921671.88 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 19.937500 | 2492187.50 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 20.463250 | 2557906.25 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 3.705917 | 57904.95 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.970250 | 77660.16 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 10.566708 | 165104.81 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 10.955625 | 171181.64 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.962208 | 245276.00 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.914958 | 239369.75 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.255708 | 656963.50 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.251875 | 656484.38 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.405333 | 178208.33 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.499750 | 85933.59 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.422416 | 84725.25 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.335959 | 83374.36 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 17.048459 | 2131057.38 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 12.346833 | 1543354.12 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.976917 | 1247114.62 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 9.514958 | 1189369.75 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.440375 | 131880.86 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.479625 | 69994.14 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.495167 | 23361.98 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.373292 | 21457.69 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.286875 | 660859.38 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.045084 | 380635.50 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.412458 | 301557.25 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.399500 | 299937.50 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.361500 | 83773.44 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.226458 | 269163.41 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 18.293375 | 285833.98 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 14.934334 | 233348.97 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.319750 | 914968.75 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 38.332708 | 4791588.50 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 32.567500 | 4070937.50 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 29.726625 | 3715828.12 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 3.824708 | 59761.06 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 11.660666 | 182197.91 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.476000 | 69937.50 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.266250 | 66660.16 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.073833 | 384229.12 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 9.095625 | 1136953.12 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.860500 | 1107562.50 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 8.133250 | 1016656.25 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.555125 | 71173.83 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 2.507375 | 39177.73 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.114375 | 48662.11 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.049208 | 63268.88 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 3.939000 | 492375.00 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 3.533625 | 441703.12 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.866250 | 483281.25 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.080917 | 510114.63 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.781792 | 74715.50 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.706000 | 57906.25 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.336459 | 52132.17 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.335375 | 67740.23 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.340000 | 417500.00 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.362458 | 170307.25 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.444750 | 305593.75 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.016583 | 377072.88 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.992875 | 109263.67 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.901542 | 76586.59 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.278584 | 98102.88 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.634125 | 88033.20 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 9.966875 | 1245859.38 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.255500 | 1281937.50 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.368833 | 1171104.12 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.828458 | 978557.25 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.754500 | 105539.06 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.571375 | 71427.73 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.760750 | 90011.72 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6.707625 | 104806.64 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.809250 | 601156.25 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.919375 | 364921.88 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.125833 | 515729.12 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.481625 | 560203.12 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.726709 | 136354.83 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.089333 | 126395.83 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.867542 | 154180.34 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.483042 | 1560380.25 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 16.339875 | 2042484.38 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 14.591583 | 1823947.88 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.402333 | 131286.45 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.056875 | 125888.67 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 9.489334 | 148270.84 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.628875 | 1328609.38 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.552125 | 694015.62 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.044792 | 630599.00 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.834291 | 44285.80 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.530250 | 8285.16 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.615208 | 9612.62 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 0.922333 | 14411.45 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 1.776875 | 222109.38 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.490083 | 61260.38 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.688333 | 86041.62 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.718667 | 89833.38 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.255333 | 66489.58 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.124625 | 48822.27 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.710084 | 11095.06 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 0.999833 | 15622.39 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.026125 | 128265.62 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.559791 | 69973.88 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.527500 | 65937.50 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 0.662084 | 82760.50 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.344375 | 67880.86 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 1.339834 | 20934.91 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.592416 | 24881.50 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.348583 | 36696.61 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 3.576791 | 447098.88 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 1.939166 | 242395.75 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 1.939458 | 242432.25 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 2.306250 | 288281.25 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.637208 | 88081.38 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 8.560500 | 133757.81 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.654958 | 25858.72 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.507333 | 39177.08 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.698750 | 337343.75 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.514166 | 314270.75 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.515584 | 189448.00 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.521917 | 190239.62 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.616917 | 72139.33 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.410250 | 84535.16 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.623666 | 40994.78 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.645417 | 41334.64 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.868166 | 733520.75 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.969666 | 996208.25 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.718750 | 714843.75 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.211083 | 901385.38 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 3.571375 | 55802.73 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.916917 | 76826.83 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.796000 | 12437.50 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 0.904750 | 14136.72 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.729083 | 341135.38 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.498666 | 312333.25 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.094084 | 261760.50 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.605917 | 200739.62 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.754417 | 89912.77 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.660209 | 72815.77 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.874959 | 44921.23 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.323500 | 51929.69 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.000208 | 875026.00 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.233917 | 1279239.62 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.184000 | 773000.00 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.050583 | 881322.88 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.410291 | 68910.80 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.051750 | 78933.59 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.017292 | 15895.19 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.133750 | 17714.84 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.751334 | 343916.75 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.496333 | 312041.62 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.539000 | 192375.00 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.887666 | 235958.25 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.011375 | 109552.73 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.050084 | 94532.56 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.599625 | 87494.14 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.940500 | 108445.31 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 9.188750 | 1148593.75 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.136041 | 1392005.12 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.671875 | 958984.38 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.789875 | 973734.38 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.890750 | 123292.97 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.497125 | 70267.58 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.619500 | 87804.69 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6.391375 | 99865.23 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.365834 | 545729.25 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.624625 | 453078.12 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.881417 | 610177.12 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.385459 | 548182.38 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 18.926666 | 295729.16 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.375833 | 271497.39 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 13.734958 | 214608.72 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 11.938417 | 186537.77 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 28.667500 | 3583437.50 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 36.271333 | 4533916.62 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 30.524792 | 3815599.00 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 24.918000 | 3114750.00 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.413916 | 178342.44 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 10.677958 | 166843.09 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.562583 | 55665.36 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.578083 | 71532.55 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.996875 | 1499609.38 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 16.201875 | 2025234.38 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.852667 | 981583.38 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 6.627583 | 828447.88 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.254666 | 97729.16 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.614584 | 103352.88 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 32.662791 | 510356.11 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 45.421000 | 709703.12 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.735000 | 841875.00 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 12.143542 | 1517942.75 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 70.860666 | 8857583.25 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 73.426167 | 9178270.88 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.787042 | 90422.53 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 8.118667 | 126854.17 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 30.682834 | 479419.28 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 38.533583 | 602087.23 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.956292 | 369536.50 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.136958 | 517119.75 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 39.137250 | 4892156.25 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 32.534750 | 4066843.75 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 3.597041 | 224815.06 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 3.049042 | 190565.12 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 40.610042 | 2538127.62 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 37.047917 | 2315494.81 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 4.245750 | 265359.38 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 2.767833 | 172989.56 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 38.223500 | 2388968.75 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 34.878209 | 2179888.06 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.014208 | 888.00 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.001167 | 72.94 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.013416 | 838.50 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.001584 | 99.00 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.012916 | 807.25 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003458 | 216.12 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000625 | 39.06 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.013750 | 859.38 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003834 | 239.62 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000084 | 5.25 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.012542 | 783.88 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.001458 | 91.12 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000125 | 7.81 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.011542 | 721.38 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.003416 | 213.50 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.010917 | 682.31 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002250 | 140.62 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.012167 | 760.44 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002208 | 138.00 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 16.3x faster than the slowest successful cell (`julia-base`, 8.498 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 21.0x faster than the slowest successful cell (`pytorch-cpu`, 14.037 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 19.7x faster than the slowest successful cell (`pytorch-cpu`, 14.037 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 19.9x faster than the slowest successful cell (`pytorch-cpu`, 14.037 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 9.494 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 14.1x faster than the slowest successful cell (`pytorch-cpu`, 9.494 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 18.2x faster than the slowest successful cell (`pytorch-cpu`, 9.494 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 24.9x faster than the slowest successful cell (`pytorch-cpu`, 9.494 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 29.9x faster than the slowest successful cell (`pytorch-cpu`, 15.161 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 17.9x faster than the slowest successful cell (`pytorch-cpu`, 15.161 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 18.5x faster than the slowest successful cell (`pytorch-cpu`, 15.161 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 9.284 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 25.9x faster than the slowest successful cell (`pytorch-cpu`, 9.284 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 31.7x faster than the slowest successful cell (`pytorch-cpu`, 9.284 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 745.1x faster than the slowest successful cell (`julia-base`, 1.273 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 901.7x faster than the slowest successful cell (`julia-base`, 1.202 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 20.4x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `pytorch-cpu` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.035 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 24.2x faster than the slowest successful cell (`jax-cpu`, 0.035 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 11.7x faster than the slowest successful cell (`jax-cpu`, 0.035 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 16.3x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `jax-cpu` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 59.3x faster than the slowest successful cell (`pytorch-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 25.1x faster than the slowest successful cell (`pytorch-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 25.0x faster than the slowest successful cell (`pytorch-cpu`, 0.064 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-trace` is 14.8x faster than the slowest successful cell (`pytorch-cpu`, 0.064 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `jax-cpu` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 21.4x faster than the slowest successful cell (`pytorch-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 21.6x faster than the slowest successful cell (`pytorch-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `jax-cpu` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 48.0x faster than the slowest successful cell (`pytorch-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 26.4x faster than the slowest successful cell (`pytorch-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 31.4x faster than the slowest successful cell (`pytorch-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 17.6x faster than the slowest successful cell (`pytorch-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 47.8x faster than the slowest successful cell (`pytorch-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 23.3x faster than the slowest successful cell (`pytorch-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `jax-cpu` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 36.5x faster than the slowest successful cell (`pytorch-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 22.5x faster than the slowest successful cell (`pytorch-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `jax-cpu` is 12.2x faster than the slowest successful cell (`pytorch-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 54.4x faster than the slowest successful cell (`pytorch-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 25.9x faster than the slowest successful cell (`pytorch-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `jax-cpu` is 10.0x faster than the slowest successful cell (`pytorch-cpu`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 37.6x faster than the slowest successful cell (`pytorch-cpu`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 23.2x faster than the slowest successful cell (`pytorch-cpu`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 27.8x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 19.5x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `jax-cpu` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 32.5x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 19.6x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `jax-cpu` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 35.8x faster than the slowest successful cell (`pytorch-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 21.5x faster than the slowest successful cell (`pytorch-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 18.0x faster than the slowest successful cell (`pytorch-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`pytorch-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 18.6x faster than the slowest successful cell (`jax-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-trace` is 14.1x faster than the slowest successful cell (`jax-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 14.3x faster than the slowest successful cell (`jax-cpu`, 0.043 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 11.1x faster than the slowest successful cell (`jax-cpu`, 0.043 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 15.9x faster than the slowest successful cell (`jax-cpu`, 0.045 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`jax-cpu`, 0.045 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 20.7x faster than the slowest successful cell (`pytorch-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 19.6x faster than the slowest successful cell (`pytorch-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 32.2x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 23.8x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x64x24`): `tenferro-direct` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 17.8x faster than the slowest successful cell (`pytorch-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 16.2x faster than the slowest successful cell (`pytorch-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 20.4x faster than the slowest successful cell (`pytorch-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 18.7x faster than the slowest successful cell (`pytorch-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 23.4x faster than the slowest successful cell (`pytorch-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 17.3x faster than the slowest successful cell (`pytorch-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 18.3x faster than the slowest successful cell (`jax-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 17.0x faster than the slowest successful cell (`jax-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 19.2x faster than the slowest successful cell (`pytorch-cpu`, 0.064 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 19.2x faster than the slowest successful cell (`pytorch-cpu`, 0.064 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 23.6x faster than the slowest successful cell (`pytorch-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 17.5x faster than the slowest successful cell (`pytorch-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 15.7x faster than the slowest successful cell (`jax-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 16.4x faster than the slowest successful cell (`jax-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `jax-cpu` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 34.8x faster than the slowest successful cell (`pytorch-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 19.3x faster than the slowest successful cell (`pytorch-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 27.3x faster than the slowest successful cell (`pytorch-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 20.6x faster than the slowest successful cell (`pytorch-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 18.0x faster than the slowest successful cell (`pytorch-cpu`, 0.032 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.032 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 13.2x faster than the slowest successful cell (`pytorch-cpu`, 0.037 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 0.032 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 12.3x faster than the slowest successful cell (`pytorch-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 15.5x faster than the slowest successful cell (`pytorch-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x2x2,rhs=1`): `tenferro-direct` is 23.9x faster than the slowest successful cell (`pytorch-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x4x4,rhs=1`): `tenferro-direct` is 15.3x faster than the slowest successful cell (`pytorch-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x8x8,rhs=1`): `tenferro-direct` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 0.055 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 10.8x faster than the slowest successful cell (`jax-cpu`, 5.961 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 11.3x faster than the slowest successful cell (`julia-base`, 5.277 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 23.2x faster than the slowest successful cell (`julia-base`, 5.277 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 10.9x faster than the slowest successful cell (`tenferro-eager`, 9.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 13.2x faster than the slowest successful cell (`tenferro-direct`, 4.892 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 8542.9x faster than the slowest successful cell (`julia-base`, 11.388 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 4294.9x faster than the slowest successful cell (`julia-base`, 12.348 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 4379.1x faster than the slowest successful cell (`pytorch-cpu`, 7.663 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 5839.4x faster than the slowest successful cell (`julia-base`, 7.299 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 60937.0x faster than the slowest successful cell (`julia-base`, 78.731 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 74960.2x faster than the slowest successful cell (`julia-base`, 90.627 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 12.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 161.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 22.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 156.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 90.1x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 136.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 152.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 9817.9 GB/s at 0.002 ms. The conservative physical lower bound is 0.168 ms, so the cell is 98.2x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 12579.9 GB/s at 0.001 ms. The conservative physical lower bound is 0.042 ms, so the cell is 31.4x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 201376.9 GB/s at 0.001 ms. The conservative physical lower bound is 2.684 ms, so the cell is 2013.8x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 93368.9 GB/s at 0.003 ms. The conservative physical lower bound is 0.671 ms, so the cell is 233.4x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 153391.7 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1533.9x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 214748.4 GB/s at 0.001 ms. The conservative physical lower bound is 0.671 ms, so the cell is 536.9x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 103883.7 GB/s at 0.001 ms. The conservative physical lower bound is 1.342 ms, so the cell is 1038.8x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 111015.5 GB/s at 0.001 ms. The conservative physical lower bound is 0.336 ms, so the cell is 277.5x faster than the bound.

# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20261105_175005/run.yaml`
- Timestamp: `2026-11-05T17:50:05.395492+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20261105_175005`.

- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

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

- CSV: `data/results/mac-cpu/cpu/public_api/20261105_175005/cpu_public_api_t1_20261105_175005.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20261105_175005/cpu_public_api_t4_20261105_175005.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 3.649 ± 0.063 | - | 3.859 ± 0.086 | 3.894 ± 0.089 | 1.209 ± 0.134 | 6.069 ± 0.718 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 3.654 ± 0.189 | - | 3.629 ± 0.134 | 3.947 ± 0.405 | 1.163 ± 0.093 | 1.851 ± 0.142 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 8.841 ± 2.798 | - | 8.364 ± 0.294 | 9.762 ± 1.503 | 8.888 ± 0.630 | 8.128 ± 0.215 | 15.416 ± 5.550 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 6.230 ± 2.703 | - | 4.866 ± 1.200 | 5.283 ± 0.498 | 5.562 ± 1.411 | 7.120 ± 0.178 | 3.832 ± 0.133 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 44.595 ± 2.046 | - | 43.290 ± 2.655 | 19.043 ± 0.426 | 36.025 ± 1.750 | 46.888 ± 1.504 | 50.770 ± 1.917 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 15.711 ± 2.335 | - | 14.872 ± 2.051 | 7.364 ± 0.407 | 17.186 ± 3.135 | 42.163 ± 4.520 | 12.174 ± 2.906 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 9.571 ± 0.298 | - | 9.420 ± 0.550 | 8.000 ± 0.300 | 46.748 ± 2.862 | 61.165 ± 1.783 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 9.392 ± 0.368 | - | 8.552 ± 0.967 | 8.524 ± 1.176 | 12.969 ± 1.445 | 13.637 ± 1.538 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 11.041 ± 2.263 | - | unsupported | 9.140 ± 0.306 | 49.573 ± 8.535 | 57.509 ± 3.446 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 9.638 ± 0.923 | - | unsupported | 9.055 ± 0.598 | 13.343 ± 0.912 | 13.867 ± 0.623 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 9.964 ± 0.533 | - | 9.292 ± 1.542 | 9.534 ± 0.273 | 16.295 ± 0.768 | 8.395 ± 1.107 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 8.836 ± 0.464 | - | 8.571 ± 0.578 | 9.217 ± 0.441 | 18.128 ± 0.961 | 11.412 ± 0.840 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 34.166 ± 1.515 | - | 31.805 ± 1.901 | 36.524 ± 7.413 | 42.945 ± 1.360 | 47.171 ± 2.276 | 46.821 ± 1.821 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 10.763 ± 5.859 | - | 13.163 ± 2.381 | 13.779 ± 0.468 | 16.204 ± 2.409 | 42.316 ± 1.382 | 14.273 ± 2.215 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 44.913 ± 1.653 | - | 43.427 ± 1.591 | 65.727 ± 2.891 | 87.333 ± 2.135 | 68.166 ± 8.258 | 70.523 ± 6.169 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 11.230 ± 0.795 | - | 13.252 ± 4.213 | 18.402 ± 0.497 | 30.484 ± 4.567 | 75.660 ± 9.651 | 21.184 ± 3.113 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 6.495 ± 0.414 | - | 6.086 ± 1.338 | 6.849 ± 0.594 | 6.978 ± 1.625 | 7.277 ± 0.630 | 7.753 ± 0.599 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 3.481 ± 1.917 | - | 4.200 ± 0.319 | 3.768 ± 0.788 | 5.091 ± 0.921 | 5.658 ± 0.459 | 4.279 ± 0.196 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.074 ± 0.047 | - | 6.642 ± 0.290 | 5.511 ± 0.165 | 4.433 ± 0.221 | 8.440 ± 0.680 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 0.649 ± 0.013 | - | 1.686 ± 0.790 | 2.397 ± 0.499 | 1.931 ± 0.334 | 7.896 ± 0.168 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 6.867 ± 0.287 | - | 6.886 ± 0.505 | 7.087 ± 0.140 | 5.405 ± 0.403 | 9.487 ± 1.297 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 6.128 ± 0.555 | - | 5.990 ± 0.344 | 6.732 ± 0.217 | 5.712 ± 0.369 | 4.676 ± 2.103 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 4.632 ± 0.435 | - | 4.901 ± 0.368 | 4.972 ± 0.461 | 2.292 ± 0.164 | 6.228 ± 0.257 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 4.698 ± 0.131 | - | 5.559 ± 0.494 | 4.982 ± 0.393 | 2.151 ± 0.065 | 2.277 ± 0.026 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 5.977 ± 0.439 | - | 5.983 ± 0.588 | 5.695 ± 0.186 | 4.167 ± 0.210 | 7.500 ± 2.740 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 5.388 ± 0.049 | - | 5.601 ± 0.491 | 5.310 ± 0.634 | 4.398 ± 0.236 | 4.848 ± 0.254 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 9.586 ± 0.236 | - | 9.418 ± 0.471 | 9.574 ± 0.745 | 51.915 ± 2.509 | 56.963 ± 2.628 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 9.156 ± 0.934 | - | 8.552 ± 0.661 | 9.339 ± 0.596 | 12.508 ± 1.848 | 14.402 ± 2.106 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 7.945 ± 0.515 | - | unsupported | 7.619 ± 0.257 | 52.338 ± 6.404 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 7.757 ± 1.169 | - | unsupported | 7.881 ± 1.269 | 14.416 ± 1.548 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 7.992 ± 0.664 | - | 8.067 ± 0.942 | 8.591 ± 0.465 | 14.331 ± 1.165 | 8.670 ± 7.344 | 13.757 ± 1.762 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 5.203 ± 0.844 | - | 4.276 ± 0.753 | 5.204 ± 1.520 | 6.378 ± 1.816 | 8.121 ± 1.058 | 4.768 ± 0.892 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 10.960 ± 1.488 | - | 11.891 ± 0.967 | 11.427 ± 0.342 | 18.419 ± 2.332 | 13.767 ± 0.954 | 15.599 ± 2.407 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 7.189 ± 2.466 | - | 6.600 ± 0.465 | 9.479 ± 1.301 | 14.500 ± 0.192 | 11.158 ± 1.359 | 7.081 ± 0.684 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 26.286 ± 0.856 | - | 35.003 ± 1.547 | 35.277 ± 1.309 | 28.805 ± 2.738 | 51.796 ± 2.280 | 55.705 ± 9.536 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 12.967 ± 5.888 | - | 9.504 ± 2.387 | 14.449 ± 2.339 | 12.329 ± 1.848 | 48.273 ± 3.444 | 14.839 ± 1.085 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 4.978 ± 0.566 | - | 4.625 ± 0.565 | 5.385 ± 0.280 | 6.998 ± 0.484 | 4.726 ± 0.677 | 6.878 ± 9.379 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 2.730 ± 0.653 | - | 2.967 ± 0.322 | 2.790 ± 0.646 | 4.875 ± 1.238 | 4.084 ± 0.265 | 2.770 ± 0.455 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 12.791 ± 1.002 | - | 12.698 ± 0.463 | 15.674 ± 1.685 | 14.215 ± 0.743 | 13.623 ± 1.544 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 6.019 ± 0.429 | - | 8.703 ± 1.383 | 6.873 ± 1.490 | 7.489 ± 0.081 | 13.700 ± 3.249 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 31.806 ± 1.515 | - | 31.034 ± 6.874 | 32.726 ± 1.671 | 28.381 ± 0.522 | 36.905 ± 12.026 | 38.076 ± 2.584 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 13.051 ± 8.447 | - | 10.240 ± 3.444 | 8.990 ± 0.808 | 9.393 ± 0.384 | 36.703 ± 4.199 | 9.194 ± 2.937 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 11.413 ± 0.549 | - | 12.864 ± 1.298 | 13.269 ± 1.100 | 19.756 ± 3.142 | 13.811 ± 1.339 | 15.683 ± 0.758 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 7.425 ± 1.598 | - | 7.380 ± 1.626 | 7.323 ± 0.994 | 11.147 ± 1.762 | 13.293 ± 2.052 | 8.114 ± 0.582 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 16.872 ± 1.174 | - | 16.794 ± 0.703 | 27.720 ± 0.469 | 16.760 ± 0.379 | 32.414 ± 1.519 | 34.037 ± 2.708 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 4.587 ± 2.632 | - | 4.462 ± 3.885 | 10.977 ± 1.119 | 5.545 ± 0.195 | 34.411 ± 2.355 | 8.436 ± 0.661 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 26.072 ± 1.221 | - | 26.098 ± 0.595 | 27.188 ± 0.902 | 16.098 ± 0.590 | 27.286 ± 1.888 | 26.582 ± 3.000 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 10.116 ± 6.017 | - | 11.678 ± 6.299 | 11.419 ± 3.253 | 5.823 ± 0.362 | 23.133 ± 1.002 | 6.097 ± 0.803 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 31.451 ± 2.149 | - | 31.582 ± 1.577 | 40.668 ± 4.267 | 26.037 ± 2.687 | 42.627 ± 0.900 | 43.047 ± 1.927 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 7.942 ± 5.492 | - | 9.634 ± 5.938 | 17.962 ± 3.983 | 8.448 ± 0.162 | 40.694 ± 8.017 | 10.934 ± 3.521 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 14.765 ± 0.988 | - | 15.992 ± 0.527 | 20.997 ± 0.724 | 17.215 ± 1.079 | 23.494 ± 3.495 | 24.281 ± 5.269 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 7.941 ± 4.785 | - | 9.232 ± 7.037 | 9.034 ± 0.670 | 5.819 ± 0.129 | 22.779 ± 2.101 | 5.566 ± 0.869 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.276 ± 0.051 | 0.297 ± 0.037 | unsupported | unsupported | 0.228 ± 0.024 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.258 ± 0.039 | 0.287 ± 0.021 | unsupported | unsupported | 0.180 ± 0.007 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 12.544 ± 0.512 | - | 12.705 ± 3.097 | 13.054 ± 0.765 | 19.276 ± 2.265 | 12.819 ± 1.106 | 14.980 ± 1.085 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 7.908 ± 0.927 | - | 8.375 ± 0.934 | 7.525 ± 1.426 | 11.229 ± 1.442 | 12.572 ± 1.728 | 8.399 ± 1.081 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 11.432 ± 0.413 | - | 13.771 ± 1.933 | 12.694 ± 1.697 | 18.743 ± 1.523 | 13.153 ± 1.703 | 15.077 ± 1.401 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 8.377 ± 1.978 | - | 8.185 ± 0.306 | 7.098 ± 0.911 | 11.048 ± 0.390 | 12.713 ± 2.028 | 7.669 ± 2.907 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 12.247 ± 0.983 | - | 11.303 ± 0.235 | 13.005 ± 0.637 | 18.535 ± 1.567 | 13.898 ± 2.315 | 15.497 ± 0.984 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 7.879 ± 1.173 | - | 7.166 ± 1.815 | 7.452 ± 1.517 | 11.022 ± 0.604 | 12.508 ± 3.291 | 8.376 ± 0.341 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 8.187 ± 0.370 | - | 8.330 ± 0.492 | 8.472 ± 0.556 | 13.450 ± 1.566 | 8.228 ± 0.801 | 13.123 ± 1.280 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 4.003 ± 1.774 | - | 4.008 ± 1.925 | 4.561 ± 0.389 | 6.282 ± 0.723 | 8.356 ± 1.532 | 4.555 ± 3.406 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 36.405 ± 0.826 | - | 35.967 ± 0.573 | 67.752 ± 2.359 | 37.484 ± 0.990 | 74.305 ± 4.913 | 83.919 ± 13.084 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 14.980 ± 4.458 | - | 16.091 ± 3.445 | 22.436 ± 4.071 | 12.697 ± 1.302 | 74.261 ± 6.710 | 19.410 ± 1.222 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 7.571 ± 0.422 | - | 7.619 ± 0.519 | 11.697 ± 0.614 | 28.954 ± 0.773 | 5.825 ± 0.388 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 3.575 ± 0.231 | - | 2.868 ± 0.853 | 4.040 ± 0.274 | 10.283 ± 0.360 | 5.958 ± 0.648 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 0.965 ± 0.078 | - | 0.965 ± 0.062 | 17.135 ± 1.416 | 2.353 ± 0.145 | 0.717 ± 0.042 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.495 ± 0.005 | - | 0.304 ± 0.010 | 8.448 ± 1.578 | 1.052 ± 0.125 | 0.780 ± 0.096 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 0.883 ± 0.102 | - | 0.916 ± 0.094 | 1.526 ± 0.138 | 0.748 ± 0.032 | 0.792 ± 0.029 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 0.823 ± 0.433 | - | 2.718 ± 2.184 | 0.689 ± 0.062 | 0.572 ± 0.030 | 0.927 ± 0.165 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 7.815 ± 2.131 | - | 7.848 ± 0.591 | 11.236 ± 1.443 | 29.327 ± 1.263 | 6.345 ± 0.531 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 3.968 ± 0.481 | - | 3.885 ± 0.389 | 3.919 ± 0.171 | 10.202 ± 0.091 | 5.890 ± 1.210 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 0.932 ± 0.067 | - | 0.898 ± 0.096 | 17.316 ± 1.362 | 2.286 ± 0.261 | 0.611 ± 0.061 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.433 ± 0.012 | - | 0.528 ± 0.032 | 8.431 ± 1.313 | 1.034 ± 0.086 | 0.731 ± 0.137 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 3.783 ± 0.392 | - | 3.542 ± 0.509 | 5.587 ± 0.130 | 3.483 ± 0.287 | 3.568 ± 0.093 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 3.905 ± 0.895 | - | 3.894 ± 1.115 | 3.109 ± 1.291 | 2.731 ± 0.078 | 4.192 ± 0.549 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 7.061 ± 0.643 | - | 7.558 ± 0.307 | 6.957 ± 0.777 | 42.544 ± 0.740 | 6.561 ± 0.312 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 3.574 ± 0.659 | - | 3.380 ± 1.118 | 3.160 ± 0.222 | 13.912 ± 0.273 | 6.315 ± 1.342 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.805 ± 0.118 | - | 0.836 ± 0.084 | 4.551 ± 0.216 | 2.492 ± 0.123 | 0.681 ± 0.068 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.410 ± 0.012 | - | 0.473 ± 0.076 | 1.686 ± 0.070 | 0.980 ± 0.162 | 0.707 ± 0.121 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 0.865 ± 0.106 | - | 0.892 ± 0.135 | 0.755 ± 0.088 | 1.093 ± 0.084 | 0.671 ± 0.045 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.987 ± 0.061 | - | 1.283 ± 3.671 | 0.391 ± 0.046 | 0.563 ± 0.037 | 0.836 ± 0.099 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 7.938 ± 0.966 | - | 6.930 ± 0.264 | 6.393 ± 0.281 | 34.257 ± 3.504 | 6.588 ± 0.300 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 3.550 ± 0.902 | - | 3.686 ± 0.333 | 3.056 ± 0.784 | 12.282 ± 0.518 | 5.553 ± 0.201 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.795 ± 0.236 | - | 0.656 ± 0.072 | 4.119 ± 0.641 | 2.133 ± 0.170 | 0.599 ± 0.037 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.403 ± 0.034 | - | 0.461 ± 0.048 | 1.649 ± 0.045 | 0.999 ± 0.024 | 0.758 ± 0.171 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 0.676 ± 0.024 | - | 0.836 ± 0.284 | 0.495 ± 0.011 | 0.864 ± 0.178 | 0.707 ± 0.046 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.708 ± 0.069 | - | 0.689 ± 0.053 | 0.441 ± 0.037 | 0.710 ± 0.060 | 1.065 ± 0.393 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.789 ± 0.078 | 0.801 ± 0.062 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.412 ± 0.024 | 0.418 ± 0.062 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 28.740 ± 1.649 | - | 27.709 ± 1.365 | 31.294 ± 1.400 | 35.948 ± 0.645 | 26.886 ± 6.366 | 28.785 ± 1.822 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 10.959 ± 3.815 | - | 8.343 ± 2.666 | 8.516 ± 3.848 | 12.030 ± 0.303 | 22.434 ± 4.718 | 8.727 ± 1.251 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 18.232 ± 1.028 | - | 18.149 ± 0.916 | 17.944 ± 1.412 | 35.269 ± 1.035 | 41.621 ± 2.126 | 41.395 ± 2.367 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 7.498 ± 5.774 | - | 6.063 ± 3.961 | 7.707 ± 0.345 | 12.020 ± 0.595 | 40.660 ± 3.295 | 13.101 ± 7.040 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 19.367 ± 0.443 | - | 19.198 ± 0.367 | 17.211 ± 1.866 | 21.897 ± 0.749 | 19.244 ± 1.738 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 11.712 ± 1.945 | - | 11.736 ± 2.317 | 9.918 ± 0.313 | 11.812 ± 0.221 | 16.521 ± 2.191 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 8.161 ± 0.429 | - | 8.567 ± 1.470 | 9.145 ± 0.420 | 15.477 ± 0.625 | 9.400 ± 0.850 | 18.177 ± 1.222 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 3.990 ± 0.898 | - | 4.127 ± 1.900 | 6.064 ± 1.039 | 7.650 ± 1.818 | 9.512 ± 2.001 | 4.919 ± 0.899 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 25.213 ± 1.360 | - | 24.808 ± 0.745 | 28.927 ± 1.670 | 26.075 ± 0.409 | 31.683 ± 2.091 | 30.386 ± 1.074 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 6.044 ± 1.733 | - | 6.167 ± 0.317 | 10.171 ± 1.424 | 8.885 ± 0.286 | 28.607 ± 3.573 | 7.399 ± 4.183 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 12.332 ± 2.114 | - | 12.070 ± 0.374 | 13.199 ± 1.124 | 20.522 ± 1.502 | 26.454 ± 0.975 | 27.128 ± 2.170 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 5.905 ± 1.987 | - | 4.764 ± 1.008 | 5.717 ± 0.341 | 7.791 ± 0.939 | 25.582 ± 1.034 | 7.194 ± 0.714 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 12.248 ± 1.990 | - | 11.645 ± 0.970 | 12.994 ± 1.298 | 19.136 ± 1.911 | 12.932 ± 1.925 | 13.858 ± 2.123 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 7.982 ± 1.604 | - | 7.900 ± 0.756 | 7.891 ± 0.605 | 11.170 ± 3.884 | 13.439 ± 2.024 | 7.425 ± 1.216 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 67.266 ± 2.216 | - | 65.300 ± 1.616 | 63.924 ± 2.263 | 14.046 ± 0.566 | 44.795 ± 1.107 | 47.116 ± 2.055 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 21.597 ± 4.725 | - | 21.938 ± 7.836 | 18.570 ± 6.284 | 4.652 ± 0.344 | 44.926 ± 5.157 | 11.326 ± 0.822 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 19.355 ± 0.544 | 18.465 ± 0.797 | unsupported | 19.591 ± 1.129 | 22.009 ± 0.400 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 10.531 ± 0.641 | 11.311 ± 2.204 | unsupported | 8.575 ± 1.569 | 11.029 ± 3.140 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.661 ± 0.232 | - | 0.733 ± 0.384 | 0.388 ± 0.062 | 0.434 ± 0.028 | 0.442 ± 0.033 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.308 ± 0.051 | - | 1.330 ± 2.550 | 0.434 ± 0.013 | 0.347 ± 0.044 | 0.481 ± 0.074 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.436 ± 0.035 | - | 0.432 ± 0.015 | 0.608 ± 0.077 | 0.607 ± 0.121 | 0.527 ± 0.051 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.413 ± 0.174 | - | 0.360 ± 1.135 | 0.342 ± 0.113 | 0.462 ± 0.198 | 0.430 ± 0.030 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 0.818 ± 0.094 | - | unsupported | 0.903 ± 0.025 | 0.594 ± 0.062 | 0.675 ± 0.180 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.379 ± 0.020 | - | unsupported | 0.695 ± 0.099 | 0.595 ± 0.107 | 0.641 ± 0.029 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.324 ± 0.050 | - | 0.508 ± 0.107 | 0.297 ± 0.018 | 0.358 ± 0.093 | 1.128 ± 0.050 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.319 ± 0.014 | - | 0.356 ± 0.025 | 0.270 ± 0.057 | 0.464 ± 0.224 | 0.658 ± 0.044 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.429 ± 0.029 | - | 0.445 ± 0.046 | 0.850 ± 0.077 | 0.621 ± 0.045 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.248 ± 1.226 | - | 0.234 ± 0.013 | 0.551 ± 0.035 | 0.662 ± 0.055 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.709 ± 0.027 | - | 0.644 ± 0.092 | 0.611 ± 0.019 | 0.651 ± 0.074 | 1.156 ± 0.239 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.315 ± 0.049 | - | 2.512 ± 3.129 | 0.408 ± 0.048 | 0.604 ± 0.026 | 1.293 ± 0.206 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.610 ± 0.166 | - | 0.671 ± 0.246 | 0.380 ± 0.014 | 0.367 ± 0.014 | 0.326 ± 0.064 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.541 ± 0.047 | - | 0.698 ± 2.587 | 0.467 ± 0.062 | 0.474 ± 0.123 | 0.309 ± 0.403 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.098 ± 0.097 | - | 0.002 ± 0.000 | 0.766 ± 0.074 | 0.775 ± 0.037 | 1.377 ± 0.117 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.472 ± 0.031 | - | 0.002 ± 0.001 | 0.435 ± 0.028 | 0.628 ± 0.060 | 1.110 ± 0.077 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.061 ± 0.001 | - | 0.069 ± 0.005 | 0.205 ± 0.009 | 0.117 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.099 ± 0.002 | - | 0.109 ± 0.012 | 0.214 ± 0.031 | 0.120 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.213 ± 0.001 | - | 0.236 ± 0.006 | 0.320 ± 0.027 | 0.225 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.019 ± 0.001 | - | 0.025 ± 0.003 | 0.028 ± 0.003 | 0.058 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.001 | 0.004 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.038 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.038 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.129 ± 0.008 | - | 0.171 ± 0.005 | 0.186 ± 0.017 | 0.222 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.011 ± 0.000 | - | 0.017 ± 0.001 | 0.017 ± 0.000 | 0.053 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.051 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.061 ± 0.005 | - | 0.065 ± 0.000 | 0.153 ± 0.013 | 0.094 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.085 ± 0.009 | - | 0.099 ± 0.006 | 0.155 ± 0.031 | 0.162 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.194 ± 0.001 | - | 0.230 ± 0.003 | 0.225 ± 0.030 | 0.249 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.018 ± 0.001 | - | 0.027 ± 0.003 | 0.126 ± 0.017 | 0.097 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.009 ± 0.001 | 0.059 ± 0.009 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.085 ± 0.027 | 0.058 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.087 ± 0.018 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.001 | - | 0.005 ± 0.000 | 0.064 ± 0.021 | 0.007 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.049 ± 0.014 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.010 ± 0.000 | 0.076 ± 0.025 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.085 ± 0.024 | 0.009 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.052 ± 0.012 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.067 ± 0.022 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.079 ± 0.007 | 0.081 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.112 ± 0.001 | - | 0.151 ± 0.003 | 0.215 ± 0.041 | 0.222 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.075 ± 0.029 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.011 ± 0.001 | - | 0.018 ± 0.001 | 0.092 ± 0.011 | 0.086 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.088 ± 0.027 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.077 ± 0.029 | 0.067 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.335 ± 0.034 | - | 0.313 ± 0.040 | 0.488 ± 0.030 | 0.837 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.625 ± 0.121 | - | 1.711 ± 0.112 | 1.877 ± 0.913 | 2.230 ± 0.096 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 5.892 ± 0.172 | - | 5.893 ± 0.340 | 6.172 ± 1.174 | 6.165 ± 0.157 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 4.702 ± 0.111 | - | 4.656 ± 0.456 | 3.509 ± 0.170 | 3.209 ± 0.132 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.088 ± 0.021 | - | 0.088 ± 0.037 | 0.089 ± 0.004 | 0.156 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.020 ± 0.022 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.007 ± 0.002 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.016 ± 0.005 | - | 0.019 ± 0.000 | 0.038 ± 0.008 | 0.067 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.002 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.009 ± 0.002 | - | 0.012 ± 0.003 | 0.013 ± 0.003 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.021 ± 0.005 | - | 0.024 ± 0.000 | 0.030 ± 0.007 | 0.058 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 36.749 ± 1.468 | - | 34.939 ± 0.866 | 27.505 ± 2.662 | 24.020 ± 1.998 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.707 ± 0.054 | - | 0.719 ± 0.049 | 0.716 ± 0.028 | 1.003 ± 0.070 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.013 ± 0.004 | - | 0.015 ± 0.005 | 0.021 ± 0.002 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.042 ± 0.013 | - | 0.055 ± 0.004 | 0.059 ± 0.015 | 0.094 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.314 ± 0.004 | - | 0.315 ± 0.010 | 0.488 ± 0.039 | 0.784 ± 0.092 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 1.598 ± 0.047 | - | 1.531 ± 0.078 | 1.750 ± 0.131 | 1.060 ± 0.222 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 6.086 ± 0.345 | - | 6.102 ± 0.205 | 5.855 ± 0.368 | 3.236 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 5.004 ± 0.500 | - | 6.210 ± 2.845 | 3.591 ± 0.210 | 3.206 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.001 | 0.009 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.104 ± 0.006 | - | 0.093 ± 0.016 | 0.083 ± 0.005 | 0.158 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.007 ± 0.000 | - | 0.010 ± 0.003 | 0.009 ± 0.002 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.008 ± 0.002 | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.017 ± 0.000 | - | 0.019 ± 0.004 | 0.020 ± 0.001 | 0.070 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.018 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.022 ± 0.000 | - | 0.024 ± 0.000 | 0.025 ± 0.005 | 0.098 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 37.664 ± 1.293 | - | 36.851 ± 0.560 | 26.973 ± 1.488 | 12.554 ± 0.434 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.005 ± 0.000 | - | 0.009 ± 0.001 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.685 ± 0.060 | - | 0.697 ± 0.049 | 0.710 ± 0.120 | 0.533 ± 0.069 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.017 ± 0.001 | - | 0.021 ± 0.001 | 0.016 ± 0.001 | 0.024 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.042 ± 0.000 | - | 0.045 ± 0.011 | 0.046 ± 0.004 | 0.098 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.296 ± 0.026 | - | 0.311 ± 0.032 | 0.294 ± 0.032 | 0.615 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.377 ± 0.136 | - | 1.280 ± 0.087 | 1.329 ± 0.482 | 1.624 ± 0.145 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 4.749 ± 0.089 | - | 4.937 ± 0.423 | 4.780 ± 0.309 | 5.126 ± 0.560 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 0.983 ± 0.084 | - | 0.986 ± 0.076 | 1.038 ± 0.339 | 0.985 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.066 ± 0.009 | - | 0.068 ± 0.004 | 0.069 ± 0.005 | 0.102 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.003 ± 0.001 | - | 0.006 ± 0.001 | 0.005 ± 0.000 | 0.009 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.005 ± 0.001 | - | 0.009 ± 0.003 | 0.011 ± 0.004 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.015 ± 0.004 | - | 0.017 ± 0.004 | 0.022 ± 0.006 | 0.024 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.006 ± 0.002 | - | 0.011 ± 0.001 | 0.009 ± 0.000 | 0.015 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.018 ± 0.000 | - | 0.021 ± 0.000 | 0.042 ± 0.012 | 0.028 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 7.790 ± 0.378 | - | 7.949 ± 0.344 | 7.888 ± 0.717 | 8.205 ± 0.332 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.512 ± 0.005 | - | 0.522 ± 0.006 | 0.472 ± 0.031 | 0.577 ± 0.065 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.012 ± 0.003 | - | 0.013 ± 0.000 | 0.013 ± 0.004 | 0.021 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.040 ± 0.007 | - | 0.039 ± 0.007 | 0.072 ± 0.014 | 0.071 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.346 ± 0.029 | - | 0.320 ± 0.038 | 0.286 ± 0.018 | 0.608 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.302 ± 0.162 | - | 1.327 ± 0.058 | 1.250 ± 0.052 | 1.071 ± 0.077 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 4.764 ± 0.120 | - | 4.771 ± 0.267 | 4.465 ± 0.132 | 2.482 ± 1.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 0.896 ± 0.016 | - | 0.977 ± 0.017 | 0.895 ± 0.011 | 0.941 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.002 ± 0.001 | - | 0.005 ± 0.000 | 0.005 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.061 ± 0.002 | - | 0.069 ± 0.001 | 0.062 ± 0.004 | 0.124 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.007 ± 0.000 | - | 0.021 ± 0.001 | 0.007 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.002 | 0.009 ± 0.002 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.036 ± 0.014 | - | 0.023 ± 0.001 | 0.016 ± 0.000 | 0.026 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.004 ± 0.005 | - | 0.005 ± 0.000 | 0.005 ± 0.001 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.003 | 0.009 ± 0.003 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.024 ± 0.002 | - | 0.028 ± 0.007 | 0.020 ± 0.000 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 7.756 ± 0.725 | - | 7.887 ± 0.139 | 8.028 ± 0.650 | 3.913 ± 0.112 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.009 ± 0.001 | 0.006 ± 0.001 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.472 ± 0.009 | - | 0.516 ± 0.012 | 0.473 ± 0.005 | 0.261 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.011 ± 0.000 | - | 0.014 ± 0.002 | 0.013 ± 0.001 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.046 ± 0.002 | - | 0.038 ± 0.000 | 0.036 ± 0.000 | 0.106 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.143 ± 0.023 | - | 0.137 ± 0.017 | 0.131 ± 0.002 | 0.619 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 0.691 ± 0.007 | - | 0.728 ± 0.052 | 0.690 ± 0.024 | 1.517 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 2.149 ± 0.087 | - | 2.199 ± 0.164 | 2.407 ± 0.093 | 5.337 ± 0.286 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.482 ± 0.018 | - | 0.459 ± 0.023 | 0.516 ± 0.096 | 0.989 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.001 ± 0.001 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.022 ± 0.000 | - | 0.025 ± 0.004 | 0.026 ± 0.000 | 0.106 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.002 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.007 ± 0.002 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.003 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.001 | 0.007 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.012 ± 0.002 | - | 0.015 ± 0.003 | 0.014 ± 0.003 | 0.025 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 3.829 ± 0.184 | - | 3.801 ± 0.072 | 3.966 ± 0.290 | 8.516 ± 0.226 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.185 ± 0.013 | - | 0.181 ± 0.021 | 0.191 ± 0.026 | 0.564 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.019 ± 0.005 | - | 0.024 ± 0.004 | 0.025 ± 0.005 | 0.084 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.134 ± 0.016 | - | 0.146 ± 0.022 | 0.140 ± 0.014 | 0.601 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 0.732 ± 0.137 | - | 0.801 ± 0.100 | 0.725 ± 0.053 | 1.060 ± 0.102 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 2.236 ± 0.062 | - | 2.466 ± 1.127 | 2.259 ± 0.251 | 2.498 ± 1.042 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.491 ± 0.044 | - | 0.494 ± 0.037 | 0.566 ± 0.037 | 1.087 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.001 | - | 0.005 ± 0.000 | 0.005 ± 0.001 | 0.006 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.024 ± 0.006 | - | 0.033 ± 0.003 | 0.044 ± 0.004 | 0.116 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.002 | 0.006 ± 0.002 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.003 ± 0.001 | - | 0.006 ± 0.000 | 0.007 ± 0.002 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.004 ± 0.001 | - | 0.008 ± 0.002 | 0.008 ± 0.002 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.010 ± 0.002 | - | 0.010 ± 0.003 | 0.011 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.002 ± 0.001 | - | 0.004 ± 0.001 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.013 ± 0.002 | - | 0.017 ± 0.000 | 0.013 ± 0.000 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 4.137 ± 0.184 | - | 4.080 ± 0.204 | 4.648 ± 0.375 | 4.170 ± 0.113 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.210 ± 0.044 | - | 0.204 ± 0.020 | 0.189 ± 0.004 | 0.369 ± 0.071 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.009 ± 0.000 | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.024 ± 0.001 | - | 0.027 ± 0.002 | 0.021 ± 0.000 | 0.101 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.131 ± 0.023 | - | 0.123 ± 0.021 | 0.301 ± 0.042 | 0.211 ± 0.059 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.237 ± 0.037 | - | 0.226 ± 0.015 | 0.394 ± 0.040 | 0.269 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.509 ± 0.064 | - | 0.569 ± 0.087 | 0.684 ± 0.095 | 0.545 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.065 ± 0.011 | - | 0.071 ± 0.006 | 0.065 ± 0.006 | 0.109 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.002 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.006 ± 0.001 | - | 0.008 ± 0.000 | 0.011 ± 0.001 | 0.046 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.009 ± 0.002 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.008 ± 0.001 | - | 0.011 ± 0.000 | 0.013 ± 0.004 | 0.059 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.008 ± 0.002 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.004 ± 0.002 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.039 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.004 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.003 | 0.011 ± 0.003 | 0.039 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.468 ± 0.021 | - | 0.503 ± 0.031 | 0.467 ± 0.280 | 0.540 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.005 | 0.009 ± 0.003 | 0.014 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.034 ± 0.004 | - | 0.037 ± 0.005 | 0.038 ± 0.004 | 0.105 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.004 ± 0.001 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.036 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.015 ± 0.001 | 0.048 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.108 ± 0.013 | - | 0.117 ± 0.022 | 0.232 ± 0.023 | 0.263 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.233 ± 0.013 | - | 0.240 ± 0.011 | 0.298 ± 0.028 | 0.368 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.489 ± 0.036 | - | 0.512 ± 0.029 | 0.552 ± 0.041 | 0.389 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.057 ± 0.001 | - | 0.064 ± 0.001 | 0.137 ± 0.019 | 0.105 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.088 ± 0.010 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.007 ± 0.001 | - | 0.009 ± 0.000 | 0.098 ± 0.012 | 0.061 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.060 ± 0.010 | 0.021 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.010 ± 0.002 | - | 0.012 ± 0.000 | 0.071 ± 0.012 | 0.066 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.082 ± 0.046 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.090 ± 0.038 | 0.009 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.061 ± 0.015 | 0.016 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.061 ± 0.015 | 0.103 ± 0.061 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.085 ± 0.012 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.066 ± 0.010 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.060 ± 0.015 | 0.083 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.455 ± 0.009 | - | 0.451 ± 0.009 | 0.418 ± 0.049 | 0.520 ± 0.074 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.002 | 0.117 ± 0.029 | 0.012 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.031 ± 0.000 | - | 0.035 ± 0.000 | 0.157 ± 0.012 | 0.171 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.084 ± 0.016 | 0.123 ± 0.067 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.007 ± 0.002 | - | 0.008 ± 0.000 | 0.098 ± 0.031 | 0.058 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.261 ± 0.008 | - | 0.245 ± 0.019 | 0.338 ± 0.022 | 0.409 ± 0.089 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.572 ± 0.056 | - | 0.596 ± 0.048 | 0.680 ± 0.134 | 0.793 ± 0.079 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.296 ± 0.175 | - | 1.280 ± 0.079 | 1.329 ± 1.045 | 1.444 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.288 ± 0.009 | - | 0.316 ± 0.027 | 0.328 ± 0.028 | 0.324 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.011 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.011 ± 0.000 | - | 0.014 ± 0.001 | 0.013 ± 0.000 | 0.048 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.013 ± 0.000 | - | 0.016 ± 0.000 | 0.016 ± 0.000 | 0.051 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.002 | 0.004 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.005 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.002 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.003 ± 0.001 | - | 0.008 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.006 ± 0.000 | - | 0.011 ± 0.003 | 0.010 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 2.312 ± 0.092 | - | 2.287 ± 0.083 | 2.564 ± 0.064 | 2.561 ± 0.324 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.089 ± 0.012 | - | 0.093 ± 0.015 | 0.086 ± 0.016 | 0.133 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.003 | 0.008 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.011 ± 0.003 | - | 0.013 ± 0.000 | 0.013 ± 0.000 | 0.019 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.233 ± 0.019 | - | 0.237 ± 0.001 | 0.309 ± 0.011 | 0.343 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.560 ± 0.044 | - | 0.536 ± 0.011 | 0.601 ± 0.029 | 0.651 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.286 ± 0.117 | - | 1.439 ± 0.156 | 1.348 ± 0.045 | 0.835 ± 0.152 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.276 ± 0.018 | - | 0.301 ± 0.030 | 0.340 ± 0.033 | 0.374 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.051 ± 0.012 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.011 ± 0.000 | - | 0.015 ± 0.000 | 0.029 ± 0.003 | 0.058 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.041 ± 0.008 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.016 ± 0.005 | - | 0.017 ± 0.003 | 0.056 ± 0.012 | 0.087 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.051 ± 0.017 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.035 ± 0.021 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.002 | 0.036 ± 0.015 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.007 ± 0.000 | - | 0.008 ± 0.003 | 0.052 ± 0.007 | 0.012 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.050 ± 0.014 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.049 ± 0.011 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.009 ± 0.000 | - | 0.009 ± 0.002 | 0.044 ± 0.014 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 2.377 ± 0.135 | - | 2.548 ± 2.489 | 2.512 ± 0.152 | 1.493 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.048 ± 0.010 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.083 ± 0.013 | - | 0.232 ± 0.025 | 0.149 ± 0.023 | 0.188 ± 0.067 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.006 ± 0.000 | - | 0.011 ± 0.002 | 0.047 ± 0.015 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.011 ± 0.000 | - | 0.014 ± 0.000 | 0.039 ± 0.009 | 0.022 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.177 ± 0.017 | - | 0.172 ± 0.029 | 0.141 ± 0.022 | 0.228 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.296 ± 0.043 | - | 0.345 ± 0.215 | 0.209 ± 0.026 | 0.291 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.417 ± 0.022 | - | 0.425 ± 0.072 | 0.379 ± 0.027 | 0.551 ± 0.069 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.044 ± 0.005 | - | 0.056 ± 0.004 | 0.063 ± 0.001 | 0.105 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.015 ± 0.009 | 0.010 ± 0.001 | 0.009 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.005 ± 0.000 | - | 0.010 ± 0.001 | 0.013 ± 0.000 | 0.045 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.003 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.019 ± 0.015 | 0.010 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.002 | 0.010 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.004 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.005 ± 0.013 | - | 0.009 ± 0.024 | 0.014 ± 0.001 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.004 | 0.010 ± 0.002 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.003 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.361 ± 0.021 | - | 0.419 ± 0.012 | 0.378 ± 0.007 | 0.486 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.003 | 0.010 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.032 ± 0.001 | - | 0.036 ± 0.007 | 0.036 ± 0.000 | 0.085 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.004 ± 0.003 | - | 0.012 ± 0.007 | 0.011 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.016 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.165 ± 0.006 | - | 0.262 ± 0.002 | 0.179 ± 0.002 | 0.202 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.205 ± 0.032 | - | 0.213 ± 0.021 | 0.201 ± 0.004 | 0.310 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.362 ± 0.026 | - | 0.349 ± 0.020 | 0.349 ± 0.019 | 0.441 ± 0.079 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.046 ± 0.001 | - | 0.060 ± 0.001 | 0.064 ± 0.014 | 0.122 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.004 ± 0.001 | - | 0.009 ± 0.004 | 0.008 ± 0.001 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.001 | 0.058 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.008 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.002 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.002 | 0.008 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.002 | 0.009 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.003 ± 0.001 | - | 0.010 ± 0.001 | 0.009 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.004 ± 0.001 | - | 0.009 ± 0.003 | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.361 ± 0.003 | - | 0.391 ± 0.012 | 0.284 ± 0.060 | 0.425 ± 0.077 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.010 ± 0.003 | 0.058 ± 0.015 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.025 ± 0.000 | - | 0.033 ± 0.000 | 0.072 ± 0.018 | 0.130 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.003 ± 0.001 | - | 0.008 ± 0.000 | 0.054 ± 0.012 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.068 ± 0.010 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.248 ± 0.105 | - | 1.307 ± 0.186 | 1.278 ± 0.105 | 3.297 ± 0.205 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 4.076 ± 0.484 | - | 4.450 ± 0.633 | 4.315 ± 0.428 | 5.829 ± 0.205 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 13.880 ± 0.142 | - | 14.338 ± 0.924 | 13.764 ± 1.180 | 15.982 ± 1.408 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 4.224 ± 0.039 | - | 4.153 ± 0.113 | 4.145 ± 0.150 | 3.207 ± 0.091 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.017 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.198 ± 0.013 | - | 0.189 ± 0.015 | 0.180 ± 0.010 | 0.259 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.001 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.181 ± 0.004 | - | 0.180 ± 0.013 | 0.178 ± 0.013 | 0.311 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.014 ± 0.000 | - | 0.017 ± 0.000 | 0.024 ± 0.006 | 0.026 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.006 ± 0.001 | - | 0.010 ± 0.002 | 0.009 ± 0.002 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.014 ± 0.003 | - | 0.015 ± 0.000 | 0.016 ± 0.002 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.038 ± 0.000 | - | 0.041 ± 0.000 | 0.043 ± 0.001 | 0.054 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.007 ± 0.000 | - | 0.011 ± 0.003 | 0.010 ± 0.000 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.017 ± 0.000 | - | 0.020 ± 0.004 | 0.022 ± 0.005 | 0.035 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.056 ± 0.006 | - | 0.062 ± 0.009 | 0.062 ± 0.010 | 0.067 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 34.641 ± 1.056 | - | 35.947 ± 4.620 | 32.614 ± 0.780 | 28.157 ± 0.701 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.011 ± 0.004 | - | 0.018 ± 0.003 | 0.015 ± 0.000 | 0.039 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.566 ± 0.140 | - | 1.567 ± 0.047 | 1.495 ± 0.076 | 1.567 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.032 ± 0.006 | - | 0.042 ± 0.005 | 0.042 ± 0.008 | 0.060 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.106 ± 0.010 | - | 0.107 ± 0.015 | 0.117 ± 0.014 | 0.167 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.234 ± 0.114 | - | 1.314 ± 0.081 | 1.494 ± 0.153 | 3.331 ± 0.339 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 4.228 ± 0.306 | - | 4.241 ± 0.181 | 4.659 ± 0.812 | 2.395 ± 0.304 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 13.355 ± 0.818 | - | 13.241 ± 1.546 | 15.119 ± 1.914 | 7.543 ± 2.570 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 3.870 ± 0.269 | - | 4.035 ± 0.233 | 4.215 ± 0.053 | 3.556 ± 0.079 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.005 ± 0.002 | - | 0.006 ± 0.000 | 0.006 ± 0.002 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.183 ± 0.016 | - | 0.197 ± 0.012 | 0.211 ± 0.019 | 0.268 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.008 ± 0.002 | - | 0.012 ± 0.001 | 0.009 ± 0.002 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.166 ± 0.012 | - | 0.180 ± 0.008 | 0.174 ± 0.002 | 0.320 ± 0.081 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.015 ± 0.001 | - | 0.018 ± 0.000 | 0.018 ± 0.000 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.008 ± 0.001 | - | 0.009 ± 0.003 | 0.011 ± 0.003 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.014 ± 0.003 | - | 0.017 ± 0.000 | 0.016 ± 0.000 | 0.028 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.040 ± 0.003 | - | 0.042 ± 0.000 | 0.050 ± 0.009 | 0.055 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.003 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.018 ± 0.000 | - | 0.020 ± 0.006 | 0.025 ± 0.006 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.064 ± 0.009 | - | 0.056 ± 0.010 | 0.064 ± 0.003 | 0.069 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 32.691 ± 1.108 | - | 32.345 ± 0.354 | 34.044 ± 1.982 | 14.165 ± 3.514 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.012 ± 0.001 | - | 0.015 ± 0.004 | 0.014 ± 0.001 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 1.518 ± 0.034 | - | 1.483 ± 0.030 | 1.560 ± 0.094 | 0.966 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.034 ± 0.000 | - | 0.036 ± 0.000 | 0.048 ± 0.003 | 0.057 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.116 ± 0.012 | - | 0.104 ± 0.004 | 0.118 ± 0.017 | 0.192 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.663 ± 0.082 | - | 0.689 ± 0.068 | 0.693 ± 0.081 | 2.636 ± 0.277 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 2.867 ± 0.222 | - | 2.771 ± 0.201 | 2.982 ± 0.423 | 4.893 ± 0.571 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 9.751 ± 0.532 | - | 9.421 ± 0.669 | 9.625 ± 0.886 | 11.013 ± 0.762 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.597 ± 0.123 | - | 1.590 ± 0.066 | 1.419 ± 0.035 | 1.502 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.001 | 0.008 ± 0.001 | 0.016 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.111 ± 0.012 | - | 0.121 ± 0.005 | 0.102 ± 0.003 | 0.151 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.009 ± 0.001 | 0.008 ± 0.001 | 0.017 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.097 ± 0.006 | - | 0.098 ± 0.002 | 0.104 ± 0.012 | 0.145 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.010 ± 0.000 | - | 0.013 ± 0.000 | 0.019 ± 0.004 | 0.031 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.002 | 0.009 ± 0.000 | 0.021 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.012 ± 0.003 | - | 0.015 ± 0.003 | 0.013 ± 0.000 | 0.026 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.024 ± 0.000 | - | 0.027 ± 0.000 | 0.034 ± 0.008 | 0.052 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.001 | 0.008 ± 0.000 | 0.024 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.026 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.041 ± 0.007 | - | 0.036 ± 0.000 | 0.046 ± 0.011 | 0.058 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 12.236 ± 0.273 | - | 12.289 ± 0.835 | 12.423 ± 0.515 | 12.645 ± 0.943 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.009 ± 0.001 | - | 0.012 ± 0.003 | 0.013 ± 0.000 | 0.035 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 0.872 ± 0.042 | - | 0.842 ± 0.038 | 0.798 ± 0.029 | 0.953 ± 0.098 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.022 ± 0.000 | - | 0.032 ± 0.001 | 0.025 ± 0.000 | 0.042 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.072 ± 0.009 | - | 0.083 ± 0.014 | 0.077 ± 0.012 | 0.145 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.671 ± 0.061 | - | 0.679 ± 0.071 | 0.795 ± 0.181 | 2.199 ± 0.060 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 2.829 ± 0.122 | - | 2.869 ± 0.238 | 3.327 ± 1.304 | 1.570 ± 0.262 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 9.713 ± 0.163 | - | 9.193 ± 1.082 | 9.939 ± 0.485 | 3.938 ± 1.287 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 1.554 ± 0.053 | - | 1.652 ± 0.405 | 1.606 ± 0.098 | 1.753 ± 0.090 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.001 | 0.011 ± 0.005 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.099 ± 0.002 | - | 0.108 ± 0.000 | 0.121 ± 0.004 | 0.175 ± 0.064 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.093 ± 0.012 | - | 0.095 ± 0.001 | 0.097 ± 0.009 | 0.138 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.011 ± 0.002 | - | 0.017 ± 0.004 | 0.022 ± 0.003 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.003 | 0.010 ± 0.012 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.016 ± 0.002 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.034 ± 0.014 | - | 0.036 ± 0.007 | 0.092 ± 0.030 | 0.039 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.006 ± 0.001 | - | 0.007 ± 0.000 | 0.010 ± 0.007 | 0.019 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.013 ± 0.001 | - | 0.015 ± 0.005 | 0.020 ± 0.003 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.045 ± 0.009 | - | 0.037 ± 0.003 | 0.051 ± 0.029 | 0.050 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 12.196 ± 0.315 | - | 12.403 ± 0.199 | 12.800 ± 1.043 | 5.221 ± 2.717 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.009 ± 0.002 | - | 0.010 ± 0.000 | 0.015 ± 0.011 | 0.027 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 0.779 ± 0.018 | - | 0.827 ± 0.016 | 0.860 ± 0.042 | 0.554 ± 0.094 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.022 ± 0.000 | - | 0.025 ± 0.000 | 0.032 ± 0.015 | 0.042 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.066 ± 0.012 | - | 0.069 ± 0.000 | 0.077 ± 0.008 | 0.133 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.239 ± 0.036 | - | unsupported | 0.254 ± 0.055 | 1.572 ± 0.122 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.433 ± 0.071 | - | unsupported | 1.419 ± 0.090 | 2.653 ± 0.128 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 4.232 ± 2.372 | - | unsupported | 4.017 ± 0.317 | 5.414 ± 0.288 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 1.184 ± 0.268 | - | unsupported | 0.958 ± 0.049 | 0.880 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.043 ± 0.010 | - | unsupported | 0.040 ± 0.013 | 0.042 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.003 ± 0.002 | - | unsupported | 0.008 ± 0.000 | 0.014 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.037 ± 0.008 | - | unsupported | 0.049 ± 0.009 | 0.040 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.001 | - | unsupported | 0.005 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.005 ± 0.001 | - | unsupported | 0.008 ± 0.000 | 0.018 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.011 ± 0.000 | - | unsupported | 0.015 ± 0.000 | 0.025 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.002 ± 0.001 | - | unsupported | 0.005 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.008 ± 0.002 | - | unsupported | 0.013 ± 0.002 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.017 ± 0.005 | - | unsupported | 0.023 ± 0.004 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 7.864 ± 0.197 | - | unsupported | 7.969 ± 0.630 | 7.838 ± 0.359 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.280 ± 0.018 | - | unsupported | 0.298 ± 0.024 | 0.339 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.011 ± 0.002 | - | unsupported | 0.019 ± 0.001 | 0.034 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.033 ± 0.006 | - | unsupported | 0.032 ± 0.004 | 0.046 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.239 ± 0.027 | - | unsupported | 0.275 ± 0.041 | 1.465 ± 0.075 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 1.438 ± 0.113 | - | unsupported | 1.478 ± 0.085 | 1.143 ± 0.210 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 3.843 ± 0.151 | - | unsupported | 4.013 ± 0.340 | 2.364 ± 0.723 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 0.872 ± 0.097 | - | unsupported | 0.851 ± 0.032 | 0.992 ± 0.107 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.002 ± 0.001 | - | unsupported | 0.004 ± 0.000 | 0.013 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.035 ± 0.001 | - | unsupported | 0.037 ± 0.007 | 0.043 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.029 ± 0.001 | - | unsupported | 0.032 ± 0.001 | 0.040 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.005 ± 0.001 | - | unsupported | 0.010 ± 0.001 | 0.012 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.012 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.007 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.015 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.011 ± 0.003 | - | unsupported | 0.019 ± 0.004 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.008 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.015 ± 0.000 | - | unsupported | 0.022 ± 0.005 | 0.027 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 7.278 ± 0.618 | - | unsupported | 6.810 ± 0.544 | 2.954 ± 1.305 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.294 ± 0.021 | - | unsupported | 0.265 ± 0.022 | 0.258 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.011 ± 0.000 | - | unsupported | 0.014 ± 0.000 | 0.029 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.036 ± 0.008 | - | unsupported | 0.032 ± 0.000 | 0.046 ± 0.002 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.864 ± 0.149 | - | unsupported | 0.877 ± 0.051 | 0.997 ± 0.053 | 2.704 ± 1.088 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.082 ± 0.002 | - | unsupported | 0.106 ± 0.013 | 0.252 ± 0.043 | 0.084 ± 0.028 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.146 ± 0.009 | - | unsupported | 0.178 ± 0.030 | 0.236 ± 0.013 | 0.157 ± 0.032 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.333 ± 0.008 | - | unsupported | 0.350 ± 0.054 | 0.385 ± 0.025 | 0.771 ± 0.099 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.903 ± 0.088 | - | unsupported | 0.496 ± 0.052 | 0.932 ± 0.082 | 2.166 ± 0.193 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.086 ± 0.097 | - | unsupported | 0.136 ± 0.053 | 0.193 ± 0.011 | 0.080 ± 0.005 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.178 ± 0.040 | - | unsupported | 0.131 ± 0.012 | 0.255 ± 0.045 | 0.144 ± 0.013 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.335 ± 0.013 | - | unsupported | 0.248 ± 0.033 | 0.445 ± 0.196 | 0.521 ± 0.021 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.199 ± 0.028 | - | unsupported | 0.173 ± 0.007 | 0.455 ± 0.084 | 0.427 ± 0.038 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.071 ± 0.009 | - | unsupported | 0.053 ± 0.014 | 0.092 ± 0.015 | 0.064 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.055 ± 0.000 | - | unsupported | 0.060 ± 0.004 | 0.096 ± 0.027 | 0.110 ± 0.025 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.081 ± 0.003 | - | unsupported | 0.078 ± 0.014 | 0.153 ± 0.018 | 0.228 ± 0.049 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.226 ± 0.019 | - | unsupported | 0.172 ± 0.026 | 0.462 ± 0.118 | 0.324 ± 0.034 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.109 ± 0.002 | - | unsupported | 0.053 ± 0.018 | 0.092 ± 0.009 | 0.060 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.142 ± 0.001 | - | unsupported | 0.045 ± 0.003 | 0.139 ± 0.021 | 0.076 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.206 ± 0.001 | - | unsupported | 0.074 ± 0.004 | 0.208 ± 0.029 | 0.149 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.102 ± 0.013 | - | 0.164 ± 0.017 | 0.081 ± 0.014 | 0.154 ± 0.004 | 0.236 ± 0.043 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.071 ± 0.008 | - | 0.052 ± 0.012 | 0.021 ± 0.018 | 0.022 ± 0.002 | 0.047 ± 0.005 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.040 ± 0.013 | - | 0.057 ± 0.009 | 0.024 ± 0.002 | 0.027 ± 0.001 | 0.047 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.054 ± 0.011 | - | 0.079 ± 0.010 | 0.036 ± 0.009 | 0.053 ± 0.002 | 0.129 ± 0.023 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.099 ± 0.013 | - | 0.155 ± 0.012 | 0.070 ± 0.001 | 0.248 ± 0.029 | 0.170 ± 0.011 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.055 ± 0.005 | - | 0.100 ± 0.003 | 0.022 ± 0.009 | 0.028 ± 0.007 | 0.034 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.041 ± 0.006 | - | 0.051 ± 0.006 | 0.024 ± 0.001 | 0.029 ± 0.003 | 0.049 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.053 ± 0.000 | - | 0.070 ± 0.013 | 0.034 ± 0.001 | 0.088 ± 0.024 | 0.090 ± 0.021 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 7.095 ± 0.495 | - | 6.794 ± 0.721 | 24.501 ± 1.365 | 16.310 ± 1.441 | 38.021 ± 12.957 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 6.777 ± 0.171 | - | 7.020 ± 0.865 | 12.420 ± 0.839 | 12.309 ± 1.105 | 12.319 ± 6.021 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 6.053 ± 0.112 | - | 6.606 ± 0.149 | 6.191 ± 0.945 | 5.552 ± 1.180 | 20.473 ± 2.710 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 6.839 ± 0.879 | - | 6.151 ± 0.609 | 5.457 ± 0.326 | 5.911 ± 0.898 | 6.389 ± 0.246 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 6.338 ± 0.153 | - | 6.397 ± 1.216 | 6.603 ± 0.381 | 9.073 ± 0.578 | 7.093 ± 0.519 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 6.293 ± 0.817 | - | 6.455 ± 0.093 | 6.446 ± 0.100 | 8.909 ± 0.285 | 6.166 ± 0.156 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 7.172 ± 0.512 | - | 7.418 ± 0.260 | 7.836 ± 0.278 | 7.290 ± 1.235 | 6.491 ± 0.314 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 7.431 ± 0.186 | - | 7.445 ± 0.616 | 7.744 ± 0.730 | 7.495 ± 0.686 | 6.475 ± 0.207 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 16.241 ± 0.162 | - | 16.311 ± 0.202 | 18.653 ± 0.800 | 24.124 ± 3.146 | 12.421 ± 0.847 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 18.066 ± 0.523 | - | 18.117 ± 0.568 | 17.438 ± 1.699 | 22.762 ± 0.439 | 8.279 ± 0.299 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.263 ± 0.003 | 0.292 ± 0.025 | unsupported | 0.281 ± 0.030 | 0.306 ± 0.025 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 3.176 ± 0.092 | 3.292 ± 0.324 | unsupported | 3.360 ± 0.100 | 4.100 ± 0.065 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.246 ± 0.006 | 0.263 ± 0.011 | unsupported | 0.250 ± 0.002 | 0.356 ± 0.031 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 3.174 ± 0.135 | 3.590 ± 0.659 | unsupported | 3.152 ± 0.243 | 3.372 ± 0.097 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 7.249 ± 0.054 | - | 7.765 ± 0.783 | 9.197 ± 2.413 | 7.250 ± 0.962 | 26.107 ± 1.354 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 7.497 ± 0.152 | - | 7.651 ± 6.671 | 7.206 ± 0.962 | 7.279 ± 0.149 | 9.558 ± 1.359 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 7.684 ± 0.718 | 7.603 ± 0.122 | 4.707 ± 0.319 | 23.474 ± 1.186 | 8.163 ± 4.792 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 13.603 ± 5.865 | 14.754 ± 6.907 | 4.528 ± 0.243 | 21.463 ± 2.190 | 4.800 ± 0.325 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 6.668 ± 0.219 | - | 6.959 ± 0.585 | 10.880 ± 0.707 | 9.467 ± 1.094 | 30.411 ± 8.008 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 5.853 ± 0.101 | - | 7.078 ± 0.754 | 8.425 ± 0.488 | 6.072 ± 0.647 | 11.287 ± 3.821 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.718 ± 0.071 | - | 0.784 ± 0.057 | 2.316 ± 0.577 | 6.282 ± 0.315 | 5.728 ± 0.701 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.445 ± 0.023 | - | 1.728 ± 1.773 | 2.020 ± 0.089 | 2.398 ± 0.502 | 5.211 ± 0.042 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 9.885 ± 0.377 | - | 9.390 ± 0.556 | 9.851 ± 1.799 | 10.726 ± 1.241 | 19.186 ± 3.002 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 13.031 ± 4.460 | - | 8.964 ± 0.885 | 9.178 ± 1.081 | 11.752 ± 0.756 | 15.709 ± 0.796 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 9.404 ± 0.086 | - | 9.241 ± 0.532 | 10.043 ± 0.812 | 11.383 ± 0.474 | 20.042 ± 1.601 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 14.084 ± 4.663 | - | 9.463 ± 1.365 | 9.669 ± 0.397 | 9.262 ± 0.485 | 15.264 ± 2.975 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.576 ± 0.136 | 1.605 ± 0.093 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 10.867 ± 0.474 | 10.646 ± 0.201 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.386 ± 0.096 | 1.410 ± 0.025 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 11.541 ± 1.680 | 16.940 ± 1.614 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.497 ± 0.095 | 1.445 ± 0.064 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 9.311 ± 0.424 | 9.708 ± 0.300 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.430 ± 0.031 | 1.424 ± 0.112 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 9.722 ± 0.412 | 12.539 ± 1.118 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 7.082 ± 0.211 | - | 6.628 ± 0.309 | 7.252 ± 0.529 | 5.504 ± 0.135 | 19.316 ± 1.272 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 6.718 ± 0.246 | - | 6.086 ± 2.117 | 5.668 ± 0.248 | 6.502 ± 1.269 | 6.532 ± 0.166 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 29.584 ± 4.215 | 29.085 ± 1.989 | 29.097 ± 3.604 | 30.808 ± 3.145 | 67.066 ± 3.899 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 35.817 ± 4.489 | 33.642 ± 6.870 | 32.093 ± 3.917 | 28.486 ± 4.003 | 47.441 ± 2.758 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 12.860 ± 1.099 | - | 18.146 ± 2.629 | 11.333 ± 0.268 | 58.499 ± 5.189 | 29.105 ± 3.722 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 13.414 ± 0.812 | - | 15.488 ± 1.367 | 10.374 ± 0.886 | 29.041 ± 4.780 | 9.269 ± 0.243 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.174 ± 0.007 | 0.188 ± 0.015 | unsupported | 0.056 ± 0.009 | 0.036 ± 0.004 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.820 ± 0.114 | 2.989 ± 0.265 | unsupported | 0.830 ± 0.083 | 0.474 ± 0.036 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.181 ± 0.010 | 0.182 ± 0.009 | unsupported | 0.059 ± 0.026 | 0.035 ± 0.027 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.722 ± 0.064 | 0.673 ± 0.069 | unsupported | 0.434 ± 0.042 | 0.344 ± 0.048 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.078 ± 0.002 | 0.073 ± 0.007 | unsupported | 0.085 ± 0.006 | 0.179 ± 0.010 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.257 ± 0.029 | 1.282 ± 0.058 | unsupported | 1.374 ± 0.038 | 2.421 ± 0.194 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.028 ± 0.006 | 0.030 ± 0.023 | unsupported | 0.101 ± 0.005 | 0.141 ± 0.009 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.295 ± 0.013 | 0.349 ± 0.057 | unsupported | 0.633 ± 0.036 | 1.292 ± 0.358 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.255 ± 0.023 | 0.275 ± 0.018 | unsupported | 0.332 ± 0.046 | 0.088 ± 0.011 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 4.621 ± 0.503 | 4.301 ± 0.244 | unsupported | 4.787 ± 0.108 | 0.844 ± 0.065 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.080 ± 0.027 | 0.116 ± 0.189 | unsupported | 0.194 ± 0.015 | 0.116 ± 0.048 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 1.035 ± 0.200 | 1.105 ± 0.134 | unsupported | 1.932 ± 0.076 | 0.896 ± 0.076 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.045 ± 0.002 | 0.071 ± 0.013 | unsupported | 0.048 ± 0.003 | 0.066 ± 0.009 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.547 ± 0.036 | 0.570 ± 0.029 | unsupported | 0.504 ± 0.016 | 0.555 ± 0.016 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.047 ± 0.001 | 0.067 ± 0.001 | unsupported | 0.054 ± 0.002 | 0.115 ± 0.033 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.556 ± 0.198 | 0.564 ± 0.216 | unsupported | 0.207 ± 0.051 | 0.644 ± 0.107 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.091 ± 0.015 | 0.105 ± 0.004 | unsupported | 0.087 ± 0.007 | 0.097 ± 0.011 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.141 ± 0.110 | 1.221 ± 0.066 | unsupported | 1.355 ± 0.125 | 1.137 ± 0.088 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.101 ± 0.012 | 0.125 ± 0.019 | unsupported | 0.098 ± 0.013 | 0.125 ± 0.048 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.566 ± 0.063 | 0.683 ± 0.281 | unsupported | 0.446 ± 0.181 | 0.964 ± 0.282 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.130 ± 0.010 | 0.148 ± 0.030 | unsupported | unsupported | 0.110 ± 0.002 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 1.837 ± 0.259 | 1.867 ± 0.054 | unsupported | unsupported | 1.612 ± 0.118 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.121 ± 0.006 | 0.149 ± 0.015 | unsupported | unsupported | 0.131 ± 0.035 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 0.904 ± 0.385 | 1.189 ± 0.479 | unsupported | unsupported | 1.832 ± 0.363 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.012 ± 0.000 | 0.017 ± 0.001 | unsupported | 0.008 ± 0.001 | 0.047 ± 0.012 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.115 ± 0.008 | 0.107 ± 0.009 | unsupported | 0.058 ± 0.005 | 0.193 ± 0.011 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.015 ± 0.001 | 0.021 ± 0.002 | unsupported | 0.039 ± 0.004 | 0.046 ± 0.012 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.068 ± 0.008 | 0.074 ± 0.015 | unsupported | 0.070 ± 0.004 | 0.233 ± 0.050 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.025 ± 0.001 | 0.038 ± 0.003 | unsupported | 0.027 ± 0.003 | 0.070 ± 0.006 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.321 ± 0.027 | 0.325 ± 0.029 | unsupported | 0.310 ± 0.030 | 0.521 ± 0.037 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.023 ± 0.000 | 0.037 ± 0.001 | unsupported | 0.142 ± 0.030 | 0.110 ± 0.031 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.289 ± 0.031 | 0.273 ± 0.089 | unsupported | 0.365 ± 0.012 | 0.493 ± 0.179 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.048 ± 0.002 | 0.049 ± 0.003 | unsupported | 0.081 ± 0.004 | 0.073 ± 0.015 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.804 ± 0.040 | 0.850 ± 0.051 | unsupported | 1.143 ± 0.096 | 0.734 ± 0.014 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.015 ± 0.003 | 0.014 ± 0.083 | unsupported | 0.081 ± 0.003 | 0.095 ± 0.037 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.185 ± 0.053 | 0.198 ± 0.021 | unsupported | 0.557 ± 0.047 | 0.515 ± 0.038 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.048 ± 0.002 | 0.051 ± 0.003 | unsupported | 0.078 ± 0.007 | 0.085 ± 0.004 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.821 ± 0.044 | 0.794 ± 0.114 | unsupported | 1.174 ± 0.045 | 0.780 ± 0.033 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.021 ± 0.005 | 0.019 ± 0.192 | unsupported | 0.084 ± 0.003 | 0.080 ± 0.041 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.194 ± 0.019 | 0.188 ± 0.047 | unsupported | 0.615 ± 0.063 | 0.599 ± 0.155 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.091 ± 0.005 | 0.109 ± 0.013 | unsupported | 0.094 ± 0.008 | 0.097 ± 0.003 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.157 ± 0.082 | 1.068 ± 0.171 | unsupported | 1.580 ± 0.125 | 1.068 ± 0.115 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.107 ± 0.009 | 0.122 ± 0.005 | unsupported | 0.100 ± 0.008 | 0.123 ± 0.010 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 0.552 ± 0.274 | 0.598 ± 0.559 | unsupported | 0.783 ± 0.081 | 0.902 ± 0.057 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.215 ± 0.014 | 0.214 ± 0.018 | unsupported | 0.290 ± 0.018 | 0.287 ± 0.017 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 3.692 ± 0.593 | 3.912 ± 0.150 | unsupported | 4.253 ± 0.192 | 3.862 ± 0.117 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.069 ± 0.006 | 0.073 ± 0.083 | unsupported | 0.177 ± 0.022 | 0.223 ± 0.027 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 0.866 ± 0.098 | 0.837 ± 0.168 | unsupported | 1.849 ± 0.088 | 2.098 ± 0.121 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.524 ± 0.015 | 0.669 ± 0.042 | unsupported | 0.115 ± 0.012 | 0.089 ± 0.013 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 8.511 ± 0.635 | 9.764 ± 0.271 | unsupported | 1.674 ± 0.092 | 0.809 ± 0.130 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.472 ± 0.041 | 0.626 ± 0.055 | unsupported | 0.131 ± 0.007 | 0.111 ± 0.005 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 3.711 ± 1.155 | 5.259 ± 0.594 | unsupported | 0.847 ± 0.032 | 0.752 ± 0.201 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 13.915 ± 1.969 | - | unsupported | 11.835 ± 1.764 | - | 12.617 ± 1.478 | 13.943 ± 0.663 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 7.749 ± 0.411 | - | unsupported | 6.613 ± 0.325 | - | 9.991 ± 0.041 | 5.563 ± 0.164 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 9.047 ± 0.356 | - | unsupported | 8.432 ± 2.284 | - | 8.252 ± 0.763 | 8.932 ± 0.833 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 5.329 ± 1.782 | - | unsupported | 4.715 ± 0.466 | - | 6.928 ± 0.138 | 3.504 ± 0.508 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 7.256 ± 0.869 | - | unsupported | 8.719 ± 0.392 | - | 6.232 ± 0.453 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 5.287 ± 1.680 | - | unsupported | 5.566 ± 0.892 | - | 5.485 ± 0.199 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 13.430 ± 0.881 | - | unsupported | 13.439 ± 0.818 | - | 12.356 ± 0.575 | 14.328 ± 1.673 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 8.014 ± 0.691 | - | unsupported | 8.758 ± 0.901 | - | 9.949 ± 0.441 | 5.729 ± 1.371 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 7.964 ± 0.271 | - | unsupported | 8.137 ± 1.221 | - | 47.621 ± 1.320 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 7.790 ± 0.904 | - | unsupported | 6.813 ± 0.459 | - | 11.427 ± 0.485 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 8.135 ± 0.319 | - | unsupported | 8.788 ± 1.611 | - | 59.436 ± 5.392 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 8.007 ± 1.316 | - | unsupported | 8.214 ± 1.196 | - | 12.429 ± 0.864 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 11.913 ± 2.143 | - | unsupported | 13.237 ± 0.759 | - | 12.110 ± 0.984 | 14.532 ± 2.088 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 6.757 ± 1.781 | - | unsupported | 6.814 ± 0.236 | - | 9.889 ± 0.159 | 6.732 ± 1.064 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 8.928 ± 0.339 | - | unsupported | 8.975 ± 0.762 | - | 8.238 ± 0.532 | 12.392 ± 1.038 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 3.700 ± 0.183 | - | unsupported | 4.973 ± 1.365 | - | 6.709 ± 0.073 | 3.628 ± 0.137 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.178 ± 0.004 | unsupported | unsupported | 0.223 ± 0.009 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 2.166 ± 0.134 | unsupported | unsupported | 2.694 ± 0.139 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.168 ± 0.007 | unsupported | unsupported | 0.240 ± 0.041 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 2.171 ± 0.102 | unsupported | unsupported | 2.278 ± 0.085 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 13.623 ± 1.597 | - | unsupported | 12.996 ± 0.773 | - | 11.552 ± 0.968 | 15.065 ± 1.534 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 7.956 ± 1.121 | - | unsupported | 6.839 ± 0.760 | - | 10.055 ± 0.542 | 5.273 ± 0.066 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 7.850 ± 2.032 | - | 0.003 ± 0.000 | 5.673 ± 0.635 | 5.474 ± 0.443 | 13.030 ± 0.885 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 5.081 ± 0.741 | - | 0.003 ± 0.001 | 3.610 ± 0.099 | 3.400 ± 1.510 | 12.533 ± 1.889 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 7.976 ± 0.550 | - | 16.289 ± 0.832 | 7.271 ± 0.383 | 8.539 ± 1.036 | 7.550 ± 0.486 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 4.446 ± 1.123 | - | 9.423 ± 1.687 | 4.241 ± 0.108 | 5.060 ± 0.069 | 6.856 ± 0.484 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 10.437 ± 1.323 | - | 11.042 ± 0.397 | 11.099 ± 0.514 | 15.176 ± 0.520 | 11.070 ± 0.815 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 6.854 ± 1.489 | - | 6.187 ± 1.797 | 7.160 ± 2.175 | 7.077 ± 2.884 | 8.895 ± 0.440 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 3.558 ± 0.255 | - | 3.735 ± 0.218 | 9.246 ± 1.379 | 7.548 ± 0.366 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 2.234 ± 0.572 | - | 1.970 ± 0.264 | 4.415 ± 0.178 | 4.962 ± 1.197 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 8.481 ± 2.462 | - | 0.002 ± 0.000 | 8.775 ± 0.903 | 6.953 ± 0.208 | 6.742 ± 0.362 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 4.705 ± 0.647 | - | 0.002 ± 0.000 | 4.390 ± 0.262 | 4.752 ± 0.564 | 5.480 ± 0.497 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 39.824 ± 2.541 | - | 0.002 ± 0.002 | 50.567 ± 1.824 | 47.870 ± 1.230 | 90.534 ± 6.182 | 31.377 ± 3.586 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 15.397 ± 0.895 | - | 0.003 ± 0.007 | 49.026 ± 2.103 | 16.124 ± 2.631 | 88.552 ± 6.142 | 10.592 ± 0.816 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 6.430 ± 0.408 | - | 6.949 ± 0.318 | 6.960 ± 0.431 | 3.939 ± 0.384 | 6.359 ± 0.564 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 3.069 ± 0.756 | - | 2.738 ± 0.442 | 2.929 ± 0.136 | 3.100 ± 0.121 | 4.991 ± 0.092 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 7.027 ± 0.954 | - | 7.377 ± 0.751 | 6.616 ± 0.518 | 4.215 ± 0.354 | 6.459 ± 0.389 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 2.829 ± 0.167 | - | 3.102 ± 0.885 | 2.855 ± 0.087 | 2.771 ± 0.196 | 6.144 ± 0.158 | - |
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
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 3.652209 | 228263.06 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 4.413375 | 275835.94 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 4.754250 | 297140.62 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 2.887958 | 180497.38 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 4.135042 | 258440.13 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 4.591834 | 286989.62 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 12.623500 | 788968.75 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 12.809750 | 800609.38 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 6.597458 | 412341.12 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 6.685958 | 417872.38 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 4.902375 | 306398.44 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 4.493708 | 280856.75 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 4.204250 | 262765.62 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 4.676500 | 292281.25 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 65.597834 | 4099864.63 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 53.753708 | 3359606.75 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 50.813041 | 3175815.06 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 52.674667 | 3292166.69 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 5.690166 | 355635.38 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 4.001500 | 250093.75 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 3.943458 | 246466.12 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 4.200292 | 262518.25 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 53.944708 | 3371544.25 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 50.428250 | 3151765.62 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 50.790625 | 3174414.06 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 57.443416 | 3590213.50 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 25.209125 | 1575570.31 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 25.682209 | 1605138.06 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 173.865458 | 10866591.12 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 170.329584 | 10645599.00 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 22.178917 | 1386182.31 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 22.562416 | 1410151.00 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 184.651667 | 11540729.19 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 271.047583 | 16940473.94 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 23.947750 | 1496734.38 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 23.127667 | 1445479.19 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 148.973250 | 9310828.12 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 155.321333 | 9707583.31 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 22.887417 | 1430463.56 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 22.779584 | 1423724.00 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 155.546542 | 9721658.88 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 200.617750 | 12538609.38 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.275417 | 35553.39 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.610125 | 56408.20 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 11.118000 | 173718.75 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 12.053334 | 188333.34 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 3.791166 | 473895.75 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 6.638250 | 829781.25 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 22.559375 | 2819921.88 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 23.909584 | 2988698.00 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.272000 | 35500.00 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.755583 | 58680.98 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 11.592584 | 181134.12 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 11.679041 | 182485.02 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.748125 | 343515.62 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.473709 | 434213.62 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.778458 | 722307.25 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.382625 | 672828.12 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.438708 | 178729.81 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.438167 | 84971.36 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.999792 | 78121.75 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.650917 | 72670.58 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 19.367417 | 2420927.12 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.994375 | 1374296.88 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.054375 | 1256796.88 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.253708 | 1281713.50 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.997166 | 140580.72 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.488834 | 101388.03 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.820083 | 28438.80 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.919958 | 29999.34 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.335291 | 1291911.38 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.062125 | 632765.62 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.361792 | 295224.00 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.791416 | 348927.00 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.633541 | 88024.08 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 21.273708 | 332401.69 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 16.313459 | 254897.80 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 17.570375 | 274537.11 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.755209 | 844401.12 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 38.296584 | 4787073.00 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 36.970792 | 4621349.00 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 34.408542 | 4301067.75 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.441541 | 116274.08 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 12.394333 | 193661.45 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.141959 | 80343.11 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.405000 | 115703.12 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.164792 | 895599.00 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 15.453625 | 1931703.12 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.277250 | 1034656.25 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 8.838042 | 1104755.25 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.243417 | 66303.39 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.094083 | 48345.05 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.878500 | 44976.56 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.513750 | 70527.34 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 4.439500 | 554937.50 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.030417 | 503802.12 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.379042 | 547380.25 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.561833 | 570229.12 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.339083 | 114673.17 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.443500 | 53804.69 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.018334 | 47161.47 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.278667 | 66854.17 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.154125 | 644265.62 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.658625 | 207328.12 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.450875 | 556359.38 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.514042 | 564255.25 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.178625 | 96541.02 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.599959 | 87499.36 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.847500 | 91367.19 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.721000 | 105015.62 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 9.094000 | 1136750.00 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.837833 | 1354729.12 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.126750 | 1140843.75 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 9.764250 | 1220531.25 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.030000 | 125468.75 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.246917 | 97608.08 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.488209 | 101378.27 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 8.008084 | 125126.31 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.711625 | 963953.12 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.568583 | 446072.88 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.528708 | 566088.50 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.460875 | 682609.38 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.010208 | 109534.50 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.293708 | 129589.19 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.496084 | 148376.31 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.898000 | 1612250.00 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 14.693000 | 1836625.00 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 14.932375 | 1866546.88 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.363125 | 130673.83 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 7.758416 | 121225.25 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 9.523834 | 148809.91 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 14.657958 | 1832244.75 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.230417 | 903802.12 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 9.515625 | 1189453.12 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.999500 | 46867.19 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.515792 | 8059.25 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.740916 | 11576.81 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.100042 | 17188.16 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 1.545250 | 193156.25 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.460375 | 57546.88 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.919334 | 114916.75 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.857292 | 107161.50 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.962750 | 46292.97 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.475417 | 38678.39 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.963333 | 15052.08 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.321625 | 20650.39 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.864084 | 233010.50 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.561042 | 70130.25 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.543708 | 67963.50 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 0.592375 | 74046.88 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.457583 | 69649.73 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 1.759708 | 27495.44 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.617166 | 25268.22 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.456500 | 38382.81 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 4.170000 | 521250.00 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 2.480292 | 310036.50 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.568458 | 321057.25 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 2.596292 | 324536.50 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.015541 | 109617.83 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 9.064708 | 141636.06 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.445000 | 22578.12 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.358042 | 36844.41 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.947083 | 493385.38 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.920042 | 365005.25 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.311375 | 288921.88 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.187500 | 273437.50 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.644750 | 72574.22 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.167500 | 80742.19 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.078458 | 48100.91 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.156667 | 49322.92 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.872833 | 734104.12 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.141667 | 1142708.38 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.434083 | 804260.38 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.800166 | 850020.75 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.066833 | 94794.27 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.202750 | 81292.97 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.964209 | 15065.77 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 0.927875 | 14498.05 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.117209 | 514651.12 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.459584 | 557448.00 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.477458 | 184682.25 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.583667 | 197958.38 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.429833 | 84841.14 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.964500 | 77570.31 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.103708 | 48495.44 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.288958 | 51389.97 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.240834 | 780104.25 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.395375 | 1174421.88 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.571125 | 821390.62 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.353667 | 794208.38 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.143292 | 80363.94 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.380500 | 84070.31 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.336750 | 20886.72 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.243042 | 19422.53 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.789959 | 598744.88 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.918459 | 614807.38 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.554375 | 194296.88 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.502166 | 187770.75 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.225083 | 97266.92 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.021334 | 94083.34 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.848750 | 91386.72 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 7.001292 | 109395.19 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 8.541042 | 1067630.25 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 12.642167 | 1580270.88 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.254917 | 1156864.62 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 8.544000 | 1068000.00 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.901209 | 123456.39 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.415833 | 100247.39 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.856458 | 107132.16 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.790875 | 121732.42 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.214958 | 901869.75 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.266500 | 783312.50 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.417834 | 552229.25 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.783959 | 597994.88 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 18.372459 | 287069.67 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 18.564208 | 290065.75 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 13.733125 | 214580.08 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 13.713291 | 214270.17 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 30.896833 | 3862104.12 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 34.022291 | 4252786.38 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 29.537875 | 3692234.38 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 31.297125 | 3912140.62 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 14.283834 | 223184.91 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 11.323292 | 176926.44 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.404709 | 68823.58 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.670542 | 72977.22 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 16.785541 | 2098192.62 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 14.791041 | 1848880.12 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.929416 | 866177.00 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 6.698917 | 837364.62 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.680250 | 88753.91 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.381875 | 115341.80 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 33.524666 | 523822.91 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 42.810834 | 668919.28 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.470458 | 808807.25 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 13.391916 | 1673989.50 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 68.088042 | 8511005.25 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 78.115416 | 9764427.00 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.120875 | 111263.67 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 8.357042 | 130578.78 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 30.208292 | 472004.56 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 40.084458 | 626319.66 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.019083 | 752385.38 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.776041 | 847005.12 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 29.686333 | 3710791.62 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 42.073209 | 5259151.12 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 3.571875 | 223242.19 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 2.847500 | 177968.75 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 43.098834 | 2693677.12 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 34.653791 | 2165861.94 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 3.843083 | 240192.69 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 2.684666 | 167791.62 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 36.455917 | 2278494.81 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 34.730792 | 2170674.50 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.014750 | 921.88 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.001708 | 106.75 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.011458 | 716.12 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.001459 | 91.19 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000084 | 5.25 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.014166 | 885.38 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003834 | 239.62 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000167 | 10.44 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.011125 | 695.31 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003542 | 221.38 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000125 | 7.81 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.015250 | 953.12 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.003083 | 192.69 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.009084 | 567.75 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.002667 | 166.69 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.013125 | 820.31 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002709 | 169.31 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000125 | 7.81 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.012208 | 763.00 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002375 | 148.44 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 12.2x faster than the slowest successful cell (`julia-base`, 7.896 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 23.9x faster than the slowest successful cell (`pytorch-cpu`, 17.135 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 17.8x faster than the slowest successful cell (`pytorch-cpu`, 17.135 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 17.8x faster than the slowest successful cell (`pytorch-cpu`, 17.135 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 8.448 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 17.1x faster than the slowest successful cell (`pytorch-cpu`, 8.448 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 27.8x faster than the slowest successful cell (`pytorch-cpu`, 8.448 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 28.4x faster than the slowest successful cell (`pytorch-cpu`, 17.316 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 18.6x faster than the slowest successful cell (`pytorch-cpu`, 17.316 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 19.3x faster than the slowest successful cell (`pytorch-cpu`, 17.316 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 8.431 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 19.5x faster than the slowest successful cell (`pytorch-cpu`, 8.431 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 16.0x faster than the slowest successful cell (`pytorch-cpu`, 8.431 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 635.4x faster than the slowest successful cell (`julia-base`, 1.377 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 512.2x faster than the slowest successful cell (`julia-base`, 1.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 19.7x faster than the slowest successful cell (`jax-cpu`, 0.038 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 24.8x faster than the slowest successful cell (`jax-cpu`, 0.038 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 21.5x faster than the slowest successful cell (`jax-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 45.4x faster than the slowest successful cell (`pytorch-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 35.0x faster than the slowest successful cell (`pytorch-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-trace` is 13.8x faster than the slowest successful cell (`pytorch-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `jax-cpu` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 50.8x faster than the slowest successful cell (`pytorch-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 18.6x faster than the slowest successful cell (`pytorch-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 35.4x faster than the slowest successful cell (`pytorch-cpu`, 0.064 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 13.4x faster than the slowest successful cell (`pytorch-cpu`, 0.064 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 30.0x faster than the slowest successful cell (`pytorch-cpu`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `jax-cpu` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 44.3x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 42.6x faster than the slowest successful cell (`pytorch-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 16.0x faster than the slowest successful cell (`pytorch-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 33.9x faster than the slowest successful cell (`pytorch-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 37.6x faster than the slowest successful cell (`pytorch-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 13.4x faster than the slowest successful cell (`pytorch-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 38.9x faster than the slowest successful cell (`jax-cpu`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 14.5x faster than the slowest successful cell (`jax-cpu`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `jax-cpu` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 42.0x faster than the slowest successful cell (`pytorch-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `jax-cpu` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 37.5x faster than the slowest successful cell (`pytorch-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 16.1x faster than the slowest successful cell (`pytorch-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 26.4x faster than the slowest successful cell (`pytorch-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`pytorch-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `jax-cpu` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 34.7x faster than the slowest successful cell (`pytorch-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 18.9x faster than the slowest successful cell (`pytorch-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x32x32`): `tenferro-trace` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 25.4x faster than the slowest successful cell (`pytorch-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 23.5x faster than the slowest successful cell (`pytorch-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 17.0x faster than the slowest successful cell (`pytorch-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 34.9x faster than the slowest successful cell (`pytorch-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 18.8x faster than the slowest successful cell (`pytorch-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 19.1x faster than the slowest successful cell (`pytorch-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 12.2x faster than the slowest successful cell (`pytorch-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 29.5x faster than the slowest successful cell (`jax-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 17.4x faster than the slowest successful cell (`jax-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 32.5x faster than the slowest successful cell (`pytorch-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 17.3x faster than the slowest successful cell (`pytorch-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 21.3x faster than the slowest successful cell (`pytorch-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 12.5x faster than the slowest successful cell (`pytorch-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 21.7x faster than the slowest successful cell (`jax-cpu`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 13.3x faster than the slowest successful cell (`jax-cpu`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 36.9x faster than the slowest successful cell (`pytorch-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 16.7x faster than the slowest successful cell (`pytorch-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 33.5x faster than the slowest successful cell (`jax-cpu`, 0.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 16.2x faster than the slowest successful cell (`jax-cpu`, 0.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 13.7x faster than the slowest successful cell (`pytorch-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 28.2x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 18.2x faster than the slowest successful cell (`pytorch-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 16.0x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.035 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 0.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x2x2,rhs=1`): `tenferro-direct` is 11.7x faster than the slowest successful cell (`tenferro-trace`, 0.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x32x32,rhs=1`): `tenferro-direct` is 12.7x faster than the slowest successful cell (`jax-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x2x2,rhs=1`): `tenferro-direct` is 17.5x faster than the slowest successful cell (`pytorch-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x4x4,rhs=1`): `tenferro-direct` is 16.3x faster than the slowest successful cell (`pytorch-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x8x8,rhs=1`): `tenferro-direct` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 11.7x faster than the slowest successful cell (`julia-base`, 5.211 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 12.1x faster than the slowest successful cell (`tenferro-eager`, 9.764 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 4738.2x faster than the slowest successful cell (`julia-base`, 13.030 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 4932.4x faster than the slowest successful cell (`julia-base`, 12.533 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 4129.4x faster than the slowest successful cell (`pytorch-cpu`, 8.775 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 2481.8x faster than the slowest successful cell (`julia-base`, 5.480 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 39500.2x faster than the slowest successful cell (`julia-base`, 90.534 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 33734.1x faster than the slowest successful cell (`julia-base`, 88.552 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 177.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 69.5x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 119.1x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 113.6x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 164.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 95.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 7738.4 GB/s at 0.002 ms. The conservative physical lower bound is 0.168 ms, so the cell is 77.4x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 7738.4 GB/s at 0.002 ms. The conservative physical lower bound is 0.042 ms, so the cell is 19.3x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 97612.9 GB/s at 0.003 ms. The conservative physical lower bound is 2.684 ms, so the cell is 976.1x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 105641.7 GB/s at 0.003 ms. The conservative physical lower bound is 0.671 ms, so the cell is 264.1x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 126322.6 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1263.2x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 121574.0 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 303.9x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 58559.2 GB/s at 0.002 ms. The conservative physical lower bound is 1.342 ms, so the cell is 585.6x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 51130.6 GB/s at 0.003 ms. The conservative physical lower bound is 0.336 ms, so the cell is 127.8x faster than the bound.

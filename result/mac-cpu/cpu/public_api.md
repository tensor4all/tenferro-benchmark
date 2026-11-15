# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20261115_181810/run.yaml`
- Timestamp: `2026-11-15T18:18:10.224336+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20261115_181810`.

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

- CSV: `data/results/mac-cpu/cpu/public_api/20261115_181810/cpu_public_api_t1_20261115_181810.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20261115_181810/cpu_public_api_t4_20261115_181810.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 4.380 ± 0.320 | - | 4.125 ± 0.157 | 3.894 ± 0.133 | 1.864 ± 2.584 | 7.174 ± 6.820 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 4.148 ± 0.659 | - | 4.795 ± 1.779 | 4.919 ± 2.159 | 1.543 ± 0.209 | 71.564 ± 21.029 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 17.643 ± 2.867 | - | 16.093 ± 2.209 | 17.294 ± 6.460 | 13.182 ± 0.167 | 13.817 ± 0.573 | 22.225 ± 38.060 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 13.256 ± 0.991 | - | 13.511 ± 0.978 | 14.096 ± 0.720 | 13.177 ± 1.393 | 18.776 ± 6.852 | 14.197 ± 4.824 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 50.585 ± 3.114 | - | 44.511 ± 1.535 | 18.675 ± 0.063 | 36.539 ± 5.135 | 60.874 ± 13.685 | 64.594 ± 7.411 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 24.364 ± 2.119 | - | 25.608 ± 3.511 | 11.563 ± 2.194 | 25.256 ± 2.997 | 57.610 ± 6.877 | 28.638 ± 5.193 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 11.747 ± 0.559 | - | 12.215 ± 0.331 | 10.422 ± 0.341 | 53.352 ± 2.985 | 62.547 ± 3.028 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 13.514 ± 4.196 | - | 11.042 ± 1.164 | 11.708 ± 0.401 | 24.104 ± 0.741 | 38.155 ± 9.533 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 13.098 ± 0.300 | - | unsupported | 11.273 ± 0.697 | 61.224 ± 3.625 | 65.302 ± 8.150 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 17.128 ± 5.542 | - | unsupported | 12.037 ± 0.310 | 23.300 ± 3.256 | 48.470 ± 9.886 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 13.430 ± 1.472 | - | 11.801 ± 0.445 | 11.133 ± 3.455 | 21.020 ± 0.802 | 10.064 ± 0.769 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 14.754 ± 3.026 | - | 12.399 ± 1.984 | 12.218 ± 0.371 | 20.770 ± 1.990 | 457.893 ± 46.835 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 35.471 ± 1.415 | - | 36.110 ± 3.185 | 34.204 ± 0.304 | 42.897 ± 0.487 | 62.322 ± 6.569 | 53.958 ± 14.019 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 19.525 ± 2.892 | - | 21.340 ± 3.708 | 19.556 ± 4.596 | 31.569 ± 5.638 | 57.932 ± 12.654 | 24.805 ± 4.973 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 49.970 ± 13.542 | - | 45.831 ± 5.063 | 45.671 ± 4.979 | 108.450 ± 25.954 | 81.477 ± 13.183 | 82.581 ± 13.602 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 26.782 ± 2.433 | - | 25.868 ± 6.062 | 27.460 ± 6.662 | 45.390 ± 10.214 | 125.406 ± 47.634 | 60.404 ± 27.326 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 12.865 ± 2.982 | - | 10.073 ± 0.378 | 9.406 ± 0.967 | 9.398 ± 1.456 | 10.726 ± 1.731 | 10.864 ± 2.828 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 11.403 ± 1.258 | - | 10.260 ± 1.318 | 10.949 ± 0.545 | 10.217 ± 1.164 | 13.465 ± 3.559 | 9.529 ± 1.027 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.373 ± 0.127 | - | 8.275 ± 0.827 | 5.972 ± 3.274 | 5.526 ± 0.974 | 8.635 ± 0.174 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 1.664 ± 0.573 | - | 5.659 ± 1.512 | 3.918 ± 0.800 | 5.042 ± 1.208 | 9.034 ± 0.238 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 8.746 ± 0.812 | - | 9.013 ± 0.524 | 7.285 ± 0.143 | 8.721 ± 2.188 | 14.062 ± 3.510 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 9.614 ± 2.203 | - | 9.381 ± 3.160 | 7.894 ± 0.514 | 7.582 ± 0.144 | 165.046 ± 31.340 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 5.511 ± 0.312 | - | 5.817 ± 0.513 | 4.949 ± 0.030 | 2.794 ± 0.301 | 6.541 ± 0.555 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 8.198 ± 3.929 | - | 7.471 ± 7.956 | 6.495 ± 0.664 | 4.564 ± 0.996 | 284.512 ± 144.287 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 7.626 ± 0.297 | - | 8.803 ± 0.312 | 7.319 ± 0.040 | 9.319 ± 1.291 | 8.572 ± 5.009 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 7.817 ± 2.282 | - | 8.075 ± 0.974 | 8.069 ± 1.410 | 7.552 ± 4.044 | 630.364 ± 223.589 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 11.637 ± 0.374 | - | 11.719 ± 0.540 | 10.677 ± 0.172 | 56.171 ± 3.819 | 71.018 ± 3.443 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 13.210 ± 4.567 | - | 11.393 ± 1.941 | 11.765 ± 0.465 | 22.606 ± 2.961 | 42.782 ± 5.599 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 12.009 ± 0.888 | - | unsupported | 10.809 ± 0.714 | 52.887 ± 7.736 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 11.257 ± 1.117 | - | unsupported | 12.901 ± 3.662 | 26.639 ± 3.162 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 13.805 ± 2.780 | - | 14.247 ± 2.467 | 15.184 ± 3.300 | 17.460 ± 3.433 | 13.020 ± 3.507 | 15.238 ± 0.774 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 13.432 ± 1.095 | - | 13.243 ± 1.335 | 13.436 ± 0.511 | 14.110 ± 0.803 | 12.862 ± 2.895 | 11.679 ± 1.203 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 33.895 ± 7.411 | - | 29.807 ± 8.270 | 20.295 ± 0.999 | 28.264 ± 2.528 | 25.608 ± 14.643 | 34.520 ± 7.503 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 17.251 ± 5.075 | - | 17.542 ± 2.214 | 21.562 ± 2.729 | 25.312 ± 2.204 | 18.483 ± 25.088 | 18.805 ± 4.461 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 30.219 ± 3.053 | - | 38.051 ± 3.809 | 43.413 ± 2.832 | 28.470 ± 0.423 | 56.188 ± 6.123 | 67.444 ± 7.831 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 18.929 ± 3.944 | - | 21.518 ± 10.444 | 26.069 ± 0.837 | 21.622 ± 0.420 | 53.309 ± 24.098 | 25.953 ± 8.858 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 9.406 ± 1.545 | - | 9.323 ± 3.082 | 7.415 ± 0.520 | 10.721 ± 1.680 | 6.533 ± 0.485 | 7.453 ± 0.677 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 5.821 ± 1.863 | - | 6.084 ± 1.452 | 7.111 ± 0.525 | 9.290 ± 0.322 | 6.550 ± 1.109 | 8.051 ± 9.592 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 18.108 ± 4.534 | - | 16.273 ± 3.368 | 17.773 ± 1.753 | 22.938 ± 4.792 | 20.019 ± 12.530 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 12.095 ± 1.698 | - | 11.830 ± 0.277 | 15.406 ± 2.952 | 16.752 ± 0.942 | 23.502 ± 9.198 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 36.083 ± 8.445 | - | 42.319 ± 10.057 | 36.985 ± 3.888 | 29.564 ± 0.589 | 41.747 ± 1.496 | 45.441 ± 4.649 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 14.733 ± 1.894 | - | 17.967 ± 5.732 | 21.435 ± 8.383 | 19.157 ± 4.002 | 43.020 ± 5.690 | 22.319 ± 4.185 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 20.633 ± 3.985 | - | 22.782 ± 1.995 | 22.679 ± 0.870 | 33.505 ± 4.832 | 21.089 ± 3.420 | 23.363 ± 2.350 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 18.997 ± 2.581 | - | 18.132 ± 3.312 | 20.128 ± 0.962 | 25.968 ± 4.980 | 22.875 ± 14.841 | 20.211 ± 4.793 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 26.361 ± 5.805 | - | 28.422 ± 5.174 | 29.184 ± 2.307 | 19.036 ± 2.459 | 40.047 ± 15.312 | 39.888 ± 7.986 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 8.404 ± 0.217 | - | 7.340 ± 0.835 | 17.387 ± 3.033 | 14.230 ± 3.796 | 40.254 ± 9.060 | 23.664 ± 5.489 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 37.248 ± 9.117 | - | 37.153 ± 5.878 | 27.786 ± 5.265 | 16.464 ± 0.478 | 27.898 ± 3.438 | 28.604 ± 2.133 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 16.799 ± 2.429 | - | 16.104 ± 7.008 | 18.737 ± 4.742 | 15.950 ± 1.845 | 27.955 ± 2.954 | 13.440 ± 4.352 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 48.629 ± 13.226 | - | 38.207 ± 11.769 | 46.515 ± 18.665 | 29.229 ± 2.537 | 49.442 ± 4.422 | 52.526 ± 3.598 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 14.328 ± 4.052 | - | 15.434 ± 0.534 | 28.537 ± 6.963 | 19.264 ± 5.055 | 43.643 ± 6.204 | 24.732 ± 4.837 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 18.667 ± 10.157 | - | 17.112 ± 2.330 | 22.018 ± 2.673 | 17.991 ± 1.156 | 25.530 ± 3.297 | 29.308 ± 5.975 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 9.036 ± 1.264 | - | 9.089 ± 5.747 | 13.975 ± 1.992 | 14.010 ± 4.336 | 24.728 ± 1.262 | 12.247 ± 2.790 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.265 ± 0.015 | 0.300 ± 0.034 | unsupported | unsupported | 0.495 ± 0.363 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.258 ± 0.011 | 0.284 ± 0.019 | unsupported | unsupported | 0.477 ± 0.203 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 28.454 ± 6.845 | - | 26.307 ± 6.147 | 22.045 ± 6.194 | 31.396 ± 7.363 | 44.659 ± 36.039 | 27.533 ± 7.386 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 19.310 ± 2.336 | - | 19.240 ± 2.001 | 22.362 ± 2.442 | 27.662 ± 3.160 | 28.703 ± 33.493 | 24.624 ± 7.461 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 26.738 ± 8.672 | - | 33.746 ± 24.420 | 21.671 ± 1.530 | 31.471 ± 3.819 | 22.294 ± 6.469 | 25.398 ± 4.908 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 17.673 ± 2.795 | - | 17.667 ± 0.817 | 21.638 ± 2.201 | 28.460 ± 5.543 | 30.728 ± 9.581 | 22.552 ± 4.998 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 19.287 ± 0.860 | - | 18.678 ± 1.074 | 23.788 ± 1.222 | 29.166 ± 2.603 | 20.214 ± 11.126 | 18.372 ± 3.213 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 19.603 ± 4.688 | - | 18.349 ± 3.159 | 21.567 ± 2.734 | 25.278 ± 2.172 | 15.800 ± 2.304 | 17.435 ± 2.311 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 13.388 ± 0.633 | - | 14.279 ± 1.048 | 12.717 ± 4.503 | 17.060 ± 1.764 | 19.772 ± 38.174 | 14.172 ± 3.023 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 12.973 ± 2.122 | - | 13.134 ± 2.146 | 13.452 ± 0.836 | 13.129 ± 0.336 | 16.713 ± 9.851 | 12.166 ± 1.100 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 39.329 ± 2.826 | - | 44.940 ± 4.996 | 86.188 ± 7.071 | 45.640 ± 0.908 | 111.785 ± 11.690 | 76.683 ± 1.544 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 28.277 ± 7.120 | - | 21.223 ± 6.554 | 35.110 ± 7.029 | 27.686 ± 6.436 | 86.213 ± 13.687 | 29.218 ± 5.329 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 8.712 ± 0.955 | - | 8.444 ± 3.182 | 11.676 ± 0.322 | 25.581 ± 3.415 | 9.115 ± 1.426 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 5.749 ± 0.455 | - | 6.159 ± 0.540 | 6.856 ± 0.083 | 17.659 ± 0.594 | 6.829 ± 1.419 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.186 ± 0.199 | - | 0.970 ± 0.057 | 17.995 ± 4.421 | 2.727 ± 2.105 | 0.936 ± 0.128 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.884 ± 0.027 | - | 0.937 ± 0.076 | 12.159 ± 1.420 | 2.224 ± 0.419 | 0.628 ± 0.159 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 0.789 ± 0.029 | - | 0.781 ± 0.023 | 1.754 ± 0.057 | 1.586 ± 1.576 | 0.960 ± 0.193 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 1.395 ± 0.048 | - | 1.410 ± 0.032 | 1.159 ± 0.088 | 1.514 ± 0.460 | 0.909 ± 0.267 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 8.262 ± 0.256 | - | 7.036 ± 0.348 | 12.685 ± 0.355 | 24.517 ± 0.328 | 7.438 ± 1.188 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 5.421 ± 0.044 | - | 5.571 ± 0.276 | 6.836 ± 0.224 | 17.369 ± 1.366 | 5.986 ± 1.603 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 0.890 ± 0.108 | - | 0.869 ± 0.024 | 16.291 ± 1.393 | 2.688 ± 2.085 | 1.062 ± 0.279 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.787 ± 0.045 | - | 0.902 ± 0.076 | 12.948 ± 0.770 | 2.066 ± 0.562 | 0.638 ± 0.041 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 4.062 ± 0.947 | - | 3.176 ± 0.127 | 6.697 ± 0.149 | 6.644 ± 3.547 | 3.699 ± 0.789 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 6.555 ± 0.914 | - | 5.373 ± 0.097 | 3.953 ± 0.067 | 3.703 ± 0.418 | 3.776 ± 0.504 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 8.397 ± 0.794 | - | 8.001 ± 0.770 | 7.602 ± 0.653 | 45.311 ± 0.615 | 8.480 ± 3.836 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 5.475 ± 0.172 | - | 5.311 ± 0.106 | 6.932 ± 0.312 | 25.759 ± 4.910 | 6.927 ± 1.959 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.990 ± 0.116 | - | 1.075 ± 0.063 | 4.660 ± 0.154 | 4.449 ± 4.441 | 1.058 ± 0.396 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.867 ± 0.024 | - | 0.745 ± 0.032 | 3.751 ± 1.385 | 1.957 ± 0.490 | 0.597 ± 0.008 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 0.980 ± 0.053 | - | 0.930 ± 0.054 | 0.820 ± 0.059 | 1.244 ± 0.174 | 1.035 ± 0.213 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 1.840 ± 0.891 | - | 1.880 ± 0.153 | 0.922 ± 0.307 | 2.112 ± 0.638 | 0.996 ± 0.082 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 7.907 ± 0.517 | - | 10.174 ± 1.492 | 6.918 ± 0.114 | 34.876 ± 0.750 | 8.997 ± 1.308 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 5.799 ± 0.251 | - | 5.684 ± 0.492 | 6.528 ± 0.892 | 21.787 ± 3.022 | 8.528 ± 2.081 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.630 ± 0.026 | - | 0.801 ± 0.288 | 4.654 ± 0.071 | 4.627 ± 3.525 | 0.857 ± 0.130 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.893 ± 0.072 | - | 0.877 ± 0.042 | 3.487 ± 1.846 | 2.153 ± 0.517 | 0.643 ± 0.040 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 0.955 ± 0.062 | - | 0.958 ± 0.060 | 0.891 ± 0.143 | 1.063 ± 0.069 | 1.025 ± 0.209 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 2.160 ± 0.718 | - | 1.926 ± 0.234 | 1.163 ± 1.326 | 1.374 ± 0.361 | 0.697 ± 0.029 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.944 ± 0.160 | 0.823 ± 0.201 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.818 ± 0.104 | 0.909 ± 0.139 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 29.068 ± 0.684 | - | 36.701 ± 3.647 | 35.225 ± 1.911 | 40.241 ± 1.535 | 78.266 ± 34.051 | 33.554 ± 5.675 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 15.348 ± 4.956 | - | 13.623 ± 0.118 | 21.807 ± 7.749 | 31.220 ± 4.337 | 36.411 ± 90.019 | 18.714 ± 1.621 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 19.344 ± 1.473 | - | 19.389 ± 3.189 | 26.343 ± 11.714 | 42.134 ± 3.803 | 48.684 ± 10.884 | 45.264 ± 3.285 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 10.810 ± 2.321 | - | 12.782 ± 0.712 | 13.169 ± 1.222 | 26.823 ± 1.332 | 44.261 ± 3.192 | 22.401 ± 9.121 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 29.791 ± 5.631 | - | 34.434 ± 3.432 | 23.906 ± 2.998 | 31.292 ± 3.039 | 28.577 ± 14.353 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 23.978 ± 0.956 | - | 24.170 ± 0.932 | 25.215 ± 3.591 | 26.953 ± 2.583 | 37.689 ± 10.346 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 14.303 ± 2.572 | - | 14.517 ± 2.509 | 12.600 ± 0.275 | 17.886 ± 2.139 | 19.666 ± 8.294 | 26.316 ± 6.918 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 12.914 ± 1.078 | - | 12.870 ± 0.644 | 13.260 ± 2.211 | 15.279 ± 0.721 | 16.169 ± 3.998 | 17.324 ± 4.105 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 32.496 ± 14.675 | - | 28.921 ± 10.709 | 32.749 ± 6.941 | 28.586 ± 8.795 | 35.148 ± 2.045 | 35.411 ± 0.654 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 11.569 ± 2.136 | - | 11.338 ± 2.153 | 24.501 ± 6.189 | 17.210 ± 1.354 | 31.117 ± 8.389 | 15.992 ± 1.854 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 13.668 ± 0.788 | - | 12.243 ± 0.349 | 15.742 ± 3.867 | 25.297 ± 3.522 | 29.270 ± 1.739 | 30.002 ± 30.504 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 12.854 ± 2.317 | - | 12.750 ± 2.163 | 12.751 ± 0.389 | 15.154 ± 6.237 | 329.930 ± 536.860 | 34.929 ± 175.707 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 20.148 ± 0.888 | - | 19.794 ± 0.990 | 24.496 ± 1.764 | 32.839 ± 5.341 | 25.162 ± 43.495 | 20.801 ± 3.677 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 17.934 ± 1.170 | - | 16.850 ± 1.621 | 19.764 ± 1.601 | 26.167 ± 0.677 | 17.848 ± 2.274 | 16.621 ± 0.542 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 75.322 ± 17.115 | - | 86.866 ± 20.433 | 66.108 ± 14.798 | 13.206 ± 0.427 | 43.514 ± 2.974 | 45.687 ± 4.086 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 37.038 ± 5.397 | - | 58.826 ± 27.230 | 46.425 ± 2.954 | 10.280 ± 2.976 | 63.243 ± 11.589 | 25.274 ± 8.355 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 42.729 ± 8.742 | 29.190 ± 2.195 | unsupported | 23.295 ± 4.605 | 34.635 ± 7.815 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 23.840 ± 1.623 | 24.867 ± 1.270 | unsupported | 25.302 ± 0.711 | 26.834 ± 0.920 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.671 ± 0.011 | - | 0.664 ± 0.018 | 0.699 ± 0.019 | 0.695 ± 0.041 | 0.885 ± 0.168 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.683 ± 0.070 | - | 0.722 ± 0.046 | 1.175 ± 0.022 | 0.756 ± 0.189 | 1.672 ± 0.637 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.935 ± 0.244 | - | 0.708 ± 0.058 | 0.849 ± 0.059 | 0.742 ± 0.022 | 0.919 ± 0.072 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.620 ± 0.033 | - | 0.638 ± 0.044 | 0.997 ± 0.065 | 1.491 ± 0.618 | 0.900 ± 0.283 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 0.958 ± 0.075 | - | unsupported | 1.404 ± 0.090 | 1.456 ± 0.824 | 1.343 ± 4.363 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 1.042 ± 0.033 | - | unsupported | 1.958 ± 0.277 | 1.331 ± 0.461 | 4.073 ± 25.679 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.407 ± 0.069 | - | 0.457 ± 0.033 | 0.323 ± 0.065 | 0.350 ± 0.031 | 1.162 ± 0.074 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.322 ± 0.028 | - | 0.824 ± 0.182 | 0.283 ± 0.018 | 0.570 ± 0.344 | 1.437 ± 2.388 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.662 ± 0.044 | - | 0.690 ± 0.086 | 1.213 ± 0.086 | 0.791 ± 0.066 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.619 ± 0.074 | - | 0.701 ± 0.059 | 1.693 ± 0.183 | 1.072 ± 0.093 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.671 ± 0.027 | - | 0.882 ± 0.251 | 1.024 ± 0.091 | 0.923 ± 0.194 | 1.186 ± 4.660 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.650 ± 0.076 | - | 0.705 ± 0.051 | 0.973 ± 0.102 | 1.536 ± 0.331 | 2.406 ± 7.808 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.692 ± 0.119 | - | 0.742 ± 0.116 | 0.489 ± 0.099 | 1.084 ± 0.681 | 1.180 ± 0.283 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.682 ± 0.032 | - | 0.850 ± 0.075 | 0.560 ± 0.048 | 0.558 ± 0.473 | 0.444 ± 0.098 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.237 ± 0.318 | - | 0.002 ± 0.000 | 1.422 ± 0.177 | 1.231 ± 0.136 | 1.646 ± 10.956 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 1.380 ± 0.048 | - | 0.002 ± 0.000 | 1.599 ± 0.096 | 1.462 ± 0.570 | 1.603 ± 0.083 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.123 ± 0.023 | - | 0.060 ± 0.002 | 0.127 ± 0.001 | 0.159 ± 0.109 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.094 ± 0.002 | - | 0.097 ± 0.003 | 0.177 ± 0.005 | 0.135 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.202 ± 0.006 | - | 0.203 ± 0.002 | 0.279 ± 0.007 | 0.258 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.025 ± 0.001 | - | 0.028 ± 0.002 | 0.039 ± 0.004 | 0.106 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.062 ± 0.103 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.090 ± 0.064 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.175 ± 0.007 | - | 0.179 ± 0.004 | 0.244 ± 0.001 | 0.305 ± 0.088 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.011 ± 0.001 | - | 0.013 ± 0.001 | 0.018 ± 0.002 | 0.052 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.054 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.124 ± 0.060 | - | 0.064 ± 0.007 | 0.194 ± 0.074 | 0.125 ± 0.069 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.105 ± 0.038 | - | 0.102 ± 0.036 | 0.292 ± 0.030 | 0.354 ± 0.124 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.217 ± 0.045 | - | 0.220 ± 0.084 | 0.402 ± 0.051 | 0.460 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.026 ± 0.027 | - | 0.029 ± 0.002 | 0.113 ± 0.038 | 0.118 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.164 ± 0.104 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.106 ± 0.033 | 0.036 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.008 ± 0.001 | 0.085 ± 0.019 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.094 ± 0.053 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.025 ± 0.019 | 0.104 ± 0.077 | 0.008 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.003 ± 0.001 | 0.103 ± 0.054 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.077 ± 0.049 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.010 ± 0.005 | - | 0.026 ± 0.028 | 0.059 ± 0.037 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.004 ± 0.001 | - | 0.009 ± 0.001 | 0.044 ± 0.060 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.102 ± 0.043 | 0.146 ± 0.088 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.229 ± 0.060 | - | 0.230 ± 0.124 | 0.381 ± 0.167 | 0.568 ± 0.236 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.009 ± 0.002 | - | 0.015 ± 0.030 | 0.029 ± 0.038 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.022 ± 0.000 | - | 0.013 ± 0.000 | 0.122 ± 0.065 | 0.090 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.005 ± 0.001 | - | 0.004 ± 0.000 | 0.072 ± 0.061 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.006 ± 0.000 | - | 0.005 ± 0.000 | 0.123 ± 0.014 | 0.011 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.250 ± 0.002 | - | 0.256 ± 0.013 | 0.422 ± 0.002 | 0.947 ± 0.115 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.571 ± 0.818 | - | 1.641 ± 1.335 | 1.670 ± 0.035 | 3.601 ± 1.456 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 7.072 ± 1.510 | - | 6.589 ± 0.382 | 6.480 ± 0.828 | 7.561 ± 3.169 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 4.620 ± 0.351 | - | 4.222 ± 0.046 | 3.544 ± 0.208 | 4.090 ± 0.063 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.091 ± 0.002 | - | 0.092 ± 0.002 | 0.096 ± 0.004 | 0.150 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.014 ± 0.006 | - | 0.026 ± 0.003 | 0.010 ± 0.000 | 0.019 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.003 | 0.009 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.053 ± 0.008 | - | 0.022 ± 0.006 | 0.023 ± 0.001 | 0.092 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.026 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.025 ± 0.000 | - | 0.026 ± 0.005 | 0.029 ± 0.001 | 0.173 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 35.247 ± 0.504 | - | 35.259 ± 2.763 | 31.523 ± 4.824 | 34.931 ± 1.694 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.708 ± 0.020 | - | 0.717 ± 0.019 | 0.773 ± 0.041 | 0.883 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.013 ± 0.000 | - | 0.014 ± 0.000 | 0.017 ± 0.000 | 0.028 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.048 ± 0.001 | - | 0.049 ± 0.001 | 0.053 ± 0.000 | 0.136 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.273 ± 0.050 | - | 0.287 ± 0.028 | 0.430 ± 0.016 | 0.960 ± 0.106 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 1.665 ± 0.968 | - | 1.951 ± 1.742 | 1.706 ± 0.036 | 1.425 ± 0.154 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 6.677 ± 0.366 | - | 6.494 ± 0.197 | 10.018 ± 4.426 | 4.828 ± 0.415 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 4.564 ± 0.252 | - | 4.188 ± 0.635 | 3.647 ± 0.348 | 4.092 ± 0.076 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.002 ± 0.001 | - | 0.012 ± 0.000 | 0.005 ± 0.001 | 0.011 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.095 ± 0.010 | - | 0.102 ± 0.009 | 0.094 ± 0.024 | 0.176 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.019 ± 0.001 | 0.006 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.008 ± 0.000 | - | 0.011 ± 0.002 | 0.010 ± 0.000 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.006 ± 0.001 | - | 0.027 ± 0.006 | 0.009 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.022 ± 0.004 | - | 0.025 ± 0.004 | 0.023 ± 0.000 | 0.099 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.023 ± 0.001 | - | 0.011 ± 0.002 | 0.011 ± 0.000 | 0.057 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.030 ± 0.005 | - | 0.027 ± 0.001 | 0.029 ± 0.000 | 0.116 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 43.997 ± 10.970 | - | 42.747 ± 8.351 | 29.079 ± 1.658 | 17.772 ± 1.685 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.005 ± 0.000 | - | 0.006 ± 0.001 | 0.008 ± 0.000 | 0.049 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.770 ± 0.049 | - | 0.773 ± 0.024 | 0.880 ± 0.075 | 0.653 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.015 ± 0.003 | - | 0.015 ± 0.000 | 0.017 ± 0.000 | 0.027 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.057 ± 0.004 | - | 0.051 ± 0.002 | 0.061 ± 0.030 | 0.142 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.242 ± 0.022 | - | 0.249 ± 0.010 | 0.241 ± 0.001 | 0.569 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.267 ± 1.205 | - | 1.255 ± 0.023 | 1.252 ± 0.020 | 2.065 ± 0.434 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 5.420 ± 1.267 | - | 5.121 ± 2.224 | 4.852 ± 0.042 | 5.830 ± 0.749 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.010 ± 0.047 | - | 1.031 ± 0.158 | 0.976 ± 0.020 | 1.179 ± 0.118 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.067 ± 0.001 | - | 0.068 ± 0.000 | 0.070 ± 0.001 | 0.103 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.015 ± 0.000 | - | 0.017 ± 0.000 | 0.018 ± 0.000 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.020 ± 0.001 | - | 0.021 ± 0.000 | 0.023 ± 0.002 | 0.031 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 7.847 ± 0.505 | - | 10.596 ± 2.633 | 7.788 ± 0.047 | 9.147 ± 0.223 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.531 ± 0.005 | - | 0.526 ± 0.011 | 0.525 ± 0.003 | 0.590 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.011 ± 0.000 | - | 0.012 ± 0.000 | 0.014 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.039 ± 0.000 | - | 0.040 ± 0.000 | 0.041 ± 0.000 | 0.257 ± 0.190 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.375 ± 0.230 | - | 0.364 ± 0.103 | 0.272 ± 0.035 | 0.547 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.472 ± 0.799 | - | 1.524 ± 0.407 | 1.407 ± 0.069 | 1.555 ± 0.145 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 5.498 ± 0.956 | - | 5.784 ± 0.505 | 5.547 ± 0.279 | 3.480 ± 0.401 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.057 ± 0.059 | - | 1.085 ± 0.073 | 1.121 ± 0.271 | 1.265 ± 0.137 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.069 ± 0.008 | - | 0.070 ± 0.003 | 0.071 ± 0.001 | 0.180 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.007 ± 0.002 | - | 0.010 ± 0.001 | 0.009 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.006 ± 0.001 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.047 ± 0.003 | - | 0.021 ± 0.004 | 0.018 ± 0.006 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.008 ± 0.001 | - | 0.010 ± 0.002 | 0.009 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.033 ± 0.009 | - | 0.039 ± 0.035 | 0.023 ± 0.003 | 0.032 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 8.496 ± 0.186 | - | 8.484 ± 0.331 | 8.473 ± 1.298 | 5.251 ± 0.757 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.544 ± 0.053 | - | 0.658 ± 0.150 | 0.563 ± 0.006 | 0.586 ± 0.080 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.013 ± 0.003 | - | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.091 ± 0.035 | - | 0.052 ± 0.019 | 0.046 ± 0.008 | 0.113 ± 0.065 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.109 ± 0.001 | - | 0.111 ± 0.010 | 0.132 ± 0.015 | 0.769 ± 0.390 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 0.737 ± 0.018 | - | 0.750 ± 0.082 | 0.917 ± 0.243 | 1.826 ± 0.310 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 2.730 ± 0.363 | - | 2.419 ± 0.318 | 2.730 ± 0.068 | 6.362 ± 1.322 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.450 ± 0.003 | - | 0.443 ± 0.009 | 0.525 ± 0.034 | 1.313 ± 0.250 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.001 | 0.005 ± 0.000 | 0.009 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.025 ± 0.000 | - | 0.026 ± 0.000 | 0.032 ± 0.007 | 0.113 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.021 ± 0.000 | 0.019 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.027 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.031 ± 0.001 | 0.055 ± 0.113 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.002 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.013 ± 0.003 | - | 0.011 ± 0.003 | 0.037 ± 0.056 | 0.118 ± 0.070 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 3.602 ± 0.041 | - | 3.770 ± 0.305 | 4.089 ± 0.285 | 9.210 ± 0.562 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.018 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.189 ± 0.009 | - | 0.190 ± 0.000 | 0.209 ± 0.041 | 0.621 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.003 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.019 ± 0.000 | - | 0.020 ± 0.000 | 0.049 ± 0.001 | 0.088 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.328 ± 0.002 | - | 0.342 ± 0.017 | 0.135 ± 0.004 | 0.789 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 0.796 ± 0.323 | - | 1.242 ± 0.669 | 0.834 ± 0.058 | 1.640 ± 0.156 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 2.776 ± 0.249 | - | 2.904 ± 0.147 | 2.781 ± 0.054 | 3.694 ± 0.252 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.575 ± 0.057 | - | 0.527 ± 0.079 | 0.603 ± 0.119 | 1.286 ± 0.058 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.032 ± 0.023 | - | 0.029 ± 0.001 | 0.030 ± 0.004 | 0.286 ± 0.169 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | 0.012 ± 0.001 | 0.006 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.008 ± 0.001 | - | 0.014 ± 0.001 | 0.007 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.005 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.009 ± 0.000 | - | 0.014 ± 0.001 | 0.009 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.017 ± 0.001 | - | 0.022 ± 0.001 | 0.015 ± 0.003 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.011 ± 0.001 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.010 ± 0.001 | - | 0.016 ± 0.000 | 0.008 ± 0.002 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.020 ± 0.000 | - | 0.027 ± 0.002 | 0.014 ± 0.003 | 0.030 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 4.240 ± 0.572 | - | 4.504 ± 0.719 | 4.391 ± 0.349 | 5.112 ± 0.726 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.012 ± 0.001 | 0.005 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.220 ± 0.069 | - | 0.242 ± 0.050 | 0.215 ± 0.047 | 0.566 ± 0.170 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.015 ± 0.000 | - | 0.021 ± 0.001 | 0.012 ± 0.003 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.037 ± 0.001 | - | 0.044 ± 0.009 | 0.023 ± 0.003 | 0.150 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.120 ± 0.004 | - | 0.121 ± 0.021 | 0.177 ± 0.055 | 0.140 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.233 ± 0.037 | - | 0.244 ± 0.059 | 0.366 ± 0.134 | 0.213 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.596 ± 0.088 | - | 0.613 ± 0.036 | 0.554 ± 0.067 | 0.587 ± 0.163 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.062 ± 0.002 | - | 0.062 ± 0.004 | 0.071 ± 0.027 | 0.161 ± 0.085 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.007 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.009 ± 0.000 | - | 0.010 ± 0.003 | 0.015 ± 0.008 | 0.098 ± 0.083 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.008 ± 0.000 | - | 0.011 ± 0.002 | 0.014 ± 0.003 | 0.055 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.032 ± 0.097 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.035 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.069 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.009 ± 0.002 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.004 ± 0.001 | - | 0.005 ± 0.000 | 0.009 ± 0.002 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.005 ± 0.000 | - | 0.013 ± 0.007 | 0.011 ± 0.002 | 0.051 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.510 ± 0.008 | - | 0.491 ± 0.023 | 0.612 ± 0.179 | 0.708 ± 0.088 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.005 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.050 ± 0.052 | - | 0.053 ± 0.009 | 0.039 ± 0.000 | 0.257 ± 0.135 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.006 ± 0.001 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.074 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.034 ± 0.018 | 0.079 ± 0.112 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.093 ± 0.004 | - | 0.098 ± 0.004 | 0.351 ± 0.048 | 0.293 ± 0.155 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.226 ± 0.018 | - | 0.233 ± 0.012 | 0.487 ± 0.086 | 0.260 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.486 ± 0.062 | - | 0.655 ± 0.361 | 0.707 ± 0.030 | 0.659 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.065 ± 0.002 | - | 0.067 ± 0.005 | 0.184 ± 0.128 | 0.178 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.001 | 0.115 ± 0.021 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.006 ± 0.000 | - | 0.007 ± 0.002 | 0.076 ± 0.062 | 0.085 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.001 | 0.095 ± 0.026 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.012 ± 0.000 | - | 0.012 ± 0.002 | 0.108 ± 0.084 | 0.074 ± 0.059 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.097 ± 0.013 | 0.010 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.187 ± 0.027 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.136 ± 0.073 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.005 ± 0.000 | - | 0.015 ± 0.001 | 0.076 ± 0.032 | 0.074 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.100 ± 0.031 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.001 | 0.065 ± 0.029 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.011 ± 0.000 | - | 0.015 ± 0.001 | 0.072 ± 0.044 | 0.078 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.532 ± 0.112 | - | 0.576 ± 0.352 | 0.782 ± 0.161 | 0.952 ± 0.347 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.169 ± 0.083 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.037 ± 0.002 | - | 0.037 ± 0.007 | 0.206 ± 0.088 | 0.323 ± 0.066 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.004 ± 0.000 | - | 0.016 ± 0.000 | 0.194 ± 0.118 | 0.048 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.014 ± 0.000 | - | 0.007 ± 0.000 | 0.050 ± 0.001 | 0.055 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.260 ± 0.001 | - | 0.264 ± 0.010 | 0.305 ± 0.001 | 0.326 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 1.342 ± 0.509 | - | 0.637 ± 0.128 | 0.653 ± 0.030 | 0.923 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.238 ± 0.050 | - | 1.565 ± 1.818 | 1.575 ± 0.024 | 2.185 ± 0.518 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.311 ± 0.038 | - | 0.317 ± 0.027 | 0.341 ± 0.074 | 0.365 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.001 | 0.003 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.021 ± 0.005 | - | 0.020 ± 0.000 | 0.020 ± 0.000 | 0.079 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.015 ± 0.000 | - | 0.016 ± 0.000 | 0.017 ± 0.001 | 0.063 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.002 ± 0.018 | - | 0.005 ± 0.001 | 0.004 ± 0.002 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.002 | 0.008 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.006 ± 0.000 | - | 0.009 ± 0.002 | 0.009 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 2.581 ± 0.657 | - | 2.956 ± 1.765 | 2.767 ± 0.022 | 2.812 ± 0.339 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.130 ± 0.004 | - | 0.133 ± 0.001 | 0.139 ± 0.010 | 0.228 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.009 ± 0.002 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.013 ± 0.003 | - | 0.015 ± 0.003 | 0.015 ± 0.000 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.309 ± 0.308 | - | 0.362 ± 0.131 | 0.415 ± 0.088 | 0.519 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.714 ± 0.319 | - | 0.735 ± 0.379 | 0.790 ± 0.033 | 1.626 ± 1.434 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.585 ± 0.264 | - | 1.304 ± 0.293 | 1.888 ± 0.295 | 1.403 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.318 ± 0.010 | - | 0.329 ± 0.037 | 0.722 ± 0.383 | 0.411 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.057 ± 0.049 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.013 ± 0.001 | - | 0.014 ± 0.000 | 0.049 ± 0.014 | 0.073 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.004 ± 0.001 | - | 0.012 ± 0.000 | 0.051 ± 0.030 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.015 ± 0.001 | - | 0.017 ± 0.005 | 0.050 ± 0.009 | 0.100 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.004 ± 0.000 | 0.052 ± 0.050 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.007 ± 0.001 | - | 0.004 ± 0.000 | 0.032 ± 0.015 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | 0.015 ± 0.001 | 0.065 ± 0.004 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.016 ± 0.001 | - | 0.007 ± 0.001 | 0.078 ± 0.033 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.007 ± 0.007 | - | 0.006 ± 0.000 | 0.064 ± 0.017 | 0.038 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | 0.016 ± 0.000 | 0.065 ± 0.003 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.007 ± 0.007 | - | 0.009 ± 0.001 | 0.041 ± 0.029 | 0.017 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 2.839 ± 0.234 | - | 2.691 ± 0.486 | 3.537 ± 0.714 | 2.464 ± 0.420 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.002 | 0.079 ± 0.045 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.090 ± 0.001 | - | 0.101 ± 0.009 | 0.314 ± 0.140 | 0.341 ± 0.076 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.008 ± 0.000 | - | 0.021 ± 0.001 | 0.073 ± 0.003 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.012 ± 0.001 | - | 0.013 ± 0.000 | 0.087 ± 0.051 | 0.025 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.153 ± 0.020 | - | 0.122 ± 0.009 | 0.109 ± 0.009 | 0.187 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.169 ± 0.012 | - | 0.248 ± 0.065 | 0.166 ± 0.004 | 0.250 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.341 ± 0.130 | - | 0.343 ± 0.040 | 0.354 ± 0.010 | 0.711 ± 0.502 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.052 ± 0.004 | - | 0.140 ± 0.003 | 0.064 ± 0.001 | 0.117 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.029 ± 0.029 | 0.010 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.004 ± 0.000 | - | 0.008 ± 0.002 | 0.013 ± 0.000 | 0.034 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.011 ± 0.002 | 0.010 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.005 ± 0.029 | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.004 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.572 ± 0.584 | - | 0.526 ± 0.691 | 0.398 ± 0.007 | 0.621 ± 0.082 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.001 | 0.010 ± 0.001 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.027 ± 0.000 | - | 0.032 ± 0.000 | 0.037 ± 0.000 | 0.098 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.035 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.044 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.152 ± 0.025 | - | 0.104 ± 0.031 | 0.181 ± 0.102 | 0.246 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.186 ± 0.029 | - | 0.202 ± 0.041 | 0.286 ± 0.088 | 0.308 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 1.045 ± 0.113 | - | 0.382 ± 0.038 | 0.467 ± 0.073 | 0.581 ± 0.067 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.068 ± 0.111 | - | 0.159 ± 0.002 | 0.066 ± 0.001 | 0.165 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.002 ± 0.003 | - | 0.009 ± 0.017 | 0.010 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.014 ± 0.001 | - | 0.029 ± 0.001 | 0.013 ± 0.003 | 0.078 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.002 | 0.011 ± 0.002 | 0.010 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.003 ± 0.015 | - | 0.007 ± 0.001 | 0.010 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.001 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.007 ± 0.019 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.031 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.022 ± 0.001 | 0.011 ± 0.002 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.004 | 0.009 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.001 | 0.011 ± 0.003 | 0.011 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.010 ± 0.014 | - | 0.024 ± 0.001 | 0.012 ± 0.003 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.636 ± 0.494 | - | 0.981 ± 0.754 | 0.531 ± 0.209 | 0.747 ± 0.059 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.008 ± 0.037 | 0.091 ± 0.050 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.104 ± 0.051 | - | 0.106 ± 0.013 | 0.115 ± 0.044 | 0.164 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.003 ± 0.001 | - | 0.013 ± 0.018 | 0.043 ± 0.061 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.012 ± 0.000 | - | 0.027 ± 0.000 | 0.041 ± 0.049 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.152 ± 0.015 | - | 1.159 ± 0.013 | 1.376 ± 0.229 | 4.225 ± 0.858 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 4.267 ± 0.018 | - | 4.309 ± 0.037 | 4.741 ± 0.527 | 7.052 ± 0.315 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 16.713 ± 0.525 | - | 16.662 ± 0.708 | 16.789 ± 0.424 | 19.036 ± 1.265 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 5.878 ± 0.335 | - | 6.191 ± 0.624 | 4.999 ± 0.041 | 5.856 ± 1.210 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.188 ± 0.042 | - | 0.192 ± 0.037 | 0.193 ± 0.041 | 0.299 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.020 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.172 ± 0.051 | - | 0.199 ± 0.040 | 0.180 ± 0.010 | 0.228 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.017 ± 0.000 | - | 0.018 ± 0.000 | 0.022 ± 0.001 | 0.030 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.024 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.014 ± 0.003 | - | 0.015 ± 0.000 | 0.019 ± 0.000 | 0.032 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.045 ± 0.000 | - | 0.046 ± 0.000 | 0.050 ± 0.007 | 0.067 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.007 ± 0.002 | - | 0.009 ± 0.002 | 0.012 ± 0.003 | 0.030 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.018 ± 0.000 | - | 0.019 ± 0.000 | 0.023 ± 0.006 | 0.042 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.061 ± 0.001 | - | 0.062 ± 0.000 | 0.065 ± 0.000 | 0.143 ± 0.165 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 51.381 ± 5.500 | - | 51.077 ± 1.350 | 39.598 ± 1.349 | 43.632 ± 4.794 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.011 ± 0.000 | - | 0.012 ± 0.000 | 0.015 ± 0.004 | 0.040 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.547 ± 0.038 | - | 1.547 ± 0.080 | 1.443 ± 0.175 | 2.231 ± 0.788 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.034 ± 0.000 | - | 0.036 ± 0.000 | 0.042 ± 0.007 | 0.059 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.116 ± 0.001 | - | 0.118 ± 0.003 | 0.121 ± 0.000 | 0.220 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.290 ± 0.127 | - | 1.210 ± 0.084 | 1.336 ± 0.061 | 3.736 ± 0.434 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 4.582 ± 0.238 | - | 4.878 ± 0.197 | 5.516 ± 1.275 | 5.504 ± 1.321 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 16.848 ± 0.788 | - | 16.436 ± 0.627 | 16.634 ± 1.781 | 10.480 ± 2.123 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 6.438 ± 0.189 | - | 6.470 ± 0.484 | 5.025 ± 0.165 | 6.082 ± 0.633 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.004 ± 0.002 | - | 0.007 ± 0.001 | 0.007 ± 0.002 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.200 ± 0.097 | - | 0.201 ± 0.061 | 0.214 ± 0.113 | 0.308 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.007 ± 0.003 | - | 0.008 ± 0.000 | 0.011 ± 0.001 | 0.027 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.168 ± 0.003 | - | 0.190 ± 0.048 | 0.180 ± 0.041 | 0.330 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.018 ± 0.001 | - | 0.020 ± 0.004 | 0.021 ± 0.001 | 0.033 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.021 ± 0.030 | - | 0.016 ± 0.000 | 0.022 ± 0.004 | 0.032 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.046 ± 0.002 | - | 0.051 ± 0.006 | 0.050 ± 0.001 | 0.069 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.001 | 0.010 ± 0.000 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.019 ± 0.000 | - | 0.020 ± 0.000 | 0.022 ± 0.000 | 0.074 ± 0.075 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.066 ± 0.008 | - | 0.071 ± 0.051 | 0.066 ± 0.004 | 0.082 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 52.145 ± 1.631 | - | 61.250 ± 3.544 | 40.155 ± 1.601 | 24.353 ± 1.626 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.012 ± 0.000 | - | 0.013 ± 0.004 | 0.015 ± 0.003 | 0.095 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 1.717 ± 0.371 | - | 1.687 ± 0.230 | 1.632 ± 0.264 | 1.549 ± 0.230 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.035 ± 0.000 | - | 0.037 ± 0.000 | 0.039 ± 0.006 | 0.062 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.167 ± 0.109 | - | 0.135 ± 0.030 | 0.135 ± 0.054 | 0.273 ± 0.064 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.580 ± 0.025 | - | 0.601 ± 0.046 | 0.723 ± 0.329 | 2.298 ± 0.117 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 2.750 ± 1.539 | - | 3.219 ± 2.044 | 3.130 ± 0.081 | 4.972 ± 0.450 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 10.567 ± 0.042 | - | 11.884 ± 1.806 | 12.467 ± 1.398 | 13.934 ± 0.464 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.512 ± 0.066 | - | 1.501 ± 0.057 | 1.642 ± 0.076 | 1.948 ± 0.162 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.003 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.121 ± 0.004 | - | 0.121 ± 0.004 | 0.129 ± 0.029 | 0.208 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.001 | 0.009 ± 0.044 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.096 ± 0.003 | - | 0.095 ± 0.001 | 0.119 ± 0.030 | 0.140 ± 0.135 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.012 ± 0.000 | - | 0.013 ± 0.000 | 0.017 ± 0.005 | 0.029 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.012 ± 0.001 | - | 0.011 ± 0.000 | 0.014 ± 0.000 | 0.033 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.029 ± 0.000 | - | 0.033 ± 0.007 | 0.034 ± 0.001 | 0.046 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.002 | 0.020 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.013 ± 0.002 | - | 0.013 ± 0.000 | 0.017 ± 0.001 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.040 ± 0.000 | - | 0.041 ± 0.000 | 0.045 ± 0.006 | 0.060 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 11.865 ± 0.075 | - | 11.864 ± 0.505 | 13.363 ± 1.150 | 13.985 ± 0.523 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.008 ± 0.000 | - | 0.007 ± 0.002 | 0.012 ± 0.003 | 0.030 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 2.346 ± 1.258 | - | 1.062 ± 0.129 | 1.048 ± 0.025 | 1.269 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.025 ± 0.008 | - | 0.025 ± 0.000 | 0.033 ± 0.008 | 0.044 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.078 ± 0.001 | - | 0.080 ± 0.001 | 0.083 ± 0.000 | 0.150 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.744 ± 0.182 | - | 0.702 ± 0.173 | 0.656 ± 0.034 | 2.183 ± 0.156 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 3.574 ± 1.807 | - | 3.321 ± 0.751 | 3.051 ± 0.269 | 3.618 ± 0.155 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 15.890 ± 4.600 | - | 12.063 ± 3.441 | 12.180 ± 0.925 | 10.721 ± 4.564 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 1.822 ± 0.262 | - | 2.057 ± 1.293 | 1.668 ± 0.212 | 2.279 ± 2.358 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.010 ± 0.001 | - | 0.015 ± 0.000 | 0.007 ± 0.001 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.134 ± 0.049 | - | 0.133 ± 0.029 | 0.135 ± 0.042 | 0.554 ± 0.203 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.007 ± 0.001 | - | 0.006 ± 0.000 | 0.010 ± 0.002 | 0.017 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.096 ± 0.010 | - | 0.097 ± 0.013 | 0.100 ± 0.050 | 0.344 ± 0.234 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.013 ± 0.001 | - | 0.014 ± 0.000 | 0.017 ± 0.004 | 0.026 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.011 ± 0.004 | - | 0.014 ± 0.003 | 0.015 ± 0.003 | 0.029 ± 0.058 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.030 ± 0.000 | - | 0.031 ± 0.004 | 0.035 ± 0.007 | 0.045 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.014 ± 0.001 | - | 0.006 ± 0.001 | 0.010 ± 0.002 | 0.054 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.013 ± 0.005 | - | 0.015 ± 0.004 | 0.016 ± 0.000 | 0.030 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.051 ± 0.004 | - | 0.059 ± 0.050 | 0.051 ± 0.009 | 0.188 ± 0.174 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 13.076 ± 0.149 | - | 13.534 ± 0.975 | 13.352 ± 0.327 | 10.078 ± 1.510 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.019 ± 0.000 | - | 0.026 ± 0.000 | 0.010 ± 0.000 | 0.028 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 1.062 ± 0.262 | - | 1.071 ± 0.226 | 1.074 ± 0.235 | 1.078 ± 0.470 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.024 ± 0.004 | - | 0.025 ± 0.001 | 0.027 ± 0.003 | 0.048 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.214 ± 0.066 | - | 0.257 ± 0.034 | 0.089 ± 0.009 | 0.199 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.227 ± 0.038 | - | unsupported | 0.238 ± 0.085 | 1.323 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.577 ± 0.114 | - | unsupported | 1.565 ± 0.078 | 2.876 ± 2.169 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 4.567 ± 0.147 | - | unsupported | 4.197 ± 0.754 | 5.251 ± 0.311 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.834 ± 0.130 | - | unsupported | 0.837 ± 0.078 | 0.791 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.039 ± 0.007 | - | unsupported | 0.050 ± 0.020 | 0.046 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.002 ± 0.001 | - | unsupported | 0.007 ± 0.045 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.037 ± 0.004 | - | unsupported | 0.036 ± 0.007 | 0.041 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.007 ± 0.001 | - | unsupported | 0.009 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | unsupported | 0.009 ± 0.002 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.013 ± 0.003 | - | unsupported | 0.016 ± 0.001 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.006 ± 0.001 | - | unsupported | 0.010 ± 0.002 | 0.020 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.017 ± 0.003 | - | unsupported | 0.022 ± 0.005 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 9.576 ± 2.743 | - | unsupported | 6.903 ± 0.304 | 6.388 ± 0.171 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.008 ± 0.002 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.311 ± 0.026 | - | unsupported | 0.339 ± 0.193 | 0.315 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.014 ± 0.001 | - | unsupported | 0.015 ± 0.000 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.033 ± 0.005 | - | unsupported | 0.035 ± 0.015 | 0.048 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.323 ± 0.024 | - | unsupported | 0.239 ± 0.036 | 1.756 ± 0.507 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 1.992 ± 0.847 | - | unsupported | 1.520 ± 0.017 | 1.846 ± 0.264 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 4.822 ± 1.228 | - | unsupported | 4.449 ± 0.094 | 3.136 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.407 ± 0.494 | - | unsupported | 0.897 ± 0.089 | 0.781 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.010 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.087 ± 0.004 | - | unsupported | 0.043 ± 0.006 | 0.047 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.009 ± 0.001 | - | unsupported | 0.006 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.036 ± 0.044 | - | unsupported | 0.035 ± 0.001 | 0.041 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.007 ± 0.002 | - | unsupported | 0.009 ± 0.004 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.003 ± 0.001 | - | unsupported | 0.005 ± 0.000 | 0.014 ± 0.083 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.015 ± 0.001 | - | unsupported | 0.009 ± 0.002 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.033 ± 0.012 | - | unsupported | 0.016 ± 0.004 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.017 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.042 ± 0.001 | - | unsupported | 0.020 ± 0.004 | 0.030 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 10.485 ± 4.053 | - | unsupported | 7.181 ± 0.381 | 4.462 ± 0.206 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.004 ± 0.001 | - | unsupported | 0.006 ± 0.000 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.409 ± 0.157 | - | unsupported | 0.315 ± 0.053 | 0.276 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.030 ± 0.001 | - | unsupported | 0.015 ± 0.004 | 0.058 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.078 ± 0.004 | - | unsupported | 0.036 ± 0.005 | 0.048 ± 0.011 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.843 ± 0.021 | - | unsupported | 0.803 ± 0.019 | 1.225 ± 0.055 | 3.865 ± 1.424 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.059 ± 0.007 | - | unsupported | 0.091 ± 0.026 | 0.136 ± 0.003 | 0.079 ± 0.014 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.115 ± 0.002 | - | unsupported | 0.129 ± 0.017 | 0.185 ± 0.020 | 0.222 ± 0.084 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.270 ± 0.007 | - | unsupported | 0.268 ± 0.010 | 0.438 ± 0.376 | 0.769 ± 0.061 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 1.205 ± 0.202 | - | unsupported | 0.875 ± 0.265 | 1.281 ± 0.113 | 3.344 ± 1.894 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.074 ± 0.020 | - | unsupported | 0.175 ± 0.041 | 0.176 ± 0.077 | 0.117 ± 0.020 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.135 ± 0.036 | - | unsupported | 0.218 ± 0.098 | 0.251 ± 0.046 | 0.244 ± 0.094 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.322 ± 0.021 | - | unsupported | 0.352 ± 0.024 | 0.589 ± 0.082 | 0.573 ± 0.168 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.223 ± 0.008 | - | unsupported | 0.186 ± 0.009 | 0.550 ± 0.061 | 0.553 ± 0.224 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.147 ± 0.006 | - | unsupported | 0.043 ± 0.001 | 0.114 ± 0.021 | 0.067 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.083 ± 0.012 | - | unsupported | 0.050 ± 0.000 | 0.123 ± 0.026 | 0.110 ± 0.034 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.100 ± 0.001 | - | unsupported | 0.080 ± 0.000 | 0.214 ± 0.037 | 0.325 ± 0.048 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.250 ± 0.041 | - | unsupported | 0.254 ± 0.335 | 0.622 ± 0.153 | 0.466 ± 0.023 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.078 ± 0.029 | - | unsupported | 0.043 ± 0.008 | 0.091 ± 0.013 | 0.059 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.107 ± 0.105 | - | unsupported | 0.058 ± 0.011 | 0.172 ± 0.064 | 0.090 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.113 ± 0.011 | - | unsupported | 0.091 ± 0.058 | 0.222 ± 0.052 | 0.218 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.121 ± 0.143 | - | 0.114 ± 0.118 | 0.087 ± 0.002 | 0.227 ± 0.022 | 0.300 ± 0.038 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.039 ± 0.000 | - | 0.041 ± 0.002 | 0.024 ± 0.002 | 0.027 ± 0.002 | 0.037 ± 0.008 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.049 ± 0.009 | - | 0.046 ± 0.001 | 0.027 ± 0.018 | 0.032 ± 0.006 | 0.059 ± 0.006 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.060 ± 0.001 | - | 0.067 ± 0.013 | 0.036 ± 0.008 | 0.064 ± 0.007 | 0.139 ± 0.036 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.126 ± 0.012 | - | 0.112 ± 0.009 | 0.106 ± 0.033 | 0.256 ± 0.013 | 0.285 ± 0.043 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.053 ± 0.006 | - | 0.067 ± 0.047 | 0.034 ± 0.032 | 0.026 ± 0.003 | 0.037 ± 0.008 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.059 ± 0.001 | - | 0.059 ± 0.009 | 0.072 ± 0.025 | 0.031 ± 0.001 | 0.079 ± 0.068 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.072 ± 0.095 | - | 0.071 ± 0.017 | 0.053 ± 0.085 | 0.082 ± 0.072 | 0.109 ± 0.054 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 10.508 ± 1.071 | - | 10.656 ± 0.238 | 26.687 ± 3.137 | 23.034 ± 5.680 | 48.843 ± 3.455 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 12.216 ± 0.600 | - | 12.159 ± 1.614 | 26.291 ± 3.143 | 15.232 ± 0.393 | 391.964 ± 294.908 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 8.619 ± 0.684 | - | 8.798 ± 0.787 | 8.995 ± 2.352 | 9.135 ± 2.129 | 28.095 ± 12.652 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 10.022 ± 1.371 | - | 9.492 ± 2.404 | 10.324 ± 1.536 | 9.203 ± 0.771 | 238.513 ± 37.601 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 7.526 ± 0.314 | - | 7.455 ± 0.197 | 7.647 ± 0.413 | 11.122 ± 1.372 | 9.383 ± 3.386 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 6.798 ± 0.101 | - | 8.226 ± 0.580 | 8.483 ± 2.120 | 8.101 ± 0.128 | 297.254 ± 6.603 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 8.448 ± 0.448 | - | 8.496 ± 0.095 | 8.759 ± 0.439 | 8.556 ± 1.011 | 9.155 ± 4.139 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 8.219 ± 1.137 | - | 7.704 ± 0.062 | 7.959 ± 3.341 | 7.200 ± 0.443 | 252.737 ± 16.009 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 22.364 ± 2.378 | - | 22.971 ± 3.318 | 23.144 ± 5.556 | 31.576 ± 6.764 | 13.691 ± 5.810 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 22.574 ± 1.128 | - | 22.136 ± 0.279 | 24.897 ± 4.095 | 29.601 ± 2.414 | 711.407 ± 595.819 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.311 ± 0.008 | 0.325 ± 0.085 | unsupported | 0.419 ± 0.194 | 0.402 ± 0.051 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 4.060 ± 0.274 | 4.127 ± 0.461 | unsupported | 5.405 ± 1.145 | 5.116 ± 1.040 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.273 ± 0.007 | 0.284 ± 0.008 | unsupported | 0.365 ± 0.223 | 0.564 ± 0.079 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 3.757 ± 0.452 | 3.896 ± 0.127 | unsupported | 3.885 ± 1.325 | 5.099 ± 0.543 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 10.942 ± 3.211 | - | 11.397 ± 4.076 | 19.049 ± 5.838 | 10.412 ± 0.948 | 35.651 ± 15.727 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 9.417 ± 0.089 | - | 10.213 ± 0.802 | 12.588 ± 4.600 | 12.360 ± 2.474 | 412.364 ± 69.058 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 10.718 ± 6.489 | 17.015 ± 6.977 | 5.471 ± 0.356 | 36.689 ± 5.260 | 9.314 ± 0.990 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 10.458 ± 0.595 | 10.240 ± 0.419 | 11.021 ± 6.348 | 35.697 ± 5.207 | 139.333 ± 35.667 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 14.957 ± 3.215 | - | 10.357 ± 3.580 | 17.479 ± 4.070 | 15.734 ± 3.078 | 29.511 ± 25.086 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 9.781 ± 2.004 | - | 9.129 ± 0.342 | 12.003 ± 5.790 | 14.001 ± 4.469 | 259.039 ± 44.754 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 1.330 ± 1.108 | - | 0.959 ± 0.083 | 2.188 ± 0.013 | 6.477 ± 0.725 | 5.725 ± 0.123 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.866 ± 0.076 | - | 0.856 ± 0.020 | 2.177 ± 0.032 | 5.135 ± 0.310 | 9.166 ± 5.018 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 14.186 ± 3.303 | - | 12.900 ± 5.382 | 19.162 ± 5.803 | 13.962 ± 0.385 | 24.287 ± 4.231 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 13.167 ± 1.584 | - | 12.440 ± 2.462 | 14.914 ± 4.127 | 17.129 ± 2.336 | 1081.306 ± 531.161 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 14.914 ± 4.123 | - | 16.736 ± 9.226 | 19.422 ± 5.078 | 14.226 ± 0.391 | 23.374 ± 0.836 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 11.394 ± 0.907 | - | 12.262 ± 3.158 | 11.427 ± 0.423 | 14.655 ± 0.384 | 1050.938 ± 186.162 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.575 ± 0.109 | 1.606 ± 0.099 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 14.900 ± 1.893 | 13.937 ± 0.905 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.376 ± 0.088 | 1.441 ± 0.066 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 16.016 ± 1.021 | 14.670 ± 0.289 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.633 ± 0.129 | 1.607 ± 0.082 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 13.490 ± 0.390 | 13.363 ± 1.917 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.480 ± 0.019 | 1.484 ± 0.049 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 12.274 ± 0.658 | 12.758 ± 0.854 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 9.877 ± 1.366 | - | 13.557 ± 10.749 | 11.298 ± 3.718 | 10.095 ± 1.026 | 30.425 ± 4.426 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 8.227 ± 0.102 | - | 9.088 ± 1.848 | 8.999 ± 2.948 | 9.304 ± 0.860 | 253.173 ± 25.351 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 42.147 ± 8.461 | 37.963 ± 2.684 | 36.529 ± 3.590 | 39.478 ± 4.393 | 73.519 ± 6.154 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 47.350 ± 9.663 | 37.513 ± 5.615 | 42.366 ± 4.989 | 42.918 ± 1.750 | 1798.947 ± 547.795 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 15.554 ± 0.238 | - | 16.022 ± 1.548 | 23.644 ± 11.733 | 66.722 ± 6.301 | 37.554 ± 3.788 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 16.241 ± 1.388 | - | 17.768 ± 4.115 | 19.048 ± 7.141 | 66.348 ± 17.799 | 25.154 ± 3.445 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.213 ± 0.052 | 0.210 ± 0.023 | unsupported | 0.061 ± 0.009 | 0.041 ± 0.009 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.986 ± 0.947 | 2.843 ± 0.077 | unsupported | 1.004 ± 0.865 | 0.495 ± 0.022 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.251 ± 0.076 | 0.222 ± 0.030 | unsupported | 0.123 ± 0.056 | 0.088 ± 0.020 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 2.166 ± 0.194 | 2.025 ± 0.103 | unsupported | 0.766 ± 0.212 | 0.707 ± 0.491 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.089 ± 0.001 | 0.093 ± 0.009 | unsupported | 0.097 ± 0.001 | 0.193 ± 0.005 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.254 ± 0.013 | 1.256 ± 0.057 | unsupported | 1.367 ± 0.007 | 2.569 ± 0.463 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.119 ± 0.031 | 0.108 ± 0.059 | unsupported | 0.162 ± 0.009 | 0.274 ± 0.010 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.995 ± 0.112 | 0.987 ± 0.257 | unsupported | 1.290 ± 0.412 | 1.870 ± 0.114 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.297 ± 0.009 | 0.298 ± 0.011 | unsupported | 0.346 ± 0.013 | 0.097 ± 0.003 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 4.205 ± 0.127 | 4.204 ± 0.011 | unsupported | 4.743 ± 0.015 | 0.992 ± 0.070 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.370 ± 0.238 | 0.279 ± 0.046 | unsupported | 0.353 ± 0.089 | 0.162 ± 0.032 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 3.032 ± 0.260 | 2.822 ± 0.318 | unsupported | 3.144 ± 0.220 | 0.943 ± 0.024 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.053 ± 0.001 | 0.075 ± 0.008 | unsupported | 0.052 ± 0.004 | 0.088 ± 0.002 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.671 ± 0.090 | 0.726 ± 0.030 | unsupported | 0.555 ± 0.008 | 0.918 ± 0.159 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.051 ± 0.001 | 0.072 ± 0.001 | unsupported | 0.115 ± 0.015 | 0.153 ± 0.020 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 1.472 ± 0.700 | 1.431 ± 0.092 | unsupported | 0.454 ± 0.038 | 1.148 ± 0.308 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.104 ± 0.093 | 0.122 ± 0.003 | unsupported | 0.118 ± 0.016 | 0.114 ± 0.009 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.216 ± 0.065 | 1.288 ± 0.090 | unsupported | 1.452 ± 0.328 | 1.456 ± 0.223 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.098 ± 0.015 | 0.112 ± 0.002 | unsupported | 0.156 ± 0.014 | 0.235 ± 0.041 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 2.079 ± 0.318 | 2.128 ± 0.244 | unsupported | 1.048 ± 0.149 | 1.665 ± 0.694 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.148 ± 0.005 | 0.168 ± 0.015 | unsupported | unsupported | 0.134 ± 0.004 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 1.820 ± 0.068 | 1.856 ± 0.047 | unsupported | unsupported | 2.048 ± 0.226 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.129 ± 0.003 | 0.150 ± 0.001 | unsupported | unsupported | 0.249 ± 0.038 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 2.531 ± 0.185 | 2.592 ± 0.251 | unsupported | unsupported | 2.457 ± 1.004 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.013 ± 0.002 | 0.020 ± 0.018 | unsupported | 0.009 ± 0.002 | 0.061 ± 0.038 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.105 ± 0.004 | 0.110 ± 0.003 | unsupported | 0.061 ± 0.008 | 0.247 ± 0.014 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.013 ± 0.000 | 0.018 ± 0.000 | unsupported | 0.050 ± 0.016 | 0.093 ± 0.009 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.203 ± 0.036 | 0.373 ± 0.094 | unsupported | 0.178 ± 0.068 | 0.529 ± 0.501 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.031 ± 0.005 | 0.045 ± 0.004 | unsupported | 0.030 ± 0.004 | 0.082 ± 0.004 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.389 ± 0.091 | 0.396 ± 0.020 | unsupported | 0.349 ± 0.061 | 0.811 ± 0.091 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.028 ± 0.001 | 0.040 ± 0.001 | unsupported | 0.169 ± 0.040 | 0.149 ± 0.023 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.657 ± 0.075 | 0.691 ± 0.053 | unsupported | 0.622 ± 0.083 | 1.059 ± 0.166 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.083 ± 0.064 | 0.058 ± 0.031 | unsupported | 0.090 ± 0.008 | 0.101 ± 0.009 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.767 ± 0.079 | 0.767 ± 0.003 | unsupported | 1.229 ± 0.125 | 0.861 ± 0.021 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.062 ± 0.037 | 0.080 ± 0.039 | unsupported | 0.137 ± 0.006 | 0.159 ± 0.016 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.786 ± 0.178 | 0.689 ± 0.039 | unsupported | 1.011 ± 0.085 | 0.802 ± 0.130 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.060 ± 0.013 | 0.058 ± 0.005 | unsupported | 0.086 ± 0.011 | 0.104 ± 0.010 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.799 ± 0.003 | 0.799 ± 0.003 | unsupported | 1.273 ± 0.023 | 1.299 ± 0.722 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.065 ± 0.039 | 0.088 ± 0.015 | unsupported | 0.148 ± 0.016 | 0.165 ± 0.023 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 1.621 ± 1.537 | 0.738 ± 0.229 | unsupported | 1.070 ± 0.125 | 0.884 ± 0.009 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.141 ± 0.068 | 0.122 ± 0.010 | unsupported | 0.109 ± 0.006 | 0.168 ± 0.162 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.197 ± 0.020 | 1.330 ± 0.332 | unsupported | 1.656 ± 0.003 | 1.284 ± 0.097 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.095 ± 0.016 | 0.112 ± 0.013 | unsupported | 0.167 ± 0.025 | 0.221 ± 0.021 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 2.038 ± 0.200 | 2.397 ± 0.343 | unsupported | 1.316 ± 0.291 | 1.381 ± 0.957 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.248 ± 0.007 | 0.234 ± 0.059 | unsupported | 0.318 ± 0.019 | 0.338 ± 0.056 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 3.508 ± 0.012 | 3.751 ± 0.568 | unsupported | 4.445 ± 0.009 | 4.456 ± 0.357 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.225 ± 0.093 | 0.287 ± 0.101 | unsupported | 0.307 ± 0.014 | 0.367 ± 0.077 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 2.612 ± 0.079 | 2.677 ± 0.144 | unsupported | 3.591 ± 0.498 | 3.765 ± 0.184 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.635 ± 0.055 | 0.754 ± 0.071 | unsupported | 0.131 ± 0.012 | 0.118 ± 0.028 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 10.987 ± 0.344 | 12.442 ± 0.984 | unsupported | 2.205 ± 0.015 | 1.604 ± 0.093 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.814 ± 0.337 | 0.945 ± 0.169 | unsupported | 0.268 ± 0.026 | 0.172 ± 0.036 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 12.034 ± 1.554 | 10.904 ± 1.421 | unsupported | 1.916 ± 0.653 | 1.705 ± 0.365 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 23.611 ± 3.248 | - | unsupported | 19.065 ± 1.302 | - | 22.381 ± 4.375 | 28.727 ± 6.281 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 21.328 ± 1.759 | - | unsupported | 26.567 ± 3.227 | - | 25.847 ± 7.544 | 27.788 ± 4.749 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 13.715 ± 2.737 | - | unsupported | 11.565 ± 0.439 | - | 12.204 ± 1.051 | 16.106 ± 4.203 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 13.346 ± 1.167 | - | unsupported | 13.265 ± 0.901 | - | 12.184 ± 2.236 | 13.001 ± 0.504 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 13.197 ± 3.116 | - | unsupported | 11.403 ± 0.087 | - | 12.974 ± 1.024 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 12.494 ± 1.646 | - | unsupported | 13.661 ± 1.810 | - | 15.421 ± 2.035 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 23.829 ± 3.469 | - | unsupported | 18.975 ± 2.917 | - | 19.873 ± 2.178 | 22.865 ± 7.840 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 22.285 ± 2.392 | - | unsupported | 23.826 ± 2.768 | - | 30.827 ± 7.210 | 21.551 ± 1.653 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 13.175 ± 3.362 | - | unsupported | 10.796 ± 0.638 | - | 72.840 ± 6.784 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 13.830 ± 2.151 | - | unsupported | 12.276 ± 0.505 | - | 34.376 ± 6.623 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 14.301 ± 2.933 | - | unsupported | 11.271 ± 0.136 | - | 72.285 ± 6.581 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 13.771 ± 1.204 | - | unsupported | 14.972 ± 4.101 | - | 26.108 ± 10.434 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 27.179 ± 6.946 | - | unsupported | 18.656 ± 2.153 | - | 18.685 ± 1.237 | 20.412 ± 1.717 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 21.748 ± 2.427 | - | unsupported | 20.592 ± 1.424 | - | 21.730 ± 4.708 | 19.151 ± 2.475 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 16.758 ± 4.972 | - | unsupported | 11.487 ± 0.542 | - | 13.794 ± 3.210 | 13.353 ± 0.352 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 13.416 ± 1.583 | - | unsupported | 15.119 ± 1.054 | - | 14.859 ± 1.937 | 12.885 ± 3.006 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.244 ± 0.059 | unsupported | unsupported | 0.575 ± 0.286 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 3.098 ± 0.073 | unsupported | unsupported | 4.658 ± 1.671 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.216 ± 0.004 | unsupported | unsupported | 0.326 ± 0.025 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 3.371 ± 0.227 | unsupported | unsupported | 3.303 ± 0.571 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 23.264 ± 7.349 | - | unsupported | 24.403 ± 6.660 | - | 22.468 ± 6.237 | 19.203 ± 2.358 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 21.536 ± 5.508 | - | unsupported | 21.308 ± 4.117 | - | 24.532 ± 3.843 | 20.380 ± 0.930 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 10.859 ± 1.886 | - | 0.004 ± 0.000 | 6.358 ± 0.167 | 5.711 ± 0.519 | 18.927 ± 5.959 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 8.664 ± 0.818 | - | 0.001 ± 0.000 | 7.841 ± 1.822 | 5.910 ± 0.503 | 13.546 ± 0.494 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 11.378 ± 2.753 | - | 23.145 ± 5.927 | 11.334 ± 0.681 | 12.055 ± 2.948 | 12.046 ± 2.238 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 8.685 ± 0.202 | - | 19.583 ± 0.634 | 10.985 ± 1.753 | 11.431 ± 0.715 | 10.496 ± 2.285 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 20.362 ± 2.831 | - | 19.858 ± 2.784 | 13.601 ± 2.919 | 17.761 ± 3.464 | 11.200 ± 1.858 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 10.423 ± 1.005 | - | 10.299 ± 0.417 | 12.498 ± 1.244 | 14.696 ± 0.574 | 14.306 ± 2.197 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 7.259 ± 1.428 | - | 6.337 ± 0.126 | 11.923 ± 1.714 | 11.055 ± 0.976 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 4.924 ± 0.319 | - | 5.149 ± 0.182 | 11.393 ± 1.161 | 10.742 ± 0.297 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 13.614 ± 2.166 | - | 0.001 ± 0.002 | 13.378 ± 0.788 | 14.212 ± 1.900 | 12.854 ± 1.488 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 11.469 ± 1.006 | - | 0.001 ± 0.000 | 13.526 ± 1.410 | 11.247 ± 1.164 | 9.508 ± 0.271 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 46.633 ± 3.928 | - | 0.001 ± 0.007 | 47.448 ± 5.831 | 58.910 ± 3.251 | 200.901 ± 50.752 | 47.346 ± 12.083 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 19.424 ± 1.231 | - | 0.001 ± 0.000 | 52.227 ± 17.242 | 32.120 ± 1.495 | 214.772 ± 37.316 | 19.345 ± 1.758 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 10.719 ± 1.814 | - | 13.452 ± 6.594 | 9.218 ± 0.765 | 7.298 ± 1.798 | 11.017 ± 133.672 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 5.562 ± 0.562 | - | 6.217 ± 0.474 | 7.505 ± 0.666 | 7.754 ± 2.696 | 10.309 ± 0.858 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 12.734 ± 2.684 | - | 12.149 ± 4.677 | 8.031 ± 0.168 | 7.382 ± 4.326 | 13.574 ± 8.667 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 5.685 ± 0.603 | - | 5.477 ± 0.345 | 6.909 ± 0.393 | 6.506 ± 0.256 | 10.618 ± 0.794 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.005 ± 0.007 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.003 ± 0.002 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 7.915000 | 494687.50 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 4.235375 | 264710.94 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 4.801292 | 300080.75 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 7.633750 | 477109.38 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 4.134750 | 258421.88 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 4.545666 | 284104.12 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 15.107083 | 944192.69 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 13.175959 | 823497.44 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 13.094166 | 818385.38 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 14.543291 | 908955.69 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 6.434417 | 402151.06 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 6.702042 | 418877.62 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 4.983125 | 311445.31 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 5.196708 | 324794.25 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 81.848708 | 5115544.25 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 86.484375 | 5405273.44 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 64.963417 | 4060213.56 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 66.033416 | 4127088.50 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 9.020250 | 563765.62 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 5.843958 | 365247.38 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 4.362833 | 272677.06 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 4.544875 | 284054.69 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 81.582625 | 5098914.06 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 62.152834 | 3884552.12 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 60.109208 | 3756825.50 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 62.329625 | 3895601.56 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 25.202042 | 1575127.62 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 25.692584 | 1605786.50 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 238.406959 | 14900434.94 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 222.989959 | 13936872.44 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 22.013500 | 1375843.75 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 23.053292 | 1440830.75 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 256.253500 | 16015843.75 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 234.719041 | 14669940.06 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 26.123083 | 1632692.69 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 25.716125 | 1607257.81 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 215.843334 | 13490208.38 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 213.815750 | 13363484.38 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 23.681125 | 1480070.31 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 23.736208 | 1483513.00 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 196.386833 | 12274177.06 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 204.121084 | 12757567.75 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.612166 | 40815.09 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.898417 | 60912.77 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 13.649250 | 213269.53 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 13.426875 | 209794.92 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 3.958000 | 494750.00 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 8.030333 | 1003791.63 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 23.889042 | 2986130.25 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 22.740417 | 2842552.12 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.606792 | 87606.12 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 7.867875 | 122935.55 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 16.040667 | 250635.42 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 14.203042 | 221922.53 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.659583 | 707447.88 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.126750 | 765843.75 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 17.331125 | 2166390.62 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 16.198791 | 2024848.88 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.354208 | 193034.50 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.230584 | 97352.88 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.678083 | 88720.05 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.933583 | 92712.23 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 20.555000 | 2569375.00 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.934542 | 1366817.75 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.030208 | 1253776.00 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.049791 | 1256223.88 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 17.564166 | 274440.09 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 10.384167 | 162252.61 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 7.621875 | 119091.80 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6.890041 | 107656.89 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 14.960541 | 1870067.62 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 10.322791 | 1290348.88 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.956542 | 994567.75 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.892042 | 986505.25 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.200833 | 96888.02 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 22.171792 | 346434.25 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 19.015500 | 297117.19 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 19.075292 | 298051.44 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.938584 | 992323.00 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 37.943167 | 4742895.88 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 33.639417 | 4204927.12 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 33.632541 | 4204067.62 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.341083 | 161579.42 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 22.603667 | 353182.30 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 23.652750 | 369574.22 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 17.848500 | 278882.81 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.545500 | 943187.50 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 25.150375 | 3143796.88 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 24.257542 | 3032192.75 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 22.574208 | 2821776.00 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.658458 | 88413.41 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.321917 | 51904.95 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.395583 | 53055.98 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.813541 | 75211.58 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.341750 | 917718.75 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.442875 | 555359.38 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.365250 | 670656.25 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.804208 | 725526.00 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.777625 | 152775.39 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 7.340791 | 114699.86 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.287333 | 51364.58 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.588667 | 71697.92 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.180083 | 1147510.38 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.633625 | 454203.12 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 11.777083 | 1472135.38 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 11.451875 | 1431484.38 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.268667 | 113572.92 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.573833 | 118341.14 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.660208 | 104065.75 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 7.796042 | 121813.16 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.650209 | 1456276.12 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.616875 | 1452109.38 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.730125 | 1216265.62 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.301083 | 1287635.38 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 15.067916 | 235436.19 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 9.969666 | 155776.03 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.280875 | 98138.67 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.153250 | 111769.53 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 13.318166 | 1664770.75 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.381750 | 1047718.75 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 16.631125 | 2078890.62 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 17.021083 | 2127635.38 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.564916 | 133826.81 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 9.487125 | 148236.33 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 10.733209 | 167706.39 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 16.384791 | 2048098.88 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 14.561542 | 1820192.75 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 14.847250 | 1855906.25 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 15.953791 | 249277.98 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.271042 | 129235.03 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 9.597834 | 149966.16 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 19.658208 | 2457276.00 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 20.249042 | 2531130.25 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 20.736250 | 2592031.25 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 3.924834 | 61325.53 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.572459 | 8944.67 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.845250 | 13207.03 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.285708 | 20089.19 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 1.975750 | 246968.75 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.487834 | 60979.25 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.840583 | 105072.88 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.880916 | 110114.50 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.944625 | 92884.77 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.195125 | 49923.83 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.857959 | 13405.61 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.181583 | 18462.23 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.234000 | 529250.00 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.427833 | 178479.12 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.622417 | 202802.12 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.986500 | 373312.50 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.267375 | 82302.73 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 1.920083 | 30001.30 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.975708 | 30870.44 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.849208 | 44518.88 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.490458 | 811307.25 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 2.794334 | 349291.75 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.112708 | 389088.50 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.167458 | 395932.25 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.554500 | 149289.06 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 10.789791 | 168590.48 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.788791 | 27949.86 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.568708 | 40136.06 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.475792 | 1059474.00 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.975542 | 621942.75 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.257208 | 657151.00 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.528875 | 691109.38 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.478750 | 101230.47 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.780917 | 90326.83 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.301667 | 82838.55 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.683375 | 57552.73 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.891291 | 861411.38 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.831250 | 1228906.25 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.139875 | 767484.38 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.137500 | 767187.50 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.166750 | 158855.47 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 8.745625 | 136650.39 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.963500 | 61929.69 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.096000 | 79625.00 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.419166 | 802395.75 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.088666 | 1011083.25 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.288084 | 786010.50 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.514792 | 689349.00 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.645375 | 103833.98 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.498250 | 85910.16 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.842875 | 60044.92 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.709625 | 57962.89 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.389167 | 1298645.88 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.184750 | 1273093.75 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.394959 | 799369.88 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.388292 | 798536.50 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.565291 | 165082.67 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 9.463542 | 147867.84 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.142541 | 64727.20 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.606250 | 87597.66 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.073625 | 884203.12 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.558583 | 1069822.88 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 12.965625 | 1620703.12 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.900167 | 737520.88 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.762709 | 168167.33 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.947792 | 108559.25 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 9.052833 | 141450.52 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 7.836709 | 122448.58 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.275750 | 1284468.75 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 13.247708 | 1655963.50 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.576500 | 1197062.50 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.638833 | 1329854.12 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 14.142917 | 220983.08 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 10.670416 | 166725.25 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.099083 | 95298.17 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.156417 | 111819.02 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.046083 | 1380760.38 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 10.530458 | 1316307.25 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 16.300167 | 2037520.87 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 19.175250 | 2396906.25 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 21.617792 | 337778.00 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 20.323208 | 317550.12 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 15.869875 | 247966.80 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 15.005541 | 234461.58 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 35.644791 | 4455598.88 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 35.563333 | 4445416.62 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 28.063042 | 3507880.25 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 30.006209 | 3750776.12 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 23.471334 | 366739.59 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 19.676250 | 307441.41 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 14.397208 | 224956.38 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 18.349583 | 286712.23 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 30.118666 | 3764833.25 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 28.725084 | 3590635.50 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 20.894208 | 2611776.00 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 21.414542 | 2676817.75 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.578750 | 118417.97 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.408125 | 131376.95 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 40.642458 | 635038.41 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 48.262000 | 754093.75 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.831791 | 1603973.88 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 17.637208 | 2204651.00 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 87.896333 | 10987041.62 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 99.533333 | 12441666.62 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.002666 | 171916.66 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 17.164875 | 268201.17 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 52.125291 | 814457.67 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 60.494625 | 945228.52 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 13.636250 | 1704531.25 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 15.325292 | 1915661.50 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 96.274292 | 12034286.50 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 87.231958 | 10903994.75 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 9.192625 | 574539.06 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 3.905792 | 244112.00 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 74.534166 | 4658385.38 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 49.567292 | 3097955.75 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 5.222459 | 326403.69 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 3.449792 | 215612.00 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 52.843458 | 3302716.12 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 53.929333 | 3370583.31 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.013125 | 820.31 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.001292 | 80.75 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.015833 | 989.56 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.001750 | 109.38 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.012542 | 783.88 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.083708 | 5231.75 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000125 | 7.81 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.013750 | 859.38 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.041875 | 2617.19 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.010916 | 682.25 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.002750 | 171.88 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000125 | 7.81 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.010542 | 658.88 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.005792 | 362.00 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.011667 | 729.19 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002667 | 166.69 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000125 | 7.81 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.011209 | 700.56 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003250 | 203.12 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `jax-cpu` is 46.4x faster than the slowest successful cell (`julia-base`, 71.564 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `pytorch-cpu` is 14.5x faster than the slowest successful cell (`julia-base`, 71.564 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-direct` is 17.3x faster than the slowest successful cell (`julia-base`, 71.564 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`julia-base`, 71.564 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `jax-cpu` is 22.0x faster than the slowest successful cell (`julia-base`, 457.893 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `pytorch-cpu` is 37.5x faster than the slowest successful cell (`julia-base`, 457.893 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-direct` is 31.0x faster than the slowest successful cell (`julia-base`, 457.893 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-trace` is 36.9x faster than the slowest successful cell (`julia-base`, 457.893 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 21.8x faster than the slowest successful cell (`julia-base`, 165.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 20.9x faster than the slowest successful cell (`julia-base`, 165.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-direct` is 17.2x faster than the slowest successful cell (`julia-base`, 165.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-trace` is 17.6x faster than the slowest successful cell (`julia-base`, 165.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 62.3x faster than the slowest successful cell (`julia-base`, 284.512 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 43.8x faster than the slowest successful cell (`julia-base`, 284.512 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-direct` is 34.7x faster than the slowest successful cell (`julia-base`, 284.512 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-trace` is 38.1x faster than the slowest successful cell (`julia-base`, 284.512 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `jax-cpu` is 83.5x faster than the slowest successful cell (`julia-base`, 630.364 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `pytorch-cpu` is 78.1x faster than the slowest successful cell (`julia-base`, 630.364 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-direct` is 80.6x faster than the slowest successful cell (`julia-base`, 630.364 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-trace` is 78.1x faster than the slowest successful cell (`julia-base`, 630.364 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 19.2x faster than the slowest successful cell (`pytorch-cpu`, 17.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 15.2x faster than the slowest successful cell (`pytorch-cpu`, 17.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 18.6x faster than the slowest successful cell (`pytorch-cpu`, 17.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 19.4x faster than the slowest successful cell (`pytorch-cpu`, 12.159 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 13.8x faster than the slowest successful cell (`pytorch-cpu`, 12.159 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 12.159 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 15.3x faster than the slowest successful cell (`pytorch-cpu`, 16.291 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 18.3x faster than the slowest successful cell (`pytorch-cpu`, 16.291 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 18.7x faster than the slowest successful cell (`pytorch-cpu`, 16.291 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 20.3x faster than the slowest successful cell (`pytorch-cpu`, 12.948 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 16.4x faster than the slowest successful cell (`pytorch-cpu`, 12.948 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 14.3x faster than the slowest successful cell (`pytorch-cpu`, 12.948 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sqrt` (f64, threads=4, shape=`33554432`): `jax-cpu` is 21.8x faster than the slowest successful cell (`julia-base`, 329.930 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sqrt` (f64, threads=4, shape=`33554432`): `pytorch-cpu` is 25.9x faster than the slowest successful cell (`julia-base`, 329.930 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sqrt` (f64, threads=4, shape=`33554432`): `tenferro-direct` is 25.7x faster than the slowest successful cell (`julia-base`, 329.930 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/sqrt` (f64, threads=4, shape=`33554432`): `tenferro-trace` is 25.9x faster than the slowest successful cell (`julia-base`, 329.930 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 898.1x faster than the slowest successful cell (`julia-base`, 1.646 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 894.4x faster than the slowest successful cell (`julia-base`, 1.603 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `pytorch-cpu` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 31.6x faster than the slowest successful cell (`jax-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-trace` is 17.2x faster than the slowest successful cell (`jax-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.009 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `pytorch-cpu` is 18.7x faster than the slowest successful cell (`jax-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 56.6x faster than the slowest successful cell (`jax-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 27.9x faster than the slowest successful cell (`jax-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 23.1x faster than the slowest successful cell (`jax-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-trace` is 13.3x faster than the slowest successful cell (`jax-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `jax-cpu` is 20.0x faster than the slowest successful cell (`pytorch-cpu`, 0.164 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 98.3x faster than the slowest successful cell (`pytorch-cpu`, 0.164 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 46.3x faster than the slowest successful cell (`pytorch-cpu`, 0.164 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 40.5x faster than the slowest successful cell (`pytorch-cpu`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-trace` is 24.3x faster than the slowest successful cell (`pytorch-cpu`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `jax-cpu` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 40.0x faster than the slowest successful cell (`pytorch-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `jax-cpu` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 40.2x faster than the slowest successful cell (`pytorch-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 28.2x faster than the slowest successful cell (`pytorch-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `jax-cpu` is 12.3x faster than the slowest successful cell (`pytorch-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 44.5x faster than the slowest successful cell (`pytorch-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `jax-cpu` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 26.0x faster than the slowest successful cell (`pytorch-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 30.1x faster than the slowest successful cell (`pytorch-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 34.4x faster than the slowest successful cell (`pytorch-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 18.1x faster than the slowest successful cell (`pytorch-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 55.8x faster than the slowest successful cell (`jax-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 36.6x faster than the slowest successful cell (`jax-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 15.0x faster than the slowest successful cell (`pytorch-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 17.6x faster than the slowest successful cell (`pytorch-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `jax-cpu` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 20.5x faster than the slowest successful cell (`pytorch-cpu`, 0.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 24.5x faster than the slowest successful cell (`pytorch-cpu`, 0.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=4, shape=`8x2x2`): `tenferro-direct` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`3x2x2`): `tenferro-direct` is 21.0x faster than the slowest successful cell (`jax-cpu`, 0.027 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`3x2x2`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`jax-cpu`, 0.027 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`4x2x2`): `tenferro-direct` is 20.6x faster than the slowest successful cell (`jax-cpu`, 0.029 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`4x2x2`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`jax-cpu`, 0.029 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`8x2x2`): `tenferro-direct` is 10.0x faster than the slowest successful cell (`pytorch-cpu`, 0.018 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 16.1x faster than the slowest successful cell (`jax-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-trace` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 13.3x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `jax-cpu` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.115 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 41.2x faster than the slowest successful cell (`pytorch-cpu`, 0.115 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 29.1x faster than the slowest successful cell (`pytorch-cpu`, 0.115 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 14.1x faster than the slowest successful cell (`jax-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x32x32`): `tenferro-trace` is 11.7x faster than the slowest successful cell (`jax-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 27.9x faster than the slowest successful cell (`pytorch-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 22.0x faster than the slowest successful cell (`pytorch-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 23.7x faster than the slowest successful cell (`pytorch-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 16.8x faster than the slowest successful cell (`pytorch-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `jax-cpu` is 17.9x faster than the slowest successful cell (`pytorch-cpu`, 0.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 54.1x faster than the slowest successful cell (`pytorch-cpu`, 0.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 39.4x faster than the slowest successful cell (`pytorch-cpu`, 0.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `jax-cpu` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 0.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 44.2x faster than the slowest successful cell (`pytorch-cpu`, 0.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 28.2x faster than the slowest successful cell (`pytorch-cpu`, 0.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 15.3x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 33.3x faster than the slowest successful cell (`pytorch-cpu`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 21.1x faster than the slowest successful cell (`pytorch-cpu`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 15.7x faster than the slowest successful cell (`pytorch-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 12.5x faster than the slowest successful cell (`pytorch-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `jax-cpu` is 15.8x faster than the slowest successful cell (`pytorch-cpu`, 0.169 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 51.3x faster than the slowest successful cell (`pytorch-cpu`, 0.169 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 31.4x faster than the slowest successful cell (`pytorch-cpu`, 0.169 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 47.0x faster than the slowest successful cell (`pytorch-cpu`, 0.194 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 12.5x faster than the slowest successful cell (`pytorch-cpu`, 0.194 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 15.2x faster than the slowest successful cell (`pytorch-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 13.6x faster than the slowest successful cell (`pytorch-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.064 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 23.3x faster than the slowest successful cell (`pytorch-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`8x4x4,rhs=1`): `tenferro-direct` is 14.0x faster than the slowest successful cell (`jax-cpu`, 0.035 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`8x8x8,rhs=1`): `tenferro-direct` is 11.9x faster than the slowest successful cell (`jax-cpu`, 0.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x2x2,rhs=1`): `tenferro-direct` is 25.8x faster than the slowest successful cell (`pytorch-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x2x2,rhs=1`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x4x4,rhs=1`): `tenferro-direct` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 0.043 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `jax-cpu` is 25.7x faster than the slowest successful cell (`julia-base`, 391.964 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `pytorch-cpu` is 14.9x faster than the slowest successful cell (`julia-base`, 391.964 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-direct` is 32.1x faster than the slowest successful cell (`julia-base`, 391.964 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-trace` is 32.2x faster than the slowest successful cell (`julia-base`, 391.964 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 25.9x faster than the slowest successful cell (`julia-base`, 238.513 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 23.1x faster than the slowest successful cell (`julia-base`, 238.513 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 23.8x faster than the slowest successful cell (`julia-base`, 238.513 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 25.1x faster than the slowest successful cell (`julia-base`, 238.513 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `jax-cpu` is 36.7x faster than the slowest successful cell (`julia-base`, 297.254 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `pytorch-cpu` is 35.0x faster than the slowest successful cell (`julia-base`, 297.254 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-direct` is 43.7x faster than the slowest successful cell (`julia-base`, 297.254 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-trace` is 36.1x faster than the slowest successful cell (`julia-base`, 297.254 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `jax-cpu` is 35.1x faster than the slowest successful cell (`julia-base`, 252.737 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `pytorch-cpu` is 31.8x faster than the slowest successful cell (`julia-base`, 252.737 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-direct` is 30.8x faster than the slowest successful cell (`julia-base`, 252.737 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-trace` is 32.8x faster than the slowest successful cell (`julia-base`, 252.737 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `jax-cpu` is 24.0x faster than the slowest successful cell (`julia-base`, 711.407 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 28.6x faster than the slowest successful cell (`julia-base`, 711.407 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-direct` is 31.5x faster than the slowest successful cell (`julia-base`, 711.407 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-trace` is 32.1x faster than the slowest successful cell (`julia-base`, 711.407 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 33.4x faster than the slowest successful cell (`julia-base`, 412.364 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 32.8x faster than the slowest successful cell (`julia-base`, 412.364 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-direct` is 43.8x faster than the slowest successful cell (`julia-base`, 412.364 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-trace` is 40.4x faster than the slowest successful cell (`julia-base`, 412.364 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 12.6x faster than the slowest successful cell (`julia-base`, 139.333 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-eager` is 13.3x faster than the slowest successful cell (`julia-base`, 139.333 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-trace` is 13.6x faster than the slowest successful cell (`julia-base`, 139.333 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 18.5x faster than the slowest successful cell (`julia-base`, 259.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 21.6x faster than the slowest successful cell (`julia-base`, 259.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 26.5x faster than the slowest successful cell (`julia-base`, 259.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 28.4x faster than the slowest successful cell (`julia-base`, 259.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 10.6x faster than the slowest successful cell (`julia-base`, 9.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`julia-base`, 9.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `jax-cpu` is 63.1x faster than the slowest successful cell (`julia-base`, 1081.306 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 72.5x faster than the slowest successful cell (`julia-base`, 1081.306 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 82.1x faster than the slowest successful cell (`julia-base`, 1081.306 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 86.9x faster than the slowest successful cell (`julia-base`, 1081.306 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `jax-cpu` is 71.7x faster than the slowest successful cell (`julia-base`, 1050.938 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 92.0x faster than the slowest successful cell (`julia-base`, 1050.938 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 92.2x faster than the slowest successful cell (`julia-base`, 1050.938 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 85.7x faster than the slowest successful cell (`julia-base`, 1050.938 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 27.2x faster than the slowest successful cell (`julia-base`, 253.173 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 28.1x faster than the slowest successful cell (`julia-base`, 253.173 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 30.8x faster than the slowest successful cell (`julia-base`, 253.173 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 27.9x faster than the slowest successful cell (`julia-base`, 253.173 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 41.9x faster than the slowest successful cell (`julia-base`, 1798.947 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 42.5x faster than the slowest successful cell (`julia-base`, 1798.947 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-eager` is 38.0x faster than the slowest successful cell (`julia-base`, 1798.947 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-trace` is 48.0x faster than the slowest successful cell (`julia-base`, 1798.947 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 5282.5x faster than the slowest successful cell (`julia-base`, 18.927 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 9851.9x faster than the slowest successful cell (`julia-base`, 13.546 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 9740.8x faster than the slowest successful cell (`jax-cpu`, 14.212 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 9836.8x faster than the slowest successful cell (`pytorch-cpu`, 13.526 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 141879.5x faster than the slowest successful cell (`julia-base`, 200.901 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `strided-jl` is 11.1x faster than the slowest successful cell (`julia-base`, 214.772 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-direct` is 11.1x faster than the slowest successful cell (`julia-base`, 214.772 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 160998.3x faster than the slowest successful cell (`julia-base`, 214.772 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 1046.4x faster than the slowest successful cell (`tenferro-direct`, 0.005 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 327.1x faster than the slowest successful cell (`tenferro-direct`, 0.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 136.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 82.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 145.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 87.6x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 9148.4 GB/s at 0.002 ms. The conservative physical lower bound is 0.168 ms, so the cell is 91.5x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 9357.7 GB/s at 0.002 ms. The conservative physical lower bound is 0.042 ms, so the cell is 23.4x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 74919.2 GB/s at 0.004 ms. The conservative physical lower bound is 2.684 ms, so the cell is 749.2x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 195225.8 GB/s at 0.001 ms. The conservative physical lower bound is 0.671 ms, so the cell is 488.1x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 183985.9 GB/s at 0.001 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1839.9x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 195225.8 GB/s at 0.001 ms. The conservative physical lower bound is 0.671 ms, so the cell is 488.1x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 94786.5 GB/s at 0.001 ms. The conservative physical lower bound is 1.342 ms, so the cell is 947.9x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 100613.0 GB/s at 0.001 ms. The conservative physical lower bound is 0.336 ms, so the cell is 251.5x faster than the bound.

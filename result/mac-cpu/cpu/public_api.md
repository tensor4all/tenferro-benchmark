# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20261022_030908/run.yaml`
- Timestamp: `20261022_030908`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Sampling: `quick` profile, 7 measured runs, 3 warmups per row

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20261022_030908`.

- tenferro-rs commit: `d5ef90538a4c05810fe63bc1b5713d871ac98279`

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

## Thread Environments

### Threads: 1

- Run metadata: `data/results/mac-cpu/cpu/public_api/20261022_030908/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/public_api/20261022_030908/run_t4.yaml`
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

- CSV: `data/results/mac-cpu/cpu/public_api/20261022_030908/cpu_public_api_t1_20261022_030908.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20261022_030908/cpu_public_api_t4_20261022_030908.csv`
- Source table: `data/results/mac-cpu/cpu/public_api/20261022_030908/cpu_public_api_20261022_030908.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 3.703 ± 0.129 | - | 4.234 ± 0.145 | 4.068 ± 0.485 | 1.854 ± 0.613 | 8.550 ± 4.076 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 4.213 ± 3.201 | - | 6.234 ± 0.921 | 4.273 ± 0.868 | 1.715 ± 0.125 | 72.175 ± 6.958 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 11.871 ± 0.521 | - | 11.853 ± 1.801 | 10.889 ± 1.271 | 10.906 ± 1.720 | 13.035 ± 2.810 | 12.155 ± 1.863 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 13.581 ± 2.045 | - | 15.675 ± 2.644 | 11.617 ± 3.424 | 10.359 ± 1.151 | 13.032 ± 4.381 | 13.840 ± 1.689 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 51.240 ± 8.728 | - | 50.603 ± 5.338 | 20.857 ± 1.563 | 45.614 ± 19.415 | 53.974 ± 7.541 | 56.462 ± 4.522 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 32.016 ± 3.305 | - | 31.533 ± 7.204 | 14.725 ± 3.966 | 24.250 ± 7.077 | 46.694 ± 0.874 | 22.633 ± 3.453 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 11.571 ± 1.201 | - | 10.010 ± 0.512 | 9.739 ± 1.235 | 54.870 ± 4.243 | 83.617 ± 6.697 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 12.361 ± 3.543 | - | 10.810 ± 0.196 | 11.235 ± 2.218 | 18.139 ± 4.739 | 28.286 ± 2.297 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 11.773 ± 0.752 | - | unsupported | 10.484 ± 1.073 | 65.564 ± 6.083 | 69.431 ± 9.772 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 11.830 ± 0.259 | - | unsupported | 11.315 ± 0.598 | 19.968 ± 4.077 | 40.076 ± 11.846 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 11.763 ± 0.527 | - | 11.990 ± 3.871 | 12.075 ± 0.176 | 21.527 ± 4.661 | 10.258 ± 1.324 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 12.767 ± 1.248 | - | 13.147 ± 1.997 | 10.457 ± 0.513 | 22.827 ± 2.278 | 435.000 ± 3.806 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 35.700 ± 1.282 | - | 36.836 ± 0.586 | 37.370 ± 3.742 | 52.536 ± 4.331 | 68.787 ± 9.371 | 60.658 ± 23.722 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 30.291 ± 10.936 | - | 28.664 ± 6.827 | 16.972 ± 6.754 | 22.411 ± 2.240 | 50.793 ± 1.113 | 26.367 ± 9.077 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 47.850 ± 2.891 | - | 47.628 ± 5.736 | 52.385 ± 5.504 | 100.192 ± 11.877 | 76.946 ± 2.781 | 78.309 ± 6.130 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 36.069 ± 4.389 | - | 32.689 ± 3.713 | 19.688 ± 1.107 | 41.077 ± 13.093 | 69.500 ± 2.440 | 26.535 ± 1.981 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 9.623 ± 2.071 | - | 9.546 ± 1.271 | 9.339 ± 1.089 | 12.080 ± 3.795 | 10.938 ± 1.429 | 9.623 ± 1.660 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 12.627 ± 0.728 | - | 11.064 ± 1.824 | 11.417 ± 3.198 | 12.350 ± 3.813 | 9.491 ± 2.130 | 8.213 ± 0.107 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 0.941 ± 0.193 | - | 8.156 ± 0.245 | 5.906 ± 0.739 | 5.243 ± 0.598 | 8.426 ± 1.493 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 1.492 ± 0.304 | - | 5.851 ± 1.294 | 5.173 ± 3.638 | 3.215 ± 0.722 | 9.612 ± 3.048 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 8.394 ± 0.467 | - | 7.779 ± 0.280 | 8.327 ± 0.414 | 7.672 ± 6.121 | 11.366 ± 0.916 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 8.473 ± 0.718 | - | 8.155 ± 0.106 | 7.518 ± 0.253 | 7.336 ± 0.661 | 174.349 ± 65.703 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 4.972 ± 1.113 | - | 5.280 ± 0.450 | 5.196 ± 0.277 | 2.559 ± 0.133 | 6.498 ± 1.187 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 7.634 ± 1.871 | - | 8.772 ± 1.771 | 6.113 ± 0.838 | 2.284 ± 0.121 | 252.418 ± 124.151 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 7.736 ± 0.398 | - | 7.878 ± 0.645 | 7.723 ± 0.371 | 6.936 ± 0.307 | 8.267 ± 0.575 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 8.024 ± 0.309 | - | 8.218 ± 0.373 | 6.920 ± 0.290 | 6.836 ± 0.397 | 626.394 ± 412.622 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 10.948 ± 0.812 | - | 11.857 ± 1.368 | 9.981 ± 0.458 | 52.263 ± 4.020 | 64.857 ± 4.173 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 11.182 ± 5.853 | - | 11.048 ± 3.974 | 9.407 ± 1.997 | 19.704 ± 2.562 | 32.187 ± 9.142 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 9.728 ± 0.467 | - | unsupported | 9.850 ± 0.832 | 57.217 ± 1.523 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 14.266 ± 6.279 | - | unsupported | 11.076 ± 1.326 | 33.052 ± 9.659 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 13.360 ± 6.893 | - | 12.842 ± 0.638 | 12.354 ± 0.845 | 17.907 ± 3.820 | 12.437 ± 1.027 | 15.340 ± 2.086 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 14.141 ± 0.915 | - | 13.088 ± 1.318 | 13.869 ± 0.815 | 13.864 ± 3.045 | 15.209 ± 1.831 | 13.936 ± 1.077 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 20.611 ± 1.597 | - | 24.462 ± 8.241 | 19.692 ± 3.727 | 24.940 ± 13.052 | 20.613 ± 17.496 | 19.647 ± 1.466 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 25.063 ± 4.216 | - | 26.813 ± 2.952 | 21.627 ± 2.018 | 23.722 ± 1.293 | 24.116 ± 3.726 | 20.839 ± 2.069 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 26.227 ± 0.546 | - | 35.116 ± 4.596 | 43.291 ± 17.832 | 30.381 ± 1.988 | 56.660 ± 9.336 | 66.275 ± 22.131 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 19.878 ± 2.380 | - | 35.023 ± 12.531 | 25.673 ± 3.647 | 20.919 ± 3.748 | 51.238 ± 3.233 | 28.923 ± 5.402 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 6.127 ± 0.109 | - | 6.050 ± 0.233 | 10.188 ± 3.857 | 10.152 ± 4.074 | 6.245 ± 1.217 | 7.751 ± 2.674 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 6.881 ± 0.273 | - | 6.596 ± 0.346 | 7.890 ± 2.117 | 9.356 ± 0.626 | 7.426 ± 1.646 | 7.437 ± 2.113 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 14.857 ± 1.848 | - | 14.828 ± 1.012 | 21.672 ± 9.100 | 18.109 ± 5.039 | 13.706 ± 1.419 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 13.167 ± 1.143 | - | 13.492 ± 0.583 | 15.684 ± 2.109 | 13.278 ± 0.871 | 21.283 ± 8.342 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 30.688 ± 0.429 | - | 32.282 ± 1.006 | 34.324 ± 1.442 | 30.475 ± 1.728 | 37.049 ± 7.940 | 38.483 ± 3.148 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 20.216 ± 1.124 | - | 19.291 ± 1.607 | 18.641 ± 5.948 | 27.863 ± 5.870 | 39.219 ± 2.992 | 18.562 ± 8.320 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 21.165 ± 7.224 | - | 18.827 ± 1.366 | 29.558 ± 7.558 | 35.516 ± 4.634 | 21.616 ± 6.384 | 21.894 ± 3.577 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 22.403 ± 1.432 | - | 21.387 ± 3.544 | 19.569 ± 2.185 | 23.826 ± 2.968 | 19.839 ± 3.688 | 23.039 ± 2.263 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 17.276 ± 1.161 | - | 16.715 ± 0.460 | 39.041 ± 10.606 | 19.880 ± 1.778 | 33.852 ± 6.161 | 36.068 ± 11.647 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 13.139 ± 4.503 | - | 12.107 ± 1.627 | 16.606 ± 3.727 | 16.418 ± 3.357 | 35.248 ± 1.133 | 17.207 ± 4.952 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 28.908 ± 6.065 | - | 26.905 ± 0.767 | 28.937 ± 0.916 | 16.849 ± 1.572 | 33.665 ± 21.021 | 33.702 ± 7.343 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 15.796 ± 2.528 | - | 17.068 ± 2.090 | 16.187 ± 2.480 | 16.146 ± 8.175 | 24.453 ± 1.884 | 12.297 ± 2.137 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 31.637 ± 2.680 | - | 30.071 ± 0.445 | 44.093 ± 3.135 | 28.328 ± 0.952 | 52.153 ± 25.788 | 68.033 ± 36.738 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 20.347 ± 0.660 | - | 16.549 ± 3.574 | 23.165 ± 1.504 | 27.573 ± 5.973 | 46.136 ± 1.585 | 24.147 ± 9.883 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 15.650 ± 1.125 | - | 15.573 ± 0.924 | 23.466 ± 0.914 | 17.684 ± 0.499 | 29.414 ± 5.390 | 30.405 ± 9.652 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 10.653 ± 2.272 | - | 11.792 ± 0.856 | 13.025 ± 2.187 | 14.953 ± 8.114 | 22.183 ± 3.092 | 8.412 ± 1.984 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.258 ± 0.009 | 0.289 ± 0.024 | unsupported | unsupported | 0.260 ± 0.182 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.243 ± 0.103 | 0.308 ± 0.201 | unsupported | unsupported | 0.406 ± 0.041 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 18.167 ± 2.001 | - | 19.604 ± 2.199 | 27.219 ± 9.539 | 32.557 ± 6.044 | 30.450 ± 15.676 | 20.059 ± 6.474 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 19.534 ± 1.792 | - | 18.955 ± 2.943 | 20.348 ± 2.019 | 23.877 ± 1.559 | 19.512 ± 4.063 | 21.002 ± 2.565 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 20.789 ± 2.973 | - | 21.307 ± 3.781 | 24.034 ± 8.049 | 29.625 ± 6.252 | 19.811 ± 3.888 | 20.904 ± 2.267 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 18.901 ± 1.168 | - | 18.662 ± 1.266 | 20.208 ± 2.585 | 24.590 ± 1.889 | 22.355 ± 3.537 | 21.316 ± 3.557 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 20.128 ± 4.646 | - | 24.335 ± 6.247 | 23.002 ± 5.652 | 26.915 ± 8.857 | 20.908 ± 8.472 | 28.086 ± 13.210 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 25.801 ± 6.260 | - | 21.000 ± 1.287 | 23.474 ± 3.632 | 24.965 ± 1.893 | 21.414 ± 2.835 | 19.644 ± 2.476 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 11.548 ± 0.791 | - | 11.049 ± 1.391 | 14.218 ± 5.601 | 18.263 ± 2.791 | 12.670 ± 43.384 | 16.247 ± 4.137 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 14.485 ± 1.517 | - | 13.015 ± 1.307 | 13.001 ± 1.705 | 12.395 ± 2.636 | 12.915 ± 2.205 | 14.318 ± 1.292 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 45.030 ± 11.686 | - | 38.439 ± 5.127 | 72.516 ± 2.980 | 45.651 ± 2.316 | 120.718 ± 47.328 | 89.707 ± 20.619 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 22.501 ± 6.876 | - | 23.200 ± 11.015 | 40.012 ± 8.085 | 28.396 ± 2.392 | 76.935 ± 12.662 | 26.655 ± 2.755 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 7.121 ± 0.529 | - | 7.228 ± 2.933 | 11.805 ± 1.258 | 29.747 ± 3.912 | 6.140 ± 0.261 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 8.598 ± 2.783 | - | 8.745 ± 2.529 | 7.287 ± 1.164 | 14.531 ± 3.837 | 6.418 ± 0.538 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 0.864 ± 0.018 | - | 0.900 ± 0.107 | 16.356 ± 1.935 | 5.886 ± 2.879 | 1.546 ± 0.684 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.965 ± 0.264 | - | 1.226 ± 0.371 | 11.623 ± 1.075 | 1.604 ± 0.764 | 0.995 ± 0.140 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 0.960 ± 0.105 | - | 0.951 ± 0.118 | 1.863 ± 0.361 | 1.117 ± 0.152 | 1.380 ± 0.784 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 1.785 ± 0.058 | - | 2.155 ± 0.277 | 1.094 ± 0.017 | 1.079 ± 0.109 | 0.849 ± 0.104 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 8.120 ± 0.508 | - | 8.351 ± 4.277 | 12.091 ± 1.272 | 40.173 ± 10.449 | 7.046 ± 1.208 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 7.286 ± 1.696 | - | 10.481 ± 3.014 | 6.912 ± 0.249 | 11.654 ± 6.868 | 6.261 ± 0.498 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.017 ± 0.173 | - | 0.974 ± 0.051 | 17.694 ± 2.002 | 5.670 ± 1.816 | 0.805 ± 0.081 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.772 ± 0.072 | - | 0.810 ± 0.150 | 12.380 ± 1.676 | 2.168 ± 0.194 | 0.711 ± 0.179 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 3.804 ± 0.554 | - | 3.952 ± 0.097 | 6.933 ± 0.143 | 3.828 ± 0.183 | 4.113 ± 0.938 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 6.584 ± 0.672 | - | 8.820 ± 1.146 | 3.910 ± 0.494 | 2.911 ± 0.265 | 3.358 ± 0.547 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 6.957 ± 0.444 | - | 8.264 ± 0.711 | 6.396 ± 0.885 | 52.498 ± 7.259 | 6.233 ± 0.316 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 6.602 ± 0.823 | - | 7.637 ± 0.382 | 5.860 ± 0.441 | 26.413 ± 1.161 | 7.152 ± 0.481 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 1.007 ± 0.234 | - | 0.923 ± 0.139 | 5.693 ± 2.022 | 4.340 ± 3.013 | 1.085 ± 0.921 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.735 ± 0.047 | - | 0.994 ± 0.336 | 2.319 ± 0.252 | 1.464 ± 0.052 | 0.787 ± 0.134 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 0.880 ± 0.026 | - | 1.053 ± 0.233 | 1.233 ± 0.335 | 1.371 ± 0.085 | 1.094 ± 0.328 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 1.881 ± 0.283 | - | 2.192 ± 1.166 | 0.813 ± 0.016 | 1.254 ± 0.091 | 0.737 ± 0.061 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 7.265 ± 1.410 | - | 6.744 ± 2.752 | 5.491 ± 0.795 | 36.002 ± 5.355 | 6.536 ± 0.475 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 7.995 ± 1.406 | - | 8.357 ± 1.954 | 5.648 ± 0.196 | 21.533 ± 3.877 | 6.416 ± 0.674 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.896 ± 0.074 | - | 0.834 ± 0.050 | 4.942 ± 0.831 | 2.845 ± 1.007 | 0.944 ± 0.073 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.697 ± 0.077 | - | 0.855 ± 0.181 | 2.402 ± 0.142 | 2.196 ± 0.063 | 0.800 ± 0.140 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 0.973 ± 0.033 | - | 0.964 ± 0.026 | 0.803 ± 0.277 | 1.040 ± 0.371 | 1.011 ± 0.199 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 1.548 ± 0.214 | - | 2.037 ± 0.361 | 0.716 ± 0.040 | 1.158 ± 0.644 | 0.872 ± 0.168 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.839 ± 0.056 | 0.786 ± 0.116 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.717 ± 0.048 | 0.815 ± 0.067 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 28.898 ± 1.644 | - | 27.703 ± 0.954 | 37.633 ± 4.613 | 43.073 ± 4.516 | 35.193 ± 49.792 | 32.430 ± 3.309 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 16.962 ± 3.966 | - | 15.824 ± 2.622 | 25.322 ± 3.502 | 27.805 ± 6.078 | 29.523 ± 26.115 | 15.146 ± 7.666 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 24.521 ± 9.572 | - | 19.982 ± 1.650 | 19.988 ± 1.068 | 40.479 ± 2.781 | 46.457 ± 2.446 | 56.867 ± 6.975 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 13.489 ± 0.758 | - | 13.426 ± 0.841 | 13.347 ± 1.891 | 23.131 ± 1.608 | 46.079 ± 2.156 | 26.257 ± 6.922 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 27.069 ± 2.112 | - | 27.244 ± 0.985 | 36.139 ± 10.731 | 30.749 ± 3.106 | 26.427 ± 3.340 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 25.417 ± 2.384 | - | 24.333 ± 1.664 | 24.007 ± 4.883 | 25.399 ± 2.124 | 28.509 ± 1.647 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 11.567 ± 0.404 | - | 11.606 ± 1.743 | 13.184 ± 7.460 | 20.807 ± 4.594 | 14.287 ± 5.136 | 22.420 ± 5.411 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 12.683 ± 0.588 | - | 13.445 ± 1.098 | 13.719 ± 2.528 | 11.737 ± 2.660 | 14.864 ± 9.397 | 14.420 ± 1.158 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 24.078 ± 1.234 | - | 23.763 ± 0.606 | 30.110 ± 2.084 | 27.827 ± 1.024 | 37.279 ± 10.539 | 38.476 ± 11.611 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 15.483 ± 1.684 | - | 15.520 ± 3.062 | 19.183 ± 5.596 | 22.359 ± 12.891 | 34.466 ± 2.760 | 16.934 ± 7.046 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 19.011 ± 5.495 | - | 13.601 ± 6.028 | 13.218 ± 1.167 | 23.575 ± 2.361 | 28.752 ± 2.727 | 30.431 ± 4.431 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 12.752 ± 0.350 | - | 12.778 ± 0.215 | 12.817 ± 0.768 | 13.519 ± 2.079 | 29.273 ± 2.483 | 15.424 ± 0.923 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 21.620 ± 4.558 | - | 21.523 ± 2.260 | 18.832 ± 1.477 | 25.287 ± 2.308 | 18.909 ± 3.972 | 26.914 ± 6.090 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 25.311 ± 6.380 | - | 24.692 ± 5.266 | 20.160 ± 1.653 | 23.613 ± 1.802 | 18.959 ± 4.027 | 21.122 ± 2.637 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 69.420 ± 10.725 | - | 73.427 ± 3.356 | 70.838 ± 5.514 | 13.717 ± 0.330 | 51.105 ± 4.793 | 55.345 ± 15.909 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 34.985 ± 6.001 | - | 39.007 ± 4.933 | 42.417 ± 11.544 | 14.752 ± 4.225 | 43.112 ± 3.362 | 24.185 ± 4.769 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 27.156 ± 1.838 | 28.214 ± 6.401 | unsupported | 32.698 ± 4.023 | 33.718 ± 5.506 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 25.014 ± 2.264 | 28.021 ± 2.553 | unsupported | 26.286 ± 3.946 | 27.360 ± 7.853 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.910 ± 0.233 | - | 1.501 ± 0.086 | 0.919 ± 0.156 | 0.853 ± 0.077 | 0.792 ± 0.042 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.941 ± 0.070 | - | 2.030 ± 0.087 | 0.850 ± 0.112 | 0.957 ± 0.178 | 0.889 ± 0.115 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.682 ± 0.030 | - | 2.299 ± 0.158 | 0.910 ± 0.130 | 0.754 ± 0.103 | 0.824 ± 0.189 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.750 ± 0.056 | - | 2.665 ± 0.402 | 1.000 ± 0.147 | 0.931 ± 0.079 | 1.121 ± 0.313 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.104 ± 0.113 | - | unsupported | 1.252 ± 0.414 | 0.988 ± 0.097 | 1.152 ± 0.145 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 1.212 ± 0.124 | - | unsupported | 1.327 ± 0.078 | 1.373 ± 0.164 | 1.536 ± 0.576 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.337 ± 0.013 | - | 0.641 ± 0.109 | 0.506 ± 0.081 | 0.903 ± 0.952 | 1.178 ± 0.492 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.418 ± 0.142 | - | 0.800 ± 0.378 | 0.364 ± 0.022 | 0.405 ± 0.020 | 1.390 ± 1.208 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.796 ± 0.109 | - | 1.608 ± 0.418 | 1.178 ± 0.197 | 0.920 ± 0.224 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.802 ± 0.075 | - | 1.957 ± 0.092 | 1.315 ± 0.117 | 0.813 ± 0.164 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.764 ± 0.085 | - | 1.713 ± 0.324 | 0.919 ± 0.303 | 1.015 ± 0.776 | 2.042 ± 11.159 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.899 ± 0.176 | - | 3.054 ± 1.279 | 0.974 ± 0.138 | 0.934 ± 0.074 | 1.399 ± 0.315 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.595 ± 0.040 | - | 0.931 ± 0.065 | 0.469 ± 0.038 | 0.535 ± 0.098 | 1.016 ± 0.692 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.578 ± 0.017 | - | 0.870 ± 0.020 | 0.700 ± 0.843 | 0.525 ± 0.116 | 1.334 ± 1.873 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.260 ± 0.082 | - | 0.003 ± 0.001 | 1.715 ± 0.366 | 1.675 ± 1.107 | 1.656 ± 0.098 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 1.435 ± 0.114 | - | 0.003 ± 0.002 | 1.553 ± 0.132 | 1.470 ± 0.146 | 1.589 ± 0.168 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.056 ± 0.002 | - | 0.061 ± 0.000 | 0.110 ± 0.006 | 0.099 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.095 ± 0.010 | - | 0.103 ± 0.002 | 0.151 ± 0.008 | 0.187 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.187 ± 0.001 | - | 0.207 ± 0.013 | 0.275 ± 0.011 | 0.229 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.023 ± 0.001 | - | 0.036 ± 0.002 | 0.040 ± 0.001 | 0.055 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.001 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.045 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.003 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.003 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.046 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.178 ± 0.015 | - | 0.211 ± 0.001 | 0.260 ± 0.018 | 0.249 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.010 ± 0.000 | - | 0.016 ± 0.001 | 0.017 ± 0.001 | 0.054 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.019 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.061 ± 0.002 | - | 0.066 ± 0.001 | 0.137 ± 0.008 | 0.137 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.099 ± 0.007 | - | 0.105 ± 0.002 | 0.266 ± 0.040 | 0.151 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.204 ± 0.022 | - | 0.264 ± 0.098 | 0.360 ± 0.016 | 0.309 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.042 ± 0.028 | - | 0.059 ± 0.022 | 0.127 ± 0.011 | 0.122 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.002 | 0.040 ± 0.016 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.077 ± 0.047 | 0.064 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.070 ± 0.045 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.001 ± 0.000 | - | 0.005 ± 0.001 | 0.080 ± 0.048 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.038 ± 0.012 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.083 ± 0.062 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.118 ± 0.121 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.041 ± 0.011 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.024 | 0.061 ± 0.029 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.074 ± 0.030 | 0.068 ± 0.070 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.247 ± 0.095 | - | 0.432 ± 0.105 | 0.368 ± 0.053 | 0.432 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.001 | 0.039 ± 0.020 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.010 ± 0.001 | - | 0.019 ± 0.002 | 0.099 ± 0.031 | 0.089 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.046 ± 0.024 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.002 ± 0.001 | - | 0.006 ± 0.000 | 0.059 ± 0.015 | 0.064 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.241 ± 0.010 | - | 0.246 ± 0.002 | 0.492 ± 0.122 | 0.747 ± 0.103 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.458 ± 0.044 | - | 1.464 ± 0.033 | 1.729 ± 0.110 | 4.438 ± 3.257 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 6.190 ± 0.663 | - | 6.021 ± 0.195 | 6.244 ± 0.790 | 8.381 ± 1.058 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 5.817 ± 0.059 | - | 7.108 ± 0.798 | 3.050 ± 0.088 | 4.706 ± 1.368 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.084 ± 0.006 | - | 0.087 ± 0.000 | 0.095 ± 0.008 | 0.160 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.003 ± 0.001 | - | 0.006 ± 0.000 | 0.006 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.008 ± 0.001 | - | 0.011 ± 0.000 | 0.011 ± 0.002 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.002 | 0.007 ± 0.000 | 0.013 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.003 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.019 ± 0.004 | - | 0.023 ± 0.004 | 0.023 ± 0.002 | 0.133 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.007 ± 0.001 | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.025 ± 0.005 | - | 0.028 ± 0.004 | 0.029 ± 0.004 | 0.151 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 51.063 ± 1.098 | - | 47.551 ± 3.205 | 27.105 ± 3.348 | 42.098 ± 9.033 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.004 ± 0.001 | - | 0.007 ± 0.000 | 0.008 ± 0.002 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.659 ± 0.008 | - | 0.666 ± 0.008 | 0.748 ± 0.030 | 1.021 ± 0.119 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.013 ± 0.000 | - | 0.016 ± 0.000 | 0.018 ± 0.004 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.051 ± 0.007 | - | 0.054 ± 0.005 | 0.051 ± 0.002 | 0.110 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.353 ± 0.027 | - | 0.388 ± 0.011 | 0.545 ± 0.224 | 0.885 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 1.958 ± 0.023 | - | 1.967 ± 0.031 | 2.127 ± 0.276 | 1.529 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 7.943 ± 2.167 | - | 6.028 ± 0.160 | 7.159 ± 0.862 | 4.676 ± 0.194 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 6.123 ± 0.333 | - | 6.402 ± 1.068 | 3.687 ± 0.115 | 4.481 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.086 ± 0.000 | - | 0.105 ± 0.014 | 0.102 ± 0.025 | 0.191 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.012 ± 0.002 | 0.008 ± 0.002 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.011 ± 0.000 | - | 0.017 ± 0.002 | 0.013 ± 0.003 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.009 ± 0.001 | 0.008 ± 0.009 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.008 ± 0.002 | - | 0.016 ± 0.002 | 0.011 ± 0.000 | 0.019 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.025 ± 0.003 | - | 0.031 ± 0.004 | 0.025 ± 0.000 | 0.080 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.004 ± 0.001 | - | 0.012 ± 0.001 | 0.019 ± 0.011 | 0.018 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.010 ± 0.002 | - | 0.018 ± 0.002 | 0.014 ± 0.002 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.035 ± 0.002 | - | 0.033 ± 0.006 | 0.036 ± 0.007 | 0.097 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 51.740 ± 7.266 | - | 51.898 ± 4.581 | 27.914 ± 4.002 | 18.204 ± 5.032 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.004 ± 0.002 | - | 0.013 ± 0.002 | 0.010 ± 0.012 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.794 ± 0.035 | - | 0.817 ± 1.301 | 0.781 ± 0.040 | 0.697 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.018 ± 0.001 | - | 0.025 ± 0.003 | 0.021 ± 0.001 | 0.028 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.055 ± 0.009 | - | 0.071 ± 0.004 | 0.074 ± 0.018 | 0.115 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.247 ± 0.022 | - | 0.241 ± 0.015 | 0.259 ± 0.001 | 0.581 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.306 ± 0.200 | - | 1.293 ± 0.104 | 1.330 ± 0.080 | 1.636 ± 0.064 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 4.901 ± 0.254 | - | 4.855 ± 0.325 | 5.516 ± 0.609 | 4.874 ± 0.117 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 0.885 ± 0.054 | - | 0.925 ± 0.015 | 0.894 ± 0.264 | 0.940 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.001 ± 0.001 | - | 0.006 ± 0.001 | 0.004 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.063 ± 0.006 | - | 0.070 ± 0.003 | 0.071 ± 0.003 | 0.100 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.001 | 0.010 ± 0.001 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.007 | 0.006 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.005 ± 0.001 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.017 ± 0.004 | - | 0.025 ± 0.005 | 0.019 ± 0.005 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.003 ± 0.005 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.006 ± 0.001 | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.021 ± 0.004 | - | 0.030 ± 0.003 | 0.027 ± 0.009 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 7.151 ± 0.216 | - | 7.494 ± 0.144 | 7.761 ± 0.609 | 9.361 ± 0.732 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.002 | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.494 ± 0.028 | - | 0.501 ± 0.020 | 0.530 ± 0.008 | 0.508 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.010 ± 0.000 | - | 0.014 ± 0.000 | 0.013 ± 0.003 | 0.044 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.039 ± 0.004 | - | 0.047 ± 0.003 | 0.042 ± 0.008 | 0.073 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.303 ± 0.165 | - | 0.332 ± 0.083 | 0.310 ± 0.098 | 0.668 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.749 ± 0.307 | - | 2.104 ± 0.844 | 2.169 ± 1.037 | 1.718 ± 0.293 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 7.477 ± 0.272 | - | 8.495 ± 1.576 | 7.480 ± 2.898 | 4.892 ± 1.855 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.133 ± 0.703 | - | 1.700 ± 0.747 | 1.289 ± 0.289 | 1.233 ± 0.666 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.006 ± 0.001 | 0.012 ± 0.012 | 0.012 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.072 ± 0.024 | - | 0.101 ± 0.030 | 0.080 ± 0.023 | 0.301 ± 0.166 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.007 ± 0.000 | - | 0.019 ± 0.000 | 0.005 ± 0.002 | 0.015 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.007 ± 0.000 | - | 0.012 ± 0.022 | 0.011 ± 0.007 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.005 ± 0.000 | - | 0.016 ± 0.000 | 0.007 ± 0.014 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.014 ± 0.001 | - | 0.026 ± 0.002 | 0.022 ± 0.008 | 0.048 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.021 ± 0.035 | - | 0.026 ± 0.005 | 0.027 ± 0.006 | 0.076 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.006 ± 0.000 | - | 0.018 ± 0.012 | 0.011 ± 0.007 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.017 ± 0.001 | - | 0.030 ± 0.000 | 0.011 ± 0.002 | 0.050 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.020 ± 0.004 | - | 0.031 ± 0.021 | 0.029 ± 0.001 | 0.038 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 11.715 ± 3.557 | - | 10.646 ± 2.704 | 9.173 ± 1.255 | 5.885 ± 0.552 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.009 ± 0.000 | - | 0.008 ± 0.001 | 0.011 ± 0.011 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.830 ± 0.333 | - | 0.747 ± 0.252 | 0.674 ± 0.160 | 0.565 ± 0.309 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.030 ± 0.000 | - | 0.019 ± 0.050 | 0.017 ± 0.003 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.055 ± 0.044 | - | 0.058 ± 0.044 | 0.071 ± 0.010 | 0.181 ± 0.118 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.141 ± 0.001 | - | 0.147 ± 0.002 | 0.140 ± 0.011 | 1.345 ± 0.235 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 0.735 ± 0.135 | - | 0.732 ± 0.026 | 0.818 ± 0.048 | 3.592 ± 1.881 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 2.154 ± 0.038 | - | 2.227 ± 0.008 | 2.298 ± 0.041 | 10.106 ± 0.436 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.466 ± 0.006 | - | 0.478 ± 0.010 | 0.472 ± 0.024 | 1.379 ± 0.383 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.001 ± 0.001 | - | 0.007 ± 0.000 | 0.006 ± 0.001 | 0.032 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.024 ± 0.000 | - | 0.031 ± 0.002 | 0.030 ± 0.004 | 0.135 ± 0.065 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.007 ± 0.001 | 0.006 ± 0.001 | 0.032 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.043 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.010 ± 0.001 | - | 0.014 ± 0.000 | 0.015 ± 0.003 | 0.030 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.006 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.005 ± 0.001 | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.015 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.012 ± 0.002 | - | 0.016 ± 0.000 | 0.014 ± 0.004 | 0.031 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 3.296 ± 0.011 | - | 3.933 ± 0.363 | 3.369 ± 0.340 | 10.817 ± 1.197 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.006 ± 0.000 | 0.025 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.179 ± 0.001 | - | 0.190 ± 0.000 | 0.195 ± 0.017 | 0.917 ± 0.559 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.008 ± 0.001 | - | 0.013 ± 0.000 | 0.010 ± 0.000 | 0.065 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.019 ± 0.000 | - | 0.025 ± 0.000 | 0.028 ± 0.005 | 0.085 ± 0.149 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.105 ± 0.000 | - | 0.108 ± 0.001 | 0.139 ± 0.013 | 0.575 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 0.703 ± 0.032 | - | 0.756 ± 0.087 | 0.910 ± 0.116 | 1.518 ± 0.167 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 2.172 ± 0.018 | - | 2.255 ± 0.536 | 2.821 ± 0.142 | 3.683 ± 0.224 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.523 ± 0.041 | - | 0.554 ± 0.040 | 0.524 ± 0.041 | 1.123 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.001 | - | 0.008 ± 0.002 | 0.006 ± 0.001 | 0.027 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.031 ± 0.001 | - | 0.033 ± 0.006 | 0.030 ± 0.004 | 0.127 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.001 | 0.007 ± 0.002 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.009 ± 0.002 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.002 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.008 ± 0.002 | - | 0.012 ± 0.003 | 0.014 ± 0.001 | 0.067 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.005 ± 0.001 | - | 0.009 ± 0.002 | 0.009 ± 0.001 | 0.016 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.012 ± 0.003 | - | 0.016 ± 0.001 | 0.014 ± 0.000 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 3.555 ± 0.110 | - | 3.874 ± 0.630 | 4.480 ± 0.929 | 5.127 ± 0.179 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.217 ± 0.014 | - | 0.230 ± 0.009 | 0.217 ± 0.017 | 0.454 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.002 | 0.014 ± 0.013 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.022 ± 0.005 | - | 0.022 ± 0.001 | 0.028 ± 0.014 | 0.114 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.099 ± 0.019 | - | 0.100 ± 0.009 | 0.159 ± 0.021 | 0.146 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.191 ± 0.000 | - | 0.197 ± 0.001 | 0.273 ± 0.030 | 0.285 ± 0.101 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.398 ± 0.011 | - | 0.412 ± 0.002 | 0.564 ± 0.023 | 0.542 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.052 ± 0.001 | - | 0.058 ± 0.001 | 0.068 ± 0.002 | 0.194 ± 0.114 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.003 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.002 | 0.037 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.009 ± 0.001 | - | 0.012 ± 0.000 | 0.016 ± 0.003 | 0.103 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.008 ± 0.001 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.002 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.045 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.001 | 0.008 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.002 | 0.096 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.408 ± 0.031 | - | 0.422 ± 0.001 | 0.528 ± 0.595 | 0.713 ± 0.154 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.002 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.029 ± 0.000 | - | 0.034 ± 0.000 | 0.043 ± 0.007 | 0.110 ± 0.067 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.001 | 0.099 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.005 ± 0.001 | - | 0.008 ± 0.000 | 0.035 ± 0.006 | 0.049 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.120 ± 0.052 | - | 0.142 ± 0.041 | 0.159 ± 0.005 | 0.389 ± 0.111 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.241 ± 0.035 | - | 0.225 ± 0.026 | 0.210 ± 0.004 | 0.658 ± 0.200 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.503 ± 0.053 | - | 0.842 ± 0.389 | 0.422 ± 0.160 | 0.647 ± 0.078 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.062 ± 0.002 | - | 0.070 ± 0.010 | 0.110 ± 0.010 | 0.136 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.131 ± 0.042 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.005 ± 0.000 | - | 0.010 ± 0.002 | 0.061 ± 0.008 | 0.078 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.003 ± 0.002 | - | 0.007 ± 0.001 | 0.046 ± 0.006 | 0.037 ± 0.062 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.008 ± 0.003 | - | 0.016 ± 0.001 | 0.058 ± 0.014 | 0.079 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.046 ± 0.018 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.002 | 0.091 ± 0.026 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.061 ± 0.016 | 0.038 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.004 ± 0.001 | - | 0.007 ± 0.000 | 0.051 ± 0.008 | 0.070 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.075 ± 0.032 | 0.013 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.003 ± 0.001 | - | 0.008 ± 0.001 | 0.056 ± 0.011 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.003 ± 0.000 | - | 0.007 ± 0.001 | 0.047 ± 0.010 | 0.067 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.852 ± 0.357 | - | 1.593 ± 0.285 | 0.377 ± 0.037 | 0.792 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.018 ± 0.002 | 0.098 ± 0.031 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.033 ± 0.005 | - | 0.044 ± 0.004 | 0.100 ± 0.010 | 0.160 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.004 ± 0.001 | - | 0.008 ± 0.001 | 0.083 ± 0.010 | 0.080 ± 0.115 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.006 ± 0.000 | - | 0.011 ± 0.001 | 0.084 ± 0.011 | 0.074 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.277 ± 0.021 | - | 0.258 ± 0.001 | 0.235 ± 0.010 | 0.368 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.577 ± 0.016 | - | 0.589 ± 0.023 | 0.539 ± 0.024 | 0.706 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.488 ± 0.036 | - | 1.485 ± 0.041 | 1.493 ± 0.028 | 1.593 ± 0.084 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.290 ± 0.028 | - | 0.299 ± 0.001 | 0.349 ± 0.013 | 0.399 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.005 ± 0.011 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.017 ± 0.003 | - | 0.023 ± 0.000 | 0.017 ± 0.001 | 0.077 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.008 ± 0.001 | 0.003 ± 0.000 | 0.011 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.014 ± 0.000 | - | 0.018 ± 0.003 | 0.017 ± 0.001 | 0.052 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.003 ± 0.001 | - | 0.009 ± 0.000 | 0.004 ± 0.000 | 0.013 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.003 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.004 ± 0.001 | - | 0.009 ± 0.001 | 0.004 ± 0.000 | 0.012 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.008 ± 0.001 | - | 0.012 ± 0.000 | 0.007 ± 0.001 | 0.014 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.004 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.009 ± 0.001 | - | 0.013 ± 0.000 | 0.009 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 2.351 ± 0.140 | - | 2.556 ± 0.904 | 2.742 ± 0.325 | 3.023 ± 0.416 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.004 ± 0.001 | - | 0.009 ± 0.000 | 0.004 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.126 ± 0.001 | - | 0.135 ± 0.001 | 0.116 ± 0.020 | 0.174 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.007 ± 0.000 | - | 0.012 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.013 ± 0.003 | - | 0.019 ± 0.000 | 0.015 ± 0.002 | 0.025 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.310 ± 0.050 | - | 0.315 ± 0.025 | 0.489 ± 0.034 | 0.382 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.717 ± 0.044 | - | 0.779 ± 0.338 | 0.561 ± 0.055 | 0.726 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.705 ± 0.140 | - | 2.246 ± 0.741 | 1.805 ± 0.638 | 1.339 ± 0.078 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.504 ± 0.102 | - | 0.456 ± 0.084 | 0.538 ± 0.202 | 0.435 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.014 ± 0.000 | 0.031 ± 0.028 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.018 ± 0.004 | - | 0.050 ± 0.059 | 0.039 ± 0.024 | 0.108 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.003 ± 0.001 | - | 0.008 ± 0.001 | 0.058 ± 0.028 | 0.015 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.017 ± 0.004 | - | 0.021 ± 0.005 | 0.057 ± 0.011 | 0.103 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.012 ± 0.003 | 0.021 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.005 ± 0.000 | - | 0.010 ± 0.002 | 0.057 ± 0.043 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.004 ± 0.001 | - | 0.011 ± 0.002 | 0.037 ± 0.022 | 0.015 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.006 ± 0.000 | - | 0.017 ± 0.008 | 0.024 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.011 ± 0.002 | 0.033 ± 0.035 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.005 ± 0.001 | - | 0.012 ± 0.002 | 0.030 ± 0.019 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.010 ± 0.001 | - | 0.014 ± 0.001 | 0.025 ± 0.004 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 3.492 ± 1.154 | - | 4.172 ± 1.179 | 2.844 ± 0.211 | 2.408 ± 2.398 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.008 ± 0.001 | - | 0.008 ± 0.001 | 0.041 ± 0.035 | 0.012 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.297 ± 0.048 | - | 0.385 ± 0.174 | 0.214 ± 0.121 | 0.312 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.008 ± 0.002 | - | 0.013 ± 0.012 | 0.038 ± 0.016 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.017 ± 0.004 | - | 0.022 ± 0.003 | 0.033 ± 0.012 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.131 ± 0.007 | - | 0.121 ± 0.021 | 0.111 ± 0.010 | 0.168 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.217 ± 0.017 | - | 0.200 ± 0.021 | 0.175 ± 0.008 | 0.264 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.347 ± 0.008 | - | 0.355 ± 0.013 | 0.357 ± 0.025 | 0.479 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.051 ± 0.001 | - | 0.056 ± 0.000 | 0.093 ± 0.013 | 0.124 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.015 ± 0.010 | 0.011 ± 0.002 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.004 ± 0.001 | - | 0.009 ± 0.000 | 0.013 ± 0.003 | 0.043 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.004 | 0.010 ± 0.002 | 0.009 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.001 | 0.012 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.002 | 0.010 ± 0.002 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.010 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.014 ± 0.003 | 0.011 ± 0.003 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.369 ± 0.056 | - | 0.508 ± 0.079 | 0.382 ± 0.018 | 0.601 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.027 ± 0.000 | - | 0.044 ± 0.008 | 0.037 ± 0.004 | 0.091 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.002 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.007 ± 0.004 | - | 0.015 ± 0.006 | 0.012 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.136 ± 0.016 | - | 0.317 ± 0.162 | 0.108 ± 0.000 | 0.225 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.294 ± 0.150 | - | 0.221 ± 0.094 | 0.156 ± 0.000 | 0.298 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.401 ± 0.110 | - | 0.436 ± 0.111 | 0.295 ± 0.001 | 0.567 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.054 ± 0.001 | - | 0.064 ± 0.020 | 0.071 ± 0.006 | 0.166 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.009 ± 0.010 | 0.009 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.004 ± 0.001 | - | 0.012 ± 0.001 | 0.012 ± 0.000 | 0.118 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.032 ± 0.041 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.002 ± 0.001 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.002 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.002 ± 0.002 | - | 0.041 ± 0.024 | 0.009 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.021 ± 0.047 | 0.011 ± 0.002 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.009 ± 0.006 | 0.009 ± 0.000 | 0.009 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.002 ± 0.003 | - | 0.009 ± 0.007 | 0.009 ± 0.000 | 0.023 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.001 | 0.010 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.427 ± 0.063 | - | 0.529 ± 0.090 | 0.289 ± 0.032 | 0.736 ± 0.112 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.003 ± 0.004 | - | 0.010 ± 0.008 | 0.035 ± 0.010 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.032 ± 0.020 | - | 0.042 ± 0.004 | 0.072 ± 0.003 | 0.271 ± 0.070 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.003 ± 0.003 | - | 0.009 ± 0.000 | 0.033 ± 0.001 | 0.013 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.004 ± 0.001 | - | 0.010 ± 0.002 | 0.042 ± 0.003 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.472 ± 0.239 | - | 1.379 ± 1.082 | 1.093 ± 0.025 | 3.146 ± 0.262 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 4.943 ± 1.194 | - | 3.909 ± 0.043 | 5.280 ± 1.001 | 6.932 ± 0.369 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 15.879 ± 0.901 | - | 13.592 ± 3.281 | 14.343 ± 1.343 | 19.584 ± 4.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 4.357 ± 0.753 | - | 5.584 ± 0.302 | 4.891 ± 0.669 | 5.768 ± 1.196 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.004 ± 0.001 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.163 ± 0.002 | - | 0.165 ± 0.002 | 0.169 ± 0.004 | 0.258 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.008 ± 0.008 | - | 0.013 ± 0.000 | 0.009 ± 0.000 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.191 ± 0.031 | - | 0.187 ± 0.040 | 0.189 ± 0.020 | 0.354 ± 0.082 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.016 ± 0.000 | - | 0.019 ± 0.001 | 0.027 ± 0.004 | 0.032 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.013 ± 0.008 | - | 0.026 ± 0.003 | 0.009 ± 0.000 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.017 ± 0.003 | - | 0.022 ± 0.003 | 0.017 ± 0.000 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.044 ± 0.002 | - | 0.045 ± 0.004 | 0.056 ± 0.009 | 0.074 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.009 ± 0.008 | - | 0.014 ± 0.006 | 0.010 ± 0.000 | 0.026 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.021 ± 0.005 | - | 0.027 ± 0.007 | 0.021 ± 0.000 | 0.041 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.056 ± 0.001 | - | 0.060 ± 0.000 | 0.079 ± 0.005 | 0.093 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 33.607 ± 5.102 | - | 44.991 ± 6.802 | 45.784 ± 4.982 | 48.392 ± 8.330 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.015 ± 0.004 | - | 0.020 ± 0.008 | 0.014 ± 0.000 | 0.036 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.541 ± 0.359 | - | 1.454 ± 0.080 | 1.271 ± 0.033 | 2.295 ± 0.633 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.043 ± 0.005 | - | 0.044 ± 0.006 | 0.036 ± 0.000 | 0.065 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.108 ± 0.003 | - | 0.112 ± 0.000 | 0.128 ± 0.017 | 0.219 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.361 ± 0.183 | - | 1.197 ± 0.022 | 1.366 ± 0.062 | 3.313 ± 0.191 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 4.330 ± 0.090 | - | 4.452 ± 0.184 | 5.740 ± 0.413 | 4.076 ± 0.312 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 15.738 ± 2.014 | - | 17.118 ± 1.799 | 14.831 ± 2.382 | 13.262 ± 2.086 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 4.493 ± 0.176 | - | 6.147 ± 0.180 | 5.503 ± 0.271 | 5.796 ± 1.151 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.001 | 0.031 ± 0.016 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.181 ± 0.034 | - | 0.186 ± 0.013 | 0.189 ± 0.010 | 0.302 ± 0.053 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.007 ± 0.001 | - | 0.010 ± 0.000 | 0.012 ± 0.002 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.286 ± 0.222 | - | 0.214 ± 0.035 | 0.176 ± 0.009 | 0.264 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.020 ± 0.002 | - | 0.021 ± 0.003 | 0.032 ± 0.014 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.006 ± 0.000 | - | 0.010 ± 0.001 | 0.028 ± 0.015 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.014 ± 0.004 | - | 0.021 ± 0.004 | 0.019 ± 0.004 | 0.119 ± 0.078 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.044 ± 0.002 | - | 0.050 ± 0.001 | 0.071 ± 0.011 | 0.168 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.007 ± 0.000 | - | 0.012 ± 0.003 | 0.034 ± 0.009 | 0.067 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.019 ± 0.005 | - | 0.022 ± 0.000 | 0.027 ± 0.001 | 0.042 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.062 ± 0.001 | - | 0.075 ± 0.013 | 0.102 ± 0.057 | 0.098 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 36.053 ± 1.866 | - | 60.818 ± 11.988 | 38.546 ± 2.789 | 22.701 ± 0.720 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.014 ± 0.001 | - | 0.015 ± 0.004 | 0.049 ± 0.025 | 0.044 ± 0.100 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 1.438 ± 0.054 | - | 1.480 ± 0.059 | 1.613 ± 0.051 | 1.295 ± 0.135 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.035 ± 0.001 | - | 0.039 ± 0.008 | 0.043 ± 0.008 | 0.160 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.121 ± 0.015 | - | 0.121 ± 0.001 | 0.316 ± 0.136 | 0.219 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.521 ± 0.015 | - | 0.545 ± 0.029 | 0.527 ± 0.012 | 2.961 ± 1.593 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 2.499 ± 0.091 | - | 2.516 ± 0.229 | 3.016 ± 0.336 | 6.110 ± 5.027 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 11.196 ± 0.928 | - | 9.652 ± 0.110 | 10.069 ± 2.047 | 11.764 ± 1.419 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.501 ± 0.123 | - | 1.837 ± 0.357 | 1.330 ± 0.009 | 2.960 ± 2.357 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.001 | 0.020 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.108 ± 0.002 | - | 0.110 ± 0.002 | 0.133 ± 0.008 | 0.164 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.005 ± 0.001 | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.093 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.099 ± 0.051 | - | 0.115 ± 0.026 | 0.088 ± 0.001 | 0.132 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.012 ± 0.003 | - | 0.015 ± 0.001 | 0.019 ± 0.002 | 0.070 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.004 ± 0.001 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.021 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.009 ± 0.001 | - | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.030 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.027 ± 0.000 | - | 0.030 ± 0.000 | 0.041 ± 0.001 | 0.125 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.004 ± 0.001 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.027 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.012 ± 0.000 | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.030 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.037 ± 0.000 | - | 0.040 ± 0.000 | 0.041 ± 0.003 | 0.074 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 10.883 ± 1.734 | - | 14.056 ± 4.362 | 11.000 ± 1.249 | 12.822 ± 2.623 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.006 ± 0.001 | - | 0.009 ± 0.000 | 0.011 ± 0.003 | 0.054 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 0.855 ± 0.028 | - | 0.852 ± 0.023 | 0.967 ± 0.282 | 1.303 ± 0.668 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.023 ± 0.000 | - | 0.026 ± 0.000 | 0.025 ± 0.000 | 0.049 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.072 ± 0.001 | - | 0.076 ± 0.001 | 0.077 ± 0.000 | 0.126 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.661 ± 0.037 | - | 0.688 ± 0.015 | 0.625 ± 0.189 | 2.008 ± 0.162 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 2.496 ± 0.035 | - | 2.548 ± 0.064 | 2.518 ± 0.056 | 2.893 ± 0.271 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 9.873 ± 0.250 | - | 9.767 ± 0.277 | 11.445 ± 2.683 | 8.082 ± 1.881 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 1.685 ± 0.454 | - | 1.348 ± 0.162 | 1.939 ± 0.586 | 1.791 ± 0.086 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.001 | 0.008 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.130 ± 0.006 | - | 0.133 ± 0.005 | 0.146 ± 0.040 | 0.227 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.006 ± 0.001 | - | 0.010 ± 0.000 | 0.008 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.091 ± 0.007 | - | 0.094 ± 0.008 | 0.149 ± 0.040 | 0.222 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.011 ± 0.001 | - | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.004 ± 0.001 | - | 0.009 ± 0.000 | 0.010 ± 0.002 | 0.021 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.012 ± 0.001 | - | 0.016 ± 0.001 | 0.013 ± 0.000 | 0.026 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.027 ± 0.001 | - | 0.031 ± 0.001 | 0.031 ± 0.000 | 0.045 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.005 ± 0.001 | - | 0.015 ± 0.014 | 0.010 ± 0.001 | 0.024 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.016 ± 0.004 | - | 0.018 ± 0.001 | 0.015 ± 0.000 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.037 ± 0.000 | - | 0.040 ± 0.000 | 0.051 ± 0.008 | 0.058 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 13.424 ± 2.199 | - | 12.173 ± 0.781 | 12.429 ± 1.910 | 8.432 ± 0.469 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.008 ± 0.032 | - | 0.013 ± 0.001 | 0.014 ± 0.000 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 1.054 ± 0.038 | - | 0.872 ± 0.103 | 1.121 ± 0.036 | 0.668 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.028 ± 0.001 | - | 0.032 ± 0.002 | 0.025 ± 0.000 | 0.043 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.073 ± 0.000 | - | 0.076 ± 0.000 | 0.097 ± 0.005 | 0.191 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.255 ± 0.029 | - | unsupported | 0.329 ± 0.009 | 1.541 ± 0.142 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.813 ± 0.231 | - | unsupported | 1.488 ± 0.280 | 3.030 ± 2.050 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 3.548 ± 0.122 | - | unsupported | 4.264 ± 0.240 | 6.150 ± 0.668 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.705 ± 0.011 | - | unsupported | 1.221 ± 0.520 | 0.921 ± 0.084 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | unsupported | 0.006 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.034 ± 0.007 | - | unsupported | 0.049 ± 0.002 | 0.061 ± 0.066 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.004 ± 0.001 | - | unsupported | 0.008 ± 0.002 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.034 ± 0.016 | - | unsupported | 0.036 ± 0.007 | 0.111 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.005 ± 0.001 | - | unsupported | 0.009 ± 0.001 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.014 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.007 ± 0.000 | - | unsupported | 0.013 ± 0.011 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.015 ± 0.003 | - | unsupported | 0.018 ± 0.002 | 0.026 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | unsupported | 0.007 ± 0.003 | 0.015 ± 0.066 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.009 ± 0.001 | - | unsupported | 0.013 ± 0.011 | 0.019 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.020 ± 0.003 | - | unsupported | 0.023 ± 0.001 | 0.037 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 5.667 ± 0.065 | - | unsupported | 7.031 ± 0.823 | 7.695 ± 1.857 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.003 ± 0.001 | - | unsupported | 0.008 ± 0.002 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.267 ± 0.005 | - | unsupported | 0.342 ± 0.013 | 0.384 ± 0.079 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.015 ± 0.003 | - | unsupported | 0.031 ± 0.009 | 0.030 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.032 ± 0.002 | - | unsupported | 0.038 ± 0.002 | 0.051 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.226 ± 0.010 | - | unsupported | 0.239 ± 0.011 | 1.609 ± 0.286 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 2.287 ± 1.934 | - | unsupported | 1.282 ± 0.158 | 1.917 ± 0.167 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 6.895 ± 1.402 | - | unsupported | 3.623 ± 1.208 | 3.112 ± 0.136 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 0.839 ± 0.068 | - | unsupported | 0.720 ± 0.028 | 0.934 ± 0.059 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.002 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.037 ± 0.003 | - | unsupported | 0.039 ± 0.001 | 0.061 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.001 | 0.016 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.035 ± 0.005 | - | unsupported | 0.041 ± 0.005 | 0.073 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.006 ± 0.001 | - | unsupported | 0.008 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | unsupported | 0.011 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.012 ± 0.001 | - | unsupported | 0.014 ± 0.000 | 0.024 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.002 ± 0.001 | - | unsupported | 0.006 ± 0.000 | 0.017 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.006 ± 0.001 | - | unsupported | 0.010 ± 0.002 | 0.023 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.016 ± 0.000 | - | unsupported | 0.018 ± 0.000 | 0.035 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 8.127 ± 2.093 | - | unsupported | 7.359 ± 1.820 | 5.182 ± 1.168 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.332 ± 0.140 | - | unsupported | 0.298 ± 0.068 | 0.381 ± 0.113 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.013 ± 0.003 | - | unsupported | 0.018 ± 0.004 | 0.077 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.078 ± 0.031 | - | unsupported | 0.032 ± 0.000 | 0.059 ± 0.002 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.752 ± 0.038 | - | unsupported | 0.860 ± 0.051 | 1.261 ± 0.270 | 7.187 ± 4.207 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.054 ± 0.006 | - | unsupported | 0.086 ± 0.015 | 0.149 ± 0.005 | 0.074 ± 0.011 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.107 ± 0.000 | - | unsupported | 0.119 ± 0.014 | 0.204 ± 0.023 | 0.144 ± 0.012 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.249 ± 0.001 | - | unsupported | 0.280 ± 0.015 | 0.366 ± 0.027 | 0.714 ± 0.036 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.918 ± 0.046 | - | unsupported | 0.444 ± 0.001 | 0.960 ± 0.033 | 4.156 ± 1.141 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.073 ± 0.056 | - | unsupported | 0.151 ± 0.033 | 0.191 ± 0.009 | 0.156 ± 0.058 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.303 ± 0.178 | - | unsupported | 0.172 ± 0.024 | 0.220 ± 0.029 | 0.246 ± 0.023 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.425 ± 0.343 | - | unsupported | 0.313 ± 0.028 | 0.455 ± 0.113 | 0.911 ± 0.332 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.209 ± 0.020 | - | unsupported | 0.186 ± 0.022 | 0.414 ± 0.040 | 0.530 ± 0.049 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.065 ± 0.006 | - | unsupported | 0.043 ± 0.004 | 0.087 ± 0.006 | 0.072 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.065 ± 0.000 | - | unsupported | 0.049 ± 0.006 | 0.095 ± 0.006 | 0.117 ± 0.022 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.081 ± 0.000 | - | unsupported | 0.080 ± 0.001 | 0.175 ± 0.014 | 0.246 ± 0.007 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.242 ± 0.016 | - | unsupported | 0.179 ± 0.002 | 0.680 ± 0.129 | 0.577 ± 0.070 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.065 ± 0.005 | - | unsupported | 0.041 ± 0.001 | 0.126 ± 0.020 | 0.083 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.079 ± 0.005 | - | unsupported | 0.048 ± 0.000 | 0.137 ± 0.069 | 0.157 ± 0.066 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.101 ± 0.011 | - | unsupported | 0.077 ± 0.001 | 0.206 ± 0.021 | 0.324 ± 0.054 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.106 ± 0.002 | - | 0.163 ± 0.003 | 0.090 ± 0.017 | 0.197 ± 0.017 | 0.278 ± 0.014 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.038 ± 0.000 | - | 0.046 ± 0.010 | 0.031 ± 0.029 | 0.028 ± 0.005 | 0.046 ± 0.009 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.045 ± 0.002 | - | 0.066 ± 0.029 | 0.030 ± 0.017 | 0.034 ± 0.004 | 0.052 ± 0.011 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.060 ± 0.004 | - | 0.076 ± 0.010 | 0.046 ± 0.014 | 0.059 ± 0.010 | 0.132 ± 0.023 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.125 ± 0.011 | - | 0.233 ± 0.098 | 0.080 ± 0.003 | 0.268 ± 0.015 | 0.437 ± 0.131 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.040 ± 0.006 | - | 0.059 ± 0.014 | 0.024 ± 0.017 | 0.030 ± 0.007 | 0.051 ± 0.039 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.053 ± 0.003 | - | 0.061 ± 0.002 | 0.027 ± 0.006 | 0.038 ± 0.003 | 0.065 ± 0.078 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.097 ± 0.022 | - | 0.124 ± 0.119 | 0.037 ± 0.011 | 0.086 ± 0.155 | 0.127 ± 0.018 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 9.036 ± 0.167 | - | 10.561 ± 0.893 | 25.893 ± 2.962 | 16.362 ± 7.367 | 49.196 ± 15.013 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 12.963 ± 6.450 | - | 12.902 ± 3.843 | 15.849 ± 3.842 | 16.305 ± 0.492 | 385.195 ± 27.014 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 8.498 ± 1.557 | - | 12.651 ± 6.113 | 5.617 ± 0.526 | 8.182 ± 1.639 | 25.252 ± 5.055 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 9.260 ± 6.169 | - | 15.314 ± 6.055 | 10.334 ± 4.158 | 8.987 ± 0.897 | 231.020 ± 30.268 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 6.971 ± 0.156 | - | 6.402 ± 0.729 | 7.268 ± 0.744 | 9.196 ± 1.315 | 7.703 ± 0.736 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 7.248 ± 1.976 | - | 7.136 ± 0.300 | 6.177 ± 0.265 | 8.429 ± 1.242 | 314.409 ± 39.939 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 7.860 ± 0.748 | - | 7.602 ± 1.049 | 6.981 ± 1.427 | 6.817 ± 0.151 | 8.383 ± 3.017 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 8.392 ± 3.963 | - | 11.656 ± 3.823 | 7.160 ± 0.268 | 7.824 ± 0.543 | 254.701 ± 20.324 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 19.760 ± 0.880 | - | 22.390 ± 1.510 | 18.800 ± 1.679 | 28.055 ± 0.958 | 14.628 ± 1.525 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 25.667 ± 1.968 | - | 36.631 ± 7.137 | 22.900 ± 4.172 | 27.950 ± 1.665 | 651.969 ± 277.752 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.296 ± 0.013 | 0.305 ± 0.100 | unsupported | 0.319 ± 0.010 | 0.378 ± 0.003 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 3.695 ± 0.546 | 3.463 ± 0.224 | unsupported | 4.059 ± 1.442 | 4.870 ± 0.682 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.392 ± 0.134 | 0.411 ± 0.079 | unsupported | 0.322 ± 0.028 | 0.460 ± 0.029 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 4.276 ± 0.754 | 4.966 ± 2.491 | unsupported | 5.068 ± 1.141 | 4.870 ± 0.115 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 8.456 ± 0.189 | - | 9.051 ± 0.272 | 7.580 ± 1.841 | 8.626 ± 0.731 | 45.282 ± 3.619 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 20.603 ± 9.875 | - | 16.740 ± 3.412 | 15.252 ± 8.086 | 16.820 ± 8.041 | 417.018 ± 311.980 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 9.119 ± 0.205 | 9.324 ± 0.269 | 6.379 ± 1.508 | 27.288 ± 2.716 | 16.283 ± 7.820 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 15.376 ± 7.253 | 11.027 ± 0.494 | 5.714 ± 1.357 | 28.875 ± 0.611 | 140.127 ± 31.014 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 9.036 ± 1.259 | - | 8.946 ± 0.600 | 15.019 ± 7.590 | 12.494 ± 2.755 | 36.163 ± 25.676 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 8.795 ± 0.411 | - | 10.104 ± 4.717 | 10.801 ± 2.395 | 16.566 ± 1.707 | 253.998 ± 35.829 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.657 ± 0.081 | - | 0.719 ± 0.085 | 2.340 ± 0.078 | 7.870 ± 1.628 | 6.919 ± 1.086 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.786 ± 0.078 | - | 0.827 ± 0.046 | 2.042 ± 0.045 | 5.105 ± 1.120 | 6.704 ± 1.130 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 11.251 ± 0.629 | - | 11.132 ± 0.373 | 9.075 ± 0.569 | 13.366 ± 0.911 | 25.776 ± 9.088 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 18.463 ± 4.578 | - | 21.181 ± 4.820 | 14.527 ± 8.536 | 15.886 ± 1.850 | 979.259 ± 35.678 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 11.672 ± 0.722 | - | 14.694 ± 1.238 | 10.740 ± 1.909 | 12.994 ± 1.511 | 23.047 ± 11.544 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 13.116 ± 1.848 | - | 13.739 ± 3.933 | 13.094 ± 1.548 | 16.710 ± 3.208 | 1006.839 ± 613.188 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.450 ± 0.047 | 1.476 ± 0.035 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 12.944 ± 0.280 | 13.660 ± 0.360 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.674 ± 0.136 | 1.540 ± 0.208 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 19.591 ± 4.068 | 15.548 ± 6.689 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.501 ± 0.327 | 1.472 ± 0.171 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 13.254 ± 1.770 | 11.497 ± 0.205 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.538 ± 0.177 | 1.529 ± 0.050 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 12.966 ± 2.322 | 15.692 ± 2.241 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 8.745 ± 0.910 | - | 8.251 ± 1.206 | 5.928 ± 1.426 | 7.217 ± 0.613 | 22.870 ± 2.846 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 12.148 ± 5.540 | - | 12.964 ± 3.251 | 9.972 ± 3.527 | 9.322 ± 1.083 | 240.306 ± 18.284 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 34.942 ± 0.702 | 34.813 ± 0.509 | 34.995 ± 0.971 | 40.128 ± 5.045 | 80.747 ± 9.048 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 45.572 ± 2.585 | 43.407 ± 1.164 | 35.684 ± 8.323 | 43.655 ± 1.532 | 1791.612 ± 946.564 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 15.609 ± 1.851 | - | 23.450 ± 4.832 | 14.001 ± 1.254 | 63.360 ± 7.144 | 37.017 ± 4.509 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 30.738 ± 7.322 | - | 31.491 ± 9.438 | 13.549 ± 2.619 | 54.910 ± 4.376 | 16.238 ± 4.186 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.171 ± 0.075 | 0.177 ± 0.058 | unsupported | 0.058 ± 0.003 | 0.041 ± 0.024 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 3.153 ± 0.154 | 3.383 ± 0.224 | unsupported | 0.927 ± 0.035 | 0.664 ± 0.266 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.319 ± 0.109 | 0.248 ± 0.061 | unsupported | 0.104 ± 0.032 | 0.072 ± 0.069 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 2.269 ± 0.180 | 2.334 ± 0.286 | unsupported | 0.754 ± 0.055 | 0.335 ± 0.276 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.069 ± 0.000 | 0.086 ± 0.021 | unsupported | 0.089 ± 0.001 | 0.312 ± 0.098 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.653 ± 0.551 | 1.589 ± 0.412 | unsupported | 1.488 ± 0.137 | 3.264 ± 1.319 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.091 ± 0.038 | 0.085 ± 0.048 | unsupported | 0.167 ± 0.044 | 0.179 ± 0.116 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 1.010 ± 0.176 | 1.084 ± 0.273 | unsupported | 1.202 ± 1.187 | 1.733 ± 0.110 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.295 ± 0.025 | 0.246 ± 0.018 | unsupported | 0.327 ± 0.013 | 0.108 ± 0.052 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 5.115 ± 0.646 | 7.001 ± 1.254 | unsupported | 5.191 ± 0.220 | 0.913 ± 0.142 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.277 ± 0.136 | 0.262 ± 0.041 | unsupported | 0.331 ± 0.055 | 0.205 ± 0.172 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 3.415 ± 0.793 | 3.055 ± 0.344 | unsupported | 3.403 ± 0.860 | 0.984 ± 0.031 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.055 ± 0.005 | 0.081 ± 0.024 | unsupported | 0.048 ± 0.002 | 0.083 ± 0.009 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.951 ± 0.437 | 0.694 ± 0.030 | unsupported | 0.609 ± 0.059 | 0.753 ± 0.059 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.184 ± 0.037 | 0.280 ± 0.059 | unsupported | 0.105 ± 0.009 | 0.105 ± 0.019 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 1.243 ± 0.221 | 1.877 ± 0.149 | unsupported | 0.453 ± 0.200 | 0.934 ± 0.411 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.097 ± 0.012 | 0.111 ± 0.027 | unsupported | 0.101 ± 0.006 | 0.129 ± 0.026 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 2.328 ± 0.864 | 2.109 ± 2.158 | unsupported | 1.803 ± 1.396 | 1.554 ± 0.882 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.304 ± 0.027 | 0.411 ± 0.085 | unsupported | 0.144 ± 0.009 | 0.249 ± 0.086 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 2.505 ± 0.361 | 2.095 ± 0.156 | unsupported | 1.131 ± 0.308 | 1.074 ± 0.627 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.114 ± 0.002 | 0.142 ± 0.100 | unsupported | unsupported | 0.166 ± 0.157 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 2.737 ± 0.630 | 2.339 ± 1.176 | unsupported | unsupported | 2.336 ± 0.512 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.423 ± 0.067 | 0.491 ± 0.101 | unsupported | unsupported | 0.178 ± 0.123 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 2.405 ± 0.479 | 2.238 ± 1.015 | unsupported | unsupported | 1.728 ± 0.829 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.014 ± 0.005 | 0.020 ± 0.001 | unsupported | 0.008 ± 0.001 | 0.056 ± 0.009 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.140 ± 0.061 | 0.152 ± 0.250 | unsupported | 0.063 ± 0.008 | 0.326 ± 0.105 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.037 ± 0.009 | 0.021 ± 0.003 | unsupported | 0.066 ± 0.007 | 0.080 ± 0.043 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.153 ± 0.032 | 0.194 ± 0.023 | unsupported | 0.104 ± 0.012 | 0.345 ± 0.060 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.026 ± 0.001 | 0.046 ± 0.002 | unsupported | 0.030 ± 0.001 | 0.078 ± 0.014 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.370 ± 0.079 | 0.419 ± 0.030 | unsupported | 0.391 ± 0.077 | 0.758 ± 0.150 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.065 ± 0.055 | 0.140 ± 0.005 | unsupported | 0.162 ± 0.043 | 0.154 ± 0.028 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.655 ± 0.156 | 0.961 ± 0.113 | unsupported | 0.513 ± 0.092 | 0.416 ± 0.035 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.045 ± 0.001 | 0.049 ± 0.005 | unsupported | 0.085 ± 0.008 | 0.113 ± 0.043 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.903 ± 0.042 | 0.886 ± 0.032 | unsupported | 1.275 ± 0.069 | 0.789 ± 0.054 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.060 ± 0.027 | 0.063 ± 0.030 | unsupported | 0.168 ± 0.042 | 0.188 ± 0.030 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.756 ± 0.246 | 0.755 ± 0.135 | unsupported | 1.062 ± 0.117 | 0.823 ± 0.422 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.046 ± 0.002 | 0.046 ± 0.001 | unsupported | 0.083 ± 0.001 | 0.104 ± 0.029 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.922 ± 0.030 | 0.925 ± 0.031 | unsupported | 1.387 ± 0.054 | 1.518 ± 0.654 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.052 ± 0.043 | 0.076 ± 0.033 | unsupported | 0.149 ± 0.034 | 0.177 ± 0.040 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.828 ± 0.128 | 0.779 ± 0.039 | unsupported | 1.315 ± 0.617 | 0.513 ± 0.023 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.108 ± 0.010 | 0.128 ± 0.014 | unsupported | 0.108 ± 0.004 | 0.145 ± 0.065 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 2.494 ± 0.791 | 2.177 ± 0.713 | unsupported | 2.277 ± 0.963 | 1.887 ± 0.878 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.275 ± 0.070 | 0.411 ± 0.049 | unsupported | 0.145 ± 0.009 | 0.146 ± 0.010 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 1.984 ± 0.152 | 2.213 ± 0.176 | unsupported | 1.170 ± 0.325 | 1.412 ± 0.016 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.235 ± 0.045 | 0.205 ± 0.017 | unsupported | 0.302 ± 0.011 | 0.459 ± 0.138 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 4.008 ± 0.420 | 4.119 ± 0.313 | unsupported | 4.909 ± 0.095 | 4.664 ± 1.004 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.221 ± 0.147 | 0.249 ± 0.095 | unsupported | 0.487 ± 0.052 | 0.308 ± 0.116 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 3.013 ± 0.489 | 2.742 ± 0.383 | unsupported | 3.472 ± 0.426 | 2.719 ± 0.604 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.641 ± 0.128 | 0.850 ± 0.101 | unsupported | 0.132 ± 0.007 | 0.099 ± 0.014 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 12.205 ± 2.572 | 12.302 ± 0.379 | unsupported | 2.741 ± 0.465 | 2.057 ± 0.578 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 1.015 ± 0.164 | 1.375 ± 0.202 | unsupported | 0.236 ± 0.009 | 0.089 ± 0.018 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 11.192 ± 0.552 | 11.108 ± 1.206 | unsupported | 1.702 ± 0.518 | 0.540 ± 0.039 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 22.953 ± 4.019 | - | unsupported | 22.315 ± 2.382 | - | 19.874 ± 2.692 | 19.667 ± 1.531 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 20.450 ± 3.371 | - | unsupported | 20.847 ± 5.287 | - | 14.141 ± 0.478 | 14.401 ± 2.276 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 12.986 ± 0.511 | - | unsupported | 14.991 ± 2.801 | - | 11.906 ± 0.440 | 12.088 ± 0.167 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 14.883 ± 3.250 | - | unsupported | 12.513 ± 0.895 | - | 12.092 ± 2.372 | 10.039 ± 1.079 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 18.540 ± 3.052 | - | unsupported | 15.152 ± 3.695 | - | 13.507 ± 2.775 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 14.467 ± 2.201 | - | unsupported | 11.834 ± 1.464 | - | 11.515 ± 1.737 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 20.333 ± 2.369 | - | unsupported | 23.259 ± 2.195 | - | 17.805 ± 1.316 | 19.212 ± 1.287 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 21.964 ± 2.822 | - | unsupported | 19.713 ± 0.512 | - | 21.328 ± 1.181 | 17.007 ± 3.197 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 17.342 ± 9.227 | - | unsupported | 11.143 ± 1.084 | - | 64.055 ± 10.919 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 13.772 ± 5.353 | - | unsupported | 11.929 ± 1.367 | - | 24.346 ± 6.951 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 13.308 ± 1.761 | - | unsupported | 11.687 ± 1.714 | - | 60.029 ± 14.065 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 18.858 ± 8.011 | - | unsupported | 12.359 ± 0.438 | - | 22.965 ± 4.114 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 22.952 ± 3.555 | - | unsupported | 19.987 ± 1.894 | - | 19.690 ± 0.761 | 19.237 ± 1.922 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 20.126 ± 3.040 | - | unsupported | 18.278 ± 2.468 | - | 21.890 ± 6.363 | 17.090 ± 1.423 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 13.273 ± 3.571 | - | unsupported | 14.364 ± 3.142 | - | 12.197 ± 1.793 | 13.375 ± 0.312 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 14.956 ± 2.988 | - | unsupported | 11.699 ± 1.626 | - | 9.815 ± 0.743 | 10.981 ± 1.347 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.222 ± 0.007 | unsupported | unsupported | 0.286 ± 0.011 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 3.393 ± 0.295 | unsupported | unsupported | 3.164 ± 0.908 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.419 ± 0.118 | unsupported | unsupported | 0.511 ± 0.190 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 3.722 ± 0.864 | unsupported | unsupported | 5.523 ± 2.397 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 18.888 ± 1.240 | - | unsupported | 24.300 ± 2.990 | - | 18.183 ± 0.612 | 20.312 ± 0.784 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 21.105 ± 4.449 | - | unsupported | 21.204 ± 3.142 | - | 19.656 ± 3.740 | 19.494 ± 2.218 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 6.806 ± 1.416 | - | 0.003 ± 0.001 | 9.898 ± 4.004 | 6.043 ± 0.254 | 15.885 ± 3.440 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 10.478 ± 3.021 | - | 0.011 ± 0.007 | 6.192 ± 0.398 | 8.359 ± 3.612 | 22.299 ± 11.222 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 10.021 ± 0.556 | - | 28.501 ± 2.344 | 9.792 ± 2.595 | 9.642 ± 0.989 | 10.519 ± 2.888 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 12.146 ± 3.573 | - | 23.644 ± 4.670 | 9.946 ± 1.268 | 12.120 ± 4.392 | 10.813 ± 3.127 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 11.706 ± 0.310 | - | 12.568 ± 0.914 | 11.823 ± 0.746 | 18.308 ± 0.773 | 10.406 ± 2.053 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 12.358 ± 0.637 | - | 12.788 ± 1.592 | 12.676 ± 5.478 | 19.175 ± 7.920 | 12.958 ± 2.740 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 5.756 ± 0.186 | - | 18.731 ± 1.405 | 14.584 ± 2.744 | 9.717 ± 0.505 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 5.704 ± 0.445 | - | 18.663 ± 0.937 | 10.123 ± 1.071 | 10.400 ± 2.134 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 11.747 ± 0.907 | - | 0.007 ± 0.071 | 11.724 ± 1.166 | 12.074 ± 0.728 | 12.755 ± 1.497 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 13.962 ± 1.654 | - | 0.003 ± 0.007 | 13.193 ± 0.412 | 11.759 ± 2.980 | 16.642 ± 1.854 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 42.890 ± 1.222 | - | 0.029 ± 0.030 | 55.246 ± 14.155 | 63.071 ± 10.854 | 206.982 ± 14.594 | 35.551 ± 1.270 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 28.359 ± 8.396 | - | 0.009 ± 0.018 | 47.217 ± 5.873 | 37.582 ± 9.454 | 219.944 ± 8.582 | 32.196 ± 8.796 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 8.316 ± 0.515 | - | 14.550 ± 0.774 | 9.883 ± 1.475 | 7.586 ± 6.968 | 15.416 ± 6.186 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 5.815 ± 0.532 | - | 12.417 ± 0.663 | 5.957 ± 0.426 | 6.644 ± 0.560 | 9.988 ± 0.984 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 8.476 ± 0.278 | - | 15.233 ± 1.255 | 9.982 ± 2.787 | 6.672 ± 1.139 | 12.170 ± 3.145 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 6.134 ± 0.259 | - | 12.833 ± 0.788 | 6.082 ± 2.149 | 6.604 ± 0.228 | 10.471 ± 0.496 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.005 ± 0.002 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.002 ± 0.000 | - | unsupported | 0.004 ± 0.002 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.002 ± 0.001 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.003 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.004 ± 0.006 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 4.165417 | 260338.56 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 4.127125 | 257945.31 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 4.631875 | 289492.19 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 6.502750 | 406421.88 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 3.893333 | 243333.31 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 4.922750 | 307671.88 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 13.427917 | 839244.81 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 12.578583 | 786161.44 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 11.469542 | 716846.38 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 13.046167 | 815385.44 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 6.040542 | 377533.88 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 5.111208 | 319450.50 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 4.728209 | 295513.06 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 4.884750 | 305296.88 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 77.926750 | 4870421.88 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 64.941458 | 4058841.12 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 59.121167 | 3695072.94 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 55.414209 | 3463388.06 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 7.356250 | 459765.62 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 5.146583 | 321661.44 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 6.279417 | 392463.56 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 6.571042 | 410690.12 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 77.920792 | 4870049.50 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 81.092125 | 5068257.81 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 68.412833 | 4275802.06 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 79.458875 | 4966179.69 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 23.192667 | 1449541.69 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 23.611708 | 1475731.75 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 207.108458 | 12944278.62 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 218.556708 | 13659794.25 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 26.776042 | 1673502.62 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 24.646916 | 1540432.25 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 313.456125 | 19591007.81 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 248.770792 | 15548174.50 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 24.012667 | 1500791.69 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 23.552709 | 1472044.31 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 212.069167 | 13254322.94 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 183.950334 | 11496895.88 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 24.614167 | 1538385.44 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 24.456083 | 1528505.19 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 207.448209 | 12965513.06 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 251.070958 | 15691934.88 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.647708 | 41370.44 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.723750 | 58183.59 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 10.924791 | 170699.86 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 11.349000 | 177328.12 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.310292 | 663786.50 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.419542 | 927442.75 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 25.226833 | 3153354.12 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 27.067083 | 3383385.38 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.612375 | 72068.36 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.652750 | 103949.22 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 20.384541 | 318508.45 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 15.854000 | 247718.75 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.683041 | 335380.12 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.031291 | 753911.38 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 18.149791 | 2268723.88 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 18.670542 | 2333817.75 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 19.984917 | 312264.33 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.711125 | 89236.33 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.435750 | 69308.59 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.499000 | 85921.88 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 26.110250 | 3263781.25 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.900125 | 1487515.62 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.224541 | 1653067.62 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 12.713833 | 1589229.12 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.475417 | 179303.39 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 10.670292 | 166723.31 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.806750 | 90730.47 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.454209 | 85222.02 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 13.865625 | 1733203.12 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 9.616750 | 1202093.75 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.081542 | 1010192.75 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 8.674125 | 1084265.62 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.926375 | 108224.61 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 20.941917 | 327217.45 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 18.904083 | 295376.30 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 15.735333 | 245864.58 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.300334 | 912541.75 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 41.530250 | 5191281.25 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 40.922417 | 5115302.12 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 56.009333 | 7001166.62 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.150250 | 205472.66 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 21.165791 | 330715.48 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 17.740125 | 277189.45 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 16.750542 | 261727.22 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.875459 | 984432.38 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 27.222666 | 3402833.25 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 27.321792 | 3415224.00 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 24.438292 | 3054786.50 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.321500 | 83148.44 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.065917 | 47904.95 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.503167 | 54736.98 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.180792 | 80949.88 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.023166 | 752895.75 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.870625 | 608828.12 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.605375 | 950671.88 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.555042 | 694380.25 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.721583 | 105024.73 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.689917 | 104529.95 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 11.766708 | 183854.81 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 17.888791 | 279512.36 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.475834 | 934479.25 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.623625 | 452953.12 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.940500 | 1242562.50 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 15.017625 | 1877203.12 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.276500 | 129320.31 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.441042 | 100641.28 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.190792 | 96731.12 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 7.085667 | 110713.55 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.429167 | 1553645.88 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.421375 | 1802671.88 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 18.626125 | 2328265.62 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 16.869667 | 2108708.38 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 15.908208 | 248565.75 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 9.217375 | 144021.48 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 19.464125 | 304126.95 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 26.328042 | 411375.66 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.588541 | 1073567.62 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 9.051291 | 1131411.38 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 20.042459 | 2505307.38 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 16.757041 | 2094630.13 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.595041 | 165547.52 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 7.278250 | 113722.66 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.107375 | 142302.73 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.686042 | 2335755.25 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 21.894750 | 2736843.75 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 18.710209 | 2338776.12 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.402167 | 178158.86 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 27.098750 | 423417.97 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 31.417500 | 490898.44 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 13.820208 | 1727526.00 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 19.241000 | 2405125.00 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 17.907209 | 2238401.12 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 3.578625 | 55916.02 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.537417 | 8397.14 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.915292 | 14301.44 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.291166 | 20174.47 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 2.609500 | 326187.50 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.501208 | 62651.00 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 1.121042 | 140130.25 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 1.217625 | 152203.12 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.095750 | 79621.09 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.204250 | 65691.41 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.349667 | 36713.55 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.347750 | 21058.59 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.756709 | 344588.62 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.829583 | 103697.88 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.223917 | 152989.62 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.553791 | 194223.88 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.991208 | 77987.62 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 1.904875 | 29763.67 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.645792 | 25715.50 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.962208 | 46284.50 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.062000 | 757750.00 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 3.131333 | 391416.62 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.960208 | 370026.00 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.351292 | 418911.50 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.828167 | 153565.11 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 10.339583 | 161555.98 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.175334 | 65239.59 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 8.988042 | 140438.16 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.329833 | 416229.12 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.103917 | 512989.62 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.243916 | 655489.50 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.686709 | 960838.62 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.221042 | 112828.78 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.414708 | 84604.81 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.848750 | 44511.72 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.109042 | 48578.78 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.311667 | 788958.38 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.198083 | 1274760.38 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.227416 | 903427.00 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.089709 | 886213.62 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.030708 | 187979.81 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 10.722917 | 167545.58 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.853167 | 60205.73 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.000083 | 62501.30 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.583291 | 822911.38 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.497625 | 1062203.12 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.048208 | 756026.00 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 6.039791 | 754973.88 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.635375 | 103677.73 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.316833 | 83075.52 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.959917 | 46248.70 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.975375 | 46490.23 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.144667 | 1518083.38 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.094833 | 1386854.12 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.377917 | 922239.62 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.398291 | 924786.38 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.327041 | 176985.02 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 9.556792 | 149324.88 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.299666 | 51557.28 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.845417 | 75709.64 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.107208 | 513401.00 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 10.521083 | 1315135.38 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.621500 | 827687.50 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 6.229375 | 778671.88 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.258916 | 144670.56 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.941875 | 108466.80 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.881167 | 107518.23 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.174708 | 127729.81 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 15.096167 | 1887020.88 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 18.213083 | 2276635.38 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 19.950292 | 2493786.50 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 17.419625 | 2177453.12 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.317375 | 145583.98 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 9.301708 | 145339.19 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 17.581041 | 274703.77 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 26.333792 | 411465.50 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.294166 | 1411770.75 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 9.356375 | 1169546.88 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 15.875417 | 1984427.12 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 17.700167 | 2212520.88 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 29.363750 | 458808.59 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 19.333083 | 302079.42 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 15.032042 | 234875.66 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 13.130750 | 205167.97 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 37.315375 | 4664421.88 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 39.269750 | 4908718.75 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 32.061875 | 4007734.38 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 32.949750 | 4118718.75 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 19.715542 | 308055.34 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 31.140667 | 486572.92 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 14.144834 | 221013.03 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 15.916125 | 248689.45 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 21.749834 | 2718729.25 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 27.773375 | 3471671.88 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 24.105583 | 3013197.88 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 21.933125 | 2741640.62 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.336625 | 99009.77 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.461958 | 132218.09 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 41.046500 | 641351.56 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 54.390542 | 849852.22 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 16.457292 | 2057161.50 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 21.924709 | 2740588.62 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 97.636125 | 12204515.62 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 98.416541 | 12302067.62 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.668917 | 88576.83 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 15.117583 | 236212.23 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 64.989958 | 1015468.09 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 87.994000 | 1374906.25 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.316750 | 539593.75 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 13.612792 | 1701599.00 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 89.537250 | 11192156.25 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 88.860334 | 11107541.75 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 4.576500 | 286031.25 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 3.552250 | 222015.62 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 50.623708 | 3163981.75 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 54.292958 | 3393309.88 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 8.183375 | 511460.94 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 6.711125 | 419445.31 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 88.362708 | 5522669.25 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 59.544083 | 3721505.19 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.080458 | 5028.62 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.001292 | 80.75 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.015541 | 971.31 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.001542 | 96.38 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000125 | 7.81 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.064541 | 4033.81 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.027625 | 1726.56 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000209 | 13.06 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.013208 | 825.50 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.029833 | 1864.56 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000250 | 15.62 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.035750 | 2234.38 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.003250 | 203.12 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000125 | 7.81 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.011334 | 708.38 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.002791 | 174.44 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000250 | 15.62 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.062417 | 3901.06 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003417 | 213.56 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.011625 | 726.56 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002667 | 166.69 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `jax-cpu` is 42.1x faster than the slowest successful cell (`julia-base`, 72.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `pytorch-cpu` is 16.9x faster than the slowest successful cell (`julia-base`, 72.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-direct` is 17.1x faster than the slowest successful cell (`julia-base`, 72.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-trace` is 11.6x faster than the slowest successful cell (`julia-base`, 72.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `jax-cpu` is 19.1x faster than the slowest successful cell (`julia-base`, 435.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `pytorch-cpu` is 41.6x faster than the slowest successful cell (`julia-base`, 435.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-direct` is 34.1x faster than the slowest successful cell (`julia-base`, 435.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-trace` is 33.1x faster than the slowest successful cell (`julia-base`, 435.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 23.8x faster than the slowest successful cell (`julia-base`, 174.349 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 23.2x faster than the slowest successful cell (`julia-base`, 174.349 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-direct` is 20.6x faster than the slowest successful cell (`julia-base`, 174.349 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-trace` is 21.4x faster than the slowest successful cell (`julia-base`, 174.349 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 110.5x faster than the slowest successful cell (`julia-base`, 252.418 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 41.3x faster than the slowest successful cell (`julia-base`, 252.418 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-direct` is 33.1x faster than the slowest successful cell (`julia-base`, 252.418 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-trace` is 28.8x faster than the slowest successful cell (`julia-base`, 252.418 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `jax-cpu` is 91.6x faster than the slowest successful cell (`julia-base`, 626.394 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `pytorch-cpu` is 90.5x faster than the slowest successful cell (`julia-base`, 626.394 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-direct` is 78.1x faster than the slowest successful cell (`julia-base`, 626.394 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-trace` is 76.2x faster than the slowest successful cell (`julia-base`, 626.394 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 16.356 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 18.9x faster than the slowest successful cell (`pytorch-cpu`, 16.356 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 18.2x faster than the slowest successful cell (`pytorch-cpu`, 16.356 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 11.623 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 11.623 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 22.0x faster than the slowest successful cell (`pytorch-cpu`, 17.694 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 17.4x faster than the slowest successful cell (`pytorch-cpu`, 17.694 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 18.2x faster than the slowest successful cell (`pytorch-cpu`, 17.694 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 17.4x faster than the slowest successful cell (`pytorch-cpu`, 12.380 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 16.0x faster than the slowest successful cell (`pytorch-cpu`, 12.380 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 15.3x faster than the slowest successful cell (`pytorch-cpu`, 12.380 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 614.3x faster than the slowest successful cell (`pytorch-cpu`, 1.715 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 552.6x faster than the slowest successful cell (`julia-base`, 1.589 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.007 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 24.4x faster than the slowest successful cell (`jax-cpu`, 0.045 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 12.9x faster than the slowest successful cell (`jax-cpu`, 0.009 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`3x2x2`): `tenferro-direct` is 11.6x faster than the slowest successful cell (`jax-cpu`, 0.009 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `pytorch-cpu` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 32.6x faster than the slowest successful cell (`jax-cpu`, 0.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 14.0x faster than the slowest successful cell (`jax-cpu`, 0.019 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 16.2x faster than the slowest successful cell (`jax-cpu`, 0.035 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 56.6x faster than the slowest successful cell (`pytorch-cpu`, 0.040 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 41.1x faster than the slowest successful cell (`pytorch-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-trace` is 13.4x faster than the slowest successful cell (`pytorch-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 98.6x faster than the slowest successful cell (`pytorch-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 95.6x faster than the slowest successful cell (`pytorch-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 15.8x faster than the slowest successful cell (`pytorch-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 43.3x faster than the slowest successful cell (`pytorch-cpu`, 0.038 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 83.1x faster than the slowest successful cell (`pytorch-cpu`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 16.5x faster than the slowest successful cell (`pytorch-cpu`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `jax-cpu` is 13.7x faster than the slowest successful cell (`pytorch-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 94.3x faster than the slowest successful cell (`pytorch-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 21.9x faster than the slowest successful cell (`pytorch-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 46.9x faster than the slowest successful cell (`pytorch-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 58.6x faster than the slowest successful cell (`pytorch-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 50.5x faster than the slowest successful cell (`pytorch-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 37.3x faster than the slowest successful cell (`pytorch-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 29.5x faster than the slowest successful cell (`pytorch-cpu`, 0.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 28.5x faster than the slowest successful cell (`jax-cpu`, 0.064 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.064 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 22.9x faster than the slowest successful cell (`jax-cpu`, 0.032 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 16.6x faster than the slowest successful cell (`jax-cpu`, 0.032 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`1x8x8`): `tenferro-direct` is 13.1x faster than the slowest successful cell (`jax-cpu`, 0.043 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 14.9x faster than the slowest successful cell (`jax-cpu`, 0.027 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`1x64x24`): `tenferro-direct` is 11.0x faster than the slowest successful cell (`jax-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 13.6x faster than the slowest successful cell (`jax-cpu`, 0.045 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `pytorch-cpu` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 25.1x faster than the slowest successful cell (`jax-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 14.5x faster than the slowest successful cell (`jax-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 24.3x faster than the slowest successful cell (`jax-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-trace` is 14.7x faster than the slowest successful cell (`jax-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `jax-cpu` is 13.8x faster than the slowest successful cell (`pytorch-cpu`, 0.131 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 50.1x faster than the slowest successful cell (`pytorch-cpu`, 0.131 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 25.0x faster than the slowest successful cell (`pytorch-cpu`, 0.131 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 14.3x faster than the slowest successful cell (`jax-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 17.0x faster than the slowest successful cell (`pytorch-cpu`, 0.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 18.6x faster than the slowest successful cell (`pytorch-cpu`, 0.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 36.9x faster than the slowest successful cell (`pytorch-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 16.0x faster than the slowest successful cell (`pytorch-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 21.1x faster than the slowest successful cell (`pytorch-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 19.9x faster than the slowest successful cell (`jax-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 25.2x faster than the slowest successful cell (`pytorch-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 21.4x faster than the slowest successful cell (`pytorch-cpu`, 0.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 20.4x faster than the slowest successful cell (`jax-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 39.4x faster than the slowest successful cell (`pytorch-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 20.3x faster than the slowest successful cell (`pytorch-cpu`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 13.2x faster than the slowest successful cell (`pytorch-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 21.2x faster than the slowest successful cell (`pytorch-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 10.4x faster than the slowest successful cell (`pytorch-cpu`, 0.037 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x2x2,rhs=1`): `tenferro-direct` is 10.4x faster than the slowest successful cell (`tenferro-trace`, 0.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x32x32,rhs=1`): `tenferro-direct` is 27.1x faster than the slowest successful cell (`jax-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x32x32,rhs=1`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`3x4x4,rhs=1`): `tenferro-direct` is 21.3x faster than the slowest successful cell (`tenferro-trace`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`4x4x4,rhs=1`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.023 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x2x2,rhs=1`): `tenferro-direct` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.035 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (c64, threads=4, shape=`4x2x2`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (f64, threads=1, shape=`1x4x4`): `pytorch-cpu` is 11.9x faster than the slowest successful cell (`jax-cpu`, 0.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 17.0x faster than the slowest successful cell (`jax-cpu`, 0.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (f64, threads=1, shape=`1x4x4`): `tenferro-trace` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `jax-cpu` is 23.6x faster than the slowest successful cell (`julia-base`, 385.195 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `pytorch-cpu` is 24.3x faster than the slowest successful cell (`julia-base`, 385.195 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-direct` is 29.7x faster than the slowest successful cell (`julia-base`, 385.195 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-trace` is 29.9x faster than the slowest successful cell (`julia-base`, 385.195 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 25.7x faster than the slowest successful cell (`julia-base`, 231.020 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 22.4x faster than the slowest successful cell (`julia-base`, 231.020 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 24.9x faster than the slowest successful cell (`julia-base`, 231.020 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 15.1x faster than the slowest successful cell (`julia-base`, 231.020 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `jax-cpu` is 37.3x faster than the slowest successful cell (`julia-base`, 314.409 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `pytorch-cpu` is 50.9x faster than the slowest successful cell (`julia-base`, 314.409 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-direct` is 43.4x faster than the slowest successful cell (`julia-base`, 314.409 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-trace` is 44.1x faster than the slowest successful cell (`julia-base`, 314.409 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `jax-cpu` is 32.6x faster than the slowest successful cell (`julia-base`, 254.701 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `pytorch-cpu` is 35.6x faster than the slowest successful cell (`julia-base`, 254.701 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-direct` is 30.3x faster than the slowest successful cell (`julia-base`, 254.701 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-trace` is 21.9x faster than the slowest successful cell (`julia-base`, 254.701 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `jax-cpu` is 23.3x faster than the slowest successful cell (`julia-base`, 651.969 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 28.5x faster than the slowest successful cell (`julia-base`, 651.969 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-direct` is 25.4x faster than the slowest successful cell (`julia-base`, 651.969 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-trace` is 17.8x faster than the slowest successful cell (`julia-base`, 651.969 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 24.8x faster than the slowest successful cell (`julia-base`, 417.018 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 27.3x faster than the slowest successful cell (`julia-base`, 417.018 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-direct` is 20.2x faster than the slowest successful cell (`julia-base`, 417.018 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-trace` is 24.9x faster than the slowest successful cell (`julia-base`, 417.018 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 24.5x faster than the slowest successful cell (`julia-base`, 140.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-trace` is 12.7x faster than the slowest successful cell (`julia-base`, 140.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 15.3x faster than the slowest successful cell (`julia-base`, 253.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 23.5x faster than the slowest successful cell (`julia-base`, 253.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 28.9x faster than the slowest successful cell (`julia-base`, 253.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 25.1x faster than the slowest successful cell (`julia-base`, 253.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 12.0x faster than the slowest successful cell (`jax-cpu`, 7.870 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`jax-cpu`, 7.870 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `jax-cpu` is 61.6x faster than the slowest successful cell (`julia-base`, 979.259 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 67.4x faster than the slowest successful cell (`julia-base`, 979.259 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 53.0x faster than the slowest successful cell (`julia-base`, 979.259 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 46.2x faster than the slowest successful cell (`julia-base`, 979.259 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `jax-cpu` is 60.3x faster than the slowest successful cell (`julia-base`, 1006.839 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 76.9x faster than the slowest successful cell (`julia-base`, 1006.839 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 76.8x faster than the slowest successful cell (`julia-base`, 1006.839 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 73.3x faster than the slowest successful cell (`julia-base`, 1006.839 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 25.8x faster than the slowest successful cell (`julia-base`, 240.306 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 24.1x faster than the slowest successful cell (`julia-base`, 240.306 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 19.8x faster than the slowest successful cell (`julia-base`, 240.306 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 18.5x faster than the slowest successful cell (`julia-base`, 240.306 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 41.0x faster than the slowest successful cell (`julia-base`, 1791.612 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 50.2x faster than the slowest successful cell (`julia-base`, 1791.612 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-eager` is 39.3x faster than the slowest successful cell (`julia-base`, 1791.612 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-trace` is 41.3x faster than the slowest successful cell (`julia-base`, 1791.612 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 15.5x faster than the slowest successful cell (`tenferro-eager`, 1.375 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 20.7x faster than the slowest successful cell (`tenferro-direct`, 11.192 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 4592.3x faster than the slowest successful cell (`julia-base`, 15.885 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 2107.1x faster than the slowest successful cell (`julia-base`, 22.299 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 1749.2x faster than the slowest successful cell (`julia-base`, 12.755 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 5256.6x faster than the slowest successful cell (`julia-base`, 16.642 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 7178.6x faster than the slowest successful cell (`julia-base`, 206.982 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 23566.3x faster than the slowest successful cell (`julia-base`, 219.944 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 62.1x faster than the slowest successful cell (`pytorch-cpu`, 0.005 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 504.3x faster than the slowest successful cell (`pytorch-cpu`, 0.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 143.5x faster than the slowest successful cell (`tenferro-direct`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 139.6x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-direct` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 88.5x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 243.8x faster than the slowest successful cell (`pytorch-cpu`, 0.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `tenferro-direct` is 18.2x faster than the slowest successful cell (`pytorch-cpu`, 0.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 145.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 6006.1 GB/s at 0.003 ms. The conservative physical lower bound is 0.168 ms, so the cell is 60.1x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 5832.7 GB/s at 0.003 ms. The conservative physical lower bound is 0.042 ms, so the cell is 14.6x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 77604.9 GB/s at 0.003 ms. The conservative physical lower bound is 2.684 ms, so the cell is 776.0x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 25364.8 GB/s at 0.011 ms. The conservative physical lower bound is 0.671 ms, so the cell is 63.4x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 36812.3 GB/s at 0.007 ms. The conservative physical lower bound is 2.684 ms, so the cell is 368.1x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 84786.9 GB/s at 0.003 ms. The conservative physical lower bound is 0.671 ms, so the cell is 212.0x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 4655.0 GB/s at 0.029 ms. The conservative physical lower bound is 1.342 ms, so the cell is 46.6x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 14381.0 GB/s at 0.009 ms. The conservative physical lower bound is 0.336 ms, so the cell is 36.0x faster than the bound.

# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20261124_182125/run.yaml`
- Timestamp: `2026-11-24T18:21:25.239363+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20261124_182125`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

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

- CSV: `data/results/mac-cpu/cpu/public_api/20261124_182125/cpu_public_api_t1_20261124_182125.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20261124_182125/cpu_public_api_t4_20261124_182125.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 5.087 ± 0.327 | - | 4.970 ± 0.423 | 4.094 ± 0.208 | 1.549 ± 0.297 | 4.801 ± 4.636 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 3.687 ± 0.865 | - | 2.967 ± 0.609 | 3.895 ± 0.191 | 1.463 ± 0.559 | 71.381 ± 15.046 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 12.738 ± 1.275 | - | 11.558 ± 3.689 | 17.240 ± 3.937 | 13.503 ± 2.778 | 13.899 ± 8.432 | 13.787 ± 3.937 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 11.225 ± 3.695 | - | 10.275 ± 4.001 | 14.230 ± 2.308 | 10.609 ± 0.732 | 10.504 ± 1.494 | 14.283 ± 11.240 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 46.323 ± 1.805 | - | 48.379 ± 9.534 | 19.728 ± 0.812 | 42.413 ± 9.331 | 47.952 ± 1.874 | 51.433 ± 3.630 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 21.165 ± 0.456 | - | 23.444 ± 3.523 | 10.324 ± 2.460 | 18.781 ± 1.724 | 50.408 ± 19.134 | 26.367 ± 1.259 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 70.350 ± 3.973 | - | 68.751 ± 3.378 | 11.486 ± 1.278 | 49.405 ± 4.480 | 62.244 ± 2.511 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 24.498 ± 4.845 | - | 28.865 ± 5.170 | 10.737 ± 0.711 | 19.907 ± 2.054 | 39.109 ± 8.747 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 69.901 ± 3.731 | - | unsupported | 12.695 ± 1.873 | 49.783 ± 2.166 | 64.636 ± 12.093 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 30.221 ± 3.808 | - | unsupported | 11.336 ± 0.604 | 19.622 ± 2.734 | 29.545 ± 20.439 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 7.203 ± 0.113 | - | 7.415 ± 1.365 | 11.330 ± 0.367 | 20.554 ± 0.646 | 8.597 ± 0.643 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 9.842 ± 1.360 | - | 8.975 ± 2.996 | 11.186 ± 0.288 | 20.388 ± 1.188 | 445.424 ± 16.631 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 36.183 ± 5.086 | - | 36.488 ± 4.793 | 34.808 ± 0.503 | 44.864 ± 0.885 | 50.207 ± 4.496 | 46.448 ± 2.453 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 13.508 ± 2.097 | - | 16.492 ± 6.221 | 15.976 ± 2.309 | 21.804 ± 1.637 | 50.902 ± 14.465 | 16.219 ± 7.244 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 61.221 ± 16.145 | - | 53.957 ± 14.452 | 56.374 ± 0.791 | 88.611 ± 4.953 | 88.438 ± 17.413 | 97.824 ± 11.831 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 22.049 ± 4.060 | - | 24.035 ± 3.430 | 25.870 ± 6.311 | 40.634 ± 5.196 | 79.998 ± 5.594 | 32.141 ± 5.849 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 10.237 ± 2.662 | - | 10.910 ± 3.414 | 12.079 ± 2.593 | 13.192 ± 3.231 | 10.926 ± 2.737 | 11.337 ± 2.547 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 11.106 ± 1.043 | - | 11.664 ± 1.674 | 10.157 ± 1.392 | 8.764 ± 0.728 | 11.522 ± 1.045 | 7.656 ± 3.710 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.155 ± 0.154 | - | 7.836 ± 0.350 | 6.104 ± 0.130 | 5.391 ± 1.761 | 8.336 ± 0.066 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 1.442 ± 0.183 | - | 3.816 ± 0.311 | 3.070 ± 0.181 | 2.988 ± 0.167 | 12.327 ± 1.760 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 9.663 ± 0.156 | - | 9.603 ± 0.334 | 7.459 ± 0.143 | 11.578 ± 2.448 | 9.265 ± 0.691 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 7.431 ± 2.143 | - | 6.525 ± 1.574 | 7.139 ± 0.318 | 6.753 ± 0.446 | 160.256 ± 191.471 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 6.279 ± 0.215 | - | 6.377 ± 0.627 | 5.152 ± 0.366 | 2.717 ± 0.267 | 6.574 ± 0.287 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 5.915 ± 0.406 | - | 4.331 ± 0.748 | 5.546 ± 0.549 | 2.558 ± 0.384 | 124.435 ± 20.041 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 8.293 ± 0.208 | - | 8.319 ± 0.285 | 7.591 ± 0.150 | 9.326 ± 3.132 | 7.530 ± 0.305 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 7.776 ± 0.415 | - | 8.375 ± 0.367 | 7.715 ± 0.588 | 6.776 ± 0.571 | 590.886 ± 45.712 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 71.306 ± 10.873 | - | 69.870 ± 1.833 | 11.813 ± 1.424 | 56.742 ± 7.218 | 63.242 ± 4.340 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 32.731 ± 2.403 | - | 28.915 ± 5.499 | 11.093 ± 1.130 | 21.201 ± 7.467 | 41.669 ± 3.604 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 53.533 ± 2.592 | - | unsupported | 11.758 ± 1.414 | 56.040 ± 2.353 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 19.344 ± 3.037 | - | unsupported | 10.412 ± 2.830 | 20.910 ± 5.800 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 13.250 ± 3.171 | - | 14.866 ± 1.764 | 13.755 ± 2.726 | 16.031 ± 1.559 | 13.992 ± 3.172 | 15.080 ± 4.610 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 13.913 ± 2.277 | - | 11.498 ± 2.096 | 14.574 ± 3.714 | 12.824 ± 2.565 | 13.827 ± 3.853 | 16.224 ± 6.273 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 32.012 ± 21.591 | - | 22.012 ± 5.910 | 21.621 ± 2.189 | 29.618 ± 1.652 | 26.664 ± 11.814 | 34.127 ± 10.665 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 22.853 ± 2.805 | - | 25.717 ± 2.446 | 24.144 ± 2.559 | 26.896 ± 4.834 | 19.379 ± 8.713 | 20.995 ± 2.833 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 26.323 ± 0.982 | - | 36.327 ± 1.182 | 51.663 ± 10.844 | 32.195 ± 7.551 | 55.517 ± 13.543 | 62.842 ± 4.087 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 19.285 ± 4.859 | - | 15.217 ± 2.315 | 23.851 ± 4.080 | 14.859 ± 1.685 | 51.017 ± 6.673 | 27.791 ± 1.357 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 9.708 ± 4.565 | - | 9.392 ± 1.358 | 7.596 ± 2.536 | 11.896 ± 4.137 | 7.947 ± 2.493 | 8.348 ± 5.990 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 9.079 ± 1.785 | - | 7.802 ± 2.386 | 7.262 ± 0.939 | 8.137 ± 0.669 | 5.754 ± 0.717 | 5.937 ± 6.927 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 17.434 ± 2.725 | - | 17.410 ± 4.633 | 18.320 ± 3.628 | 18.247 ± 2.941 | 18.085 ± 7.758 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 13.784 ± 3.196 | - | 13.158 ± 2.004 | 16.462 ± 1.299 | 20.021 ± 4.870 | 26.819 ± 3.736 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 31.715 ± 1.484 | - | 31.910 ± 5.406 | 41.830 ± 8.985 | 28.534 ± 0.090 | 37.491 ± 7.793 | 39.158 ± 1.931 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 16.921 ± 4.437 | - | 15.366 ± 5.773 | 18.539 ± 6.007 | 21.358 ± 5.725 | 41.547 ± 4.591 | 16.134 ± 2.804 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 20.738 ± 7.358 | - | 20.375 ± 7.372 | 22.874 ± 9.581 | 34.277 ± 2.345 | 25.906 ± 23.492 | 21.913 ± 3.632 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 19.306 ± 5.109 | - | 18.541 ± 3.713 | 22.214 ± 3.794 | 25.600 ± 6.630 | 25.415 ± 3.131 | 23.303 ± 5.584 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 17.751 ± 7.649 | - | 18.844 ± 13.270 | 27.045 ± 0.407 | 17.235 ± 0.945 | 33.140 ± 1.498 | 34.913 ± 1.402 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 8.608 ± 4.091 | - | 11.219 ± 3.692 | 16.499 ± 2.109 | 11.397 ± 4.614 | 35.642 ± 7.160 | 16.045 ± 3.268 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 26.626 ± 5.066 | - | 27.750 ± 3.895 | 32.381 ± 8.758 | 17.649 ± 0.918 | 26.718 ± 2.494 | 26.714 ± 9.542 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 13.671 ± 2.487 | - | 13.744 ± 1.353 | 14.310 ± 1.186 | 14.532 ± 6.433 | 26.184 ± 1.413 | 12.450 ± 5.503 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 32.305 ± 11.984 | - | 42.977 ± 16.852 | 41.937 ± 1.905 | 26.378 ± 3.387 | 43.163 ± 1.631 | 45.597 ± 9.349 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 15.170 ± 2.799 | - | 15.172 ± 0.457 | 24.773 ± 6.692 | 15.432 ± 6.710 | 45.504 ± 6.683 | 19.127 ± 6.111 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 16.181 ± 3.669 | - | 15.581 ± 0.409 | 23.585 ± 3.707 | 18.213 ± 0.804 | 23.517 ± 1.507 | 23.664 ± 1.334 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 9.560 ± 3.132 | - | 7.965 ± 0.127 | 9.400 ± 0.176 | 11.779 ± 6.687 | 21.742 ± 0.487 | 8.452 ± 1.596 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.286 ± 0.145 | 0.441 ± 0.256 | unsupported | unsupported | 0.309 ± 0.208 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.291 ± 0.042 | 0.303 ± 0.055 | unsupported | unsupported | 0.451 ± 0.023 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 20.928 ± 5.390 | - | 25.016 ± 10.809 | 21.279 ± 6.784 | 30.469 ± 6.308 | 22.207 ± 9.682 | 23.075 ± 2.393 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 23.720 ± 2.760 | - | 21.008 ± 3.384 | 22.345 ± 1.903 | 24.437 ± 4.096 | 33.480 ± 38.381 | 21.741 ± 2.432 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 24.637 ± 10.235 | - | 23.097 ± 3.781 | 19.656 ± 6.443 | 30.020 ± 4.668 | 24.426 ± 10.909 | 27.399 ± 13.325 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 18.053 ± 2.851 | - | 18.852 ± 2.053 | 22.790 ± 1.923 | 25.652 ± 3.691 | 28.208 ± 9.230 | 32.035 ± 11.610 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 23.757 ± 8.768 | - | 21.059 ± 3.455 | 22.706 ± 5.733 | 32.820 ± 6.486 | 32.669 ± 19.947 | 25.903 ± 5.418 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 23.290 ± 10.676 | - | 18.460 ± 7.998 | 22.105 ± 2.859 | 25.844 ± 4.536 | 36.259 ± 10.747 | 23.095 ± 1.762 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 12.177 ± 6.115 | - | 15.068 ± 4.013 | 13.637 ± 1.470 | 17.891 ± 2.567 | 14.711 ± 4.351 | 13.542 ± 3.674 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 10.855 ± 2.439 | - | 14.083 ± 2.686 | 14.350 ± 3.018 | 14.590 ± 5.538 | 25.713 ± 17.552 | 19.265 ± 5.359 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 36.301 ± 3.322 | - | 38.234 ± 5.770 | 75.021 ± 7.468 | 45.773 ± 0.746 | 97.665 ± 30.596 | 79.860 ± 8.142 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 18.064 ± 4.106 | - | 18.471 ± 7.857 | 28.778 ± 4.081 | 19.137 ± 7.463 | 85.787 ± 8.324 | 40.371 ± 11.839 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 10.784 ± 7.136 | - | 11.093 ± 6.456 | 12.319 ± 0.341 | 24.960 ± 1.493 | 7.620 ± 4.858 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 5.564 ± 0.801 | - | 6.106 ± 1.409 | 7.796 ± 1.028 | 15.765 ± 1.933 | 5.533 ± 0.909 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.128 ± 0.109 | - | 1.109 ± 0.425 | 21.986 ± 3.833 | 3.693 ± 1.446 | 1.189 ± 0.124 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 1.137 ± 0.076 | - | 0.822 ± 0.027 | 13.655 ± 3.092 | 2.024 ± 0.348 | 0.580 ± 0.030 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 1.046 ± 0.481 | - | 0.912 ± 0.040 | 1.905 ± 0.160 | 1.237 ± 0.015 | 1.056 ± 0.115 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 2.254 ± 0.710 | - | 1.423 ± 0.115 | 1.079 ± 0.122 | 1.452 ± 0.617 | 0.867 ± 0.048 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 12.035 ± 7.540 | - | 9.681 ± 2.047 | 12.173 ± 0.460 | 23.827 ± 0.215 | 6.342 ± 1.499 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 7.582 ± 2.118 | - | 5.635 ± 1.535 | 6.461 ± 1.644 | 15.967 ± 3.135 | 4.882 ± 0.085 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.255 ± 0.787 | - | 1.611 ± 0.380 | 20.284 ± 7.315 | 3.354 ± 1.944 | 1.021 ± 0.045 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.698 ± 0.019 | - | 0.771 ± 0.084 | 13.667 ± 2.244 | 2.809 ± 1.168 | 0.832 ± 0.082 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 4.799 ± 0.379 | - | 5.410 ± 2.120 | 6.491 ± 0.316 | 4.491 ± 0.788 | 4.378 ± 1.146 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 8.220 ± 0.881 | - | 7.086 ± 2.496 | 3.728 ± 0.309 | 4.134 ± 0.957 | 3.119 ± 0.273 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 8.648 ± 0.410 | - | 8.089 ± 1.098 | 9.918 ± 5.019 | 44.298 ± 2.589 | 6.616 ± 2.338 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 8.253 ± 3.146 | - | 7.405 ± 1.671 | 6.917 ± 1.533 | 28.010 ± 5.119 | 6.744 ± 0.574 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.865 ± 0.076 | - | 1.525 ± 0.088 | 5.344 ± 0.506 | 2.752 ± 0.369 | 1.074 ± 0.251 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 1.255 ± 0.432 | - | 1.015 ± 0.497 | 2.889 ± 0.322 | 1.716 ± 0.799 | 0.920 ± 0.071 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.305 ± 0.135 | - | 1.129 ± 0.330 | 0.820 ± 0.096 | 1.181 ± 0.098 | 0.914 ± 0.079 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 2.196 ± 0.099 | - | 2.191 ± 0.312 | 0.843 ± 0.020 | 1.081 ± 0.485 | 0.886 ± 0.036 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 12.067 ± 7.242 | - | 8.963 ± 2.642 | 9.818 ± 5.413 | 32.925 ± 0.502 | 6.463 ± 2.435 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 5.972 ± 1.269 | - | 6.340 ± 0.523 | 6.552 ± 2.553 | 27.859 ± 6.252 | 5.604 ± 0.828 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 1.294 ± 0.134 | - | 1.189 ± 0.359 | 6.070 ± 2.295 | 2.799 ± 0.904 | 1.280 ± 0.159 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 1.005 ± 0.100 | - | 1.053 ± 0.032 | 3.333 ± 0.806 | 2.101 ± 0.484 | 0.905 ± 0.152 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.407 ± 0.301 | - | 0.798 ± 0.112 | 0.892 ± 0.219 | 1.154 ± 0.193 | 0.871 ± 0.152 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 2.087 ± 0.481 | - | 1.968 ± 0.079 | 0.827 ± 0.118 | 1.011 ± 0.066 | 0.942 ± 0.121 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 1.166 ± 0.328 | 1.251 ± 0.298 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.856 ± 0.190 | 0.930 ± 0.283 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 31.123 ± 3.886 | - | 27.968 ± 1.832 | 32.566 ± 1.391 | 37.928 ± 1.733 | 65.964 ± 44.191 | 31.052 ± 2.934 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 18.510 ± 2.708 | - | 19.690 ± 4.642 | 21.386 ± 1.956 | 23.296 ± 4.912 | 68.625 ± 45.402 | 18.121 ± 9.651 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 26.375 ± 5.047 | - | 31.663 ± 5.258 | 19.622 ± 1.134 | 44.368 ± 2.920 | 42.691 ± 1.429 | 42.933 ± 10.717 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 17.918 ± 1.606 | - | 16.778 ± 3.237 | 12.756 ± 2.404 | 21.838 ± 3.901 | 45.427 ± 4.874 | 15.045 ± 0.721 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 34.658 ± 9.338 | - | 30.846 ± 8.706 | 24.077 ± 4.975 | 33.688 ± 4.632 | 33.898 ± 16.154 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 24.965 ± 3.489 | - | 26.630 ± 2.606 | 23.148 ± 3.403 | 27.055 ± 3.034 | 54.391 ± 74.813 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 15.740 ± 2.107 | - | 15.744 ± 7.486 | 13.839 ± 1.705 | 16.931 ± 2.675 | 16.367 ± 8.476 | 19.829 ± 5.128 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 11.339 ± 1.454 | - | 13.564 ± 0.937 | 14.830 ± 2.112 | 12.846 ± 2.980 | 32.916 ± 25.147 | 14.196 ± 1.141 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 25.056 ± 3.206 | - | 24.913 ± 0.338 | 29.382 ± 0.485 | 25.865 ± 0.636 | 31.770 ± 0.774 | 34.203 ± 4.666 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 14.133 ± 3.453 | - | 16.793 ± 2.389 | 16.696 ± 3.268 | 20.458 ± 2.495 | 33.051 ± 1.737 | 13.347 ± 3.153 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 19.151 ± 13.457 | - | 23.814 ± 5.436 | 14.363 ± 3.183 | 22.290 ± 2.059 | 27.886 ± 1.857 | 27.978 ± 5.879 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 14.112 ± 2.732 | - | 16.462 ± 2.788 | 15.114 ± 2.679 | 14.737 ± 2.550 | 30.013 ± 9.035 | 27.570 ± 14.213 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 22.446 ± 3.185 | - | 23.683 ± 15.018 | 20.397 ± 3.537 | 29.875 ± 4.910 | 27.699 ± 20.089 | 29.748 ± 7.823 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 22.991 ± 7.765 | - | 21.294 ± 4.754 | 22.794 ± 3.111 | 25.460 ± 6.968 | 42.162 ± 39.670 | 25.576 ± 6.726 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 67.802 ± 1.646 | - | 68.421 ± 1.986 | 67.065 ± 0.182 | 13.520 ± 0.291 | 47.665 ± 6.658 | 46.624 ± 4.467 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 33.094 ± 11.318 | - | 35.105 ± 11.238 | 33.398 ± 3.000 | 7.049 ± 7.605 | 47.577 ± 5.840 | 27.480 ± 11.353 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 30.415 ± 8.321 | 50.331 ± 15.558 | unsupported | 27.390 ± 6.022 | 31.277 ± 5.256 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 26.042 ± 2.616 | 33.251 ± 6.176 | unsupported | 21.860 ± 3.493 | 28.170 ± 3.375 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.950 ± 0.247 | - | 0.741 ± 0.095 | 0.876 ± 0.067 | 0.697 ± 0.067 | 0.792 ± 0.189 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.812 ± 0.175 | - | 0.648 ± 0.029 | 0.878 ± 0.157 | 0.613 ± 0.076 | 0.803 ± 0.070 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.868 ± 0.272 | - | 1.090 ± 0.318 | 0.707 ± 0.073 | 0.800 ± 0.221 | 0.970 ± 0.345 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.926 ± 0.176 | - | 0.884 ± 0.318 | 0.834 ± 0.124 | 0.880 ± 0.192 | 0.821 ± 0.049 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.501 ± 1.615 | - | unsupported | 1.395 ± 0.086 | 1.452 ± 0.334 | 1.754 ± 4.980 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 1.335 ± 0.174 | - | unsupported | 1.328 ± 0.302 | 0.917 ± 0.046 | 1.235 ± 3.015 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.492 ± 0.098 | - | 0.468 ± 0.112 | 0.332 ± 0.093 | 1.403 ± 1.563 | 1.373 ± 0.466 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.307 ± 0.083 | - | 0.543 ± 0.186 | 0.307 ± 0.030 | 0.236 ± 0.017 | 1.102 ± 0.103 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.798 ± 0.092 | - | 1.067 ± 0.806 | 1.013 ± 0.184 | 0.947 ± 0.115 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.718 ± 0.069 | - | 0.814 ± 0.086 | 1.279 ± 0.143 | 0.953 ± 0.035 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.754 ± 0.109 | - | 1.007 ± 0.091 | 1.066 ± 0.121 | 0.829 ± 0.098 | 1.643 ± 9.156 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.663 ± 0.237 | - | 0.854 ± 0.193 | 0.910 ± 0.259 | 0.781 ± 0.059 | 1.448 ± 4.254 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.761 ± 0.154 | - | 0.806 ± 0.094 | 0.415 ± 0.038 | 1.011 ± 0.368 | 1.402 ± 0.067 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.628 ± 0.086 | - | 0.679 ± 0.076 | 0.450 ± 0.021 | 0.555 ± 0.080 | 0.348 ± 0.026 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 2.114 ± 0.468 | - | 0.004 ± 0.001 | 1.176 ± 0.075 | 2.011 ± 0.425 | 2.129 ± 14.691 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 1.001 ± 0.207 | - | 0.001 ± 0.001 | 1.251 ± 0.050 | 1.283 ± 0.228 | 1.543 ± 0.075 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.132 ± 0.019 | - | 0.142 ± 0.031 | 0.141 ± 0.001 | 0.131 ± 0.070 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.100 ± 0.011 | - | 0.101 ± 0.001 | 0.196 ± 0.007 | 0.136 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.216 ± 0.001 | - | 0.219 ± 0.010 | 0.304 ± 0.005 | 0.231 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.049 ± 0.000 | - | 0.052 ± 0.009 | 0.039 ± 0.004 | 0.098 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.001 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.100 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.006 ± 0.003 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.001 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.003 ± 0.002 | - | 0.007 ± 0.001 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.001 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.008 ± 0.003 | 0.005 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.061 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.391 ± 0.020 | - | 0.401 ± 0.030 | 0.253 ± 0.012 | 0.252 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.037 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.018 ± 0.005 | - | 0.020 ± 0.000 | 0.018 ± 0.002 | 0.121 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.004 ± 0.000 | - | 0.008 ± 0.001 | 0.005 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.006 ± 0.000 | 0.056 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.048 ± 0.000 | - | 0.049 ± 0.000 | 0.206 ± 0.054 | 0.183 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.099 ± 0.001 | - | 0.101 ± 0.002 | 0.246 ± 0.079 | 0.136 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.216 ± 0.014 | - | 0.217 ± 0.001 | 0.331 ± 0.033 | 0.244 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.050 ± 0.003 | - | 0.051 ± 0.001 | 0.097 ± 0.029 | 0.062 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.125 ± 0.029 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.051 ± 0.016 | 0.062 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.066 ± 0.013 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.054 ± 0.030 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.001 | 0.138 ± 0.023 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.048 ± 0.014 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.002 ± 0.001 | - | 0.004 ± 0.000 | 0.057 ± 0.022 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.001 | 0.143 ± 0.039 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.045 ± 0.054 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.067 ± 0.020 | 0.061 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.225 ± 0.035 | - | 0.239 ± 0.016 | 0.227 ± 0.039 | 0.463 ± 0.093 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.141 ± 0.057 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.019 ± 0.000 | - | 0.020 ± 0.000 | 0.061 ± 0.004 | 0.039 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.059 ± 0.015 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.068 ± 0.026 | 0.038 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.277 ± 0.004 | - | 0.298 ± 0.022 | 0.436 ± 0.016 | 0.815 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.466 ± 0.203 | - | 1.438 ± 0.056 | 1.736 ± 0.146 | 2.276 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 4.734 ± 0.069 | - | 4.764 ± 0.161 | 6.728 ± 1.355 | 7.367 ± 0.338 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 2.419 ± 0.084 | - | 2.497 ± 0.099 | 3.324 ± 0.042 | 4.225 ± 3.080 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.002 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.076 ± 0.000 | - | 0.078 ± 0.008 | 0.096 ± 0.022 | 0.163 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.006 ± 0.000 | - | 0.007 ± 0.002 | 0.011 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.002 ± 0.001 | - | 0.003 ± 0.000 | 0.006 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.007 ± 0.001 | - | 0.008 ± 0.001 | 0.010 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.015 ± 0.003 | - | 0.016 ± 0.000 | 0.024 ± 0.007 | 0.061 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.007 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.007 ± 0.000 | - | 0.010 ± 0.002 | 0.011 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.019 ± 0.000 | - | 0.021 ± 0.004 | 0.029 ± 0.003 | 0.070 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 19.393 ± 0.406 | - | 19.776 ± 3.141 | 26.677 ± 0.663 | 34.813 ± 1.506 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.004 ± 0.001 | - | 0.005 ± 0.001 | 0.008 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.610 ± 0.008 | - | 0.606 ± 0.017 | 0.713 ± 0.017 | 0.847 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.012 ± 0.003 | - | 0.013 ± 0.003 | 0.018 ± 0.004 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.038 ± 0.000 | - | 0.039 ± 0.000 | 0.054 ± 0.001 | 0.099 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.273 ± 0.013 | - | 0.281 ± 0.022 | 0.425 ± 0.003 | 0.821 ± 0.086 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 0.759 ± 0.041 | - | 0.828 ± 0.079 | 1.764 ± 0.248 | 1.790 ± 0.594 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 2.541 ± 0.297 | - | 2.728 ± 0.227 | 6.615 ± 0.159 | 3.665 ± 0.105 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 2.288 ± 0.028 | - | 2.843 ± 0.600 | 3.455 ± 0.117 | 3.755 ± 0.255 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.212 ± 0.051 | - | 0.089 ± 0.022 | 0.092 ± 0.001 | 0.173 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.007 ± 0.001 | - | 0.011 ± 0.001 | 0.011 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.002 | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.002 | 0.010 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.019 ± 0.002 | - | 0.021 ± 0.003 | 0.024 ± 0.004 | 0.049 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.002 | 0.008 ± 0.001 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.008 ± 0.000 | - | 0.009 ± 0.005 | 0.011 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.020 ± 0.001 | - | 0.022 ± 0.000 | 0.029 ± 0.000 | 0.082 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 8.974 ± 1.309 | - | 8.766 ± 1.266 | 34.690 ± 9.862 | 16.370 ± 0.740 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.342 ± 0.075 | - | 0.346 ± 0.010 | 0.764 ± 0.123 | 0.522 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.013 ± 0.003 | - | 0.014 ± 0.000 | 0.017 ± 0.000 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.039 ± 0.001 | - | 0.093 ± 0.001 | 0.054 ± 0.000 | 0.143 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.266 ± 0.003 | - | 0.268 ± 0.010 | 0.242 ± 0.003 | 0.543 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.226 ± 0.032 | - | 1.210 ± 0.021 | 1.334 ± 0.081 | 1.541 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 4.165 ± 0.047 | - | 4.271 ± 0.615 | 4.962 ± 0.070 | 5.626 ± 0.286 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.393 ± 0.050 | - | 1.408 ± 0.090 | 1.002 ± 0.052 | 1.134 ± 0.087 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.014 ± 0.001 | 0.033 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.061 ± 0.001 | - | 0.062 ± 0.000 | 0.071 ± 0.001 | 0.126 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.006 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.001 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.015 ± 0.001 | 0.012 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.003 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.013 ± 0.000 | - | 0.014 ± 0.000 | 0.019 ± 0.001 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.003 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.002 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.017 ± 0.000 | - | 0.018 ± 0.000 | 0.024 ± 0.000 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 11.359 ± 1.596 | - | 11.601 ± 0.865 | 7.953 ± 0.245 | 8.725 ± 0.166 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.003 ± 0.001 | - | 0.006 ± 0.001 | 0.019 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.492 ± 0.020 | - | 0.485 ± 0.010 | 0.532 ± 0.018 | 0.599 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.011 ± 0.000 | - | 0.012 ± 0.000 | 0.014 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.033 ± 0.000 | - | 0.034 ± 0.000 | 0.043 ± 0.001 | 0.087 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.274 ± 0.026 | - | 0.270 ± 0.021 | 0.261 ± 0.014 | 0.558 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.254 ± 0.086 | - | 1.251 ± 0.030 | 1.392 ± 0.083 | 1.059 ± 0.101 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 2.119 ± 0.202 | - | 2.098 ± 0.035 | 4.864 ± 0.256 | 3.460 ± 0.790 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.454 ± 0.241 | - | 1.414 ± 0.049 | 0.936 ± 0.010 | 1.105 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.006 ± 0.005 | 0.010 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.062 ± 0.001 | - | 0.063 ± 0.007 | 0.066 ± 0.001 | 0.130 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.004 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.003 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.006 ± 0.001 | - | 0.007 ± 0.001 | 0.008 ± 0.002 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.014 ± 0.004 | - | 0.015 ± 0.000 | 0.019 ± 0.001 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.008 ± 0.001 | - | 0.008 ± 0.002 | 0.011 ± 0.002 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.018 ± 0.002 | - | 0.019 ± 0.000 | 0.026 ± 0.006 | 0.037 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 5.751 ± 0.909 | - | 5.725 ± 0.286 | 8.034 ± 0.184 | 4.393 ± 0.109 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.004 ± 0.001 | - | 0.006 ± 0.001 | 0.006 ± 0.002 | 0.014 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.279 ± 0.018 | - | 0.306 ± 0.095 | 0.501 ± 0.017 | 0.383 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.013 ± 0.003 | - | 0.012 ± 0.003 | 0.015 ± 0.003 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.034 ± 0.000 | - | 0.035 ± 0.001 | 0.047 ± 0.010 | 0.081 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.203 ± 0.013 | - | 0.192 ± 0.003 | 0.132 ± 0.273 | 1.403 ± 0.617 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 1.136 ± 0.066 | - | 1.109 ± 0.040 | 0.771 ± 0.029 | 1.979 ± 2.113 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 3.769 ± 0.199 | - | 3.880 ± 0.100 | 2.457 ± 0.101 | 7.204 ± 5.462 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.813 ± 0.169 | - | 0.937 ± 0.204 | 0.464 ± 0.200 | 1.487 ± 0.435 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.027 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.046 ± 0.005 | - | 0.047 ± 0.001 | 0.030 ± 0.000 | 0.107 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.019 ± 0.045 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.005 | 0.016 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.059 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.019 ± 0.007 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.033 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.016 ± 0.015 | - | 0.016 ± 0.000 | 0.015 ± 0.005 | 0.086 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 6.116 ± 0.141 | - | 5.979 ± 0.073 | 3.747 ± 0.045 | 12.884 ± 5.046 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.392 ± 0.143 | - | 0.383 ± 0.015 | 0.195 ± 0.004 | 0.759 ± 0.765 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.011 ± 0.002 | - | 0.012 ± 0.003 | 0.011 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.029 ± 0.000 | - | 0.031 ± 0.000 | 0.028 ± 0.006 | 0.196 ± 0.088 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.197 ± 0.002 | - | 0.189 ± 0.006 | 0.126 ± 0.001 | 0.563 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 1.078 ± 0.023 | - | 1.118 ± 0.023 | 0.741 ± 0.015 | 1.387 ± 0.398 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 1.914 ± 0.090 | - | 1.915 ± 0.075 | 2.525 ± 0.098 | 3.346 ± 0.507 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.718 ± 0.033 | - | 0.710 ± 0.013 | 0.541 ± 0.104 | 1.439 ± 0.460 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.046 ± 0.000 | - | 0.061 ± 0.002 | 0.040 ± 0.010 | 0.150 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.006 ± 0.000 | - | 0.007 ± 0.002 | 0.008 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.001 | 0.008 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.012 ± 0.003 | - | 0.014 ± 0.000 | 0.012 ± 0.000 | 0.024 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.006 ± 0.001 | - | 0.007 ± 0.002 | 0.008 ± 0.002 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.016 ± 0.004 | - | 0.019 ± 0.005 | 0.015 ± 0.000 | 0.030 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 2.996 ± 0.098 | - | 3.135 ± 0.613 | 4.722 ± 1.702 | 4.819 ± 0.384 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.212 ± 0.001 | - | 0.214 ± 0.001 | 0.227 ± 0.015 | 0.721 ± 0.158 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.012 ± 0.002 | - | 0.011 ± 0.001 | 0.012 ± 0.002 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.029 ± 0.000 | - | 0.031 ± 0.001 | 0.024 ± 0.000 | 0.121 ± 0.150 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.208 ± 0.137 | - | 0.072 ± 0.005 | 0.184 ± 0.016 | 0.145 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.160 ± 0.003 | - | 0.176 ± 0.011 | 0.278 ± 0.020 | 0.229 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.499 ± 0.076 | - | 0.467 ± 0.073 | 0.548 ± 0.042 | 0.447 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.132 ± 0.002 | - | 0.133 ± 0.003 | 0.069 ± 0.002 | 0.132 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.029 ± 0.020 | - | 0.019 ± 0.007 | 0.011 ± 0.001 | 0.067 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.011 ± 0.003 | - | 0.013 ± 0.003 | 0.016 ± 0.004 | 0.051 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.009 ± 0.001 | 0.008 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.068 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.002 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.072 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 1.074 ± 0.364 | - | 1.162 ± 1.521 | 0.476 ± 0.016 | 0.697 ± 0.121 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.009 ± 0.002 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.071 ± 0.010 | - | 0.055 ± 0.013 | 0.039 ± 0.005 | 0.090 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.002 | 0.067 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.122 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.072 ± 0.001 | - | 0.070 ± 0.000 | 0.318 ± 0.040 | 0.312 ± 0.334 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.162 ± 0.007 | - | 0.159 ± 0.003 | 0.410 ± 0.031 | 0.214 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.451 ± 0.009 | - | 0.450 ± 0.011 | 0.690 ± 0.053 | 0.600 ± 0.176 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.171 ± 0.006 | - | 0.171 ± 0.003 | 0.154 ± 0.013 | 0.117 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.090 ± 0.007 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.009 ± 0.003 | - | 0.010 ± 0.000 | 0.094 ± 0.008 | 0.075 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.085 ± 0.015 | 0.010 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.012 ± 0.001 | - | 0.013 ± 0.000 | 0.088 ± 0.013 | 0.081 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.004 ± 0.001 | - | 0.004 ± 0.000 | 0.114 ± 0.020 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.090 ± 0.010 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.004 ± 0.000 | 0.083 ± 0.007 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.116 ± 0.003 | 0.086 ± 0.078 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.001 | 0.091 ± 0.031 | 0.014 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.085 ± 0.005 | 0.035 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.001 | 0.076 ± 0.049 | 0.071 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.552 ± 0.014 | - | 0.549 ± 0.011 | 0.803 ± 0.030 | 0.470 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.139 ± 0.044 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.059 ± 0.010 | - | 0.056 ± 0.001 | 0.187 ± 0.030 | 0.174 ± 0.053 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.005 ± 0.001 | - | 0.006 ± 0.000 | 0.123 ± 0.068 | 0.150 ± 0.096 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.007 ± 0.001 | - | 0.007 ± 0.000 | 0.117 ± 0.023 | 0.045 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.163 ± 0.000 | - | 0.164 ± 0.009 | 0.298 ± 0.350 | 0.330 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.372 ± 0.010 | - | 0.374 ± 0.013 | 0.654 ± 0.018 | 0.656 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.051 ± 0.054 | - | 1.068 ± 0.036 | 1.643 ± 0.148 | 1.698 ± 0.801 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.420 ± 0.016 | - | 0.398 ± 0.019 | 0.358 ± 0.011 | 0.430 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.020 ± 0.000 | - | 0.021 ± 0.000 | 0.019 ± 0.000 | 0.057 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.022 ± 0.000 | - | 0.023 ± 0.000 | 0.018 ± 0.000 | 0.066 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.002 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.002 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 3.309 ± 0.120 | - | 3.202 ± 0.073 | 2.888 ± 0.119 | 3.049 ± 0.294 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.148 ± 0.001 | - | 0.154 ± 0.004 | 0.153 ± 0.006 | 0.147 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.009 ± 0.001 | - | 0.010 ± 0.000 | 0.015 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.171 ± 0.007 | - | 0.166 ± 0.008 | 0.482 ± 0.352 | 0.346 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.411 ± 0.022 | - | 0.409 ± 0.049 | 0.585 ± 0.040 | 0.681 ± 0.108 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 0.798 ± 0.235 | - | 0.825 ± 0.042 | 1.360 ± 0.044 | 0.940 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.408 ± 0.026 | - | 0.403 ± 0.026 | 0.465 ± 0.198 | 0.391 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.036 ± 0.004 | 0.011 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.021 ± 0.000 | - | 0.022 ± 0.000 | 0.048 ± 0.019 | 0.074 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.003 ± 0.001 | - | 0.012 ± 0.000 | 0.027 ± 0.009 | 0.010 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.023 ± 0.001 | - | 0.025 ± 0.000 | 0.040 ± 0.019 | 0.078 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.003 ± 0.001 | - | 0.014 ± 0.001 | 0.040 ± 0.039 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.025 ± 0.010 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.070 ± 0.015 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.006 ± 0.000 | - | 0.019 ± 0.000 | 0.037 ± 0.021 | 0.045 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.022 ± 0.011 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.014 ± 0.000 | 0.034 ± 0.010 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.007 ± 0.001 | - | 0.007 ± 0.001 | 0.052 ± 0.027 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 2.337 ± 0.591 | - | 1.755 ± 0.145 | 2.741 ± 0.249 | 1.572 ± 0.114 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.027 ± 0.013 | 0.012 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.149 ± 0.001 | - | 0.150 ± 0.001 | 0.185 ± 0.312 | 0.163 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.005 ± 0.001 | - | 0.008 ± 0.002 | 0.074 ± 0.050 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.012 ± 0.002 | - | 0.011 ± 0.000 | 0.035 ± 0.015 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.095 ± 0.009 | - | 0.097 ± 0.002 | 0.129 ± 0.003 | 0.185 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.148 ± 0.005 | - | 0.163 ± 0.026 | 0.189 ± 0.002 | 0.269 ± 0.087 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.361 ± 0.002 | - | 0.398 ± 0.070 | 0.407 ± 0.021 | 0.480 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.085 ± 0.002 | - | 0.089 ± 0.000 | 0.171 ± 0.002 | 0.125 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.016 ± 0.013 | 0.011 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.038 ± 0.001 | 0.076 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.001 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.031 ± 0.027 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.011 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.003 ± 0.001 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.678 ± 0.033 | - | 0.730 ± 0.043 | 0.406 ± 0.005 | 0.832 ± 0.947 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.003 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.034 ± 0.000 | - | 0.042 ± 0.009 | 0.103 ± 0.001 | 0.091 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.001 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.007 ± 0.002 | 0.013 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.094 ± 0.008 | - | 0.101 ± 0.002 | 0.202 ± 0.019 | 0.199 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.147 ± 0.001 | - | 0.161 ± 0.003 | 0.206 ± 0.029 | 0.265 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.225 ± 0.037 | - | 0.284 ± 0.154 | 0.428 ± 0.045 | 0.448 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.185 ± 0.084 | - | 0.149 ± 0.031 | 0.068 ± 0.002 | 0.216 ± 0.238 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.002 ± 0.001 | - | 0.020 ± 0.015 | 0.010 ± 0.001 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.007 ± 0.000 | - | 0.011 ± 0.001 | 0.014 ± 0.001 | 0.063 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.002 ± 0.001 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.007 ± 0.001 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.020 ± 0.000 | 0.011 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.014 ± 0.003 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.008 ± 0.023 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.001 | 0.012 ± 0.000 | 0.014 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.653 ± 0.063 | - | 0.461 ± 0.151 | 0.484 ± 0.045 | 0.713 ± 0.103 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.035 ± 0.002 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.036 ± 0.000 | - | 0.042 ± 0.001 | 0.092 ± 0.002 | 0.297 ± 0.199 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.001 | 0.036 ± 0.003 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.005 ± 0.001 | - | 0.009 ± 0.000 | 0.047 ± 0.030 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 0.773 ± 0.020 | - | 0.690 ± 0.014 | 1.210 ± 0.057 | 3.042 ± 0.190 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 2.819 ± 0.064 | - | 2.680 ± 0.117 | 4.385 ± 0.205 | 6.393 ± 2.115 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 9.446 ± 0.945 | - | 9.180 ± 1.299 | 15.034 ± 0.711 | 17.724 ± 0.355 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 5.682 ± 5.312 | - | 5.075 ± 2.514 | 4.968 ± 0.249 | 5.515 ± 0.082 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.008 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.144 ± 0.006 | - | 0.144 ± 0.003 | 0.187 ± 0.008 | 0.264 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.028 | 0.011 ± 0.000 | 0.021 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.130 ± 0.007 | - | 0.138 ± 0.012 | 0.175 ± 0.008 | 0.247 ± 0.065 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.010 ± 0.000 | - | 0.011 ± 0.000 | 0.022 ± 0.001 | 0.031 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.012 ± 0.003 | - | 0.012 ± 0.002 | 0.019 ± 0.000 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.027 ± 0.000 | - | 0.027 ± 0.000 | 0.051 ± 0.001 | 0.063 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.015 ± 0.003 | - | 0.016 ± 0.003 | 0.024 ± 0.005 | 0.037 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.036 ± 0.000 | - | 0.037 ± 0.001 | 0.067 ± 0.009 | 0.080 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 56.491 ± 9.761 | - | 47.142 ± 14.655 | 39.382 ± 0.619 | 43.519 ± 4.187 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.036 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.773 ± 1.850 | - | 1.186 ± 0.838 | 1.478 ± 0.111 | 1.708 ± 0.075 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.023 ± 0.000 | - | 0.023 ± 0.000 | 0.040 ± 0.003 | 0.059 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.073 ± 0.001 | - | 0.071 ± 0.014 | 0.136 ± 0.019 | 0.230 ± 0.065 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 0.823 ± 0.023 | - | 0.710 ± 0.047 | 1.279 ± 0.440 | 2.966 ± 0.157 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 1.446 ± 0.108 | - | 1.419 ± 0.140 | 4.253 ± 0.047 | 4.118 ± 1.195 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 4.833 ± 0.463 | - | 4.637 ± 0.279 | 14.987 ± 2.210 | 8.635 ± 0.319 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 5.199 ± 0.371 | - | 5.020 ± 0.198 | 4.934 ± 0.168 | 5.231 ± 0.398 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.009 ± 0.002 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.172 ± 0.040 | - | 0.156 ± 0.011 | 0.211 ± 0.037 | 0.260 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.020 ± 0.001 | 0.012 ± 0.002 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.123 ± 0.001 | - | 0.124 ± 0.001 | 0.174 ± 0.014 | 0.254 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.014 ± 0.003 | - | 0.012 ± 0.000 | 0.022 ± 0.001 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.002 | 0.013 ± 0.003 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.011 ± 0.002 | - | 0.011 ± 0.000 | 0.020 ± 0.003 | 0.032 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.029 ± 0.006 | - | 0.048 ± 0.040 | 0.050 ± 0.004 | 0.071 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.006 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.013 ± 0.003 | - | 0.015 ± 0.004 | 0.027 ± 0.006 | 0.036 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.038 ± 0.003 | - | 0.041 ± 0.009 | 0.067 ± 0.005 | 0.081 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 18.197 ± 1.715 | - | 17.038 ± 3.541 | 41.452 ± 3.513 | 21.793 ± 0.817 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.010 ± 0.001 | - | 0.010 ± 0.002 | 0.016 ± 0.001 | 0.038 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 0.822 ± 0.177 | - | 0.601 ± 0.009 | 1.562 ± 0.194 | 0.945 ± 0.061 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.026 ± 0.006 | - | 0.059 ± 0.001 | 0.045 ± 0.004 | 0.059 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.072 ± 0.007 | - | 0.072 ± 0.001 | 0.122 ± 0.002 | 0.196 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.510 ± 0.042 | - | 0.516 ± 0.024 | 0.594 ± 0.056 | 2.048 ± 0.078 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 2.150 ± 0.830 | - | 2.183 ± 0.567 | 2.923 ± 0.353 | 4.319 ± 0.199 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 8.852 ± 2.470 | - | 7.653 ± 1.595 | 10.701 ± 0.199 | 13.865 ± 2.748 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 2.241 ± 4.198 | - | 2.090 ± 1.562 | 1.510 ± 0.043 | 4.172 ± 0.185 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.006 ± 0.000 | - | 0.010 ± 0.000 | 0.007 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.106 ± 0.008 | - | 0.106 ± 0.001 | 0.125 ± 0.001 | 0.414 ± 0.060 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.002 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.076 ± 0.001 | - | 0.076 ± 0.001 | 0.357 ± 0.024 | 0.139 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.010 ± 0.002 | - | 0.009 ± 0.000 | 0.056 ± 0.013 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.009 ± 0.000 | - | 0.013 ± 0.000 | 0.008 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.002 | 0.014 ± 0.003 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.021 ± 0.000 | - | 0.023 ± 0.000 | 0.055 ± 0.073 | 0.043 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.010 ± 0.000 | - | 0.014 ± 0.000 | 0.008 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.024 ± 0.000 | - | 0.028 ± 0.000 | 0.017 ± 0.000 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.028 ± 0.005 | - | 0.029 ± 0.000 | 0.045 ± 0.003 | 0.057 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 18.697 ± 2.064 | - | 19.201 ± 6.812 | 18.871 ± 8.917 | 12.658 ± 0.297 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.015 ± 0.000 | - | 0.019 ± 0.000 | 0.011 ± 0.000 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 2.037 ± 1.551 | - | 0.837 ± 0.019 | 0.983 ± 0.029 | 1.140 ± 1.400 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.041 ± 0.000 | - | 0.046 ± 0.000 | 0.027 ± 0.000 | 0.044 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.057 ± 0.000 | - | 0.057 ± 0.000 | 0.083 ± 0.001 | 0.149 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.508 ± 0.019 | - | 0.533 ± 0.009 | 0.572 ± 0.025 | 2.615 ± 2.406 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 2.221 ± 0.043 | - | 2.226 ± 0.073 | 3.080 ± 0.932 | 2.861 ± 0.575 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 3.967 ± 0.239 | - | 4.179 ± 0.546 | 10.722 ± 0.441 | 8.130 ± 2.061 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 2.227 ± 0.093 | - | 2.256 ± 0.068 | 1.653 ± 0.292 | 1.592 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.115 ± 0.018 | - | 0.154 ± 0.169 | 0.121 ± 0.002 | 0.233 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.001 | 0.009 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.202 ± 0.100 | - | 0.088 ± 0.007 | 0.104 ± 0.006 | 0.171 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.009 ± 0.000 | - | 0.013 ± 0.003 | 0.017 ± 0.005 | 0.072 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.004 ± 0.001 | - | 0.005 ± 0.000 | 0.008 ± 0.002 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.011 ± 0.002 | - | 0.012 ± 0.002 | 0.014 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.022 ± 0.001 | - | 0.024 ± 0.000 | 0.036 ± 0.007 | 0.045 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.001 | 0.009 ± 0.000 | 0.061 ± 0.060 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.011 ± 0.001 | - | 0.014 ± 0.004 | 0.017 ± 0.004 | 0.030 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.029 ± 0.005 | - | 0.031 ± 0.005 | 0.044 ± 0.006 | 0.058 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 9.082 ± 1.082 | - | 9.268 ± 1.181 | 13.133 ± 0.464 | 7.601 ± 0.765 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.007 ± 0.001 | - | 0.008 ± 0.000 | 0.011 ± 0.002 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 0.578 ± 0.023 | - | 0.540 ± 0.092 | 1.214 ± 0.279 | 1.158 ± 0.935 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.021 ± 0.003 | - | 0.023 ± 0.004 | 0.027 ± 0.000 | 0.047 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.058 ± 0.010 | - | 0.059 ± 0.001 | 0.090 ± 0.012 | 0.244 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.320 ± 0.004 | - | unsupported | 0.231 ± 0.043 | 1.369 ± 0.068 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.520 ± 0.411 | - | unsupported | 1.542 ± 0.329 | 2.964 ± 1.174 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 4.640 ± 3.271 | - | unsupported | 4.012 ± 0.065 | 5.394 ± 0.376 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 1.253 ± 0.423 | - | unsupported | 0.795 ± 0.020 | 0.790 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | unsupported | 0.007 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.047 ± 0.000 | - | unsupported | 0.041 ± 0.001 | 0.047 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.038 ± 0.002 | - | unsupported | 0.035 ± 0.003 | 0.042 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | unsupported | 0.009 ± 0.003 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.007 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | unsupported | 0.011 ± 0.002 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.013 ± 0.001 | - | unsupported | 0.018 ± 0.005 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.007 ± 0.000 | - | unsupported | 0.012 ± 0.002 | 0.053 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.017 ± 0.000 | - | unsupported | 0.023 ± 0.004 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 10.411 ± 1.715 | - | unsupported | 6.428 ± 0.292 | 6.678 ± 0.125 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.366 ± 0.022 | - | unsupported | 0.305 ± 0.020 | 0.315 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.012 ± 0.000 | - | unsupported | 0.019 ± 0.003 | 0.034 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.034 ± 0.000 | - | unsupported | 0.036 ± 0.007 | 0.048 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.306 ± 0.001 | - | unsupported | 0.231 ± 0.010 | 1.525 ± 1.136 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 1.435 ± 0.027 | - | unsupported | 1.610 ± 0.349 | 1.450 ± 0.195 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 2.167 ± 0.047 | - | unsupported | 4.645 ± 0.048 | 3.535 ± 0.953 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.120 ± 0.046 | - | unsupported | 1.206 ± 0.030 | 1.463 ± 2.046 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.007 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.047 ± 0.002 | - | unsupported | 0.048 ± 0.007 | 0.049 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.019 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.039 ± 0.004 | - | unsupported | 0.038 ± 0.006 | 0.048 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.003 ± 0.001 | - | unsupported | 0.006 ± 0.001 | 0.014 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.013 ± 0.000 | - | unsupported | 0.018 ± 0.005 | 0.037 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.007 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.019 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.017 ± 0.000 | - | unsupported | 0.022 ± 0.006 | 0.038 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 4.873 ± 0.311 | - | unsupported | 8.385 ± 2.572 | 4.623 ± 0.734 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | unsupported | 0.009 ± 0.002 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.212 ± 0.013 | - | unsupported | 0.350 ± 0.006 | 0.352 ± 0.110 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.012 ± 0.000 | - | unsupported | 0.016 ± 0.001 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.033 ± 0.001 | - | unsupported | 0.038 ± 0.022 | 0.064 ± 0.007 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.783 ± 0.025 | - | unsupported | 0.812 ± 0.063 | 1.173 ± 0.027 | 6.036 ± 3.700 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.052 ± 0.009 | - | unsupported | 0.084 ± 0.022 | 0.142 ± 0.007 | 0.069 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.101 ± 0.010 | - | unsupported | 0.130 ± 0.020 | 0.199 ± 0.017 | 0.222 ± 0.031 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.228 ± 0.002 | - | unsupported | 0.268 ± 0.001 | 0.345 ± 0.004 | 1.416 ± 0.244 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.610 ± 0.156 | - | unsupported | 0.485 ± 0.202 | 0.796 ± 0.134 | 2.690 ± 0.786 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.054 ± 0.005 | - | unsupported | 0.151 ± 0.011 | 0.123 ± 0.014 | 0.066 ± 0.012 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.112 ± 0.007 | - | unsupported | 0.196 ± 0.015 | 0.182 ± 0.023 | 0.144 ± 0.020 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.223 ± 0.039 | - | unsupported | 0.321 ± 0.034 | 0.394 ± 0.143 | 0.759 ± 0.247 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.440 ± 0.022 | - | unsupported | 0.197 ± 0.014 | 0.747 ± 0.184 | 0.787 ± 0.648 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.031 ± 0.000 | - | unsupported | 0.043 ± 0.001 | 0.094 ± 0.014 | 0.073 ± 0.026 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.041 ± 0.000 | - | unsupported | 0.054 ± 0.002 | 0.140 ± 0.029 | 0.090 ± 0.023 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.132 ± 0.006 | - | unsupported | 0.082 ± 0.004 | 0.201 ± 0.106 | 0.274 ± 0.233 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.305 ± 0.009 | - | unsupported | 0.190 ± 0.050 | 0.544 ± 0.076 | 0.778 ± 0.541 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.032 ± 0.004 | - | unsupported | 0.041 ± 0.001 | 0.107 ± 0.020 | 0.057 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.042 ± 0.000 | - | unsupported | 0.048 ± 0.000 | 0.127 ± 0.019 | 0.082 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.134 ± 0.001 | - | unsupported | 0.078 ± 0.000 | 0.176 ± 0.026 | 0.223 ± 0.030 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.196 ± 0.005 | - | 0.200 ± 0.002 | 0.096 ± 0.048 | 0.195 ± 0.025 | 0.737 ± 0.354 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.019 ± 0.000 | - | 0.022 ± 0.021 | 0.047 ± 0.029 | 0.026 ± 0.002 | 0.034 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.024 ± 0.000 | - | 0.027 ± 0.000 | 0.037 ± 0.009 | 0.044 ± 0.022 | 0.046 ± 0.003 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.065 ± 0.005 | - | 0.065 ± 0.004 | 0.043 ± 0.033 | 0.059 ± 0.002 | 0.130 ± 0.005 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.198 ± 0.012 | - | 0.210 ± 0.019 | 0.085 ± 0.018 | 0.220 ± 0.029 | 0.245 ± 0.006 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.020 ± 0.000 | - | 0.023 ± 0.010 | 0.031 ± 0.015 | 0.025 ± 0.001 | 0.037 ± 0.005 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.025 ± 0.000 | - | 0.027 ± 0.000 | 0.032 ± 0.016 | 0.030 ± 0.001 | 0.046 ± 0.004 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.063 ± 0.000 | - | 0.065 ± 0.001 | 0.037 ± 0.007 | 0.073 ± 0.007 | 0.117 ± 0.020 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 39.989 ± 0.508 | - | 40.039 ± 1.834 | 26.107 ± 1.308 | 19.168 ± 3.497 | 38.053 ± 4.906 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 25.654 ± 3.909 | - | 27.193 ± 2.001 | 20.452 ± 2.472 | 20.275 ± 2.226 | 390.580 ± 51.717 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 26.386 ± 1.677 | - | 23.305 ± 1.201 | 7.566 ± 1.026 | 9.427 ± 2.298 | 24.027 ± 5.997 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 19.440 ± 1.402 | - | 17.334 ± 1.388 | 10.541 ± 6.469 | 7.567 ± 1.056 | 236.411 ± 32.613 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 6.723 ± 0.101 | - | 6.986 ± 0.444 | 6.964 ± 0.386 | 9.452 ± 2.523 | 7.004 ± 0.160 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 7.190 ± 0.680 | - | 7.427 ± 1.627 | 7.892 ± 0.261 | 8.654 ± 0.073 | 512.933 ± 462.339 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 7.826 ± 0.295 | - | 7.934 ± 0.204 | 7.677 ± 0.190 | 7.910 ± 0.608 | 7.408 ± 0.113 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 9.521 ± 0.391 | - | 10.170 ± 2.634 | 8.761 ± 0.609 | 8.642 ± 1.142 | 270.374 ± 22.932 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 19.662 ± 0.827 | - | 19.915 ± 1.161 | 21.910 ± 1.494 | 28.539 ± 1.289 | 13.445 ± 1.195 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 19.598 ± 0.779 | - | 19.032 ± 0.763 | 23.564 ± 1.303 | 26.166 ± 1.508 | 702.814 ± 349.806 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.418 ± 0.176 | 0.354 ± 0.018 | unsupported | 0.321 ± 0.008 | 0.393 ± 0.061 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 6.529 ± 0.163 | 6.552 ± 0.375 | unsupported | 3.989 ± 0.105 | 4.709 ± 0.223 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.417 ± 0.280 | 0.421 ± 0.297 | unsupported | 0.318 ± 0.032 | 0.593 ± 0.137 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 5.294 ± 0.578 | 5.664 ± 0.844 | unsupported | 4.372 ± 0.760 | 6.308 ± 0.937 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 36.463 ± 2.502 | - | 37.282 ± 0.582 | 11.446 ± 2.086 | 9.926 ± 1.269 | 29.432 ± 3.297 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 24.525 ± 5.025 | - | 24.657 ± 3.765 | 10.051 ± 0.851 | 9.563 ± 1.476 | 381.907 ± 461.623 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 26.652 ± 9.130 | 30.601 ± 10.691 | 5.738 ± 0.425 | 32.357 ± 0.863 | 13.374 ± 6.877 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 11.733 ± 1.129 | 12.019 ± 1.413 | 5.262 ± 0.334 | 32.978 ± 2.042 | 157.368 ± 12.131 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 36.251 ± 3.093 | - | 40.211 ± 21.056 | 17.478 ± 3.352 | 16.234 ± 10.920 | 29.399 ± 17.400 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 18.266 ± 1.794 | - | 18.279 ± 1.217 | 11.213 ± 2.527 | 11.653 ± 4.087 | 349.025 ± 239.012 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 1.214 ± 1.923 | - | 0.890 ± 0.228 | 2.413 ± 0.749 | 6.285 ± 0.345 | 5.797 ± 0.160 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.901 ± 0.279 | - | 0.800 ± 0.061 | 2.305 ± 0.094 | 3.293 ± 0.076 | 7.681 ± 0.733 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 18.251 ± 1.098 | - | 18.018 ± 1.892 | 12.718 ± 2.067 | 13.578 ± 1.260 | 24.955 ± 4.678 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 16.429 ± 1.018 | - | 16.834 ± 1.314 | 13.945 ± 4.180 | 12.380 ± 1.175 | 988.778 ± 70.304 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 20.348 ± 4.660 | - | 21.877 ± 8.034 | 12.651 ± 6.901 | 12.940 ± 1.092 | 24.165 ± 5.491 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 16.639 ± 0.753 | - | 16.599 ± 0.984 | 11.612 ± 1.865 | 13.643 ± 1.760 | 946.520 ± 45.238 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.121 ± 0.024 | 1.146 ± 0.028 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 27.712 ± 0.869 | 29.580 ± 8.524 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.070 ± 0.044 | 1.126 ± 0.190 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 19.339 ± 4.609 | 20.214 ± 3.378 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.132 ± 0.029 | 1.213 ± 0.446 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 28.829 ± 4.148 | 29.061 ± 5.181 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.055 ± 0.022 | 1.089 ± 0.038 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 26.050 ± 12.207 | 20.820 ± 2.950 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 26.069 ± 0.353 | - | 22.904 ± 1.676 | 8.439 ± 1.107 | 10.284 ± 1.888 | 20.565 ± 3.587 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 23.064 ± 1.539 | - | 19.746 ± 4.330 | 8.359 ± 0.413 | 7.696 ± 1.108 | 239.325 ± 41.139 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 63.772 ± 2.699 | 62.064 ± 1.879 | 41.630 ± 5.545 | 36.546 ± 0.716 | 68.277 ± 1.580 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 58.113 ± 5.147 | 60.789 ± 2.806 | 40.917 ± 4.882 | 38.810 ± 10.117 | 1691.971 ± 102.663 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 40.226 ± 1.385 | - | 39.708 ± 0.496 | 13.907 ± 1.047 | 73.125 ± 4.182 | 33.775 ± 4.082 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 26.844 ± 4.938 | - | 29.617 ± 4.080 | 15.576 ± 3.049 | 46.362 ± 2.146 | 31.432 ± 4.901 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.212 ± 0.054 | 0.214 ± 0.079 | unsupported | 0.135 ± 0.087 | 0.037 ± 0.011 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.842 ± 0.388 | 2.854 ± 0.066 | unsupported | 0.867 ± 0.020 | 0.502 ± 0.062 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.177 ± 0.005 | 0.171 ± 0.011 | unsupported | 0.114 ± 0.069 | 0.028 ± 0.011 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 1.662 ± 0.803 | 1.545 ± 0.205 | unsupported | 0.762 ± 0.237 | 0.566 ± 0.569 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.081 ± 0.004 | 0.080 ± 0.001 | unsupported | 0.096 ± 0.067 | 0.188 ± 0.004 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.279 ± 0.032 | 1.301 ± 0.079 | unsupported | 1.411 ± 0.050 | 4.148 ± 1.152 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.062 ± 0.031 | 0.054 ± 0.009 | unsupported | 0.089 ± 0.050 | 0.193 ± 0.081 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.884 ± 0.223 | 0.698 ± 0.075 | unsupported | 1.090 ± 0.097 | 1.663 ± 0.820 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.273 ± 0.022 | 0.271 ± 0.005 | unsupported | 0.417 ± 0.219 | 0.088 ± 0.007 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 4.444 ± 0.185 | 4.518 ± 0.303 | unsupported | 4.794 ± 0.232 | 1.704 ± 0.791 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.171 ± 0.070 | 0.175 ± 0.033 | unsupported | 0.309 ± 0.197 | 0.069 ± 0.009 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 2.863 ± 0.619 | 3.310 ± 0.734 | unsupported | 3.150 ± 0.394 | 1.039 ± 0.348 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.047 ± 0.001 | 0.067 ± 0.002 | unsupported | 0.048 ± 0.075 | 0.100 ± 0.012 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.685 ± 0.121 | 0.667 ± 0.111 | unsupported | 0.602 ± 0.035 | 0.786 ± 0.224 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.056 ± 0.005 | 0.072 ± 0.005 | unsupported | 0.099 ± 0.004 | 0.134 ± 0.005 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 1.247 ± 0.322 | 1.557 ± 0.246 | unsupported | 0.419 ± 0.010 | 0.469 ± 0.139 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.096 ± 0.007 | 0.107 ± 0.002 | unsupported | 0.104 ± 0.053 | 0.116 ± 0.014 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.317 ± 0.086 | 1.435 ± 0.163 | unsupported | 1.524 ± 0.043 | 1.379 ± 0.145 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.097 ± 0.005 | 0.113 ± 0.004 | unsupported | 0.121 ± 0.037 | 0.222 ± 0.057 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 1.883 ± 0.581 | 1.882 ± 1.000 | unsupported | 0.735 ± 0.069 | 2.165 ± 0.690 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.130 ± 0.002 | 0.152 ± 0.007 | unsupported | unsupported | 0.138 ± 0.024 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 2.016 ± 0.070 | 2.073 ± 0.122 | unsupported | unsupported | 1.894 ± 0.264 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.133 ± 0.020 | 0.158 ± 0.018 | unsupported | unsupported | 0.251 ± 0.015 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 2.370 ± 0.582 | 2.177 ± 0.994 | unsupported | unsupported | 1.011 ± 0.141 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.012 ± 0.001 | 0.017 ± 0.005 | unsupported | 0.008 ± 0.001 | 0.057 ± 0.011 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.114 ± 0.011 | 0.124 ± 0.004 | unsupported | 0.069 ± 0.111 | 0.250 ± 0.054 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.013 ± 0.001 | 0.018 ± 0.001 | unsupported | 0.061 ± 0.013 | 0.042 ± 0.047 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.099 ± 0.053 | 0.273 ± 0.233 | unsupported | 0.101 ± 0.012 | 0.516 ± 0.223 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.026 ± 0.002 | 0.040 ± 0.001 | unsupported | 0.038 ± 0.007 | 0.123 ± 0.038 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.383 ± 0.023 | 0.358 ± 0.081 | unsupported | 0.470 ± 0.070 | 0.713 ± 0.051 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.032 ± 0.002 | 0.041 ± 0.004 | unsupported | 0.137 ± 0.052 | 0.164 ± 0.013 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.729 ± 0.295 | 0.647 ± 0.343 | unsupported | 0.494 ± 0.047 | 0.447 ± 0.135 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.051 ± 0.002 | 0.052 ± 0.008 | unsupported | 0.147 ± 0.099 | 0.084 ± 0.010 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.791 ± 0.010 | 0.800 ± 0.024 | unsupported | 1.308 ± 0.133 | 1.019 ± 0.330 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.029 ± 0.001 | 0.032 ± 0.004 | unsupported | 0.171 ± 0.036 | 0.088 ± 0.006 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.438 ± 0.032 | 0.432 ± 0.137 | unsupported | 0.741 ± 0.158 | 0.948 ± 0.162 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.054 ± 0.003 | 0.052 ± 0.003 | unsupported | 0.093 ± 0.021 | 0.098 ± 0.009 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.826 ± 0.018 | 0.837 ± 0.036 | unsupported | 1.290 ± 0.050 | 1.701 ± 0.966 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.030 ± 0.001 | 0.032 ± 0.062 | unsupported | 0.167 ± 0.049 | 0.175 ± 0.017 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.516 ± 0.194 | 0.580 ± 0.505 | unsupported | 1.097 ± 0.210 | 0.974 ± 0.176 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.097 ± 0.019 | 0.106 ± 0.003 | unsupported | 0.146 ± 0.130 | 0.119 ± 0.005 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.322 ± 0.157 | 1.428 ± 0.107 | unsupported | 1.678 ± 0.031 | 2.144 ± 2.418 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.094 ± 0.006 | 0.110 ± 0.006 | unsupported | 0.162 ± 0.033 | 0.224 ± 0.017 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 1.636 ± 0.710 | 2.049 ± 1.139 | unsupported | 1.142 ± 0.235 | 1.976 ± 0.403 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.223 ± 0.008 | 0.230 ± 0.012 | unsupported | 0.402 ± 0.064 | 0.310 ± 0.012 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 3.649 ± 0.104 | 3.639 ± 0.047 | unsupported | 4.546 ± 0.150 | 5.218 ± 1.194 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.141 ± 0.028 | 0.139 ± 0.023 | unsupported | 0.436 ± 0.112 | 0.403 ± 0.022 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 1.810 ± 0.298 | 2.027 ± 0.507 | unsupported | 2.409 ± 0.368 | 3.801 ± 0.967 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.570 ± 0.042 | 0.863 ± 0.156 | unsupported | 0.138 ± 0.084 | 0.163 ± 0.061 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 10.418 ± 0.392 | 11.884 ± 1.291 | unsupported | 2.428 ± 0.151 | 1.344 ± 0.263 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.633 ± 0.041 | 0.827 ± 0.093 | unsupported | 0.155 ± 0.099 | 0.191 ± 0.120 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 7.377 ± 2.036 | 8.379 ± 0.553 | unsupported | 1.673 ± 0.272 | 0.587 ± 0.081 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 20.750 ± 6.242 | - | unsupported | 23.576 ± 5.052 | - | 20.128 ± 4.509 | 18.329 ± 9.415 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 17.717 ± 3.236 | - | unsupported | 19.050 ± 3.633 | - | 17.876 ± 1.215 | 28.593 ± 13.811 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 12.888 ± 3.750 | - | unsupported | 17.084 ± 6.735 | - | 13.829 ± 1.594 | 15.517 ± 1.827 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 13.677 ± 2.310 | - | unsupported | 13.969 ± 1.828 | - | 11.442 ± 0.214 | 10.179 ± 0.964 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 14.579 ± 4.927 | - | unsupported | 13.072 ± 2.141 | - | 13.673 ± 3.341 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 11.851 ± 2.798 | - | unsupported | 14.737 ± 2.342 | - | 10.392 ± 0.667 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 23.353 ± 4.856 | - | unsupported | 25.438 ± 10.010 | - | 20.867 ± 3.835 | 21.671 ± 4.429 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 20.054 ± 1.265 | - | unsupported | 22.199 ± 5.516 | - | 17.619 ± 2.121 | 17.261 ± 2.437 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 53.124 ± 4.410 | - | unsupported | 12.500 ± 1.691 | - | 51.395 ± 1.607 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 22.181 ± 6.342 | - | unsupported | 12.181 ± 1.659 | - | 23.423 ± 5.571 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 52.900 ± 0.632 | - | unsupported | 13.446 ± 1.389 | - | 48.841 ± 1.086 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 21.414 ± 4.748 | - | unsupported | 18.485 ± 6.009 | - | 21.837 ± 3.605 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 20.740 ± 3.344 | - | unsupported | 22.194 ± 6.496 | - | 20.290 ± 5.745 | 20.192 ± 6.629 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 19.150 ± 2.548 | - | unsupported | 20.315 ± 3.183 | - | 18.910 ± 1.969 | 17.845 ± 2.927 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 13.007 ± 1.459 | - | unsupported | 13.712 ± 3.517 | - | 14.974 ± 1.749 | 15.003 ± 8.313 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 14.703 ± 1.875 | - | unsupported | 14.793 ± 3.979 | - | 10.468 ± 1.121 | 10.427 ± 2.020 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.537 ± 0.025 | unsupported | unsupported | 0.306 ± 0.018 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 10.693 ± 0.467 | unsupported | unsupported | 3.784 ± 1.439 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.562 ± 0.149 | unsupported | unsupported | 0.489 ± 0.138 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 8.238 ± 0.471 | unsupported | unsupported | 4.287 ± 0.397 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 26.067 ± 6.693 | - | unsupported | 22.754 ± 6.139 | - | 23.933 ± 5.321 | 21.233 ± 4.365 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 16.975 ± 1.961 | - | unsupported | 20.741 ± 2.854 | - | 18.449 ± 1.956 | 19.838 ± 8.360 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 8.782 ± 0.788 | - | 0.002 ± 0.013 | 5.877 ± 0.883 | 6.096 ± 0.900 | 14.766 ± 0.864 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 11.681 ± 3.058 | - | 0.001 ± 0.000 | 7.049 ± 1.016 | 4.890 ± 1.915 | 15.165 ± 1.828 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 10.570 ± 3.231 | - | 11.039 ± 2.549 | 11.504 ± 1.337 | 9.521 ± 1.134 | 14.053 ± 7.954 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 10.326 ± 3.293 | - | 11.133 ± 2.075 | 10.863 ± 1.489 | 9.509 ± 0.736 | 10.474 ± 1.542 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 12.663 ± 1.538 | - | 12.747 ± 0.462 | 14.170 ± 2.184 | 21.090 ± 5.122 | 13.672 ± 2.093 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 12.509 ± 1.515 | - | 10.915 ± 2.376 | 12.729 ± 1.399 | 14.971 ± 3.965 | 13.217 ± 9.121 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 7.244 ± 3.438 | - | 6.216 ± 1.081 | 13.751 ± 4.129 | 10.071 ± 2.906 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 6.790 ± 1.530 | - | 5.253 ± 0.258 | 10.869 ± 2.742 | 11.084 ± 2.855 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 14.137 ± 4.048 | - | 0.001 ± 0.000 | 12.486 ± 2.915 | 11.869 ± 4.249 | 16.668 ± 7.054 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 14.512 ± 2.468 | - | 0.001 ± 0.000 | 14.503 ± 3.367 | 11.058 ± 1.566 | 11.784 ± 5.104 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 44.015 ± 3.429 | - | 0.003 ± 0.001 | 55.214 ± 5.222 | 56.078 ± 5.554 | 204.911 ± 20.082 | 41.540 ± 2.414 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 21.526 ± 2.003 | - | 0.001 ± 0.000 | 59.846 ± 5.921 | 33.289 ± 4.473 | 207.852 ± 29.745 | 17.675 ± 1.373 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 10.955 ± 8.733 | - | 9.221 ± 2.451 | 10.149 ± 0.582 | 7.122 ± 1.421 | 11.302 ± 1.657 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 6.327 ± 1.038 | - | 6.051 ± 0.801 | 9.378 ± 1.800 | 5.623 ± 1.731 | 11.476 ± 1.576 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 9.766 ± 0.603 | - | 9.050 ± 1.309 | 9.125 ± 1.607 | 5.939 ± 0.133 | 11.753 ± 1.647 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 6.197 ± 0.475 | - | 5.427 ± 0.268 | 8.078 ± 1.825 | 5.729 ± 1.360 | 10.393 ± 3.987 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.001 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.002 ± 0.004 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.001 ± 0.002 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.001 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.001 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 4.948291 | 309268.19 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 4.571916 | 285744.75 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 7.062917 | 441432.31 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 7.218625 | 451164.06 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 4.648042 | 290502.62 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 4.850834 | 303177.12 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 18.660834 | 1166302.12 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 20.017959 | 1251122.44 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 13.694917 | 855932.31 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 14.882541 | 930158.81 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 6.291166 | 393197.88 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 5.143958 | 321497.38 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 6.681042 | 417565.12 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 5.664333 | 354020.81 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 75.337916 | 4708619.75 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 63.817500 | 3988593.75 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 104.468083 | 6529255.19 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 104.829000 | 6551812.50 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 9.482625 | 592664.06 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 5.084708 | 317794.25 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 6.669166 | 416822.88 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 6.733625 | 420851.56 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 100.925791 | 6307861.94 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 69.950583 | 4371911.44 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 84.697250 | 5293578.12 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 90.631584 | 5664474.00 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 17.939500 | 1121218.75 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 18.334875 | 1145929.69 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 443.398292 | 27712393.25 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 473.287750 | 29580484.38 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 17.115625 | 1069726.56 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 18.016875 | 1126054.69 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 309.430125 | 19339382.81 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 323.425333 | 20214083.31 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 18.109208 | 1131825.50 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 19.413083 | 1213317.69 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 461.262292 | 28828893.25 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 464.972750 | 29060796.88 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 16.875875 | 1054742.19 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 17.422625 | 1088914.06 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 416.793209 | 26049575.56 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 333.124791 | 20820299.44 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.363959 | 36936.86 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.623541 | 134742.83 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 13.579208 | 212175.12 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 13.665000 | 213515.62 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 4.017375 | 502171.88 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 6.936375 | 867046.88 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 22.732125 | 2841515.62 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 22.830916 | 2853864.50 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 1.777917 | 27779.95 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 7.281375 | 113771.48 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 11.303709 | 176620.45 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 10.957917 | 171217.45 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.525000 | 565625.00 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.093084 | 761635.50 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 13.299417 | 1662427.12 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 12.357833 | 1544729.12 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.022834 | 187856.78 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.114833 | 95544.27 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.202375 | 81287.11 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.145583 | 80399.73 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 33.180500 | 4147562.50 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.288167 | 1411020.88 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.235083 | 1279385.38 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.405875 | 1300734.38 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.329333 | 192645.83 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.707750 | 89183.59 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.998750 | 62480.47 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3.441791 | 53777.98 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 13.300167 | 1662520.88 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.720917 | 1090114.62 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.075459 | 884432.38 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.585375 | 698171.88 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.614292 | 87723.31 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 26.714166 | 417408.84 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 17.471875 | 272998.05 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 17.342125 | 270970.70 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.631959 | 1703994.88 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 38.352333 | 4794041.62 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 35.556000 | 4444500.00 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 36.141958 | 4517744.75 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.439833 | 69372.39 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 19.766833 | 308856.77 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 10.921958 | 170655.59 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 11.188542 | 174820.97 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.312916 | 1039114.50 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 25.200917 | 3150114.62 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 22.906417 | 2863302.12 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 26.481291 | 3310161.38 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.373125 | 99580.08 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.075125 | 48048.83 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.026459 | 47288.42 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.319834 | 67497.41 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.286125 | 785765.62 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.812417 | 601552.12 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.480750 | 685093.75 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.338000 | 667250.00 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.550333 | 133598.95 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.350583 | 99227.86 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.571292 | 55801.44 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.577292 | 71520.19 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.753000 | 469125.00 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.354750 | 419343.75 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.976083 | 1247010.38 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 12.459500 | 1557437.50 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.398500 | 115601.56 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.659209 | 104050.14 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.118042 | 95594.41 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.874167 | 107408.86 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.028542 | 1378567.75 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 12.191166 | 1523895.75 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.537167 | 1317145.88 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.479459 | 1434932.38 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 14.214042 | 222094.41 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 7.724958 | 120702.47 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.236417 | 97444.02 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.238917 | 113108.08 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 17.318625 | 2164828.12 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.879125 | 734890.62 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 15.064584 | 1883073.00 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 15.055500 | 1881937.50 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.863917 | 138498.70 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.314959 | 129921.23 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.755125 | 152423.83 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 15.149208 | 1893651.00 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 16.130500 | 2016312.50 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 16.580083 | 2072510.37 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 16.044542 | 250695.97 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.515833 | 133059.89 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 10.105333 | 157895.83 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.090458 | 1011307.25 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 18.961292 | 2370161.50 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 17.416625 | 2177078.12 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 3.677458 | 57460.28 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.514667 | 8041.67 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.770875 | 12044.92 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.071625 | 16744.14 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 1.998583 | 249822.88 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.551167 | 68895.88 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.914292 | 114286.50 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.988333 | 123541.62 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.656041 | 41500.64 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.926000 | 61343.75 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.822584 | 12852.88 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.144084 | 17876.31 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.129167 | 516145.88 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.807416 | 100927.00 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.788791 | 98598.88 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.181625 | 272703.12 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.844792 | 122574.88 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 2.410291 | 37660.80 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.679458 | 26241.53 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.537917 | 39654.95 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.706250 | 713281.25 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 3.760125 | 470015.62 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.060291 | 382536.38 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 2.860917 | 357614.62 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.506125 | 164158.20 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 8.783125 | 137236.33 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.022917 | 31608.08 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.596208 | 40565.75 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.573875 | 446734.38 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.951625 | 493953.12 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.834000 | 729250.00 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.176708 | 647088.50 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.403375 | 84427.73 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 9.400500 | 146882.81 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.237250 | 50582.03 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.358166 | 52471.34 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 8.155666 | 1019458.25 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.464875 | 1308109.38 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.331042 | 791380.25 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.403791 | 800473.88 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.660125 | 88439.45 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 10.957167 | 171205.73 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.884125 | 29439.45 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.051750 | 32058.59 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.580125 | 947515.62 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.926834 | 740854.25 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.504708 | 438088.50 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.454041 | 431755.12 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.256708 | 97761.06 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.936709 | 92761.08 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.446584 | 53852.88 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.334041 | 52094.39 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.610208 | 1701276.00 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.317583 | 1289697.88 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.605583 | 825697.88 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.699792 | 837474.00 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.199875 | 174998.05 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 10.669375 | 166708.98 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.909375 | 29833.98 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.042542 | 31914.72 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.795875 | 974484.38 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.777000 | 1097125.00 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.131583 | 516447.88 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.642583 | 580322.88 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.630500 | 119226.56 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 9.322042 | 145656.91 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.234416 | 97412.75 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.799334 | 106239.59 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 17.149000 | 2143625.00 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 13.422209 | 1677776.12 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.575709 | 1321963.62 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.425083 | 1428135.38 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 14.351875 | 224248.05 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 10.390458 | 162350.91 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.033708 | 94276.69 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.059292 | 110301.44 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 15.808250 | 1976031.25 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 9.136375 | 1142046.88 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 13.086250 | 1635781.25 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 16.392667 | 2049083.38 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 19.809167 | 309518.23 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 25.745042 | 402266.28 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 14.300500 | 223445.31 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 14.705375 | 229771.48 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 41.744167 | 5218020.88 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 36.371833 | 4546479.12 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 29.190958 | 3648869.75 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 29.114750 | 3639343.75 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 25.767875 | 402623.05 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 27.930125 | 436408.20 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 9.011750 | 140808.59 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 8.902000 | 139093.75 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 30.410875 | 3801359.38 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 19.272417 | 2409052.12 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 14.478750 | 1809843.75 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 16.213875 | 2026734.38 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.445333 | 163208.33 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.818667 | 137791.67 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 36.504125 | 570376.95 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 55.224375 | 862880.86 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.755166 | 1344395.75 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 19.421250 | 2427656.25 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 83.340541 | 10417567.62 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 95.072666 | 11884083.25 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.238708 | 191229.81 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 9.930125 | 155158.20 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 40.503375 | 632865.23 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 52.929625 | 827025.39 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.698125 | 587265.62 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 13.380250 | 1672531.25 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 59.018416 | 7377302.00 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 67.035625 | 8379453.12 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 4.889041 | 305565.06 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 8.590250 | 536890.62 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 60.541833 | 3783864.56 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 171.087750 | 10692984.38 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 7.826833 | 489177.06 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 8.996208 | 562263.00 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 68.588083 | 4286755.19 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 131.811375 | 8238210.94 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.016667 | 1041.69 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.001375 | 85.94 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.033959 | 2122.44 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.001333 | 83.31 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.012458 | 778.62 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.031959 | 1997.44 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.014416 | 901.00 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.022500 | 1406.25 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.011292 | 705.75 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.003500 | 218.75 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.017625 | 1101.56 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.004166 | 260.38 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.011500 | 718.75 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002875 | 179.69 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.015625 | 976.56 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002625 | 164.06 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `jax-cpu` is 48.8x faster than the slowest successful cell (`julia-base`, 71.381 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `pytorch-cpu` is 18.3x faster than the slowest successful cell (`julia-base`, 71.381 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-direct` is 19.4x faster than the slowest successful cell (`julia-base`, 71.381 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-trace` is 24.1x faster than the slowest successful cell (`julia-base`, 71.381 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `jax-cpu` is 21.8x faster than the slowest successful cell (`julia-base`, 445.424 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `pytorch-cpu` is 39.8x faster than the slowest successful cell (`julia-base`, 445.424 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-direct` is 45.3x faster than the slowest successful cell (`julia-base`, 445.424 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-trace` is 49.6x faster than the slowest successful cell (`julia-base`, 445.424 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 23.7x faster than the slowest successful cell (`julia-base`, 160.256 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 22.4x faster than the slowest successful cell (`julia-base`, 160.256 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-direct` is 21.6x faster than the slowest successful cell (`julia-base`, 160.256 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-trace` is 24.6x faster than the slowest successful cell (`julia-base`, 160.256 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 48.7x faster than the slowest successful cell (`julia-base`, 124.435 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 22.4x faster than the slowest successful cell (`julia-base`, 124.435 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-direct` is 21.0x faster than the slowest successful cell (`julia-base`, 124.435 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-trace` is 28.7x faster than the slowest successful cell (`julia-base`, 124.435 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `jax-cpu` is 87.2x faster than the slowest successful cell (`julia-base`, 590.886 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `pytorch-cpu` is 76.6x faster than the slowest successful cell (`julia-base`, 590.886 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-direct` is 76.0x faster than the slowest successful cell (`julia-base`, 590.886 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-trace` is 70.6x faster than the slowest successful cell (`julia-base`, 590.886 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 18.5x faster than the slowest successful cell (`pytorch-cpu`, 21.986 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 19.5x faster than the slowest successful cell (`pytorch-cpu`, 21.986 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 19.8x faster than the slowest successful cell (`pytorch-cpu`, 21.986 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 23.6x faster than the slowest successful cell (`pytorch-cpu`, 13.655 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 13.655 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 16.6x faster than the slowest successful cell (`pytorch-cpu`, 13.655 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 19.9x faster than the slowest successful cell (`pytorch-cpu`, 20.284 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 16.2x faster than the slowest successful cell (`pytorch-cpu`, 20.284 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 20.284 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 16.4x faster than the slowest successful cell (`pytorch-cpu`, 13.667 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 19.6x faster than the slowest successful cell (`pytorch-cpu`, 13.667 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 17.7x faster than the slowest successful cell (`pytorch-cpu`, 13.667 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 532.2x faster than the slowest successful cell (`julia-base`, 2.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 1058.0x faster than the slowest successful cell (`julia-base`, 1.543 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `pytorch-cpu` is 18.1x faster than the slowest successful cell (`jax-cpu`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 33.9x faster than the slowest successful cell (`jax-cpu`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-trace` is 24.6x faster than the slowest successful cell (`jax-cpu`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `pytorch-cpu` is 11.9x faster than the slowest successful cell (`jax-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 36.6x faster than the slowest successful cell (`jax-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 21.8x faster than the slowest successful cell (`jax-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x2x2`): `tenferro-direct` is 30.7x faster than the slowest successful cell (`jax-cpu`, 0.037 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x2x2`): `tenferro-trace` is 14.3x faster than the slowest successful cell (`jax-cpu`, 0.037 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 18.3x faster than the slowest successful cell (`jax-cpu`, 0.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-trace` is 12.2x faster than the slowest successful cell (`jax-cpu`, 0.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `jax-cpu` is 16.9x faster than the slowest successful cell (`pytorch-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 85.6x faster than the slowest successful cell (`pytorch-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 46.8x faster than the slowest successful cell (`pytorch-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 14.3x faster than the slowest successful cell (`jax-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-trace` is 12.2x faster than the slowest successful cell (`jax-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 45.1x faster than the slowest successful cell (`pytorch-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 19.0x faster than the slowest successful cell (`pytorch-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 33.3x faster than the slowest successful cell (`pytorch-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `jax-cpu` is 18.2x faster than the slowest successful cell (`pytorch-cpu`, 0.138 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 92.3x faster than the slowest successful cell (`pytorch-cpu`, 0.138 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 40.5x faster than the slowest successful cell (`pytorch-cpu`, 0.138 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 23.8x faster than the slowest successful cell (`pytorch-cpu`, 0.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 13.1x faster than the slowest successful cell (`pytorch-cpu`, 0.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 27.5x faster than the slowest successful cell (`pytorch-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `jax-cpu` is 17.6x faster than the slowest successful cell (`pytorch-cpu`, 0.143 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 81.9x faster than the slowest successful cell (`pytorch-cpu`, 0.143 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 47.8x faster than the slowest successful cell (`pytorch-cpu`, 0.143 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 23.1x faster than the slowest successful cell (`pytorch-cpu`, 0.045 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 15.1x faster than the slowest successful cell (`pytorch-cpu`, 0.045 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 25.3x faster than the slowest successful cell (`pytorch-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `jax-cpu` is 17.0x faster than the slowest successful cell (`pytorch-cpu`, 0.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 84.8x faster than the slowest successful cell (`pytorch-cpu`, 0.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 36.9x faster than the slowest successful cell (`pytorch-cpu`, 0.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 26.3x faster than the slowest successful cell (`pytorch-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 16.7x faster than the slowest successful cell (`pytorch-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 21.9x faster than the slowest successful cell (`pytorch-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 22.3x faster than the slowest successful cell (`jax-cpu`, 0.033 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=1, shape=`1x2x2`): `tenferro-trace` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.033 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`1024x2x2`): `pytorch-cpu` is 10.7x faster than the slowest successful cell (`jax-cpu`, 1.403 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 27.2x faster than the slowest successful cell (`jax-cpu`, 0.027 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`1x2x2`): `tenferro-trace` is 12.3x faster than the slowest successful cell (`jax-cpu`, 0.027 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.019 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`3x4x4`): `tenferro-direct` is 14.3x faster than the slowest successful cell (`jax-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`3x4x4`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`4x2x2`): `tenferro-direct` is 20.2x faster than the slowest successful cell (`jax-cpu`, 0.033 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`4x2x2`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`jax-cpu`, 0.033 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 21.3x faster than the slowest successful cell (`jax-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-trace` is 16.0x faster than the slowest successful cell (`jax-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 19.7x faster than the slowest successful cell (`jax-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`jax-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 21.3x faster than the slowest successful cell (`jax-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-trace` is 16.7x faster than the slowest successful cell (`jax-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x8x8`): `pytorch-cpu` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.122 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 22.2x faster than the slowest successful cell (`jax-cpu`, 0.122 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x8x8`): `tenferro-trace` is 18.7x faster than the slowest successful cell (`jax-cpu`, 0.122 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 33.1x faster than the slowest successful cell (`pytorch-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 25.0x faster than the slowest successful cell (`pytorch-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 32.3x faster than the slowest successful cell (`pytorch-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 21.9x faster than the slowest successful cell (`pytorch-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `jax-cpu` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.114 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 32.3x faster than the slowest successful cell (`pytorch-cpu`, 0.114 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 29.5x faster than the slowest successful cell (`pytorch-cpu`, 0.114 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 35.6x faster than the slowest successful cell (`pytorch-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 22.1x faster than the slowest successful cell (`pytorch-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 23.4x faster than the slowest successful cell (`pytorch-cpu`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 20.1x faster than the slowest successful cell (`pytorch-cpu`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 31.0x faster than the slowest successful cell (`pytorch-cpu`, 0.116 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 24.3x faster than the slowest successful cell (`pytorch-cpu`, 0.116 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 32.8x faster than the slowest successful cell (`pytorch-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 21.7x faster than the slowest successful cell (`pytorch-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 28.3x faster than the slowest successful cell (`pytorch-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 20.1x faster than the slowest successful cell (`pytorch-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 17.8x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `jax-cpu` is 13.1x faster than the slowest successful cell (`pytorch-cpu`, 0.139 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 43.8x faster than the slowest successful cell (`pytorch-cpu`, 0.139 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 33.6x faster than the slowest successful cell (`pytorch-cpu`, 0.139 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 31.8x faster than the slowest successful cell (`jax-cpu`, 0.150 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 26.1x faster than the slowest successful cell (`jax-cpu`, 0.150 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 16.8x faster than the slowest successful cell (`pytorch-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 16.7x faster than the slowest successful cell (`pytorch-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 16.9x faster than the slowest successful cell (`pytorch-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.027 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 12.5x faster than the slowest successful cell (`pytorch-cpu`, 0.040 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.025 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 21.1x faster than the slowest successful cell (`pytorch-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x2x2,rhs=1`): `tenferro-direct` is 11.3x faster than the slowest successful cell (`tenferro-trace`, 0.016 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x32x32,rhs=1`): `tenferro-direct` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`4x2x2,rhs=1`): `tenferro-direct` is 18.8x faster than the slowest successful cell (`pytorch-cpu`, 0.031 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x2x2,rhs=1`): `tenferro-direct` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 0.035 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x4x4,rhs=1`): `tenferro-direct` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 14.2x faster than the slowest successful cell (`jax-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `jax-cpu` is 19.3x faster than the slowest successful cell (`julia-base`, 390.580 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `pytorch-cpu` is 19.1x faster than the slowest successful cell (`julia-base`, 390.580 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-direct` is 15.2x faster than the slowest successful cell (`julia-base`, 390.580 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-trace` is 14.4x faster than the slowest successful cell (`julia-base`, 390.580 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 31.2x faster than the slowest successful cell (`julia-base`, 236.411 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 22.4x faster than the slowest successful cell (`julia-base`, 236.411 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 12.2x faster than the slowest successful cell (`julia-base`, 236.411 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 13.6x faster than the slowest successful cell (`julia-base`, 236.411 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `jax-cpu` is 59.3x faster than the slowest successful cell (`julia-base`, 512.933 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `pytorch-cpu` is 65.0x faster than the slowest successful cell (`julia-base`, 512.933 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-direct` is 71.3x faster than the slowest successful cell (`julia-base`, 512.933 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-trace` is 69.1x faster than the slowest successful cell (`julia-base`, 512.933 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `jax-cpu` is 31.3x faster than the slowest successful cell (`julia-base`, 270.374 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `pytorch-cpu` is 30.9x faster than the slowest successful cell (`julia-base`, 270.374 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-direct` is 28.4x faster than the slowest successful cell (`julia-base`, 270.374 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-trace` is 26.6x faster than the slowest successful cell (`julia-base`, 270.374 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `jax-cpu` is 26.9x faster than the slowest successful cell (`julia-base`, 702.814 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 29.8x faster than the slowest successful cell (`julia-base`, 702.814 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-direct` is 35.9x faster than the slowest successful cell (`julia-base`, 702.814 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-trace` is 36.9x faster than the slowest successful cell (`julia-base`, 702.814 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 39.9x faster than the slowest successful cell (`julia-base`, 381.907 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 38.0x faster than the slowest successful cell (`julia-base`, 381.907 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-direct` is 15.6x faster than the slowest successful cell (`julia-base`, 381.907 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-trace` is 15.5x faster than the slowest successful cell (`julia-base`, 381.907 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 29.9x faster than the slowest successful cell (`julia-base`, 157.368 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-eager` is 13.4x faster than the slowest successful cell (`julia-base`, 157.368 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-trace` is 13.1x faster than the slowest successful cell (`julia-base`, 157.368 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 30.0x faster than the slowest successful cell (`julia-base`, 349.025 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 31.1x faster than the slowest successful cell (`julia-base`, 349.025 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 19.1x faster than the slowest successful cell (`julia-base`, 349.025 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 19.1x faster than the slowest successful cell (`julia-base`, 349.025 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `jax-cpu` is 79.9x faster than the slowest successful cell (`julia-base`, 988.778 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 70.9x faster than the slowest successful cell (`julia-base`, 988.778 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 60.2x faster than the slowest successful cell (`julia-base`, 988.778 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 58.7x faster than the slowest successful cell (`julia-base`, 988.778 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `jax-cpu` is 69.4x faster than the slowest successful cell (`julia-base`, 946.520 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 81.5x faster than the slowest successful cell (`julia-base`, 946.520 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 56.9x faster than the slowest successful cell (`julia-base`, 946.520 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 57.0x faster than the slowest successful cell (`julia-base`, 946.520 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 31.1x faster than the slowest successful cell (`julia-base`, 239.325 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 28.6x faster than the slowest successful cell (`julia-base`, 239.325 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 10.4x faster than the slowest successful cell (`julia-base`, 239.325 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`julia-base`, 239.325 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 43.6x faster than the slowest successful cell (`julia-base`, 1691.971 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 41.4x faster than the slowest successful cell (`julia-base`, 1691.971 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-eager` is 29.1x faster than the slowest successful cell (`julia-base`, 1691.971 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-trace` is 27.8x faster than the slowest successful cell (`julia-base`, 1691.971 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 14.3x faster than the slowest successful cell (`tenferro-eager`, 8.379 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 6948.6x faster than the slowest successful cell (`julia-base`, 14.766 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 10709.7x faster than the slowest successful cell (`julia-base`, 15.165 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 11762.7x faster than the slowest successful cell (`julia-base`, 16.668 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 10554.0x faster than the slowest successful cell (`tenferro-direct`, 14.512 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 59257.1x faster than the slowest successful cell (`julia-base`, 204.911 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `strided-jl` is 11.8x faster than the slowest successful cell (`julia-base`, 207.852 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 155928.3x faster than the slowest successful cell (`julia-base`, 207.852 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 12.1x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 25.6x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 399.4x faster than the slowest successful cell (`tenferro-direct`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 281.2x faster than the slowest successful cell (`tenferro-direct`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 141.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 220.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 143.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 195.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 4192.3 GB/s at 0.004 ms. The conservative physical lower bound is 0.168 ms, so the cell is 41.9x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 11501.4 GB/s at 0.001 ms. The conservative physical lower bound is 0.042 ms, so the cell is 28.8x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 126322.6 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1263.2x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 189573.1 GB/s at 0.001 ms. The conservative physical lower bound is 0.671 ms, so the cell is 473.9x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 189439.3 GB/s at 0.001 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1894.4x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 195225.8 GB/s at 0.001 ms. The conservative physical lower bound is 0.671 ms, so the cell is 488.1x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 38813.7 GB/s at 0.003 ms. The conservative physical lower bound is 1.342 ms, so the cell is 388.1x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 100688.5 GB/s at 0.001 ms. The conservative physical lower bound is 0.336 ms, so the cell is 251.7x faster than the bound.
